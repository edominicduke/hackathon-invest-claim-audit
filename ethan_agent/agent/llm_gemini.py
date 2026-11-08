"""Gemini adapter: same ``structured()`` contract as ClaudeLLM, so the pipeline,
prompts, code checks and verdict rules are unchanged when switching providers.

Select it with:  $env:AUDITOR_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "..."
Options:         $env:GEMINI_MODEL = "..."        model (default below)
                 $env:GEMINI_FAST = "1"           one thinking level lower + claims checked
                                                  one at a time (fewer rate-limit bursts)
                 $env:GEMINI_TIMEOUT_S = "120"    per-attempt timeout
                 $env:GEMINI_MAX_THINKING = "LOW" never think more than this (LOW/MEDIUM/HIGH);
                                                  default LOW for every model - set HIGH to lift
                 $env:GEMINI_FALLBACK_MODELS = "a,b"  models to switch to when the current one's
                                                  daily quota is used up (default: discovered)
                 $env:GEMINI_AUTO_SWITCH = "0"    turn automatic switching off

Differences from ClaudeLLM, handled here:
* effort low/medium/high -> Gemini thinking level (or a token budget on 2.x models)
* safety blocks (prompt- or response-level) -> LLMError, like a Claude refusal
* no server-side fallback model: a block simply fails that step
* our own retry loop (not the SDK's silent one) so every retry is logged
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import CallLog, LLMError, short_error, strict_schema

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("agent.llm")

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_MAX_THINKING = "LOW"

# Finish reasons that mean "the model declined / output was blocked".
_BLOCKED = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}
# Rate limits, server errors, server-side deadline: worth retrying.
RETRYABLE = {429, 500, 502, 503, 504}

_LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_FAST_LEVELS = {"low": "LOW", "medium": "LOW", "high": "MEDIUM"}  # one step down, floor LOW
_BUDGETS = {"low": 1024, "medium": 4096, "high": -1}  # Gemini 2.x; -1 = automatic
_FAST_BUDGETS = {"low": 512, "medium": 1024, "high": 4096}


_ORDER = ["LOW", "MEDIUM", "HIGH"]
# Model-name fragments that mean "not a plain text model" - never auto-switch to these.
_NOT_TEXT = ("tts", "image", "live", "audio", "embedding", "transcribe", "computer-use", "customtools", "native")
# Models that hit their daily quota in this process. Shared by all instances/threads.
_EXHAUSTED: set[str] = set()
_SWITCH_LOCK = threading.Lock()
_CAP_BUDGETS = {"LOW": 512, "MEDIUM": 4096, "HIGH": -1}


def quota_info(exc: Exception) -> tuple[str | None, float | None]:
    """From a 429: (which quota was hit, server-suggested wait in seconds)."""
    quota, delay = None, None
    details = getattr(exc, "details", None)
    error = details.get("error", details) if isinstance(details, dict) else {}
    for d in (error.get("details") or []) if isinstance(error, dict) else []:
        for v in d.get("violations") or []:
            quota = quota or v.get("quotaId") or v.get("quotaMetric")
        m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
        if m:
            delay = float(m.group(1))
    return quota, delay


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class GeminiLLM:
    def __init__(
        self,
        model: str | None = None,
        client: Any = None,
        max_tokens: int = 16000,
        *,
        fast: bool | None = None,
        attempts: int = 5,
        initial_delay: float = 2.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._api_error = errors.APIError
        timeout_s = int(os.environ.get("GEMINI_TIMEOUT_S", "120"))
        # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment - never hardcode it.
        # SDK retries are off (attempts=1): structured() retries itself and logs each one.
        self.client = client or genai.Client(
            http_options=types.HttpOptions(
                timeout=timeout_s * 1000, retry_options=types.HttpRetryOptions(attempts=1)
            )
        )
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.max_tokens = max_tokens
        self.fast = env_flag("GEMINI_FAST") if fast is None else fast
        # Every call, on every model (incl. auto-switched fallbacks), thinks at most this much.
        # Default LOW: fastest, fewest tokens, gentlest on free-tier quotas. Raise deliberately.
        cap = (os.environ.get("GEMINI_MAX_THINKING", "").strip() or DEFAULT_MAX_THINKING).upper()
        if cap not in _ORDER:
            raise ValueError(f"GEMINI_MAX_THINKING must be one of {_ORDER}, got {cap!r}")
        self.max_thinking = cap
        self.attempts = max(1, attempts)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._sleep = sleep
        # Auto-switch to another model when one's daily quota is used up (on by default).
        self.auto_switch = os.environ.get("GEMINI_AUTO_SWITCH", "1").strip().lower() not in {"0", "false", "no", "off"}
        env_fallbacks = os.environ.get("GEMINI_FALLBACK_MODELS")
        self._fallbacks: list[str] | None = (
            [m.strip() for m in env_fallbacks.split(",") if m.strip()] if env_fallbacks is not None else None
        )

    # ----------------------------------------------------------- model fallback

    def fallback_models(self) -> list[str]:
        """Models to switch to, in order. From GEMINI_FALLBACK_MODELS, else discovered
        once from the models this key can use (text 'flash' models, newest first)."""
        if self._fallbacks is None:
            self._fallbacks = self._discover_models()
            log.info("   Gemini fallback models: %s", ", ".join(self._fallbacks) or "(none found)")
        return [m for m in self._fallbacks if m != self.model]

    def _discover_models(self) -> list[str]:
        try:
            listed = list(self.client.models.list())
        except Exception as exc:
            log.warning("!  could not list Gemini models for fallback: %s", short_error(exc))
            return []
        names = []
        for m in listed:
            name = (getattr(m, "name", "") or "").removeprefix("models/")
            if "generateContent" not in (getattr(m, "supported_actions", None) or []):
                continue
            if "flash" not in name or any(x in name for x in _NOT_TEXT):
                continue
            names.append(name)

        def rank(name: str):
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
            return (-(float(v.group(1)) if v else 0.0), "lite" in name, "preview" in name or "exp" in name, name)

        return sorted(set(names), key=rank)

    def _switch_from(self, failed: str, why: str) -> bool:
        """Mark ``failed`` as used up and move to the next fallback. Thread-safe: if another
        claim's thread already switched away from ``failed``, just use its choice."""
        with _SWITCH_LOCK:
            _EXHAUSTED.add(failed)
            if self.model != failed:
                return True
            for candidate in self.fallback_models():
                if candidate not in _EXHAUSTED:
                    self.model = candidate
                    log.warning("!! %s %s - switching to %s", failed, why, candidate)
                    return True
            return False

    def _thinking(self, effort: str, model: str | None = None):
        types = self._types
        if (model or self.model).startswith("gemini-2"):
            budget = (_FAST_BUDGETS if self.fast else _BUDGETS).get(effort, -1)
            if self.max_thinking:
                cap = _CAP_BUDGETS[self.max_thinking]
                budget = cap if budget == -1 or (cap != -1 and budget > cap) else budget
            return types.ThinkingConfig(thinking_budget=budget)
        level = (_FAST_LEVELS if self.fast else _LEVELS).get(effort, "MEDIUM")
        if self.max_thinking and _ORDER.index(level) > _ORDER.index(self.max_thinking):
            level = self.max_thinking
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel(level))

    def _retryable(self, exc: Exception) -> str | None:
        """Short reason if ``exc`` is worth retrying, else None."""
        if isinstance(exc, self._api_error):
            code = getattr(exc, "code", None)
            return f"{code} {getattr(exc, 'status', '')}".strip() if code in RETRYABLE else None
        kind = type(exc).__name__.lower()
        if "timeout" in kind:  # client-side timeout (httpx)
            return "client timeout"
        if any(k in kind for k in ("connect", "network", "protocol")) or isinstance(exc, ConnectionError):
            # e.g. "getaddrinfo failed": DNS/Wi-Fi/VPN blip - almost always temporary
            return f"network error ({type(exc).__name__})"
        return None

    @staticmethod
    def _is_deadline(reason: str) -> bool:
        return reason.startswith("504") or reason == "client timeout"

    def _lighter(self, config: Any) -> Any | None:
        """Same request with less thinking, or None if already at the floor."""
        types, tc = self._types, config.thinking_config
        if tc.thinking_level is not None:
            order = ["LOW", "MEDIUM", "HIGH"]
            current = tc.thinking_level.value
            if current not in order or order.index(current) == 0:
                return None
            lower = types.ThinkingConfig(thinking_level=types.ThinkingLevel(order[order.index(current) - 1]))
        else:
            if tc.thinking_budget is not None and 0 <= tc.thinking_budget <= 512:
                return None
            lower = types.ThinkingConfig(thinking_budget=512)
        return config.model_copy(update={"thinking_config": lower})

    def _unavailable(self, exc: Exception) -> str | None:
        """Why this *model* can't be used right now (vs. a transient error), or None."""
        code = getattr(exc, "code", None)
        if code == 429:
            quota, _ = quota_info(exc)
            if quota and "perday" in quota.replace("_", "").lower():
                return f"daily quota used up ({quota})"
        if code == 404:
            return "not available to this key (404)"
        return None

    def _generate(self, name: str, config: Any, user: str, effort: str = "medium"):
        slow_retries = 0
        attempt = 0
        while True:
            attempt += 1
            model = self.model
            start = time.perf_counter()
            try:
                return self.client.models.generate_content(model=model, contents=user, config=config)
            except Exception as exc:
                elapsed = time.perf_counter() - start

                why = self._unavailable(exc)
                if why:
                    if self.auto_switch and self._switch_from(model, why):
                        # A different model: fresh quota, so this isn't a "retry" of the same call.
                        config = config.model_copy(update={"thinking_config": self._thinking(effort, self.model)})
                        attempt -= 1
                        continue
                    raise LLMError(
                        f"Gemini model {model}: {why}. "
                        + ("No fallback model left - quotas reset daily. " if self.auto_switch else "")
                        + "Set $env:GEMINI_MODEL or $env:GEMINI_FALLBACK_MODELS to models with quota left."
                    ) from exc

                reason = self._retryable(exc)
                if reason is None:
                    raise LLMError(f"Gemini error in {name}: {exc}") from exc
                if attempt >= self.attempts:
                    raise LLMError(f"Gemini {reason} in {name} after {attempt} attempts") from exc

                if self._is_deadline(reason):
                    # Too slow, not flaky: resending the identical request would just time
                    # out again. Retry once, with less thinking if possible, then give up.
                    if slow_retries >= 1:
                        raise LLMError(f"Gemini {reason} in {name}: too slow even after a lighter retry") from exc
                    slow_retries += 1
                    lighter = self._lighter(config)
                    change = "same settings"
                    if lighter is not None:
                        old = config.thinking_config.thinking_level or config.thinking_config.thinking_budget
                        new = lighter.thinking_config.thinking_level or lighter.thinking_config.thinking_budget
                        change = f"thinking {getattr(old, 'value', old)} -> {getattr(new, 'value', new)}"
                        config = lighter
                    log.warning("!  %-16s %s after %.1fs - retrying once with %s", name, reason, elapsed, change)
                    continue

                delay = min(self.max_delay, self.initial_delay * 2 ** (attempt - 1))
                if getattr(exc, "code", None) == 429:  # per-minute limit: wait, same model
                    quota, suggested = quota_info(exc)
                    if suggested:
                        delay = min(max(suggested, delay), 90.0)  # honour the server's hint
                    reason = f"{reason} [{quota or 'quota unknown'}]"
                log.warning(
                    "!  %-16s %s after %.1fs - retrying in %.0fs (attempt %d/%d)  [%s]",
                    name, reason, elapsed, delay, attempt + 1, self.attempts, short_error(exc, 90),
                )
                self._sleep(delay)

    def structured(self, *, system: str, user: str, output: type[T], effort: str = "medium") -> T:
        types = self._types
        name = output.__name__
        thinking = self._thinking(effort)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=strict_schema(output),
            thinking_config=thinking,
            max_output_tokens=self.max_tokens,
        )
        level = thinking.thinking_level.value if thinking.thinking_level else f"budget {thinking.thinking_budget}"
        with CallLog(name, f"{self.model}, thinking={level}") as call:
            resp = self._generate(name, config, user, effort)
            usage = getattr(resp, "usage_metadata", None)
            call.note = (
                f"thought_tokens={getattr(usage, 'thoughts_token_count', None)} "
                f"output_tokens={getattr(usage, 'candidates_token_count', None)}"
            )

        feedback = getattr(resp, "prompt_feedback", None)
        if feedback is not None and feedback.block_reason:
            raise LLMError(f"Gemini blocked the prompt ({_name(feedback.block_reason)})")
        if not resp.candidates:
            raise LLMError("Gemini returned no candidates")
        finish = _name(resp.candidates[0].finish_reason)
        if finish in _BLOCKED:
            raise LLMError(f"Gemini blocked the response ({finish})")
        if finish == "MAX_TOKENS":
            raise LLMError("Output hit max_output_tokens; raise it or shrink the input")

        text = resp.text or ""
        try:
            return output.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"Output failed {output.__name__} validation: {exc}") from exc


def _name(value: Any) -> str:
    """Enum or string -> plain name ("SAFETY")."""
    return str(getattr(value, "value", value) or "")
