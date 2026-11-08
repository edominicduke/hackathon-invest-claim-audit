"""Thin LLM layer: one method, ``structured()``, that returns a validated Pydantic object.

Why structured outputs everywhere: every downstream consumer (code checks, verdict
rules, Abdullah's storage, Bogdan's number check) needs fields, not prose. The API's
JSON-schema mode guarantees parseable output; Pydantic re-validates it.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import threading
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"

log = logging.getLogger("agent.llm")


def enable_call_logging(level: int = logging.INFO) -> None:
    """Print one line per model call / pipeline step to the terminal (stderr).

    Configures the "agent" logger directly (not the root logger, which Streamlit owns).
    Safe to call repeatedly, e.g. on every Streamlit rerun.
    """
    logger = logging.getLogger("agent")
    logger.setLevel(level)
    if not any(getattr(h, "_agent_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        handler._agent_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False


def short_error(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {text[:limit]}"


class CallLog:
    """Logs one model call: start, a heartbeat every few seconds while waiting, and
    the outcome. Set ``.note`` before the block ends to add detail to the success line.

        with CallLog("ClaimJudgment", "gemini-3.8-flash, thinking=HIGH") as call:
            resp = ...
            call.note = "output_tokens=412"
    """

    def __init__(self, name: str, detail: str, heartbeat_s: float = 15.0):
        self.name, self.detail, self.heartbeat_s = name, detail, heartbeat_s
        self.note = ""
        self._done = threading.Event()

    def _beat(self) -> None:
        while not self._done.wait(self.heartbeat_s):
            log.info("   %-16s still waiting... %.0fs", self.name, time.perf_counter() - self._start)

    def __enter__(self) -> "CallLog":
        self._start = time.perf_counter()
        log.info("-> %-16s calling %s", self.name, self.detail)
        threading.Thread(target=self._beat, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        self._done.set()
        elapsed = time.perf_counter() - self._start
        if exc is None:
            log.info("<- %-16s done in %.1fs  %s", self.name, elapsed, self.note)
        else:
            log.warning("x  %-16s FAILED after %.1fs: %s", self.name, elapsed, short_error(exc))


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    model: str

    def structured(self, *, system: str, user: str, output: type[T], effort: str = "medium") -> T: ...


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> strict structured-output schema.

    Inlines $refs, marks every property required, forbids extra keys, and drops
    annotation-only keys (title/default) the format does not need.
    """
    root = model.model_json_schema()
    defs = root.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties":
                out[key] = {name: walk(sub) for name, sub in value.items()}
            elif key in ("title", "default") and not isinstance(value, (dict, list)):
                continue
            else:
                out[key] = walk(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(root)


class ClaudeLLM:
    def __init__(self, model: str | None = None, client: Any = None, max_tokens: int = 16000):
        import anthropic

        self.client = client or anthropic.Anthropic(max_retries=3)
        self.model = model or os.environ.get("AUDITOR_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def structured(self, *, system: str, user: str, output: type[T], effort: str = "medium") -> T:
        with CallLog(output.__name__, f"{self.model}, effort={effort}") as call:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # Server-side fallback: if a safety classifier declines (e.g. a filing
                # chunk trips it), the API re-runs on a recommended model instead of failing.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": strict_schema(output)},
                },
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            call.note = f"output_tokens={getattr(getattr(resp, 'usage', None), 'output_tokens', None)}"
        if resp.stop_reason == "refusal":
            raise LLMError(f"Model declined ({getattr(resp.stop_details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Output hit max_tokens; raise max_tokens or shrink the input")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return output.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"Output failed {output.__name__} validation: {exc}") from exc
