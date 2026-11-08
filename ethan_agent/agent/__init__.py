"""Investment Thesis Auditor - agent reasoning, Challenge My Trade, and explanations."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .pipeline import ThesisAuditor
from .schemas import ThesisAudit

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def make_llm(provider: str | None = None):
    """Pick the model provider: $env:AUDITOR_PROVIDER = "claude" (default) or "gemini"."""
    provider = (provider or os.environ.get("AUDITOR_PROVIDER", "claude")).strip().lower()
    if provider == "gemini":
        from .llm_gemini import GeminiLLM  # imported lazily: google-genai is optional

        return GeminiLLM()
    if provider == "claude":
        from .llm import ClaudeLLM

        return ClaudeLLM()
    raise ValueError(f"Unknown AUDITOR_PROVIDER {provider!r}; use 'claude' or 'gemini'")


def build_default_auditor(**kwargs) -> ThesisAuditor:
    """LLM (per AUDITOR_PROVIDER) + fixture retriever + JSON store. Replace the stubs
    with teammates' versions."""
    from .stubs import FixtureRetriever, JsonFileStore, NoopNumberChecker

    llm = make_llm()
    if "max_workers" not in kwargs:
        # Claims checked in parallel by default; one at a time in Gemini fast mode, where
        # free-tier rate limits make bursts slower than going sequentially.
        default = 1 if getattr(llm, "fast", False) else 4
        kwargs["max_workers"] = int(os.environ.get("AUDITOR_MAX_WORKERS", default))

    logging.getLogger("agent").info(
        "Auditor ready: provider=%s model=%s fast=%s claims_in_parallel=%s",
        type(llm).__name__, llm.model, getattr(llm, "fast", False), kwargs["max_workers"],
    )
    return ThesisAuditor(
        llm=llm,
        retriever=FixtureRetriever(FIXTURES),
        store=JsonFileStore(),
        number_checker=NoopNumberChecker(),
        **kwargs,
    )


__all__ = ["ThesisAuditor", "ThesisAudit", "build_default_auditor", "make_llm"]
