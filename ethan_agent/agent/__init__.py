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


def build_team_components() -> dict:
    """The team's stock-agent/ code behind the agent's interfaces, each falling back to
    its stand-in if it can't be loaded. AUDITOR_TEAM_COMPONENTS=0 uses stand-ins only."""
    from .stubs import FixtureRetriever, JsonFileStore, NoopNumberChecker

    fixture = FixtureRetriever(FIXTURES)
    parts = {"retriever": fixture, "store": JsonFileStore(), "number_checker": NoopNumberChecker(),
             "citation_verifier": None}
    if os.environ.get("AUDITOR_TEAM_COMPONENTS", "1").strip() == "0":
        return parts

    from . import integrations as team

    demo_tickers = {c.ticker for c in fixture.chunks}
    builders = {
        "retriever": lambda: team.StockAgentRetriever(fallback=fixture),
        "store": lambda: team.StockAgentStore(skip_tickers=demo_tickers),
        "number_checker": team.StockAgentNumberChecker,
        "citation_verifier": team.StockAgentCitationVerifier,
    }
    log = logging.getLogger("agent")
    for name, build in builders.items():
        try:
            parts[name] = build()
        except Exception as exc:  # missing package, moved folder...: keep the stand-in
            log.warning("Team %s unavailable, using the stand-in: %s", name, exc)
    return parts


def build_default_auditor(**kwargs) -> ThesisAuditor:
    """LLM (per AUDITOR_PROVIDER) + the team's search, checks and storage."""
    llm = make_llm()
    if "max_workers" not in kwargs:
        # Claims checked in parallel by default; one at a time in Gemini fast mode, where
        # free-tier rate limits make bursts slower than going sequentially.
        default = 1 if getattr(llm, "fast", False) else 4
        kwargs["max_workers"] = int(os.environ.get("AUDITOR_MAX_WORKERS", default))

    parts = build_team_components()
    logging.getLogger("agent").info(
        "Auditor ready: provider=%s model=%s fast=%s claims_in_parallel=%s | %s",
        type(llm).__name__, llm.model, getattr(llm, "fast", False), kwargs["max_workers"],
        ", ".join(f"{k}={type(v).__name__}" for k, v in parts.items()),
    )
    return ThesisAuditor(llm=llm, **parts, **kwargs)


__all__ = ["ThesisAuditor", "ThesisAudit", "build_default_auditor", "build_team_components", "make_llm"]
