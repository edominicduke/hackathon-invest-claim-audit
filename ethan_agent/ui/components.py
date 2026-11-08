"""Shared UI pieces for every page: constants, escaping/markup helpers, the cached
auditor and the stylesheet loader. Page scripts live in views/."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import streamlit as st

import agent
from agent.schemas import ClaimStatus

log = logging.getLogger("agent.app")
STYLES = Path(__file__).resolve().parent.parent / "styles" / "theme.css"

# ------------------------------------------------------------------ constants

TONE_ICON = {"good": "✓", "warn": "◐", "bad": "✕", "neutral": "?"}  # icon + word + colour, never colour alone

VERDICT = {  # label, explanation, tone. The rating grades evidence for the reasons, not the price.
    "green": ("Strong evidence", "The filings back every core claim that could be checked.", "good"),
    "yellow": ("Mixed evidence", "Some core claims couldn't be confirmed from the filings.", "warn"),
    "red": ("Contradicted", "The filings contradict at least one core claim.", "bad"),
    "none": ("Can't audit this thesis", "", "neutral"),
}
STATUS = {  # label, tone
    ClaimStatus.SUPPORTED: ("Supported", "good"),
    ClaimStatus.CONTRADICTED: ("Contradicted", "bad"),
    ClaimStatus.UNVERIFIABLE: ("Not verifiable", "neutral"),
}
TYPE_LABEL = {"fact": "Fact", "causal": "Cause", "forecast": "Forecast", "vague": "Needs definition", "premise": "Assumption"}
DIRECTION_LABEL = {"buy": "Buy", "sell": "Sell", "short": "Short", "hold": "Hold", "unclear": "Unclear"}

EXAMPLES = {
    "EXMP · AI demand": (
        "I'm buying EXMP because AI demand will keep data center revenue growing 50%+ "
        "and they have no real competition."
    ),
    "EXMP · Data center": (
        "I'm buying EXMP because AI demand drove data center revenue up more than 60% last year, "
        "and data center is now most of the company's revenue."
    ),
    "EXMP · Gaming +20%": "Buying EXMP because its gaming business is booming and grew 20% last year.",
    "AMD · AI chip share": (
        "I'm buying AMD because its data center business is growing fast on demand for its "
        "Instinct AI accelerators, and it keeps taking server CPU share from Intel."
    ),
}
CHECK_LABEL = {  # Guru's sentence statuses -> (label, tone)
    "supported": ("Supported", "good"),
    "cited": ("Citation valid", "good"),
    "partial": ("Partly supported", "warn"),
    "weak": ("Weakly supported", "warn"),
    "unsupported": ("Not supported", "bad"),
    "invalid_citation": ("Invalid citation", "bad"),
    "uncited_number": ("Number without source", "bad"),
    "uncited_claim": ("Claim without source", "warn"),
    "interpretation": ("Interpretation", "neutral"),  # framed analysis, uncited by design
    "no_claim": ("No factual claim", "neutral"),
}
FAILING = {"unsupported", "invalid_citation", "uncited_number"}

# -------------------------------------------------------------------- helpers


def esc(text: object) -> str:
    return html.escape(str(text or ""))


def md(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def pill(text: str, tone: str) -> str:
    return f'<span class="ta-pill tone-{tone}">{TONE_ICON.get(tone, "")} {esc(text)}</span>'


def tag(text: str) -> str:
    return f'<span class="ta-tag">{esc(text)}</span>'


@st.cache_resource
def auditor():
    # Looked up on the module at call time, so tests can monkeypatch agent.build_default_auditor.
    return agent.build_default_auditor()


def reset() -> None:
    for key in ("parsed", "audit", "review"):
        st.session_state.pop(key, None)


def load_example(text: str) -> None:
    st.session_state.raw = text
    reset()


def analyst() -> bool:
    return st.session_state.get("view", "Plain English") == "Analyst"


def load_css() -> None:
    """All custom CSS lives in styles/theme.css; injected once per run."""
    md("<style>" + STYLES.read_text(encoding="utf-8") + "</style>")
