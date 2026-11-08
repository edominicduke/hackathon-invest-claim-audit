"""Deterministic citation checks: no LLM involved.

Two citation channels, two checks:

* Structured evidence ``{source: "S3", quote, stance}`` - for verification.
  The label must be one shown to the model in that call, and the quote must appear
  in that source (verbatim after normalisation, or >=90% near-exact).
* Inline ``[S3]`` labels in prose - for attribution. Unknown labels are stripped;
  sentences that state a number with no label are flagged.

What this CANNOT catch: a real quote that doesn't support the claim (wrong period,
wrong segment, forward-looking cited as fact). That is the entailment check's job.
Nor a factual sentence with no number and no label - prose checks are a net, not proof.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .interfaces import Chunk
from .schemas import Evidence

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
LABEL = re.compile(r"\[(S\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_NUMBER = re.compile(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?")  # skips "Q1", "FY2026", "S3"
_YEAR = re.compile(r"^(19|20)\d\d$")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


def quote_in_text(quote: str, text: str, threshold: float = 0.9) -> bool:
    q, t = normalize(quote), normalize(text)
    if not q:
        return False
    if q in t:
        return True
    match = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    return match.size / len(q) >= threshold


def check_evidence(evidence: list[Evidence], allowed: dict[str, Chunk]) -> tuple[list[Evidence], list[str]]:
    """Split structured evidence into (valid, issues). ``allowed`` is keyed by label."""
    valid, issues = [], []
    for ev in evidence:
        label = ev.source.strip().strip("[]")
        chunk = allowed.get(label)
        if chunk is None:
            issues.append(f"Dropped citation [{label}]: not one of the sources provided")
        elif not quote_in_text(ev.quote, chunk.text):
            issues.append(f"Dropped citation [{label}]: quote not found in that source")
        else:
            valid.append(ev.model_copy(update={"source": label}))
    return valid, issues


def labels_in(text: str) -> set[str]:
    return set(LABEL.findall(text))


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check_inline(text: str, allowed: set[str], ignore_numbers_from: str = "") -> tuple[str, list[str]]:
    """Strip unknown [S#] labels; flag sentences stating a number with no label.

    Numbers the user wrote in their own thesis, and bare years, are not flagged:
    restating "you said 50%+" or "fiscal 2025" needs no source.
    """
    issues = []
    for bad in sorted(labels_in(text) - allowed):
        issues.append(f"Removed invented citation [{bad}]")
        text = text.replace(f"[{bad}]", "")
    text = re.sub(r" +([.,;:])", r"\1", re.sub(r"  +", " ", text)).strip()

    exempt = _numbers(ignore_numbers_from)
    for sentence in _SENTENCE.split(text):
        if LABEL.search(sentence):
            continue
        uncited = {n for n in _numbers(sentence) if n not in exempt and not _YEAR.match(n)}
        if uncited:
            issues.append(f"Number(s) {', '.join(sorted(uncited))} stated without a source: \"{sentence.strip()[:120]}\"")
    return text, issues
