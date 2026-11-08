"""Number check (#3): does every number in the AI answer appear in its sources?

check_numbers(answer, sources) ->
  {"total": 3, "found": 2, "all_found": False,
   "numbers": [{"text": "$4.2 billion", "found": True, "source": "10-Q p.8", "matched": "$4,213 million"}, ...],
   "not_found": ["31%"]}

- Understands $, %, x (multiples), commas, and units: thousand/million/billion/trillion, K/M/B/bn/mn.
  "$4.2B" = "$4.2 billion" = "$4,200 million".
- Allows rounding: "$4.2 billion" matches "$4,213 million" because 4.213 rounds to 4.2.
- Ignores citations in [brackets], years (2026), quarters (Q3), and form names (10-K).
- check_numbers_with_sec() also compares against official SEC numbers (see sec_facts.py).
  Each number says "official_sec": true when it matched an official SEC number.
"""

from __future__ import annotations

import re

SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6, "mm": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}

NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9.\-])                       # not part of a word, Q3, 10-K, p.8
    (?P<dollar>\$\s?)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<unit>
        \s?%|\s?percent(?:age\s+points?)?\b
      | \s?(?:thousand|million|billion|trillion)\b
      | (?:bn|mn|mm|tn|[kKmMbBtT])\b
      | x\b
    )?
    (?![A-Za-z0-9\-])                         # not 10-Q, not 3rd
    """,
    re.VERBOSE,
)


def _parse(match: re.Match) -> dict | None:
    raw_num = match.group("num")
    unit = (match.group("unit") or "").strip().lower()
    has_dollar = bool(match.group("dollar"))
    decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
    value = float(raw_num.replace(",", ""))

    if unit == "%" or unit.startswith("percent"):
        kind, scale = "percent", 1.0
    elif unit == "x":
        kind, scale = "multiple", 1.0
    elif unit:
        kind, scale = ("money" if has_dollar else "amount"), SCALES[unit]
    else:
        kind, scale = ("money" if has_dollar else "plain"), 1.0
        # A bare 4-digit number between 1900 and 2100 is almost always a year.
        if kind == "plain" and decimals == 0 and "," not in raw_num and 1900 <= value <= 2100:
            return None

    # Half of the last written digit = how much rounding the writer may have done.
    tolerance = 0.5 * (10 ** -decimals) * scale
    return {
        "text": match.group(0).strip(),
        "value": value * scale,
        "kind": kind,
        "tolerance": tolerance,
    }


TIME_WORDS = re.compile(r"\s*(?:days?|weeks?|months?|quarters?|years?)\b", re.IGNORECASE)


def find_numbers(text: str) -> list[dict]:
    text = re.sub(r"\[[^\]]*\]", " ", text or "")  # drop citations like [10-Q, p.8]
    out = []
    for m in NUMBER_RE.finditer(text):
        parsed = _parse(m)
        # "12 months", "3 years": time spans, not financial facts to check.
        if parsed and parsed["kind"] == "plain" and TIME_WORDS.match(text, m.end()):
            continue
        if parsed:
            out.append(parsed)
    return out


COMPATIBLE = {
    "percent": {"percent"},
    "multiple": {"multiple", "plain"},
    "money": {"money", "amount", "plain"},
    "amount": {"money", "amount", "plain"},
    "plain": {"plain", "money", "amount", "multiple"},
}


def _matches(claim: dict, fact: dict) -> bool:
    if fact["kind"] not in COMPATIBLE[claim["kind"]]:
        return False
    allowed = max(claim["tolerance"], fact["tolerance"], 1e-9)
    return abs(claim["value"] - fact["value"]) <= allowed + 1e-9 * abs(fact["value"])


def check_numbers(answer: str, sources: list) -> dict:
    """sources: list of {"label": "10-Q p.8", "text": "...passage..."} (plain strings also work)."""
    facts = []
    for i, src in enumerate(sources or []):
        if isinstance(src, str):
            src = {"label": f"source {i + 1}", "text": src}
        for f in find_numbers(src.get("text") or ""):
            facts.append({**f, "label": src.get("label") or f"source {i + 1}"})

    numbers = []
    for claim in find_numbers(answer):
        hit = next((f for f in facts if _matches(claim, f)), None)
        numbers.append({
            "text": claim["text"],
            "kind": claim["kind"],
            "found": hit is not None,
            "source": hit["label"] if hit else None,
            "matched": hit["text"] if hit else None,
        })

    for n in numbers:
        n["official_sec"] = bool(n["source"] and n["source"].startswith("SEC "))

    found = sum(1 for n in numbers if n["found"])
    return {
        "total": len(numbers),
        "found": found,
        "all_found": found == len(numbers),
        "numbers": numbers,
        "not_found": [n["text"] for n in numbers if not n["found"]],
        "note": None if facts else "No source text was given, so no number could be checked.",
    }


def check_numbers_with_sec(answer: str, sources: list, ticker: str, sec=None, today=None) -> dict:
    """Same as check_numbers, but also compares against official SEC numbers for the company.

    If the SEC can't be reached, it still checks against the normal sources and
    explains why in "sec_error".
    """
    from src.checks.sec_facts import SecError, SecFacts

    sec = sec or SecFacts()
    try:
        official, error = sec.as_sources(ticker, today), None
    except SecError as exc:
        official, error = [], str(exc)

    result = check_numbers(answer, list(sources or []) + official)
    result["sec_numbers_used"] = len(official)
    result["sec_error"] = error
    return result
