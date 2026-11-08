"""Evaluate the agent on what it controls (Stage 6) - not on stock returns.

Claim level: labeled claims built by *perturbation*. Start from claims the corpus
states (gold = supported), break their numbers or direction words (gold =
contradicted), and add claims the corpus is silent on (gold = unverifiable). No
financial expert needed to label.

Headline metrics:
  * false-support rate  P(pred=supported | gold!=supported)  - the costliest error
  * supported precision / recall - recall is the "doesn't always say no" test
  * contradicted recall

Verdict level: a few theses with expected colours, incl. one that should be GREEN.

Runs real API calls (Claude Opus 5.5 by default; ~2 calls per claim, ~10-15 per
thesis). Usage:
    python -m eval.run_eval              # claim-level only
    python -m eval.run_eval --verdicts   # also run full audits
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from agent import build_default_auditor
from agent.llm import enable_call_logging
from agent.schemas import Claim, ClaimStatus, ParsedThesis

S, C, U = ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED, ClaimStatus.UNVERIFIABLE

# True statements from data/fixtures/exmp_demo.json (fictional company).
SEEDS = [
    "EXMP's Data Center segment revenue was $18.4 billion in fiscal 2025.",
    "EXMP's Data Center revenue grew 61% in fiscal 2025 compared with fiscal 2024.",
    "EXMP's gross margin was 71.2% in fiscal 2025.",
    "EXMP's largest customer accounted for 23% of total revenue in fiscal 2025.",
    "EXMP's Gaming segment revenue decreased 4% in fiscal 2025.",
    "EXMP's Data Center revenue grew 44% year over year in the first quarter of fiscal 2026.",
    "EXMP's inventory increased to $4.2 billion at the end of fiscal 2025.",
]

# Hard cases written by hand: period/scope traps, forward-looking-as-fact, analyst and competitor traps.
HARD = [
    ("EXMP's Data Center revenue grew 61% year over year in the first quarter of fiscal 2026.", C),
    ("EXMP's total revenue was $18.4 billion in fiscal 2025.", C),
    ("EXMP reported that Data Center growth slowed in the second half of fiscal 2026.", U),
    # Analyst-question trap: only an analyst's question says "more than 80%"; the 10-K says 61%.
    ("EXMP's data center revenue is growing more than 80% a year.", C),
    # Competitor trap: Rival's own 10-K boasts "best performance per dollar" - a self-claim, not a fact.
    ("Rival Micro's RM-9 offers the best performance per dollar in the industry.", U),
]

SILENT = [
    "EXMP's CEO has led the company for more than 20 years.",
    "EXMP announced a special dividend for 2026.",
    "EXMP's automotive segment revenue doubled in fiscal 2025.",
]

_NUM = re.compile(r"\d+(?:\.\d+)?")
_FLIPS = {"increased": "decreased", "decreased": "increased", "grew": "shrank"}


def perturb(text: str) -> str:
    """Break a true claim: flip its direction word, else scale its first number by 0.75x (plausible, not absurd)."""
    for word, flipped in _FLIPS.items():
        if f" {word} " in text:
            return text.replace(f" {word} ", f" {flipped} ", 1)
    m = _NUM.search(text)
    value = float(m.group()) * 0.75
    new = f"{value:.1f}" if "." in m.group() else str(round(value))
    return text[: m.start()] + new + text[m.end():]


def labeled_claims() -> list[tuple[str, ClaimStatus]]:
    return [(s, S) for s in SEEDS] + [(perturb(s), C) for s in SEEDS] + HARD + [(s, U) for s in SILENT]


THESES = [
    ("I'm buying EXMP because AI demand drove data center revenue up more than 60% last year, "
     "and data center is now most of the company's revenue.", "green"),
    ("Buying EXMP because its gaming business is booming and grew 20% last year.", "red"),
    ("EXMP will keep growing data center revenue 50%+ for years because nobody can compete with them.", "yellow|red"),
]


def claim_eval(auditor) -> None:
    parsed = ParsedThesis(
        ticker="EXMP", company_name="Exemplar Semiconductor", other_tickers=[], direction="buy",
        horizon_months=None, entry_price=None, ambiguities=[], is_auditable=True, not_auditable_reason=None,
    )
    rows = []
    for i, (text, gold) in enumerate(labeled_claims()):
        claim = Claim(
            id=f"e{i}", text=text, type="fact", importance="core", stated_by_user=True,
            support_queries=[text], challenge_queries=[f"{text} decrease decline"], clarifying_question=None,
        )
        result, _ = auditor.check_claim(claim, parsed)
        rows.append((text, gold, result.status))
        mark = "ok " if gold == result.status else "BAD"
        print(f"[{mark}] gold={gold.value:<12} pred={result.status.value:<12} {text}")

    confusion = Counter((g, p) for _, g, p in rows)

    def prf(cls):
        tp = confusion[(cls, cls)]
        pred = sum(v for (g, p), v in confusion.items() if p == cls)
        gold = sum(v for (g, p), v in confusion.items() if g == cls)
        return (tp / pred if pred else float("nan")), (tp / gold if gold else float("nan"))

    not_supported = [r for r in rows if r[1] != S]
    false_support = sum(1 for r in not_supported if r[2] == S) / len(not_supported)
    sp, sr = prf(S)
    _, cr = prf(C)
    print("\n== Claim-level ==")
    print(f"accuracy             {sum(g == p for _, g, p in rows) / len(rows):.2f}  (n={len(rows)})")
    print(f"false-support rate   {false_support:.2f}   target <= 0.05")
    print(f"supported precision  {sp:.2f}   target >= 0.95")
    print(f"supported recall     {sr:.2f}   target >= 0.80  ('doesn't always say no')")
    print(f"contradicted recall  {cr:.2f}")


def verdict_eval(auditor) -> None:
    print("\n== Verdict-level ==")
    for thesis, expected in THESES:
        audit = auditor.audit(thesis)
        got = audit.verdict.level.value
        print(f"[{'ok ' if got in expected.split('|') else 'BAD'}] expected={expected:<10} got={got:<7} {thesis[:70]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", action="store_true", help="also run full audits (more API calls)")
    args = ap.parse_args()
    enable_call_logging()
    auditor = build_default_auditor(verify_entailment=True)
    claim_eval(auditor)
    if args.verdicts:
        verdict_eval(auditor)
