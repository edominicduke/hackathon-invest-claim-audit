"""Rule-based verdict (Stage 5). The LLM supplies judgment (claim status, core vs
supporting); plain code makes the call, so the verdict is deterministic, explainable
and unit-testable - and immune to the "Challenge My Trade" framing pushing it negative.

What the colours mean: how well the *thesis* holds up against evidence in filings.
Not a price prediction, not advice.

    RED     any core claim is contradicted by specific, cited evidence
    GREEN   every core checkable claim (fact/causal/premise) is supported,
            at least one exists, and no claim at all is contradicted
    YELLOW  everything else (incl. a contradicted *supporting* claim)

Anti-"always says no" guarantees encoded here:
* ``unverifiable`` can cap a thesis at YELLOW but can never cause RED
  (absence of evidence is not evidence against; retrieval gaps aren't the user's fault).
* Forecasts are never required to be "supported" for GREEN - they can't be.
* Counterarguments are shown to the user but do not feed the verdict;
  contradiction only counts through a claim's status, which requires cited evidence.
"""

from __future__ import annotations

from .schemas import CHECKABLE_TYPES, ClaimResult, ClaimStatus, Importance, Verdict, VerdictLevel


def compute_verdict(results: list[ClaimResult]) -> Verdict:
    core = [r for r in results if r.claim.importance == Importance.CORE]
    if not core:  # model tagged nothing core: treat every claim as core rather than guess
        core = results

    contradicted = [r for r in core if r.status == ClaimStatus.CONTRADICTED]
    if contradicted:
        return Verdict(
            level=VerdictLevel.RED,
            rule="core_claim_contradicted",
            reasons=[f"Contradicted by the filings: {r.claim.text}" for r in contradicted],
        )

    checkable = [r for r in core if r.claim.type in CHECKABLE_TYPES]
    unsupported = [r for r in checkable if r.status != ClaimStatus.SUPPORTED]
    minor_contradicted = [r for r in results if r.status == ClaimStatus.CONTRADICTED]
    if minor_contradicted:  # a refuted supporting claim blocks GREEN but isn't fatal
        return Verdict(
            level=VerdictLevel.YELLOW,
            rule="supporting_claim_contradicted",
            reasons=[f"A supporting claim is contradicted by the filings: {r.claim.text}" for r in minor_contradicted]
            + [f"Couldn't verify from the filings: {r.claim.text}" for r in unsupported],
        )
    if checkable and not unsupported:
        reasons = [f"Supported by the filings: {r.claim.text}" for r in checkable]
        forecasts = [r for r in core if r.claim.type not in CHECKABLE_TYPES]
        if forecasts:
            reasons.append(
                f"{len(forecasts)} forward-looking or vague claim(s) can't be proven from filings, "
                "but nothing in the filings contradicts them"
            )
        return Verdict(level=VerdictLevel.GREEN, rule="core_facts_supported", reasons=reasons)

    if not checkable:
        reasons = ["The thesis rests only on forecasts or vague claims, which filings can't confirm"]
    else:
        reasons = [f"Couldn't verify from the filings: {r.claim.text}" for r in unsupported]
    return Verdict(level=VerdictLevel.YELLOW, rule="core_claims_unverified", reasons=reasons)
