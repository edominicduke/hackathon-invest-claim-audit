from agent.schemas import Claim, ClaimResult, ClaimStatus, ClaimType, Importance, VerdictLevel
from agent.verdict import compute_verdict

S, C, U = ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED, ClaimStatus.UNVERIFIABLE
CORE, SUP = Importance.CORE, Importance.SUPPORTING


def r(type_: ClaimType, importance: Importance, status: ClaimStatus, text: str = "claim") -> ClaimResult:
    claim = Claim(
        id="c", text=text, type=type_, importance=importance, stated_by_user=True,
        support_queries=[], challenge_queries=[], clarifying_question=None,
    )
    return ClaimResult(claim=claim, status=status, llm_status=status, reasoning="", evidence=[])


def test_solid_thesis_can_be_green_even_with_unprovable_forecast():
    # The "doesn't always say no" guarantee: forecasts are never required to be supported.
    v = compute_verdict([r(ClaimType.FACT, CORE, S), r(ClaimType.FORECAST, CORE, U)])
    assert v.level == VerdictLevel.GREEN


def test_core_fact_contradicted_is_red():
    v = compute_verdict([r(ClaimType.FACT, CORE, C, "grew 50%"), r(ClaimType.FACT, CORE, S)])
    assert v.level == VerdictLevel.RED
    assert "grew 50%" in v.reasons[0]


def test_core_forecast_contradicted_is_red():
    assert compute_verdict([r(ClaimType.FACT, CORE, S), r(ClaimType.FORECAST, CORE, C)]).level == VerdictLevel.RED


def test_unverifiable_never_causes_red():
    for status_mix in ([U], [U, U], [S, U]):
        v = compute_verdict([r(ClaimType.FACT, CORE, s) for s in status_mix])
        assert v.level == VerdictLevel.YELLOW


def test_only_forecasts_is_yellow():
    assert compute_verdict([r(ClaimType.FORECAST, CORE, U)]).level == VerdictLevel.YELLOW


def test_contradicted_supporting_claim_blocks_green_but_not_red():
    v = compute_verdict([r(ClaimType.FACT, CORE, S), r(ClaimType.FACT, SUP, C)])
    assert v.level == VerdictLevel.YELLOW
    assert v.rule == "supporting_claim_contradicted"


def test_supporting_unverifiable_does_not_block_green():
    v = compute_verdict([r(ClaimType.FACT, CORE, S), r(ClaimType.VAGUE, SUP, U)])
    assert v.level == VerdictLevel.GREEN


def test_no_core_tagged_treats_all_as_core():
    assert compute_verdict([r(ClaimType.FACT, SUP, C)]).level == VerdictLevel.RED
