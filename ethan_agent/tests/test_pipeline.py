"""End-to-end pipeline tests with a scripted LLM: they test the code-side enforcement
(citations, labels, status rules, balancing, verdict), not the model's judgment."""

import json

import pytest

from agent import FIXTURES
from agent.interfaces import Chunk
from agent.pipeline import SourceRegistry, ThesisAuditor, format_sources
from agent.schemas import (
    AnswerReview,
    Challenge,
    ClaimJudgment,
    ClaimStatus,
    Counterargument,
    Decomposition,
    DraftClaim,
    EntailmentItem,
    EntailmentReport,
    Evidence,
    Explanation,
    Materiality,
    ParsedThesis,
    StressTest,
    UserAnswer,
    VerdictLevel,
)
from agent.stubs import FixtureRetriever, JsonFileStore
from tests.fakes import FakeLLM, by_claim, cite, label_for

THESIS = "Buying EXMP: AI demand keeps data center revenue growing 50%+ and they have no real competition."

PARSED = ParsedThesis(
    ticker="exmp", company_name=None, other_tickers=[], direction="buy", horizon_months=None,
    entry_price=None, ambiguities=[], is_auditable=True, not_auditable_reason=None,
)

FACT = DraftClaim(
    text="EXMP data center revenue grew 61% in fiscal 2025", type="fact", importance="core",
    stated_by_user=True,
    support_queries=["data center segment revenue fiscal 2025 increase", "data center revenue growth", "extra"],
    challenge_queries=["data center revenue decrease"],  # unbalanced on purpose
    clarifying_question=None,
)
FORECAST = DraftClaim(
    text="EXMP data center revenue will keep growing 50%+", type="forecast", importance="core",
    stated_by_user=True, support_queries=["demand for AI infrastructure"],
    challenge_queries=["data center growth moderate second half"], clarifying_question=None,
)
VAGUE = DraftClaim(
    text="EXMP has no real competition", type="vague", importance="supporting", stated_by_user=True,
    support_queries=["market share leadership"], challenge_queries=["competitors accelerators RM-9"],
    clarifying_question="By 'no real competition', do you mean today or in 3 years?",
)

FACT_QUOTE = "an increase of 61% compared with $11.4 billion for fiscal 2024"


def judge_fact(user):
    s = label_for(user, FACT_QUOTE)
    return ClaimJudgment(
        status="supported", reasoning=f"Data center revenue rose 61% in fiscal 2025 [{s}].",
        evidence=[cite(user, FACT_QUOTE)], follow_up_query=None,
    )


def judge_forecast(user):  # model wrongly marks a forecast "supported" -> code downgrades
    return ClaimJudgment(
        status="supported", reasoning="Management still sees strong demand.",
        evidence=[cite(user, "we continue to see strong demand for AI infrastructure")], follow_up_query=None,
    )


def judge_vague(user):  # fabricated quote + invented label + uncited number
    s = label_for(user, "RM-9")
    return ClaimJudgment(
        status="supported",
        reasoning=f"Rival Micro shipped the RM-9 [{s}]. EXMP holds 90% market share [S99].",
        evidence=[Evidence(source=s, quote="we have no meaningful competitors", stance="supports")],
        follow_up_query=None,
    )


JUDGMENTS = {"grew 61%": judge_fact, "will keep growing": judge_forecast, "no real competition": judge_vague}


def challenge(_system, user):
    return Challenge(
        strongest_point_for=f"Data center revenue grew 61% [{label_for(user, FACT_QUOTE)}].",
        counterarguments=[
            Counterargument(  # claims "specific" but cites a label that doesn't exist -> generic
                text="Customers are building their own chips [S99].", materiality="specific",
                evidence=[Evidence(source="S99", quote="custom chips", stance="contradicts")],
            ),
            Counterargument(
                text=f"Growth already slowed to 44% [{label_for(user, 'up 44% from a year ago')}].",
                materiality="specific",
                evidence=[cite(user, "up 44% from a year ago", "contradicts")],
            ),
        ],
        stress_test=StressTest(scenario="Growth could slow to 20% next year.", what_would_have_to_be_true=["x"], early_warning_signs=["y"]),
        questions_for_user=[],  # model forgot the clarifying question -> code appends it
    )


def explain(_system, user):
    fact_label = next(l for l in ["S1", "S2", "S3", "S4"] if f"[{l}]" in user)
    return Explanation(plain_summary=f"The filings back the revenue claim [{fact_label}]. See also [S77].",
                       key_takeaways=["a", "b", "c"])


def make_llm(entail=None, judgments=None) -> FakeLLM:
    return FakeLLM({
        ParsedThesis: PARSED,
        Decomposition: Decomposition(claims=[FACT, FORECAST, VAGUE]),
        ClaimJudgment: by_claim(judgments or JUDGMENTS),
        EntailmentReport: entail or (lambda _s, user: EntailmentReport(
            items=[EntailmentItem(index=i, entails=True, problem=None) for i in range(user.count("<item "))]
        )),
        Challenge: challenge,
        Explanation: explain,
        AnswerReview: AnswerReview(notes="n", clarified_claims=[], new_concerns=[]),
    })


@pytest.fixture
def retriever():
    return FixtureRetriever(FIXTURES)


def run(llm, retriever, tmp_path):
    auditor = ThesisAuditor(llm, retriever, store=JsonFileStore(tmp_path), max_workers=1)
    return auditor, auditor.audit(THESIS)


def claim(audit, draft):
    return next(r for r in audit.claims if r.claim.text == draft.text)


def test_code_enforcement_and_green_verdict(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    assert audit.parsed.ticker == "EXMP"  # normalised
    assert claim(audit, FACT).status == ClaimStatus.SUPPORTED
    forecast = claim(audit, FORECAST)
    assert forecast.llm_status == ClaimStatus.SUPPORTED and forecast.status == ClaimStatus.UNVERIFIABLE
    vague = claim(audit, VAGUE)
    assert vague.status == ClaimStatus.UNVERIFIABLE and not vague.evidence
    assert any("quote not found" in i for i in vague.issues)
    # Core fact supported + forecast unverifiable -> GREEN. The tool can say yes.
    assert audit.verdict.level == VerdictLevel.GREEN


def test_inline_labels_checked_in_reasoning(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    vague = claim(audit, VAGUE)
    assert "[S99]" not in vague.reasoning
    assert any("invented citation [S99]" in i for i in vague.issues)
    assert any("90" in i and "without a source" in i for i in vague.issues)
    # The thesis's own "50" is never flagged as uncited.
    assert not any("50 " in i for r in audit.claims for i in r.issues)


def test_inline_labels_checked_in_challenge_and_summary(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    assert "[S99]" not in audit.challenge.counterarguments[0].text
    assert "[S77]" not in audit.explanation.plain_summary
    joined = " ".join(audit.citation_issues)
    assert "[S99]" in joined and "[S77]" in joined
    assert "20" in joined  # uncited "20%" in the stress test is flagged


def test_labels_are_audit_wide_and_resolvable(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    table = {s.label: s.chunk_id for s in audit.sources}
    fact_ev = claim(audit, FACT).evidence[0]
    assert table[fact_ev.source] == "EXMP-10K-FY2025-012"
    # The same chunk cited from the challenge step resolves to the same label.
    assert audit.challenge.strongest_point_for.endswith(f"[{fact_ev.source}].")
    # Every label used anywhere is in the table.
    used = {e.source for r in audit.claims for e in r.evidence}
    assert used <= set(table)


def test_registry_reuses_labels():
    reg = SourceRegistry()
    a = Chunk(chunk_id="a", ticker="X", doc_type="10-K", text="t")
    b = Chunk(chunk_id="b", ticker="X", doc_type="10-K", text="t")
    assert [reg.label(a), reg.label(b), reg.label(a)] == ["S1", "S2", "S1"]


def test_competitor_and_analyst_sources_rendered_for_model(retriever):
    sources = {"S1": next(c for c in retriever.chunks if c.ticker == "RVMC")}
    assert 'company="RVMC" relation="COMPETITOR"' in format_sources(sources, "EXMP")
    own = {"S1": next(c for c in retriever.chunks if c.chunk_id == "EXMP-10K-FY2025-012")}
    assert "COMPETITOR" not in format_sources(own, "EXMP")
    # The stub retriever returns competitor chunks tagged as related to EXMP.
    assert any(c.ticker == "RVMC" for c in retriever.search("RM-9 accelerator shipments", ticker="EXMP"))
    assert not retriever.search("RM-9 accelerator shipments", ticker="AAPL")


def test_balanced_queries(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    fact = claim(audit, FACT)
    assert len(fact.claim.support_queries) == len(fact.claim.challenge_queries) == 1


def test_challenge_checks(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    cas = audit.challenge.counterarguments
    assert cas[0].materiality == Materiality.GENERIC and not cas[0].evidence
    assert cas[1].materiality == Materiality.SPECIFIC
    assert any("no real competition" in q.question for q in audit.challenge.questions_for_user)


def test_sources_glossary_and_storage(retriever, tmp_path):
    _, audit = run(make_llm(), retriever, tmp_path)
    cited = {s.chunk_id for s in audit.sources}
    assert {"EXMP-10K-FY2025-012", "EXMP-10Q-Q1FY2026-004"} <= cited
    assert {"revenue", "10-K"} <= {g.term for g in audit.glossary}
    saved = json.loads((tmp_path / "audits" / f"{audit.audit_id}.json").read_text())
    assert saved["verdict"]["level"] == "green"


def test_failed_entailment_downgrades_claim(retriever, tmp_path):
    reject_all = lambda _s, user: EntailmentReport(  # noqa: E731
        items=[EntailmentItem(index=i, entails=False, problem="wrong period") for i in range(user.count("<item "))]
    )
    _, audit = run(make_llm(entail=reject_all), retriever, tmp_path)
    fact = claim(audit, FACT)
    assert fact.status == ClaimStatus.UNVERIFIABLE
    assert any("wrong period" in i for i in fact.issues)
    assert audit.verdict.level == VerdictLevel.YELLOW


def test_bounded_follow_up_searches(retriever, tmp_path):
    stuck = ClaimJudgment(status="unverifiable", reasoning="Unclear.", evidence=[], follow_up_query="export license regions")
    llm = make_llm(judgments={**JUDGMENTS, "grew 61%": stuck})
    _, audit = run(llm, retriever, tmp_path)
    fact_calls = sum(1 for n, u in llm.calls if n == "ClaimJudgment" and "grew 61%" in u)
    assert 1 < fact_calls <= 3  # 1 initial + at most 2 follow-ups
    assert "export license regions" in claim(audit, FACT).searches


def test_not_auditable_skips_pipeline(retriever, tmp_path):
    llm = make_llm()
    llm.handlers[ParsedThesis] = PARSED.model_copy(update={"ticker": None, "is_auditable": False, "not_auditable_reason": "Private company"})
    _, audit = run(llm, retriever, tmp_path)
    assert audit.status == "not_auditable" and audit.verdict.level == VerdictLevel.NONE
    assert llm.count("Decomposition") == 0


def test_answers_saved_and_verdict_unchanged(retriever, tmp_path):
    auditor, audit = run(make_llm(), retriever, tmp_path)
    before = audit.verdict.level
    auditor.review_answers(audit, [UserAnswer(question="q", answer="I'd sell if growth drops below 30%")])
    assert (tmp_path / "answers" / f"{audit.audit_id}.json").exists()
    assert audit.verdict.level == before
