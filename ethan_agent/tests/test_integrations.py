"""The team's stock-agent/ code wired into the agent. No API calls: keyword search,
structural citation checks only, no SEC downloads."""

import json

import pytest

from agent import FIXTURES, build_team_components
from agent.integrations import (
    StockAgentCitationVerifier,
    StockAgentNumberChecker,
    StockAgentRetriever,
    StockAgentStore,
    hit_to_chunk,
    recommendation_payload,
)
from agent.interfaces import Chunk
from agent.pipeline import ThesisAuditor, format_sources
from agent.schemas import UserAnswer, VerdictLevel
from agent.stubs import FixtureRetriever, JsonFileStore
from tests.test_pipeline import THESIS, make_llm


@pytest.fixture(scope="module")
def retriever():
    return StockAgentRetriever(fallback=FixtureRetriever(FIXTURES), mode="keyword", competitor_k=1)


def fake_prices(growth_per_day: float = 0.1):
    """Bogdan's PriceBook with made-up prices: the stock rises, SPY stays flat. No network."""
    from datetime import timedelta

    from src.checks.prices import PriceBook

    def fetch(ticker, start, end):
        days = (end - start).days
        out = {}
        for i in range(days):
            d = start + timedelta(days=i)
            out[d] = 100.0 if ticker == "SPY" else 100.0 + growth_per_day * (d.toordinal() % 100000)
        return out

    return PriceBook(fetch=fetch)


def team_auditor(retriever, tmp_path, **store_kwargs):
    store_kwargs.setdefault("record_price", False)
    store = StockAgentStore(tmp_path / "track", archive=JsonFileStore(tmp_path / "archive"), **store_kwargs)
    return ThesisAuditor(
        make_llm(), retriever, store=store,
        number_checker=StockAgentNumberChecker(use_sec=False),
        citation_verifier=StockAgentCitationVerifier(check_support=False),
        max_workers=1,
    )


# ---------------------------------------------------------------- Guru: search


def test_real_index_search_with_competitor(retriever):
    assert {"AMD", "NVDA", "INTC"} <= set(retriever.tickers)
    chunks = retriever.search("data center revenue growth AI accelerators", ticker="AMD", k=3)
    own = [c for c in chunks if c.ticker == "AMD"]
    rivals = [c for c in chunks if c.ticker != "AMD"]
    assert len(own) == 3 and len(rivals) == 1
    assert rivals[0].related_tickers == ["AMD"]
    assert "COMPETITOR" in format_sources({"S1": rivals[0]}, "AMD")


def test_papers_off_by_default(retriever):
    query = "FPGA accelerator neural network inference energy efficiency"
    assert all(c.doc_type != "paper" for c in retriever.search(query, ticker="AMD", k=5))
    with_papers = StockAgentRetriever(mode="keyword", competitor_k=0, include_papers=True)
    assert any(c.doc_type == "paper" for c in with_papers.search(query, ticker="AMD", k=5))


def test_guru_pinned_to_project_even_without_vertex_env(monkeypatch):
    from google import genai
    from src.ingest import embed as guru_embed

    from agent.integrations import pin_guru_to_vertex

    made = []
    monkeypatch.setattr(genai, "Client", lambda **kw: made.append(kw) or object())
    monkeypatch.setattr(guru_embed, "_client", None)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert not pin_guru_to_vertex() and not made
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-123")
    assert pin_guru_to_vertex()
    assert made == [{"vertexai": True, "project": "proj-123", "location": "global"}]


def test_unindexed_ticker_falls_back_to_fixture(retriever):
    chunks = retriever.search("data center revenue", ticker="EXMP", k=2)
    assert chunks and all(c.ticker in ("EXMP", "RVMC") for c in chunks)


def test_hit_mapping_and_paper_relation():
    paper = hit_to_chunk({
        "chunk_id": "p#0", "ticker": "RESEARCH", "doc_type": "paper", "text": "t", "section": "2509.1",
        "date": "2025-09-19", "period": None, "url": "u", "tags": ["AMD", "NVDA"],
    }, "AMD")
    assert paper.filed_date == "2025-09-19" and paper.related_tickers == ["AMD", "NVDA"]
    assert 'relation="RESEARCH"' in format_sources({"S1": paper}, "AMD")
    call = hit_to_chunk({
        "chunk_id": "c#0", "ticker": "AMD", "doc_type": "earnings_call", "text": "t", "section": "full",
        "date": "2025Q3", "period": "2025Q3",
    }, "AMD")
    assert (call.period, call.section, call.filed_date) == ("2025Q3", None, None)


# ------------------------------------------------- full pipeline with team parts


def test_pipeline_with_team_components(retriever, tmp_path):
    audit = team_auditor(retriever, tmp_path).audit(THESIS)
    assert audit.verdict.level == VerdictLevel.GREEN
    # Guru's final check ran over the summary
    assert audit.summary_check is not None and not audit.summary_check.support_checked
    assert any(s.cited for s in audit.summary_check.sentences)
    # sources now carry their text, for storage and number checks
    assert all(s.text for s in audit.sources)
    # Abdullah's store got a recommendation + challenge, linked to the audit
    recs = list((tmp_path / "track" / "recommendations").glob("*.json"))
    assert len(recs) == 1
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    assert rec["company"] == "EXMP" and rec["verdict"] == "green"
    assert rec["sources"] and rec["sources"][0]["label"].startswith("S") and rec["sources"][0]["text"]
    challenges = list((tmp_path / "track" / "challenges").glob("*.json"))
    assert len(challenges) == 1
    assert (tmp_path / "archive" / "audits" / f"{audit.audit_id}.json").exists()


def test_answers_reach_the_challenge(retriever, tmp_path):
    auditor = team_auditor(retriever, tmp_path)
    audit = auditor.audit(THESIS)
    auditor.review_answers(audit, [UserAnswer(question="Why now?", answer="Earnings next week")])
    ch = json.loads(next((tmp_path / "track" / "challenges").glob("*.json")).read_text(encoding="utf-8"))
    assert ch["answers"] == {"Why now?": "Earnings next week"}
    assert ch["verdict"] == "green"  # answers never change the verdict


def test_demo_tickers_kept_out_of_track_record(retriever, tmp_path):
    team_auditor(retriever, tmp_path, skip_tickers={"EXMP"}).audit(THESIS)
    assert not list((tmp_path / "track" / "recommendations").glob("*.json"))
    assert list((tmp_path / "archive" / "audits").glob("*.json"))  # still archived


def test_store_failure_becomes_warning(retriever, tmp_path):
    class Broken:
        def save_audit(self, audit):
            raise RuntimeError("disk full")

    auditor = ThesisAuditor(make_llm(), retriever, store=Broken(), max_workers=1)
    audit = auditor.audit(THESIS)
    assert audit.verdict.level == VerdictLevel.GREEN
    assert any("disk full" in w for w in audit.warnings)


# ------------------------------------------------------------ Bogdan: numbers


def test_number_checker(retriever, tmp_path):
    audit = team_auditor(retriever, tmp_path).audit(THESIS)
    pool = {s.label: Chunk(**{k: v for k, v in s.model_dump().items() if k in Chunk.model_fields})
            for s in audit.sources}
    label = audit.claims[0].evidence[0].source
    checker = StockAgentNumberChecker(use_sec=False)
    assert checker.check(audit, pool).issues == []
    audit.claims[0].reasoning = f"Revenue was $987.6 billion [{label}]. You expect 50% growth [{label}]."
    issues = checker.check(audit, pool).issues
    assert any("$987.6 billion" in i for i in issues)
    assert not any("50%" in i for i in issues)  # the investor's own number is allowed


class FakeSec:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, 0

    def as_sources(self, ticker, today=None):
        self.calls += 1
        if self.fail:
            from src.checks.sec_facts import SecError

            raise SecError(self.fail)
        return [{"label": "SEC 10-K: Revenue (2025 FY, 2025-01-01 to 2025-12-31)", "text": "$987600000000"}]


def test_number_checker_with_official_sec_figures(retriever, tmp_path):
    audit = team_auditor(retriever, tmp_path).audit(THESIS)
    pool = {s.label: Chunk(**{k: v for k, v in s.model_dump().items() if k in Chunk.model_fields})
            for s in audit.sources}
    label = audit.claims[0].evidence[0].source
    audit.claims[0].reasoning = f"Revenue was $987.6 billion [{label}]."
    sec = FakeSec()
    report = StockAgentNumberChecker(sec=sec).check(audit, pool)
    assert report.sec_checked and report.sec_error is None
    assert any("$987.6 billion matches 10-K: Revenue" in c for c in report.sec_confirmed)
    # missing from the cited source, but official: a softer note, not a plain miss
    assert any("$987.6 billion" in i and "matches an official SEC figure" in i for i in report.issues)
    sec.calls = 0
    checker = StockAgentNumberChecker(sec=sec)
    checker.check(audit, pool)
    checker.check(audit, pool)
    assert sec.calls == 1  # downloaded once per ticker, not once per sentence or audit


def test_sec_failure_is_reported_not_raised(retriever, tmp_path):
    audit = team_auditor(retriever, tmp_path).audit(THESIS)
    report = StockAgentNumberChecker(sec=FakeSec(fail="SEC needs a contact email")).check(audit, {})
    assert not report.sec_checked and "contact email" in report.sec_error


def test_sec_on_only_with_contact_email(monkeypatch):
    monkeypatch.delenv("AUDITOR_SEC_NUMBERS", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert not StockAgentNumberChecker().uses_sec
    monkeypatch.setenv("SEC_USER_AGENT", "Team1J-StockAgent someone@example.com")
    assert StockAgentNumberChecker().uses_sec
    monkeypatch.setenv("AUDITOR_SEC_NUMBERS", "0")
    assert not StockAgentNumberChecker().uses_sec


def test_number_check_report_stored_on_audit(retriever, tmp_path):
    store = StockAgentStore(tmp_path / "t", archive=JsonFileStore(tmp_path / "a"), record_price=False)
    auditor = ThesisAuditor(make_llm(), retriever, store=store,
                            number_checker=StockAgentNumberChecker(sec=FakeSec()), max_workers=1)
    audit = auditor.audit(THESIS)
    assert audit.number_check is not None and audit.number_check.sec_checked
    assert audit.number_check_issues == audit.number_check.issues


# ------------------------------------------------- track record (Abdullah + Bogdan)


def test_start_price_recorded_and_results_check(retriever, tmp_path):
    from datetime import date, timedelta

    prices = fake_prices()
    auditor = team_auditor(retriever, tmp_path, record_price=True, prices=prices)
    audit = auditor.audit(THESIS)
    store = auditor.store
    rec_id = store.recommendation_id(audit.audit_id)
    rec = store.storage.recommendations.get(rec_id)
    assert rec["price_on_date"] and rec["price_on_date"] > 0

    # Nothing is due on the audit day...
    assert store.update_results(prices=prices)["updated"] == []
    summary = store.track_record()
    assert summary["total_recommendations"] == 1 and summary["waiting_for_first_check"] == 1

    # ...but a year later every horizon is: back-date the saved record to simulate that.
    path = tmp_path / "track" / "recommendations" / f"{rec_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["as_of_date"] = (date.today() - timedelta(days=400)).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")
    report = store.update_results(prices=prices)
    assert report["updated"][0]["horizons"] == ["12m", "1m", "3m", "6m"]
    summary = store.track_record()
    assert summary["horizons"]["12m"]["correct"] == 1  # green + stock beat flat SPY = right
    recs = store.recommendations()
    assert recs[0]["results"]["1m"]["beat_market"] and recs[0]["challenges"]


def test_start_price_failure_doesnt_block_saving(retriever, tmp_path):
    from src.checks.prices import PriceBook

    auditor = team_auditor(retriever, tmp_path, record_price=True, prices=PriceBook(fetch=lambda *a: {}))
    audit = auditor.audit(THESIS)
    rec = auditor.store.storage.recommendations.get(auditor.store.recommendation_id(audit.audit_id))
    assert rec["price_on_date"] is None and not audit.warnings


# ------------------------------------------------------- Guru: citation check


def test_citation_verifier_flags_uncited_numbers():
    src = {"S1": Chunk(chunk_id="a", ticker="AMD", doc_type="10-Q", text="Revenue was $9.2 billion.")}
    check = StockAgentCitationVerifier(check_support=False).verify(
        "Revenue was $9.2 billion [S1]. Margins rose to 54%.", src
    )
    assert not check.ok
    assert [s.status for s in check.sentences] == ["cited", "uncited_number"]


def test_payload_fits_storage_limits(retriever, tmp_path):
    audit = team_auditor(retriever, tmp_path).audit(THESIS)
    audit.raw_input = "x" * 9000
    payload = recommendation_payload(audit)
    assert len(payload["user_reason"]) == 4000
    assert all(len(s["label"]) <= 120 and len(s.get("page", "")) <= 20 for s in payload["sources"])


def test_build_team_components_uses_real_parts(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCK_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("AUDITOR_TEAM_COMPONENTS", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    parts = build_team_components()
    assert type(parts["retriever"]).__name__ == "StockAgentRetriever"
    assert type(parts["store"]).__name__ == "StockAgentStore"
    assert "EXMP" in parts["store"].skip
    monkeypatch.setenv("AUDITOR_TEAM_COMPONENTS", "0")
    assert type(build_team_components()["retriever"]).__name__ == "FixtureRetriever"
