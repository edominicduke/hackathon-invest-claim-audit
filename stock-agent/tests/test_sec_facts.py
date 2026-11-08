"""SEC official numbers tests (Bogdan's part). Uses fake SEC data, no internet."""

from datetime import date

import pytest

from src.checks.number_check import check_numbers_with_sec
from src.checks.sec_facts import FACTS_URL, TICKERS_URL, SecError, SecFacts

TODAY = date(2026, 11, 7)

FAKE_TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

FAKE_FACTS = {
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2026-04-28", "end": "2026-07-27", "val": 46743000000, "form": "10-Q", "fy": 2027, "fp": "Q2"},
            {"start": "2026-04-28", "end": "2026-07-27", "val": 46743000000, "form": "10-Q", "fy": 2027, "fp": "Q2"},  # duplicate
            {"start": "2015-01-01", "end": "2015-03-31", "val": 999000000, "form": "10-Q", "fy": 2015, "fp": "Q1"},    # too old
            {"start": "2026-04-28", "end": "2026-07-27", "val": 12345, "form": "8-K", "fy": 2027, "fp": "Q2"},         # not 10-K/10-Q
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2026-04-28", "end": "2026-07-27", "val": 26422000000, "form": "10-Q", "fy": 2027, "fp": "Q2"},
        ]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"start": "2026-04-28", "end": "2026-07-27", "val": 1.08, "form": "10-Q", "fy": 2027, "fp": "Q2"},
        ]}},
        "LongTermDebt": {"units": {"USD": [
            {"end": "2026-07-27", "val": 8463000000, "form": "10-Q", "fy": 2027, "fp": "Q2"},
        ]}},
        "SomethingWeIgnore": {"units": {"USD": [
            {"end": "2026-07-27", "val": 31000000000, "form": "10-Q", "fy": 2027, "fp": "Q2"},
        ]}},
    }}
}


def fake_sec():
    calls = []

    def fetch(url, user_agent):
        calls.append(url)
        assert "@" in user_agent
        if url == TICKERS_URL:
            return FAKE_TICKERS
        if url == FACTS_URL.format(cik=1045810):
            return FAKE_FACTS
        raise RuntimeError("404")

    sec = SecFacts(user_agent="Team1J test@example.com", fetch=fetch, pause=0)
    return sec, calls


def test_needs_contact_email():
    sec = SecFacts(user_agent="no email here", fetch=lambda u, a: {}, pause=0)
    with pytest.raises(SecError, match="SEC_USER_AGENT"):
        sec.cik("NVDA")


def test_ticker_to_cik_and_unknown_ticker():
    sec, _ = fake_sec()
    assert sec.cik("nvda") == 1045810
    with pytest.raises(SecError, match="ZZZZ"):
        sec.cik("ZZZZ")


def test_official_numbers_keeps_recent_10k_10q_and_known_concepts():
    sec, _ = fake_sec()
    nums = sec.official_numbers("NVDA", TODAY)
    names = sorted(n["name"] for n in nums)
    assert names == ["EPS (diluted)", "Long-term debt", "Net income", "Revenue"]  # no dupes, no old, no 8-K
    rev = next(n for n in nums if n["name"] == "Revenue")
    assert rev["value"] == 46743000000 and rev["form"] == "10-Q" and rev["fiscal"] == "2027 Q2"


def test_downloads_are_cached():
    sec, calls = fake_sec()
    sec.official_numbers("NVDA", TODAY)
    sec.official_numbers("NVDA", TODAY)
    assert calls == [TICKERS_URL, FACTS_URL.format(cik=1045810)]


def test_number_check_against_sec():
    sec, _ = fake_sec()
    answer = "Revenue was $46.7 billion and net income $26.4 billion. EPS was $1.08. Debt is $9.9 billion."
    result = check_numbers_with_sec(answer, [], "NVDA", sec, TODAY)

    by_text = {n["text"]: n for n in result["numbers"]}
    assert by_text["$46.7 billion"]["found"] and by_text["$46.7 billion"]["official_sec"]
    assert "Revenue" in by_text["$46.7 billion"]["source"]
    assert by_text["$26.4 billion"]["official_sec"]
    assert by_text["$1.08"]["official_sec"]
    assert result["not_found"] == ["$9.9 billion"]   # real debt is $8.5 billion
    assert result["sec_numbers_used"] == 4
    assert result["sec_error"] is None


def test_normal_sources_still_work_with_sec():
    sec, _ = fake_sec()
    sources = [{"label": "Q2 call", "text": "Data center grew 56%."}]
    result = check_numbers_with_sec("Data center grew 56%, revenue $46.7B.", sources, "NVDA", sec, TODAY)
    assert result["all_found"]
    assert [n["official_sec"] for n in result["numbers"]] == [False, True]


def test_sec_unreachable_falls_back_to_sources():
    sec = SecFacts(user_agent="x a@b.com", fetch=lambda u, a: (_ for _ in ()).throw(OSError("offline")), pause=0)
    sources = [{"label": "10-Q p.8", "text": "Revenue was $4,213 million."}]
    result = check_numbers_with_sec("Revenue was $4.2 billion.", sources, "NVDA", sec, TODAY)
    assert result["all_found"]
    assert "could not reach SEC" in result["sec_error"]
