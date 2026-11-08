"""Prices + results check tests (Bogdan's part). Uses fake prices, no internet."""

from datetime import date, timedelta

import pytest

from src.app.store import Storage
from src.checks.prices import PriceBook, PriceNotFoundError
from src.checks.results_check import check_recommendation, track_record, update_all, verdict_correct


def fake_market(prices_by_ticker):
    """prices_by_ticker: {"NVDA": {date: price}, "SPY": {...}}. Records how often it was called."""
    calls = []

    def fetch(ticker, start, end):
        calls.append(ticker)
        return {d: p for d, p in prices_by_ticker.get(ticker, {}).items() if start <= d < end}

    fetch.calls = calls
    return fetch


def daily(start, days, start_price, end_price):
    """Straight-line prices on weekdays only."""
    out = {}
    for i in range(days + 1):
        d = start + timedelta(days=i)
        if d.weekday() < 5:
            out[d] = start_price + (end_price - start_price) * i / days
    return out


START = date(2025, 1, 6)  # a Monday


def test_weekend_uses_friday():
    friday = date(2025, 1, 10)
    book = PriceBook(fake_market({"NVDA": {friday: 100.0}}))
    assert book.get_close("nvda", date(2025, 1, 12)) == (friday, 100.0)  # Sunday


def test_missing_price_raises():
    book = PriceBook(fake_market({}))
    with pytest.raises(PriceNotFoundError):
        book.get_close("NVDA", START)


def test_prices_are_cached():
    fetch = fake_market({"NVDA": {START: 100.0}})
    book = PriceBook(fetch)
    book.get_close("NVDA", START)
    book.get_close("NVDA", START)
    assert fetch.calls == ["NVDA"]


def test_verdict_rules():
    assert verdict_correct("green", True) is True
    assert verdict_correct("green", False) is False
    assert verdict_correct("red", False) is True
    assert verdict_correct("yellow", True) is None
    assert verdict_correct("not_enough_evidence", False) is None


def test_check_recommendation_only_due_horizons():
    market = {
        "NVDA": daily(START, 400, 100, 200),   # stock doubles over the year
        "SPY": daily(START, 400, 500, 550),    # market +10%
    }
    book = PriceBook(fake_market(market))
    rec = {"company": "NVDA", "verdict": "green", "as_of_date": START.isoformat(), "results": {}}

    results, start_price = check_recommendation(rec, book, today=START + timedelta(days=100))
    assert sorted(results) == ["1m", "3m"]          # 6m and 12m not due yet
    assert start_price == 100
    r = results["3m"]
    assert r["beat_market"] is True and r["verdict_correct"] is True
    assert r["stock_return_pct"] > r["market_return_pct"]


def test_update_all_saves_into_storage_and_track_record(tmp_path):
    storage = Storage(tmp_path)
    market = {
        "NVDA": daily(START, 400, 100, 200),   # beats market
        "INTC": daily(START, 400, 50, 40),     # loses
        "SPY": daily(START, 400, 500, 550),
    }
    book = PriceBook(fake_market(market))
    base = {"user_reason": "test", "as_of_date": START.isoformat()}
    good_buy = storage.recommendations.create({**base, "company": "NVDA", "verdict": "green"})
    good_avoid = storage.recommendations.create({**base, "company": "INTC", "verdict": "red"})
    bad_buy = storage.recommendations.create({**base, "company": "INTC", "verdict": "green"})
    storage.recommendations.create({**base, "company": "NVDA", "verdict": "yellow"})
    no_data = storage.recommendations.create({**base, "company": "ZZZZ", "verdict": "green"})

    report = update_all(storage.recommendations, book, today=START + timedelta(days=200))
    assert report["checked"] == 5
    assert len(report["updated"]) == 4
    assert report["errors"][0]["id"] == no_data["id"]

    saved = storage.recommendations.get(good_buy["id"])
    assert sorted(saved["results"]) == ["1m", "3m", "6m"]
    assert saved["price_on_date"] == 100

    # Running again the same day changes nothing (already checked).
    again = update_all(storage.recommendations, book, today=START + timedelta(days=200))
    assert again["updated"] == []

    summary = track_record(storage.recommendations.all())
    assert summary["total_recommendations"] == 5
    assert summary["by_verdict"] == {"green": 3, "red": 1, "yellow": 1}
    six = summary["horizons"]["6m"]
    assert six["checked"] == 4 and six["scored"] == 3 and six["correct"] == 2
    assert six["accuracy_pct"] == 66.7
    assert summary["horizons"]["12m"]["checked"] == 0
    assert summary["waiting_for_first_check"] == 1
    assert storage.recommendations.get(good_avoid["id"])["results"]["6m"]["verdict_correct"] is True
    assert storage.recommendations.get(bad_buy["id"])["results"]["6m"]["verdict_correct"] is False
