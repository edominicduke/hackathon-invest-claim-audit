"""Was the agent right? Compares each recommendation with the S&P 500.

For every saved recommendation, after 1, 3, 6 and 12 months:
  stock % change  vs  S&P 500 (SPY) % change  ->  beat_market true/false

Was the verdict right?
  green (buy)  -> right if the stock BEAT the market
  red (avoid)  -> right if the stock did NOT beat the market
  yellow / not_enough_evidence -> not scored (no clear call was made)

Run on a schedule (e.g. daily with Cloud Scheduler or cron):
    python -m src.checks.results_check --data-dir data
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

from src.checks.prices import PriceBook, PriceNotFoundError

HORIZONS = {"1m": 30, "3m": 91, "6m": 182, "12m": 365}  # days


def _pct(start: float, end: float) -> float:
    return round((end / start - 1) * 100, 2)


def verdict_correct(verdict: str, beat_market: bool) -> bool | None:
    if verdict == "green":
        return beat_market
    if verdict == "red":
        return not beat_market
    return None


def check_recommendation(rec: dict, prices: PriceBook, today: date | None = None) -> tuple[dict, float | None]:
    """Returns (new_results, start_price). Only horizons that are due and not done yet."""
    today = today or date.today()
    as_of = date.fromisoformat(rec["as_of_date"])
    due = {h: d for h, d in HORIZONS.items() if as_of + timedelta(days=d) <= today and h not in rec.get("results", {})}
    if not due:
        return {}, None

    start_day, stock_start = prices.get_close(rec["company"], as_of)
    _, market_start = prices.get_benchmark(as_of)

    results = {}
    for horizon, days in due.items():
        end = as_of + timedelta(days=days)
        end_day, stock_end = prices.get_close(rec["company"], end)
        _, market_end = prices.get_benchmark(end)
        stock_ret = _pct(stock_start, stock_end)
        market_ret = _pct(market_start, market_end)
        beat = stock_ret > market_ret
        results[horizon] = {
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "stock_start": round(stock_start, 4),
            "stock_end": round(stock_end, 4),
            "stock_return_pct": stock_ret,
            "market_return_pct": market_ret,
            "excess_return_pct": round(stock_ret - market_ret, 2),
            "beat_market": beat,
            "verdict_correct": verdict_correct(rec["verdict"], beat),
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    return results, stock_start


def update_all(rec_store, prices: PriceBook | None = None, today: date | None = None) -> dict:
    """Checks every saved recommendation and saves results into Abdullah's store."""
    prices = prices or PriceBook()
    report = {"checked": 0, "updated": [], "errors": []}
    for rec in rec_store.all():
        report["checked"] += 1
        try:
            results, start_price = check_recommendation(rec, prices, today)
        except PriceNotFoundError as exc:
            report["errors"].append({"id": rec["id"], "error": str(exc)})
            continue
        if results:
            rec_store.update_results(rec["id"], results, price_on_date=start_price)
            report["updated"].append({"id": rec["id"], "horizons": sorted(results)})
    return report


def track_record(recs: list[dict]) -> dict:
    """Summary for the track record page."""
    by_verdict: dict[str, int] = {}
    for r in recs:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    horizons = {}
    for h in HORIZONS:
        rows = [r["results"][h] for r in recs if h in r.get("results", {})]
        scored = [x for x in rows if x["verdict_correct"] is not None]
        correct = sum(1 for x in scored if x["verdict_correct"])
        horizons[h] = {
            "checked": len(rows),
            "scored": len(scored),
            "correct": correct,
            "accuracy_pct": round(100 * correct / len(scored), 1) if scored else None,
            "avg_excess_return_pct": round(sum(x["excess_return_pct"] for x in rows) / len(rows), 2) if rows else None,
        }
    return {
        "total_recommendations": len(recs),
        "by_verdict": by_verdict,
        "waiting_for_first_check": sum(1 for r in recs if not r.get("results")),
        "horizons": horizons,
        "note": "Yellow and not_enough_evidence verdicts are not scored. Past results do not predict future results.",
    }


if __name__ == "__main__":
    from src.app.store import Storage

    parser = argparse.ArgumentParser(description="Check saved recommendations against the S&P 500")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    print(update_all(Storage(args.data_dir).recommendations))
