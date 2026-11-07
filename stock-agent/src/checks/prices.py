"""Stock and S&P 500 prices for the results check.

get_close("NVDA", date(2026, 11, 7)) -> (trading_date, close_price)

- Weekends/holidays: uses the closest EARLIER trading day (looks back up to 10 days).
- Prices are cached, so the same day is never downloaded twice in one run.
- The price source can be swapped (tests use fake prices; you could later use
  Alpha Vantage instead of yfinance without changing the results check).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Dict

BENCHMARK = "SPY"  # ETF that tracks the S&P 500
LOOKBACK_DAYS = 10

# fetch(ticker, start, end_exclusive) -> {date: close}
Fetcher = Callable[[str, date, date], Dict[date, float]]


class PriceNotFoundError(Exception):
    pass


def yfinance_fetch(ticker: str, start: date, end: date) -> Dict[date, float]:
    import yfinance as yf  # imported here so tests don't need it

    hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        return {}
    return {ts.date(): float(close) for ts, close in hist["Close"].items()}


class PriceBook:
    def __init__(self, fetch: Fetcher | None = None):
        self.fetch = fetch or yfinance_fetch
        self._cache: dict[str, dict[date, float]] = {}

    def get_close(self, ticker: str, on: date) -> tuple[date, float]:
        ticker = ticker.upper()
        known = self._cache.setdefault(ticker, {})
        earliest = on - timedelta(days=LOOKBACK_DAYS)

        def lookup():
            for d in sorted((d for d in known if earliest <= d <= on), reverse=True):
                return d, known[d]
            return None

        hit = lookup()
        if hit is None:
            known.update(self.fetch(ticker, earliest, on + timedelta(days=1)))
            hit = lookup()
        if hit is None:
            raise PriceNotFoundError(f"no price for {ticker} on or before {on}")
        return hit

    def get_benchmark(self, on: date) -> tuple[date, float]:
        return self.get_close(BENCHMARK, on)
