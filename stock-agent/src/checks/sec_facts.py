"""Official company numbers from SEC EDGAR (XBRL "company facts").

Every 10-K / 10-Q includes tagged numbers (revenue, net income, EPS, debt...).
The SEC publishes them free at data.sec.gov. We use them as an extra, official
source for the number check (#3).

SEC rules (or you get blocked):
  - Send a User-Agent with your name/team and contact email.
    Set it once in the terminal before starting the app:
        export SEC_USER_AGENT="Team1J-StockAgent yourname@example.com"
  - At most 10 requests per second. We stay well under that and cache results.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from datetime import date, timedelta
from typing import Callable

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# The numbers an AI answer usually mentions. Name shown to the user -> SEC tags.
CONCEPTS = {
    "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "Cost of revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "Gross profit": ["GrossProfit"],
    "Operating income": ["OperatingIncomeLoss"],
    "Net income": ["NetIncomeLoss"],
    "R&D spending": ["ResearchAndDevelopmentExpense"],
    "EPS (diluted)": ["EarningsPerShareDiluted"],
    "EPS (basic)": ["EarningsPerShareBasic"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "Total assets": ["Assets"],
    "Total liabilities": ["Liabilities"],
    "Long-term debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "Shareholders' equity": ["StockholdersEquity"],
}
UNITS = ("USD", "USD/shares")
YEARS_BACK = 3

# fetch(url, user_agent) -> parsed JSON
JsonFetcher = Callable[[str, str], dict]


class SecError(Exception):
    pass


def _ssl_context() -> ssl.SSLContext:
    # Python from python.org on a Mac doesn't use the Mac's certificates, which
    # causes CERTIFICATE_VERIFY_FAILED. certifi provides a trusted certificate list.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_fetch(url: str, user_agent: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))


class SecFacts:
    def __init__(self, user_agent: str | None = None, fetch: JsonFetcher | None = None, pause: float = 0.15):
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "")
        self.fetch = fetch or http_fetch
        self.pause = pause  # seconds between requests (keeps us far below 10/second)
        self._ciks: dict[str, int] | None = None
        self._facts: dict[int, dict] = {}

    def _get(self, url: str) -> dict:
        if "@" not in self.user_agent:
            raise SecError(
                'SEC needs a contact email. In the terminal run: '
                'export SEC_USER_AGENT="Team1J-StockAgent yourname@example.com"'
            )
        try:
            data = self.fetch(url, self.user_agent)
        except Exception as exc:  # network problems, 403, 404...
            raise SecError(f"could not reach SEC ({exc})") from exc
        time.sleep(self.pause)
        return data

    def cik(self, ticker: str) -> int:
        if self._ciks is None:
            data = self._get(TICKERS_URL)
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        ticker = ticker.upper()
        if ticker not in self._ciks:
            raise SecError(f"{ticker} is not a ticker the SEC knows")
        return self._ciks[ticker]

    def company_facts(self, ticker: str) -> dict:
        cik = self.cik(ticker)
        if cik not in self._facts:
            self._facts[cik] = self._get(FACTS_URL.format(cik=cik))
        return self._facts[cik]

    def official_numbers(self, ticker: str, today: date | None = None) -> list[dict]:
        """Recent official numbers, e.g.
        {"name": "Revenue", "value": 4213000000.0, "unit": "USD", "period_end": "2026-07-27",
         "form": "10-Q", "fiscal": "2026 Q2", "concept": "Revenues"}
        """
        today = today or date.today()
        cutoff = (today - timedelta(days=365 * YEARS_BACK)).isoformat()
        gaap = self.company_facts(ticker).get("facts", {}).get("us-gaap", {})

        out, seen = [], set()
        for name, tags in CONCEPTS.items():
            for tag in tags:
                for unit, rows in gaap.get(tag, {}).get("units", {}).items():
                    if unit not in UNITS:
                        continue
                    for row in rows:
                        if row.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                            continue
                        if row.get("end", "") < cutoff or row.get("val") is None:
                            continue
                        key = (name, row["end"], row.get("start"), float(row["val"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append({
                            "name": name,
                            "value": float(row["val"]),
                            "unit": unit,
                            "period_start": row.get("start"),
                            "period_end": row["end"],
                            "form": row.get("form"),
                            "fiscal": f'{row.get("fy", "")} {row.get("fp", "")}'.strip(),
                            "concept": tag,
                        })
        return out

    def as_sources(self, ticker: str, today: date | None = None) -> list[dict]:
        """Official numbers in the same shape as normal sources, so number_check can use them."""
        sources = []
        for n in self.official_numbers(ticker, today):
            period = f'{n["period_start"]} to {n["period_end"]}' if n["period_start"] else f'as of {n["period_end"]}'
            sources.append({
                "label": f'SEC {n["form"]}: {n["name"]} ({n["fiscal"]}, {period})',
                "text": f'${n["value"]:.2f}' if n["unit"] == "USD/shares" else f'${n["value"]:.0f}',
            })
        return sources
