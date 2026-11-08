"""Download SEC filings for a list of tickers and save them section by section.

What it grabs per ticker:
  10-K  (latest N)  -> Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 7A
  10-Q  (latest N)  -> Part I Item 2 MD&A, Part II Item 1A Risk Factors
  8-K   (latest N with Item 2.02 "Results of Operations") -> the earnings press
        release exhibit (EX-99.1). Also scans 8-K exhibits for filed call
        transcripts (some companies file them; most don't).
  XBRL company facts -> structured actuals (revenue, net income, EPS...) saved
        as JSON, so the agent compares real numbers instead of reading them
        out of text.

Setup:
  pip install edgartools requests
  export SEC_IDENTITY="Your Name your.email@example.com"   # SEC requires this

Run:
  python -m ingest.edgar_fetch NVDA AMD INTC --n-10k 2 --n-10q 4 --n-8k 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import requests
from edgar import Company, set_identity

from ingest.common import DATA_DIR, Doc, save

TEN_K_SECTIONS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 7A": "Market Risk",
}
TEN_Q_SECTIONS = [  # (part, item, label)
    ("Part I", "Item 2", "MD&A"),
    ("Part II", "Item 1A", "Risk Factors"),
]
# 8-K items worth scanning: 2.02 = earnings results; 7.01/8.01 = where
# companies sometimes file the call transcript a few days later.
EIGHT_K_ITEMS = ("2.02", "7.01", "8.01")

# XBRL tags worth keeping. Companies use different tags for the same idea, so
# several revenue tags are listed; the chunker / agent can use whichever exists.
XBRL_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "ResearchAndDevelopmentExpense",
    "CashAndCashEquivalentsAtCarryingValue",
]


def looks_like_transcript(text: str) -> bool:
    """Cheap heuristic: call transcripts have an operator and many speaker turns."""
    if not text:
        return False
    head = text[:5000].lower()
    mentions_call = "conference call" in head or "earnings call" in head
    operator_turns = len(re.findall(r"^\s*\**operator\**\s*[:\-]?", text, re.I | re.M))
    return mentions_call and operator_turns >= 3


def fetch_10k(company, ticker: str, n: int) -> int:
    saved = 0
    filings = company.get_filings(form="10-K", amendments=False).latest(n)
    for filing in _as_list(filings):
        tenk = filing.obj()
        for item, label in TEN_K_SECTIONS.items():
            text = tenk.get(item)
            saved += save(Doc(
                ticker=ticker, source="sec_edgar", doc_type="10-K",
                section=f"{item} {label}", date=str(filing.filing_date),
                period=str(filing.period_of_report), accession=filing.accession_number,
                url=filing.homepage_url, text=text or "",
            ))
    return saved


def fetch_10q(company, ticker: str, n: int) -> int:
    saved = 0
    filings = company.get_filings(form="10-Q", amendments=False).latest(n)
    for filing in _as_list(filings):
        tenq = filing.obj()
        for part, item, label in TEN_Q_SECTIONS:
            text = tenq.get_item_with_part(part, item)
            saved += save(Doc(
                ticker=ticker, source="sec_edgar", doc_type="10-Q",
                section=f"{part} {item} {label}", date=str(filing.filing_date),
                period=str(filing.period_of_report), accession=filing.accession_number,
                url=filing.homepage_url, text=text or "",
            ))
    return saved


def fetch_8k(company, ticker: str, n: int) -> int:
    """Earnings press releases (+ any filed transcripts) from recent 8-Ks."""
    saved = 0
    kept = 0
    # Look further back than n, since most 8-Ks aren't earnings-related.
    for filing in _as_list(company.get_filings(form="8-K", amendments=False).latest(n * 6)):
        # The filing index lists 8-K items ("2.02,9.01") so we can filter
        # before downloading anything.
        items = str(getattr(filing, "items", "") or "")
        if not any(code in items for code in EIGHT_K_ITEMS):
            continue
        is_earnings = "2.02" in items
        for att in filing.attachments:
            if not str(att.document_type).upper().startswith("EX-99"):
                continue
            text = att.markdown() if att.is_html() else att.text()
            if looks_like_transcript(text):
                doc_type = "8-K_transcript"
            elif is_earnings:
                doc_type = "8-K_press_release"
            else:
                continue  # 7.01 / 8.01 exhibit that isn't a transcript: skip
            saved += save(Doc(
                ticker=ticker, source="sec_edgar", doc_type=doc_type,
                section=str(att.document_type), date=str(filing.filing_date),
                accession=filing.accession_number, url=att.url, text=text or "",
                extra={"8k_items": items, "description": att.description},
            ))
        kept += is_earnings
        if kept >= n:
            break
    return saved


def fetch_xbrl_facts(cik: int, ticker: str, identity: str) -> int:
    """Structured financial actuals from SEC's company-facts API."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    resp = requests.get(url, headers={"User-Agent": identity}, timeout=30)
    resp.raise_for_status()
    gaap = resp.json().get("facts", {}).get("us-gaap", {})

    rows = []
    for tag in XBRL_TAGS:
        for unit, facts in gaap.get(tag, {}).get("units", {}).items():
            for f in facts:
                if f.get("form") not in ("10-K", "10-Q"):
                    continue
                rows.append({
                    "tag": tag, "unit": unit, "value": f["val"],
                    "period_start": f.get("start"), "period_end": f["end"],
                    "fiscal_year": f.get("fy"), "fiscal_period": f.get("fp"),
                    "form": f["form"], "filed": f["filed"], "accession": f["accn"],
                })
    out = DATA_DIR / ticker / "xbrl_facts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    (DATA_DIR / ticker / "xbrl_meta.json").write_text(json.dumps({"cik": cik, "ticker": ticker}))
    return len(rows)


def _as_list(filings):
    """.latest(1) returns a single Filing; .latest(n>1) returns a collection."""
    if filings is None:
        return []
    return [filings] if not hasattr(filings, "__iter__") else list(filings)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--n-10k", type=int, default=2)
    p.add_argument("--n-10q", type=int, default=4)
    p.add_argument("--n-8k", type=int, default=6, help="number of earnings 8-Ks per ticker")
    args = p.parse_args()

    identity = os.environ.get("SEC_IDENTITY")
    if not identity:
        sys.exit('Set SEC_IDENTITY="Your Name you@example.com" first (SEC requires it).')
    set_identity(identity)

    for ticker in (t.upper() for t in args.tickers):
        print(f"== {ticker}")
        company = Company(ticker)
        steps = [
            ("10-K sections", lambda: fetch_10k(company, ticker, args.n_10k)),
            ("10-Q sections", lambda: fetch_10q(company, ticker, args.n_10q)),
            ("8-K exhibits", lambda: fetch_8k(company, ticker, args.n_8k)),
            ("XBRL facts", lambda: fetch_xbrl_facts(company.cik, ticker, identity)),
        ]
        for name, step in steps:
            try:
                print(f"   {name}: {step()} saved")
            except Exception as e:  # keep going: one bad filing shouldn't kill the run
                print(f"   {name}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()