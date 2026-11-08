"""Structured financials from SEC XBRL data (data/raw/<TICKER>/xbrl_facts.json).

    from ingest.financials import get_financials, clean_facts

    rows = get_financials("INTC", metrics=["revenue", "net_income"], period="quarterly", last=6)
    # each row: metric, value, unit, value_text ("$13.65 billion"), period_start/end,
    #           period_label, form, filed, accession, url, derived, restated,
    #           plus chunk_id / citation / text so rows can be cited like search results

    table = clean_facts("INTC")   # the whole cleaned table (for the number check)

What the cleanup does (the raw file is NOT usable as-is):
  1. Duplicates: every filing re-reports earlier periods for comparison. Keep one
     value per (metric, period), from the LATEST filing (so restatements win);
     flag restated=True if an earlier filing reported a different number.
  2. Period types: 10-Qs report the 3-month quarter AND 6/9-month year-to-date
     totals under the same tag. Classified by length: ~3 mo = quarter,
     ~6/9 mo = ytd (kept out of results), ~12 mo = annual, no start = instant
     (balance-sheet items like cash, "as of" a date).
  3. Q4: companies don't file a Q4 report, so Q4 = annual - 9-month YTD,
     marked derived=True. Not derived for EPS (share counts change, so
     subtracting EPS is wrong).
  4. Revenue: companies use different tags. Per period, "Revenues" (total) is
     preferred over "RevenueFromContractWithCustomer..." (which can exclude
     e.g. leasing revenue), if both exist.

Period labels use the period END date ("quarter ended 2025-09-27") plus the
approximate calendar quarter. Fiscal quarters differ by company (NVIDIA's
fiscal year ends in late January), so don't treat "CY2025Q3" as the company's
own "Q3".
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from .common import DATA_DIR

# friendly metric -> XBRL tags in order of preference
METRICS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("Revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
    "gross_profit": ("Gross profit", ["GrossProfit"]),
    "operating_income": ("Operating income", ["OperatingIncomeLoss"]),
    "net_income": ("Net income", ["NetIncomeLoss"]),
    "eps_diluted": ("Diluted EPS", ["EarningsPerShareDiluted"]),
    "rd_expense": ("R&D expense", ["ResearchAndDevelopmentExpense"]),
    "cash": ("Cash and cash equivalents", ["CashAndCashEquivalentsAtCarryingValue"]),
}
NOT_DERIVABLE = {"eps_diluted"}

# Fallback CIKs for the demo tickers (used to build filing links)
CIKS = {"NVDA": 1045810, "AMD": 2488, "INTC": 50863, "TSLA": 1318605, "GOOGL": 1652044}


# ---------- loading + cleanup ----------

def _d(s: str) -> date:
    return date.fromisoformat(s)


def _period_kind(start: str | None, end: str) -> str:
    if not start:
        return "instant"
    days = (_d(end) - _d(start)).days
    if 80 <= days <= 100:
        return "quarter"
    if 170 <= days <= 190:
        return "ytd6"
    if 260 <= days <= 285:
        return "ytd9"
    if 350 <= days <= 380:
        return "annual"
    return "other"


def _calendar_label(start: str | None, end: str, kind: str) -> str:
    """Approximate calendar period from the midpoint (e.g. a quarter ending 2025-10-26 -> CY2025Q3)."""
    e = _d(end)
    if kind == "instant":
        return f"as of {end}"
    mid = _d(start) + (e - _d(start)) / 2
    if kind == "annual":
        return f"~CY{mid.year}"
    return f"~CY{mid.year}Q{(mid.month - 1) // 3 + 1}"


def _cik(ticker: str) -> int | None:
    meta = DATA_DIR / ticker / "xbrl_meta.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("cik")
    return CIKS.get(ticker)


def _filing_url(ticker: str, accession: str) -> str | None:
    cik = _cik(ticker)
    if not cik or not accession:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"


def _fmt_value(v: float, unit: str) -> str:
    if unit == "USD/shares":
        return f"{'-' if v < 0 else ''}${abs(v):,.2f} per share"
    if unit == "USD":
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 1e9:
            return f"{sign}${a / 1e9:,.2f} billion"
        if a >= 1e6:
            return f"{sign}${a / 1e6:,.1f} million"
        return f"{sign}${a:,.0f}"
    return f"{v:,} {unit}"


@lru_cache(maxsize=None)
def clean_facts(ticker: str) -> tuple[dict, ...]:
    """The full cleaned table for one ticker: one row per (metric, period),
    quarters + annual + instants, with derived Q4s. Sorted by metric, period end."""
    ticker = ticker.upper()
    path = DATA_DIR / ticker / "xbrl_facts.json"
    if not path.exists():
        raise ValueError(f"No XBRL facts for {ticker}. Run: python -m ingest.edgar_fetch {ticker}")
    raw = json.loads(path.read_text())
    tag_rank = {tag: (metric, rank) for metric, (_, tags) in METRICS.items() for rank, tag in enumerate(tags)}

    # 1) keep latest filing per (metric, tag, start, end); remember if values changed
    best: dict[tuple, dict] = {}
    seen_values: dict[tuple, set] = {}
    for r in raw:
        if r["tag"] not in tag_rank:
            continue
        metric, rank = tag_rank[r["tag"]]
        key = (metric, rank, r.get("period_start"), r["period_end"])
        seen_values.setdefault(key, set()).add(r["value"])
        if key not in best or r["filed"] > best[key]["filed"]:
            best[key] = r

    # 2) one tag per (metric, period): the most preferred tag that has it
    chosen: dict[tuple, dict] = {}
    for (metric, rank, start, end), r in sorted(best.items(), key=lambda kv: kv[0][1]):
        pkey = (metric, start, end)
        if pkey in chosen:
            continue
        kind = _period_kind(start, end)
        if kind == "other":
            continue
        chosen[pkey] = {
            "ticker": ticker, "metric": metric, "metric_label": METRICS[metric][0], "tag": r["tag"],
            "value": r["value"], "unit": r["unit"], "period_start": start, "period_end": end, "kind": kind,
            "form": r["form"], "filed": r["filed"], "accession": r["accession"],
            "derived": False, "restated": len(seen_values[(metric, rank, start, end)]) > 1,
        }

    rows = [r for r in chosen.values() if r["kind"] in ("quarter", "annual", "instant", "ytd9")]

    # 3) derive Q4 = annual - 9-month YTD (same start date)
    ytd9 = {(r["metric"], r["period_start"]): r for r in rows if r["kind"] == "ytd9"}
    have_q = {(r["metric"], r["period_end"]) for r in rows if r["kind"] == "quarter"}
    for a in [r for r in rows if r["kind"] == "annual" and r["metric"] not in NOT_DERIVABLE]:
        nine = ytd9.get((a["metric"], a["period_start"]))
        if not nine or (a["metric"], a["period_end"]) in have_q:
            continue
        q4_start = (_d(nine["period_end"]) + timedelta(days=1)).isoformat()
        rows.append({
            **a, "value": a["value"] - nine["value"], "period_start": q4_start, "kind": "quarter",
            "derived": True, "restated": a["restated"] or nine["restated"],
            "derived_from": {"annual": {"form": a["form"], "filed": a["filed"], "accession": a["accession"]},
                             "ytd9": {"form": nine["form"], "filed": nine["filed"], "accession": nine["accession"]}},
        })
    rows = [r for r in rows if r["kind"] != "ytd9"]

    for r in rows:
        r.update(_presentation(r))
    return tuple(sorted(rows, key=lambda r: (r["metric"], r["period_end"], r["kind"])))


def _presentation(r: dict) -> dict:
    """Fields that let a row be cited exactly like a search hit."""
    t, label, end = r["ticker"], r["metric_label"], r["period_end"]
    period = {"quarter": f"quarter ended {end}", "annual": f"fiscal year ended {end}",
              "instant": f"as of {end}"}[r["kind"]]
    approx = _calendar_label(r["period_start"], end, r["kind"])
    value_text = _fmt_value(r["value"], r["unit"])
    if r["derived"]:
        src = r["derived_from"]
        citation = (f"{t} SEC XBRL: {label}, {period} (DERIVED: {src['annual']['form']} filed "
                    f"{src['annual']['filed']} minus 9-month total from {src['ytd9']['form']} filed {src['ytd9']['filed']})")
        text = (f"{t} {label.lower()} for the {period} ({approx}): {value_text}. Derived as full-year "
                f"value minus first nine months; not directly reported.")
        accession = src["annual"]["accession"]
    else:
        citation = f"{t} {r['form']} (filed {r['filed']}), SEC XBRL: {label}, {period}"
        text = f"{t} {label.lower()} for the {period} ({approx}): {value_text}. Reported in {r['form']} filed {r['filed']}."
        accession = r["accession"]
    if r["restated"]:
        text += " This value was revised in a later filing; the latest reported value is shown."
    return {
        "chunk_id": f"XBRL::{t}::{r['metric']}::{r['kind']}::{r['period_start']}::{end}",
        "doc_type": "xbrl_fact", "source": "sec_xbrl", "date": r["filed"],
        "period_label": f"{period} ({approx})", "value_text": value_text,
        "citation": citation, "text": text, "url": _filing_url(t, accession), "similarity": None,
    }


# ---------- the query function ----------

def get_financials(ticker: str, metrics: list[str] | None = None, period: str = "quarterly",
                   last: int = 6, include_derived: bool = True) -> list[dict]:
    """Most recent `last` periods of each metric, oldest first within each metric.

    period: "quarterly" (3-month periods, incl. derived Q4s; for cash: quarter-end balances)
            or "annual" (fiscal years; for cash: fiscal-year-end balances).
    """
    if isinstance(metrics, str):
        raise TypeError("metrics must be a list, e.g. metrics=['revenue']")
    metrics = metrics or list(METRICS)
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise ValueError(f"Unknown metric(s) {bad}. Valid metrics: {list(METRICS)}")
    if period not in ("quarterly", "annual"):
        raise ValueError("period must be 'quarterly' or 'annual'")

    table = clean_facts(ticker)
    annual_ends = {r["period_end"] for r in table if r["kind"] == "annual"}
    out = []
    for m in metrics:
        if m == "cash":   # balance-sheet item: pick instants at quarter or year ends
            rows = [r for r in table if r["metric"] == m and
                    (period == "quarterly" or r["period_end"] in annual_ends)]
        else:
            kind = "quarter" if period == "quarterly" else "annual"
            rows = [r for r in table if r["metric"] == m and r["kind"] == kind and
                    (include_derived or not r["derived"])]
        out.extend(rows[-last:])
    return [dict(r) for r in out]


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    for r in get_financials(t, ["revenue", "net_income", "eps_diluted"], "quarterly", last=5):
        flag = " [derived]" if r["derived"] else ""
        print(f"{r['metric']:<12} {r['period_label']:<36} {r['value_text']:>22}{flag}   {r['form']} filed {r['filed']}")
