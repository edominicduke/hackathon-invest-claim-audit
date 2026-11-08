"""Did the company deliver what it guided? Revenue guidance vs. reported results.

    python -m ingest.guidance --extract        # Gemini reads each earnings press release (~cents)
    python -m ingest.guidance --review         # print every extracted guide next to its quote: CHECK THESE
    python -m ingest.guidance --show NVDA      # the record: guided vs. actual, verdict, gap

How it works:
  1. Extract: Gemini pulls revenue guidance for the NEXT quarter out of each 8-K
     earnings press release (low / high / point, GAAP or not, the exact quote).
     The quote is then checked against the press release text: if it isn't
     really there, the guide is flagged and not used.
  2. Match: guidance in a release filed on date D is for the first fiscal quarter
     that ENDS after D. That works for any fiscal calendar, with no quarter
     names to translate (NVIDIA's "Q3 FY2027" is calendar 2026).
  3. Compare with the reported revenue for that quarter (financials.py):
       beat   = above the top of the range
       met    = inside the range
       missed = below the bottom
       pending = that quarter hasn't been reported yet

Saved to data/guidance/guidance.json. It's ~15 rows: open it, compare each
number with its quote, fix anything wrong and set "reviewed": true.

Read the record carefully: many companies guide conservatively on purpose, so
"beat every quarter" mostly measures how cautious the guidance is. Misses (how
often, how big) are the stronger signal of over-promising.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from .common import DATA_ROOT, MANIFEST, resolve_path
from .financials import _fmt_value, clean_facts

GUIDANCE_PATH = DATA_ROOT / "guidance" / "guidance.json"
MAX_DAYS_AHEAD = 130     # target quarter must end within this many days after the release

EXTRACT_PROMPT = """Below is a company's earnings press release ({ticker}, filed {filed}).

Extract REVENUE guidance (the company's forecast of future revenue), if any.
- Only forward-looking guidance ("outlook", "expects", "anticipates"). NOT reported results.
- Convert to absolute US dollars: "$54.0 billion" -> 54000000000.
- "plus or minus 2%" around a point: give point, and low/high = point -/+ 2%.
- "plus or minus $300 million": low/high = point -/+ 300000000.
- A range "$12.8 billion to $13.8 billion": low and high; point = midpoint.
- period_type: "next_quarter" if it's for the upcoming quarter, "full_year" for a fiscal year, else "other".
- quote: copy the guidance sentence EXACTLY as written in the text (verbatim, no paraphrase).
- notes: anything that affects comparison (e.g. "excludes H20 revenue", "assumes no China sales"), else "".
- If there is no numeric revenue guidance, return an empty list.

Press release:
\"\"\"
{text}
\"\"\"

Return JSON only:
{{"guidance": [{{"metric": "revenue", "period_type": "...", "period_text": "...", "low": number|null,
  "high": number|null, "point": number|null, "basis": "GAAP"|"non-GAAP"|"unspecified",
  "quote": "...", "notes": "..."}}]}}"""


def _norm(s: str) -> str:
    s = s.replace("\\", "").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s.lower()).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """Verbatim check (ignoring whitespace, case, markdown escapes); falls back to
    95% word overlap in order, for small formatting differences."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    words = q.split()
    if len(words) < 6:
        return False
    hits, pos = 0, 0
    for w in words:
        i = t.find(w, pos)
        if i >= 0:
            hits, pos = hits + 1, i + len(w)
    return hits / len(words) >= 0.95


# ---------- 1) extract ----------

def _press_releases() -> list[dict]:
    recs = [json.loads(l) for l in MANIFEST.open()]
    return [r for r in recs if r["doc_type"] == "8-K_press_release"]


def extract(workers: int = 6) -> None:
    from .llm import ask_json
    docs = _press_releases()
    print(f"Reading {len(docs)} earnings press releases...")

    def one(rec):
        text = resolve_path(rec["path"]).read_text(encoding="utf-8")
        try:
            out = ask_json(EXTRACT_PROMPT.format(ticker=rec["ticker"], filed=rec["date"], text=text[:40000]))
        except Exception as e:
            print(f"  {rec['ticker']} {rec['date']}: failed ({e})")
            return []
        rows = []
        for g in out.get("guidance", []):
            if g.get("metric") != "revenue":
                continue
            low, high, point = g.get("low"), g.get("high"), g.get("point")
            if point is None and low is not None and high is not None:
                point = (low + high) / 2
            if low is None and high is None and point is not None:
                low = high = point
            rows.append({
                "ticker": rec["ticker"], "filed": rec["date"], "doc_id": rec["doc_id"], "url": rec["url"],
                "exhibit": rec["section"], "metric": "revenue", "period_type": g.get("period_type", "other"),
                "period_text": g.get("period_text", ""), "low": low, "high": high, "point": point,
                "basis": g.get("basis", "unspecified"), "quote": g.get("quote", ""), "notes": g.get("notes", ""),
                "quote_found": quote_in_text(g.get("quote", ""), text), "reviewed": False,
            })
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for batch in pool.map(one, docs) for r in batch]

    # one guide per (ticker, filing date, period type): prefer a verified quote
    best: dict[tuple, dict] = {}
    for r in sorted(rows, key=lambda r: not r["quote_found"]):
        best.setdefault((r["ticker"], r["filed"], r["period_type"]), r)
    rows = sorted(best.values(), key=lambda r: (r["ticker"], r["filed"]))

    # keep human edits from a previous run
    if GUIDANCE_PATH.exists():
        old = {(r["ticker"], r["filed"], r["period_type"]): r for r in json.loads(GUIDANCE_PATH.read_text())}
        rows = [old[k] if (k := (r["ticker"], r["filed"], r["period_type"])) in old and old[k].get("reviewed") else r
                for r in rows]

    GUIDANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDANCE_PATH.write_text(json.dumps(rows, indent=1))
    bad = [r for r in rows if not r["quote_found"]]
    tickers = sorted({r["ticker"] for r in docs})
    found = sorted({r["ticker"] for r in rows})
    print(f"{len(rows)} revenue guides saved -> {GUIDANCE_PATH}")
    print(f"  with guidance: {found}   no numeric guidance found: {[t for t in tickers if t not in found]}")
    if bad:
        print(f"  {len(bad)} quote(s) NOT found verbatim in the press release: these are excluded until fixed")
    print("Next: python -m ingest.guidance --review   (check each number against its quote)")


# ---------- 2) match + compare ----------

def _load() -> list[dict]:
    if not GUIDANCE_PATH.exists():
        return []
    return json.loads(GUIDANCE_PATH.read_text())


def _verdict(actual: float, low: float, high: float) -> str:
    if actual > high:
        return "beat"
    if actual < low:
        return "missed"
    return "met"


def guidance_record(ticker: str, last: int = 6) -> dict:
    """Comparisons (oldest first) + a summary. Each comparison carries two citable
    items: the guidance quote and the reported revenue row."""
    ticker = ticker.upper()
    mine = [g for g in _load() if g["ticker"] == ticker and g["period_type"] == "next_quarter"]
    guides = [g for g in mine if g["quote_found"] and g["low"] is not None]
    if not guides:
        msg = (f"{ticker}'s revenue guidance couldn't be verified against its press releases, so no record is shown."
               if mine else f"No numeric revenue guidance found in {ticker}'s earnings press releases.")
        return {"ticker": ticker, "guidance_given": False, "comparisons": [], "summary": msg}
    try:
        quarters = [r for r in clean_facts(ticker) if r["metric"] == "revenue" and r["kind"] == "quarter"]
    except ValueError:
        quarters = []

    comps = []
    for g in sorted(guides, key=lambda g: g["filed"])[-last:]:
        filed = date.fromisoformat(g["filed"])
        target = next((q for q in sorted(quarters, key=lambda q: q["period_end"])
                       if filed < date.fromisoformat(q["period_end"]) <= filed + timedelta(days=MAX_DAYS_AHEAD)), None)
        guide_item = _guide_item(g)
        if target is None:
            comps.append({"guide": guide_item, "actual": None, "verdict": "pending",
                          "period": f"quarter after {g['filed']}", "gap_vs_mid_pct": None})
            continue
        actual = target["value"]
        comps.append({
            "guide": guide_item, "actual": dict(target), "verdict": _verdict(actual, g["low"], g["high"]),
            "period": target["period_label"],
            "gap_vs_mid_pct": round((actual - g["point"]) / g["point"] * 100, 1),
        })

    done = [c for c in comps if c["verdict"] != "pending"]
    counts = {v: sum(c["verdict"] == v for c in done) for v in ("beat", "met", "missed")}
    gaps = [c["gap_vs_mid_pct"] for c in done]
    parts = [f"{ticker} revenue vs. its own guidance, last {len(done)} reported quarters: "
             f"{counts['beat']} beat, {counts['met']} within range, {counts['missed']} missed."]
    if gaps:
        parts.append(f"Average result vs. guidance midpoint: {sum(gaps) / len(gaps):+.1f}%.")
        worst = min(done, key=lambda c: c["gap_vs_mid_pct"])
        if worst["gap_vs_mid_pct"] < 0:
            parts.append(f"Largest shortfall: {worst['gap_vs_mid_pct']:+.1f}% ({worst['period']}).")
    if len(done) < len(comps):
        parts.append("Latest guidance is pending (quarter not reported yet).")
    if any(c["actual"] and c["actual"].get("derived") for c in done):
        parts.append("Some actuals are derived Q4 values (full year minus 9 months).")
    return {"ticker": ticker, "guidance_given": True, "comparisons": comps, "counts": counts,
            "summary": " ".join(parts), "unreviewed": sum(not g.get("reviewed") for g in guides)}


def _guide_item(g: dict) -> dict:
    """The guidance as a citable source (same shape as a search hit)."""
    low, high = _fmt_value(g["low"], "USD"), _fmt_value(g["high"], "USD")
    rng = low if g["low"] == g["high"] else f"{low} to {high}"
    text = (f"{g['ticker']} revenue guidance for the next quarter, issued {g['filed']}: {rng} "
            f"(midpoint {_fmt_value(g['point'], 'USD')}, {g['basis']}). Quote: \"{g['quote']}\""
            + (f" Note: {g['notes']}" if g.get("notes") else ""))
    return {"chunk_id": f"GUIDE::{g['ticker']}::{g['filed']}::revenue", "ticker": g["ticker"],
            "doc_type": "guidance", "date": g["filed"], "section": "Outlook",
            "citation": f"{g['ticker']} 8-K earnings press release (filed {g['filed']}), revenue outlook",
            "text": text, "url": g["url"], "similarity": None, "range_text": rng}


# ---------- CLI ----------

def review() -> None:
    rows = _load()
    if not rows:
        raise SystemExit("Nothing extracted yet. Run: python -m ingest.guidance --extract")
    for r in rows:
        flag = "" if r["quote_found"] else "   <-- QUOTE NOT FOUND IN RELEASE (excluded)"
        rng = f"{_fmt_value(r['low'], 'USD')} to {_fmt_value(r['high'], 'USD')}" if r["low"] is not None else "?"
        print(f"\n{r['ticker']}  filed {r['filed']}  [{r['period_type']}, {r['basis']}]  "
              f"{'reviewed' if r.get('reviewed') else 'NOT reviewed'}{flag}")
        print(f"   extracted: {rng}" + (f"   notes: {r['notes']}" if r["notes"] else ""))
        print(f"   quote:     {r['quote']}")
    print(f"\nEdit {GUIDANCE_PATH} to fix any number, then set \"reviewed\": true.")


def show(ticker: str) -> None:
    rec = guidance_record(ticker)
    print(rec["summary"] + "\n")
    for c in rec["comparisons"]:
        actual = c["actual"]["value_text"] + (" (derived)" if c["actual"].get("derived") else "") if c["actual"] else "-"
        gap = f"{c['gap_vs_mid_pct']:+.1f}%" if c["gap_vs_mid_pct"] is not None else ""
        print(f"  {c['period']:<40} guided {c['guide']['range_text']:<38} actual {actual:<26} {c['verdict']:<8} {gap}")
    if rec.get("unreviewed"):
        print(f"\n  ({rec['unreviewed']} guide(s) not reviewed yet: python -m ingest.guidance --review)")


def main() -> None:
    p = argparse.ArgumentParser(description="Revenue guidance vs. reported results.")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--review", action="store_true")
    p.add_argument("--show", metavar="TICKER")
    a = p.parse_args()
    if a.extract:
        extract()
    elif a.review:
        review()
    elif a.show:
        show(a.show)
    else:
        p.print_help()


if __name__ == "__main__":
    main()