"""Earnings call transcripts from Alpha Vantage (free key, 25 requests/day).

One request = one transcript (ticker + quarter), so 6 tickers x 4 quarters
= 24 requests, which just fits one day's quota. The script skips anything
already on disk, so if you hit the limit, re-run tomorrow (or with a
teammate's key) and it picks up where it stopped.

Setup:
  export ALPHAVANTAGE_API_KEY=...   # free: https://www.alphavantage.co/support/#api-key

Run:
  python -m ingest.transcripts_fetch NVDA AMD --quarters 2025Q1 2025Q2 2025Q3 2025Q4

NOTE: check that the quarter labels line up with each company's fiscal
calendar (NVIDIA's fiscal year ends in late January, for example). Fetch one
transcript and read its first lines before fetching everything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

from .common import Doc, already_have, save

API = "https://www.alphavantage.co/query"


class RateLimited(Exception):
    pass


def fetch_transcript(ticker: str, quarter: str, api_key: str) -> list[dict]:
    resp = requests.get(API, params={
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": ticker, "quarter": quarter, "apikey": api_key,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Alpha Vantage returns HTTP 200 with a message instead of data when you
    # are over quota, so check the body.
    msg = data.get("Information") or data.get("Note") or data.get("Error Message")
    if msg:
        if "rate limit" in msg.lower() or "requests per day" in msg.lower() or "premium" in msg.lower():
            raise RateLimited(msg)
        raise RuntimeError(msg)
    return data.get("transcript", [])


def turns_to_markdown(turns: list[dict]) -> str:
    """One block per speaker turn. Keep turns intact when chunking later:
    a turn is the natural unit to cite ("CFO, Q3 2025 call: ...")."""
    blocks = []
    for t in turns:
        who = t.get("speaker", "Unknown")
        title = t.get("title")
        header = f"**{who}" + (f" ({title})" if title else "") + "**"
        blocks.append(f"{header}\n{t.get('content', '').strip()}")
    return "\n\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--quarters", nargs="+", required=True, help="e.g. 2025Q1 2025Q2")
    args = p.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        sys.exit("Set ALPHAVANTAGE_API_KEY first.")

    for ticker in (t.upper() for t in args.tickers):
        for q in args.quarters:
            probe = Doc(ticker=ticker, source="alpha_vantage", doc_type="earnings_call",
                        section="full", date=q, url="", text="")
            if already_have(probe.doc_id):
                print(f"{ticker} {q}: already have it")
                continue
            try:
                turns = fetch_transcript(ticker, q, api_key)
            except RateLimited as e:
                sys.exit(f"Hit the Alpha Vantage limit at {ticker} {q}. Re-run later.\n  {e}")
            except Exception as e:
                print(f"{ticker} {q}: FAILED ({e})")
                continue
            if not turns:
                print(f"{ticker} {q}: no transcript returned")
                continue
            ok = save(Doc(
                ticker=ticker, source="alpha_vantage", doc_type="earnings_call",
                section="full", date=q, period=q,
                url=f"{API}?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={q}",
                text=turns_to_markdown(turns),
                extra={"n_turns": len(turns)},
            ))
            print(f"{ticker} {q}: {'saved' if ok else 'skipped'} ({len(turns)} turns)")
            time.sleep(13)  # free tier also caps you at 5 requests/minute


if __name__ == "__main__":
    main()
