"""Validate the LLM's ticker guess against a lookup table: LLM for fuzzy understanding,
code for facts.

Uses SEC's company_tickers.json if it is present at data/company_tickers.json (Guru
already pulls from EDGAR and can drop it there); otherwise a small built-in table.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SEC_FILE = Path(__file__).resolve().parent.parent / "data" / "company_tickers.json"

_BUILTIN = {
    "EXMP": "Exemplar Semiconductor Inc. (fictional demo company)",
    "RVMC": "Rival Micro Devices (fictional demo company)",
    "NVDA": "NVIDIA Corp",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corp",
    "AVGO": "Broadcom Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing Co.",
    "BRK.B": "Berkshire Hathaway Inc. (Class B)",
    "JPM": "JPMorgan Chase & Co.",
}


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    table = dict(_BUILTIN)
    if SEC_FILE.exists():
        rows = json.loads(SEC_FILE.read_text(encoding="utf-8"))
        table.update({r["ticker"].upper(): r["title"] for r in rows.values()})
    return table


def lookup(ticker: str | None) -> str | None:
    """Return the company name for a known ticker, else None."""
    if not ticker:
        return None
    return _table().get(ticker.strip().upper().replace("-", "."))
