"""Shared helpers for every fetcher: one document record format + one manifest.

Every fetcher writes:
  data/raw/<TICKER>/<doc_id>.md      the document text
  data/raw/manifest.jsonl            one JSON line per document (metadata)

The chunker reads the manifest, so every source ends up with the same labels
(ticker, source, doc_type, section, date, url). Those labels become the
citation shown to the user.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data/raw")
MANIFEST = DATA_DIR / "manifest.jsonl"


@dataclass
class Doc:
    ticker: str
    source: str            # "sec_edgar" | "alpha_vantage" | "arxiv" | "patentsview"
    doc_type: str          # "10-K" | "10-Q" | "8-K_press_release" | "earnings_call" | ...
    section: str           # "Item 1A Risk Factors", "full", "EX-99.1", ...
    date: str              # filing / call / publication date, YYYY-MM-DD
    url: str               # link the user can click to check the claim
    text: str = field(repr=False)
    period: Optional[str] = None      # fiscal period the doc covers, e.g. "2025-10-26" or "2025Q3"
    accession: Optional[str] = None   # SEC accession number, when there is one
    extra: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        parts = [self.ticker, self.doc_type, self.date, self.section]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", "__".join(parts))[:180]


def already_have(doc_id: str) -> bool:
    """Lets fetchers skip work on re-runs (important for rate-limited APIs)."""
    if not MANIFEST.exists():
        return False
    with MANIFEST.open() as f:
        return any(json.loads(line).get("doc_id") == doc_id for line in f)


def save(doc: Doc, min_chars: int = 200) -> bool:
    """Write the text file + append a manifest line. Returns False if skipped."""
    if not doc.text or len(doc.text.strip()) < min_chars:
        return False
    if already_have(doc.doc_id):
        return False
    folder = DATA_DIR / doc.ticker
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc.doc_id}.md"
    path.write_text(doc.text, encoding="utf-8")

    record = {k: v for k, v in asdict(doc).items() if k != "text"}
    record.update(doc_id=doc.doc_id, path=str(path), n_chars=len(doc.text))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return True