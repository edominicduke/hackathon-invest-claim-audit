"""Split every document in the manifest into citable chunks.

Rules:
  - Split on natural units, never mid-unit:
      * transcripts: one speaker turn ("**Name (Title)**" blocks)
      * everything else: paragraphs, with markdown tables kept whole
  - Pack units together up to ~TARGET tokens, never over MAX (the embedding
    model silently truncates past 2,048 tokens).
  - A unit bigger than MAX is split by sentences (tables: by rows, repeating
    the header row so each piece still makes sense).
  - Every chunk gets a short header ("NVDA | 10-K | Item 1A Risk Factors |
    2026-02-25") added to the text that gets embedded and keyword-indexed, so
    a chunk that never names the company still matches "NVIDIA risks".

Output: data/index/chunks.jsonl, one chunk per line.

Run:  python -m ingest.chunk
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ingest.common import MANIFEST

TARGET_TOKENS = 700
MAX_TOKENS = 1400
CHUNKS_PATH = Path("data/index/chunks.jsonl")

_TURN_START = re.compile(r"^\*\*[^*\n]{1,120}\*\*\s*$", re.M)   # "**Colette Kress (CFO)**"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def n_tokens(text: str) -> int:
    return len(text) // 4 + 1   # rough, but fine for sizing chunks


def quarter_to_date(q: str) -> str:
    """'2025Q3' -> '2025-09-30' so transcripts sort/filter alongside filings.
    Calendar quarters; fiscal quarters may be offset (e.g. NVIDIA)."""
    m = re.fullmatch(r"(\d{4})Q([1-4])", q or "")
    if not m:
        return q
    return f"{m[1]}-{['03-31', '06-30', '09-30', '12-31'][int(m[2]) - 1]}"


# ---------- splitting a document into units ----------

def split_units(text: str, doc_type: str) -> list[str]:
    if doc_type in ("earnings_call", "8-K_transcript") and _TURN_START.search(text):
        starts = [m.start() for m in _TURN_START.finditer(text)]
        if starts[0] > 0:
            starts.insert(0, 0)
        return [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)]) if text[a:b].strip()]
    return _paragraphs_and_tables(text)


def _paragraphs_and_tables(text: str) -> list[str]:
    units, para, table = [], [], []

    def flush(buf):
        if buf and "\n".join(buf).strip():
            units.append("\n".join(buf).strip())
        buf.clear()

    for line in text.splitlines():
        is_table_row = line.lstrip().startswith("|")
        if is_table_row:
            flush(para)
            table.append(line)
        else:
            flush(table)
            if line.strip():
                para.append(line)
            else:
                flush(para)
    flush(para)
    flush(table)
    return units


def split_oversized(unit: str) -> list[str]:
    if n_tokens(unit) <= MAX_TOKENS:
        return [unit]
    lines = unit.splitlines()
    if lines and lines[0].lstrip().startswith("|"):          # table: split by rows, repeat header
        header = lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ") else lines[:1]
        pieces, cur = [], list(header)
        for row in lines[len(header):]:
            if n_tokens("\n".join(cur + [row])) > MAX_TOKENS and len(cur) > len(header):
                pieces.append("\n".join(cur))
                cur = list(header)
            cur.append(row)
        pieces.append("\n".join(cur))
        return pieces
    pieces, cur = [], ""                                       # prose: split by sentences
    for sent in _SENTENCE_END.split(unit):
        if cur and n_tokens(cur + " " + sent) > MAX_TOKENS:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    # a single absurdly long "sentence" (e.g. a run-on list): hard-cut by characters
    return [p[i:i + MAX_TOKENS * 4] for p in pieces for i in range(0, len(p), MAX_TOKENS * 4)]


def pack(units: list[str]) -> list[str]:
    chunks, cur = [], []
    for unit in (u for raw in units for u in split_oversized(raw)):
        if cur and n_tokens("\n\n".join(cur + [unit])) > TARGET_TOKENS:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(unit)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ---------- manifest -> chunks ----------

def citation_label(rec: dict) -> str:
    """Human-readable source line shown next to a claim."""
    t, d = rec["doc_type"], rec["date"]
    if t == "paper":
        title = rec.get("extra", {}).get("title", rec["section"])
        return f'arXiv {rec["section"]}: "{title}" ({d})'
    if t == "earnings_call":
        return f"{rec['ticker']} {d} earnings call"
    if t.startswith("8-K"):
        kind = "transcript" if t.endswith("transcript") else "earnings press release"
        return f"{rec['ticker']} 8-K {kind} (filed {d})"
    return f"{rec['ticker']} {t} (filed {d}), {rec['section']}"


def chunk_manifest() -> list[dict]:
    records = [json.loads(line) for line in MANIFEST.open()]
    out = []
    for rec in records:
        text = Path(rec["path"]).read_text(encoding="utf-8")
        extra = rec.get("extra") or {}
        # Papers carry the tickers whose technology they cover; everything else is its own ticker.
        tags = extra.get("tickers") or [rec["ticker"]]
        if rec["doc_type"] == "paper":
            header = " | ".join(["research paper", "relevant to " + ", ".join(tags),
                                 "; ".join(extra.get("topics", [])), rec["date"]])
        else:
            header = " | ".join([rec["ticker"], rec["doc_type"], rec["section"], rec["date"]])
        for n, body in enumerate(pack(split_units(text, rec["doc_type"]))):
            out.append({
                "chunk_id": f"{rec['doc_id']}#{n:03d}",
                "doc_id": rec["doc_id"],
                "ticker": rec["ticker"],
                "tags": tags,
                "source": rec["source"],
                "doc_type": rec["doc_type"],
                "section": rec["section"],
                "date": rec["date"],
                "date_norm": quarter_to_date(rec["date"]),
                "period": rec.get("period"),
                "url": rec["url"],
                "citation": citation_label(rec),
                "text": body,                          # what the user sees / gets quoted
                "index_text": f"{header}\n{body}",     # what gets embedded + keyword-indexed
            })
    return out


def main() -> None:
    chunks = chunk_manifest()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    toks = sum(n_tokens(c["index_text"]) for c in chunks)
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1
    print(f"{len(chunks)} chunks, ~{toks:,} tokens -> {CHUNKS_PATH}")
    for t, n in sorted(by_type.items()):
        print(f"   {t}: {n}")


if __name__ == "__main__":
    main()