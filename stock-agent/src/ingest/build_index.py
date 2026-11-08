"""Chunk the manifest and embed every chunk (only new ones are paid for).

Run:  python -m ingest.build_index
      python -m ingest.build_index --yes      # skip the cost confirmation
"""

from __future__ import annotations

import argparse
import json

from . import chunk
from .embed import INDEX_DIR, embed_chunks_cached

PRICE_PER_M_TOKENS = 0.15   # gemini-embedding-001 on Vertex, standard (non-batch) rate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true", help="don't ask before spending credits")
    args = p.parse_args()

    chunk.main()
    chunks = [json.loads(line) for line in chunk.CHUNKS_PATH.open()]

    ids_path = INDEX_DIR / "ids.json"
    have = set(json.loads(ids_path.read_text())) if ids_path.exists() else set()
    todo = [c for c in chunks if c["chunk_id"] not in have]
    toks = sum(chunk.n_tokens(c["index_text"]) for c in todo)
    print(f"\nTo embed: {len(todo)} new chunks, ~{toks:,} tokens, ~${toks / 1e6 * PRICE_PER_M_TOKENS:.2f}")
    if todo and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    embed_chunks_cached([c["chunk_id"] for c in chunks], [c["index_text"] for c in chunks])
    print("Index ready.")


if __name__ == "__main__":
    main()
