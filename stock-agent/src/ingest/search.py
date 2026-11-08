"""Hybrid search over the index: keyword (BM25) + embeddings, merged with
reciprocal rank fusion. This is the function teammates call.

    from ingest.search import search, format_for_prompt

    hits = search("Did Intel hit its 18A process timeline?",
                  tickers=["INTC"], doc_types=["10-K", "earnings_call"],
                  date_from="2025-01-01", k=8)
    prompt_block, sources = format_for_prompt(hits)
    # prompt_block: "[S1] INTC 10-K (filed ...), Item 7 MD&A\n<text>\n\n[S2] ..."
    # sources:      {"S1": hit, "S2": hit, ...}  -> keep this to check citations

    # Rivals' evidence (e.g. for Challenge My Trade):
    rival_hits = search_competitors("NVDA", "AI accelerator market share", k=4)

Valid filter values: DOC_TYPES below (with descriptions) and whatever tickers
are indexed (see describe_index()). Bad values raise ValueError listing the
valid ones, so an agent can correct itself instead of getting an empty list.

Try it from the terminal:
    python -m ingest.search --describe
    python -m ingest.search "data center revenue growth" --tickers NVDA AMD
    python -m ingest.search "AI accelerator demand" --competitors-of NVDA
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import CHUNKS_PATH
from .embed import INDEX_DIR, embed_query

RRF_K = 60          # standard constant for reciprocal rank fusion
CANDIDATES = 50     # how many each method contributes before merging

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")   # keeps "18a", "30.8", "h100", "q3-2025"


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# Every doc_type the fetchers produce. Copy these into the agent's tool
# definition (as an enum) so the model can't invent values like "transcript".
DOC_TYPES: dict[str, str] = {
    "10-K": "Annual report. Sections: Item 1 Business, Item 1A Risk Factors, "
            "Item 7 MD&A (management's discussion of results), Item 7A Market Risk.",
    "10-Q": "Quarterly report. Sections: Part I Item 2 MD&A, Part II Item 1A Risk Factors.",
    "8-K_press_release": "Earnings press release filed with the SEC (results + often next-quarter guidance).",
    "8-K_transcript": "Earnings call transcript filed with the SEC (rare; most come from earnings_call).",
    "earnings_call": "Earnings call transcript (Alpha Vantage): management remarks and analyst Q&A. "
                     "Turns labeled (Analyst) are questions, not company statements.",
    "paper": "Research paper abstract (arXiv), tagged with the companies whose technology area it "
             "covers. Evidence about the technology, NOT a statement by the company; authors are "
             "usually not company employees.",
}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def describe_index() -> dict:
    """What's actually searchable right now: tickers, doc types, date ranges.
    Useful for writing the tool description and for debugging empty results."""
    chunks, _, _ = _load()
    out: dict = {"tickers": {}, "doc_types": DOC_TYPES}
    for c in chunks:
        t = out["tickers"].setdefault(c["ticker"], {"chunks": 0, "doc_types": set(), "dates": [c["date_norm"]] * 2})
        t["chunks"] += 1
        t["doc_types"].add(c["doc_type"])
        t["dates"] = [min(t["dates"][0], c["date_norm"]), max(t["dates"][1], c["date_norm"])]
    for t in out["tickers"].values():
        t["doc_types"] = sorted(t["doc_types"])
    return out


def _validate(tickers, doc_types, date_from, date_to, chunks) -> None:
    """Fail loudly on bad filters. A silent [] reads to an agent as
    'no evidence exists', which is worse than an error it can correct."""
    if isinstance(tickers, str) or isinstance(doc_types, str):
        raise TypeError("tickers and doc_types must be lists, e.g. tickers=['NVDA']")
    known_tickers = sorted({t for c in chunks for t in [c["ticker"], *c.get("tags", [])]})
    bad = [t for t in (tickers or []) if t.upper() not in known_tickers]
    if bad:
        raise ValueError(f"Unknown ticker(s) {bad}. Indexed tickers: {known_tickers}")
    bad = [d for d in (doc_types or []) if d not in DOC_TYPES]
    if bad:
        raise ValueError(f"Unknown doc_type(s) {bad}. Valid doc_types: {list(DOC_TYPES)}")
    for name, val in (("date_from", date_from), ("date_to", date_to)):
        if val is not None and not _DATE.fullmatch(val):
            raise ValueError(f"{name}={val!r} must be 'YYYY-MM-DD'")


@lru_cache(maxsize=1)
def _load():
    chunks = [json.loads(line) for line in CHUNKS_PATH.open()]
    ids = json.loads((INDEX_DIR / "ids.json").read_text())
    vecs = np.load(INDEX_DIR / "embeddings.npy")
    row_of = {cid: i for i, cid in enumerate(ids)}
    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in row_of]
    if missing:
        raise RuntimeError(f"{len(missing)} chunks have no embedding. Run: python -m ingest.build_index")
    vecs = vecs[[row_of[c["chunk_id"]] for c in chunks]]       # align vectors to chunk order
    bm25 = BM25Okapi([tokenize(c["index_text"]) for c in chunks])
    return chunks, vecs, bm25


def search(query: str, tickers: list[str] | None = None, doc_types: list[str] | None = None,
           date_from: str | None = None, date_to: str | None = None, k: int = 8,
           min_similarity: float | None = None, mode: str = "hybrid") -> list[dict]:
    """Returns up to k chunks, best first. Filters are applied BEFORE ranking,
    so when `tickers` is given, other companies' chunks are excluded.
    Dates are 'YYYY-MM-DD' (transcript quarters are mapped to quarter-end).

    Each hit has two scores:
      score       rank-fusion score, for ORDERING only. Always ~0.016-0.033 for
                  the top hits, even when nothing relevant exists.
      similarity  cosine similarity between query and chunk (-1..1), an
                  ABSOLUTE relevance signal. Use it to decide "is there real
                  evidence?" Calibrate the cutoff on your data first.
    min_similarity drops hits below that cosine similarity (so a query with no
    good evidence returns [] instead of the least-bad chunks).
    mode: "hybrid" (default), or "dense" / "keyword" alone (used by the eval).
    """
    if mode not in ("hybrid", "dense", "keyword"):
        raise ValueError("mode must be 'hybrid', 'dense' or 'keyword'")
    chunks, vecs, bm25 = _load()
    _validate(tickers, doc_types, date_from, date_to, chunks)

    want = {t.upper() for t in tickers} if tickers else None
    mask = np.ones(len(chunks), dtype=bool)
    for i, c in enumerate(chunks):
        # a chunk matches if it IS the company, or (papers) is tagged with it
        if want and c["ticker"] not in want and not want.intersection(c.get("tags", [])):
            mask[i] = False
        elif doc_types and c["doc_type"] not in doc_types:
            mask[i] = False
        elif date_from and c["date_norm"] < date_from:
            mask[i] = False
        elif date_to and c["date_norm"] > date_to:
            mask[i] = False
    allowed = np.flatnonzero(mask)
    if allowed.size == 0:
        return []

    sparse = bm25.get_scores(tokenize(query))[allowed]
    dense = None
    if mode != "keyword":
        try:
            dense = vecs[allowed] @ embed_query(query)
        except Exception as e:
            # No Vertex (credits gone, offline, auth expired): keyword search still works.
            _warn_once(f"Embedding unavailable ({type(e).__name__}); using keyword-only search.")
            mode = "keyword"

    fused: dict[int, float] = {}
    lists = {"hybrid": (dense, sparse), "dense": (dense,), "keyword": (sparse,)}[mode]
    for scores in lists:
        top = np.argsort(-scores)[:CANDIDATES]
        for rank, j in enumerate(top):
            if scores[j] <= 0 and scores is sparse:
                break                       # no keyword overlap at all: don't reward it
            idx = int(allowed[j])
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    similarity = ({int(allowed[j]): float(dense[j]) for j in range(allowed.size)}
                  if dense is not None else {})
    ranked = sorted(fused, key=fused.get, reverse=True)
    if min_similarity is not None and similarity:          # can't apply a cosine cutoff without embeddings
        ranked = [i for i in ranked if similarity[i] >= min_similarity]
    return [{**{f: chunks[i][f] for f in chunks[i] if f != "index_text"},
             "score": round(fused[i], 5),
             "similarity": round(similarity[i], 3) if similarity else None}
            for i in ranked[:k]]


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[search] {msg}", file=sys.stderr)


# Direct competitors within the demo set. A ticker with no peers here gets no
# competitor evidence; add tickers to the fetch list to fill those in.
PEERS: dict[str, list[str]] = {
    "NVDA": ["AMD", "INTC"],
    "AMD": ["NVDA", "INTC"],
    "INTC": ["AMD", "NVDA"],
    "TSLA": [],
    "GOOGL": [],
}


def search_competitors(ticker: str, query: str, k: int = 5, **filters) -> list[dict]:
    """Evidence from a company's rivals, e.g. threats to its market position.

    Every hit's citation names the rival ("AMD 8-K earnings press release..."),
    so the answer must attribute the fact to that rival, never to `ticker`.
    Accepts the same doc_types / date_from / date_to filters as search().
    """
    peers = PEERS.get(ticker.upper(), [])
    if not peers:
        return []
    # Papers are about technology, not a rival's own statements: leave them out
    # unless the caller explicitly asks for doc_types including "paper".
    if not filters.get("doc_types"):
        filters["doc_types"] = [d for d in DOC_TYPES if d != "paper"]
    hits = search(query, tickers=peers, k=k, **filters)
    for h in hits:
        h["competitor_of"] = ticker.upper()   # lets the prompt / UI flag it as rival evidence
    return hits


def format_for_prompt(hits: list[dict]) -> tuple[str, dict[str, dict]]:
    """Label hits [S1]..[Sn] for the model and return the label->hit map
    you need later to check that every cited label was actually retrieved."""
    sources = {f"S{n}": h for n, h in enumerate(hits, 1)}
    def label(h):
        rival = f" [COMPETITOR of {h['competitor_of']}: attribute to {h['ticker']}]" if h.get("competitor_of") else ""
        return f"{h['citation']}{rival}"
    block = "\n\n".join(f"[{sid}] {label(h)}\n{h['text']}" for sid, h in sources.items())
    return block, sources


def main() -> None:
    if sys.argv[1:] == ["--describe"]:
        info = describe_index()
        for t, d in sorted(info["tickers"].items()):
            print(f"{t}: {d['chunks']} chunks, {d['dates'][0]} .. {d['dates'][1]}, {', '.join(d['doc_types'])}")
        return
    p = argparse.ArgumentParser(epilog="Run with only --describe to list what's indexed.")
    p.add_argument("query")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--doc-types", nargs="*")
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--competitors-of", help="search this ticker's rivals instead (ignores --tickers)")
    p.add_argument("--min-sim", type=float, help="drop hits below this cosine similarity")
    a = p.parse_args()
    if a.competitors_of:
        hits = search_competitors(a.competitors_of, a.query, k=a.k, doc_types=a.doc_types,
                                  date_from=a.date_from, date_to=a.date_to, min_similarity=a.min_sim)
    else:
        hits = search(a.query, a.tickers, a.doc_types, a.date_from, a.date_to, a.k, a.min_sim)
    if not hits:
        print("No results" + (f" above similarity {a.min_sim}" if a.min_sim else ""))
    for n, h in enumerate(hits, 1):
        preview = " ".join(h["text"].split())[:220]
        print(f"[S{n}] {h['citation']}  (rank score {h['score']}, similarity {h['similarity']})\n"
              f"     id: {h['chunk_id']}\n     {preview}...\n")


if __name__ == "__main__":
    main()
