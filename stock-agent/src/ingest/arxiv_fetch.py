"""Research papers from arXiv, tagged with the companies whose technology they cover.

Papers don't belong to one company: a paper on low-precision GPU training is
evidence for NVIDIA, AMD and Intel at once. So each search topic below lists
the tickers it's relevant to, and every paper it finds gets those tags. A
paper found by several topics gets all their tags. search(tickers=["AMD"])
then includes papers tagged AMD.

Stored as ticker="RESEARCH", doc_type="paper", one document per paper
(title + authors + abstract; abstracts are ~250 words, so one chunk each).

No API key needed. arXiv asks for <= 1 request / 3 s; the client enforces it.

Run:
  pip install arxiv
  python -m ingest.arxiv_fetch                    # all topics, papers since 2024-01-01
  python -m ingest.arxiv_fetch --since 2025-01-01 --per-topic 15
Then rebuild the index:  python -m ingest.build_index
"""

from __future__ import annotations

import argparse
import re
from datetime import date

import arxiv

from .common import Doc, save

# topic name -> (arXiv query, tickers it's evidence for)
# Queries use arXiv syntax: abs: = abstract, ti: = title, cat: = category.
TOPICS: dict[str, tuple[str, list[str]]] = {
    "AI accelerators / GPU training": (
        '(abs:"GPU" OR abs:"accelerator") AND abs:"deep learning" AND (cat:cs.AR OR cat:cs.DC OR cat:cs.LG)',
        ["NVDA", "AMD", "INTC"]),
    "Low-precision training/inference (FP8/FP4)": (
        '(abs:"FP8" OR abs:"FP4" OR abs:"low-precision") AND (abs:"training" OR abs:"inference") AND cat:cs.LG',
        ["NVDA", "AMD", "INTC"]),
    "Multi-GPU interconnect / collective communication": (
        '(abs:"NVLink" OR abs:"collective communication" OR abs:"all-reduce") AND (cat:cs.DC OR cat:cs.NI)',
        ["NVDA", "AMD"]),
    "LLM inference serving efficiency": (
        'abs:"LLM inference" AND (abs:"serving" OR abs:"throughput" OR abs:"KV cache") AND (cat:cs.DC OR cat:cs.LG)',
        ["NVDA", "AMD", "GOOGL"]),
    "Chiplets / advanced packaging": (
        '(abs:"chiplet" OR abs:"advanced packaging" OR abs:"3D integration") AND (cat:cs.AR OR cat:physics.app-ph)',
        ["AMD", "INTC", "NVDA"]),
    "Transistor process technology (GAA, backside power)": (
        '(abs:"gate-all-around" OR abs:"backside power" OR abs:"nanosheet transistor") AND (cat:physics.app-ph OR cat:cond-mat.mes-hall OR cat:cs.ET)',
        ["INTC"]),
    "End-to-end autonomous driving": (
        'abs:"autonomous driving" AND (abs:"end-to-end" OR abs:"planning") AND (cat:cs.CV OR cat:cs.RO)',
        ["TSLA", "GOOGL"]),
    "Lithium-ion battery cells": (
        '(abs:"lithium-ion" OR abs:"battery cell") AND (abs:"electric vehicle" OR abs:"cathode" OR abs:"energy density")',
        ["TSLA"]),
    "Humanoid robots": (
        'abs:"humanoid" AND (abs:"robot" OR abs:"manipulation" OR abs:"locomotion") AND cat:cs.RO',
        ["TSLA", "GOOGL"]),
    "TPUs / ML hardware at scale": (
        '(abs:"TPU" OR abs:"tensor processing unit") AND (cat:cs.AR OR cat:cs.DC OR cat:cs.LG)',
        ["GOOGL"]),
    "Quantum error correction (superconducting qubits)": (
        'abs:"error correction" AND abs:"superconducting" AND cat:quant-ph',
        ["GOOGL"]),
}


def fetch_topic(client: arxiv.Client, query: str, since: date, per_topic: int) -> list[arxiv.Result]:
    """Most relevant papers for the query, keeping only those published since `since`."""
    search = arxiv.Search(query=query, max_results=100, sort_by=arxiv.SortCriterion.Relevance)
    out = []
    for r in client.results(search):
        if r.published.date() >= since:
            out.append(r)
            if len(out) >= per_topic:
                break
    return out


def paper_text(r: arxiv.Result) -> str:
    authors = ", ".join(a.name for a in r.authors[:12]) + (" et al." if len(r.authors) > 12 else "")
    return f"{r.title}\n\nAuthors: {authors}\n\nAbstract: {' '.join(r.summary.split())}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--per-topic", type=int, default=25)
    p.add_argument("--topics", nargs="*", help="subset of topic names (default: all)")
    a = p.parse_args()
    since = date.fromisoformat(a.since)

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    papers: dict[str, dict] = {}      # arxiv id -> {"result", "tickers", "topics"}
    for name, (query, tickers) in TOPICS.items():
        if a.topics and name not in a.topics:
            continue
        try:
            results = fetch_topic(client, query, since, a.per_topic)
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
            continue
        print(f"{name}: {len(results)} papers")
        for r in results:
            pid = re.sub(r"v\d+$", "", r.get_short_id())    # "2501.01234v2" -> "2501.01234"
            entry = papers.setdefault(pid, {"result": r, "tickers": set(), "topics": set()})
            entry["tickers"].update(tickers)
            entry["topics"].add(name)

    saved = 0
    for pid, e in papers.items():
        r = e["result"]
        saved += save(Doc(
            ticker="RESEARCH", source="arxiv", doc_type="paper", section=pid,
            date=r.published.date().isoformat(), url=r.entry_id, text=paper_text(r),
            extra={"title": " ".join(r.title.split()), "tickers": sorted(e["tickers"]),
                   "topics": sorted(e["topics"]), "authors": [x.name for x in r.authors],
                   "categories": r.categories},
        ), min_chars=100)
    print(f"\n{len(papers)} unique papers, {saved} new saved. Now run: python -m ingest.build_index")


if __name__ == "__main__":
    main()
