"""Retrieval eval: does the right chunk come back, and how high?

Recommended workflow (no answer phrases to hunt for):

  1) Generate questions from random chunks. Gemini writes one realistic,
     paraphrased question per chunk; that chunk's ID is the answer key.
       python -m ingest.eval_retrieval --generate 40
     Skim data/eval/questions.jsonl afterwards and delete any bad questions.

  2) Add 5-10 NEGATIVE questions by hand (nothing in the index answers them),
     used to calibrate the similarity cutoff:
       {"question": "What is Tesla's plan for its own smartphone?", "tickers": ["TSLA"], "negative": true}

  3) Run:
       python -m ingest.eval_retrieval --judge -v

What "--judge" fixes: with chunk-ID keys, a question can have several right
answers (the same risk factor appears in the 10-K and in a 10-Q). Strict scoring
counts those as misses. With --judge, when the key chunk isn't in the top 5,
Gemini checks whether the chunks that DID come back answer the question.
You get both numbers: strict (lower bound) and judged (closer to reality).

Output: recall@1/5/10 and MRR for hybrid vs embeddings-only vs keyword-only,
plus a suggested min_similarity cutoff.

Manual questions still work too:
  {"question": "...", "tickers": ["INTC"], "expect_chunk_ids": ["INTC__10-K__...#004"]}
  {"question": "...", "tickers": ["INTC"], "expect_phrase": "commit to Intel 18A"}
Helpers:  --sample N  (print random chunks)   --find "text" --ticker X  (grep chunks)

Uses gemini-3.8-flash on Vertex (override with EVAL_MODEL=...). Cost for
40 questions + judging: a few cents.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from ingest import search as S
from ingest.chunk import CHUNKS_PATH

EVAL_PATH = Path("data/eval/questions.jsonl")
MODES = ("hybrid", "dense", "keyword")
KS = (1, 5, 10)


# ---------- helpers ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS_PATH.open()]


def _tickers_of(c: dict) -> list[str]:
    return [t for t in c.get("tags", [c["ticker"]]) if t != "RESEARCH"]


def pick_chunks(n: int, seed: int | None, exclude: set[str] = frozenset()) -> list[dict]:
    """Random chunks spread evenly across (company, doc type), skipping tiny fragments."""
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for c in _chunks():
        if len(c["text"]) > 400 and c["chunk_id"] not in exclude:
            groups.setdefault((c["ticker"], c["doc_type"]), []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    keys = list(groups)
    rng.shuffle(keys)
    out, i = [], 0
    while len(out) < n and any(groups.values()):
        g = groups[keys[i % len(keys)]]
        if g:
            out.append(g.pop())
        i += 1
    return out


from ingest.llm import MODEL, ask_json  # noqa: E402  (shared helper)


# ---------- 1) question generation ----------

GEN_PROMPT = """You are helping build a search evaluation for an investment research tool.

Source: {citation}
Passage:
\"\"\"
{text}
\"\"\"

Write ONE question that a retail investor researching {who} might type, which this passage answers.
Rules:
- Paraphrase. Do NOT copy distinctive phrases, product names spelled the same way, or exact numbers from the passage.
- Ask about substance: a risk, a trend, a plan, a result, guidance, a research finding.
- The question should make sense on its own, without seeing the passage.
- If the passage is boilerplate (legal disclaimers, forward-looking-statement warnings, table of contents,
  signatures, exhibit lists, an operator's call logistics) or has no substantive fact, skip it.

Return JSON only: {{"question": "..."}} or {{"skip": true}}"""


def generate(n: int, seed: int | None) -> None:
    existing = [json.loads(l) for l in EVAL_PATH.open() if l.strip()] if EVAL_PATH.exists() else []
    used = {cid for q in existing for cid in q.get("expect_chunk_ids", [])}
    picks = pick_chunks(int(n * 1.4) + 2, seed, exclude=used)   # extra, since some get skipped

    def one(c):
        who = ", ".join(_tickers_of(c)) or "this company"
        try:
            r = ask_json(GEN_PROMPT.format(citation=c["citation"], text=c["text"][:6000], who=who), temperature=0.7)
        except Exception as e:
            print(f"  failed on {c['chunk_id']}: {e}")
            return None
        if r.get("skip") or not r.get("question"):
            return None
        return {"question": r["question"].strip(), "tickers": _tickers_of(c),
                "expect_chunk_ids": [c["chunk_id"]], "generated": True, "source": c["citation"]}

    # Each question is saved the moment it's ready, so Ctrl+C keeps everything so far.
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    made, done, lock = [], 0, threading.Lock()
    print(f"Writing up to {n} questions from {len(picks)} passages (Ctrl+C is safe)...")
    with ThreadPoolExecutor(max_workers=12) as pool, EVAL_PATH.open("a") as f:
        futures = [pool.submit(one, c) for c in picks]
        try:
            for fut in as_completed(futures):
                q = fut.result()
                with lock:
                    done += 1
                    if q and len(made) < n:
                        made.append(q)
                        f.write(json.dumps(q) + "\n")
                        f.flush()
                    print(f"\r  {done}/{len(picks)} passages checked, {len(made)} questions saved", end="", flush=True)
                    if len(made) >= n:
                        for other in futures:
                            other.cancel()
                        break
        except KeyboardInterrupt:
            for other in futures:
                other.cancel()
            print("\nStopped early.")
    print(f"\nAdded {len(made)} questions -> {EVAL_PATH}")
    for q in made[:5]:
        print(f"  - {q['question']}\n      from: {q['source']}")
    print("Skim the file and delete weak questions, then add 5-10 negatives by hand.")


# ---------- 2) judging (lenient scoring) ----------

JUDGE_PROMPT = """Question: {question}

For each passage below, decide whether it contains the information needed to answer the question
(fully or for the main part). Be strict: related topic is not enough.

{passages}

Return JSON only: {{"answers": [list of passage numbers that answer the question]}}"""

_judged: dict[tuple[str, str], bool] = {}


def judge(question: str, hits: list[dict]) -> list[bool]:
    todo = [h for h in hits if (question, h["chunk_id"]) not in _judged]
    if todo:
        passages = "\n\n".join(f"[{i}] ({h['citation']})\n{h['text'][:2500]}" for i, h in enumerate(todo, 1))
        try:
            ok = set(ask_json(JUDGE_PROMPT.format(question=question, passages=passages)).get("answers", []))
        except Exception:
            ok = set()
        for i, h in enumerate(todo, 1):
            _judged[(question, h["chunk_id"])] = i in ok
    return [_judged[(question, h["chunk_id"])] for h in hits]


# ---------- 3) evaluation ----------

def is_match(hit: dict, q: dict) -> bool:
    if q.get("expect_chunk_ids"):
        return hit["chunk_id"] in q["expect_chunk_ids"]
    if q.get("expect_ticker") and q["expect_ticker"].upper() not in [hit["ticker"], *hit.get("tags", [])]:
        return False
    if q.get("expect_doc_type") and hit["doc_type"] != q["expect_doc_type"]:
        return False
    return _norm(q["expect_phrase"]) in _norm(hit["text"])


def load_questions() -> list[dict]:
    if not EVAL_PATH.exists():
        raise SystemExit(f"No {EVAL_PATH} yet. Start with: python -m ingest.eval_retrieval --generate 40")
    qs = [json.loads(l) for l in EVAL_PATH.open() if l.strip()]
    keep = [q for q in qs if not str(q.get("expect_phrase", "")).startswith("<")]
    if len(keep) < len(qs):
        print(f"Skipping {len(qs) - len(keep)} template question(s) still containing <REPLACE>.")
    return keep


def _filters(q: dict) -> dict:
    return {k: q[k] for k in ("tickers", "doc_types", "date_from", "date_to") if q.get(k)}


def evaluate(questions: list[dict], use_judge: bool = False, verbose: bool = False) -> None:
    S.embed_query = lru_cache(maxsize=None)(S.embed_query)   # embed each question once, not per mode
    pos = [q for q in questions if not q.get("negative")]
    neg = [q for q in questions if q.get("negative")]
    if not pos:
        raise SystemExit("No positive questions to evaluate.")

    print(f"\n{len(pos)} positive questions, {len(neg)} negative"
          + ("  (judging misses with " + MODEL + ")" if use_judge else ""))
    header = f"{'mode':<8} " + " ".join(f"R@{k:<4}" for k in KS) + "  MRR@10"
    print("\n" + header + ("   R@5 judged" if use_judge else ""))

    misses: dict[str, list] = {}
    for mode in MODES:
        ranks, judged_hits = [], []
        for q in pos:
            hits = S.search(q["question"], k=max(KS), mode=mode, **_filters(q))
            rank = next((i + 1 for i, h in enumerate(hits) if is_match(h, q)), None)
            ranks.append(rank)
            ok5 = rank is not None and rank <= 5
            if use_judge and not ok5:
                ok5 = any(judge(q["question"], hits[:5]))
            judged_hits.append(ok5)
            if not ok5:
                misses.setdefault(mode, []).append((q["question"], rank, [h["citation"] for h in hits[:3]],
                                                    q.get("source")))
        rec = [np.mean([r is not None and r <= k for r in ranks]) for k in KS]
        mrr = np.mean([1 / r if r else 0 for r in ranks])
        line = f"{mode:<8} " + " ".join(f"{x:<6.0%}" for x in rec) + f"  {mrr:.3f}"
        print(line + (f"   {np.mean(judged_hits):.0%}" if use_judge else ""))

    # --- similarity calibration ---
    print("\nSimilarity calibration (cosine, embeddings):")
    pos_sims = []
    for q in pos:
        m = next((h for h in S.search(q["question"], k=50, mode="dense", **_filters(q)) if is_match(h, q)), None)
        if m:
            pos_sims.append(m["similarity"])
    neg_sims = [h[0]["similarity"] for q in neg if (h := S.search(q["question"], k=1, mode="dense", **_filters(q)))]
    if pos_sims:
        print(f"  right chunk, positives:  min {min(pos_sims):.3f}  p10 {np.percentile(pos_sims, 10):.3f}  "
              f"median {np.median(pos_sims):.3f}")
    if neg_sims:
        print(f"  best chunk, negatives:   median {np.median(neg_sims):.3f}  p90 {np.percentile(neg_sims, 90):.3f}  "
              f"max {max(neg_sims):.3f}")
    if pos_sims and neg_sims:
        lo, hi = np.percentile(neg_sims, 90), np.percentile(pos_sims, 10)
        if lo < hi:
            print(f"  suggested min_similarity ~ {(lo + hi) / 2:.2f}")
        else:
            print("  positives and negatives overlap: no clean cutoff. Use a low cutoff only to drop clear junk.")
    elif not neg_sims:
        print("  add some negative questions to get a suggested cutoff")

    if verbose:
        for mode, ms in misses.items():
            print(f"\nMisses (not in top 5{', even after judging' if use_judge else ''}), {mode}:")
            for question, rank, top, src in ms:
                print(f"  - {question}  (key rank {rank or '>10'})")
                if src:
                    print(f"      written from: {src}")
                print(f"      got: {top}")


# ---------- small CLI helpers ----------

def sample(n: int, seed: int | None) -> None:
    for c in pick_chunks(n, seed):
        print("=" * 90)
        print(f"{c['citation']}\nticker(s): {_tickers_of(c)}   doc_type: {c['doc_type']}   id: {c['chunk_id']}\n")
        print(c["text"][:1500] + ("..." if len(c["text"]) > 1500 else ""))
    print("=" * 90)


def find(phrase: str, ticker: str | None) -> None:
    n = 0
    for c in _chunks():
        if ticker and ticker.upper() not in [c["ticker"], *c.get("tags", [])]:
            continue
        txt = _norm(c["text"])
        i = txt.find(_norm(phrase))
        if i >= 0:
            n += 1
            print(f"{c['chunk_id']}\n   {c['citation']}\n   ...{txt[max(0, i - 80):i + 120]}...\n")
    print(f"{n} chunk(s) contain {phrase!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Retrieval eval. See module docstring for the workflow.")
    p.add_argument("--generate", type=int, metavar="N", help="have Gemini write N questions from random chunks")
    p.add_argument("--judge", action="store_true", help="also score misses with a Gemini relevance judge")
    p.add_argument("--sample", type=int, metavar="N", help="print N random chunks")
    p.add_argument("--find", help="list chunks containing this text")
    p.add_argument("--ticker")
    p.add_argument("--seed", type=int)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    if a.generate:
        generate(a.generate, a.seed)
    elif a.sample:
        sample(a.sample, a.seed)
    elif a.find:
        find(a.find, a.ticker)
    else:
        evaluate(load_questions(), a.judge, a.verbose)


if __name__ == "__main__":
    main()