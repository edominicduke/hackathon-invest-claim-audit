"""End-to-end check of the evidence layer, stage by stage.

    python -m ingest.check_pipeline            # everything (makes a few Vertex calls: ~cents)
    python -m ingest.check_pipeline --offline  # skip anything that calls Google
    python -m ingest.check_pipeline --no-agent # skip the full agent runs (the slowest part)

Stages:
  1. data       every ticker has 10-K, 10-Q, 8-K press releases, transcripts, XBRL; papers exist
  2. index      chunks, ids and vectors line up; no chunk too long for the embedder
  3. embeddings Vertex auth works and a query embeds
  4. search     filters really filter; bad filters raise; competitors exclude self + papers
  5. financials quarterly data exists; Q1+Q2+Q3 matches the 9-month total; no impossible values
  6. grounding  a sentence copied from a chunk is supported; an invented one is not
  7. agent      3 real questions: tools get called, answers cite, no invented labels

Stops at the first FAIL in stages 1-3 (later stages depend on them).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

RESULTS: list[tuple[str, str, str]] = []


def report(status: str, stage: str, msg: str) -> None:
    RESULTS.append((status, stage, msg))
    print(f"  [{status}] {msg}")


def stage(name: str) -> None:
    print(f"\n== {name}")


def _tickers(manifest) -> list[str]:
    return sorted({r["ticker"] for r in manifest if r["ticker"] != "RESEARCH"})


# ---------- 1. data ----------

def check_data() -> list[str]:
    stage("1. data")
    from ingest.common import DATA_DIR, MANIFEST
    if not MANIFEST.exists():
        report("FAIL", "data", f"{MANIFEST} missing: run the fetchers first")
        return []
    manifest = [json.loads(l) for l in MANIFEST.open()]
    tickers = _tickers(manifest)
    by = defaultdict(Counter)
    for r in manifest:
        by[r["ticker"]][r["doc_type"]] += 1
    for t in tickers:
        c = by[t]
        missing = [d for d in ("10-K", "10-Q", "8-K_press_release", "earnings_call") if not c[d]]
        summary = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        report("WARN" if missing else "PASS", "data", f"{t}: {summary}" + (f"  MISSING {missing}" if missing else ""))
        if not (DATA_DIR / t / "xbrl_facts.json").exists():
            report("WARN", "data", f"{t}: no xbrl_facts.json (get_financials won't work for it)")
    papers = by["RESEARCH"]["paper"]
    report("PASS" if papers else "WARN", "data", f"{papers} research papers" + ("" if papers else ": run ingest.arxiv_fetch"))
    return tickers


# ---------- 2. index ----------

def check_index() -> bool:
    stage("2. index")
    import numpy as np
    from ingest.chunk import CHUNKS_PATH, n_tokens
    from ingest.common import MANIFEST
    from ingest.embed import DIM, INDEX_DIR
    if not CHUNKS_PATH.exists() or not (INDEX_DIR / "embeddings.npy").exists():
        report("FAIL", "index", "index missing: run python -m ingest.build_index")
        return False
    chunks = [json.loads(l) for l in CHUNKS_PATH.open()]
    ids = json.loads((INDEX_DIR / "ids.json").read_text())
    vecs = np.load(INDEX_DIR / "embeddings.npy")
    ok = True
    if vecs.shape != (len(ids), DIM):
        report("FAIL", "index", f"embeddings shape {vecs.shape} != ({len(ids)}, {DIM})"); ok = False
    missing = {c["chunk_id"] for c in chunks} - set(ids)
    if missing:
        report("FAIL", "index", f"{len(missing)} chunks not embedded: run python -m ingest.build_index"); ok = False
    docs_in_manifest = {json.loads(l)["doc_id"] for l in MANIFEST.open()}
    docs_in_chunks = {c["doc_id"] for c in chunks}
    if docs_in_manifest - docs_in_chunks:
        report("WARN", "index", f"{len(docs_in_manifest - docs_in_chunks)} documents not chunked: re-run build_index")
    longest = max(n_tokens(c["index_text"]) for c in chunks)
    norms = np.linalg.norm(vecs, axis=1)
    report("PASS" if longest < 2000 else "FAIL", "index", f"{len(chunks)} chunks, longest ~{longest} tokens (limit 2048)")
    report("PASS" if np.allclose(norms, 1, atol=1e-3) else "WARN", "index", "vectors are normalized")
    return ok


# ---------- 3. embeddings ----------

def check_embeddings() -> bool:
    stage("3. embeddings (Vertex)")
    try:
        from ingest.embed import embed_query
        v = embed_query("revenue growth")
        report("PASS", "embed", f"query embedded, dim {len(v)}")
        return True
    except Exception as e:
        report("FAIL", "embed", f"{type(e).__name__}: {e}  (check gcloud auth / env vars)")
        return False


# ---------- 4. search ----------

def check_search(tickers: list[str]) -> None:
    stage("4. search")
    from ingest import search as S
    for t in tickers:
        hits = S.search("risks to the business", tickers=[t], k=10)
        wrong = [h["citation"] for h in hits if t not in [h["ticker"], *h.get("tags", [])]]
        report("FAIL" if wrong or not hits else "PASS", "search",
               f"tickers=[{t}]: {len(hits)} hits" + (f", WRONG COMPANY: {wrong[:2]}" if wrong else ""))
    for dt in ("10-K", "earnings_call", "paper"):
        hits = S.search("demand outlook", doc_types=[dt], k=5)
        wrong = [h["doc_type"] for h in hits if h["doc_type"] != dt]
        report("FAIL" if wrong else ("PASS" if hits else "WARN"), "search", f"doc_types=[{dt}]: {len(hits)} hits")
    try:
        S.search("x", doc_types=["transcript"])
        report("FAIL", "search", "bad doc_type did not raise")
    except ValueError:
        report("PASS", "search", "bad doc_type raises a helpful error")
    for t, peers in S.PEERS.items():
        if t in tickers and peers:
            hits = S.search_competitors(t, "data center revenue", k=8)
            bad = [h["citation"] for h in hits if h["ticker"] == t or h["doc_type"] == "paper"]
            report("FAIL" if bad else "PASS", "search", f"competitors of {t}: {sorted({h['ticker'] for h in hits})}")
            break
    sims = [h["similarity"] for h in S.search("What did management say about AI demand?", k=5)]
    report("PASS" if sims and sims[0] > 0.5 else "WARN", "search", f"top similarities for a real question: {sims}")


# ---------- 5. financials ----------

def check_financials(tickers: list[str]) -> None:
    stage("5. financials (XBRL)")
    from ingest.common import DATA_DIR
    from ingest.financials import _period_kind, clean_facts, get_financials
    for t in tickers:
        try:
            q = get_financials(t, ["revenue"], "quarterly", last=8)
        except ValueError as e:
            report("WARN", "fin", f"{t}: {e}")
            continue
        if len(q) < 4:
            report("WARN", "fin", f"{t}: only {len(q)} quarters of revenue")
        neg = [r["period_label"] for r in q if r["value"] <= 0]
        if neg:
            report("FAIL", "fin", f"{t}: non-positive revenue in {neg}")
        # consistency: reported Q1+Q2+Q3 should equal the reported 9-month total
        raw = json.loads((DATA_DIR / t / "xbrl_facts.json").read_text())
        table = clean_facts(t)
        rev_tags = {r["tag"] for r in table if r["metric"] == "revenue"}
        nines = sorted((r for r in raw if r["tag"] in rev_tags and r.get("period_start")
                        and _period_kind(r["period_start"], r["period_end"]) == "ytd9"),
                       key=lambda r: (r["period_end"], r["filed"]), reverse=True)
        checked = False
        for nine in nines:
            qs = [r for r in table if r["metric"] == "revenue" and r["kind"] == "quarter" and not r["derived"]
                  and nine["period_start"] <= r["period_start"] and r["period_end"] <= nine["period_end"]]
            if len(qs) == 3:
                total = sum(r["value"] for r in qs)
                diff = abs(total - nine["value"]) / nine["value"]
                report("PASS" if diff < 0.01 else "FAIL", "fin",
                       f"{t}: Q1+Q2+Q3 revenue vs 9-month total ending {nine['period_end']}: off by {diff:.2%}")
                checked = True
                break
        if not checked:
            report("WARN", "fin", f"{t}: couldn't find 3 quarters + a 9-month total to cross-check")
        latest = q[-1] if q else None
        if latest:
            report("PASS", "fin", f"{t}: latest revenue {latest['value_text']} ({latest['period_label']})"
                   + (" [derived]" if latest["derived"] else ""))


# ---------- 6. grounding ----------

def check_grounding() -> None:
    stage("6. grounding check (Vertex)")
    from ingest import search as S
    from ingest.verify import verify_citations
    hits = S.search("risk factors competition", doc_types=["10-K"], k=3)
    if not hits:
        report("WARN", "ground", "no 10-K chunks to test with"); return
    h = hits[0]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", h["text"]) if 60 < len(s.strip()) < 400 and not s.strip().startswith("|")]
    if not sents:
        report("WARN", "ground", "couldn't find a clean sentence to test with"); return
    sources = {"S1": h}
    true_claim = sents[0].rstrip(".") + " [S1]."
    false_claim = f"{h['ticker']} announced it will stop selling all of its products next year [S1]."
    r = verify_citations(true_claim + "\n" + false_claim, sources, check_support=True)
    s_true, s_false = r["sentences"][0], r["sentences"][-1]
    err = s_true.get("support_error") or s_false.get("support_error")
    if err:
        report("FAIL", "ground", f"grounding API error: {err}  (gcloud services enable discoveryengine.googleapis.com)")
        return
    report("PASS" if s_true["support"] >= 0.6 else "FAIL", "ground", f"sentence copied from the source: support {s_true['support']}")
    report("PASS" if s_false["support"] < 0.3 else "FAIL", "ground", f"invented sentence: support {s_false['support']}")


# ---------- 7. agent ----------

QUESTIONS = [
    "What are the biggest risks to {t}'s business right now?",
    "How has {t}'s quarterly revenue changed over the last year, and what explains it?",
    "What are {t}'s competitors doing that could hurt it?",
]


def check_agent(tickers: list[str]) -> None:
    stage("7. agent end to end (Gemini + tools)")
    from ingest.tools import answer_with_tools
    t = "INTC" if "INTC" in tickers else tickers[0]
    for q in (x.format(t=t) for x in QUESTIONS):
        try:
            r = answer_with_tools(q, check_support=True)
        except Exception as e:
            report("FAIL", "agent", f"{q[:50]}...: {type(e).__name__}: {e}"); continue
        v = r["verification"]
        tools_used = Counter(c["tool"] for c in r["tool_calls"])
        errors = [c["error"] for c in r["tool_calls"] if c["error"]]
        cited = sum(1 for s in v["sentences"] if s["cited"])
        status = "PASS"
        if not r["tool_calls"] or cited == 0 or v["invalid_labels"]:
            status = "FAIL"
        elif not v["ok"]:
            status = "WARN"
        report(status, "agent", f"{q}\n         tools {dict(tools_used)}, {cited} cited sentences, "
               f"counts {v['counts']}" + (f", invalid {v['invalid_labels']}" if v["invalid_labels"] else "")
               + (f", tool errors {errors}" if errors else ""))
        print("         --- answer ---")
        print("         " + r["answer"].replace("\n", "\n         ")[:1500])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true", help="skip stages that call Google")
    p.add_argument("--no-agent", action="store_true", help="skip the agent runs")
    a = p.parse_args()

    tickers = check_data()
    if not tickers or not check_index():
        return summary()
    if a.offline:
        check_financials(tickers)
        return summary()
    if not check_embeddings():
        return summary()
    check_search(tickers)
    check_financials(tickers)
    check_grounding()
    if not a.no_agent:
        check_agent(tickers)
    summary()


def summary() -> None:
    c = Counter(s for s, _, _ in RESULTS)
    print(f"\n== summary: {c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail")
    for s, st, m in RESULTS:
        if s == "FAIL":
            print(f"  FAIL [{st}] {m.splitlines()[0]}")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()