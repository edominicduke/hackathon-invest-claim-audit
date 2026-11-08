"""Measure how often the citation checker agrees with you.

Three steps:

  1) Collect cited sentences from real agent answers (runs ~6 questions, a few cents):
       python -m ingest.label_checker --collect 30
     Sentences are sampled across checker verdicts (supported / partial / unsupported)
     so the test isn't just easy "supported" cases.

  2) Label them yourself. You see the sentence and its cited sources, NOT the
     checker's verdict (seeing it would bias you):
       python -m ingest.label_checker --label
     Keys:  s = supported   p = partly supported   u = unsupported
            m = show full source text   k = skip   b = back one   q = save & quit
     Progress saves after every answer; re-run --label to continue.

  3) Compare:
       python -m ingest.label_checker --report
     Agreement, a confusion table, and how well it catches bad sentences:
       - "When the checker flags a sentence, how often is it really a problem?"
       - "Of the sentences that really are a problem, how many does it catch?"

What counts as supported (use the same rule as the checker):
  every fact in the sentence (names, numbers, dates, products, strong words like
  "dominant") is stated or directly paraphrased in the cited sources.
  Some facts present, some missing = partly. Nothing (or the key claim) missing = unsupported.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from collections import Counter
from pathlib import Path

from .common import DATA_ROOT

LABELS_PATH = DATA_ROOT / "eval" / "checker_labels.jsonl"

DEFAULT_QUESTIONS = [
    "What are the biggest risks to {t}'s business right now?",
    "What are {t}'s competitors doing that could hurt it?",
    "How has {t}'s revenue changed over the last year, and what explains it?",
    "What guidance has {t}'s management given about the next quarters?",
    "What is {t} betting on for future growth, and what could go wrong?",
]

# terminal colors (fall back to plain text if the terminal doesn't support them)
BOLD, DIM, YEL, CYAN, RESET = "\033[1m", "\033[2m", "\033[33m", "\033[36m", "\033[0m"


def _load() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    return [json.loads(l) for l in LABELS_PATH.open() if l.strip()]


def _save(items: list[dict]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_PATH.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    tmp.replace(LABELS_PATH)


def _bucket(status: str) -> str:
    return {"supported": "supported", "partial": "partial", "weak": "partial"}.get(status, "unsupported")


# ---------- 1) collect ----------

def collect(n: int, tickers: list[str], seed: int | None) -> None:
    from .tools import answer_with_tools
    from .verify import clean_claim

    items = _load()
    have = {i["sentence"] for i in items}
    pool: list[dict] = []
    questions = [q.format(t=t) for t in tickers for q in DEFAULT_QUESTIONS]
    rng = random.Random(seed)
    rng.shuffle(questions)

    for q in questions:
        print(f"Running: {q}")
        try:
            r = answer_with_tools(q, check_support=True)
        except Exception as e:
            print(f"   failed: {type(e).__name__}: {e}")
            continue
        for s in r["verification"]["sentences"]:
            if not s["cited"] or s["status"] in ("invalid_citation",) or s["support"] is None:
                continue
            sent = clean_claim(s["sentence"])
            if sent in have or len(sent.split()) < 5:
                continue
            have.add(sent)
            pool.append({
                "question": q, "sentence": sent, "cited": s["cited"],
                "sources": {l: {k: r["sources"][l].get(k) for k in ("citation", "text", "url")}
                            for l in s["cited"] if l in r["sources"]},
                "checker": {"status": s["status"], "bucket": _bucket(s["status"]), "support": s["support"],
                            "method": s.get("method"), "unsupported_parts": s.get("unsupported_parts", [])},
                "human": None, "note": "",
            })
        counts = Counter(p["checker"]["bucket"] for p in pool)
        print(f"   pool so far: {dict(counts)}")
        # stop once we can fill a balanced sample
        if all(counts[b] >= n // 3 for b in ("supported", "partial", "unsupported")) or len(pool) >= n * 3:
            break

    # balanced sample: up to n/3 from each verdict, fill the rest from whatever is left
    rng.shuffle(pool)
    picked, per = [], n // 3
    for b in ("supported", "partial", "unsupported"):
        picked += [p for p in pool if p["checker"]["bucket"] == b][:per]
    rest = [p for p in pool if p not in picked]
    picked += rest[: n - len(picked)]
    rng.shuffle(picked)
    start = len(items)
    for k, p in enumerate(picked):
        p["id"] = start + k + 1
    _save(items + picked)
    print(f"\nAdded {len(picked)} sentences -> {LABELS_PATH}  "
          f"(checker verdicts: {dict(Counter(p['checker']['bucket'] for p in picked))})")
    if any(Counter(p["checker"]["bucket"] for p in picked)[b] < per for b in ("partial", "unsupported")):
        print("Few flagged sentences turned up (the citation rules work!). That's fine; the report will say so.")
    print("Next: python -m ingest.label_checker --label")


# ---------- 2) label ----------

_STOP = set("the a an and or of to in on for with by as at from that this these those is are was were be been "
            "its it their our has have had which who also more than into over such".split())


def _terms(sentence: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-%$]*", sentence)
    return {w.lower().strip(".") for w in words if (len(w) > 3 or any(ch.isdigit() for ch in w))
            and w.lower() not in _STOP}


def _highlight(text: str, terms: set[str]) -> str:
    if not terms:
        return text
    pat = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b")
    return pat.sub(lambda m: f"{YEL}{BOLD}{m.group(0)}{RESET}", text)


def _best_paragraphs(text: str, terms: set[str], k: int = 3) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:   # one long block: split into ~600-char windows on sentence ends
        sents = re.split(r"(?<=[.!?])\s+", text)
        paras, cur = [], ""
        for s in sents:
            if len(cur) + len(s) > 600 and cur:
                paras.append(cur); cur = ""
            cur = f"{cur} {s}".strip()
        if cur:
            paras.append(cur)
    scored = sorted(paras, key=lambda p: -len(terms & _terms(p)))
    return scored[:k]


def _show(item: dict, full: bool = False) -> None:
    terms = _terms(item["sentence"])
    print("\n" + "=" * 100)
    print(f"{DIM}#{item['id']}  question: {item['question']}{RESET}\n")
    print(f"{BOLD}SENTENCE:{RESET} {item['sentence']}  {CYAN}{' '.join('[' + c + ']' for c in item['cited'])}{RESET}\n")
    for label, src in item["sources"].items():
        print(f"{CYAN}[{label}] {src['citation']}{RESET}")
        chunks = [src["text"]] if full else _best_paragraphs(src["text"], terms)
        for para in chunks:
            body = textwrap.fill(" ".join(para.split()), 100, initial_indent="   ", subsequent_indent="   ")
            print(_highlight(body, terms) + "\n")
        if not full:
            print(f"   {DIM}(most relevant passages shown; press m for the full text){RESET}\n")


def label() -> None:
    items = _load()
    if not items:
        raise SystemExit("Nothing to label. Run: python -m ingest.label_checker --collect 30")
    order = [i for i, it in enumerate(items) if it["human"] is None]
    done = sum(1 for it in items if it["human"])
    print(f"{done} labeled, {len(order)} to go. Keys: s=supported  p=partly  u=unsupported  "
          f"m=more  k=skip  b=back  q=quit")
    history: list[int] = []
    pos = 0
    full = False
    while pos < len(order):
        idx = order[pos]
        _show(items[idx], full)
        ans = input(f"[{done + 1}/{done + len(order) - pos}] s / p / u  (m more, k skip, b back, q quit): ").strip().lower()
        full = False
        if ans in ("s", "p", "u"):
            items[idx]["human"] = {"s": "supported", "p": "partial", "u": "unsupported"}[ans]
            note = input("   optional note (Enter to skip): ").strip()
            items[idx]["note"] = note
            _save(items)
            history.append(pos)
            done += 1
            pos += 1
        elif ans == "m":
            full = True
        elif ans == "k":
            pos += 1
        elif ans == "b" and history:
            pos = history.pop()
            items[order[pos]]["human"] = None
            done -= 1
            _save(items)
        elif ans == "q":
            break
    _save(items)
    left = sum(1 for it in items if it["human"] is None)
    print(f"\nSaved. {sum(1 for it in items if it['human'])} labeled, {left} left."
          + ("" if left else "  Next: python -m ingest.label_checker --report"))


# ---------- 3) report ----------

def report() -> None:
    items = [i for i in _load() if i["human"]]
    if not items:
        raise SystemExit("No labels yet. Run --label first.")
    cats = ("supported", "partial", "unsupported")
    n = len(items)
    agree = sum(i["human"] == i["checker"]["bucket"] for i in items)

    print(f"\n{n} sentences labeled\n")
    print(f"Exact agreement (supported / partly / unsupported): {agree}/{n} = {agree / n:.0%}")

    # Cohen's kappa: agreement corrected for chance
    ph = Counter(i["human"] for i in items)
    pc = Counter(i["checker"]["bucket"] for i in items)
    pe = sum(ph[c] * pc[c] for c in cats) / (n * n)
    kappa = (agree / n - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"Cohen's kappa (chance-corrected): {kappa:.2f}   (0.6+ is usually called substantial)")

    # binary: is the sentence a problem? (partly or unsupported)
    tp = sum(i["human"] != "supported" and i["checker"]["bucket"] != "supported" for i in items)
    fp = sum(i["human"] == "supported" and i["checker"]["bucket"] != "supported" for i in items)
    fn = sum(i["human"] != "supported" and i["checker"]["bucket"] == "supported" for i in items)
    tn = n - tp - fp - fn
    bin_acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"\nProblem / no problem agreement: {bin_acc:.0%}")
    print(f"  When the checker flags a sentence, it's really a problem: {prec:.0%}  ({tp}/{tp + fp})")
    print(f"  Of the real problems, the checker catches:               {rec:.0%}  ({tp}/{tp + fn})")
    print(f"  Missed problems (checker said supported, you didn't):    {fn}")
    print(f"  False alarms (checker flagged, you said supported):      {fp}")

    print("\nConfusion table (rows = you, columns = checker):")
    print(f"{'':>14}" + "".join(f"{c:>13}" for c in cats))
    for h in cats:
        print(f"{h:>14}" + "".join(f"{sum(i['human'] == h and i['checker']['bucket'] == c for i in items):>13}" for c in cats))

    by_method = Counter((i["checker"].get("method") or "grounding", i["human"] == i["checker"]["bucket"]) for i in items)
    print("\nAgreement by how the checker decided:")
    for m in ("grounding", "judge"):
        tot = by_method[(m, True)] + by_method[(m, False)]
        if tot:
            print(f"  {m:<10} {by_method[(m, True)]}/{tot} = {by_method[(m, True)] / tot:.0%}"
                  + ("   (grounding API score >= 0.6)" if m == "grounding" else "   (Gemini fact-by-fact check)"))

    mism = [i for i in items if i["human"] != i["checker"]["bucket"]]
    if mism:
        print(f"\nDisagreements ({len(mism)}):")
        for i in mism:
            print(f"  #{i['id']} you: {i['human']:<11} checker: {i['checker']['status']:<11} "
                  f"(support {i['checker']['support']}, {i['checker'].get('method') or 'grounding'})")
            print(f"      {i['sentence'][:160]}")
            if i["checker"]["unsupported_parts"]:
                print(f"      checker said missing: {i['checker']['unsupported_parts']}")
            if i["note"]:
                print(f"      your note: {i['note']}")
    if n < 20:
        print(f"\nNote: {n} labels is a small sample; each disagreement moves the numbers a lot.")


def main() -> None:
    p = argparse.ArgumentParser(description="Measure the citation checker against your own labels.")
    p.add_argument("--collect", type=int, metavar="N", help="collect N cited sentences from real agent answers")
    p.add_argument("--tickers", nargs="*", default=["INTC", "NVDA", "TSLA"])
    p.add_argument("--label", action="store_true", help="label collected sentences")
    p.add_argument("--report", action="store_true", help="compare your labels with the checker")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    if a.collect:
        collect(a.collect, a.tickers, a.seed)
    elif a.label:
        label()
    elif a.report:
        report()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
