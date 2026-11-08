"""Feature #3: check that every claim in a generated answer cites a source that
was actually retrieved, and (optionally) that the source supports it.

    from ingest.search import search, format_for_prompt
    from ingest.verify import verify_citations

    block, sources = format_for_prompt(search("...", tickers=["INTC"]))
    answer = <model output written with [S#] labels>
    report = verify_citations(answer, sources, check_support=True)

    report["ok"]              # False if anything is invented, uncited, or unsupported
    report["sentences"]       # per-sentence status, cited labels, support score
    report["used_sources"]    # {"S1": hit, ...} only the sources the answer cites (for the UI)
    report["clean_answer"]    # answer with [S#] labels removed (labels stay in report)

Checks, in order:
  1. invented labels   [S9] when only S1..S5 were retrieved      -> "invalid_citation"
  2. uncited facts     sentence with a number and no label         -> "uncited_number"
                       longer declarative sentence with no label   -> "uncited_claim" (warning)
  3. support           (check_support=True) Vertex check grounding
                       API scores each cited sentence against ONLY
                       the chunks it cites. >= 0.6                  -> "supported"
  4. second opinion    (judge=True, default) sentences under 0.6 go
                       to Gemini, which splits them into individual
                       facts and checks each against the cited
                       sources (the grounding API is harsh on
                       sentences combining several facts)            -> "supported" / "partial" / "unsupported"
                       "partial" lists the parts with no source in s["unsupported_parts"]
  table rows           lines starting with "|": every $ amount and % must appear in the
                       row's cited sources (rounding allowed). Deterministic, no API call.

Number-vs-SEC-data checking is the Checking Results teammate's job; this
module only checks citations.

Support check setup (once per project):
  gcloud services enable discoveryengine.googleapis.com
  pip install google-cloud-discoveryengine
Cost: about $0.00075 per 1,000 answer characters, i.e. negligible.
"""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

SUPPORTED = 0.6     # support score at/above this = supported
WEAK = 0.3          # between WEAK and SUPPORTED = weakly supported; below WEAK = unsupported
UNCITED_CLAIM_MIN_WORDS = 10

# [S1]  [S1][S3]  [S1, S3]  [S1,S3]  [S 1]  [s1]  [S1; S3]  [S1, 3]
_TAG = re.compile(r"\[\s*[Ss]\s*\d+(?:\s*[,;]\s*(?:[Ss]\s*)?\d+)*\s*\]")
_TAG_NUM = re.compile(r"\d+")
# tags placed after the period: "grew 22%. [S1]" -> move before the period
_TAG_AFTER_PUNCT = re.compile(r"([.!?])((?:\s*" + _TAG.pattern + r")+)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*\[])")
_HAS_NUMBER = re.compile(r"\d")
# A QUANTITY (not a product name like "Intel 18A"/"MI300", a year, "Q4 2026" or a date):
# money, percentages, numbers with units, decimals, or numbers with thousands separators.
_QUANTITY = re.compile(
    r"\$\s?\d"
    r"|\b\d+(?:\.\d+)?\s?(?:%|(?:percent|billion|million|thousand|trillion|bps|basis points)\b)"
    r"|(?<![\w.-])\d+\.\d+(?![\w-])"
    r"|\b\d{1,3}(?:,\d{3})+\b", re.I)
_META = re.compile(r"^\W*(whether|evaluating|to evaluate|to answer|based on|in summary|here is|here's|below)\b", re.I)
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


_MD_EMPH = re.compile(r"(\*\*|__|\*|`)")
_LEAD_LABEL = re.compile(r"^([^.:!?]{2,80}):\s+(?=\S)")      # "Massive Asset Impairments: Intel holds..."


def clean_claim(s: str) -> str:
    """What the grounding API should judge: the claim itself, without markdown
    emphasis, bullet markers or a bold lead-in label the model added."""
    s = _LIST_MARK.sub("", s.strip())
    s = _MD_EMPH.sub("", s).strip()
    m = _LEAD_LABEL.match(s)
    if m and len(m.group(1).split()) <= 8:
        s = s[m.end():]
    return re.sub(r"\s+", " ", s).strip()


def _labels(tag: str) -> list[str]:
    return [f"S{n}" for n in _TAG_NUM.findall(tag)]


def split_sentences(answer: str) -> list[str]:
    """Lines first (bullets, headings), then sentences within each line."""
    answer = _TAG_AFTER_PUNCT.sub(lambda m: f"{m.group(2).strip()}{m.group(1)}", answer)
    out = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("|"):   # headings and table rows stay whole
            out.append(line)
            continue
        # keep a list marker ("1.", "-", "*") attached to its first sentence, so "1." is never a sentence
        m = _LIST_MARK.match(line)
        prefix, rest = (line[:m.end()], line[m.end():]) if m else ("", line)
        parts = [s.strip() for s in _SENT_SPLIT.split(rest) if s.strip()]
        if parts:
            parts[0] = prefix + parts[0]
            out.extend(parts)
        elif prefix.strip():
            out.append(prefix.strip())
    return out


_INTERPRETATION = re.compile(
    r"^\W*(this|that|these|those|overall|taken together|in short|in other words|together)\b"
    r"(?:\W+\w+){0,10}?\W+(suggests?|indicates?|implies|imply|may|might|could|appears?|seems?)\b", re.I)


def _needs_citation(sentence_text: str) -> str | None:
    """Returns the status to assign if this UNCITED sentence looks like a claim."""
    s = _MD_EMPH.sub("", sentence_text).strip()
    if s.startswith("#") or s.endswith("?") or s.endswith(":") or set(s) <= set("-_=* "):
        return None                                   # heading, question, lead-in, divider
    body = _LIST_MARK.sub("", s)
    if _QUANTITY.search(body):
        return "uncited_number"
    if _INTERPRETATION.match(body):
        return "interpretation"                       # framed analysis, uncited by design (not a warning)
    if _META.match(body):
        return None                                   # framing sentence ("Whether you should buy... depends on...")
    if len(body.split()) >= UNCITED_CLAIM_MIN_WORDS:
        return "uncited_claim"
    return None


# ---------- table rows: deterministic number check ----------

_MONEY = re.compile(r"(-?)\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|bn|b|m|k)?\b", re.I)
_PCT = re.compile(r"([+\-−]?\d+(?:\.\d+)?)\s?%")
_SCALE = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_TABLE_SEP = re.compile(r"^\|?[\s:\-|]+\|?$")


def _money_values(text: str) -> list[float]:
    out = []
    for sign, num, unit in _MONEY.findall(text):
        v = float(num.replace(",", "")) * _SCALE.get((unit or "").lower(), 1)
        out.append(-v if sign else v)
    return out


def _pct_values(text: str) -> list[float]:
    return [float(p.replace("−", "-")) for p in _PCT.findall(text)]


def _check_table_row(row: str, cited: list[str], sources: dict[str, dict]) -> tuple[str, list[str]]:
    """Every $ amount and % in a table row must appear in its cited sources (rounding allowed).
    Percentages may also come from a computed guidance record in the conversation."""
    body = _TAG.sub("", row)
    ref = " ".join(sources[l]["text"] for l in cited if l in sources)
    ref_pct = ref + " " + " ".join(h["text"] for h in sources.values() if h.get("doc_type") == "guidance_record")
    have_money, have_pct = _money_values(ref), _pct_values(ref_pct)
    missing = []
    for v in _money_values(body):
        if not any(abs(v - h) <= 0.006 * max(abs(v), abs(h), 1) for h in have_money):
            missing.append(f"${v / 1e9:,.2f} billion" if abs(v) >= 1e9 else f"${v:,.0f}")
    for p in _pct_values(body):
        if not any(abs(p - h) <= 0.051 for h in have_pct):
            missing.append(f"{p:+.1f}%")
    checked = len(_money_values(body)) + len(_pct_values(body))

    # Result column: if the row has guidance (range or point) + actual, the verdict word must match the numbers
    vals = _money_values(body)
    word = re.search(r"\b(beat|met|within|in range|missed|miss)\b", body, re.I)
    if word and len(vals) in (2, 3):
        low, high, actual = (vals[0], vals[1], vals[2]) if len(vals) == 3 else (vals[0], vals[0], vals[1])
        expected = "beat" if actual > high else "missed" if actual < low else "met"
        said = {"within": "met", "in range": "met", "miss": "missed"}.get(word.group(1).lower(), word.group(1).lower())
        if said != expected:
            return "unsupported", missing + [f"result says '{word.group(1)}' but the numbers show '{expected}'"]

    if not missing:
        return "supported", []
    return ("partial" if len(missing) < checked else "unsupported"), missing


# ---------- support check (Vertex AI check grounding API) ----------

_gclient = None
_glock = threading.Lock()


def _grounding_client():
    global _gclient
    with _glock:
        if _gclient is None:
            from google.cloud import discoveryengine_v1 as de
            _gclient = de.GroundedGenerationServiceClient()
        return _gclient


def support_detail(claim: str, fact_texts: list[tuple[str, str]],
                   citation_threshold: float = SUPPORTED) -> tuple[float, list[int]]:
    """(score 0..1, indexes of the facts the API says support the claim).
    fact_texts: [(citation_label, chunk_text), ...]"""
    from google.cloud import discoveryengine_v1 as de
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = _grounding_client()
    req = de.CheckGroundingRequest(
        grounding_config=client.grounding_config_path(project, "global", "default_grounding_config"),
        answer_candidate=claim[:1024],                 # API limit: 1,024 characters
        facts=[de.GroundingFact(fact_text=text, attributes={"title": label}) for label, text in fact_texts],
        grounding_spec=de.CheckGroundingSpec(citation_threshold=citation_threshold),
    )
    resp = client.check_grounding(request=req)
    supporting = set()
    for claim_result in resp.claims:
        for ci in claim_result.citation_indices:
            if ci < len(resp.cited_chunks):
                src = resp.cited_chunks[ci].source
                if str(src).isdigit():
                    supporting.add(int(src))
    return float(resp.support_score), sorted(supporting)


def support_score(claim: str, fact_texts: list[tuple[str, str]]) -> float:
    return support_detail(claim, fact_texts)[0]


# ---------- second opinion: fact-by-fact check with Gemini ----------

JUDGE_PROMPT = """You are checking whether a sentence in an investment-research answer is supported by its cited sources.

Sentence: {claim}

Cited sources:
{sources}

Split the sentence into its individual factual claims (each named product, number, date, customer,
action or characterization is its own claim). For each one, decide whether the cited sources state it
or directly paraphrase it. Rules:
- Same meaning in different words counts as supported. Numbers, names and dates must match.
- Every item in a list ("A, B and C") must appear in a source to be supported.
- Strong characterizations ("massive", "dominant", "relentless") are unsupported unless a source says so.
- Use only the sources above, not your own knowledge.

Return JSON only:
{{"claims": [{{"claim": "...", "supported": true/false, "source": "S#" or null,
               "evidence": "short quote (max 20 words) from the source, or null"}}]}}"""


def judge_sentence(claim: str, cited: list[str], sources: dict[str, dict]) -> dict:
    from .llm import ask_json
    block = "\n\n".join(f"[{l}] {sources[l]['citation']}\n{sources[l]['text'][:6000]}"
                        for l in cited if l in sources)
    out = ask_json(JUDGE_PROMPT.format(claim=claim, sources=block))
    parts = [c for c in out.get("claims", []) if isinstance(c, dict) and c.get("claim")]
    supported = [c for c in parts if c.get("supported")]
    if not parts:
        verdict = "unsupported"
    elif len(supported) == len(parts):
        verdict = "supported"
    elif supported:
        verdict = "partial"
    else:
        verdict = "unsupported"
    return {"verdict": verdict, "claims": parts,
            "unsupported_parts": [c["claim"] for c in parts if not c.get("supported")]}


def diagnose(report: dict, all_sources: dict[str, dict], workers: int = 6) -> list[dict]:
    """For each partial/unsupported sentence, re-check its missing parts against
    EVERY source retrieved in the conversation (registry.sources):
      missing_citation - the missing parts ARE in other retrieved sources (says which):
                         the model should have cited them
      not_in_sources   - still missing: the model added it from its own knowledge
      mixed            - some parts found elsewhere, some not
    """
    labels = list(all_sources)
    todo = [s for s in report["sentences"] if s["status"] in ("partial", "unsupported", "weak")]

    def run(s):
        claim = clean_claim(s["sentence"])
        missing_before = s.get("unsupported_parts") or [claim]
        try:
            j = judge_sentence(claim, labels, {l: {**all_sources[l], "text": all_sources[l]["text"][:3000]}
                                                for l in labels})
        except Exception as e:
            return {"sentence": claim, "cited": s["cited"], "diagnosis": "error", "error": str(e)}
        still_missing = j["unsupported_parts"]
        found_in = sorted({c["source"] for c in j["claims"]
                           if c.get("supported") and c.get("source") and c["source"] not in s["cited"]})
        if not still_missing and not found_in:
            why = "recheck_supported"     # re-check found it supported by its own sources: first verdict was noise
        elif not still_missing:
            why = "missing_citation"
        elif found_in and len(still_missing) < len(missing_before):
            why = "mixed"
        else:
            why = "not_in_sources"
        return {"sentence": claim, "cited": s["cited"], "status": s["status"],
                "missing_before": missing_before, "still_missing": still_missing,
                "found_in": found_in, "diagnosis": why}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, todo))


# ---------- main entry point ----------

def verify_citations(answer: str, sources: dict[str, dict], check_support: bool = False,
                     judge: bool = True, workers: int = 6) -> dict:
    """check_support: score each cited sentence with the grounding API.
    judge: for sentences the grounding API scores below SUPPORTED, get a fact-by-fact
    second opinion from Gemini (the grounding API is harsh on sentences that combine
    several facts or several sources). Final statuses: supported / partial / unsupported.
    """
    sentences = []
    for text in split_sentences(answer):
        tags = _TAG.findall(text)
        cited = list(dict.fromkeys(l for t in tags for l in _labels(t)))   # unique, in order
        claim = _TAG.sub("", text).strip()
        claim = re.sub(r"\s+([.!?,;:])", r"\1", claim)
        invalid = [l for l in cited if l not in sources]
        entry = {"sentence": claim, "cited": cited, "invalid": invalid, "support": None}
        if text.startswith("|"):
            if _TABLE_SEP.match(text) or (not cited and not _money_values(text) and not _pct_values(text)):
                entry["status"] = "no_claim"                      # separator or header row
            elif invalid:
                entry["status"] = "invalid_citation"
            elif not cited:
                entry["status"] = "uncited_number"
            else:
                entry["status"], entry["unsupported_parts"] = _check_table_row(text, cited, sources)
                entry["method"] = "table"
            sentences.append(entry)
            continue
        if invalid:
            status = "invalid_citation"
        elif not cited:
            status = _needs_citation(claim) or "no_claim"
        else:
            status = "cited"
        entry["status"] = status
        sentences.append(entry)

    if check_support:
        todo = [s for s in sentences if s["status"] == "cited"]

        def run(s):
            facts = [(sources[l]["citation"], sources[l]["text"]) for l in s["cited"]]
            try:
                s["support"] = round(support_score(clean_claim(s["sentence"]), facts), 3)
            except Exception as e:             # don't fail the whole answer on one API error
                s["support_error"] = f"{type(e).__name__}: {e}"
                return
            if s["support"] >= SUPPORTED:
                s["status"], s["method"] = "supported", "grounding"
                return
            s["status"] = "weak" if s["support"] >= WEAK else "unsupported"
            if judge:
                try:
                    j = judge_sentence(clean_claim(s["sentence"]), s["cited"], sources)
                except Exception as e:
                    s["judge_error"] = f"{type(e).__name__}: {e}"
                    return
                s["judge"] = j
                s["unsupported_parts"] = j["unsupported_parts"]
                s["status"], s["method"] = j["verdict"], "judge"

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, todo))

    counts: dict[str, int] = {}
    for s in sentences:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    hard_fail = {"invalid_citation", "uncited_number", "unsupported"}   # "partial" and "weak" are warnings
    used = list(dict.fromkeys(l for s in sentences for l in s["cited"] if l in sources))

    return {
        "ok": not any(s["status"] in hard_fail for s in sentences),
        "counts": counts,
        "sentences": sentences,
        "invalid_labels": sorted({l for s in sentences for l in s["invalid"]}),
        "used_sources": {l: sources[l] for l in used},
        "clean_answer": re.sub(r"\s+([.!?,;:])", r"\1", _TAG.sub("", answer)).strip(),
    }


def print_report(report: dict) -> None:
    marks = {"supported": "OK ", "cited": "OK ", "no_claim": "   ", "interpretation": "   ",
             "weak": "WEAK", "partial": "PART",
             "uncited_claim": "WARN", "uncited_number": "FAIL", "invalid_citation": "FAIL",
             "unsupported": "FAIL"}
    for s in report["sentences"]:
        extra = f" support={s['support']}" if s["support"] is not None else ""
        if s.get("method") == "judge":
            extra += " (checked fact by fact)"
        elif s.get("method") == "table":
            extra += " (table row: numbers checked)"
        bad = f" invalid={s['invalid']}" if s["invalid"] else ""
        print(f"[{marks.get(s['status'], '?'):4}] {s['status']:<16}{s['cited']}{bad}{extra}\n        {s['sentence']}")
        for part in s.get("unsupported_parts", []):
            print(f"          not in sources: {part}")
    print(f"\nok={report['ok']}  {report['counts']}")


def print_diagnosis(rows: list[dict]) -> None:
    from collections import Counter
    if not rows:
        print("Nothing weak or unsupported to diagnose.")
        return
    for r in rows:
        if r["diagnosis"] == "error":
            print(f"\n[error] {r['sentence']}\n   {r['error']}")
            continue
        print(f"\n[{r['diagnosis']}] ({r['status']}, cited {r['cited']})"
              + (f" -> also supported by {r['found_in']}" if r["found_in"] else ""))
        print(f"   {r['sentence']}")
        for part in r["still_missing"]:
            print(f"     not in ANY retrieved source: {part}")
    c = Counter(r["diagnosis"] for r in rows)
    print(f"\nSummary: {dict(c)}")
    print("  missing_citation -> facts were retrieved but cited to the wrong/too few [S#]: citation discipline")
    print("  not_in_sources   -> the model added outside knowledge: prompt (or retrieval missed it)")
    print("  mixed            -> some of each")
    print("  recheck_supported-> a second check found it supported by its own sources (checker noise)")


if __name__ == "__main__":
    # python -m ingest.verify "What are INTC's competitors doing that could hurt it?"
    import sys
    from .tools import answer_with_tools
    q = " ".join(sys.argv[1:]) or "What are INTC's competitors doing that could hurt it?"
    r = answer_with_tools(q, check_support=True)
    print(f"tool calls: {len(r['tool_calls'])}, hit limit: {r['hit_tool_limit']}, sources: {len(r['sources'])}\n")
    print(r["answer"][:3000] + "\n")
    print_report(r["verification"])
    print("\n==== diagnosis ====")
    print_diagnosis(diagnose(r["verification"], r["sources"]))