"""Everything the agent needs to use the evidence layer as Gemini tools.

FOR THE AI ANSWERS TEAMMATE: the simplest path is one call:

    from ingest.tools import answer_with_tools
    result = answer_with_tools(
        question="Should I buy NVDA because AI demand will keep growing?",
        system_prompt=YOUR_PROMPT,          # e.g. the Challenge My Trade prompt
    )
    result["answer"]          # model's answer with [S#] citations
    result["verification"]    # verify_citations() report (ok, per-sentence status)
    result["sources"]         # {"S1": {...citation, url, text...}, ...} for the UI
    result["tool_calls"]      # what the agent searched for (debugging)

Or build your own chat loop with the pieces:
    SourceRegistry       keeps [S#] labels unique across ALL searches in one conversation
    gemini_tool()        the tool declarations to pass to Gemini
    run_tool(...)        executes a function call from Gemini and returns its response
    CITATION_RULES       append to your system prompt

Tools the agent gets: search_documents, search_competitors, get_financials, get_guidance_record.
Financial rows get [S#] labels from the same registry as text results, so one
answer can cite a reported number [S7] and the MD&A explanation [S3] together.

Why the registry matters: an answer usually needs several searches (own claims,
competitors, papers). Each search must continue the numbering (S1-S5, then
S6-S10...), and a chunk found twice keeps its first label. verify_citations()
then checks the answer against everything retrieved in the conversation.
"""

from __future__ import annotations

from google.genai import types

from .financials import METRICS, get_financials
from .guidance import guidance_record
from .search import DOC_TYPES, PEERS, describe_index, search, search_competitors
from .verify import verify_citations

MODEL = "gemini-3.8-flash"
DEFAULT_K = 6
MAX_K = 10
MIN_SIMILARITY = None     # set this from the eval's suggested cutoff once you've run it
MAX_TOOL_ROUNDS = 10      # Gemini 3 often makes one call per round; 6 cut thorough answers off

CITATION_RULES = """
## Evidence and citations
You have tools to search SEC filings, earnings call transcripts and research papers.
Every cited sentence is automatically checked against its sources; unsupported sentences are flagged to the user.
- Search before making any factual claim about a company. Never answer facts from memory.
- Use ONLY what the retrieved sources say. Do not add background knowledge, examples, product names,
  customer names, dates or numbers that are not in a cited source, even if you believe they are true.
  If something relevant isn't in the sources, leave it out or say the evidence doesn't cover it.
- One fact per sentence. Keep sentences short; don't pack lists of examples into one sentence.
- Report what sources say, not your characterization: avoid words like "dominate", "relentlessly",
  "directly eliminates" unless a source uses them. Put your own analysis in separate sentences
  WITHOUT citations, clearly framed as interpretation ("This suggests...").
- Every search result has an id like S3. Cite ids at the end of each sentence that states a fact,
  before the period: "Data center revenue grew 112% year over year [S1]." Several: "[S1][S4]".
- Only cite ids that appeared in search results. Never invent one.
- Sentences with no factual claim (e.g. "Here's what the evidence shows.") need no id.
- Results marked as competitor evidence describe ANOTHER company: name that company in the sentence.
- Earnings call turns labeled (Analyst) are questions, not company statements.
- Research papers describe a technology area; they are not statements by the company.
- For reported financial numbers (revenue, gross profit, operating income, net income, diluted EPS,
  R&D expense, cash), use get_financials instead of reading numbers out of text, and cite the row ids.
  Each row is one value for one period. Rows marked derived were calculated (full year minus nine
  months), so say "about" or note it. Compute changes (e.g. % growth) only from cited rows.
- Combine numbers with text: cite the figure from get_financials and the explanation from search results.
- To judge whether management delivers what it promises, call get_guidance_record. Report it in one or
  two sentences, citing the record's summary id (and the guidance/actual ids for any specific quarter).
  If it helps, add ONE small markdown table, at most 5 rows: | Quarter | Guided | Actual | Result |,
  copying values exactly as given, with ids in the Guided and Actual cells. Never more than one table.
  Beating guidance every quarter can just mean cautious guidance; misses are the stronger warning sign.
  If no numeric guidance exists, say so in one sentence.
- If the searches don't support a point, say the evidence is missing instead of guessing.
"""


class SourceRegistry:
    """One per conversation. Assigns stable, unique [S#] labels across searches."""

    def __init__(self) -> None:
        self.sources: dict[str, dict] = {}      # "S1" -> hit
        self._label_of: dict[str, str] = {}     # chunk_id -> "S1"

    def add(self, hits: list[dict]) -> list[tuple[str, dict]]:
        out = []
        for h in hits:
            label = self._label_of.get(h["chunk_id"])
            if label is None:
                label = f"S{len(self.sources) + 1}"
                self._label_of[h["chunk_id"]] = label
                self.sources[label] = h
            out.append((label, h))
        return out


# ---------- tool declarations ----------

def _indexed_tickers() -> list[str]:
    return sorted(t for t in describe_index()["tickers"] if t != "RESEARCH")


def declarations() -> list[types.FunctionDeclaration]:
    tickers = _indexed_tickers()
    doc_types_desc = "; ".join(f"{k}: {v}" for k, v in DOC_TYPES.items())
    common = {
        "doc_types": {"type": "array", "items": {"type": "string", "enum": list(DOC_TYPES)},
                      "description": "Optional. Restrict to these document types. " + doc_types_desc},
        "date_from": {"type": "string", "description": "Optional. Earliest date, YYYY-MM-DD."},
        "date_to": {"type": "string", "description": "Optional. Latest date, YYYY-MM-DD."},
        "k": {"type": "integer", "description": f"Number of results (default {DEFAULT_K}, max {MAX_K})."},
    }
    return [
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the evidence base (SEC 10-K/10-Q/8-K filings, earnings call transcripts, "
                "research papers) for passages relevant to a question. Use specific, focused queries; "
                "make several searches for different angles (e.g. risks, guidance, results). "
                "Pass tickers to restrict results to those companies; research papers tagged with "
                "those companies are included."),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in plain words."},
                    "tickers": {"type": "array", "items": {"type": "string", "enum": tickers},
                                "description": "Optional. Only these companies."},
                    **common,
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="search_competitors",
            description=(
                "Search the filings and calls of a company's direct competitors, e.g. to find "
                "competitive threats or compare results. Results describe the COMPETITOR, not the "
                "company you pass. Competitors: " +
                "; ".join(f"{t} -> {', '.join(p) or 'none indexed'}" for t, p in PEERS.items() if t in tickers)),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "enum": tickers,
                               "description": "The company whose competitors to search."},
                    "query": {"type": "string"},
                    **common,
                },
                "required": ["ticker", "query"],
            },
        ),
        types.FunctionDeclaration(
            name="get_financials",
            description=(
                "Reported financial figures from the company's SEC filings (XBRL data), one row per "
                "period, newest last. Use for revenue, profit, EPS, R&D and cash numbers and trends. "
                "Quarters are labeled by period end date; fiscal quarters differ by company. "
                "Rows marked derived (Q4) were calculated as full year minus nine months."),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "enum": tickers},
                    "metrics": {"type": "array", "items": {"type": "string", "enum": list(METRICS)},
                                "description": "Which figures. " +
                                "; ".join(f"{k}: {v[0]}" for k, v in METRICS.items())},
                    "period": {"type": "string", "enum": ["quarterly", "annual"],
                               "description": "quarterly = 3-month periods; annual = fiscal years."},
                    "last": {"type": "integer", "description": "How many recent periods per metric (default 6, max 12)."},
                },
                "required": ["ticker", "metrics"],
            },
        ),
        types.FunctionDeclaration(
            name="get_guidance_record",
            description=(
                "Did the company deliver the revenue it guided? For each recent quarter: the revenue "
                "guidance from its earnings press release vs. the revenue it then reported, with a verdict "
                "(beat / met / missed / pending) and the gap vs. the guidance midpoint, plus a summary. "
                "Use it to judge management credibility and for 'what management said vs. delivered'."),
            parameters_json_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string", "enum": tickers}},
                "required": ["ticker"],
            },
        ),
    ]


def gemini_tool() -> types.Tool:
    return types.Tool(function_declarations=declarations())


# ---------- executing a tool call ----------

def _format(labeled: list[tuple[str, dict]]) -> list[dict]:
    out = []
    for label, h in labeled:
        if h["doc_type"] == "xbrl_fact":
            r = {"id": label, "metric": h["metric"], "period": h["period_label"], "value": h["value"],
                 "unit": h["unit"], "value_text": h["value_text"], "source": h["citation"],
                 "derived": h["derived"]}
            if h["derived"]:
                r["note"] = "Calculated (full year minus first nine months), not directly reported."
            if h["restated"]:
                r["restated"] = True
            out.append(r)
            continue
        r = {"id": label, "source": h["citation"], "doc_type": h["doc_type"], "date": h["date"],
             "similarity": h["similarity"], "text": h["text"]}
        if h.get("competitor_of"):
            r["note"] = f"Competitor evidence: this is about {h['ticker']}, not {h['competitor_of']}."
        out.append(r)
    return out


def _guidance_tool(ticker: str, registry: SourceRegistry) -> dict:
    rec = guidance_record(ticker)
    if not rec["guidance_given"]:
        return {"results": [], "summary": rec["summary"]}
    rows, lines = [], []
    for c in rec["comparisons"]:
        (g_id, _), *rest = registry.add([c["guide"]] + ([c["actual"]] if c["actual"] else []))
        a_id = rest[0][0] if rest else None
        gap = f"{c['gap_vs_mid_pct']:+.1f}%" if c["gap_vs_mid_pct"] is not None else None
        actual = c["actual"]["value_text"] if c["actual"] else None
        rows.append({"period": c["period"], "guided": c["guide"]["range_text"], "guidance_id": g_id,
                     "actual": actual, "actual_id": a_id, "verdict": c["verdict"], "gap_vs_midpoint": gap,
                     **({"derived_actual": True} if c["actual"] and c["actual"].get("derived") else {})})
        lines.append(f"{c['period']}: guided {c['guide']['range_text']}, actual {actual or 'not reported yet'}, "
                     f"{c['verdict']}" + (f" ({gap} vs. midpoint)" if gap else ""))
    # the computed summary is itself a citable source, so "beat 4 of 5" can be verified
    summary_item = {"chunk_id": f"GUIDE_SUMMARY::{rec['ticker']}", "ticker": rec["ticker"], "doc_type": "guidance_record",
                    "date": rec["comparisons"][-1]["guide"]["date"], "section": "Guidance record",
                    "citation": f"{rec['ticker']} guidance record (computed from its earnings press releases and SEC filings)",
                    "text": rec["summary"] + "\n" + "\n".join(lines) + "\nHow to read this: beating guidance every "
                            "quarter can reflect cautious (conservative) guidance rather than outperformance; misses "
                            "are the stronger warning sign.", "url": None, "similarity": None}
    (s_id, _), = registry.add([summary_item])
    return {"summary": rec["summary"], "summary_id": s_id, "comparisons": rows,
            "how_to_read": "Beating guidance every quarter can mean cautious guidance; misses are the stronger signal."}


def run_tool(name: str, args: dict, registry: SourceRegistry) -> dict:
    """Run one Gemini function call. Never raises: errors go back to the model
    as {"error": ...} so it can fix its arguments and retry."""
    args = dict(args or {})
    k = max(1, min(int(args.pop("k", DEFAULT_K) or DEFAULT_K), MAX_K))
    try:
        if name == "get_guidance_record":
            return _guidance_tool(args["ticker"], registry)
        if name == "get_financials":
            last = max(1, min(int(args.get("last") or 6), 12))
            rows = get_financials(args["ticker"], args.get("metrics"), args.get("period", "quarterly"), last)
            if not rows:
                return {"results": [], "note": "No reported values for that request."}
            return {"results": _format(registry.add(rows)),
                    "note": "Values as reported to the SEC (latest filing wins if restated). "
                            "Quarter labels use period end dates."}
        if name == "search_documents":
            hits = search(args.pop("query"), k=k, min_similarity=MIN_SIMILARITY, **args)
        elif name == "search_competitors":
            hits = search_competitors(args.pop("ticker"), args.pop("query"), k=k,
                                      min_similarity=MIN_SIMILARITY, **args)
        else:
            return {"error": f"Unknown tool {name!r}"}
    except (ValueError, TypeError, KeyError) as e:
        return {"error": str(e)}
    if not hits:
        return {"results": [], "note": "No matching evidence. Try different wording or filters. "
                                       "If still nothing, say the evidence is missing; do not answer from memory."}
    return {"results": _format(registry.add(hits))}


# ---------- reference agent loop ----------

def answer_with_tools(question: str, system_prompt: str = "", model: str = MODEL,
                      check_support: bool = True, history: list[types.Content] | None = None) -> dict:
    """Ask Gemini a question, letting it search as needed, then verify citations.

    history: optional earlier turns (types.Content) for multi-turn chats. Reuse the
    returned "registry" across turns of the same conversation by passing registry=...
    if you extend this; a fresh registry per question is fine for single questions.
    """
    from .embed import client

    registry = SourceRegistry()
    log: list[dict] = []
    contents = list(history or []) + [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    config = types.GenerateContentConfig(
        system_instruction=(system_prompt + "\n" + CITATION_RULES).strip(),
        tools=[gemini_tool()],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    resp = None
    hit_limit = False
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client().models.generate_content(model=model, contents=contents, config=config)
        calls = resp.function_calls or []
        if not calls:
            break
        # Append the model turn exactly as returned: Gemini 3 needs its thought
        # signatures echoed back in the next request.
        contents.append(resp.candidates[0].content)
        parts = []
        for c in calls:
            result = run_tool(c.name, dict(c.args or {}), registry)
            returned = ([r["id"] for r in result.get("results", [])]
                        + [i for row in result.get("comparisons", []) for i in (row["guidance_id"], row["actual_id"]) if i]
                        + ([result["summary_id"]] if result.get("summary_id") else []))
            log.append({"tool": c.name, "args": dict(c.args or {}), "returned": returned, "error": result.get("error")})
            parts.append(types.Part.from_function_response(name=c.name, response=result))
        contents.append(types.Content(role="user", parts=parts))
    else:
        # Still searching at the limit. Ask for the final answer in a FRESH request:
        # the question plus every source already retrieved (same [S#] labels), no
        # tools and no tool-call history. Appending to the tool-call history
        # breaks Gemini 3's turn rules ("Requests ending with a model turn are
        # not supported") and disabling tools alone can return no text.
        hit_limit = True
        final_contents = list(history or []) + [types.Content(role="user", parts=[
            types.Part.from_text(text=_final_prompt(question, registry))])]
        final_cfg = types.GenerateContentConfig(system_instruction=config.system_instruction)
        for _ in range(2):
            resp = client().models.generate_content(model=model, contents=final_contents, config=final_cfg)
            if _text(resp):
                break

    answer = _text(resp)
    return {
        "answer": answer,
        "verification": verify_citations(answer, registry.sources, check_support=check_support),
        "sources": registry.sources,
        "tool_calls": log,
        "hit_tool_limit": hit_limit,
        "registry": registry,
    }


def to_storage_sources(result: dict, max_sources: int = 50, max_text: int = 4000,
                       include_url: bool = False) -> list[dict]:
    """Convert answer_with_tools() sources into the storage API's format
    (POST /api/recommendations "sources": label <= 120 chars, text <= 4,000, <= 50 sources).
    Only the sources the answer actually cites are sent."""
    used = result["verification"].get("used_sources") or result["sources"]
    out = []
    for sid, h in list(used.items())[:max_sources]:
        src = {
            "label": f"{sid} {h['citation']}"[:120],
            "document": h["doc_type"],
            "date": h.get("date_norm") or h.get("date"),
            "page": h.get("section") or h.get("period_label") or "",
            "text": h["text"][:max_text],
        }
        if include_url and h.get("url"):
            src["url"] = h["url"]
        out.append(src)
    return out


def _final_prompt(question: str, registry: SourceRegistry) -> str:
    block = "\n\n".join(f"[{label}] {h['citation']}\n{h['text']}" for label, h in registry.sources.items())
    return (f"{question}\n\n"
            "You have finished searching. Below is everything you retrieved. Write your final answer now, "
            "using ONLY these sources and citing their ids exactly as shown. If something the question asks "
            "about isn't covered, say the evidence is missing.\n\n"
            f"{block or '(no sources were retrieved)'}")


def _text(resp) -> str:
    """Only the written answer: skips function-call and thinking parts."""
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


if __name__ == "__main__":
    import sys
    from .verify import print_report
    q = " ".join(sys.argv[1:]) or "What are the biggest risks to Intel's foundry plans?"
    r = answer_with_tools(q)
    print("Searches:")
    for t in r["tool_calls"]:
        print(f"  {t['tool']}({t['args']}) -> {t['returned']}{'  ERROR: ' + t['error'] if t['error'] else ''}")
    print("\n" + r["answer"] + "\n")
    print_report(r["verification"])