# Investment Thesis Auditor — Challenge My Trade

An AI agent that takes an investor's free-text thesis, breaks it into claims, checks each one against company filings, builds the strongest honest case against the trade, and returns a source-backed 🟢 / 🟡 / 🔴 assessment in plain language.

**What the verdict means:** how well the evidence in filings backs the *thesis*. It is not a stock-price prediction and not investment advice.

## Quick start

This is Ethan's part of the team project (agent reasoning, Challenge My Trade, explanations, UI). Everything lives in `ethan_agent/`, and **all commands below run from inside this folder**.

```bash
cd ethan_agent
python -m venv ../.venv && ../.venv/Scripts/activate   # one venv at the repo root; use bin/activate on macOS/Linux
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="sk-ant-..."                  # PowerShell; bash: export ANTHROPIC_API_KEY=...
# ...or use Gemini instead of Claude:
#   $env:AUDITOR_PROVIDER="gemini"; $env:GEMINI_API_KEY="..."   (optional: $env:GEMINI_MODEL="...")
#   $env:GEMINI_FAST="1"   one thinking level lower + claims checked one at a time (free-tier friendly)
#   Thinking is LOW for every Gemini model by default; lift with $env:GEMINI_MAX_THINKING="MEDIUM" or "HIGH"
#   Daily quota used up? The app switches to the next model automatically. Choose the order with
#   $env:GEMINI_FALLBACK_MODELS="model-a,model-b" (default: discovered from your key); off: GEMINI_AUTO_SWITCH=0
streamlit run app.py                                 # UI
python -m agent "I'm buying EXMP because ..."        # CLI -> audit JSON
pytest                                               # 98 tests, no API calls
python -m eval.run_eval [--verdicts]                 # eval against the real model (costs API calls)
```

Until the real retriever is connected, the agent searches `data/fixtures/exmp_demo.json`, a **fictional** company (EXMP). The fixture includes a planted prompt-injection chunk to test against. The default provider is Claude (`claude-opus-5-5`, override with `AUDITOR_MODEL`), with server-side refusal fallbacks (`fallbacks="default"`). Set `AUDITOR_PROVIDER=gemini` to use [agent/llm_gemini.py](agent/llm_gemini.py) instead. It defaults to `gemini-3.8-flash` (override with `GEMINI_MODEL`), maps effort to Gemini thinking levels, and treats safety blocks as refusals, with no fallback model. Both adapters implement the same `structured()` method, so the pipeline, prompts and checks are identical. API keys are read only from environment variables; never commit them. The terminal running the app logs every pipeline step, every model call (time, thinking tokens) and every retry. Set `AUDITOR_MAX_WORKERS` to control how many claims are checked in parallel.

## Pipeline

```
free text ─► parse ─► user confirms "I read this as: BUY EXMP"
                │
                ▼
          decompose into 3–7 claims (fact / causal / forecast / vague / hidden premise,
                │                   core vs supporting, balanced support+challenge queries)
                ▼
   per claim (parallel):  retrieve ─► judge ─► ≤2 follow-up searches if unverifiable
                │                      └─► code: chunk_id retrieved? quote verbatim? status rules
                ▼
          entailment check (does each real quote actually support its claim?)
                ▼
          Challenge My Trade: strongest point for, counterarguments, "what if it dropped 30%?",
                │             questions for the user  (code: "specific" requires a valid citation)
                ▼
          verdict = plain code over claim statuses ─► plain-language summary ─► glossary
                ▼
          Bogdan's number check ─► Abdullah's store
```

| File | Responsibility |
|---|---|
| [agent/schemas.py](agent/schemas.py) | All data contracts. `ThesisAudit` is the team hand-off object |
| [agent/interfaces.py](agent/interfaces.py) | `Retriever` (Guru), `AuditStore` (Abdullah), `NumberChecker` (Bogdan) |
| [agent/pipeline.py](agent/pipeline.py) | The agent: orchestration plus code-side enforcement |
| [agent/prompts.py](agent/prompts.py) | Every prompt (Features #6 and #9) |
| [agent/verdict.py](agent/verdict.py) | Rule-based 🟢🟡🔴 |
| [agent/citations.py](agent/citations.py) | Deterministic citation checks |
| [agent/glossary.py](agent/glossary.py) | Reviewed glossary, matched by code |
| [agent/stubs.py](agent/stubs.py) | Stand-ins until teammates' parts land |
| [eval/run_eval.py](eval/run_eval.py) | Claim- and verdict-level evaluation |

## Integration contract for teammates

- **Guru:** implement `search(query, *, ticker, k) -> list[Chunk]`. Fill in `period` and `section` on every chunk, because the period and scope checks depend on them. Return relevant competitor chunks with `related_tickers=[audited ticker]`; the agent marks them `relation="COMPETITOR"`. In transcript chunks, keep the speaker tags such as `(Analyst)`. The agent decides *what* to search for, including adversarial queries. You decide *how* to find it. The Feature #3 citation check currently lives in `agent/citations.py`, so agree on who owns it.
- **Abdullah:** implement `save_audit(ThesisAudit)` and `save_answers(audit_id, list[UserAnswer])`. `ThesisAudit` has company (`parsed.ticker`), reason (`raw_input`), verdict, sources and `created_at`. Add the price from Bogdan at save time.
- **Bogdan:** implement `check(audit, chunks) -> list[str]`, where `chunks` is keyed by label (`"S3"`). Numbers appear in `Evidence.quote` (checkable against `chunks[ev.source].text`) and in prose sentences ending in `[S#]`. `audit.sources` maps each label to its `chunk_id`.

Then wire them in: `ThesisAuditor(llm=ClaudeLLM(), retriever=..., store=..., number_checker=...)`.

## Design decisions

1. **Free-text input, structured internally, confirmed by the user.** Extraction errors (wrong ticker, a sell read as a buy) would otherwise corrupt storage and scoring. The LLM guesses the ticker and code validates it against a lookup table.
2. **Verdict = strength of evidence, not a price call.** Filings are public, so their contents are largely priced in. Good evidence for a thesis does not predict returns. So the agent is evaluated on claim accuracy, not on 1/3/6/12-month outcomes, which are only a long-run reference signal.
3. **A fixed pipeline with one bounded agentic step, not an autonomous loop.** The product's value is that it *always* searches for counter-evidence. An agent that decides when it's done could stop after confirming searches. Autonomy is limited to the one place it helps: at most 2 follow-up searches for unverifiable claims.
4. **Claim breakdown with types.** Judging a whole thesis lets a strong claim cover for a weak one. Each type needs a different strategy: facts are checked, forecasts are stress-tested on their assumptions (never "supported"), vague claims become questions for the user, and hidden premises are surfaced.
5. **Agent-written, balanced queries.** Searching on the thesis text returns passages that agree with it, which is confirmation bias in the embeddings. Code enforces an equal number of support and challenge queries.
6. **Every prompt rule is backed by a code check.** Citations must point to a retrieved chunk with a verbatim quote. "Supported" requires a surviving supporting citation. Forecasts can't be supported. A second LLM pass checks that each quote actually entails its claim (wrong period, segment, or forward-looking language cited as fact).
7. **"I don't know" is a feature.** No evidence means `unverifiable`, never a number from model memory. Not finding evidence is not evidence against the claim, so `unverifiable` can cap a verdict at 🟡 but can't cause 🔴.
8. **The verdict comes from rules, the judgment from the LLM.** The LLM labels claim status and importance, and code turns that into a colour. The verdict is deterministic, explainable, testable, and not pushed negative by the "challenge" framing.
9. **Built not to always say no.** Boilerplate risk factors are tagged "generic" and don't affect the verdict. A "specific" counterargument needs a valid citation. Counterarguments are shown to the user but not scored. Retrieval is balanced. Tests assert that a solid thesis gets 🟢.
10. **Filing text is untrusted.** Documents are wrapped in tags and the prompts say embedded instructions are data. The fixture's injection chunk exercises this.
11. **User answers are saved but never change the verdict.** They are the user's views, not evidence about the company.
12. **Two citation channels: `[S#]` labels for attribution, verbatim quotes for verification** (format from Guru's spec). Labels are numbered once per audit, so `[S3]` means the same source in every step. Per-call numbering would silently corrupt citations once outputs are combined. Code strips invented labels and flags sentences that state a number without a label. Competitor sources are marked so the model names the company. Analyst questions in transcripts are never treated as company statements; there's an eval case for this.

## Evaluation

Labels are built **by perturbation**, with no expert needed. Claims the corpus states are labeled `supported`. The same claims with a number scaled or a direction word flipped are labeled `contradicted`. Hand-written period and scope traps, plus claims the corpus is silent on, are labeled `unverifiable`.

The headline metric is the **false-support rate** (a false claim labeled supported), because that's the costliest error. **Recall on supported** is the "doesn't always say no" test. Verdict-level cases include a thesis that should get 🟢.
