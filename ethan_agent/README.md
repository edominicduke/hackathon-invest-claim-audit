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
streamlit run app.py                                 # UI (see "UI design" below)
python -m agent "I'm buying EXMP because ..."        # CLI -> audit JSON
pytest                                               # 119 tests, no API calls
python -m eval.run_eval [--verdicts]                 # eval against the real model (costs API calls)
```

The agent uses the team's code in `../stock-agent/` (see [How the team's code is connected](#how-the-teams-code-is-connected)): real SEC filings, earnings calls and papers for **NVDA, AMD, INTC, TSLA and GOOGL**. The fictional company **EXMP** is served from `data/fixtures/exmp_demo.json` (with a planted prompt-injection chunk to test against). Any other ticker has no documents to search, so its claims come back `unverifiable`. The default provider is Claude (`claude-opus-5-5`, override with `AUDITOR_MODEL`), with server-side refusal fallbacks (`fallbacks="default"`). Set `AUDITOR_PROVIDER=gemini` to use [agent/llm_gemini.py](agent/llm_gemini.py) instead. It defaults to `gemini-3.8-flash` (override with `GEMINI_MODEL`), maps effort to Gemini thinking levels, and treats safety blocks as refusals, with no fallback model. Both adapters implement the same `structured()` method, so the pipeline, prompts and checks are identical. API keys are read only from environment variables; never commit them. The terminal running the app logs every pipeline step, every model call (time, thinking tokens) and every retry. Set `AUDITOR_MAX_WORKERS` to control how many claims are checked in parallel.

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
          Bogdan's number check ─► Guru's independent check of the summary ─► Abdullah's store
```

| File | Responsibility |
|---|---|
| [agent/schemas.py](agent/schemas.py) | All data contracts. `ThesisAudit` is the team hand-off object |
| [agent/interfaces.py](agent/interfaces.py) | `Retriever` and `CitationVerifier` (Guru), `AuditStore` (Abdullah), `NumberChecker` (Bogdan) |
| [agent/integrations.py](agent/integrations.py) | Adapters from those interfaces to the team's `stock-agent/` code |
| [agent/pipeline.py](agent/pipeline.py) | The agent: orchestration plus code-side enforcement |
| [agent/prompts.py](agent/prompts.py) | Every prompt (Features #6 and #9) |
| [agent/verdict.py](agent/verdict.py) | Rule-based 🟢🟡🔴 |
| [agent/citations.py](agent/citations.py) | Deterministic citation checks |
| [agent/glossary.py](agent/glossary.py) | Reviewed glossary, matched by code |
| [agent/stubs.py](agent/stubs.py) | Stand-ins: fixture search for EXMP, and fallbacks if `stock-agent/` can't load |
| [eval/run_eval.py](eval/run_eval.py) | Claim- and verdict-level evaluation |

## How the team's code is connected

`build_default_auditor()` loads the adapters in [agent/integrations.py](agent/integrations.py). Each adapter only translates formats; none of them changes how the agent reasons. If one can't load (a missing package, a moved folder), the terminal logs a warning and that part falls back to its stand-in. Set `AUDITOR_TEAM_COMPONENTS=0` to use the stand-ins only.

| Interface | Teammate's code | What the adapter does |
|---|---|---|
| `Retriever` | Guru: `src/ingest/search.py` | Searches the audited company's documents, plus 1 chunk from its rivals per search (`AUDITOR_COMPETITOR_K`). Rival chunks are marked `COMPETITOR`. Research papers are left out by default, because they can't confirm company facts and crowd out filings on broad queries. `AUDITOR_INCLUDE_PAPERS=1` lets them back in, marked `RESEARCH`. Tickers outside Guru's index go to the EXMP fixture |
| `CitationVerifier` | Guru: `src/ingest/verify.py` | Runs once over the finished plain-language summary |
| `NumberChecker` | Bogdan: `src/checks/number_check.py`, `sec_facts.py` | Every number in a quote, or in a sentence with `[S#]`, must appear in the source(s) it cites, with rounding allowed. Numbers from the investor's own thesis are allowed. With `SEC_USER_AGENT` set, dollar figures are also compared with the company's official SEC (XBRL) numbers; matches are shown as confirmations |
| `AuditStore` | Abdullah: `src/app/store.py` | Archives the full audit JSON in `data/store/`. It also saves a recommendation and a Challenge My Trade session to `stock-agent/data/`, the folder the results check and the track record read. Investors' answers are added to that session. Fictional demo tickers stay out of the track record |
| Track record page | Abdullah's storage + Bogdan: `src/checks/results_check.py`, `prices.py` | The app's **Track record** page lists every saved audit and its returns vs the S&P 500 at 1/3/6/12 months. The start price is recorded when the audit is saved (`AUDITOR_RECORD_PRICE=0` to skip). **Check results now** runs Bogdan's results check for every horizon that's due |

**Team decisions:**
- **Two citation checks with different jobs.** The agent's strict, quote-level checks (verbatim quote, then entailment) run *inside* the pipeline and can change a claim's status. Guru's sentence-level check runs *after*, over the summary the user reads, and is shown as an independent second opinion. It never changes the verdict.
- **The verdict means strength of evidence, not a buy call.** It is stored as `green` / `yellow` / `red` for the track record, which compares it with the S&P 500 only as a long-run reference.

**Optional settings:**
- **Hybrid search:** `GOOGLE_CLOUD_PROJECT` (with Guru's Vertex setup) turns on embedding search, plus Vertex re-reading of each summary sentence (`AUDITOR_SUPPORT_CHECK`). Without it, search is keyword-only and the summary check is structural: invented labels, uncited numbers.
- **Official SEC figures:** on whenever `SEC_USER_AGENT` is set to a team name plus contact email (the SEC requires one), e.g. `$env:SEC_USER_AGENT = "Team1J-StockAgent you@example.com"`. `AUDITOR_SEC_NUMBERS=0` turns it off.
- **Folders:** `STOCK_AGENT_DIR` and `STOCK_AGENT_DATA_DIR` override the folder locations.

## UI design

Direction: **"Research Note"**, an editorial research report (serif headlines, paper and ink) with a modern fintech's clarity. Light theme only.

- **Tokens live in one place.** Colours, fonts, radius and spacing are CSS variables in [styles/theme.css](styles/theme.css); Streamlit's own widgets are matched in [.streamlit/config.toml](.streamlit/config.toml). Python code refers to tones (`good` / `warn` / `bad` / `neutral`), never hex codes.
- **Fonts are bundled** in `static/fonts/` (Inter, Newsreader, JetBrains Mono; SIL OFL), so the app looks the same offline.
- **Evidence colours are not market colours.** The rating is shown as *Evidence strength* (Strong / Mixed / Contradicted), and every status carries an icon and a word, so green never reads as "buy" and meaning never depends on colour alone.
- **Layered depth.** Scorecard, then reasons, then quotes, then raw sources. The Plain English / Analyst switch adds claim types, checks applied and searches run.

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
