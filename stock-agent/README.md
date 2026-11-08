# Stock Agent: Storage + Checks (Features #3, #4, #9)

The "memory" and "proof" of the agent. Saves every recommendation and Challenge My Trade
session, checks later whether each call beat the S&P 500, and checks that numbers the
AI writes really appear in its sources. No LLM calls in this part.

## Files

| File | What it does |
|---|---|
| `src/app/store.py` | Storage: `RecommendationStore`, `ChallengeStore` (one JSON file per item, KPMG `JsonStore` pattern) |
| `src/app/server.py` | API routes (FastAPI) |
| `src/checks/prices.py` | Stock + S&P 500 (SPY) prices via yfinance; weekends use Friday's price |
| `src/checks/results_check.py` | 1/3/6/12-month check vs. the S&P 500, saves results, track record summary |
| `src/checks/number_check.py` | Number check: every number in an answer must appear in a source |
| `src/checks/sec_facts.py` | Downloads official company numbers (revenue, net income, EPS, debt...) from SEC EDGAR for the number check |
| `tests/` | Tests for storage, routes, results check, number check, SEC numbers |

## Who built what

| Part | Owner | Files |
|---|---|---|
| Storage + routes (#4, #9) | Abdullah | `src/app/store.py`, `src/app/server.py`, `src/app/__main__.py`, `tests/test_storage.py`, `tests/test_api.py` |
| Prices + results check (#4) | Bogdan | `src/checks/prices.py`, `src/checks/results_check.py`, `tests/test_results_check.py` |
| Number check + SEC numbers (#3) | Bogdan | `src/checks/number_check.py`, `src/checks/sec_facts.py`, `tests/test_number_check.py`, `tests/test_sec_facts.py` |
| Shared | Both | `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `__init__.py` files |

`src/app/server.py` is the only file that uses both parts: its routes call the storage and the checks.

## Run it (Mac, VS Code terminal)

```bash
cd stock-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # all tests should pass
python -m src.app            # then open http://127.0.0.1:8000/docs
```

For the SEC number check, set your contact once per terminal (the SEC requires it):

```bash
export SEC_USER_AGENT="Team1J-StockAgent yourname@example.com"
```

Check recommendations against the market (run daily, e.g. with Cloud Scheduler or cron):

```bash
python -m src.checks.results_check --data-dir data
```

## Routes

**Recommendations (#4)**
- `POST /api/recommendations`: save one
- `GET /api/recommendations?company=&verdict=`: list
- `GET /api/recommendations/{id}`: open one (includes its challenges)
- `DELETE /api/recommendations/{id}`: also deletes its challenges
- No PUT: a saved recommendation can't be edited, so the track record stays honest. Only the results check fills in `results`.

**Challenge My Trade (#9)**
- `POST /api/recommendations/{id}/challenges`
- `GET /api/recommendations/{id}/challenges`
- `GET / PUT / DELETE /api/challenges/{id}`: PUT keeps fields you leave out; new answers are added to old ones

**Track record + number check (#4, #3)**
- `GET /api/track-record`: how often the agent was right at 1/3/6/12 months
- `POST /api/number-check`: body `{"answer": "...", "sources": [{"label": "...", "text": "..."}], "company": "NVDA"}`. Add `company` to also check against official SEC numbers
- `GET /api/recommendations/{id}/number-check?use_sec=true`: checks a saved summary against its saved sources (and SEC numbers when `use_sec=true`)
- Each number comes back with `found`, the `source` it matched, and `official_sec: true` if it matched an official SEC number

## The format teammates send (agree on this!)

**Ethan → `POST /api/recommendations`**
```json
{
  "company": "NVDA",
  "user_reason": "AI chip demand is growing",
  "verdict": "yellow",
  "confidence": "medium",
  "summary": "Data center revenue was $4.2 billion, up 18%...",
  "sources": [
    {"label": "10-Q p.8", "document": "10-Q", "date": "2026-08-27", "page": "8",
     "text": "Data center revenue was $4,213 million, up 18.4%..."}
  ],
  "created_by": "agent"
}
```
- `verdict`: `green` (buy) · `yellow` (wait) · `red` (avoid) · `not_enough_evidence`
- `as_of_date` (optional, defaults to today): set a past date to test the results check right away
- **Guru:** each source needs a `label`, and its `text` (the passage the AI used) is required for the number check

**Ethan → `POST /api/recommendations/{id}/challenges`**
```json
{
  "bear_case": ["AI revenue is only 4% of sales", "Margins fell two quarters in a row"],
  "drop_reason": "Missing earnings expectations on Nov 20",
  "answers": {"Why now?": "...", "What would make you sell?": "...", "How much of your money is this?": "..."},
  "verdict": "yellow",
  "verdict_reason": "Good company, risky timing"
}
```

## Rules
- Limits: reason/answers/text ≤ 4,000 characters, labels ≤ 120, ≤ 50 sources, ≤ 20 answers. Ticker like `NVDA` or `BRK.B`.
- Bad input → 400, unknown id → 404.
- Scoring: green is right if the stock beat the S&P 500; red is right if it didn't. Yellow and not_enough_evidence aren't scored.
- SEC numbers used: revenue, cost of revenue, gross profit, operating income, net income, R&D, EPS, cash, assets, liabilities, long-term debt, equity, from 10-K/10-Q filings in the last 3 years. Growth percentages ("up 18%") are not in SEC data, so they must come from the sources.
- Number check allows rounding ($4,213 million = "$4.2 billion") and ignores citations, years, quarters (Q3), form names (10-K) and time spans (12 months).

## Demo (no AI needed)
1. `/docs` → `POST /api/recommendations` with `"as_of_date": "2025-06-02"` (a past date)
2. `POST /api/recommendations/{id}/challenges` with some answers
3. Run `python -m src.checks.results_check --data-dir data`
4. `GET /api/recommendations/{id}`: results vs. the S&P 500 are saved
5. `GET /api/track-record`: accuracy summary
6. Restart the app, reload: everything is still there

## Not financial advice
This is a student research project. It doesn't connect to any brokerage and makes no trades.

# Stock Agent: Data + Search + Citations (Feature #3, evidence for #6 and #9)

The "evidence" of the agent. Downloads primary sources (SEC filings, earnings call transcripts,
SEC financial data, research papers), makes them searchable, gives the AI tools to use them, and
checks that every sentence the AI writes is supported by the source it cites.

## Files

| File | What it does |
|---|---|
| `src/ingest/edgar_fetch.py` | Downloads 10-K, 10-Q and 8-K earnings press releases from SEC EDGAR, plus official financial data (XBRL) |
| `src/ingest/transcripts_fetch.py` | Earnings call transcripts from Alpha Vantage (free key, 25/day) |
| `src/ingest/arxiv_fetch.py` | Research papers from arXiv, tagged with the companies whose technology they cover |
| `src/ingest/chunk.py` | Splits documents into small chunks with a source label (company, document, date, section) |
| `src/ingest/embed.py`, `build_index.py` | Embeddings (Vertex `gemini-embedding-001`) and building the search index |
| `src/ingest/search.py` | Hybrid search (keyword + embeddings), competitor search, filter checks |
| `src/ingest/financials.py` | Cleaned official numbers (revenue, net income, EPS...) for the AI to cite |
| `src/ingest/guidance.py` | Said vs. delivered: revenue guidance from press releases vs. reported revenue |
| `src/ingest/tools.py` | The AI's tools, `[S#]` source labels, `answer_with_tools()`, citation rules |
| `src/ingest/verify.py` | Feature #3: checks every cited sentence against its sources |
| `src/ingest/check_pipeline.py` | End-to-end health check |
| `src/ingest/eval_retrieval.py`, `label_checker.py` | Evaluations: search quality, checker accuracy |
| `src/ingest/common.py`, `llm.py` | Shared helpers |
| `src/data/` | Downloaded documents, search index, eval files |

## Who built what

| Part | Owner | Files |
|---|---|---|
| Data, search, citations (#3) | Gurinder | everything in `src/ingest/`, `src/data/` |

`tools.py` is where this part meets the AI Answers part: Ethan's prompts call `answer_with_tools()`.

## Run it (Mac, VS Code terminal)

```bash
cd stock-agent
source .venv/bin/activate
pip install -r requirements.txt
```

Google Cloud, once per laptop (your account must be added to the project in IAM):

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
gcloud auth application-default set-quota-project <PROJECT_ID>
```

Once per terminal (or add to `~/.zshrc`):

```bash
export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
```

Then, from `stock-agent/`:

```bash
python -m src.ingest.check_pipeline --no-agent              # health check, should end with 0 fail
python -m src.ingest.tools "What are the biggest risks to Intel?"   # ask the agent
```

The data is already in the repo, so there's nothing to download. To add a company:

```bash
export SEC_IDENTITY="Team1J-StockAgent yourname@example.com"   # the SEC requires a contact
export ALPHAVANTAGE_API_KEY=...
python -m src.ingest.edgar_fetch MSFT
python -m src.ingest.transcripts_fetch MSFT --quarters 2026Q1 2026Q2
python -m src.ingest.build_index                               # always run after fetching
python -m src.ingest.guidance --extract                        # re-read press releases for guidance
```

Guidance (said vs. delivered), after fetching press releases:

```bash
python -m src.ingest.guidance --extract       # Gemini pulls revenue guidance out of each press release (~cents)
python -m src.ingest.guidance --review        # CHECK each number against its quote, then set "reviewed": true in the file
python -m src.ingest.guidance --show NVDA     # guided vs. actual, verdict, gap
```

## Functions

**Ask the agent (Ethan)**
- `answer_with_tools(question, system_prompt)`: the AI searches, writes an answer with `[S#]` citations, and the answer is checked. Returns:
  - `answer`: text with `[S1]`, `[S2]`... citations
  - `verification`: the citation check (below)
  - `sources`: `{"S1": {citation, url, text, ...}}` for showing sources in the UI
  - `tool_calls`: what the AI searched for
- `to_storage_sources(result)`: converts `sources` into the storage format (below)

**The AI's tools** (called by Gemini, not by you)
- `search_documents`: filings, transcripts and papers; filter by company, document type, date
- `search_competitors`: a company's rivals (NVDA → AMD, INTC); results are marked so the answer names the rival
- `get_financials`: official reported numbers, one row per quarter or year, each with its own `[S#]`
- `get_guidance_record`: revenue guidance vs. what was then reported, per quarter (beat / met / missed / pending), plus a summary. The AI reports it in 1-2 sentences and, if useful, one small table (max 5 rows)

**Citation check (#3)**
- `verify_citations(answer, sources, check_support=True)`: each sentence comes back with a `status`:
  - `supported`: every fact is in the cited sources
  - `partial`: some details aren't, and `unsupported_parts` lists them (e.g. `["MI400"]`). Show these in the UI
  - `unsupported`: the claim isn't in the cited sources
  - `invalid_citation`: cites an `[S#]` that was never retrieved
  - `uncited_number`: a number with no citation
- `ok` is `false` if anything is `unsupported`, `invalid_citation` or `uncited_number`. `partial` is a warning.

**Direct use**
- `search(query, tickers, doc_types, date_from, date_to, k)`: search without the AI
- `get_financials(ticker, metrics, period, last)`, `clean_facts(ticker)`: official numbers (cleaned)

## The format we send (agree on this!)

**Ethan → `POST /api/recommendations`**, using this part's output:
```python
result = answer_with_tools(question, system_prompt=CHALLENGE_PROMPT)
payload = {
    "company": "NVDA",
    "user_reason": "...",
    "verdict": "yellow",
    "confidence": "medium",
    "summary": result["answer"],                      # keeps [S#]; the number check ignores citations
    "sources": to_storage_sources(result),            # only the cited sources, in the storage format
    "created_by": "agent",
}
```
`to_storage_sources()` produces exactly what storage expects:
```json
{"label": "S3 INTC 10-Q (filed 2025-10-31), Item 2 MD&A",
 "document": "10-Q", "date": "2025-10-31", "page": "Part I Item 2 MD&A",
 "text": "Revenue for the quarter was $13.7 billion..."}
```
- `label` starts with the `[S#]` used in the summary, so the UI can link citations to sources
- SEC filings have no page numbers (they're HTML), so `page` holds the section
- Financial rows come through with `document: "xbrl_fact"`
- Limits are applied for you: label ≤ 120 characters, text ≤ 4,000, ≤ 50 sources

## Rules

- Search filters are checked: an unknown ticker or document type gives an error listing the valid ones.
  Valid document types: `10-K`, `10-Q`, `8-K_press_release`, `8-K_transcript`, `earnings_call`, `paper`.
- A ticker filter never returns another company's documents. Papers count if they're tagged with that company.
- The AI only uses retrieved sources, one fact per sentence. Its own analysis goes in separate, uncited sentences ("This suggests...").
- Official numbers: restated values use the latest filing, and Q4 is computed (full year minus 9 months), marked `derived`. Never for EPS.
- Guidance: guidance in a press release filed on date D is matched to the first quarter ending after D (works for any fiscal calendar). beat = above the range, met = inside, missed = below. Each guidance quote is checked word for word against the press release; unverified ones are left out. NVDA, AMD and INTC give numeric guidance; GOOGL doesn't, and TSLA's is mostly qualitative.
- Beating guidance every quarter can just mean cautious guidance; misses are the stronger warning sign.
- Table rows in answers are checked by number: every $ amount and % must appear in the row's cited sources.
- If Google Cloud can't be reached, search falls back to keyword-only. The AI and the citation check still need Google Cloud.
- Data: NVDA, AMD, INTC, TSLA, GOOGL, as of the last download.

## Demo

1. `python -m src.ingest.search --describe`: what's in the evidence base
2. `python -m src.ingest.verify "What are INTC's competitors doing that could hurt it?"`: an answer with every sentence checked, unsupported details named, and a diagnosis
3. `python -m src.ingest.financials INTC`: official numbers, with computed rows marked
4. `python -m src.ingest.guidance --show NVDA`: what NVIDIA guided vs. what it delivered
5. Before and after (same question, as the citation rules and checks were added):

| Version | Supported | Partly | Unsupported | Passes |
|---|---|---|---|---|
| Original | 1 | – | 15 | no |
| Stricter citation rules | 8 | – | 7 | no |
| + fact-by-fact check | 10 | 6 | 1 | no |
| Final | 12 | 6 | 0 | yes |

## Tests

- `python -m src.ingest.check_pipeline`: data, index, search, financials, citation check, and 3 real agent questions
- `python -m src.ingest.eval_retrieval --generate 40`, then `--judge -v`: does search find the right passage
- `python -m src.ingest.label_checker --collect 30`, then `--label`, then `--report`: does the citation check agree with a person

## Not financial advice
This is a student research project. It doesn't connect to any brokerage and makes no trades.