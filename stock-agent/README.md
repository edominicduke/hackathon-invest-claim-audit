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
