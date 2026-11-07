# Invest Claim Audit

An AI agent that checks whether your reason for buying a stock is actually true.

You enter a company and your reason. The agent finds evidence in SEC reports, earnings calls and research papers, argues against your trade (Challenge My Trade), gives a cited verdict in plain words, and keeps an honest track record against the S&P 500.

**Demo stocks:** NVDA, AMD, INTC, TSLA, GOOGL

## Team
- **Guru:** data + search (SEC reports, earnings calls, papers)
- **Ethan:** AI answers, Challenge My Trade, simple explanations
- **Abdullah:** storage + track record (`stock-agent/`)
- **Bogdan:** number check, SEC numbers, results vs. S&P 500 (`stock-agent/`)

## Code
- `stock-agent/`: storage, routes, number check and results check (60 tests). See its README to run it.

Research project, not financial advice.
