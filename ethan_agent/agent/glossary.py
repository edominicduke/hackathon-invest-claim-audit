"""Feature #6 glossary. Deterministic: terms are matched in the audit text by code, so
the definitions shown are reviewed text, not model output that could drift."""

from __future__ import annotations

import re

from .schemas import GlossaryEntry, ThesisAudit

GLOSSARY: dict[str, str] = {
    "10-K": "A company's yearly report to the SEC, with audited financial results and a list of risks.",
    "10-Q": "A company's quarterly report to the SEC. Less detailed than a 10-K and not fully audited.",
    "8-K": "A report a company must file with the SEC soon after a major event, like a deal or a regulatory notice.",
    "SEC": "The U.S. Securities and Exchange Commission, the government agency that oversees public companies.",
    "earnings call": "A call where company leaders discuss the latest results with analysts and answer questions.",
    "revenue": "The total money a company brings in from sales, before any costs are subtracted.",
    "segment": "A division of a company that reports its own results, like 'Data Center' or 'Gaming'.",
    "year over year": "Compared with the same period one year earlier.",
    "sequential": "Compared with the period right before (e.g. this quarter vs. last quarter).",
    "fiscal year": "The 12-month period a company uses for its accounts. It doesn't always match the calendar year.",
    "gross margin": "The share of revenue left after paying the direct cost of making the product.",
    "guidance": "The company's own forecast of its future results.",
    "forward-looking statement": "A statement about what a company expects to happen. It is a hope or plan, not a result.",
    "risk factors": "A required section of a filing that lists things that could hurt the company. Many are standard legal wording.",
    "market share": "The company's slice of all sales in its market.",
    "customer concentration": "When a few customers make up a large share of sales, so losing one would hurt a lot.",
    "export license": "Government permission needed to sell certain products to certain countries.",
    "inventory": "Products made or bought but not yet sold. Rising inventory can mean demand is slowing.",
    "thesis": "Your reasons for believing an investment will work out.",
    "short": "A bet that a stock's price will fall.",
    "position size": "How much of your total money is in one investment.",
    "priced in": "When the market already expects something, so the stock price already reflects it.",
    "valuation": "How expensive a stock is compared with the company's sales or profits.",
    "accelerator": "A specialised computer chip, like a GPU, built to run tasks such as AI much faster.",
}

_PATTERNS = {
    term: re.compile(r"(?<![\w-])" + re.escape(term).replace(r"\ ", r"[\s-]") + r"s?(?![\w-])", re.I)
    for term in GLOSSARY
}


def find_terms(text: str) -> list[GlossaryEntry]:
    return [GlossaryEntry(term=t, definition=GLOSSARY[t]) for t, p in _PATTERNS.items() if p.search(text)]


def audit_text(audit: ThesisAudit) -> str:
    parts = [audit.raw_input]
    parts += [f"{r.claim.text} {r.reasoning}" for r in audit.claims]
    if audit.challenge:
        parts.append(audit.challenge.model_dump_json())
    if audit.explanation:
        parts.append(audit.explanation.plain_summary)
    parts += [f"{s.doc_type} {s.section or ''}" for s in audit.sources]
    return "\n".join(parts)
