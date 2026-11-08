"""Methodology page: how a thesis is checked, what the rating means, what the data is, and where
the tool is limited. Every statement here is meant to match the code (agent/pipeline.py,
agent/verdict.py, agent/citations.py, agent/integrations.py); update both together."""

from __future__ import annotations

import streamlit as st

from ui.components import VERDICT, md, pill

md('<div class="ta-eyebrow">Methodology</div>')
md('<h1 class="ta-hero-title">How Thesis Audit reaches a rating</h1>')
md('<div class="ta-hero-sub">A fixed, inspectable pipeline: the AI reads and judges, plain code enforces the rules '
   "and makes the call. Quotes are machine-checked against the sources, and the Analyst view lists every check that fired.</div>")

# ---- pipeline
md('<div class="ta-section">The pipeline</div>')
steps = [
    ("Read", "Your free-text idea becomes a ticker, a direction and a horizon. You confirm it before anything runs."),
    ("Decompose", "The thesis is split into up to 7 claims: facts, causes, forecasts, vague wording and hidden assumptions."),
    ("Search", "Each claim gets an equal number of queries seeking support and seeking contradiction, plus an excerpt "
               "from a rival's filings. If a claim stays unverifiable, up to two follow-up searches run."),
    ("Judge", "The model labels each claim and quotes the filing. Code confirms each quote is in the retrieved passage "
              "(ignoring spacing and punctuation, allowing a 90% near-match) and drops any that are not."),
    ("Challenge", "A model-written case against the trade, a \"what if the stock fell 30%?\" stress test and questions for you."),
    ("Verify", "Numbers must appear in the source they cite. Dollar figures are also compared with official SEC data "
               "when that is configured. A separate checker then re-reads the plain-English summary."),
]
md('<div class="ta-flow">' + "".join(
    f'<div><div class="n">{i:02d}</div><h4>{name}</h4><p>{text}</p></div>' for i, (name, text) in enumerate(steps, start=1)
) + "</div>")

# ---- rating rules
md('<div class="ta-section">How the rating is decided</div>')
md('<div class="ta-prose">The rating comes from rules, not from the model\'s opinion. The model labels each claim as '
   "supported, contradicted or not verifiable; code turns those labels into a rating. The rules are deterministic, "
   "but the model's labels can still vary between runs.</div>")
rules = {
    "green": "At least one core claim can be checked against filings, every such claim is supported, "
             "and no claim is contradicted.",
    "yellow": "Anything in between: a core claim couldn't be verified, a supporting claim is contradicted, "
              "or the thesis rests only on forecasts and vague claims.",
    "red": "A core claim is contradicted by a specific, quoted passage.",
}
md("".join(
    f'<div class="ta-rule"><div>{pill(VERDICT[lv][0], VERDICT[lv][2])}</div><div class="ta-prose">{rules[lv]}</div></div>'
    for lv in ("green", "yellow", "red")
))
md('<div class="ta-prose"><b>Built to be fair, not to always say no.</b> Failing to find evidence is never treated '
   "as evidence against: it can cap a rating at Mixed, but never makes it Contradicted. Forecasts and vague claims "
   "can never be marked supported, so they are not required to be. The counterarguments in Challenge My Trade are "
   "shown to you but never feed the rating; only a claim's own cited evidence can contradict it.</div>")

# ---- trust
md('<div class="ta-section">What is checked, and how far to trust it</div>')
left, right = st.columns(2, gap="large")
with left:
    md('<div class="ta-card"><h4>Quotes are checked by code</h4>'
       '<div class="ta-prose">A claim is only marked supported if the model quotes the filing and our code finds '
       "that text in the retrieved passage. A second, model-based pass then checks that each quote actually backs "
       "its claim (wrong period, wrong segment, forward-looking language cited as fact), and drops quotes that fail. "
       "These checks are a safety net, not proof: a sentence with no number and no source label is not flagged.</div></div>")
with right:
    md('<div class="ta-card"><h4>Filing text is treated as data</h4>'
       '<div class="ta-prose">Documents are wrapped in tags and the model is told that instructions inside them are '
       "content, not commands. That is a prompt-level defence, not a guarantee. Text shown on screen is "
       "HTML-escaped, so source text cannot alter the page.</div></div>")

# ---- data and limits
md('<div class="ta-section">Data and limits</div>')
md('<div class="ta-prose"><b>Sources.</b> SEC filings, earnings call transcripts and research papers for NVDA, AMD, '
   "INTC, TSLA and GOOGL. The fictional company EXMP uses built-in demo data so the product can be tried without "
   "real filings. Any other ticker has no documents to search, so its claims come back as not verifiable. "
   "Research papers are left out of company-fact checks by default because they cannot confirm what a company "
   "reported.<br><br>"
   "<b>What a rating is not.</b> It is not a price forecast. Filings are public, so what they say is largely "
   "already priced in; strong evidence for your reasons does not mean the stock will rise. That is why the Track "
   "record page compares results with the S&amp;P 500 only as a long-run reference.<br><br>"
   "<b>Known limits.</b> The model can misread a passage, and search can miss a relevant one. That is why "
   "unverifiable claims are shown as such, and why the Analyst view exposes the searches run and the checks "
   "applied. The independent summary check re-reads sources only when Google Cloud is configured; otherwise it "
   "checks citations and numbers only.</div>")
md('<div class="ta-disclaimer">Research project, not financial advice.</div>')
