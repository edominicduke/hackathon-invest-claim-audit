"""Thesis Audit UI.  Run:  streamlit run app.py

Entry point only: page setup, stylesheet, top navigation and the header row. The pages
themselves live in views/ (audit, track record, methodology); shared helpers in ui/.

Security: every model- or document-derived string is HTML-escaped before rendering,
so text inside a filing can't inject markup into the page.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from agent.llm import enable_call_logging
from ui.components import auditor, esc, load_css, md

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Thesis Audit", page_icon=str(HERE / "static" / "icon.svg"), layout="wide")
enable_call_logging()  # one line per model call / step in the terminal running streamlit
load_css()
st.logo(str(HERE / "static" / "wordmark.svg"), size="large")

nav = st.navigation(
    [
        # Absolute paths: pages resolve the same however the app is launched.
        st.Page(HERE / "views" / "audit.py", title="Audit", default=True),
        st.Page(HERE / "views" / "track_record.py", title="Track record"),
        st.Page(HERE / "views" / "methodology.py", title="Methodology"),
    ],
    position="top",
)

# Header row: coverage badge on the right (the product name sits in the top bar as the logo).
# Shown only when the real index is loaded; demo-only setups show nothing here.
coverage = ""
try:
    retriever = auditor().retriever
    covered = getattr(retriever, "tickers", None)
    if covered:
        coverage = f"{len(covered)} companies · 10-K, 10-Q & earnings calls"
        through = getattr(retriever, "data_through", None)
        if through:
            coverage += f" · through {date.fromisoformat(through[:10]).strftime('%d %b %Y').lstrip('0')}"
except Exception as exc:  # unknown provider / missing SDK: show it instead of crashing
    st.error(f"Model setup failed: {exc}")
if coverage:
    md(f'<div class="ta-header-row"><span class="ta-chip">{esc(coverage)}</span></div>')

nav.run()
