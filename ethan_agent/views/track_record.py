"""Track record page: saved audits of real companies and their results vs the S&P 500."""

from __future__ import annotations

import re

import streamlit as st

from ui.components import VERDICT, analyst, auditor, esc, log, md

# --------------------------------------------------------------- track record

HORIZON_LABEL = {"1m": "1 month", "3m": "3 months", "6m": "6 months", "12m": "12 months"}
HORIZON_DAYS = {"1m": 30, "3m": 91, "6m": 182, "12m": 365}


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:+.1f}%"


def no_labels(text: str) -> str:
    """Saved summaries keep [S#] labels, but their sources aren't shown on this page."""
    return re.sub(r"\s*\[S\d+\]", "", text or "")


def render_track_record() -> None:
    from datetime import date, timedelta

    store = auditor().store
    md('<div class="ta-section">Track record</div>')
    if not hasattr(store, "track_record"):
        st.info("The track record needs the team's storage (the stock-agent folder), which isn't loaded.")
        return
    md('<div class="ta-muted ta-muted-block">Every audit of a real company is saved here. Later, each '
       "stock's return is compared with the S&P 500 at 1, 3, 6 and 12 months. A rating grades the "
       "<b>evidence for the reasons</b>, not the price, so this is a long-run reference rather than a score: "
       "a strongly supported thesis counts as right if the stock beat the market, a contradicted one if it didn't, "
       "and mixed-evidence theses aren't scored.</div>")

    c_btn, c_msg = st.columns([1, 3], vertical_alignment="center")
    if c_btn.button("Check results now", help="Downloads prices and fills in every horizon that's due."):
        log.info("[UI] Results check started")
        with st.spinner("Downloading prices and comparing with the S&P 500..."):
            try:
                report = store.update_results()
                st.session_state.results_report = report
            except Exception as exc:
                st.session_state.results_report = {"failed": str(exc)}
        log.info("[UI] Results check: %s", st.session_state.results_report)
    report = st.session_state.get("results_report")
    if report:
        if "failed" in report:
            c_msg.error(f"Results check failed: {report['failed']}")
        else:
            c_msg.caption(f"Checked {report['checked']} audits · {len(report['updated'])} updated"
                          + (f" · {len(report['errors'])} couldn't be priced" if report["errors"] else ""))

    summary = store.track_record()
    recs = store.recommendations()
    by = summary["by_verdict"]
    kpis = [
        ("Audits saved", summary["total_recommendations"], ""),
        ("Strong evidence", by.get("green", 0), "tone-good"),
        ("Mixed evidence", by.get("yellow", 0), "tone-warn"),
        ("Contradicted", by.get("red", 0), "tone-bad"),
        ("Awaiting first check", summary["waiting_for_first_check"], "tone-neutral"),
    ]
    md('<div class="ta-kpis">' + "".join(
        f'<div class="ta-kpi {tone and "accent " + tone}"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v, tone in kpis
    ) + "</div>")
    if not recs:
        st.info("No audits of real companies yet. Run one (e.g. the AMD example) and it will appear here.")
        return

    md('<div class="ta-section">Results vs the S&P 500</div>')
    chart = [{"After": HORIZON_LABEL[h], "Avg. return vs S&P 500 (%)": v["avg_excess_return_pct"]}
             for h, v in summary["horizons"].items() if v["avg_excess_return_pct"] is not None]
    if chart:
        st.bar_chart(chart, x="After", y="Avg. return vs S&P 500 (%)", color="#0F5F66", height=240, sort=False)
    st.dataframe(
        [{
            "After": HORIZON_LABEL[h],
            "Audits checked": v["checked"],
            "Scored (strong/contradicted)": v["scored"],
            "Right": v["correct"],
            "Accuracy": f"{v['accuracy_pct']:.0f}%" if v["accuracy_pct"] is not None else "–",
            "Avg. return vs S&P 500": pct(v["avg_excess_return_pct"]),
        } for h, v in summary["horizons"].items()],
        hide_index=True, width="stretch",
    )
    md('<div class="ta-muted">Mixed-evidence ratings aren\'t scored. Past results don\'t predict future results.</div>')

    md('<div class="ta-section">Saved audits</div>')
    today = date.today()

    def horizon_cell(rec: dict, h: str) -> str:
        res = rec.get("results", {}).get(h)
        if res:
            return f"{pct(res['excess_return_pct'])} vs S&P"
        due = date.fromisoformat(rec["as_of_date"]) + timedelta(days=HORIZON_DAYS[h])
        return f"due {due:%d %b %Y}" if due > today else "due now"

    st.dataframe(
        [{
            "Date": r["as_of_date"],
            "Company": r["company"],
            "Rating": VERDICT.get(r["verdict"], (r["verdict"],))[0],
            "Price at audit": f"${r['price_on_date']:,.2f}" if r.get("price_on_date") else "–",
            **{HORIZON_LABEL[h]: horizon_cell(r, h) for h in HORIZON_DAYS},
            **({"ID": r["id"]} if analyst() else {}),
        } for r in recs],
        hide_index=True, width="stretch",
    )

    for r in recs[:15]:
        label = f"{r['as_of_date']} · {r['company']} · {VERDICT.get(r['verdict'], (r['verdict'],))[0]}"
        with st.expander(label):
            md(f'<div class="ta-thesis"><div class="ta-muted">Thesis</div>“{esc(r["user_reason"])}”</div>')
            if r.get("summary"):
                md(f'<div class="ta-card">{esc(no_labels(r["summary"]))}</div>')
            for ch in r.get("challenges", []):
                if ch.get("bear_case"):
                    items = ch["bear_case"] if isinstance(ch["bear_case"], list) else [ch["bear_case"]]
                    md('<div class="ta-muted"><b>Case against</b></div><ul class="ta-list">'
                       + "".join(f"<li>{esc(no_labels(x))}</li>" for x in items) + "</ul>")
                answers = {q: a for q, a in (ch.get("answers") or {}).items() if a}
                if answers:
                    md('<div class="ta-muted ta-mt8"><b>Investor\'s answers</b></div>'
                       + "".join(f"<div class='ta-muted'><i>{esc(q)}</i></div><div>{esc(a)}</div>"
                                 for q, a in answers.items()))
            if r.get("sources"):
                st.caption(f"{len(r['sources'])} sources saved with this audit")


render_track_record()
