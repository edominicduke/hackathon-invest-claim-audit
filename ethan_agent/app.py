"""Thesis Audit UI.  Run:  streamlit run app.py

Flow: describe a trade -> confirm how it was read -> review the evidence-checked audit.
Two audiences, one interface: "Plain English" for retail investors, "Analyst" adds the
full audit trail (claim types, checks applied, searches, source table, JSON export).

Security: every model- or document-derived string is HTML-escaped before rendering,
so text inside a filing can't inject markup into the page.
"""

from __future__ import annotations

import html
import logging
import re

import streamlit as st

from agent import build_default_auditor
from agent.llm import enable_call_logging
from agent.schemas import (
    ClaimStatus,
    Direction,
    Importance,
    Materiality,
    ParsedThesis,
    ThesisAudit,
    UserAnswer,
)

st.set_page_config(page_title="Thesis Audit", page_icon="📊", layout="wide")
enable_call_logging()  # one line per model call / step in the terminal running streamlit
log = logging.getLogger("agent.app")

# ------------------------------------------------------------------ constants

VERDICT = {  # label, explanation, accent, background
    "green": ("Thesis well supported", "The filings back every core claim that could be checked.", "#067647", "#ECFDF3"),
    "yellow": ("Partly supported", "Some core claims couldn't be confirmed from the filings.", "#B54708", "#FFFAEB"),
    "red": ("Thesis contradicted", "The filings contradict at least one core claim.", "#B42318", "#FEF3F2"),
    "none": ("Can't audit this thesis", "", "#475467", "#F2F4F7"),
}
STATUS = {  # label, text colour, background
    ClaimStatus.SUPPORTED: ("Supported", "#067647", "#ECFDF3"),
    ClaimStatus.CONTRADICTED: ("Contradicted", "#B42318", "#FEF3F2"),
    ClaimStatus.UNVERIFIABLE: ("Not verifiable", "#475467", "#F2F4F7"),
}
TYPE_LABEL = {"fact": "Fact", "causal": "Cause", "forecast": "Forecast", "vague": "Needs definition", "premise": "Assumption"}
DIRECTION_LABEL = {"buy": "Buy", "sell": "Sell", "short": "Short", "hold": "Hold", "unclear": "Unclear"}

EXAMPLES = {
    "Growth thesis": (
        "I'm buying EXMP because AI demand will keep data center revenue growing 50%+ "
        "and they have no real competition."
    ),
    "Well-supported thesis": (
        "I'm buying EXMP because AI demand drove data center revenue up more than 60% last year, "
        "and data center is now most of the company's revenue."
    ),
    "Weak thesis": "Buying EXMP because its gaming business is booming and grew 20% last year.",
}

CSS = """
<style>
.block-container { padding-top: 1.4rem; max-width: 1180px; }
.ta-header { display:flex; justify-content:space-between; align-items:flex-end; gap:12px;
  border-bottom:1px solid #E4E7EC; padding-bottom:14px; margin-bottom:16px; }
.ta-brand { font-size:1.55rem; font-weight:700; letter-spacing:-0.02em; color:#0F172A; line-height:1.2; }
.ta-brand span { color:#1D4ED8; }
.ta-tagline { color:#475467; font-size:0.95rem; margin-top:4px; }
.ta-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.74rem; font-weight:600;
  border:1px solid #D0D5DD; color:#344054; background:#FFFFFF; margin:2px 0 2px 6px; white-space:nowrap; }
.ta-steps { display:flex; gap:8px; margin:0 0 18px; }
.ta-step { flex:1; padding:9px 12px; border-radius:8px; background:#F4F6FA; color:#667085; font-size:0.84rem; font-weight:500; }
.ta-step b { font-weight:700; margin-right:6px; }
.ta-step.active { background:#EEF4FF; color:#1D4ED8; box-shadow:inset 0 0 0 1px #C7D7FE; }
.ta-step.done { color:#067647; }
.ta-verdict { border-radius:12px; padding:18px 22px; border:1px solid; margin-bottom:12px; }
.ta-verdict .eyebrow { font-size:0.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; opacity:.85; }
.ta-verdict .label { font-size:1.55rem; font-weight:700; letter-spacing:-0.01em; margin:2px 0 4px; }
.ta-verdict .meta { font-size:0.85rem; color:#475467; }
.ta-kpis { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:10px; margin-bottom:6px; }
.ta-kpi { border:1px solid #E4E7EC; border-radius:10px; padding:11px 14px; background:#FFFFFF; }
.ta-kpi .v { font-size:1.55rem; font-weight:700; font-variant-numeric:tabular-nums; color:#0F172A; }
.ta-kpi .k { font-size:0.72rem; color:#667085; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
.ta-card { border:1px solid #E4E7EC; border-radius:12px; padding:16px 18px; margin-bottom:12px; background:#FFFFFF; }
.ta-card.accent { border-left-width:4px; }
.ta-card h4 { margin:0 0 6px; font-size:1rem; font-weight:600; color:#0F172A; }
.ta-pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.72rem; font-weight:600; margin:0 6px 6px 0; }
.ta-tag { display:inline-block; padding:1px 8px; border-radius:6px; font-size:0.72rem; font-weight:500;
  color:#475467; background:#F2F4F7; margin:0 6px 6px 0; }
.ta-quote { border-left:3px solid #D0D5DD; padding:7px 12px; margin:10px 0 2px; color:#344054;
  background:#F9FAFB; border-radius:0 6px 6px 0; font-size:0.92rem; }
.ta-quote.contra { border-left-color:#F04438; }
.ta-quote.support { border-left-color:#12B76A; }
.ta-src { font-size:0.76rem; color:#667085; margin-bottom:4px; }
.cite { display:inline-block; font-size:0.66rem; font-weight:700; color:#1D4ED8; background:#EEF4FF;
  border-radius:4px; padding:0 5px; margin-left:2px; vertical-align:1px; cursor:help; }
.ta-section { font-size:0.76rem; font-weight:700; color:#475467; text-transform:uppercase; letter-spacing:.07em; margin:20px 0 8px; }
.ta-muted { color:#667085; font-size:0.84rem; }
.ta-list { margin:4px 0 0 0; padding-left:18px; }
.ta-list li { margin-bottom:4px; }
.ta-legend { margin-bottom:10px; }
.ta-legend .ta-pill { margin-bottom:2px; }
.ta-thesis { border:1px solid #E4E7EC; border-radius:10px; padding:10px 14px; background:#F9FAFB; color:#0F172A; margin-bottom:12px; }
.ta-thesis .ta-muted { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin-bottom:2px; }
.ta-disclaimer { font-size:0.78rem; color:#667085; border-top:1px solid #E4E7EC; padding-top:10px; margin-top:24px; }
@media (max-width: 820px) {
  .ta-kpis { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .ta-header { flex-direction:column; align-items:flex-start; }
  .ta-steps { flex-direction:column; }
  .block-container { padding-top:3.2rem; }
}
</style>
"""

# -------------------------------------------------------------------- helpers


def esc(text: object) -> str:
    return html.escape(str(text or ""))


def md(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def pill(text: str, fg: str, bg: str) -> str:
    return f'<span class="ta-pill" style="color:{fg};background:{bg}">{esc(text)}</span>'


def tag(text: str) -> str:
    return f'<span class="ta-tag">{esc(text)}</span>'


@st.cache_resource
def auditor():
    return build_default_auditor()


def reset() -> None:
    for key in ("parsed", "audit", "review"):
        st.session_state.pop(key, None)


def load_example(text: str) -> None:
    st.session_state.raw = text
    reset()


def analyst() -> bool:
    return st.session_state.get("view", "Plain English") == "Analyst"


# ----------------------------------------------------------------- page chrome

md(CSS)

try:
    model_name = auditor().llm.model
    model_chip = f'<span class="ta-chip">Model · {esc(model_name)}</span>'
except Exception as exc:  # unknown provider / missing SDK: show it instead of crashing
    model_chip = ""
    st.error(f"Model setup failed: {exc}")

md(
    '<div class="ta-header"><div>'
    '<div class="ta-brand">Thesis <span>Audit</span></div>'
    '<div class="ta-tagline">Test the reasons behind a trade against company filings - before you place it.</div>'
    f'</div><div>{model_chip}<span class="ta-chip">Demo data · fictional EXMP</span></div></div>'
)

parsed: ParsedThesis | None = st.session_state.get("parsed")
audit: ThesisAudit | None = st.session_state.get("audit")
step = 3 if audit else 2 if parsed else 1
steps_html = "".join(
    f'<div class="ta-step {"active" if i == step else "done" if i < step else ""}"><b>{"✓" if i < step else i}</b>{name}</div>'
    for i, name in enumerate(["Describe your trade", "Confirm the details", "Review the audit"], start=1)
)
md(f'<div class="ta-steps">{steps_html}</div>')

# --------------------------------------------------------------------- sidebar

with st.sidebar:
    md('<div class="ta-brand" style="font-size:1.2rem">Thesis <span>Audit</span></div>')
    st.segmented_control("View", ["Plain English", "Analyst"], default="Plain English", key="view",
                         help="Analyst adds claim types, checks applied, searches run and the full source table.")
    md('<div class="ta-section">How it works</div>')
    md(
        '<ol class="ta-list ta-muted">'
        "<li>Your thesis is split into separate claims, including assumptions you didn't state.</li>"
        "<li>Each claim is checked against filings and transcripts, with exact quotes.</li>"
        "<li>The tool builds the strongest honest case against your trade.</li></ol>"
    )
    md('<div class="ta-section">Ratings</div>')
    md("".join(
        f'<div class="ta-legend">{pill(VERDICT[lv][0], VERDICT[lv][2], VERDICT[lv][3])}'
        f'<div class="ta-muted">{esc(VERDICT[lv][1])}</div></div>'
        for lv in ("green", "yellow", "red")
    ) + '<div class="ta-muted" style="margin-top:4px">Ratings grade the <b>evidence for your reasons</b> '
        '- not the future stock price.</div>')
    if audit and audit.glossary:
        md('<div class="ta-section">Glossary</div>')
        for g in audit.glossary:
            with st.expander(g.term):
                st.write(g.definition)

# --------------------------------------------------------------- step 1: input

if not audit:
    md('<div class="ta-section">Your trade idea</div>')
    ex_cols = st.columns(len(EXAMPLES) + 2)
    ex_cols[0].markdown('<div class="ta-muted" style="padding-top:6px">Try an example:</div>', unsafe_allow_html=True)
    for col, (name, text) in zip(ex_cols[1:], EXAMPLES.items()):
        col.button(name, on_click=load_example, args=(text,), width="stretch")

if "raw" not in st.session_state:
    st.session_state.raw = EXAMPLES["Growth thesis"]

if audit:
    # Once audited, the thesis is a record: show it, don't invite editing.
    raw = st.session_state.raw
    c_quote, c_btn = st.columns([5, 1], vertical_alignment="center")
    c_quote.markdown(f'<div class="ta-thesis"><div class="ta-muted">Your thesis</div>“{esc(audit.raw_input)}”</div>',
                     unsafe_allow_html=True)
    if c_btn.button("New audit", width="stretch"):
        reset()
        st.rerun()
else:
    raw = st.text_area(
        "Describe the trade and why you're making it",
        key="raw", height=96, on_change=reset, label_visibility="collapsed",
        placeholder="e.g. I'm buying EXMP because data center revenue keeps growing and margins are expanding.",
    )

if not audit and st.button("Check my thesis", type="primary", disabled=not raw.strip()):
    reset()
    log.info("[UI] 'Check my thesis' clicked")
    with st.spinner("Reading your thesis..."):
        st.session_state.parsed = auditor().parse(raw)
    p = st.session_state.parsed
    log.info("[UI] Read as: ticker=%s direction=%s ambiguities=%d", p.ticker, p.direction.value, len(p.ambiguities))
    st.rerun()

# ------------------------------------------------------------- step 2: confirm

if parsed and not audit:
    md('<div class="ta-section">Confirm the details</div>')
    with st.container(border=True):
        st.markdown("**Here's how we read your trade.** Correct anything that's wrong before we run the audit.")
        if not parsed.is_auditable:
            st.warning(parsed.not_auditable_reason or "This doesn't look like a thesis about a public company.")
        for a in parsed.ambiguities:
            st.info(f"Please check: {a}")
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value=parsed.ticker or "", help=parsed.company_name or None)
        directions = [d.value for d in Direction]
        direction = c2.selectbox("Direction", directions, index=directions.index(parsed.direction.value),
                                 format_func=lambda d: DIRECTION_LABEL.get(d, d))
        horizon = c3.number_input("Time horizon (months, 0 = not stated)", min_value=0, value=parsed.horizon_months or 0)
        if st.button("Confirm and run audit", type="primary", disabled=not ticker.strip()):
            confirmed = parsed.model_copy(update={
                "ticker": ticker.strip().upper(),
                "direction": Direction(direction),
                "horizon_months": horizon or None,
                "is_auditable": True,
            })
            log.info("[UI] Audit started for %s (%s)", confirmed.ticker, confirmed.direction.value)
            with st.status("Auditing your thesis - this usually takes a minute or two...", expanded=True) as status:
                st.session_state.audit = auditor().audit(raw, parsed=confirmed, on_progress=st.write)
                status.update(label="Audit complete", state="complete", expanded=False)
            st.rerun()

# -------------------------------------------------------------- step 3: result

if audit:
    own_ticker = (audit.parsed.ticker or "").upper()
    src_by_label = {s.label: s for s in audit.sources}

    def src_name(label: str) -> str:
        s = src_by_label.get(label)
        if not s:
            return label
        parts = [s.ticker, s.doc_type, s.period or "", s.section or ""]
        return " · ".join(p for p in parts if p)

    def cited(text: str) -> str:
        """Escape model text, then turn [S#] labels into hoverable citation chips."""
        return re.sub(
            r"\[(S\d+)\]",
            lambda m: f'<span class="cite" title="{esc(src_name(m.group(1)))}">{m.group(1)}</span>',
            esc(text),
        )

    def quote_block(ev) -> str:
        s = src_by_label.get(ev.source)
        competitor = s is not None and s.ticker.upper() != own_ticker
        kind = "support" if ev.stance == "supports" else "contra"
        stance = "Supports" if ev.stance == "supports" else "Contradicts"
        extra = f" · {tag('Competitor filing')}" if competitor else ""
        return (
            f'<div class="ta-quote {kind}">“{esc(ev.quote)}”</div>'
            f'<div class="ta-src"><span class="cite">{esc(ev.source)}</span> {esc(src_name(ev.source))} · {stance}{extra}</div>'
        )

    # ---- verdict banner + KPIs
    level = audit.verdict.level.value
    v_label, v_meaning, v_fg, v_bg = VERDICT[level]
    p = audit.parsed
    meta = " · ".join(x for x in [
        f"{esc(p.ticker)}" + (f" ({esc(p.company_name)})" if p.company_name else "") if p.ticker else "",
        DIRECTION_LABEL.get(p.direction.value, p.direction.value),
        f"{p.horizon_months}-month horizon" if p.horizon_months else "",
        audit.created_at.strftime("%d %b %Y, %H:%M UTC"),
    ] if x)
    md(
        f'<div class="ta-verdict" style="border-color:{v_fg}33;background:{v_bg}">'
        f'<div class="eyebrow" style="color:{v_fg}">Audit result</div>'
        f'<div class="label" style="color:{v_fg}">{esc(v_label)}</div>'
        f'<div>{esc(v_meaning)}</div><div class="meta" style="margin-top:6px">{meta}</div></div>'
    )

    counts = {s: sum(1 for r in audit.claims if r.status == s) for s in ClaimStatus}
    kpis = [
        ("Claims checked", len(audit.claims)),
        ("Supported", counts[ClaimStatus.SUPPORTED]),
        ("Contradicted", counts[ClaimStatus.CONTRADICTED]),
        ("Not verifiable", counts[ClaimStatus.UNVERIFIABLE]),
        ("Sources cited", len(audit.sources)),
    ]
    md('<div class="ta-kpis">' + "".join(
        f'<div class="ta-kpi"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in kpis
    ) + "</div>")
    for w in audit.warnings:
        st.warning(w)

    tab_overview, tab_claims, tab_challenge, tab_sources = st.tabs(
        ["Overview", "Claims & evidence", "Challenge my trade", "Sources & audit trail"]
    )

    # ---- overview
    with tab_overview:
        left, right = st.columns([3, 2], gap="large")
        with left:
            if audit.explanation:
                md('<div class="ta-section">In plain English</div>')
                md(f'<div class="ta-card">{cited(audit.explanation.plain_summary)}</div>')
                md('<div class="ta-section">Key takeaways</div>')
                md('<ul class="ta-list">' + "".join(f"<li>{cited(t)}</li>" for t in audit.explanation.key_takeaways) + "</ul>")
        with right:
            md('<div class="ta-section">Why this rating</div>')
            md('<div class="ta-card"><ul class="ta-list">'
               + "".join(f"<li>{esc(r)}</li>" for r in audit.verdict.reasons) + "</ul></div>")
            ch = audit.challenge
            if ch:
                md('<div class="ta-section">Both sides</div>')
                md(f'<div class="ta-card accent" style="border-left-color:#12B76A"><h4>Strongest point for</h4>'
                   f'{cited(ch.strongest_point_for)}</div>')
                if ch.counterarguments:
                    # Prefer an evidence-backed counterargument over a generic risk.
                    top = next((c for c in ch.counterarguments if c.materiality == Materiality.SPECIFIC),
                               ch.counterarguments[0])
                    md(f'<div class="ta-card accent" style="border-left-color:#F04438"><h4>Strongest point against</h4>'
                       f'{cited(top.text)}</div>')

    # ---- claims
    with tab_claims:
        md('<div class="ta-muted" style="margin:6px 0 12px">Your thesis broken into separate claims, each checked '
           'against the filings. <b>Assumptions</b> are things your thesis depends on that you didn\'t state.</div>')
        if analyst():
            st.dataframe(
                [{
                    "ID": r.claim.id,
                    "Claim": r.claim.text,
                    "Type": TYPE_LABEL.get(r.claim.type.value, r.claim.type.value),
                    "Weight": r.claim.importance.value.title(),
                    "Status": STATUS[r.status][0],
                    "Model said": STATUS[r.llm_status][0],
                    "Quotes": len(r.evidence),
                    "Checks fired": len(r.issues),
                } for r in audit.claims],
                hide_index=True, width="stretch",
            )
        for r in audit.claims:
            s_label, s_fg, s_bg = STATUS[r.status]
            tags = pill(s_label, s_fg, s_bg)
            tags += tag("Core claim" if r.claim.importance == Importance.CORE else "Supporting claim")
            tags += tag(TYPE_LABEL.get(r.claim.type.value, r.claim.type.value))
            if not r.claim.stated_by_user:
                tags += tag("Unstated assumption")
            quotes = "".join(quote_block(ev) for ev in r.evidence)
            md(f'<div class="ta-card"><div>{tags}</div><h4>{esc(r.claim.text)}</h4>'
               f'<div>{cited(r.reasoning)}</div>{quotes}</div>')
            if analyst() and (r.issues or r.searches):
                with st.expander(f"Audit trail · {r.claim.id}"):
                    if r.status != r.llm_status:
                        st.markdown(f"Model said **{STATUS[r.llm_status][0]}**; code checks changed it to **{s_label}**.")
                    if r.issues:
                        st.markdown("**Checks applied**\n" + "\n".join(f"- {esc(i)}" for i in r.issues))
                    if r.searches:
                        st.markdown("**Searches run**\n" + "\n".join(f"- `{esc(q)}`" for q in r.searches))

    # ---- challenge my trade
    with tab_challenge:
        ch = audit.challenge
        if not ch:
            st.info("The challenge step didn't complete for this audit.")
        else:
            md('<div class="ta-section">The case against your trade</div>')
            for ca in ch.counterarguments:
                backed = ca.materiality == Materiality.SPECIFIC
                badge = pill("Evidence-backed", "#B42318", "#FEF3F2") if backed else pill("General risk", "#475467", "#F2F4F7")
                md(f'<div class="ta-card">{badge}<div>{cited(ca.text)}</div>'
                   + "".join(quote_block(ev) for ev in ca.evidence) + "</div>")

            md('<div class="ta-section">What if the stock fell 30%?</div>')
            st_ = ch.stress_test
            md(f'<div class="ta-card">{cited(st_.scenario)}</div>')
            c1, c2 = st.columns(2, gap="large")
            c1.markdown('<div class="ta-card"><h4>Must stay true for your thesis</h4><ul class="ta-list">'
                        + "".join(f"<li>{cited(x)}</li>" for x in st_.what_would_have_to_be_true) + "</ul></div>",
                        unsafe_allow_html=True)
            c2.markdown('<div class="ta-card"><h4>Warning signs to watch</h4><ul class="ta-list">'
                        + "".join(f"<li>{cited(x)}</li>" for x in st_.early_warning_signs) + "</ul></div>",
                        unsafe_allow_html=True)

            md('<div class="ta-section">Questions for you</div>')
            md('<div class="ta-muted" style="margin-bottom:8px">Answering these helps you pressure-test your own '
               'reasoning. Your answers are saved but don\'t change the rating - they\'re your views, not evidence.</div>')
            with st.form("answers"):
                answers = []
                for i, q in enumerate(ch.questions_for_user):
                    a = st.text_area(q.question, help=q.why_it_matters, key=f"q{i}", height=80)
                    answers.append(UserAnswer(question=q.question, answer=a))
                if st.form_submit_button("Save my answers", type="primary"):
                    filled = [a for a in answers if a.answer.strip()]
                    if filled:
                        log.info("[UI] %d answer(s) submitted", len(filled))
                        with st.spinner("Reviewing your answers..."):
                            st.session_state.review = auditor().review_answers(audit, filled)
                    else:
                        st.warning("Write at least one answer first.")
            review = st.session_state.get("review")
            if review:
                body = f"<div>{esc(review.notes)}</div>"
                if review.new_concerns:
                    body += '<ul class="ta-list" style="margin-top:8px">' + "".join(
                        f"<li>{esc(c)}</li>" for c in review.new_concerns) + "</ul>"
                md(f'<div class="ta-card accent" style="border-left-color:#1D4ED8"><h4>Feedback on your answers</h4>{body}</div>')

    # ---- sources & audit trail
    with tab_sources:
        md('<div class="ta-section">Sources cited</div>')
        if audit.sources:
            st.dataframe(
                [{
                    "Ref": s.label,
                    "Company": s.ticker,
                    "Relation": "Competitor" if s.ticker.upper() != own_ticker else "Company",
                    "Document": s.doc_type,
                    "Period": s.period or "",
                    "Section": s.section or "",
                    "Filed": s.filed_date or "",
                    **({"Chunk ID": s.chunk_id} if analyst() else {}),
                } for s in audit.sources],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No sources were cited.")

        if audit.citation_issues or analyst():
            md('<div class="ta-section">Citation checks</div>')
            if audit.citation_issues:
                md('<div class="ta-muted" style="margin-bottom:6px">Problems our code caught in the model\'s '
                   'citations and fixed or flagged before showing you the result.</div>')
                md('<ul class="ta-list ta-muted">' + "".join(f"<li>{esc(i)}</li>" for i in audit.citation_issues) + "</ul>")
            else:
                st.caption("No citation problems found.")

        if analyst():
            md('<div class="ta-section">Audit record</div>')
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="ta-muted">Audit ID</div><code>{esc(audit.audit_id)}</code>', unsafe_allow_html=True)
            c2.markdown(f'<div class="ta-muted">Model</div><code>{esc(audit.model)}</code>', unsafe_allow_html=True)
            c3.markdown(f'<div class="ta-muted">Verdict rule</div><code>{esc(audit.verdict.rule)}</code>', unsafe_allow_html=True)
            st.download_button(
                "Download audit (JSON)", audit.model_dump_json(indent=2),
                file_name=f"thesis-audit-{audit.audit_id}.json", mime="application/json",
            )

    md(f'<div class="ta-disclaimer">{esc(audit.disclaimer)} Demo data covers fictional companies only.</div>')
