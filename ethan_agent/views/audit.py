"""Audit page: describe a trade -> confirm how it was read -> review the evidence-checked audit."""

from __future__ import annotations

import re

import streamlit as st

from agent.schemas import ClaimStatus, Direction, Importance, Materiality, ParsedThesis, ThesisAudit, UserAnswer
from ui.components import (
    CHECK_LABEL, DIRECTION_LABEL, EXAMPLES, FAILING, STATUS, TONE_ICON, TYPE_LABEL, VERDICT,
    analyst, auditor, esc, load_example, log, md, pill, reset, tag,
)

parsed: ParsedThesis | None = st.session_state.get("parsed")
audit: ThesisAudit | None = st.session_state.get("audit")

# Static preview of a finished audit (fictional company EXMP, figures from the demo fixture).
def _sample_claim(tone: str, text: str, source: str) -> str:
    return (f'<div class="s-claim tone-{tone}"><span class="ci">{TONE_ICON[tone]}</span>'
            f'<div>{text}<div class="ta-src">{source}</div></div></div>')


SAMPLE_AUDIT = (
    '<div class="ta-sample tone-warn"><span class="ta-sample-tag">Sample audit</span>'
    '<div class="s-head"><span class="s-label">Mixed evidence</span><span class="ta-tag">EXMP</span></div>'
    '<div class="ta-muted">Illustrative example on a fictional company.</div>'
    '<div class="ta-mix"><i class="m-good" style="flex:2"></i><i class="m-neutral" style="flex:1"></i></div>'
    + _sample_claim("good", "Data Center revenue grew 61% in fiscal 2025.", "EXMP · 10-K · FY2025 · MD&amp;A")
    + _sample_claim("good", "Gross margin expanded to 71.2% from 64.0%.", "EXMP · 10-K · FY2025 · Gross Margin")
    + _sample_claim("neutral", "Data Center revenue will keep growing 50%+.",
                    "Forecast · couldn’t be confirmed from filings")
    + "</div>"
)

# ------------------------------------------------------------------- stepper

step = 3 if audit else 2 if parsed else 1
steps_html = "".join(
    f'<div class="ta-step {"active" if i == step else "done" if i < step else ""}"><b>{"✓" if i < step else i}</b>{name}</div>'
    for i, name in enumerate(["Describe your trade", "Confirm the details", "Review the audit"], start=1)
)
STEPPER = f'<div class="ta-steps">{steps_html}</div>'
if audit:
    md(STEPPER)  # on the landing view the stepper sits inside the hero, right above "Your trade idea"

# --------------------------------------------------------------- step 1: input

if "raw" not in st.session_state:
    st.session_state.raw = ""


def pick_example() -> None:
    """Pills callback: fill the text box from the chosen example, then clear the pill."""
    name = st.session_state.get("example_pick")
    if name:
        load_example(EXAMPLES[name])
        st.session_state.example_pick = None


if audit:
    # Once audited, the thesis is a record: show it, don't invite editing.
    raw = st.session_state.raw
    c_quote, c_view, c_btn = st.columns([4, 1.6, 1], vertical_alignment="center")
    c_quote.markdown(f'<div class="ta-thesis"><div class="ta-muted">Your thesis</div>“{esc(audit.raw_input)}”</div>',
                     unsafe_allow_html=True)
    c_view.segmented_control("View", ["Plain English", "Analyst"], default="Plain English", key="view",
                             help="Analyst adds claim types, checks applied, searches run and the full source table.")
    if c_btn.button("New audit", width="stretch"):
        reset()
        st.rerun()
else:
    hero_left, hero_right = st.columns([1.45, 1], gap="large")
    with hero_left:
        md('<div class="ta-eyebrow">Evidence-first investing research</div>')
        md('<h1 class="ta-hero-title">Test the reasons behind a trade</h1>')
        md('<div class="ta-hero-sub">Thesis Audit splits your thesis into claims and checks each one against '
           "company filings and earnings calls, then argues the case against it, before you place the trade.</div>")
        md('<div class="ta-trust"><span>SEC filings</span><span>Earnings calls</span><span>Every claim quoted</span></div>')
        md(STEPPER)
        md('<div class="ta-section">Your trade idea</div>')
        md('<div class="ta-muted ta-mb6">Start from an example, or write your own:</div>')
        st.pills("Examples", list(EXAMPLES), key="example_pick", on_change=pick_example, label_visibility="collapsed")
        raw = st.text_area(
            "Describe the trade and why you're making it",
            key="raw", height=110, on_change=reset, label_visibility="collapsed",
            placeholder="e.g. I'm buying NVDA because data center revenue keeps growing and margins are expanding.",
        )
        check = st.button("Check my thesis", type="primary", disabled=not raw.strip(), key="cta")
        md('<div class="ta-cta-note">About 1–2 minutes · cites the filing passages behind each claim</div>')
    with hero_right:
        md(SAMPLE_AUDIT)

    if check:
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

# ------------------------------------------- landing content (was the sidebar)

if not audit:
    c_how, c_ratings = st.columns(2, gap="large")
    with c_how:
        md('<div class="ta-section">How it works</div>')
        md(
            '<ul class="ta-how">'
            '<li><span class="ta-ico ico-claims"></span>Your thesis is split into separate claims, including assumptions you didn&rsquo;t state.</li>'
            '<li><span class="ta-ico ico-search"></span>Each claim is checked against filings and transcripts, with quoted passages.</li>'
            '<li><span class="ta-ico ico-scale"></span>The tool builds the strongest honest case against your trade.</li></ul>'
        )
    with c_ratings:
        md('<div class="ta-section">Evidence ratings</div>')
        md("".join(
            f'<div class="ta-legend">{pill(VERDICT[lv][0], VERDICT[lv][2])}'
            f'<div class="ta-muted">{esc(VERDICT[lv][1])}</div></div>'
            for lv in ("green", "yellow", "red")
        ) + '<div class="ta-muted">Ratings grade the <b>evidence for your reasons</b> '
            '- not the future stock price, and not a buy or sell call.</div>')

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

    def relation_of(s) -> str:
        if s.doc_type == "paper":
            return "Research paper"
        return "Competitor" if s.ticker.upper() != own_ticker else "Company"

    def quote_block(ev) -> str:
        s = src_by_label.get(ev.source)
        rel = relation_of(s) if s is not None else "Company"
        kind = "support" if ev.stance == "supports" else "contra"
        stance = "Supports" if ev.stance == "supports" else "Contradicts"
        extra = {"Competitor": f" · {tag('Competitor filing')}",
                 "Research paper": f" · {tag('Research paper')}"}.get(rel, "")
        return (
            f'<div class="ta-quote {kind}">“{esc(ev.quote)}”</div>'
            f'<div class="ta-src"><span class="cite">{esc(ev.source)}</span> {esc(src_name(ev.source))} · {stance}{extra}</div>'
        )

    # ---- verdict banner + KPIs
    level = audit.verdict.level.value
    v_label, v_meaning, v_tone = VERDICT[level]
    p = audit.parsed
    meta = " \u00b7 ".join(x for x in [
        f"{esc(p.ticker)}" + (f" ({esc(p.company_name)})" if p.company_name else "") if p.ticker else "",
        DIRECTION_LABEL.get(p.direction.value, p.direction.value),
        f"{p.horizon_months}-month horizon" if p.horizon_months else "",
        audit.created_at.strftime("%d %b %Y, %H:%M UTC"),
    ] if x)
    counts = {s: sum(1 for r in audit.claims if r.status == s) for s in ClaimStatus}
    mix_spec = ((ClaimStatus.SUPPORTED, "m-good", "var(--good)", "supported"),
                (ClaimStatus.CONTRADICTED, "m-bad", "var(--bad)", "contradicted"),
                (ClaimStatus.UNVERIFIABLE, "m-neutral", "#9AA3B2", "not verifiable"))
    mix = "".join(f'<i class="{cls}" style="flex:{counts[st_]}" title="{counts[st_]} {name}"></i>'
                  for st_, cls, _dot, name in mix_spec if counts[st_])
    legend = "".join(f'<span><i style="background:{dot}"></i><b>{counts[st_]}</b> {name}</span>'
                     for st_, _cls, dot, name in mix_spec)
    md(
        f'<div class="ta-score tone-{v_tone}"><div class="ta-score-top"><div>'
        f'<div class="eyebrow">Evidence strength</div>'
        f'<div class="label"><span class="icon">{TONE_ICON[v_tone]}</span>{esc(v_label)}</div>'
        f'<div class="meaning">{esc(v_meaning)}</div><div class="meta">{meta}</div></div>'
        f'<div class="ta-score-side"><div class="ta-stat">{len(audit.claims)}</div><div class="ta-stat-k">Claims checked</div>'
        f'<div class="ta-stat ta-mt8">{len(audit.sources)}</div><div class="ta-stat-k">Sources cited</div></div></div>'
        f'<div class="ta-mix">{mix}</div><div class="ta-mix-legend">{legend}</div></div>'
    )
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
                sc = audit.summary_check
                if sc:
                    checked = [s for s in sc.sentences if s.cited]
                    passed = sum(1 for s in checked if s.status not in FAILING)
                    how = "sources re-read sentence by sentence" if sc.support_checked else "citations and numbers checked"
                    md(f'<div class="ta-muted ta-mb8">'
                       f'{pill("Independent check passed" if sc.ok else "Independent check flagged issues", "good" if sc.ok else "bad")}'
                       f'{passed} of {len(checked)} cited sentences passed ({how}). Details under Sources.</div>')
                md('<div class="ta-section">Key takeaways</div>')
                md('<ul class="ta-list">' + "".join(f"<li>{cited(t)}</li>" for t in audit.explanation.key_takeaways) + "</ul>")
            if audit.glossary:
                with st.expander(f"Glossary · {len(audit.glossary)} terms used in this audit"):
                    for g in audit.glossary:
                        md(f"<div><b>{esc(g.term)}</b></div><div class='ta-muted ta-mb8'>"
                           f"{esc(g.definition)}</div>")
        with right:
            md('<div class="ta-section">Why this rating</div>')
            md('<div class="ta-card"><ul class="ta-list">'
               + "".join(f"<li>{esc(r)}</li>" for r in audit.verdict.reasons) + "</ul></div>")
            ch = audit.challenge
            if ch:
                md('<div class="ta-section">Both sides</div>')
                md(f'<div class="ta-card accent tone-good"><h4>Strongest point for</h4>'
                   f'{cited(ch.strongest_point_for)}</div>')
                if ch.counterarguments:
                    # Prefer an evidence-backed counterargument over a generic risk.
                    top = next((c for c in ch.counterarguments if c.materiality == Materiality.SPECIFIC),
                               ch.counterarguments[0])
                    md(f'<div class="ta-card accent tone-bad"><h4>Strongest point against</h4>'
                       f'{cited(top.text)}</div>')

    # ---- claims
    with tab_claims:
        md('<div class="ta-muted ta-muted-block">Your thesis broken into separate claims, each checked '
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
            s_label, s_tone = STATUS[r.status]
            tags = pill(s_label, s_tone)
            tags += tag("Core claim" if r.claim.importance == Importance.CORE else "Supporting claim")
            tags += tag(TYPE_LABEL.get(r.claim.type.value, r.claim.type.value))
            if not r.claim.stated_by_user:
                tags += tag("Unstated assumption")
            quotes = "".join(quote_block(ev) for ev in r.evidence)
            md(f'<div class="ta-card ta-claim tone-{s_tone}"><span class="ci">{TONE_ICON[s_tone]}</span>'
               f'<div class="cb"><div>{tags}</div><h4>{esc(r.claim.text)}</h4>'
               f'<div>{cited(r.reasoning)}</div>{quotes}</div></div>')
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
                badge = pill("Evidence-backed", "bad") if backed else pill("General risk", "neutral")
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
            md('<div class="ta-muted ta-mb8">Answering these helps you pressure-test your own '
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
                    body += '<ul class="ta-list ta-mt8">' + "".join(
                        f"<li>{esc(c)}</li>" for c in review.new_concerns) + "</ul>"
                md(f'<div class="ta-card accent tone-good"><h4>Feedback on your answers</h4>{body}</div>')

    # ---- sources & audit trail
    with tab_sources:
        md('<div class="ta-section">Sources cited</div>')
        if audit.sources:
            st.dataframe(
                [{
                    "Ref": s.label,
                    "Company": s.ticker,
                    "Relation": relation_of(s),
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
                md('<div class="ta-muted ta-mb6">Problems our code caught in the model\'s '
                   'citations and fixed or flagged before showing you the result.</div>')
                md('<ul class="ta-list ta-muted">' + "".join(f"<li>{esc(i)}</li>" for i in audit.citation_issues) + "</ul>")
            else:
                st.caption("No citation problems found.")

        sc = audit.summary_check
        if sc:
            md('<div class="ta-section">Independent check of the summary</div>')
            md('<div class="ta-muted ta-mb6">A second, separate checker re-reads every sentence of '
               'the plain-English summary against the sources it cites'
               + ("" if sc.support_checked else " (citation and number checks only; source re-reading needs Google Cloud)")
               + ".</div>")
            rows = ""
            for s in sc.sentences:
                if s.status == "no_claim" and not analyst():
                    continue
                label, tone = CHECK_LABEL.get(s.status, (s.status, "neutral"))
                missing = ("<div class='ta-muted'>Not found in sources: " + esc("; ".join(s.unsupported_parts)) + "</div>"
                           if s.unsupported_parts else "")
                note = f"<div class='ta-muted'>{esc(s.note)}</div>" if s.note and analyst() else ""
                chips = "".join(cited(f"[{c}]") for c in s.cited)
                rows += f'<div class="ta-mb8">{pill(label, tone)}{esc(s.sentence)} {chips}{missing}{note}</div>'
            md(f'<div class="ta-card">{rows or "<span class=ta-muted>No sentences to check.</span>"}</div>')

        nc = audit.number_check
        if audit.number_check_issues or (nc and nc.sec_confirmed) or analyst():
            md('<div class="ta-section">Number checks</div>')
            if audit.number_check_issues:
                md('<div class="ta-muted ta-mb6">Numbers we couldn\'t find in the source they cite. '
                   'Treat these with caution.</div>')
                md('<ul class="ta-list ta-muted">' + "".join(f"<li>{esc(i)}</li>" for i in audit.number_check_issues) + "</ul>")
            else:
                st.caption("Every cited number was found in its source.")
            if nc and nc.sec_confirmed:
                md('<div class="ta-muted ta-mb6">'
                   f'{pill("Matches official SEC data", "good")}'
                   "These dollar figures also match the company's official numbers filed with the SEC.</div>")
                md('<ul class="ta-list ta-muted">' + "".join(f"<li>{esc(c)}</li>" for c in nc.sec_confirmed) + "</ul>")
            elif nc and nc.sec_checked:
                st.caption("Official SEC figures were checked; no dollar figure in this audit matched one directly.")
            if analyst() and nc and nc.sec_error and own_ticker not in ("EXMP", "RVMC"):
                st.caption(f"SEC comparison not run: {nc.sec_error}")

        if analyst():
            md('<div class="ta-section">Audit record</div>')
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="ta-muted">Audit ID</div><code>{esc(audit.audit_id)}</code>', unsafe_allow_html=True)
            c2.markdown(f'<div class="ta-muted">Model</div><code>{esc(audit.model)}</code>', unsafe_allow_html=True)
            c3.markdown(f'<div class="ta-muted">Verdict rule</div><code>{esc(audit.verdict.rule)}</code>', unsafe_allow_html=True)
            rec_id = getattr(auditor().store, "recommendation_id", lambda _id: None)(audit.audit_id)
            if rec_id:
                st.caption(f"Saved to the track record as {rec_id} (see the Track record page).")
            st.download_button(
                "Download audit (JSON)", audit.model_dump_json(indent=2),
                file_name=f"thesis-audit-{audit.audit_id}.json", mime="application/json",
            )

    demo = " EXMP and RVMC are fictional demo companies." if own_ticker in ("EXMP", "RVMC") else ""
    md(f'<div class="ta-disclaimer">{esc(audit.disclaimer)}{demo}</div>')
