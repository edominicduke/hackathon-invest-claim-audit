"""The agent: a fixed pipeline with one bounded agentic step (Stage 2).

    parse -> decompose -> [per claim: balanced retrieval -> judge -> <=2 follow-up searches
    -> code citation checks] -> entailment check -> challenge -> rule-based verdict
    -> plain-language explanation -> glossary -> number check -> store

Why a pipeline and not an autonomous loop: the product's value is that it *always*
searches for counter-evidence. A fixed skeleton guarantees that; an agent deciding
when it's "done" could stop after confirming searches. Autonomy is limited to the one
place it helps: when a claim is unverifiable, the model may pick up to 2 more queries.

Citations: every retrieved chunk gets an audit-wide label (S1, S2, ...) the first time
it is seen. Per-call numbering would make [S1] mean different sources in different
steps, silently corrupting citations once outputs are combined.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Lock
from typing import Callable, Optional

from . import prompts as P
from .citations import check_evidence, check_inline, labels_in, normalize
from .glossary import audit_text, find_terms
from .interfaces import AuditStore, Chunk, NumberChecker, Retriever
from .llm import LLM, LLMError
from .schemas import (
    AnswerReview,
    Challenge,
    Claim,
    ClaimJudgment,
    ClaimResult,
    ClaimStatus,
    ClaimType,
    Decomposition,
    EntailmentReport,
    Evidence,
    Explanation,
    Materiality,
    ParsedThesis,
    SourceRef,
    ThesisAudit,
    UserAnswer,
    UserQuestion,
    Verdict,
    VerdictLevel,
)
from .tickers import lookup
from .verdict import compute_verdict

Progress = Callable[[str], None]
log = logging.getLogger("agent.pipeline")

MAX_CLAIMS = 7
MAX_QUERIES_PER_SIDE = 2
CHALLENGE_SOURCE_LIMIT = 25


class SourceRegistry:
    """Assigns stable audit-wide labels to chunks. Thread-safe (claims run in parallel)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._label_of: dict[str, str] = {}
        self.by_label: dict[str, Chunk] = {}

    def label(self, chunk: Chunk) -> str:
        with self._lock:
            if chunk.chunk_id not in self._label_of:
                label = f"S{len(self._label_of) + 1}"
                self._label_of[chunk.chunk_id] = label
                self.by_label[label] = chunk
            return self._label_of[chunk.chunk_id]


def _label_key(label: str) -> int:
    return int(label[1:]) if label[1:].isdigit() else 0


def format_sources(sources: dict[str, Chunk], ticker: str) -> str:
    """Wrap chunks in labelled tags. Other companies' chunks are marked COMPETITOR."""
    blocks = []
    for label in sorted(sources, key=_label_key):
        c = sources[label]
        attrs = f'label="{label}" company="{c.ticker}"'
        if c.ticker.upper() != (ticker or "").upper():
            attrs += ' relation="COMPETITOR"'
        attrs += f' doc_type="{c.doc_type}" period="{c.period or "unknown"}"'
        if c.section:
            attrs += f' section="{c.section}"'
        if c.filed_date:
            attrs += f' filed="{c.filed_date}"'
        blocks.append(f"<source {attrs}>\n{c.text}\n</source>")
    return "\n\n".join(blocks) if blocks else "(no sources found)"


def apply_status_rules(claim: Claim, status: ClaimStatus, evidence: list[Evidence]) -> tuple[ClaimStatus, list[str]]:
    """Code enforcement of the judging rules. Returns (status, issues)."""
    issues = []
    if status == ClaimStatus.SUPPORTED and claim.type == ClaimType.FORECAST:
        issues.append("Forecasts can't be confirmed by filings; only whether their assumptions hold")
        status = ClaimStatus.UNVERIFIABLE
    if status == ClaimStatus.SUPPORTED and claim.type == ClaimType.VAGUE:
        issues.append("Too vague to confirm as stated")
        status = ClaimStatus.UNVERIFIABLE
    if status == ClaimStatus.SUPPORTED and not any(e.stance == "supports" for e in evidence):
        issues.append("Marked supported, but no valid supporting citation remained")
        status = ClaimStatus.UNVERIFIABLE
    if status == ClaimStatus.CONTRADICTED and not any(e.stance == "contradicts" for e in evidence):
        issues.append("Marked contradicted, but no valid contradicting citation remained")
        status = ClaimStatus.UNVERIFIABLE
    return status, issues


class ThesisAuditor:
    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        store: Optional[AuditStore] = None,
        number_checker: Optional[NumberChecker] = None,
        *,
        k: int = 4,
        max_follow_ups: int = 2,
        max_workers: int = 4,
        verify_entailment: bool = True,
    ):
        self.llm = llm
        self.retriever = retriever
        self.store = store
        self.number_checker = number_checker
        self.k = k
        self.max_follow_ups = max_follow_ups
        self.max_workers = max_workers
        self.verify_entailment = verify_entailment

    # ------------------------------------------------------------------ stage 1

    def parse(self, raw: str) -> ParsedThesis:
        """Free text -> structured fields. The UI shows the result for confirmation."""
        user = f"Today's date: {date.today()}\n\n<thesis>\n{raw}\n</thesis>"
        parsed = self.llm.structured(system=P.EXTRACT, user=user, output=ParsedThesis, effort="low")
        if parsed.ticker:
            parsed.ticker = parsed.ticker.strip().upper()
            name = lookup(parsed.ticker)
            if name is None:
                parsed.ambiguities.append(f"'{parsed.ticker}' isn't in our ticker list - please check it")
            elif not parsed.company_name:
                parsed.company_name = name
        return parsed

    # ------------------------------------------------------------------ stage 3

    def decompose(self, raw: str, parsed: ParsedThesis) -> list[Claim]:
        user = (
            f"Today's date: {date.today()}\n"
            f"Company: {parsed.ticker} ({parsed.company_name or 'name unknown'})\n"
            f"Direction: {parsed.direction.value}\n"
            f"Horizon: {f'{parsed.horizon_months} months' if parsed.horizon_months else 'not stated'}\n\n"
            f"<thesis>\n{raw}\n</thesis>"
        )
        draft = self.llm.structured(system=P.DECOMPOSE, user=user, output=Decomposition, effort="high")
        claims = []
        for i, d in enumerate(draft.claims[:MAX_CLAIMS], start=1):
            claim = Claim(id=f"c{i}", **d.model_dump())
            # Enforce balanced retrieval: same number of supporting and challenging queries.
            n = max(1, min(len(claim.support_queries), len(claim.challenge_queries), MAX_QUERIES_PER_SIDE))
            claim.support_queries = claim.support_queries[:n] or [claim.text]
            claim.challenge_queries = claim.challenge_queries[:n] or [f"{claim.text} risk decline competition"]
            claims.append(claim)
        return claims

    # ------------------------------------------------------------- stages 3 + 4

    def _judge(self, claim: Claim, parsed: ParsedThesis, sources: dict[str, Chunk]) -> ClaimJudgment:
        user = (
            f"Today's date: {date.today()}\n"
            f"Company: {parsed.ticker} ({parsed.company_name or ''}); investor direction: {parsed.direction.value}\n"
            f"Claim ({claim.type.value}, {claim.importance.value}): {claim.text}\n\n"
            f"Sources:\n{format_sources(sources, parsed.ticker)}"
        )
        return self.llm.structured(system=P.JUDGE, user=user, output=ClaimJudgment, effort="high")

    def check_claim(
        self,
        claim: Claim,
        parsed: ParsedThesis,
        registry: Optional[SourceRegistry] = None,
        thesis: str = "",
    ) -> tuple[ClaimResult, dict[str, Chunk]]:
        """Retrieve, judge and verify one claim. Returns (result, sources keyed by label)."""
        registry = registry or SourceRegistry()
        sources: dict[str, Chunk] = {}
        searches: list[str] = []

        def search(query: str) -> int:
            searches.append(query)
            added = 0
            for c in self.retriever.search(query, ticker=parsed.ticker, k=self.k):
                label = registry.label(c)
                if label not in sources:
                    sources[label] = c
                    added += 1
            return added

        for q in claim.support_queries + claim.challenge_queries:
            search(q)

        if not sources:
            return (
                ClaimResult(
                    claim=claim,
                    status=ClaimStatus.UNVERIFIABLE,
                    llm_status=ClaimStatus.UNVERIFIABLE,
                    reasoning="No relevant sources were found for this claim.",
                    evidence=[],
                    searches=searches,
                ),
                sources,
            )

        log.info("   claim %s: judging against %d sources - %s", claim.id, len(sources), claim.text[:70])
        start = time.perf_counter()
        judgment = self._judge(claim, parsed, sources)
        # Bounded agentic step: the model may ask for up to N more searches.
        follow_ups = 0
        while (
            judgment.status == ClaimStatus.UNVERIFIABLE
            and judgment.follow_up_query
            and follow_ups < self.max_follow_ups
        ):
            follow_ups += 1
            log.info("   claim %s: unverifiable, follow-up search %d: %s", claim.id, follow_ups, judgment.follow_up_query)
            if search(judgment.follow_up_query) == 0:
                log.info("   claim %s: follow-up found nothing new, stopping", claim.id)
                break
            judgment = self._judge(claim, parsed, sources)

        valid, issues = check_evidence(judgment.evidence, sources)
        status, rule_issues = apply_status_rules(claim, judgment.status, valid)
        log.info(
            "   claim %s -> %s (%.1fs, %d sources, %d follow-up searches): %s",
            claim.id, status.value, time.perf_counter() - start, len(sources), follow_ups, claim.text[:60],
        )
        reasoning, inline_issues = check_inline(judgment.reasoning, set(sources), ignore_numbers_from=thesis)
        result = ClaimResult(
            claim=claim,
            status=status,
            llm_status=judgment.status,
            reasoning=reasoning,
            evidence=valid,
            issues=issues + rule_issues + inline_issues,
            searches=searches,
        )
        return result, sources

    def _check_entailment(self, results: list[ClaimResult], pool: dict[str, Chunk], ticker: str) -> None:
        """Second-pass check that each real quote actually bears on its claim."""
        index: list[tuple[int, int]] = []
        items = []
        for ri, r in enumerate(results):
            if r.status == ClaimStatus.UNVERIFIABLE:
                continue
            for ei, ev in enumerate(r.evidence):
                c = pool[ev.source]
                company = c.ticker + (" (COMPETITOR)" if c.ticker.upper() != ticker.upper() else "")
                items.append(
                    f'<item index="{len(index)}">\nClaim: {r.claim.text}\nCited as: {ev.stance}\n'
                    f"Source: {company} {c.doc_type}, period {c.period or 'unknown'}, "
                    f"section {c.section or 'unknown'}\n"
                    f'Quote: "{ev.quote}"\n</item>'
                )
                index.append((ri, ei))
        if not items:
            return
        report = self.llm.structured(
            system=P.ENTAILMENT, user="\n\n".join(items), output=EntailmentReport, effort="medium"
        )
        failed: dict[int, set[int]] = {}
        for item in report.items:
            if not item.entails and 0 <= item.index < len(index):
                ri, ei = index[item.index]
                failed.setdefault(ri, set()).add(ei)
                results[ri].issues.append(
                    f"Dropped citation [{results[ri].evidence[ei].source}]: {item.problem or 'does not support the claim'}"
                )
        for ri, bad in failed.items():
            r = results[ri]
            r.evidence = [ev for ei, ev in enumerate(r.evidence) if ei not in bad]
            r.status, extra = apply_status_rules(r.claim, r.status, r.evidence)
            r.issues += extra

    # ------------------------------------------------------------------ stage 9

    def _challenge(
        self, raw: str, parsed: ParsedThesis, results: list[ClaimResult], pool: dict[str, Chunk]
    ) -> tuple[Challenge, list[str]]:
        cited = {ev.source for r in results for ev in r.evidence} | {l for r in results for l in labels_in(r.reasoning)}
        ordered = sorted(cited, key=_label_key) + sorted(set(pool) - cited, key=_label_key)
        sources = {label: pool[label] for label in ordered[:CHALLENGE_SOURCE_LIMIT] if label in pool}
        claim_lines = "\n".join(
            f"- [{r.claim.importance.value} {r.claim.type.value}] {r.claim.text} -> {r.status.value}: {r.reasoning}"
            for r in results
        )
        clarifying = [r.claim.clarifying_question for r in results if r.claim.clarifying_question]
        user = (
            f"Today's date: {date.today()}\n"
            f"Parsed thesis: {parsed.model_dump_json()}\n\n<thesis>\n{raw}\n</thesis>\n\n"
            f"Claim results:\n{claim_lines}\n\n"
            f"Clarifying questions to include: {clarifying or 'none'}\n\n"
            f"Sources:\n{format_sources(sources, parsed.ticker)}"
        )
        challenge = self.llm.structured(system=P.CHALLENGE, user=user, output=Challenge, effort="high")

        allowed = set(sources)
        issues: list[str] = []

        def clean(text: str) -> str:
            text, found = check_inline(text, allowed, ignore_numbers_from=raw)
            issues.extend(found)
            return text

        challenge.strongest_point_for = clean(challenge.strongest_point_for)
        challenge.stress_test.scenario = clean(challenge.stress_test.scenario)
        for ca in challenge.counterarguments:
            ca.text = clean(ca.text)
            ca.evidence, _ = check_evidence(ca.evidence, sources)
            # "Specific" must be earned with a valid citation.
            if ca.materiality == Materiality.SPECIFIC and not ca.evidence:
                ca.materiality = Materiality.GENERIC
        asked = {normalize(q.question) for q in challenge.questions_for_user}
        for q in clarifying:
            if normalize(q) not in asked:
                challenge.questions_for_user.append(
                    UserQuestion(question=q, why_it_matters="This claim is too vague to check until it's pinned down.")
                )
        return challenge, issues

    def _explain(
        self, raw: str, verdict: Verdict, results: list[ClaimResult], challenge: Optional[Challenge]
    ) -> tuple[Explanation, list[str]]:
        lines = [f"Verdict: {verdict.level.value} ({'; '.join(verdict.reasons)})", "Claims:"]
        lines += [f"- {r.claim.text} -> {r.status.value}: {r.reasoning}" for r in results]
        if challenge:
            lines.append(f"Strongest point for: {challenge.strongest_point_for}")
            lines += [f"Counterargument ({ca.materiality.value}): {ca.text}" for ca in challenge.counterarguments[:3]]
        user = "\n".join(lines)
        explanation = self.llm.structured(system=P.EXPLAIN, user=user, output=Explanation, effort="low")
        # The summary may only carry labels that were already in the audit it summarises.
        explanation.plain_summary, issues = check_inline(
            explanation.plain_summary, labels_in(user), ignore_numbers_from=raw
        )
        return explanation, issues

    # --------------------------------------------------------------- entrypoint

    def audit(self, raw: str, parsed: Optional[ParsedThesis] = None, on_progress: Optional[Progress] = None) -> ThesisAudit:
        audit_start = time.perf_counter()

        def progress(msg: str) -> None:
            log.info("== %s (t=%.0fs)", msg, time.perf_counter() - audit_start)
            if on_progress:
                on_progress(msg)
        if parsed is None:
            progress("Reading your thesis")
            parsed = self.parse(raw)

        if not parsed.is_auditable or not parsed.ticker:
            audit = ThesisAudit(
                raw_input=raw,
                parsed=parsed,
                status="not_auditable",
                verdict=Verdict(
                    level=VerdictLevel.NONE,
                    rule="not_auditable",
                    reasons=[parsed.not_auditable_reason or "No public company ticker was identified"],
                ),
                model=self.llm.model,
            )
            self._save(audit)
            return audit

        progress("Breaking the thesis into checkable claims")
        claims = self.decompose(raw, parsed)

        progress(f"Checking {len(claims)} claims against the filings")
        registry = SourceRegistry()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            outcomes = list(ex.map(lambda c: self.check_claim(c, parsed, registry, raw), claims))
        results = [r for r, _ in outcomes]
        pool = dict(registry.by_label)

        warnings: list[str] = []
        citation_issues: list[str] = []
        if self.verify_entailment:
            progress("Double-checking that each quote really backs its claim")
            try:
                self._check_entailment(results, pool, parsed.ticker)
            except LLMError as exc:
                warnings.append(f"Citation double-check skipped: {exc}")

        progress("Building the case against your trade")
        challenge = None
        try:
            challenge, found = self._challenge(raw, parsed, results, pool)
            citation_issues += found
        except LLMError as exc:
            warnings.append(f"Challenge step failed: {exc}")

        verdict = compute_verdict(results)

        progress("Writing the plain-language summary")
        explanation = None
        try:
            explanation, found = self._explain(raw, verdict, results, challenge)
            citation_issues += found
        except LLMError as exc:
            warnings.append(f"Summary step failed: {exc}")

        # Every label that appears anywhere in the audit goes into the source table.
        prose = [r.reasoning for r in results]
        cited = {ev.source for r in results for ev in r.evidence}
        if challenge:
            cited |= {ev.source for ca in challenge.counterarguments for ev in ca.evidence}
            prose += [challenge.strongest_point_for, challenge.stress_test.scenario]
            prose += [ca.text for ca in challenge.counterarguments]
        if explanation:
            prose.append(explanation.plain_summary)
        cited |= {label for text in prose for label in labels_in(text)}
        sources = [
            SourceRef(label=label, **pool[label].model_dump(include=set(SourceRef.model_fields) - {"label"}))
            for label in sorted(cited & set(pool), key=_label_key)
        ]

        audit = ThesisAudit(
            raw_input=raw,
            parsed=parsed,
            status="complete",
            claims=results,
            challenge=challenge,
            verdict=verdict,
            explanation=explanation,
            sources=sources,
            retrieved_chunk_ids=sorted(c.chunk_id for c in pool.values()),
            citation_issues=citation_issues,
            warnings=warnings,
            model=self.llm.model,
        )
        audit.glossary = find_terms(re.sub(r"\[S\d+\]", "", audit_text(audit)))
        if self.number_checker:
            audit.number_check_issues = self.number_checker.check(audit, pool)
        self._save(audit)
        log.info("== Audit done: %s in %.0fs", verdict.level.value, time.perf_counter() - audit_start)
        return audit

    def review_answers(self, audit: ThesisAudit, answers: list[UserAnswer]) -> AnswerReview:
        """Feature #9 follow-up. Answers are saved and reviewed, but never change the
        verdict: the user's statements are not evidence about the company."""
        if self.store:
            self.store.save_answers(audit.audit_id, answers)
        qa = "\n".join(f"Q: {a.question}\nA: {a.answer}" for a in answers)
        claims = "\n".join(f"- {r.claim.text} -> {r.status.value}" for r in audit.claims)
        user = (
            f"Thesis:\n<thesis>\n{audit.raw_input}\n</thesis>\n\nVerdict: {audit.verdict.level.value}\n"
            f"Claims:\n{claims}\n\nInvestor's answers:\n{qa}"
        )
        return self.llm.structured(system=P.REVIEW_ANSWERS, user=user, output=AnswerReview, effort="low")

    def _save(self, audit: ThesisAudit) -> None:
        if self.store:
            self.store.save_audit(audit)
