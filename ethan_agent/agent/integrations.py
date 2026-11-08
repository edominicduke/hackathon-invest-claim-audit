"""Adapters that plug the team's ``stock-agent/`` code into the agent's interfaces.

    Guru      search.search / search_competitors  -> StockAgentRetriever      (Retriever)
              verify.verify_citations             -> StockAgentCitationVerifier (CitationVerifier)
    Bogdan    number_check.check_numbers          -> StockAgentNumberChecker  (NumberChecker)
    Abdullah  app.store.Storage                   -> StockAgentStore          (AuditStore)

Nothing here changes how the agent reasons: each adapter only translates formats.
Division of labour on citations (agreed with Guru): the agent's strict, quote-level
checks run inside the pipeline; Guru's check runs once over the finished summary.

Environment variables (all optional):
    STOCK_AGENT_DIR          where stock-agent/ lives (default: next to ethan_agent/)
    STOCK_AGENT_DATA_DIR     Abdullah's storage folder (default: stock-agent/data, the folder
                             `python -m src.checks.results_check --data-dir data` reads)
    STOCK_AGENT_SEARCH_MODE  hybrid | keyword (default: hybrid if GOOGLE_CLOUD_PROJECT is set)
    AUDITOR_COMPETITOR_K     competitor chunks added per search (default 1, 0 = off)
    AUDITOR_INCLUDE_PAPERS   1: let research papers into searches (default off: they can't
                             confirm company facts and crowd out filings on broad queries)
    AUDITOR_SUPPORT_CHECK    1/0: run Guru's Vertex support check on the summary
                             (default: on if GOOGLE_CLOUD_PROJECT is set)
    AUDITOR_SEC_NUMBERS      compare numbers to official SEC figures (default: on when
                             SEC_USER_AGENT = "Team name your@email" is set; 0 = off)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from .citations import labels_in
from .interfaces import Chunk, Retriever
from .schemas import NumberCheckReport, SentenceCheck, SummaryCheck, ThesisAudit, UserAnswer, VerdictLevel
from .stubs import JsonFileStore

log = logging.getLogger("agent.integrations")

REPO_ROOT = Path(__file__).resolve().parents[2]
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")


def stock_agent_dir() -> Path:
    return Path(os.environ.get("STOCK_AGENT_DIR") or REPO_ROOT / "stock-agent").resolve()


def _ensure_importable() -> None:
    """stock-agent's code is a package called ``src`` under stock-agent/."""
    root = stock_agent_dir()
    if not (root / "src").is_dir():
        raise ImportError(f"stock-agent code not found at {root} (set STOCK_AGENT_DIR)")
    if str(root) not in sys.path:
        sys.path.append(str(root))


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _cloud_configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


def pin_guru_to_vertex() -> bool:
    """Make Guru's embedding search and citation-check model calls use the Google Cloud
    project, whatever the agent's own LLM uses.

    Guru's code builds its client from environment variables. Without
    GOOGLE_GENAI_USE_VERTEXAI=True (which the agent may leave unset so its own calls use
    a Gemini API key), his calls would silently go to that API key instead of the project.
    Returns True if pinned.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return False
    from google import genai
    from src.ingest import embed as guru_embed

    with guru_embed._client_lock:
        if guru_embed._client is None:
            guru_embed._client = genai.Client(
                vertexai=True, project=project, location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
            )
    return True


# ------------------------------------------------------------------- Guru: search


def hit_to_chunk(hit: dict, audited: str) -> Chunk:
    """Guru's search hit -> the agent's Chunk."""
    date = hit.get("date") or ""
    section = hit.get("section")
    related = []
    if hit.get("competitor_of"):
        related = [hit["competitor_of"]]
    elif hit.get("doc_type") == "paper":
        related = list(hit.get("tags") or [])
    return Chunk(
        chunk_id=hit["chunk_id"],
        ticker=hit["ticker"],
        doc_type=hit["doc_type"],
        text=hit["text"],
        period=hit.get("period") or (date if not _DATE.match(date) else None),
        section=None if section == "full" else section,
        filed_date=date if _DATE.match(date) else None,
        url=hit.get("url"),
        related_tickers=related if hit["ticker"].upper() != audited.upper() else [],
    )


class StockAgentRetriever:
    """Guru's hybrid search over real SEC filings, transcripts and papers.

    Each search returns the company's own chunks plus ``competitor_k`` chunks from its
    rivals. Tickers that aren't in Guru's index (e.g. the fictional EXMP demo company)
    go to ``fallback``.
    """

    def __init__(self, fallback: Optional[Retriever] = None, *, competitor_k: Optional[int] = None,
                 mode: Optional[str] = None, include_papers: Optional[bool] = None):
        _ensure_importable()
        from src.ingest import search as guru_search

        self._search = guru_search
        if pin_guru_to_vertex():
            log.info("Guru's search uses Google Cloud project %s", os.environ["GOOGLE_CLOUD_PROJECT"])
        self.fallback = fallback
        self.competitor_k = int(os.environ.get("AUDITOR_COMPETITOR_K", 1)) if competitor_k is None else competitor_k
        self.mode = mode or os.environ.get("STOCK_AGENT_SEARCH_MODE") or ("hybrid" if _cloud_configured() else "keyword")
        # Research papers match broad queries strongly but can't confirm anything about the
        # company, so by default they don't take slots from filings and transcripts.
        self.include_papers = _flag("AUDITOR_INCLUDE_PAPERS", False) if include_papers is None else include_papers
        self.doc_types = None if self.include_papers else [d for d in guru_search.DOC_TYPES if d != "paper"]
        info = guru_search.describe_index()
        self.tickers = sorted(t for t in info["tickers"] if t != "RESEARCH")
        # Display only (landing-page coverage badge): newest document date across the covered companies.
        self.data_through = max((v["dates"][1] for t, v in info["tickers"].items() if t != "RESEARCH"), default=None)
        log.info("Retriever: Guru's index (%s search%s) covers %s", self.mode,
                 "" if self.include_papers else ", papers off", ", ".join(self.tickers))

    def covers(self, ticker: str) -> bool:
        return (ticker or "").upper() in self.tickers

    def search(self, query: str, *, ticker: str, k: int = 4) -> list[Chunk]:
        t = (ticker or "").upper()
        if not self.covers(t):
            return self.fallback.search(query, ticker=t, k=k) if self.fallback else []
        try:
            hits = self._search.search(query, tickers=[t], doc_types=self.doc_types, k=k, mode=self.mode)
            if self.competitor_k:
                hits += self._search.search_competitors(t, query, k=self.competitor_k, mode=self.mode)
        except Exception as exc:  # a failed search reads as "no evidence", never crashes the audit
            log.warning("Search failed for %r: %s", query, exc)
            return []
        return [hit_to_chunk(h, t) for h in hits]


# ------------------------------------------------------- Guru: final citation check


def _citation(c: Chunk) -> str:
    parts = [c.ticker, c.doc_type, f"(filed {c.filed_date})" if c.filed_date else c.period or "", c.section or ""]
    return " ".join(p for p in parts if p)


class StockAgentCitationVerifier:
    """Guru's sentence-by-sentence check, run once over the plain-language summary.

    Without Google Cloud it runs the structural checks only (invented labels, numbers
    with no citation, long claims with no citation). With it (or AUDITOR_SUPPORT_CHECK=1)
    it also asks Vertex whether each cited sentence is supported by its sources.
    """

    def __init__(self, check_support: Optional[bool] = None):
        _ensure_importable()
        from src.ingest import verify as guru_verify

        self._verify = guru_verify
        pin_guru_to_vertex()  # its fact-by-fact second opinion calls Gemini through Guru's client
        self.check_support = _flag("AUDITOR_SUPPORT_CHECK", _cloud_configured()) if check_support is None else check_support

    def verify(self, text: str, sources: dict[str, Chunk]) -> SummaryCheck:
        srcs = {label: {"citation": _citation(c), "text": c.text} for label, c in sources.items()}
        report = self._verify.verify_citations(text, srcs, check_support=self.check_support)
        return SummaryCheck(
            ok=report["ok"],
            support_checked=self.check_support,
            counts=report["counts"],
            sentences=[
                SentenceCheck(
                    sentence=s["sentence"],
                    cited=s["cited"],
                    status=s["status"],
                    unsupported_parts=s.get("unsupported_parts") or [],
                    note=s.get("support_error") or s.get("judge_error"),
                )
                for s in report["sentences"]
            ],
        )


# ------------------------------------------------------------ Bogdan: number check


def prose_fields(audit: ThesisAudit) -> Iterable[tuple[str, str]]:
    """(where, text) for every model-written text that can carry [S#] citations."""
    for r in audit.claims:
        yield f"Claim {r.claim.id} reasoning", r.reasoning
    ch = audit.challenge
    if ch:
        yield "Strongest point for", ch.strongest_point_for
        yield "Stress test", ch.stress_test.scenario
        for x in ch.stress_test.what_would_have_to_be_true + ch.stress_test.early_warning_signs:
            yield "Stress test", x
        for i, ca in enumerate(ch.counterarguments, 1):
            yield f"Counterargument {i}", ca.text
    if audit.explanation:
        yield "Summary", audit.explanation.plain_summary
        for t in audit.explanation.key_takeaways:
            yield "Key takeaway", t


class StockAgentNumberChecker:
    """Bogdan's number check: every number in a quote or a cited sentence must appear in
    the source(s) it cites (rounding allowed). Numbers the investor wrote in their own
    thesis are allowed too.

    With official SEC figures on (default when SEC_USER_AGENT holds a contact email;
    AUDITOR_SEC_NUMBERS=0 turns it off), dollar amounts are also compared with the
    company's XBRL numbers from SEC EDGAR: matches are listed as confirmations, and a
    number missing from its cited source but matching an SEC figure gets a softer note.
    """

    THESIS = "your thesis"
    SEC_KINDS = {"money", "amount"}  # percentages and plain counts aren't in SEC data

    def __init__(self, use_sec: Optional[bool] = None, sec=None):
        _ensure_importable()
        from src.checks import number_check as bogdan
        from src.ingest.verify import split_sentences

        self._n = bogdan
        self._split = split_sentences
        self._sec = sec
        if self._sec is None:
            default = "@" in os.environ.get("SEC_USER_AGENT", "")
            if _flag("AUDITOR_SEC_NUMBERS", default) if use_sec is None else use_sec:
                from src.checks.sec_facts import SecFacts

                self._sec = SecFacts()
        self._sec_sources: dict[str, list[dict]] = {}

    @property
    def uses_sec(self) -> bool:
        return self._sec is not None

    def _official(self, ticker: str) -> list[dict]:
        """Official SEC numbers for the ticker, downloaded once per ticker. Raises SecError."""
        if ticker not in self._sec_sources:
            self._sec_sources[ticker] = self._sec.as_sources(ticker)
        return self._sec_sources[ticker]

    def check(self, audit: ThesisAudit, chunks: dict[str, Chunk]) -> NumberCheckReport:
        ticker = (audit.parsed.ticker or "").upper()
        thesis = {"label": self.THESIS, "text": audit.raw_input}
        report = NumberCheckReport()
        official: list[dict] = []
        if self._sec is not None and ticker:
            try:
                official = self._official(ticker)
                report.sec_checked = True
                log.info("Number check: comparing with %d official SEC figures for %s", len(official), ticker)
            except Exception as exc:  # no contact email, offline, ticker unknown to the SEC
                report.sec_error = str(exc)
                log.info("Number check: SEC comparison skipped (%s)", exc)

        def src(label: str) -> Optional[dict]:
            c = chunks.get(label)
            return {"label": label, "text": c.text} if c else None

        def run(text: str, sources: list[dict], where: str, cited: list[str], quote: bool) -> None:
            result = self._n.check_numbers(text, sources)
            sec_hits = {}
            if official:
                for n in self._n.check_numbers(text, official)["numbers"]:
                    if n["found"] and n["kind"] in self.SEC_KINDS:
                        sec_hits[n["text"]] = n["source"]
                        report.sec_confirmed.append(f"{n['text']} matches {n['source'].removeprefix('SEC ')}")
            for n in result["numbers"]:
                if n["found"]:
                    continue
                refs = ", ".join(cited)
                where_from = f"the quote from [{refs}]" if quote else f"the cited source{'s' if len(cited) > 1 else ''} [{refs}]"
                if n["text"] in sec_hits:
                    report.issues.append(f"{where}: {n['text']} isn't in {where_from}, but it matches an official "
                                         f"SEC figure ({sec_hits[n['text']].removeprefix('SEC ')})")
                else:
                    report.issues.append(f"{where}: {n['text']} isn't in {where_from}")

        quotes = [(f"Claim {r.claim.id}", ev) for r in audit.claims for ev in r.evidence]
        if audit.challenge:
            quotes += [(f"Counterargument {i}", ev)
                       for i, ca in enumerate(audit.challenge.counterarguments, 1) for ev in ca.evidence]
        for where, ev in quotes:
            s = src(ev.source)
            if s:
                run(ev.quote, [s], where, [ev.source], quote=True)

        for where, text in prose_fields(audit):
            for sentence in self._split(text or ""):
                cited = sorted(labels_in(sentence))
                if not cited:
                    continue  # uncited numbers are already flagged by the agent's own check
                run(sentence, [s for s in map(src, cited) if s] + [thesis], where, cited, quote=False)

        report.issues = list(dict.fromkeys(report.issues))
        report.sec_confirmed = list(dict.fromkeys(report.sec_confirmed))
        return report


# --------------------------------------------------------------- Abdullah: storage

VERDICT_TO_STORE = {
    VerdictLevel.GREEN: "green",
    VerdictLevel.YELLOW: "yellow",
    VerdictLevel.RED: "red",
    VerdictLevel.NONE: "not_enough_evidence",
}


def _cap(text: Optional[str], n: int) -> Optional[str]:
    text = (text or "").strip()
    return text[:n] if text else None


def recommendation_payload(audit: ThesisAudit) -> dict:
    """ThesisAudit -> Abdullah's POST /api/recommendations body.

    The verdict is strength of evidence for the thesis (team decision), not a buy call;
    the results check compares it with the S&P 500 only as a long-run reference.
    """
    sources = []
    for s in audit.sources[:50]:
        filed = f" (filed {s.filed_date})" if s.filed_date else f" ({s.period})" if s.period else ""
        sources.append({
            "label": _cap(f"{s.label} {s.ticker} {s.doc_type}{filed}{', ' + s.section if s.section else ''}", 120),
            "document": _cap(s.doc_type, 120),
            "date": _cap(s.filed_date or s.period, 20),
            "page": _cap(s.section, 20),
            "url": _cap(s.url, 500),
            "text": _cap(s.text, 4000),
        })
    summary = audit.explanation.plain_summary if audit.explanation else "; ".join(audit.verdict.reasons)
    return {
        "company": audit.parsed.ticker,
        "user_reason": _cap(audit.raw_input, 4000),
        "verdict": VERDICT_TO_STORE[audit.verdict.level],
        "summary": _cap(summary, 4000),
        "sources": [{k: v for k, v in s.items() if v is not None} for s in sources],
        "created_by": "thesis-auditor",
    }


def challenge_payload(audit: ThesisAudit) -> dict:
    """ThesisAudit.challenge -> Abdullah's POST /api/recommendations/{id}/challenges body."""
    ch = audit.challenge
    return {
        "bear_case": [t for t in (_cap(ca.text, 4000) for ca in ch.counterarguments[:50]) if t],
        "drop_reason": _cap(ch.stress_test.scenario, 4000),
        "verdict": VERDICT_TO_STORE[audit.verdict.level],
        "verdict_reason": _cap("; ".join(audit.verdict.reasons), 4000),
        "created_by": "thesis-auditor",
    }


class StockAgentStore:
    """Saves each audit twice: the full audit JSON (our archive, for replay and debugging)
    and a recommendation + Challenge My Trade session in Abdullah's storage, which feeds
    the track record. Fictional demo tickers are kept out of the track record."""

    def __init__(self, data_dir: Optional[str | Path] = None, archive: Optional[JsonFileStore] = None,
                 skip_tickers: Iterable[str] = (), *, record_price: Optional[bool] = None, prices=None):
        _ensure_importable()
        from src.app.store import Storage

        self.record_price = _flag("AUDITOR_RECORD_PRICE", True) if record_price is None else record_price
        self.prices = prices  # Bogdan's PriceBook (tests pass a fake one)
        self.data_dir = Path(data_dir or os.environ.get("STOCK_AGENT_DATA_DIR") or stock_agent_dir() / "data")
        self.storage = Storage(self.data_dir)
        self.archive = archive or JsonFileStore()
        self.skip = {t.upper() for t in skip_tickers}
        log.info("Store: track record in %s (skipping demo tickers %s)", self.data_dir, sorted(self.skip) or "none")

    def _link_path(self, audit_id: str) -> Path:
        return Path(self.archive.root) / "links" / f"{audit_id}.json"

    def save_audit(self, audit: ThesisAudit) -> None:
        self.archive.save_audit(audit)
        ticker = (audit.parsed.ticker or "").upper()
        if audit.status != "complete" or not ticker or ticker in self.skip:
            return
        rec = self.storage.recommendations.create(recommendation_payload(audit))
        link = {"recommendation_id": rec["id"], "challenge_id": None}
        if audit.challenge:
            link["challenge_id"] = self.storage.challenges.create(rec["id"], challenge_payload(audit))["id"]
        path = self._link_path(audit.audit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(link), encoding="utf-8")
        log.info("Saved to the track record: recommendation %s (%s %s)", rec["id"], ticker, rec["verdict"])
        if self.record_price:
            self._record_start_price(rec["id"], ticker)

    def _record_start_price(self, rec_id: str, ticker: str) -> None:
        """Bogdan's price on the audit date, so the track record shows it from day one.
        Best effort: the results check fills it in later anyway."""
        try:
            from datetime import date

            from src.checks.prices import PriceBook

            _, price = (self.prices or PriceBook()).get_close(ticker, date.today())
            self.storage.recommendations.update_results(rec_id, {}, price_on_date=price)
        except Exception as exc:  # offline, yfinance missing, market data delayed
            log.info("Start price for %s not recorded yet (%s); the results check will add it", ticker, exc)

    # ---- track record (read side): Abdullah's storage + Bogdan's results check

    def recommendation_id(self, audit_id: str) -> Optional[str]:
        path = self._link_path(audit_id)
        return json.loads(path.read_text(encoding="utf-8"))["recommendation_id"] if path.exists() else None

    def recommendations(self) -> list[dict]:
        """Every saved recommendation, newest first, each with its Challenge My Trade sessions."""
        recs = self.storage.recommendations.all()
        challenges = self.storage.challenges.all()
        for r in recs:
            r["challenges"] = [c for c in challenges if c["recommendation_id"] == r["id"]]
        return sorted(recs, key=lambda r: r.get("created_at", ""), reverse=True)

    def track_record(self) -> dict:
        """Bogdan's summary: accuracy and average excess return vs the S&P 500 per horizon."""
        from src.checks.results_check import track_record

        return track_record(self.storage.recommendations.all())

    def update_results(self, prices=None) -> dict:
        """Run Bogdan's results check now (downloads prices with yfinance). Only horizons
        that are due (30/91/182/365 days after the audit) are filled in."""
        from src.checks.results_check import update_all

        return update_all(self.storage.recommendations, prices=prices)

    def save_answers(self, audit_id: str, answers: list[UserAnswer]) -> None:
        self.archive.save_answers(audit_id, answers)
        path = self._link_path(audit_id)
        if not path.exists():
            return  # demo ticker or not saved to the track record
        link = json.loads(path.read_text(encoding="utf-8"))
        payload = {"answers": {a.question[:300]: a.answer[:4000] for a in answers[:20]}}
        if link.get("challenge_id"):
            self.storage.challenges.update(link["challenge_id"], payload)
        else:
            link["challenge_id"] = self.storage.challenges.create(link["recommendation_id"], payload)["id"]
            path.write_text(json.dumps(link), encoding="utf-8")
