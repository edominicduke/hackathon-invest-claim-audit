"""Team boundaries. Each teammate implements one Protocol; the agent depends only on these.

* Guru      -> ``Retriever``      (the "R": filings, transcripts, papers, chunked + searchable)
* Abdullah  -> ``AuditStore``     (saves audits, Challenge My Trade answers, track record)
* Bogdan    -> ``NumberChecker``  (every number the agent writes must appear in the source)

Stand-in implementations live in ``agent.stubs`` so this part runs before theirs land.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel, Field

from .schemas import ThesisAudit, UserAnswer


class Chunk(BaseModel):
    """One searchable piece of a source document, with the label Guru attaches."""

    chunk_id: str
    ticker: str
    doc_type: str                    # "10-K", "10-Q", "8-K", "transcript", "paper", "patent"
    text: str
    period: Optional[str] = None     # "FY2025", "Q1 FY2026" - critical for period checks
    section: Optional[str] = None    # "MD&A", "Risk Factors", ...
    filed_date: Optional[str] = None
    url: Optional[str] = None
    # Set on chunks from *other* companies' documents that are relevant to these tickers
    # (e.g. a competitor's 10-K). The agent marks them COMPETITOR when shown to the model.
    related_tickers: list[str] = Field(default_factory=list)


class Retriever(Protocol):
    def search(self, query: str, *, ticker: str, k: int = 4) -> list[Chunk]:
        """Return the k most relevant chunks for ``query`` about ``ticker`` (its own
        documents, plus competitor documents tagged with ``related_tickers``).

        The agent owns *what* to ask (including adversarial queries); the retriever
        owns *how* to find it (chunking, ranking, filters).
        """
        ...


class AuditStore(Protocol):
    def save_audit(self, audit: ThesisAudit) -> None: ...

    def save_answers(self, audit_id: str, answers: list[UserAnswer]) -> None: ...


class NumberChecker(Protocol):
    def check(self, audit: ThesisAudit, chunks: dict[str, Chunk]) -> list[str]:
        """Return human-readable issues, one per number not found in its cited source.

        ``chunks`` is keyed by audit label ("S3"). Numbers appear in ``Evidence.quote``
        and in prose fields as sentences ending in [S#] labels.
        """
        ...
