"""Data contracts for the Thesis Auditor.

Two kinds of models live here:

* **LLM output models** (ParsedThesis, Decomposition, ClaimJudgment, ...) are the
  exact shapes the model must return. They are converted to strict JSON schemas in
  ``agent.llm`` so the API guarantees well-formed output. Keep them free of
  defaults and validators: every field is required (nullable where optional).
* **Record models** (ClaimResult, ThesisAudit, ...) are what the pipeline assembles
  after code-side checks. ``ThesisAudit`` is the hand-off object for the team:
  Abdullah stores it, Bogdan number-checks and scores it, the UI renders it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- enums


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    HOLD = "hold"
    UNCLEAR = "unclear"


class ClaimType(str, Enum):
    FACT = "fact"          # historical, checkable against filings
    CAUSAL = "causal"      # "X is driving Y" - partly checkable via mgmt commentary
    FORECAST = "forecast"  # about the future - never "supported", only stress-tested
    VAGUE = "vague"        # needs operationalising or a question to the user
    PREMISE = "premise"    # unstated assumption the thesis depends on


CHECKABLE_TYPES = {ClaimType.FACT, ClaimType.CAUSAL, ClaimType.PREMISE}


class Importance(str, Enum):
    CORE = "core"              # thesis falls apart without it
    SUPPORTING = "supporting"  # nice-to-have


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Materiality(str, Enum):
    SPECIFIC = "specific"  # backed by a concrete, cited fact
    GENERIC = "generic"    # boilerplate risk that applies to most companies


class VerdictLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    NONE = "none"  # audit could not run (e.g. private company)


# ------------------------------------------------------------- LLM output models


class ParsedThesis(BaseModel):
    ticker: Optional[str]
    company_name: Optional[str]
    other_tickers: list[str]
    direction: Direction
    horizon_months: Optional[int]
    entry_price: Optional[float]
    ambiguities: list[str]
    is_auditable: bool
    not_auditable_reason: Optional[str]


class DraftClaim(BaseModel):
    text: str
    type: ClaimType
    importance: Importance
    stated_by_user: bool
    support_queries: list[str]
    challenge_queries: list[str]
    clarifying_question: Optional[str]


class Decomposition(BaseModel):
    claims: list[DraftClaim]


class Evidence(BaseModel):
    source: str  # audit-wide label, e.g. "S3"; resolve via ThesisAudit.sources
    quote: str
    stance: Literal["supports", "contradicts"]


class ClaimJudgment(BaseModel):
    status: ClaimStatus
    reasoning: str
    evidence: list[Evidence]
    follow_up_query: Optional[str]


class EntailmentItem(BaseModel):
    index: int
    entails: bool
    problem: Optional[str]


class EntailmentReport(BaseModel):
    items: list[EntailmentItem]


class Counterargument(BaseModel):
    text: str
    materiality: Materiality
    evidence: list[Evidence]


class StressTest(BaseModel):
    scenario: str
    what_would_have_to_be_true: list[str]
    early_warning_signs: list[str]


class UserQuestion(BaseModel):
    question: str
    why_it_matters: str


class Challenge(BaseModel):
    strongest_point_for: str
    counterarguments: list[Counterargument]
    stress_test: StressTest
    questions_for_user: list[UserQuestion]


class Explanation(BaseModel):
    plain_summary: str
    key_takeaways: list[str]


class AnswerReview(BaseModel):
    notes: str
    clarified_claims: list[str]
    new_concerns: list[str]


# ------------------------------------------------------------------ record models


class Claim(DraftClaim):
    id: str


class ClaimResult(BaseModel):
    claim: Claim
    status: ClaimStatus
    llm_status: ClaimStatus  # what the model said before code checks (kept for eval)
    reasoning: str
    evidence: list[Evidence]
    issues: list[str] = Field(default_factory=list)  # what the code checks changed
    searches: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    level: VerdictLevel
    rule: str
    reasons: list[str]


class GlossaryEntry(BaseModel):
    term: str
    definition: str


class SourceRef(BaseModel):
    label: str  # "S3" - the label used in evidence and inline [S3] citations
    chunk_id: str
    ticker: str
    doc_type: str
    period: Optional[str] = None
    section: Optional[str] = None
    filed_date: Optional[str] = None
    url: Optional[str] = None


class UserAnswer(BaseModel):
    question: str
    answer: str


DISCLAIMER = (
    "This audit grades how well the evidence in public filings supports the stated "
    "thesis. It is not a prediction of the stock price and not investment advice."
)


class ThesisAudit(BaseModel):
    audit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_input: str
    parsed: ParsedThesis
    status: Literal["complete", "not_auditable"]
    claims: list[ClaimResult] = Field(default_factory=list)
    challenge: Optional[Challenge] = None
    verdict: Verdict
    explanation: Optional[Explanation] = None
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)  # label -> chunk, cited sources only
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    number_check_issues: list[str] = Field(default_factory=list)  # filled by Bogdan's checker
    citation_issues: list[str] = Field(default_factory=list)  # inline [S#] problems in prose fields
    warnings: list[str] = Field(default_factory=list)  # optional steps that failed
    model: str = ""
    disclaimer: str = DISCLAIMER
