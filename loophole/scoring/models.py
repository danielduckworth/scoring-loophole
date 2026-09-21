from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FindingType(str, Enum):
    UNDER_CREDITING = "under_crediting"
    OVER_CREDITING = "over_crediting"
    AMBIGUOUS = "ambiguous"
    DISMISSED = "dismissed"


class ReviewAction(str, Enum):
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    DEFER = "defer"


class CreditCode(BaseModel):
    code: str
    label: str
    description: str
    order: int | None = None
    administrative: bool = False


class SuppliedExample(BaseModel):
    id: str
    response: str
    supplied_code: str
    source: str = "guide"


class ConfirmedIntent(BaseModel):
    evidence: list[str] = Field(default_factory=list)
    codes: list[CreditCode] = Field(default_factory=list)
    examples: list[SuppliedExample] = Field(default_factory=list)
    confirmed: bool = False
    corrections: list[str] = Field(default_factory=list)


class GuideVersion(BaseModel):
    version: int
    text: str
    created_at: datetime = Field(default_factory=datetime.now)
    reason: str = "original guide"


class RevisionEdit(BaseModel):
    old_text: NonBlankText
    new_text: str


class ValidationCheck(BaseModel):
    scope: str
    target_id: str
    passed: bool
    details: NonBlankText


class ScoringCase(BaseModel):
    id: str
    round: int
    finding_type: FindingType
    response: str
    source: str = "illustrative generated response"
    likely_code: str | None = None
    proposed_code: str | None = None
    proposed_revision: str | None = None
    proposal_type: str | None = None
    proposal_edits: list[RevisionEdit] = Field(default_factory=list)
    proposal_supporting_passages: list[str] = Field(default_factory=list)
    guide_passages: list[str] = Field(default_factory=list)
    reasoning: str = ""
    uncertainty: str | None = None
    status: str = "pending"
    decision: ReviewAction | None = None
    decision_note: str | None = None
    # The guide used to obtain likely_code. This is immutable scoring provenance.
    guide_version: int = 1
    scoring_history: list[dict] = Field(default_factory=list)
    accepted_guide_version: int | None = None


class ValidationResult(BaseModel):
    case_id: str
    passes: bool
    details: str = ""
    checks: list[str] = Field(default_factory=list)
    check_results: list[ValidationCheck] = Field(default_factory=list)
    guide_version: int
    created_at: datetime = Field(default_factory=datetime.now)


class ScoringSessionState(BaseModel):
    session_id: str
    name: str
    stem: str
    original_guide: str
    context: str = ""
    intent: ConfirmedIntent | None = None
    current_guide: GuideVersion
    guide_history: list[GuideVersion] = Field(default_factory=list)
    cases: list[ScoringCase] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    processing_failures: list[dict] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 10
    cases_per_agent: int = 3
    next_case_number: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
