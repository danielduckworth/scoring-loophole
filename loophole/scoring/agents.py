from __future__ import annotations

import json
from typing import Any, Literal, TypeVar
from pydantic import BaseModel, Field, ValidationError, model_validator

from loophole.llm import LLMProvider
from .models import ConfirmedIntent, NonBlankText, RevisionEdit, ScoringCase, ScoringSessionState, ValidationCheck, ValidationResult, FindingType
from .prompts import INTENT_SYSTEM, GENERATE_SYSTEM, MODIFY_SYSTEM, SCORE_SYSTEM, REVIEW_SYSTEM, VALIDATE_SYSTEM

T = TypeVar("T", bound=BaseModel)


def _json(provider: LLMProvider, system: str, message: str, model: type[T], temperature: float) -> T | None:
    try:
        raw = provider.call(system, message, temperature=temperature).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, AttributeError, TypeError, ValueError):
        return None


class _GeneratedCase(BaseModel):
    response: NonBlankText
    proposed_code: str | None = None
    reasoning: str = ""
    uncertainty: str | None = None


class _Cases(BaseModel):
    cases: list[_GeneratedCase]


class _Score(BaseModel):
    likely_code: NonBlankText
    guide_passages: list[NonBlankText] = Field(min_length=1)
    reasoning: NonBlankText
    uncertainty: str | None = None


class _Proposal(BaseModel):
    kind: Literal["full_guide", "patch"]
    complete_guide: NonBlankText | None = None
    edits: list[RevisionEdit] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_or_patch(self):
        if self.kind == "full_guide" and not (self.complete_guide or "").strip():
            raise ValueError("full_guide requires complete_guide")
        if self.kind == "patch" and not self.edits:
            raise ValueError("patch requires at least one edit")
        return self


class _Review(BaseModel):
    finding_type: FindingType
    proposed_code: str | None = None
    reasoning: NonBlankText
    uncertainty: str | None = None
    supporting_guide_passages: list[NonBlankText] = Field(min_length=1)
    proposal: _Proposal | None = None


class _Validation(BaseModel):
    passes: bool
    details: NonBlankText
    check_results: list[ValidationCheck] = Field(min_length=1)


class _ModifiedProposal(_Proposal):
    supporting_guide_passages: list[NonBlankText] = Field(min_length=1)


class IntentAnalyst:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def analyze(self, stem: str, guide: str, context: str = "") -> ConfirmedIntent | None:
        msg = f"STEM:\n{stem}\n\nGUIDE:\n{guide}\n\nOPTIONAL CONTEXT:\n{context}"
        return _json(self.provider, INTENT_SYSTEM, msg, ConfirmedIntent, self.temperature)
    def refine(self, stem: str, guide: str, context: str, intent: ConfirmedIntent, corrections: str) -> ConfirmedIntent | None:
        msg = f"STEM:\n{stem}\n\nGUIDE:\n{guide}\n\nOPTIONAL CONTEXT:\n{context}\n\nPRIOR INFERENCE:\n{intent.model_dump_json()}\n\nUSER CORRECTIONS:\n{corrections}\n\nReturn a corrected intent, incorporating the corrections into evidence, codes, and examples."
        revised = _json(self.provider, INTENT_SYSTEM, msg, ConfirmedIntent, self.temperature)
        if revised:
            revised.confirmed = False
            revised.corrections = [*intent.corrections, corrections]
        return revised


class Adversary:
    def __init__(self, provider: LLMProvider, finding_type: FindingType, temperature: float = .8, cases_per_agent: int = 3):
        self.provider, self.finding_type, self.temperature, self.cases_per_agent = provider, finding_type, temperature, cases_per_agent
    def generate(self, state: ScoringSessionState) -> list[dict[str, Any]] | None:
        msg = f"FAILURE TYPE: {self.finding_type.value}\nSTEM:\n{state.stem}\nGUIDE:\n{state.current_guide.text}\nINTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\nGenerate at most {self.cases_per_agent} cases."
        result = _json(self.provider, GENERATE_SYSTEM, msg, _Cases, self.temperature)
        return [case.model_dump() for case in result.cases] if result else None


class BlindScorer:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def score(self, state: ScoringSessionState, response: str) -> _Score | None:
        result = _json(self.provider, SCORE_SYSTEM, f"STEM:\n{state.stem}\nGUIDE:\n{state.current_guide.text}\nRESPONSE:\n{response}", _Score, self.temperature)
        valid_codes = {code.code for code in (state.intent.codes if state.intent else [])}
        return result if result and result.likely_code in valid_codes else None


class Reviewer:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def review(self, state: ScoringSessionState, response: str, score: _Score, finding_type: FindingType) -> _Review | None:
        msg = f"STEM:\n{state.stem}\n\nCURRENT GUIDE v{state.current_guide.version}:\n{state.current_guide.text}\n\nINTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\nCONTEXT:\n{state.context}\nORIGINAL EXAMPLES:\n{[x.model_dump() for x in (state.intent.examples if state.intent else [])]}\nCASE TYPE: {finding_type.value}\nRESPONSE:\n{response}\nBLIND SCORE:\n{score.model_dump_json()}"
        result = _json(self.provider, REVIEW_SYSTEM, msg, _Review, self.temperature)
        if not result or any(passage not in state.current_guide.text for passage in result.supporting_guide_passages):
            return None
        return result
    def modify(self, state: ScoringSessionState, case: ScoringCase, instructions: str) -> _ModifiedProposal | None:
        msg = f"STEM:\n{state.stem}\n\nCURRENT GUIDE:\n{state.current_guide.text}\n\nCURRENT CANDIDATE:\n{case.proposed_revision or state.current_guide.text}\n\nTRIGGERING RESPONSE:\n{case.response}\n\nMODIFICATION INSTRUCTIONS:\n{instructions}"
        result = _json(self.provider, MODIFY_SYSTEM, msg, _ModifiedProposal, self.temperature)
        if not result or any(passage not in state.current_guide.text for passage in result.supporting_guide_passages):
            return None
        return result


class Validator:
    def __init__(self, provider: LLMProvider, temperature: float = .1): self.provider, self.temperature = provider, temperature
    def validate(self, state: ScoringSessionState, case: ScoringCase, proposed_guide: str) -> ValidationResult:
        accepted = [x for x in state.cases if x.decision == "accept" and x.id != case.id]
        msg = f"STEM:\n{state.stem}\n\nCONTEXT:\n{state.context}\n\nCURRENT GUIDE:\n{state.current_guide.text}\n\nINTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\n\nTRIGGERING RESPONSE:\n{case.model_dump_json()}\n\nSUPPLIED EXAMPLES:\n{[x.model_dump() for x in (state.intent.examples if state.intent else [])]}\n\nPREVIOUSLY ACCEPTED CASES:\n{[{'id': x.id, 'response': x.response, 'proposed_code': x.proposed_code, 'decision_note': x.decision_note} for x in accepted]}\n\nCANDIDATE GUIDE:\n{proposed_guide}"
        result = _json(self.provider, VALIDATE_SYSTEM, msg, _Validation, self.temperature)
        if not result:
            return ValidationResult(case_id=case.id, passes=False, details="Validation output was malformed or missing.", guide_version=state.current_guide.version)
        required = {("triggering_response", case.id)}
        required.update(("supplied_example", x.id) for x in (state.intent.examples if state.intent else []))
        required.update(("accepted_case", x.id) for x in accepted)
        observed = {(x.scope, x.target_id) for x in result.check_results}
        complete = required <= observed
        all_pass = all(x.passed for x in result.check_results if (x.scope, x.target_id) in required)
        passes = bool(result.passes and complete and all_pass)
        missing = sorted(required - observed)
        details = result.details if complete else f"Missing required validation checks: {missing}. {result.details}"
        return ValidationResult(case_id=case.id, guide_version=state.current_guide.version, passes=passes, details=details, checks=[f"{x.scope}:{x.target_id}" for x in result.check_results], check_results=result.check_results)
