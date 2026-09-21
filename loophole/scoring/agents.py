from __future__ import annotations

import json
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError

from loophole.llm import LLMProvider
from .models import ConfirmedIntent, ScoringCase, ScoringSessionState, ValidationResult, FindingType
from .prompts import INTENT_SYSTEM, GENERATE_SYSTEM, SCORE_SYSTEM, REVIEW_SYSTEM, VALIDATE_SYSTEM

T = TypeVar("T", bound=BaseModel)


def _json(provider: LLMProvider, system: str, message: str, model: type[T], temperature: float) -> T | None:
    try:
        raw = provider.call(system, message, temperature=temperature).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, AttributeError, TypeError, ValueError):
        return None


class _Cases(BaseModel):
    cases: list[dict[str, Any]]


class _Score(BaseModel):
    likely_code: str | None = None
    guide_passages: list[str] = []
    reasoning: str = ""
    uncertainty: str | None = None


class _Review(BaseModel):
    finding_type: FindingType
    proposed_code: str | None = None
    reasoning: str = ""
    uncertainty: str | None = None
    proposal: str | None = None


class _Validation(BaseModel):
    passes: bool
    details: str = ""
    checks: list[str] = []


class IntentAnalyst:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def analyze(self, stem: str, guide: str, context: str = "") -> ConfirmedIntent | None:
        msg = f"STEM:\n{stem}\n\nGUIDE:\n{guide}\n\nOPTIONAL CONTEXT:\n{context}"
        return _json(self.provider, INTENT_SYSTEM, msg, ConfirmedIntent, self.temperature)


class Adversary:
    def __init__(self, provider: LLMProvider, finding_type: FindingType, temperature: float = .8, cases_per_agent: int = 3):
        self.provider, self.finding_type, self.temperature, self.cases_per_agent = provider, finding_type, temperature, cases_per_agent
    def generate(self, state: ScoringSessionState) -> list[dict[str, Any]]:
        msg = f"FAILURE TYPE: {self.finding_type.value}\nSTEM:\n{state.stem}\nGUIDE:\n{state.current_guide.text}\nINTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\nGenerate at most {self.cases_per_agent} cases."
        result = _json(self.provider, GENERATE_SYSTEM, msg, _Cases, self.temperature)
        return result.cases if result else []


class BlindScorer:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def score(self, state: ScoringSessionState, response: str) -> _Score | None:
        return _json(self.provider, SCORE_SYSTEM, f"STEM:\n{state.stem}\nGUIDE:\n{state.current_guide.text}\nRESPONSE:\n{response}", _Score, self.temperature)


class Reviewer:
    def __init__(self, provider: LLMProvider, temperature: float = .2): self.provider, self.temperature = provider, temperature
    def review(self, state: ScoringSessionState, response: str, score: _Score, finding_type: FindingType) -> _Review | None:
        msg = f"INTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\nCONTEXT:\n{state.context}\nORIGINAL EXAMPLES:\n{[x.model_dump() for x in (state.intent.examples if state.intent else [])]}\nCASE TYPE: {finding_type.value}\nRESPONSE:\n{response}\nBLIND SCORE:\n{score.model_dump_json()}"
        return _json(self.provider, REVIEW_SYSTEM, msg, _Review, self.temperature)


class Validator:
    def __init__(self, provider: LLMProvider, temperature: float = .1): self.provider, self.temperature = provider, temperature
    def validate(self, state: ScoringSessionState, case: ScoringCase, proposed_guide: str) -> ValidationResult:
        msg = f"INTENT:\n{state.intent.model_dump_json() if state.intent else '{}'}\nCASE:\n{case.model_dump_json()}\nPROPOSED GUIDE:\n{proposed_guide}\nPRIOR DECISIONS:\n{state.decisions}"
        result = _json(self.provider, VALIDATE_SYSTEM, msg, _Validation, self.temperature)
        if not result:
            return ValidationResult(case_id=case.id, passes=False, details="Validation output was malformed or missing.", guide_version=state.current_guide.version)
        return ValidationResult(case_id=case.id, guide_version=state.current_guide.version, **result.model_dump())

