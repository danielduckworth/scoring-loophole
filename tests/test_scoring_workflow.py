import json
import tempfile
import unittest
from pathlib import Path

from loophole.scoring.agents import Adversary, BlindScorer, Validator, _Proposal
from loophole.scoring.main import _apply_proposal
from loophole.scoring.models import (
    ConfirmedIntent,
    CreditCode,
    FindingType,
    GuideVersion,
    RevisionEdit,
    ScoringCase,
    ScoringSessionState,
    SuppliedExample,
)
from loophole.scoring.session import ScoringSessionManager
from loophole.scoring.visualize import generate_html


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload

    def call(self, system, message, temperature=0):
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def state_with_example():
    intent = ConfirmedIntent(
        confirmed=True,
        evidence=["A reason"],
        codes=[CreditCode(code="1", label="credit", description="A reason")],
        examples=[SuppliedExample(id="E1", response="Because X", supplied_code="1")],
    )
    return ScoringSessionState(
        session_id="test",
        name="Test",
        stem="Give a reason.",
        original_guide="Rule A\nRule B",
        intent=intent,
        current_guide=GuideVersion(version=1, text="Rule A\nRule B"),
    )


class ScoringWorkflowTests(unittest.TestCase):
    def test_patch_must_match_exactly_once(self):
        patch = _Proposal(kind="patch", edits=[RevisionEdit(old_text="Rule A", new_text="Rule A revised")])
        self.assertEqual(_apply_proposal("Rule A\nRule B", patch), "Rule A revised\nRule B")
        self.assertIsNone(_apply_proposal("Rule A\nRule A", patch))

    def test_malformed_generation_differs_from_valid_empty_generation(self):
        state = state_with_example()
        malformed = Adversary(FakeProvider("not json"), FindingType.UNDER_CREDITING)
        empty = Adversary(FakeProvider({"cases": []}), FindingType.UNDER_CREDITING)
        self.assertIsNone(malformed.generate(state))
        self.assertEqual(empty.generate(state), [])

    def test_score_requires_substance_and_known_code(self):
        state = state_with_example()
        self.assertIsNone(BlindScorer(FakeProvider({})).score(state, "response"))
        unknown = {"likely_code": "9", "guide_passages": ["Rule A"], "reasoning": "Applied rule"}
        self.assertIsNone(BlindScorer(FakeProvider(unknown)).score(state, "response"))

    def test_validator_rejects_missing_required_checks(self):
        state = state_with_example()
        case = ScoringCase(id="S0001", round=1, finding_type=FindingType.UNDER_CREDITING, response="Response")
        result = Validator(FakeProvider({"passes": True, "details": "Looks fine", "check_results": [
            {"scope": "triggering_response", "target_id": "S0001", "passed": True, "details": "Gets code 1"}
        ]})).validate(state, case, "Rule A revised\nRule B")
        self.assertFalse(result.passes)
        self.assertIn("E1", result.details)

    def test_validator_rejects_bare_pass_boolean(self):
        state = state_with_example()
        case = ScoringCase(id="S0001", round=1, finding_type=FindingType.UNDER_CREDITING, response="Response")
        result = Validator(FakeProvider({"passes": True})).validate(state, case, "Rule A revised\nRule B")
        self.assertFalse(result.passes)

    def test_default_report_uses_session_manager_directory(self):
        state = state_with_example()
        with tempfile.TemporaryDirectory() as directory:
            manager = ScoringSessionManager(directory)
            path = Path(generate_html(state, manager=manager))
            self.assertEqual(path, manager.report_path(state.session_id))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
