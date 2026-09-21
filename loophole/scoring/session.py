from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from .models import ScoringSessionState


class ScoringSessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir) / "scoring"
        self.base_dir.mkdir(parents=True, exist_ok=True)
    def save(self, state: ScoringSessionState) -> None:
        directory = self.base_dir / state.session_id
        directory.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2)
        fd, temp_name = tempfile.mkstemp(prefix="state-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f: f.write(payload); f.flush(); os.fsync(f.fileno())
            os.replace(temp_name, directory / "state.json")
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        (directory / "current_guide.md").write_text(f"# Scoring Guide v{state.current_guide.version}\n\n{state.current_guide.text}\n", encoding="utf-8")
        (directory / "case_log.md").write_text(render_case_log(state), encoding="utf-8")
    def load(self, session_id: str) -> ScoringSessionState:
        return ScoringSessionState.model_validate_json((self.base_dir / session_id / "state.json").read_text(encoding="utf-8"))
    def list_sessions(self) -> list[dict]:
        result = []
        if not self.base_dir.exists(): return result
        for p in sorted(self.base_dir.iterdir()):
            path = p / "state.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8")); result.append({"id": data["session_id"], "name": data["name"], "round": data["current_round"], "cases": len(data["cases"]), "guide_version": data["current_guide"]["version"]})
        return result


def render_case_log(state: ScoringSessionState) -> str:
    lines = [f"# Scoring Case Log — {state.name}", f"*Session: {state.session_id}*", "", "## Confirmed intent", "\n".join(f"- {x}" for x in (state.intent.evidence if state.intent else [])), ""]
    for case in state.cases:
        lines += [f"## Case {case.id} — {case.finding_type.value}", f"**Status:** {case.status}", f"**Response (illustrative):** {case.response}", f"**Likely code under guide:** {case.likely_code}", f"**Proposed intended code:** {case.proposed_code}", f"**Proposed revision:** {case.proposed_revision or 'None.'}", f"**Guide passages:** {' | '.join(case.guide_passages)}", f"**Reasoning:** {case.reasoning}", f"**Uncertainty:** {case.uncertainty or 'None recorded.'}", ""]
    return "\n".join(lines)
