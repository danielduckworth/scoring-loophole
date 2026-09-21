from __future__ import annotations
from datetime import datetime
from pathlib import Path
import difflib
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from loophole.llm import _infer_provider, create_provider
from .agents import IntentAnalyst, Adversary, BlindScorer, Reviewer, Validator
from .models import GuideVersion, ScoringCase, ScoringSessionState, FindingType, ReviewAction
from .session import ScoringSessionManager
from .visualize import generate_html

app = typer.Typer(name="scoring", add_completion=False)
console = Console()

def _config() -> dict:
    path = Path("config.yaml")
    base = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    base.setdefault("model", {"default": "claude-sonnet-4-20250514", "max_tokens": 4096})
    base.setdefault("loop", {"max_rounds": 10, "cases_per_agent": 3})
    base.setdefault("session_dir", "sessions")
    base.setdefault("scoring", {})
    return base

def _provider(config: dict, role: str):
    override = config.get("scoring", {}).get("providers", {}).get(role) or config.get("model", {}).get("providers", {}).get(role)
    model = (override or {}).get("model", config["model"]["default"]); provider = (override or {}).get("provider", _infer_provider(model))
    return create_provider(provider, model, config["model"].get("max_tokens", 4096), base_url=(override or {}).get("base_url"))

def _agents(config: dict):
    temps = config.get("scoring", {}).get("temperatures", {})
    n = config["loop"].get("cases_per_agent", 3)
    return (IntentAnalyst(_provider(config, "intent"), temps.get("intent", .2)), Adversary(_provider(config, "under_crediting"), FindingType.UNDER_CREDITING, temps.get("under_crediting", .8), n), Adversary(_provider(config, "over_crediting"), FindingType.OVER_CREDITING, temps.get("over_crediting", .8), n), BlindScorer(_provider(config, "scorer"), temps.get("scorer", .2)), Reviewer(_provider(config, "reviewer"), temps.get("reviewer", .2)), Validator(_provider(config, "validator"), temps.get("validator", .1)))

def _input(label: str, required: bool = True) -> str:
    console.print(f"\n[bold]{label}[/bold]\n[dim]Paste text; finish with a line containing END. Blank lines are preserved.[/dim]")
    lines = []
    while True:
        line = Prompt.ask("", default="")
        if line == "END": break
        lines.append(line)
    value = "\n".join(lines)
    if required and not value.strip(): raise typer.BadParameter(f"{label} cannot be empty")
    return value

def _content(value: str | None, label: str, required: bool = True) -> str:
    if value:
        p = Path(value)
        return p.read_text(encoding="utf-8") if p.exists() else value
    return _input(label, required)

def _show_intent(state: ScoringSessionState):
    intent = state.intent
    console.print(Panel("\n".join(f"• {x}" for x in (intent.evidence if intent else [])) or "No intent inferred", title="Provisional intended evidence"))
    if intent:
        table = Table(title="Credit codes"); table.add_column("Code"); table.add_column("Description"); table.add_column("Order")
        for code in intent.codes: table.add_row(code.code, code.description, "administrative" if code.administrative else str(code.order or "—"))
        console.print(table)
        for ex in intent.examples: console.print(f"[dim]{ex.id} | supplied {ex.supplied_code}[/dim] {ex.response}")
        if intent.corrections:
            console.print(Panel("\n".join(f"• {x}" for x in intent.corrections if x.strip()), title="Corrections incorporated"))


def _apply_proposal(current: str, proposal) -> str | None:
    if proposal is None:
        return None
    if proposal.kind == "full_guide":
        return proposal.complete_guide.strip()
    candidate = current
    for edit in proposal.edits:
        if candidate.count(edit.old_text) != 1:
            return None
        candidate = candidate.replace(edit.old_text, edit.new_text, 1)
    return candidate


def _show_diff(old: str, new: str) -> None:
    diff = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile="current guide", tofile="candidate guide", lineterm=""))
    console.print(Panel(diff or "No textual change.", title="Resulting guide diff"))


def _record_score(case: ScoringCase, version: int) -> None:
    case.guide_version = version
    case.scoring_history.append({"guide_version": version, "likely_code": case.likely_code, "guide_passages": list(case.guide_passages)})


def _refresh_case(state, case, scorer, reviewer) -> bool:
    score = scorer.score(state, case.response)
    if not score:
        case.status = "retry"
        case.uncertainty = "Structured scoring failed or returned a code outside the confirmed code set; retry required."
        return False
    review = reviewer.review(state, case.response, score, case.finding_type)
    if not review:
        case.status = "retry"
        case.uncertainty = "Structured review failed; retry required."
        return False
    candidate = _apply_proposal(state.current_guide.text, review.proposal)
    if review.proposal and candidate is None:
        case.status = "retry"
        case.uncertainty = "The proposed patch did not apply exactly once to the current guide; retry required."
        return False
    case.finding_type = review.finding_type
    case.likely_code = score.likely_code
    case.proposed_code = review.proposed_code
    case.proposed_revision = candidate
    case.proposal_type = review.proposal.kind if review.proposal else None
    case.proposal_edits = review.proposal.edits if review.proposal else []
    case.proposal_supporting_passages = review.supporting_guide_passages
    case.guide_passages = score.guide_passages
    case.reasoning = review.reasoning
    case.uncertainty = review.uncertainty
    case.status = "pending"
    _record_score(case, state.current_guide.version)
    return True


def _review_case(state, case, scorer, reviewer, validator, manager):
    legacy_untyped_proposal = bool(case.proposed_revision and case.proposal_type not in ("full_guide", "patch"))
    if case.status == "retry" or case.guide_version != state.current_guide.version or legacy_untyped_proposal:
        console.print(f"[dim]Re-evaluating {case.id} against guide v{state.current_guide.version}.[/dim]")
        if not _refresh_case(state, case, scorer, reviewer):
            manager.save(state)
            console.print("[red]Re-evaluation failed and was retained for retry.[/red]")
            return
    console.print(Panel(f"[bold]Response (illustrative):[/bold]\n{case.response}\n\n[bold]Guide interpretation:[/bold] {case.likely_code}\n[bold]Proposed intended code:[/bold] {case.proposed_code}\n\n{case.reasoning}\n\n[bold]Uncertainty:[/bold] {case.uncertainty or 'None'}", title=f"{case.id} — {case.finding_type.value}"))
    if case.proposed_revision:
        console.print(Panel("\n".join(case.proposal_supporting_passages), title="Supporting current-guide passages"))
        _show_diff(state.current_guide.text, case.proposed_revision)
    action = Prompt.ask("Review", choices=[x.value for x in ReviewAction], default="defer")
    try: action_enum = ReviewAction(action)
    except ValueError: action_enum = ReviewAction.DEFER
    note = None
    if action_enum == ReviewAction.MODIFY:
        note = _input("Modification instructions")
        modified = reviewer.modify(state, case, note)
        proposed = _apply_proposal(state.current_guide.text, modified)
        if not modified or proposed is None:
            case.status = "retry"
            console.print("[red]Modification output was incomplete or could not be applied; retained for retry.[/red]")
        else:
            case.proposed_revision = proposed
            case.proposal_type = modified.kind
            case.proposal_edits = modified.edits
            case.proposal_supporting_passages = modified.supporting_guide_passages
            _show_diff(state.current_guide.text, proposed)
            validation = validator.validate(state, case, proposed)
            state.validations.append(validation)
            case.status = "pending"
            if validation.passes:
                console.print("[green]Modified candidate validated. It remains pending for explicit acceptance.[/green]")
            else:
                console.print(f"[red]Modified candidate failed validation: {validation.details}[/red]")
        case.decision = ReviewAction.MODIFY
        case.decision_note = note
    elif action_enum == ReviewAction.ACCEPT:
        proposed = case.proposed_revision
        if not proposed or proposed == state.current_guide.text:
            case.status = "pending"
            console.print("[red]Acceptance requires a complete changed guide or an applicable explicit patch.[/red]")
        else:
            validation = validator.validate(state, case, proposed)
            state.validations.append(validation)
            if validation.passes:
                new = GuideVersion(version=state.current_guide.version + 1, text=proposed, reason=f"Accepted after {case.id}")
                state.current_guide = new
                state.guide_history.append(new)
                case.accepted_guide_version = new.version
                case.decision = ReviewAction.ACCEPT
                case.status = "resolved"
            else:
                case.status = "pending"
                console.print(f"[red]Validation failed; guide unchanged: {validation.details}[/red]")
    elif action_enum == ReviewAction.REJECT:
        case.decision = ReviewAction.REJECT
        case.status = "resolved"
    else:
        case.decision = ReviewAction.DEFER
        case.status = "pending"
    state.decisions.append({"case_id": case.id, "action": action_enum.value, "note": note, "guide_version": state.current_guide.version})
    manager.save(state)


def _process_generated_items(state, items, kind, round_number, scorer, reviewer, manager):
    found = []
    for item in items:
        blind = scorer.score(state, item.get("response", ""))
        if not blind:
            case = ScoringCase(id=f"S{state.next_case_number:04d}", round=round_number, finding_type=kind, response=item.get("response", ""), reasoning="Blind scorer output was malformed, incomplete, or used an unknown code; no code discrepancy is asserted.", uncertainty="Unresolved because structured scoring output was unavailable.", status="retry", guide_version=state.current_guide.version)
            state.next_case_number += 1; state.cases.append(case); found.append(case); manager.save(state); continue
        review = reviewer.review(state, item.get("response", ""), blind, kind)
        if not review:
            case = ScoringCase(id=f"S{state.next_case_number:04d}", round=round_number, finding_type=kind, response=item.get("response", ""), likely_code=blind.likely_code, guide_passages=blind.guide_passages, reasoning="Reviewer output was malformed or missing; no code discrepancy is asserted.", uncertainty="Unresolved because structured review output was unavailable.", status="retry", guide_version=state.current_guide.version, scoring_history=[{"guide_version": state.current_guide.version, "likely_code": blind.likely_code, "guide_passages": blind.guide_passages}])
            state.next_case_number += 1; state.cases.append(case); found.append(case); manager.save(state); continue
        candidate = _apply_proposal(state.current_guide.text, review.proposal)
        status = "pending" if not review.proposal or candidate is not None else "retry"
        case = ScoringCase(id=f"S{state.next_case_number:04d}", round=round_number, finding_type=review.finding_type, response=item.get("response", ""), likely_code=blind.likely_code, proposed_code=review.proposed_code, proposed_revision=candidate, proposal_type=review.proposal.kind if review.proposal else None, proposal_edits=review.proposal.edits if review.proposal else [], proposal_supporting_passages=review.supporting_guide_passages, guide_passages=blind.guide_passages, reasoning=review.reasoning, uncertainty=review.uncertainty, status=status, guide_version=state.current_guide.version, scoring_history=[{"guide_version": state.current_guide.version, "likely_code": blind.likely_code, "guide_passages": blind.guide_passages}])
        state.next_case_number += 1; state.cases.append(case); found.append(case); manager.save(state)
    return found


def _run(state, manager, config):
    _, under, over, scorer, reviewer, validator = _agents(config)
    by_kind = {FindingType.UNDER_CREDITING: under, FindingType.OVER_CREDITING: over}
    retry_cases = []
    for failure in [x for x in state.processing_failures if x.get("stage") == "generation" and x.get("status") == "pending_retry"]:
        kind = FindingType(failure["finding_type"])
        items = by_kind[kind].generate(state)
        failure["attempts"] = failure.get("attempts", 1) + 1
        if items is None:
            continue
        failure["status"] = "resolved"
        retry_cases.extend(_process_generated_items(state, items, kind, failure["round"], scorer, reviewer, manager))
    for case in retry_cases:
        _review_case(state, case, scorer, reviewer, validator, manager)
    manager.save(state)
    max_rounds = state.max_rounds
    while state.current_round < max_rounds:
        state.current_round += 1
        generated = [(under, FindingType.UNDER_CREDITING), (over, FindingType.OVER_CREDITING)]
        found = []
        for adversary, kind in generated:
            items = adversary.generate(state)
            if items is None:
                state.processing_failures.append({"stage": "generation", "round": state.current_round, "finding_type": kind.value, "guide_version": state.current_guide.version, "status": "pending_retry"})
                manager.save(state)
                console.print(f"[red]{kind.value} generation failed and was retained for retry.[/red]")
                continue
            found.extend(_process_generated_items(state, items, kind, state.current_round, scorer, reviewer, manager))
        failures = [x for x in state.processing_failures if x.get("round") == state.current_round and x.get("status") == "pending_retry"]
        if not found and not failures: console.print("[green]No additional issues found in this round.[/green]")
        for case in found: _review_case(state, case, scorer, reviewer, validator, manager)
        manager.save(state)
        if not found or not Confirm.ask("Run another round?", default=False): break
    generate_html(state, manager=manager)
    console.print(f"[green]Scoring session saved:[/green] {manager.base_dir / state.session_id}")

@app.command()
def new(name: str = typer.Option(None, "--name"), stem: str = typer.Option(None, "--stem"), guide: str = typer.Option(None, "--guide"), context: list[str] = typer.Option([], "--context")):
    """Start a Mode 4 human scoring-guide stress test."""
    config = _config(); name = name or Prompt.ask("Item name", default="Scoring item")
    stem_text = _content(stem, "Question stem"); guide_text = _content(guide, "Draft scoring guide")
    context_text = "\n\n".join(_content(x, "Context file", False) for x in context)
    manager = ScoringSessionManager(config["session_dir"]); sid = f"{name.lower().replace(' ', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyst, *_ = _agents(config); intent = analyst.analyze(stem_text, guide_text, context_text)
    if intent is None: raise typer.Exit("Intent analysis returned malformed output; guide unchanged.")
    state = ScoringSessionState(session_id=sid, name=name, stem=stem_text, original_guide=guide_text, context=context_text, intent=intent, current_guide=GuideVersion(version=1, text=guide_text), guide_history=[GuideVersion(version=1, text=guide_text)], max_rounds=config["loop"].get("max_rounds", 10), cases_per_agent=config["loop"].get("cases_per_agent", 3))
    _show_intent(state)
    if not Confirm.ask("Confirm this inferred intent and begin testing?", default=False):
        corrections = _input("Corrections to intended evidence", False)
        state.intent.confirmed = False
        if corrections.strip():
            revised = analyst.refine(stem_text, guide_text, context_text, state.intent, corrections)
            if revised: state.intent = revised
            else: state.intent.corrections.append(corrections)
        manager.save(state); _show_intent(state); console.print("Saved pending intent confirmation."); return
    state.intent.confirmed = True; manager.save(state); _run(state, manager, config)

@app.command()
def resume(session_id: str = typer.Argument(None)):
    """Resume a scoring session, including pending reviews."""
    config = _config(); manager = ScoringSessionManager(config["session_dir"])
    if not session_id:
        sessions = manager.list_sessions()
        if not sessions: console.print("[dim]No scoring sessions found.[/dim]"); raise typer.Exit()
        for i, s in enumerate(sessions, 1): console.print(f"{i}. {s['id']} — {s['cases']} cases, guide v{s['guide_version']}")
        session_id = sessions[int(Prompt.ask("Select session number")) - 1]["id"]
    state = manager.load(session_id)
    if not state.intent or not state.intent.confirmed:
        _show_intent(state)
        if not Confirm.ask("Confirm this inferred intent and begin testing?", default=False):
            corrections = _input("Corrections to intended evidence", False)
            if corrections.strip() and state.intent:
                analyst, *_ = _agents(config)
                revised = analyst.refine(state.stem, state.original_guide, state.context, state.intent, corrections)
                if revised: state.intent = revised
                else: state.intent.corrections.append(corrections)
            if state.intent: state.intent.confirmed = False
            manager.save(state); _show_intent(state); console.print("Saved pending intent confirmation."); return
        state.intent.confirmed = True
        manager.save(state)
    pending = [x for x in state.cases if x.status in ("pending", "retry")]
    if pending:
        _, _, _, scorer, reviewer, validator = _agents(config)
        for case in pending: _review_case(state, case, scorer, reviewer, validator, manager)
    _run(state, manager, config)

@app.command(name="list")
def list_sessions():
    """List scoring sessions only."""
    sessions = ScoringSessionManager(_config()["session_dir"]).list_sessions()
    for s in sessions: console.print(f"{s['id']} — {s['cases']} cases — guide v{s['guide_version']}")

@app.command()
def visualize(session_id: str = typer.Argument(...), output: str = typer.Option(None, "--output", "-o")):
    """Export a scoring session as an escaped HTML report."""
    manager = ScoringSessionManager(_config()["session_dir"]); state = manager.load(session_id); console.print(generate_html(state, output, manager))

if __name__ == "__main__": app()
