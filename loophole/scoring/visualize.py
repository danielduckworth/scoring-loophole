from __future__ import annotations
import html
from pathlib import Path
from .models import ScoringSessionState
from .session import ScoringSessionManager

def generate_html(state: ScoringSessionState, output_path: str | None = None, manager: ScoringSessionManager | None = None) -> str:
    out = Path(output_path) if output_path else (manager or ScoringSessionManager()).report_path(state.session_id)
    cases = []
    for c in state.cases:
        accepted = f"v{c.accepted_guide_version}" if c.accepted_guide_version else "Not accepted"
        cases.append(f"<article><h2>Case {html.escape(c.id)} — {html.escape(c.finding_type.value)}</h2><p><b>Status:</b> {html.escape(c.status)} &nbsp; <b>Scored with:</b> v{c.guide_version} &nbsp; <b>Accepted as:</b> {accepted}</p><h3>Illustrative response</h3><pre>{html.escape(c.response)}</pre><h3>Guide passages</h3><p>{html.escape(' | '.join(c.guide_passages))}</p><h3>Interpretation</h3><p>{html.escape(c.reasoning)}</p><p><b>Guide code:</b> {html.escape(c.likely_code or 'Unresolved')} &nbsp; <b>Proposed code:</b> {html.escape(c.proposed_code or 'Unresolved')}</p><p><b>Uncertainty:</b> {html.escape(c.uncertainty or 'None recorded.')}</p></article>")
    body = f"<!doctype html><html><head><meta charset='utf-8'><title>Scoring report — {html.escape(state.name)}</title><style>body{{font:16px system-ui;max-width:960px;margin:2rem auto;color:#222}} article{{border:1px solid #bbb;border-radius:8px;padding:1rem;margin:1rem 0}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}.meta{{background:#eef5ff;padding:1rem}}</style></head><body><h1>Scoring-guide stress test</h1><div class='meta'><p><b>Item:</b> {html.escape(state.name)}</p><p><b>Round:</b> {state.current_round} &nbsp; <b>Guide:</b> v{state.current_guide.version}</p><h2>Confirmed intended evidence</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in (state.intent.evidence if state.intent else []))}</ul></div>{''.join(cases) or '<p>No cases recorded.</p>'}<h2>Current guide</h2><pre>{html.escape(state.current_guide.text)}</pre></body></html>"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(body, encoding="utf-8"); return str(out)
