"""Self-contained HTML audit report of one planning session (what was asked, what changed, what it cost).

Built from deterministic data only (requests, knob changes, metric checks, before/after trade-offs). The
agent's own reply is quoted verbatim and labelled as such, so a reader can tell narrative from measurement.
"""
from __future__ import annotations

import html
import time

from .agent.goals import TRADEOFF_ROWS, tradeoffs

CSS = """
body { font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color: #1d232a; margin: 0; background: #f6f7f9; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 8px; }
h3 { font-size: 15px; margin: 0 0 6px; }
.sub { color: #5b6470; margin: 0 0 16px; }
.warn { background: #fff4e5; border: 1px solid #f5c27a; border-radius: 8px; padding: 8px 12px; margin: 12px 0; }
.card { background: #fff; border: 1px solid #dde1e6; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 13px; }
th, td { border-bottom: 1px solid #e6e9ed; padding: 4px 8px; text-align: left; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.better { color: #0a7a3d; } .worse { color: #b42318; font-weight: 600; } .same { color: #6b7480; }
.pass { color: #0a7a3d; font-weight: 600; } .fail { color: #b42318; font-weight: 600; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #eef1f4; font-size: 12px; }
blockquote { margin: 6px 0; padding: 6px 12px; border-left: 3px solid #c8ced6; color: #3b434c; background: #fafbfc; }
.mono { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
.scroll { overflow-x: auto; }
"""


def _e(x) -> str:
    return html.escape(str(x))


def _trade_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="sub">No metric changes.</p>'
    out = ['<div class="scroll"><table><tr><th>ROI</th><th>metric</th><th class="n">before</th>'
           '<th class="n">after</th><th class="n">Δ</th><th></th></tr>']
    for r in rows:
        out.append(f"<tr><td>{_e(r['roi'])}{' *' if r['partial'] else ''}</td><td>{_e(r['metric'])}</td>"
                   f"<td class='n'>{r['before']:.2f}</td><td class='n'>{r['after']:.2f}</td>"
                   f"<td class='n'>{r['delta']:+.2f} {r['unit']}</td><td class='{r['flag']}'>{r['flag']}</td></tr>")
    out.append('</table></div>')
    return ''.join(out)


def _checks_table(checks: list[dict]) -> str:
    if not checks:
        return ''
    out = ['<div class="scroll"><table><tr><th>role</th><th>check</th><th class="n">start</th>'
           '<th class="n">result</th><th></th></tr>']
    for c in checks:
        unit = '%' if c['metric'].startswith('V') else 'Gy'
        act = '' if c.get('actual') is None else f"{c['actual']:.2f}"
        ok = c.get('passed')
        out.append(f"<tr><td>{_e(c['role'])}</td><td>{_e(c['roi'])} {_e(c['metric'])} {_e(c['op'])} "
                   f"{c['threshold']:.2f} {unit}</td><td class='n'>{c['ref']:.2f}</td><td class='n'>{act}</td>"
                   f"<td class='{'pass' if ok else 'fail'}'>{'pass' if ok else 'not met'}</td></tr>")
    out.append('</table></div>')
    return ''.join(out)


def build_report(session, turns: list[dict], patient_id: str, protocol: str, model: str,
                 reference_metrics: dict | None = None) -> str:
    """turns: dicts recorded by the app per request (time, request, outcome, start, final, checks, reply)."""
    cur = session.current
    base = session.get_version(0)
    now = time.strftime('%Y-%m-%d %H:%M')
    parts = [f'<title>BeamOpt session report · {_e(patient_id)}</title><style>{CSS}</style><main>',
             f'<h1>BeamOpt session report</h1><p class="sub">{_e(patient_id)} · {_e(protocol)} · model {_e(model)} · '
             f'generated {now}</p>',
             '<div class="warn"><b>Research prototype — not for clinical use.</b> Doses are total-course Gy from a '
             'beamlet (fluence) optimization without leaf sequencing. Organs marked * are only partly inside the '
             'dose-calculation box; their metrics describe the represented sub-volume.</div>']

    parts.append(f'<h2>Summary</h2><div class="card"><p>Working plan at report time: <b>v{cur.id}</b> '
                 f'({_e(cur.label)}). Requests in this session: <b>{len(turns)}</b>.</p>')
    parts.append('<h3>Working plan vs. protocol baseline (v0)</h3>' + _trade_table(tradeoffs(base.metrics, cur.metrics)))
    if reference_metrics:
        rows = []
        for roi, metric in TRADEOFF_ROWS:
            a, b = cur.metrics.get(roi, {}), reference_metrics.get(roi, {})
            if a.get(metric) is not None and b.get(metric) is not None:
                rows.append(f"<tr><td>{_e(roi)}</td><td>{_e(metric)}</td><td class='n'>{a[metric]:.2f}</td>"
                            f"<td class='n'>{b[metric]:.2f}</td></tr>")
        if rows:
            parts.append('<h3>Working plan vs. ECHO benchmark plan (same beams)</h3><div class="scroll"><table>'
                         '<tr><th>ROI</th><th>metric</th><th class="n">working</th><th class="n">ECHO</th></tr>'
                         + ''.join(rows) + '</table></div>')
    counts = session.criteria_counts(cur.id)
    if counts:
        parts.append('<p>Protocol screening of the working plan: ' + ' · '.join(
            f'<span class="tag">{_e(k)} {v}</span>' for k, v in sorted(counts.items())) + '</p>')
    parts.append('</div>')

    parts.append('<h2>Requests</h2>')
    if not turns:
        parts.append('<p class="sub">No requests yet.</p>')
    for i, t in enumerate(turns, 1):
        start, final = t.get('start'), t.get('final')
        changes = []
        if final is not None and start is not None and final != start:
            v = session.get_version(final)
            while v is not None and v.id != start:
                changes = v.changes + changes
                v = session.get_version(v.parent) if v.parent is not None else None
        rows = []
        if final is not None and start is not None and final != start:
            rows = tradeoffs(session.get_version(start).metrics, session.get_version(final).metrics)
        parts.append(
            f'<div class="card"><h3>{i}. “{_e(t["request"])}”</h3>'
            f'<p class="sub">{_e(t.get("time", ""))} · outcome <span class="tag">{_e(t.get("outcome"))}</span> · '
            f'v{start} → v{final}</p>'
            + (('<p><b>Optimizer changes:</b></p><ul class="mono">' + ''.join(f'<li>{_e(c)}</li>' for c in changes)
                + '</ul>') if changes else '<p><b>Optimizer changes:</b> none (plan unchanged)</p>')
            + (('<p><b>Deterministic checks</b> (how success was judged):</p>' + _checks_table(t['checks']))
               if t.get('checks') else '')
            + (('<p><b>Trade-offs</b> (before → after this request):</p>' + _trade_table(rows)) if rows else '')
            + f'<p><b>Agent reply</b> (model-written narrative):</p><blockquote>{_e(t.get("reply_text", "")).replace(chr(10), "<br>")}</blockquote>'
            + '</div>')

    parts.append('<h2>All solved versions</h2><div class="card scroll"><table><tr><th>v</th><th>parent</th>'
                 '<th>label</th><th>status</th><th class="n">PTV D95</th><th class="n">ESO Dmean</th>'
                 '<th class="n">CORD Dmax</th><th class="n">Lung Dmean</th></tr>')
    for v in session.history:
        m = v.metrics or {}

        def g(roi, k):
            x = m.get(roi, {}).get(k)
            return '' if x is None else f'{x:.2f}'
        parts.append(f"<tr><td>{v.id}</td><td>{'' if v.parent is None else v.parent}</td><td>{_e(v.label)}</td>"
                     f"<td>{_e(v.status)}</td><td class='n'>{g('PTV', 'D95')}</td><td class='n'>{g('ESOPHAGUS', 'Dmean')}</td>"
                     f"<td class='n'>{g('CORD', 'Dmax')}</td><td class='n'>{g('LUNGS_NOT_GTV', 'Dmean')}</td></tr>")
    parts.append('</table></div></main>')
    return '<!doctype html><meta charset="utf-8">' + ''.join(parts)
