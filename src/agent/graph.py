"""LangGraph planning agent.

    understand ─┬─(optimize)──► propose ─► apply ─► optimize ─► evaluate ─┬─► finalize ─► options ─► report
                │                  ▲                                   │
                │                  └──────────── iterate ◄─────────────┘
                ├─(question)─► answer
                ├─(revert/reset)─► restore
                └─(out_of_scope / unsafe / infeasible)─► refuse

LLM nodes: understand, propose, report, answer. Everything that touches the
plan or judges success is deterministic Python. The heavy PlanSession lives
in the closure (not in graph state) so state stays small and serializable.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..plan_session import AGENT_OARS, OVERDOSE_ROIS, TARGET_ROIS, VOXELWISE_ROIS, PlanSession, ValidationError
from .. import explain as X
from . import feasibility as F
from . import options as O
from . import prompts as P
from . import reply_check as RC
from .goals import GUARD_HEADROOM, ResolvedGoal, Check, evaluate, resolve_goal, score_tuple, tradeoffs
from .llm import structured
from .schemas import Action, LimitChange, PhysicistGoal, Proposal, make_proposal_schema, proposal_actions

TARGET_COVERAGE_METRICS = {'D95', 'D98', 'V60', 'V95%Rx', 'Dmean'}
PTV_HOLD_TERMS = ('quadratic_underdose', 'quadratic_overdose')   # hold target coverage; never lowered by the agent


class AgentState(TypedDict, total=False):
    request: str
    history: list[str]
    goal: dict
    resolved: dict
    start_version: int
    iteration: int
    attempts: list[dict]
    events: Annotated[list[dict], operator.add]
    outcome: str
    final_version: int
    reply: str
    llm_calls: int
    llm_errors: int
    action_errors: int
    tradeoffs: list[dict]
    reply_checks: list[str]
    feasibility: dict          # floor probes: {'when', 'probes': [{roi, metric, goal, probe, verdict, text}], 'unreachable'}
    options: list[dict]        # trade-off options (src/agent/options.py), each a solved plan the physicist can pick


def fallback_reply(state: dict, changes: list[str], trade: list[dict]) -> str:
    """Plain, fully deterministic reply used when the LLM's text fails the reply check twice."""
    outcome = state['outcome']
    head = {'met': 'The requested goal was met.', 'partial': 'The goal was only partly met.',
            'not_met': 'The goal was not met; the plan is unchanged.',
            'needs_decision': 'The goal was met, but at a cost to an organ you did not mention; please decide '
                              'whether to keep it.',
            'not_reachable': 'The goal is not reachable while protecting the PTV; the plan is unchanged.',
            }.get(outcome, f'Outcome: {outcome}.')
    parts = [head]
    if outcome == 'not_reachable':
        parts.append(state['resolved'].get('notes', ''))
        return ' '.join(p for p in parts if p)
    if changes and outcome != 'not_met':
        parts.append('Changes: ' + '; '.join(changes) + '.')
    worse = [r for r in trade if r['flag'] == 'worse']
    if worse and outcome != 'not_met':
        parts.append('Worse elsewhere: ' + ', '.join(f"{r['roi']} {r['metric']} {r['before']:.2f} -> "
                                                    f"{r['after']:.2f} {r['unit']}" for r in worse) + '.')
    notes = state['resolved'].get('notes', '')
    if notes:
        parts.append(notes)
    if state.get('options'):
        parts.append('Options to choose from:\n' + O.options_text(state['options']))
    return ' '.join(parts)


def _ev(node: str, text: str, **data) -> list[dict]:
    return [{'node': node, 'text': text, 't': time.time(), **data}]


def _resolved_from(d: dict) -> ResolvedGoal:
    return ResolvedGoal([Check(**c) for c in d['checks']], list(d['issues']))


def apply_action(session: PlanSession, a: Action) -> str:
    if a.action == 'set_weight':
        if not a.term_id or a.weight is None:
            raise ValidationError('set_weight needs term_id and weight')
        return session.set_objective_weight(a.term_id, a.weight)
    if a.action == 'add_objective':
        if not a.roi or not a.kind or a.weight is None:
            raise ValidationError('add_objective needs roi, kind and weight')
        return session.add_objective(a.roi, a.kind, a.weight, a.dose_gy)
    if a.action == 'set_constraint':
        if not a.roi or not a.kind or a.limit_gy is None:
            raise ValidationError('set_constraint needs roi, kind and limit_gy')
        return session.set_dose_constraint(a.roi, a.kind, a.limit_gy)
    if a.action == 'remove':
        if not a.term_id:
            raise ValidationError('remove needs term_id')
        return session.remove_agent_item(a.term_id)
    raise ValidationError(f'Unknown action {a.action!r}')


LUNG_GROUP = {'LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R'}


def _action_roi(a: dict) -> str | None:
    """ROI an action acts on (actions are dicts from Action.model_dump)."""
    if a.get('roi'):
        return str(a['roi']).upper()
    tid = a.get('term_id') or ''
    return tid.split(':')[-1].upper() if ':' in tid else None


def adjust_repeat(session: PlanSession, actions: list[dict], aggressive: bool, ref_metrics: dict) -> list[dict]:
    """A setting identical to an earlier attempt was proposed: step it instead of wasting the attempt.

    aggressive=True  (the earlier identical setting broke a protection/guard): go halfway from the current knob.
    aggressive=False (it was safe but too weak): double the step from the current knob.
    """
    f = 0.5 if aggressive else 2.0
    out = []
    for a in actions:
        a = dict(a)
        if a['action'] == 'set_weight' and a.get('term_id') in session.terms:
            cur = session.terms[a['term_id']].weight
            a['weight'] = round(max(0.0, cur + (float(a['weight']) - cur) * f), 4)
        elif a['action'] == 'add_objective':
            t = session.terms.get(f"{a.get('kind')}:{str(a.get('roi', '')).upper()}")
            cur = t.weight if t is not None else 0.0
            a['weight'] = round(max(0.0, cur + (float(a['weight']) - cur) * f), 4)
        elif a['action'] == 'set_constraint' and a.get('limit_gy') is not None:
            roi = str(a.get('roi', '')).upper()
            ref = (ref_metrics.get(roi) or {}).get('Dmax' if a.get('kind') == 'max' else 'Dmean')
            lim = float(a['limit_gy'])
            if aggressive and ref is not None:
                a['limit_gy'] = round((lim + float(ref)) / 2, 2)          # halfway back toward the current dose
            else:
                a['limit_gy'] = round(lim - max(0.5, 0.05 * lim), 2)      # one step tighter
        out.append(a)
    return out


def partial_gain_cost(ev: dict, before: dict, after: dict) -> tuple[float, float]:
    """gain = improvement on the objective metrics; cost = sum of 'worse' changes on other organs (Gy or %-pt)."""
    obj = [c for c in ev['checks'] if c['role'] == 'objective']
    gain = sum(max(0.0, (c['ref'] - c['actual']) if c['op'] == '<=' else (c['actual'] - c['ref'])) for c in obj
               if c.get('actual') is not None)
    obj_rois = {c['roi'] for c in obj}
    if obj_rois & LUNG_GROUP:
        obj_rois |= LUNG_GROUP
    cost = sum(abs(r['delta']) for r in tradeoffs(before, after) if r['flag'] == 'worse' and r['roi'] not in obj_rois)
    return round(gain, 3), round(cost, 3)


def why_not_met(attempts: list[dict], checks: list[dict]) -> str:
    """Deterministic explanation of what blocked the goal (numbers the report must use)."""
    out, best = [], {}
    for a in attempts:
        ev = a.get('eval')
        if not ev:
            continue
        objs = [c for c in ev['checks'] if c['role'] == 'objective' and c.get('actual') is not None]
        for c in objs:
            k = (c['roi'], c['metric'])
            if k not in best:
                better = True
            elif c['op'] == '<=':
                better = c['actual'] < best[k]['actual']
            else:
                better = c['actual'] > best[k]['actual']
            if better:
                best[k] = c
        if a.get('rolled_back_to') is not None and len(out) < 2:
            bad = [c for c in ev['checks'] if c['role'] in ('protection', 'guard') and not c['passed']]
            if objs and bad:
                o, b = objs[0], bad[0]
                out.append(f"Attempt {a['iteration']} reached {o['roi']} {o['metric']} {o['actual']:.2f} "
                           f"(goal {o['op']} {o['threshold']:.2f}) but {b['roi']} {b['metric']} went to "
                           f"{b['actual']:.2f} (limit {b['op']} {b['threshold']:.2f}), so it was rolled back.")
    for (roi, metric), c in best.items():
        if not c['passed']:
            unit = '%' if metric.startswith('V') else 'Gy'
            out.append(f"Closest any attempt came: {roi} {metric} {c['actual']:.2f} {unit} vs goal "
                       f"{c['op']} {c['threshold']:.2f} {unit} (start {c['ref']:.2f}).")
            want, got = abs(c['threshold'] - c['ref']), abs(c['actual'] - c['ref'])
            if want > 0 and got < 0.25 * want and want > max(2.0, 0.3 * abs(c['ref'])):
                out.append('The requested change is far beyond what the optimizer can reach with these beams while '
                           'keeping PTV coverage and the other organs within tolerance.')
    rejected = sum(1 for a in attempts if not a.get('applied'))
    if rejected:
        out.append(f'{rejected} proposed change set(s) were rejected before solving.')
    return ' '.join(out)


def guard_excess(ev: dict) -> float:
    return sum(c['actual'] - c['threshold'] for c in ev['checks']
               if c['role'] == 'guard' and not c['passed'] and c.get('actual') is not None)


def decision_note(ask: dict, safe_best: dict | None, start: int) -> str:
    """Deterministic cost statement + question for a result that met the goal by breaking only side-effect guards."""
    parts = []
    for c in ask['eval']['checks']:
        if c['role'] != 'guard' or c['passed'] or c.get('actual') is None:
            continue
        unit = '%' if c['metric'].startswith('V') else 'Gy'
        lim = f", protocol limit {c['limit']:g} {unit}" if c.get('limit') is not None else ''
        parts.append(f"{c['roi']} {c['metric']} rose from {c['ref']:.2f} to {c['actual']:.2f} {unit} "
                     f"(side-effect allowance {c['threshold']:.2f} {unit}{lim})")
    text = (f"Attempt {ask['iteration']} met the goal, but " + '; '.join(parts) + '. '
            f"This plan is kept for your review: keep it to accept that cost, or discard it to return to "
            f"version {start}.")
    if safe_best is not None and safe_best['eval']['progress'] > 0:
        text += (f" The best attempt without that cost (version {safe_best['version']}) reached "
                 f"{safe_best['eval']['progress']:.0%} of the requested change.")
    return text


GPU_LIMIT_MARGIN_GY = 0.05   # = PlanSession.solve tol_gy: how far a GPU answer may exceed a hard limit (total Gy)
MIN_PROTOCOL_WEIGHT = 0.25   # the agent may lower a protocol OAR objective to at most this fraction of its weight


def _below(x: float) -> float:
    """A suggested limit that really meets an '<= x' goal: 0.01 below x after rounding down (v6 pilot: a mean limit
    rounded to 4.59 left Lung_Patient_36 at 4.590 against a 4.587 goal, 99.4 % of the change)."""
    import math
    return math.floor(x * 100 + 1e-6) / 100 - 0.01


def direct_levers(checks: list[dict], rx_gy: float) -> str:
    """The most direct knob for each objective metric, with a concrete starting value (measured 2026-11-09)."""
    out = []
    for c in checks:
        if c['role'] != 'objective' or c['op'] != '<=':
            continue
        roi, met, thr = c['roi'], c['metric'], _below(float(c['threshold']))
        if roi in TARGET_ROIS and met in ('Dmax', 'D2'):
            lim = max(thr, 1.02 * rx_gy)
            out.append(f"{roi} {met}: set_constraints kind=max roi={roi} limit_gy={lim:.1f} is the direct knob (a "
                       f"hard cap on the hot spot; coverage objectives stay). Raising quadratic_overdose:{roi} by "
                       f"x1.5 barely moves the max.")
        elif met == 'Dmean' and roi in AGENT_OARS:
            out.append(f"{roi} Dmean: set_constraints kind=mean roi={roi} limit_gy={thr:.2f} is the direct knob (the "
                       f"optimizer finds the cheapest way to reach it). Alternative: add_objectives kind=mean weight "
                       f"100-500.")
        elif met.startswith('V') and roi in OVERDOSE_ROIS:
            t = max(1.0, float(met[1:]) - 2)
            out.append(f"{roi} {met}: add_objectives kind=linear_overdose roi={roi} dose_gy={t:g} weight=1000 is the "
                       f"direct knob (penalizes the voxels above {t:g} Gy). If it costs PTV coverage, halve the "
                       f"weight; if too weak, x3.")
        elif met == 'Dmax' and roi in VOXELWISE_ROIS:
            out.append(f"{roi} Dmax: set_constraints kind=max roi={roi} limit_gy={thr:.2f} is the direct knob.")
        elif met == 'D2' and roi in VOXELWISE_ROIS:
            out.append(f"{roi} D2: add_objectives kind=linear_overdose roi={roi} dose_gy={max(1.0, thr - 2):g} "
                       f"weight=500, or set_constraints kind=max roi={roi} limit_gy={thr:.2f} (stronger).")
    return '\n'.join(out) or '(no specific suggestion)'


def reach_options(feas: dict) -> str:
    """What the physicist can choose instead, from the floor probes (rounded up so the suggestion is reachable)."""
    import math
    opts = []
    for o in feas.get('probes', []):
        if o['verdict'] != 'unreachable':
            continue
        unit = '%' if o['metric'].startswith('V') else 'Gy'
        aim = math.ceil((o['probe']['value'] + 0.1) * 10) / 10
        if o['probe'].get('bound'):          # a geometric lower bound, not a solved plan: the aim is a starting point
            opts.append(f"aim for {o['roi']} {o['metric']} <= {aim:.1f} {unit} or higher (the geometric limit; the "
                        f"plan may need more room)")
        else:
            opts.append(f"aim for {o['roi']} {o['metric']} <= {aim:.1f} {unit} (close to the lowest found; it will "
                        f"cost other organs)")
    return ('Options: ' + '; '.join(opts) + ', or allow lower PTV coverage.') if opts else ''


DEFAULT_HINTS = ('(no measurements yet) Typical: x5-x10 on quadratic:<OAR> lowers its Dmean by a few Gy with '
                 'small PTV cost; hard constraints far below current values tend to break PTV coverage.')


def load_hints(patient_id: str | None) -> str:
    """Sensitivity table written by experiments/stage1_tools.py (runs/cache/hints_<patient>.txt)."""
    from pathlib import Path
    if patient_id:
        f = Path(__file__).resolve().parents[2] / 'runs' / 'cache' / f'hints_{patient_id}.txt'
        if f.exists():
            return f.read_text(encoding='utf-8').strip()
    return DEFAULT_HINTS


def build_graph(session: PlanSession, llm, cfg: dict, checkpointer=None, hints: str | None = None):
    acfg = cfg.get('agent', {})
    if hints is None:
        hints = load_hints(cfg.get('data', {}).get('patient_id')) if acfg.get('use_hints', True) else DEFAULT_HINTS
    max_iter = int(acfg.get('max_iterations', 4))
    ptv_tol = float(acfg.get('ptv_d95_tolerance_gy', 0.5))
    guard = acfg.get('oar_guard', {}) or {}
    guard_tol = (guard.get('tolerance') or None) if guard.get('enabled', True) else {}
    headroom = float(guard.get('headroom', GUARD_HEADROOM)) if guard.get('mode', 'headroom') == 'headroom' else 0.0
    max_rise = guard.get('max_rise')                     # None = goals.GUARD_MAX_RISE
    limits = session.protocol_limits()
    require_approval = bool(acfg.get('require_approval', False))
    keep_ratio = float(acfg.get('partial_keep_cost_ratio', 1.5))   # keep partial only if cost <= ratio * gain
    min_partial = float(acfg.get('partial_keep_min_progress', 0.2))
    want_options = bool(acfg.get('options', True))
    geo_cache: dict = {}
    goal_llm = structured(llm, PhysicistGoal)

    def _margin() -> float:
        return GPU_LIMIT_MARGIN_GY if getattr(session, '_gpu_solver', None) is not None else 0.0

    def _prop_llm():
        # Rebuilt per call: term ids change when the agent adds objectives.
        avail = [r for r, info in session.optimizable_rois().items() if info['optimizable']]
        oars = [r for r in AGENT_OARS if r in avail]
        vox = [r for r in VOXELWISE_ROIS if r in avail]
        cons = sorted(set(oars) | set(vox))
        over = [r for r in OVERDOSE_ROIS if r in avail and r != 'PTV']
        return structured(llm, make_proposal_schema(list(session.terms), oars, cons, over))

    def opt_rois():
        return [n for n, r in session.optimizable_rois().items() if r['optimizable']]

    # ------------------------------------------------------------ nodes
    def understand(state: AgentState):
        cur = session.current
        prompt = P.UNDERSTAND.format(rois=', '.join(session.structures), metrics=P.fmt_metrics(cur.metrics),
                                     history='\n'.join(state.get('history', [])[-6:]) or '(none)')
        calls, errs = state.get('llm_calls', 0) + 1, state.get('llm_errors', 0)
        goal = None
        for attempt in range(2):
            try:
                goal = goal_llm.invoke([('system', prompt), ('human', state['request'])])
                if goal is not None:
                    break
            except Exception as exc:  # noqa: BLE001 - surface any model/parse failure
                errs += 1
                last = str(exc)
        if goal is None:
            return {'goal': {'intent': 'error'}, 'llm_calls': calls, 'llm_errors': errs,
                    'events': _ev('understand', f'Could not parse the request ({last[:200]})')}
        # Deterministic backup for "relax / loosen / tighten the <organ> limit": a verb acting on a LIMIT is a
        # limit change, never a dose goal (v4 read "relax the cord limit to 55 Gy" as "cord <= 55").
        lim = F.parse_limit_request(state['request'])
        if lim is not None and (goal.intent != 'change_limit' or goal.limit is None or not goal.limit.roi):
            llm_lim = goal.limit.model_dump() if goal.limit is not None else {}
            merged = {**{k: v for k, v in llm_lim.items() if v is not None}, **{k: v for k, v in lim.items() if v}}
            if merged.get('roi'):
                goal = goal.model_copy(update={'intent': 'change_limit', 'limit': LimitChange(**merged)})
        # ...and the reverse: without any limit wording ("make sure the cord max is below 23 Gy") the LLM's
        # change_limit is a dose goal (round 2: 9/10 such requests went to the limit node, 4 got a
        # "cannot raise the protocol limit" reply to a request that was already satisfied).
        if goal.intent == 'change_limit' and lim is None and not F.mentions_limit(state['request']):
            goal = F.limit_as_goal(goal)
        if goal.intent == 'change_limit':
            return {'goal': goal.model_dump(), 'resolved': {'checks': [], 'issues': []}, 'start_version': cur.id,
                    'iteration': 0, 'attempts': [], 'llm_calls': calls, 'llm_errors': errs, 'outcome': None,
                    'events': _ev('understand', f'Intent: change_limit. {goal.summary}\n{goal.limit}',
                                  goal=goal.model_dump())}

        # Data-availability judge: organs without dose-influence data cannot be optimized. Decide this
        # deterministically (before any optimization) and give the physicist the explicit reason.
        obj_rois = [o.roi.strip().upper() for o in goal.objectives if o.change != 'keep']
        if not obj_rois and goal.intent in ('optimize', 'out_of_scope'):
            obj_rois = [r for r in F.mentioned_rois(state['request']) if r != 'PTV']
        why = {r: F.explain_unavailable(session, r, geo_cache) for r in dict.fromkeys(obj_rois)}
        why = {r: t for r, t in why.items() if t}
        resolved = resolve_goal(goal.model_copy(deep=True), cur.metrics, ptv_tol, guard_tol=guard_tol,
                                limits=limits, headroom=headroom, max_rise=max_rise)
        if why:
            resolved.issues = list(why.values()) + [i for i in resolved.issues if not i.startswith(tuple(why))]
            if not resolved.objectives or goal.intent == 'out_of_scope':
                resolved.checks = []
                text = 'Data check: ' + ' '.join(why.values())
                return {'goal': goal.model_dump(), 'resolved': {'checks': [], 'issues': resolved.issues},
                        'start_version': cur.id, 'iteration': 0, 'attempts': [], 'llm_calls': calls,
                        'llm_errors': errs, 'outcome': 'unavailable', 'events': _ev('understand', text,
                                                                                   goal=goal.model_dump())}
        caveats = [n for n in (F.partial_note(session, c.roi) for c in resolved.objectives) if n]
        # Deterministic safety: never optimize toward lower target coverage.
        unsafe = [o for o in goal.objectives if o.roi.upper() in ('PTV', 'GTV', 'CTV')
                  and o.metric in TARGET_COVERAGE_METRICS and o.direction in ('decrease', 'at_most')]
        if unsafe:
            resolved.issues.append('Requests to reduce target (PTV/GTV) coverage are outside what this assistant may do.')
            resolved.checks = []
        already = False
        if goal.intent == 'optimize' and resolved.objectives:
            ev0 = evaluate(resolved, cur.metrics)
            already = all(c['passed'] for c in ev0['checks'] if c['role'] == 'objective')
        text = (f"Intent: {goal.intent}. {goal.summary}\n" + P.fmt_checks([c.as_dict() for c in resolved.checks])
                + (('\nIssues: ' + ' '.join(resolved.issues)) if resolved.issues else ''))
        return {'goal': goal.model_dump(), 'resolved': {'checks': [c.as_dict() for c in resolved.checks],
                                                        'issues': resolved.issues,
                                                        'caveats': list(dict.fromkeys(caveats))},
                'start_version': cur.id, 'iteration': 0, 'attempts': [], 'llm_calls': calls, 'llm_errors': errs,
                'outcome': 'already_met' if already else None,
                'events': _ev('understand', text, goal=goal.model_dump())}

    def route_intent(state: AgentState):
        intent = state['goal'].get('intent')
        if state.get('outcome') in ('already_met', 'unavailable'):
            return 'refuse'
        if intent == 'change_limit':
            return 'limit'
        if intent == 'optimize':
            if not any(c['role'] == 'objective' for c in state['resolved']['checks']):
                return 'refuse'
            return 'reach' if F.wants_upfront(session, state['resolved']['checks']) else 'propose'
        return {'question': 'answer', 'revert': 'restore', 'reset': 'restore'}.get(intent, 'refuse')

    def _attempt_text(attempts):
        lines = []
        for a in attempts:
            head = f"attempt {a['iteration']}: "
            if a['applied']:
                head += 'applied [' + '; '.join(a['applied']) + ']'
            if a['errors']:
                head += ' REJECTED [' + '; '.join(a['errors']) + ']'
            if a.get('version') is None:
                lines.append(head + ' -> not solved')
                continue
            ev = a.get('eval') or {}
            res = ' | '.join(f"{c['roi']} {c['metric']}={c['actual']:.2f} ({'ok' if c['passed'] else 'NOT MET'})"
                             for c in ev.get('checks', []) if c.get('actual') is not None)
            prog = ev.get('progress')
            lines.append(f"{head} -> {a['status']}: {res}"
                         + (f" | achieved {prog:.0%} of the requested change" if prog is not None else ''))
            if prog is not None and 0 < prog < 1 and ev.get('protections_ok'):
                lines.append(f"   -> protections still OK: scale this step up (about {1 / max(prog, 0.1):.1f}x the "
                             f"change so far), or add a second knob aimed at the same metric.")
            elif prog == 0 and ev.get('protections_ok'):
                lines.append("   -> no progress: that knob is too weak or wrong; try a different / stronger knob.")
            if a.get('rolled_back_to') is not None and ev.get('askable') and ev.get('objectives_met'):
                lines.append(f"   -> it MET the goal but pushed an unmentioned organ past its side-effect allowance "
                             f"({', '.join(ev.get('violations') or [])}). It is saved to show the physicist. Knobs "
                             f"were reset to version {a['rolled_back_to']}: look for a cheaper way (a smaller step, "
                             f"or protect that organ with a max / mean constraint near its current value).")
            elif a.get('rolled_back_to') is not None:
                lines.append(f"   -> that change broke a protection/guard ({', '.join(ev.get('violations') or []) or '?'}"
                             f"; too aggressive). Knobs were RESET to version "
                             f"{a['rolled_back_to']}. Next: a smaller step in the same direction "
                             f"(e.g. half the weight increase), or a different knob.")
        return '\n'.join(lines) or '(first attempt)'

    def _ptv_protection(state):
        chk = state['resolved']['checks']
        d95 = next((c['threshold'] for c in chk if c['roi'] == 'PTV' and c['metric'] == 'D95' and c['op'] == '>='),
                   None)
        dmx = next((c['threshold'] for c in chk if c['roi'] == 'PTV' and c['metric'] == 'Dmax' and c['op'] == '<='),
                   None)
        m = session.get_version(state['start_version']).metrics['PTV']
        return (d95 if d95 is not None else m['D95'] - ptv_tol), (dmx if dmx is not None else m['Dmax'] + 1.0)

    def reach(state: AgentState):
        """Floor probes: how low can each requested metric go while PTV coverage and hard limits are kept?"""
        d95, dmx = _ptv_protection(state)
        cur_id = session.current_id
        session.revert(state['start_version'])             # probe from the plan the request started on
        out = []
        try:
            for c in F.probe_targets(state['resolved']['checks']):
                try:
                    pr = session.floor_probe(c['roi'], c['metric'], d95, dmx,
                                             stop_above=c['threshold'] + F.floor_margin(c))
                except Exception as exc:  # noqa: BLE001 - a failed probe only means "unknown"
                    pr = {'status': f'error: {exc}'}
                verdict = F.floor_verdict(c, pr)
                out.append({'roi': c['roi'], 'metric': c['metric'], 'goal': c['threshold'], 'probe': pr,
                            'verdict': verdict,
                            'text': F.floor_text(session, c, pr, d95) if verdict != 'unknown' else ''})
        finally:
            session.revert(cur_id)
        when = ('before the first attempt' if not state.get('attempts')
                else f"after attempt {state['attempts'][-1]['iteration']}")
        feas = {'when': when, 'probes': out, 'unreachable': any(o['verdict'] == 'unreachable' for o in out)}
        txt = '\n'.join(o['text'] or f"{o['roi']} {o['metric']}: probe failed ({o['probe'].get('status')})"
                        for o in out)
        return {'feasibility': feas, 'events': _ev('reach', f'Reachability check {when}:\n{txt}')}

    def route_reach(state: AgentState):
        return 'finalize' if state['feasibility']['unreachable'] else 'propose'

    def _reach_text(state):
        f = state.get('feasibility')
        if not f:
            return ''
        return '\nReachability (deterministic probe): ' + ' '.join(o['text'] for o in f['probes'] if o['text'])

    def propose(state: AgentState):
        cur = session.current
        attempts = state.get('attempts', [])
        prompt = P.PROPOSE.format(opt_rois=', '.join(opt_rois()), checks=P.fmt_checks(state['resolved']['checks']),
                                  knobs=P.fmt_knobs(session.knobs_table()), version=cur.id,
                                  metrics=P.fmt_metrics(cur.metrics), attempts=_attempt_text(attempts),
                                  hints=hints, levers=direct_levers(state['resolved']['checks'], session.rx_gy),
                                  binding=X.fmt_binding(X.binding_limits(session, cur.id)) + _reach_text(state))
        calls, errs = state.get('llm_calls', 0) + 1, state.get('llm_errors', 0)
        proposal, last = None, ''
        for _ in range(2):
            try:
                proposal = _prop_llm().invoke([('system', prompt), ('human', f"Physicist request: {state['request']}")])
                if proposal is not None and proposal.n_actions():
                    break
            except Exception as exc:  # noqa: BLE001
                errs += 1
                last = str(exc)
        auto = False
        if proposal is None or not proposal.n_actions():
            # Round 4: after a rollback the model often proposed nothing (Lung_Patient_197 V20: 4 of 5 attempts).
            # Fall back to the most direct knob: the full goal, or halfway if the last attempt broke a limit.
            prev = attempts[-1] if attempts else None
            frac = 0.5 if prev is not None and prev.get('rolled_back_to') is not None else 1.0
            acts = [x for c in state['resolved']['checks'] if c['role'] == 'objective'
                    for x in O.lever_actions(c, c['ref'] + frac * (float(c['threshold']) - c['ref']), session,
                                             _margin())]
            why = f'no valid proposal: {last[:120]}' if last else 'the model proposed no change'
            if acts:
                auto = True
                proposal = Proposal(reasoning=f"({why}; using the most direct knob"
                                              f"{' at half the step' if frac < 1 else ''})")
            else:
                proposal = Proposal(reasoning=f'({why})')
        actions = (acts if auto else [a.model_dump(exclude_none=True) for a in proposal_actions(proposal)])
        return {'llm_calls': calls, 'llm_errors': errs,
                'attempts': attempts + [{'iteration': state.get('iteration', 0) + 1, 'proposal': proposal.model_dump(),
                                         'actions': actions, 'applied': [], 'errors': [], 'auto_lever': auto}],
                'iteration': state.get('iteration', 0) + 1,
                'events': _ev('propose', proposal.reasoning + '\n' + '\n'.join(f'- {a}' for a in actions),
                              proposal=proposal.model_dump())}

    def apply(state: AgentState):
        attempts = [dict(a) for a in state['attempts']]
        att = attempts[-1]
        if require_approval:
            from langgraph.types import interrupt
            decision = interrupt({'proposal': att['proposal']})
            if not decision:
                att['errors'] = ['physicist rejected the proposal']
                return {'attempts': attempts, 'events': _ev('apply', 'Proposal rejected by physicist.')}
        applied, errors = [], []
        # Structural rule: at least one change must act on an organ the request is about (small models
        # otherwise copy the prompt's example knob and push an unrelated organ, see pilot on Lung_Patient_36).
        targets = {c['roi'] for c in state['resolved']['checks'] if c['role'] == 'objective'}
        related = set(targets)
        if targets & LUNG_GROUP:
            related |= LUNG_GROUP
        if att['actions'] and targets and not any(_action_roi(a) in related for a in att['actions'][:3]):
            knobs = [t for t in session.terms if t.split(':')[-1] in targets]
            errors.append(f"None of these changes acts on {', '.join(sorted(targets))}, which is what the physicist "
                          f"asked about. Change a knob of that organ, e.g. "
                          f"{', '.join(knobs) or 'add_objectives kind=mean'} or add an objective / constraint on "
                          f"{', '.join(sorted(targets))}; you may ADD a protective change on another organ.")
            att['applied'], att['errors'], att['fingerprint'] = [], errors, None
            return {'attempts': attempts, 'action_errors': state.get('action_errors', 0) + 1,
                    'events': _ev('apply', 'Rejected: ' + errors[0])}
        start_metrics = session.get_version(state['start_version']).metrics
        named = {c['roi'] for c in state['resolved']['checks'] if c['role'] in ('objective', 'protection')}
        actions, notes = [], []
        for a in att['actions'][:3]:
            a = dict(a)
            if a.get('action') == 'set_weight' and a.get('term_id') in session.terms and a.get('weight') is not None:
                t = session.terms[a['term_id']]
                if float(a['weight']) < t.weight and (t.kind in PTV_HOLD_TERMS or (t.roi or '') in named):
                    errors.append(f"Lowering {a['term_id']} ({t.weight:g} -> {float(a['weight']):g}) is not allowed: "
                                  + ('it holds PTV coverage.' if t.kind in PTV_HOLD_TERMS else
                                     f"{t.roi} is an organ the physicist asked to improve or protect; lowering its "
                                     f"weight pushes dose onto it."))
                    continue
                base = session.protocol_terms.get(a['term_id'])
                floor_w = None if base is None or t.roi in TARGET_ROIS else round(MIN_PROTOCOL_WEIGHT * base.weight, 4)
                if floor_w is not None and float(a['weight']) < floor_w:
                    # Round 4: the model set the cord objective weight to 0 to make room. Keep part of the protection.
                    notes.append(f"{a['term_id']} weight {float(a['weight']):g} raised to {floor_w:g} "
                                 f"({MIN_PROTOCOL_WEIGHT:.0%} of the protocol weight; the protocol objective stays)")
                    a['weight'] = floor_w
            if (a.get('action') == 'set_constraint' and a.get('limit_gy') is not None
                    and str(a.get('roi', '')).upper() not in targets | set(TARGET_ROIS)):
                # A protective cap on another organ sits at its starting dose, not at the guard edge.
                roi = str(a['roi']).upper()
                ref = (start_metrics.get(roi) or {}).get('Dmax' if a.get('kind') == 'max' else 'Dmean')
                if ref is not None and float(a['limit_gy']) > round(float(ref), 2) + 0.005:
                    notes.append(f"protective cap on {roi} lowered from {float(a['limit_gy']):g} to its starting "
                                 f"value {float(ref):.2f} Gy")
                    a['limit_gy'] = round(float(ref), 2)
            if a.get('action') == 'set_constraint' and a.get('limit_gy') is not None:
                # Rounding slip: a hard limit on the requested organ just ABOVE its goal cannot meet it (round 3:
                # Lung_Patient_36 got mean <= 4.59 for a 4.587 goal, reached 99.4 % and spent 4 more attempts).
                roi, lim = str(a.get('roi', '')).upper(), float(a['limit_gy'])
                met = 'Dmean' if a.get('kind') == 'mean' else 'Dmax'
                goal = next((float(c['threshold']) for c in state['resolved']['checks']
                             if c['role'] == 'objective' and c['op'] == '<=' and c['roi'] == roi
                             and c['metric'] == met), None)
                # With the GPU solver a hard limit may be overshot by up to GPU_LIMIT_MARGIN_GY (its accepted
                # violation), so a limit AT the goal misses it too (round 4: PTV Dmax limit 62.0 -> 62.009).
                margin = GPU_LIMIT_MARGIN_GY if getattr(session, '_gpu_solver', None) is not None else 0.0
                if goal is not None and goal - margin < lim <= goal + max(0.05, 0.01 * goal):
                    a['limit_gy'] = round(_below(goal) - margin, 2)
                    notes.append(f"{roi} {a.get('kind')} limit {lim:g} would not reliably meet the {goal:g} Gy goal "
                                 f"-> set to {a['limit_gy']:g} Gy")
            actions.append(a)

        def _apply_all(acts):
            ok, bad = [], []
            for a in acts:
                try:
                    ok.append(apply_action(session, Action(**a)))
                except (ValidationError, ValueError, TypeError) as exc:
                    bad.append(str(exc))
            return ok, bad

        applied, errs2 = _apply_all(actions)
        errors += errs2
        fp = session.knobs_fingerprint()
        dup = next((p for p in attempts[:-1] if applied and p.get('fingerprint') == fp), None)
        if dup is not None:
            # Same setting as an earlier attempt: step it deterministically instead of wasting the attempt.
            session.revert(session.current_id)
            aggressive = dup.get('rolled_back_to') is not None
            actions = adjust_repeat(session, actions, aggressive, start_metrics)
            applied, errs2 = _apply_all(actions)
            fp = session.knobs_fingerprint()
            if applied and not any(p.get('fingerprint') == fp for p in attempts[:-1]):
                notes.append(f"same setting as attempt {dup['iteration']}, which "
                             + ('broke ' + ', '.join((dup.get('eval') or {}).get('violations') or ['a guard'])
                                + ' -> auto-adjusted halfway' if aggressive else
                                'was too weak -> auto-adjusted to twice the step'))
                att['auto_adjusted'] = True
                att['actions'] = actions
            else:
                session.revert(session.current_id)
                errors.append(f"This exact setting was already tried in attempt {dup['iteration']} "
                              f"(it {'broke a protection' if aggressive else 'did not meet the goal'}). "
                              "Choose a different value or a different knob.")
                applied = []
        if applied and notes:
            applied = applied + [f'(note: {n})' for n in notes]
        att['fingerprint'] = fp if applied else None
        att['applied'], att['errors'] = applied, errors
        return {'attempts': attempts, 'action_errors': state.get('action_errors', 0) + len(errors),
                'events': _ev('apply', '\n'.join(['Applied: ' + a for a in applied] + ['Rejected: ' + e for e in errors]))}

    def route_apply(state: AgentState):
        if state['attempts'][-1]['applied']:
            return 'optimize'
        return 'propose' if state['iteration'] < max_iter else 'finalize'

    def optimize(state: AgentState):
        attempts = [dict(a) for a in state['attempts']]
        att = attempts[-1]
        v = session.solve(label=f"{state['request'][:40]} [iter {att['iteration']}]")
        att['version'], att['status'], att['solve_s'] = v.id, v.status, v.solve_seconds
        if v.ok:
            att['eval'] = evaluate(_resolved_from(state['resolved']), v.metrics)
            txt = f'Solved version {v.id} in {v.solve_seconds:.1f}s.\n' + P.fmt_checks(att['eval']['checks'])
            if not att['eval']['protections_ok']:
                # Deterministic back-off: don't let harmful changes accumulate across attempts.
                safe = [a for a in attempts[:-1] if a.get('eval') and a['eval']['protections_ok']]
                best = max(safe, key=lambda a: score_tuple(a['eval'])) if safe else None
                target = best['version'] if best else state['start_version']
                session.revert(target)
                att['rolled_back_to'] = target
                txt += f'\nProtection violated -> knobs reset to version {target} before the next attempt.'
        else:
            txt = f'Optimization failed ({v.status}); knobs rolled back to version {session.current_id}.'
        return {'attempts': attempts, 'events': _ev('optimize', txt, version=v.id)}

    def route_eval(state: AgentState):
        ev = state['attempts'][-1].get('eval')
        if ev and ev['objectives_met'] and ev['protections_ok']:
            return 'finalize'
        if state['iteration'] >= max_iter:
            return 'finalize'
        if not state.get('feasibility') and F.probe_targets(state['resolved']['checks']):
            # Lazy probe: the last attempt broke PTV coverage, or two attempts made little progress.
            last = state['attempts'][-1]
            evals = [a['eval'] for a in state['attempts'] if a.get('eval') and a['eval']['protections_ok']]
            best = max((e['progress'] for e in evals), default=0.0)
            ptv_broke = any(str(v).startswith('PTV') for v in ((last.get('eval') or {}).get('violations') or []))
            if ptv_broke or (state['iteration'] >= 2 and best < 0.5):
                return 'reach'
        return 'propose'

    def finalize(state: AgentState):
        start = state['start_version']
        best, best_score = None, None
        for a in state.get('attempts', []):
            ev = a.get('eval')
            if ev is None or not ev['protections_ok']:
                continue
            s = score_tuple(ev)
            if best_score is None or s > best_score:
                best, best_score = a, s
        start_metrics = session.get_version(start).metrics
        cost_note = ''
        if best is not None and not best['eval']['objectives_met'] and best['eval']['progress'] > 0:
            # A partial result is kept only if what it gains on the goal outweighs what it costs elsewhere.
            gain, cost = partial_gain_cost(best['eval'], start_metrics, session.get_version(best['version']).metrics)
            # Headroom mode: the guards already decide which costs are acceptable, so a partial result within them
            # is kept for review (v6 pilot: 84-99 % results were dropped by the ratio rule). Fixed mode: v5 rule.
            too_costly = headroom <= 0 and cost > keep_ratio * gain
            if best['eval']['progress'] < min_partial or too_costly:
                cost_note = (f' The best attempt (iteration {best["iteration"]}) achieved only '
                             f'{best["eval"]["progress"]:.0%} of the requested change (gain {gain:.2f}) at a cost of '
                             f'{cost:.2f} Gy / %-points on other organs, so it was not kept.')
                best = None
        ask = None
        if best is None or not best['eval']['objectives_met']:
            cands = [a for a in state.get('attempts', []) if a.get('eval') and a['eval'].get('askable')
                     and a['eval']['objectives_met'] and a.get('version') is not None]
            if cands:
                ask = min(cands, key=lambda a: guard_excess(a['eval']))
        if ask is not None:
            safe_best = best
            session.revert(ask['version'])
            final, final_eval = ask['version'], ask['eval']
            outcome = 'needs_decision'
            notes = decision_note(ask, safe_best, start)
        elif best is None or best['eval']['progress'] <= 0:
            session.revert(start)
            feas = state.get('feasibility') or {}
            if feas.get('unreachable'):
                outcome = 'not_reachable'
                notes = (' '.join(o['text'] for o in feas['probes'] if o['verdict'] == 'unreachable')
                         + ' ' + reach_options(feas) + f' The plan is unchanged (version {start}).')
            else:
                outcome = 'not_met'
                notes = (f'No attempt improved the goal enough while keeping protections; the plan was restored to '
                         f'version {start}.' + cost_note)
            final, final_eval = start, evaluate(_resolved_from(state['resolved']), session.current.metrics)
        else:
            session.revert(best['version'])
            final, final_eval = best['version'], best['eval']
            outcome = 'met' if final_eval['objectives_met'] else 'partial'
            notes = f'Selected best attempt (iteration {best["iteration"]}).'
        if outcome not in ('met', 'needs_decision', 'not_reachable'):
            notes += ' ' + why_not_met(state.get('attempts', []), state['resolved']['checks'])
            feas = state.get('feasibility') or {}
            if feas.get('unreachable'):
                notes += (' ' + ' '.join(o['text'] for o in feas['probes'] if o['verdict'] == 'unreachable')
                          + ' ' + reach_options(feas))
        caveats = state['resolved'].get('caveats') or []
        if caveats:
            notes += ' Caveat: ' + ' '.join(caveats)
        if final != start:
            session.checkpoints.append(start)   # lets 'undo' return to the plan before this request
        return {'final_version': final, 'outcome': outcome,
                'resolved': {**state['resolved'], 'final_eval': final_eval, 'notes': notes.strip()},
                'events': _ev('finalize', f'Outcome: {outcome}. {notes}', version=final)}

    def _solve_option(key, base_vid, acts, label, resolved, start_metrics, organs=''):
        session.revert(base_vid)
        applied = []
        for a in acts:
            try:
                applied.append(apply_action(session, Action(**a)))
            except (ValidationError, ValueError, TypeError):
                pass
        if not applied:
            return None
        v = session.solve(label=label)
        if not v.ok:
            return None
        return O.summarize(key, v, evaluate(resolved, v.metrics), start_metrics, applied, organs)

    def options(state: AgentState):
        """2-3 solved alternatives when the request ended without a clean answer (needs_decision / partial / not_met).
        If one of them meets the goal and holds every limit, it becomes the answer (outcome met)."""
        outcome = state['outcome']
        if not want_options or outcome not in ('needs_decision', 'partial', 'not_met'):
            return {'options': []}
        start, final = state['start_version'], state['final_version']
        resolved = _resolved_from(state['resolved'])
        start_metrics = session.get_version(start).metrics
        objs = [c for c in state['resolved']['checks'] if c['role'] == 'objective']
        tried = [a for a in state.get('attempts', []) if a.get('eval') and a.get('version') is not None]
        safe = [a for a in tried if a['eval']['protections_ok'] and a['eval']['progress'] > 0]
        best_safe = max(safe, key=lambda a: score_tuple(a['eval'])) if safe else None
        tag, m, opts, new = state['request'][:30], _margin(), [], []

        def existing(key, att):
            v = session.get_version(att['version'])
            how = [x for x in att.get('applied') or [] if not x.startswith('(note')]
            opts.append(O.summarize(key, v, evaluate(resolved, v.metrics), start_metrics,
                                    [f"attempt {att['iteration']}: " + '; '.join(how)]))

        def solved(o):
            if o is not None:
                opts.append(o)
                new.append(o)

        try:
            if outcome == 'needs_decision':
                ask = next((a for a in tried if a['version'] == final), None)
                if ask is not None:
                    existing('accept', ask)
                    broken = [c for c in ask['eval']['checks'] if c['role'] == 'guard' and not c['passed']]
                    acts = [x for c in broken for x in O.cap_actions(c, m)]
                    organs = ' and '.join(dict.fromkeys(c['roi'] for c in broken))
                    solved(_solve_option('capped', final, acts, f'{tag} [option: cap]', resolved, start_metrics,
                                         organs))
                if best_safe is not None:
                    existing('safe', best_safe)
            else:
                base = best_safe['version'] if best_safe is not None else start
                if best_safe is not None:
                    existing('safe', best_safe)
                acts = [x for c in objs for x in O.lever_actions(c, float(c['threshold']), session, m)]
                solved(_solve_option('push', base, acts, f'{tag} [option: push]', resolved, start_metrics))
                if best_safe is None:
                    acts = [x for c in objs
                            for x in O.lever_actions(c, c['ref'] + 0.5 * (float(c['threshold']) - c['ref']), session, m)]
                    solved(_solve_option('half', start, acts, f'{tag} [option: half]', resolved, start_metrics))
        finally:
            session.revert(final)

        upd: dict = {}
        win = next((o for o in new if o['objectives_met'] and o['protections_ok']), None)
        notes = state['resolved'].get('notes', '')
        if win is not None:
            session.revert(win['version'])
            if start not in session.checkpoints[-1:]:
                session.checkpoints.append(start)
            final_eval = evaluate(resolved, session.current.metrics)
            why = {'needs_decision': 'met the goal only by pushing other organs past their allowance',
                   'partial': 'held every limit but reached only part of the goal',
                   'not_met': 'did not improve the goal while holding every limit'}[outcome]
            notes = (f"Met by the option \"{win['title']}\" (version {win['version']}). The agent's own attempts "
                     f"{why}.")
            upd = {'outcome': 'met', 'final_version': win['version'],
                   'resolved': {**state['resolved'], 'final_eval': final_eval, 'notes': notes}}
            final = win['version']
        for o in opts:
            o['chosen'] = o['version'] == final
        if len(opts) < 2 and win is None:
            opts = []                     # a single option is just the result itself
        text = O.options_text(opts) if opts else 'No alternative plans could be built.'
        return {**upd, 'options': opts,
                'events': _ev('options', ('Promoted to met. ' + notes + '\n' if win else '') + text)}

    def report(state: AgentState):
        v = session.get_version(state['final_version'])
        changes = []
        cur = v
        while cur is not None and cur.id != state['start_version']:
            changes = cur.changes + changes
            cur = session.get_version(cur.parent) if cur.parent is not None else None
        trade = tradeoffs(session.get_version(state['start_version']).metrics, v.metrics)
        prompt = P.REPORT.format(request=state['request'], version=v.id, label=v.label,
                                 changes='; '.join(changes) or 'none (plan unchanged)',
                                 checks=P.fmt_checks(state['resolved']['final_eval']['checks']),
                                 tradeoffs=P.fmt_tradeoffs_text(trade),
                                 notes=state['resolved'].get('notes', ''),
                                 options=O.options_text(state.get('options') or []) or '(none)')
        calls = state.get('llm_calls', 0)
        outcome = state['outcome']
        facts = RC.numbers(prompt) + [float(v.id), float(state['start_version'])]
        msgs = [('system', prompt), ('human', 'Write the reply.')]
        reply, issues, check_log = None, [], []
        for _ in range(2):
            calls += 1
            try:
                text = _strip_think(llm.invoke(msgs).content)
            except Exception as exc:  # noqa: BLE001
                check_log.append(f'report LLM failed: {exc}')
                break
            text, dropped = RC.strip_out_of_scope(text)
            if dropped:
                check_log.append(f'removed {len(dropped)} out-of-scope sentence(s)')
            issues = RC.problems(text, facts, outcome)
            if not issues and text.strip():
                reply = text
                break
            check_log.append('reply check: ' + '; '.join(issues))
            msgs = [('system', prompt), ('human', 'Write the reply.'), ('assistant', text),
                    ('human', 'Your reply has errors: ' + '; '.join(issues) + '. Rewrite it using ONLY the numbers '
                     'given above, copied exactly, and state plainly whether the goal was met.')]
        if reply is None:
            reply = fallback_reply(state, changes, trade)
            check_log.append('used deterministic reply')
        reply += ('\n\nResults (deterministic check, version %d):\n' % v.id
                  + P.fmt_checks(state['resolved']['final_eval']['checks']))
        if v.id != state['start_version']:
            reply += f"\n\nTrade-offs vs. version {state['start_version']}:\n\n" + P.fmt_tradeoffs_md(trade)
        return {'reply': reply, 'llm_calls': calls, 'tradeoffs': trade, 'reply_checks': check_log,
                'events': _ev('report', reply, reply_checks=check_log)}

    def answer(state: AgentState):
        cur = session.current
        notes = [F.explain_unavailable(session, r, geo_cache) for r in F.mentioned_rois(state['request'])]
        notes += [F.partial_note(session, r) for r in F.mentioned_rois(state['request'])]
        prompt = P.ANSWER.format(version=cur.id, label=cur.label, metrics=P.fmt_metrics(cur.metrics),
                                 criteria=session.criteria_counts(), question=state['request'],
                                 notes=' '.join(n for n in notes if n) or '(none)')
        reply = _strip_think(llm.invoke([('system', prompt), ('human', state['request'])]).content)
        return {'reply': reply, 'outcome': 'answered', 'final_version': cur.id,
                'llm_calls': state.get('llm_calls', 0) + 1, 'events': _ev('answer', reply)}

    def restore(state: AgentState):
        intent = state['goal']['intent']
        if intent == 'reset':
            session.reset_to_protocol()
            target = 0
        else:
            cur = session.current
            target = session.checkpoints.pop() if session.checkpoints else (
                cur.parent if cur.parent is not None else cur.id)
        msg = session.revert(target)
        return {'reply': f'Done: {msg}.', 'outcome': intent, 'final_version': target,
                'events': _ev('restore', msg, version=target)}

    def limit(state: AgentState):
        """Deterministic handling of 'relax / loosen / tighten the <organ> limit to X Gy'."""
        lim = state['goal'].get('limit') or {}
        roi, kind, direction, value = (str(lim.get('roi') or '').upper(), lim.get('kind') or 'max',
                                       lim.get('direction') or 'set', lim.get('value'))
        start = session.current_id

        def done(reply, outcome, final=None, trade=None):
            return {'reply': reply, 'outcome': outcome, 'final_version': start if final is None else final,
                    'tradeoffs': trade, 'events': _ev('limit', reply)}

        why = F.explain_unavailable(session, roi, geo_cache) if roi else None
        if not roi or why:
            return done(why or "Which organ's limit should I change? Please name the organ and the new limit in Gy.",
                        'unavailable' if why else 'clarify')
        metric = 'Dmax' if kind == 'max' else 'Dmean'
        cur_val = (session.current.metrics.get(roi) or {}).get(metric)
        proto = session.protocol_constraints.get(f'{kind}:{roi}')
        agent = session.constraints.get(f'agent_{kind}:{roi}')
        p_txt = f'{proto.limit_gy:g} Gy' if proto else None
        what = f'{roi} {kind}-dose limit'
        now = f'The current {roi} {metric} is {cur_val:.2f} Gy' if cur_val is not None else ''
        if value is None and not (direction == 'loosen' and agent is not None):
            return done(f'I understood that you want to {direction} the {what}, but no new value was given. {now}'
                        + (f'; the protocol limit is {p_txt}' if p_txt else '') + '. What limit (in Gy) should I use?',
                        'clarify')
        effective = min([c.limit_gy for c in (proto, agent) if c is not None], default=None)
        loosen = direction == 'loosen' or (value is not None and effective is not None and value > effective)
        if loosen:
            if agent is None:
                if proto is None:
                    return done(f'There is no {what} in this plan to relax, so nothing changed. {now}.', 'no_change')
                msg = (f'I understood this as a request to raise the {what} from {p_txt} to {value:g} Gy, i.e. to '
                       f'allow more {roi} dose. I did not change it: {p_txt} is the {roi} limit of the clinical '
                       f'protocol, and this assistant can tighten protocol limits but never loosen them; going above '
                       f'a protocol limit needs a physician decision outside this tool.')
                if cur_val is not None and proto.limit_gy - cur_val > 1.0:
                    msg += (f' {now}, {proto.limit_gy - cur_val:.2f} Gy below the limit, so the limit is not what '
                            f'restricts this plan; raising it would not change the optimized dose.')
                return done(msg, 'limit_refused')
            new = value if value is not None else (proto.limit_gy if proto else None)
            capped = proto is not None and (new is None or new >= proto.limit_gy)
            if capped or new is None:
                session.remove_agent_item(agent.id)
                head = (f'I removed the {what} of {agent.limit_gy:g} Gy that I had added earlier'
                        + (f', so the protocol limit of {p_txt} applies again' if proto else '')
                        + (f' (you asked for {value:g} Gy; the protocol limit cannot be exceeded)'
                           if value is not None and proto is not None and value > proto.limit_gy else '') + '.')
            else:
                session.set_dose_constraint(roi, kind, new)
                head = f'I relaxed the {what} that I had added earlier from {agent.limit_gy:g} Gy to {new:g} Gy.'
        else:
            if effective is not None and value >= effective:
                return done(f'The {what} is already {effective:g} Gy, at or below the requested {value:g} Gy, so '
                            f'nothing changed. {now}.', 'no_change')
            try:
                session.set_dose_constraint(roi, kind, value)
            except ValidationError as exc:
                return done(f'I could not set that limit: {exc}', 'refused')
            head = f'I set a hard {what} of {value:g} Gy.'
        v = session.solve(label=f"{state['request'][:40]} [limit]")
        if not v.ok:
            session.revert(start)
            return done(head + f' The optimizer could not find a plan with it ({v.status}), so I restored version '
                        f'{start}.', 'limit_failed')
        before = session.get_version(start).metrics
        safety = resolve_goal(PhysicistGoal(intent='optimize'), before, ptv_tol, guard_tol=guard_tol,
                              limits=limits, headroom=headroom, max_rise=max_rise)
        safety.checks = [c for c in safety.checks if c.roi != roi]
        ev = evaluate(safety, v.metrics)
        trade = tradeoffs(before, v.metrics)
        if not ev['protections_ok']:
            session.revert(start)
            return done(head + ' But the re-optimized plan broke ' + ', '.join(ev['violations'])
                        + f', so I restored version {start}.', 'limit_failed', trade=trade)
        session.checkpoints.append(start)
        new_val = (v.metrics.get(roi) or {}).get(metric)
        reply = head
        if new_val is not None and cur_val is not None:
            reply += f' After re-optimizing (version {v.id}), {roi} {metric} is {new_val:.2f} Gy (was {cur_val:.2f}).'
        reply += f"\n\nTrade-offs vs. version {start}:\n\n" + P.fmt_tradeoffs_md(trade)
        return done(reply, 'limit_changed', final=v.id, trade=trade)

    def refuse(state: AgentState):
        goal = state.get('goal', {})
        issues = state.get('resolved', {}).get('issues', [])
        if state.get('outcome') == 'already_met':
            reply = (f'The current plan (version {session.current_id}) already satisfies that request, '
                     'so I made no changes:\n') + P.fmt_checks(evaluate(_resolved_from(state['resolved']),
                                                                    session.current.metrics)['checks'])
            outcome = 'already_met'
        elif state.get('outcome') == 'unavailable':
            reply, outcome = ' '.join(issues), 'unavailable'
        elif goal.get('intent') == 'error':
            reply, outcome = 'Sorry, I could not interpret that request. Could you rephrase it with an ROI and metric?', 'error'
        elif issues:
            reply = 'I did not change the plan. ' + ' '.join(issues)
            outcome = 'infeasible' if not any('outside' in i for i in issues) else 'refused'
        elif goal.get('intent') == 'out_of_scope':
            reply = ('That request is outside what I can do here: I can adjust OAR objectives/constraints of this '
                     'research plan, answer questions about it, or revert changes. Prescription and target '
                     'coverage are fixed.')
            outcome = 'refused'
        else:
            reply = 'I could not identify a measurable goal (ROI + metric). Could you be more specific?'
            outcome = 'clarify'
        return {'reply': reply, 'outcome': outcome, 'final_version': session.current_id,
                'events': _ev('refuse', reply)}

    g = StateGraph(AgentState)
    for name, fn in [('understand', understand), ('propose', propose), ('apply', apply), ('optimize', optimize),
                     ('finalize', finalize), ('report', report), ('answer', answer), ('restore', restore),
                     ('refuse', refuse), ('limit', limit), ('reach', reach), ('options', options)]:
        g.add_node(name, fn)
    g.add_edge(START, 'understand')
    g.add_conditional_edges('understand', route_intent, ['reach', 'propose', 'answer', 'restore', 'refuse', 'limit'])
    g.add_conditional_edges('reach', route_reach, ['propose', 'finalize'])
    g.add_edge('propose', 'apply')
    g.add_conditional_edges('apply', route_apply, ['optimize', 'propose', 'finalize'])
    g.add_conditional_edges('optimize', route_eval, ['finalize', 'propose', 'reach'])
    g.add_edge('finalize', 'options')
    g.add_edge('options', 'report')
    for n in ('report', 'answer', 'restore', 'refuse', 'limit'):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer)


def _strip_think(text: str) -> str:
    import re
    return re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()


def run_turn(graph, request: str, history: list[str] | None = None, on_event=None, config=None) -> dict:
    """Run one physicist turn; returns final state. on_event(event_dict) is called as nodes finish."""
    state: dict[str, Any] = {'request': request, 'history': history or [], 'events': []}
    final: dict[str, Any] = dict(state)
    for update in graph.stream(state, config=config or {}, stream_mode='updates'):
        for node, delta in update.items():
            if not isinstance(delta, dict):
                continue
            for e in delta.get('events', []):
                if on_event:
                    on_event(e)
            for k, v in delta.items():
                if k == 'events':
                    final['events'] = final.get('events', []) + v
                else:
                    final[k] = v
    return final
