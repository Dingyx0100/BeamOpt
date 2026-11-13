"""Trade-off options (roadmap step 3): when a request ends without a clean answer, show 2-3 real, solved plans.

Every option is a solved PlanVersion, built by deterministic knob changes (no LLM):
- 'accept'  meet the goal and accept the side effect   (the attempt that met the goal but broke a guard)
- 'capped'  meet the goal while capping that organ      (the 'accept' plan + a hard cap / penalty on the costly organ)
- 'safe'    hold every limit, go part of the way        (the best attempt that kept all protections and guards)
- 'push'    push to the goal with its most direct knob  (from the best safe plan; shows what the full goal costs)
- 'half'    halfway to the goal                         (from the start plan, when nothing safe was found)

The physicist picks one in the UI; the numbers on each option are the deterministic evaluator's, never the LLM's.
"""
from __future__ import annotations

import math

from ..plan_session import AGENT_OARS, OVERDOSE_ROIS, TARGET_ROIS, VOXELWISE_ROIS
from .goals import tradeoffs

TITLES = {
    'accept': 'Meet the goal, accept the cost',
    'capped': 'Meet the goal, cap {organs}',
    'safe': 'Hold every limit, part of the way',
    'push': 'Push to the goal',
    'half': 'Halfway to the goal',
}


def below(x: float, margin: float = 0.0) -> float:
    """A limit value that really meets an '<= x' goal (0.01 below x after rounding down), minus a solver margin."""
    return round(math.floor(x * 100 + 1e-6) / 100 - 0.01 - margin, 2)


def lever_actions(check: dict, target: float, session, margin: float = 0.0) -> list[dict]:
    """The most direct knob that drives one objective metric to `target` (an '<=' goal on an OAR or a PTV max)."""
    roi, met = check['roi'], check['metric']
    if check.get('op') != '<=':
        return []
    if met == 'Dmean' and roi in AGENT_OARS:
        return [{'action': 'set_constraint', 'roi': roi, 'kind': 'mean', 'limit_gy': below(target, margin)}]
    if met in ('Dmax', 'D2') and roi in VOXELWISE_ROIS:
        lim = below(target, margin)
        if roi in TARGET_ROIS:
            lim = max(lim, round(1.02 * session.rx_gy, 2))
        if met == 'Dmax' or roi in TARGET_ROIS:
            return [{'action': 'set_constraint', 'roi': roi, 'kind': 'max', 'limit_gy': lim}]
        # D2 of an organ: a max cap at the D2 goal over-constrains the hottest 2 %; penalize dose above it instead
        t = session.terms.get(f'linear_overdose:{roi}')
        w = max(1000.0, 3.0 * t.weight) if t is not None else 1000.0
        return [{'action': 'add_objective', 'roi': roi, 'kind': 'linear_overdose', 'dose_gy': max(1.0, lim - 1.0),
                 'weight': w}]
    if met.startswith('V') and roi in OVERDOSE_ROIS:
        try:
            dose = max(1.0, float(met[1:]) - 2.0)
        except ValueError:
            return []
        t = session.terms.get(f'linear_overdose:{roi}')
        w = max(1000.0, 3.0 * t.weight) if t is not None else 1000.0
        return [{'action': 'add_objective', 'roi': roi, 'kind': 'linear_overdose', 'dose_gy': dose, 'weight': w}]
    return []


def cap_actions(guard: dict, margin: float = 0.0) -> list[dict]:
    """Hold an organ that broke its side-effect guard at that guard's allowance."""
    roi, met, thr = guard['roi'], guard['metric'], float(guard['threshold'])
    if met == 'Dmean' and roi in AGENT_OARS:
        return [{'action': 'set_constraint', 'roi': roi, 'kind': 'mean', 'limit_gy': below(thr, margin)}]
    if met == 'Dmax' and roi in VOXELWISE_ROIS:
        return [{'action': 'set_constraint', 'roi': roi, 'kind': 'max', 'limit_gy': below(thr, margin)}]
    if met == 'D2' and roi in OVERDOSE_ROIS:
        return [{'action': 'add_objective', 'roi': roi, 'kind': 'linear_overdose', 'dose_gy': max(1.0, thr - 1.0),
                 'weight': 1000.0}]
    if met.startswith('V') and roi in OVERDOSE_ROIS:
        try:
            dose = max(1.0, float(met[1:]) - 2.0)
        except ValueError:
            return []
        return [{'action': 'add_objective', 'roi': roi, 'kind': 'linear_overdose', 'dose_gy': dose, 'weight': 1000.0}]
    return []


def summarize(key: str, version, ev: dict, start_metrics: dict, how: list[str], organs: str = '') -> dict:
    """What the UI and the reply need about one option: goal rows, broken limits and the cost vs. the start plan."""
    rows = [{'roi': c['roi'], 'metric': c['metric'], 'op': c['op'], 'threshold': c['threshold'], 'ref': c['ref'],
             'actual': c['actual'], 'passed': c['passed'], 'role': c['role']}
            for c in ev['checks'] if c['role'] in ('objective', 'protection') or not c['passed']]
    trade = tradeoffs(start_metrics, version.metrics)
    return {'key': key, 'title': TITLES[key].format(organs=organs or 'the organ'), 'version': version.id,
            'objectives_met': ev['objectives_met'], 'protections_ok': ev['protections_ok'],
            'targets_ok': ev['targets_ok'], 'progress': round(ev['progress'], 3), 'violations': ev['violations'],
            'rows': rows, 'checks': ev['checks'], 'cost': [t for t in trade if t['flag'] == 'worse'], 'tradeoffs': trade,
            'how': how, 'chosen': False}


def options_text(opts: list[dict]) -> str:
    """Plain lines for the report prompt and the deterministic reply."""
    out = []
    for i, o in enumerate(opts, 1):
        goal = '; '.join(f"{r['roi']} {r['metric']} {r['actual']:.2f}" for r in o['rows']
                         if r['role'] == 'objective' and r['actual'] is not None)
        if o['objectives_met'] and o['protections_ok']:
            state = 'meets the goal and holds every limit'
        elif o['objectives_met']:
            state = 'meets the goal but breaks ' + ', '.join(o['violations'])
        elif o['protections_ok']:
            state = f"holds every limit and reaches {o['progress']:.0%} of the change"
        else:
            state = f"reaches {o['progress']:.0%} of the change and breaks " + ', '.join(o['violations'])
        cost = ', '.join(f"{c['roi']} {c['metric']} {c['before']:.2f} -> {c['after']:.2f} {c['unit']}"
                         for c in o['cost'])
        out.append(f"Option {i} (version {o['version']}, {o['title']}{', chosen' if o['chosen'] else ''}): {goal}; "
                   f"{state}" + (f"; worse: {cost}" if cost else '; nothing else worse') + '.')
    return '\n'.join(out)
