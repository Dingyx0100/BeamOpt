"""Agent v6: headroom-sized side-effect guards, 'needs_decision' results, direct levers, hot-subset overdose."""
import numpy as np

import src.plan_session as PS
from app.revisions import status_for
from src.agent.goals import evaluate, guard_allowance, resolve_goal
from src.agent.graph import build_graph, direct_levers, run_turn
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetWeight
from tests.test_graph import FakeLLM, _session


def test_guard_allowance_uses_room_to_protocol_limit():
    assert abs(guard_allowance(7.8, 2.0, 50.0, 'Gy') - 28.9) < 1e-9       # cord far below its limit
    assert guard_allowance(7.8, 2.0, None, 'Gy') == 9.8                   # no protocol limit: fixed tolerance
    assert guard_allowance(7.8, 2.0, 50.0, 'Gy', headroom=0) == 9.8       # fixed mode
    assert guard_allowance(49.0, 2.0, 50.0, 'Gy') == 50.0                 # never past the limit
    assert abs(guard_allowance(50.0, 2.0, 50.0, 'Gy') - 50.2) < 1e-9      # at the limit: noise only
    assert guard_allowance(7.8, 2.0, 50.0, 'Gy', cap=5.0) == 12.8          # ask cap: +5 Gy at most without asking
    assert guard_allowance(7.8, 2.0, 50.0, 'Gy', cap=1.0) == 9.8           # the fixed tolerance always applies


def test_ask_cap_per_metric():
    from src.agent.goals import max_rise_for
    assert max_rise_for('D2', None) == 5.0 and max_rise_for('Dmean', None) == 3.0 and max_rise_for('V20', None) == 5.0
    assert max_rise_for('Dmean', {'Dmean': 1.0}) == 1.0 and max_rise_for('Dmax', {}) is None


def test_protocol_limits_from_criteria():
    lim = _session().protocol_limits()
    assert lim[('CORD', 'Dmax')] == 50 and lim[('CORD', 'D2')] == 50
    assert lim[('ESOPHAGUS', 'Dmean')] == 34 and lim[('ESOPHAGUS', 'Dmax')] == 66


def test_headroom_guard_and_askable():
    s = _session()
    m = s.current.metrics
    g = PhysicistGoal(intent='optimize', objectives=[GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by',
                                                              value=0.1)])
    r = resolve_goal(g, m, limits=s.protocol_limits())
    cord = next(c for c in r.checks if c.role == 'guard' and c.roi == 'CORD')
    ref = m['CORD']['Dmax']
    assert cord.limit == 50 and abs(cord.threshold - min(ref + max(2.0, min(0.5 * (50 - ref), 5.0)), 50)) < 1e-9
    worse = {**m, 'CORD': {**m['CORD'], 'Dmax': cord.threshold + 1}}
    ev = evaluate(r, worse)
    assert not ev['protections_ok'] and ev['targets_ok'] and ev['askable'] == (cord.threshold + 1 <= 50)
    past = {**m, 'CORD': {**m['CORD'], 'Dmax': 51.0}}
    assert not evaluate(r, past)['askable']                               # past the protocol limit: never asked


def test_goal_met_only_by_breaking_a_guard_asks_the_physicist():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    cfg = {'agent': {'max_iterations': 3, 'ptv_d95_tolerance_gy': 0.5,
                     'oar_guard': {'tolerance': {'Dmax': -1.0}, 'mode': 'fixed'}}}   # every cord result breaks it
    llm = FakeLLM({
        'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
            GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=w)])
                     for w in (100, 60, 150)],
    })
    out = run_turn(build_graph(s, llm, cfg), 'lower esophagus mean a bit')
    assert out['outcome'] == 'needs_decision'
    assert out['final_version'] != 0 and s.current_id == out['final_version']
    notes = out['resolved']['notes']
    assert 'met the goal, but CORD Dmax rose from' in notes and 'discard it to return to version 0' in notes
    assert status_for('needs_decision', 0, out['final_version']) == 'review'


def test_direct_levers_name_the_knob():
    checks = [{'role': 'objective', 'op': '<=', 'roi': 'PTV', 'metric': 'Dmax', 'threshold': 60.0},
              {'role': 'objective', 'op': '<=', 'roi': 'LUNGS_NOT_GTV', 'metric': 'Dmean', 'threshold': 4.587},
              {'role': 'objective', 'op': '<=', 'roi': 'LUNGS_NOT_GTV', 'metric': 'V20', 'threshold': 6.4}]
    txt = direct_levers(checks, 60.0)
    assert 'kind=max roi=PTV limit_gy=61.2' in txt                       # clamped to 1.02 x Rx
    assert 'kind=mean roi=LUNGS_NOT_GTV limit_gy=4.57' in txt          # rounded DOWN below the goal
    assert 'linear_overdose roi=LUNGS_NOT_GTV dose_gy=18' in txt
    assert 'kind=max roi=PTV limit_gy=62.5' in direct_levers([{**checks[0], 'threshold': 62.5}], 60.0)


def test_hot_subset_overdose_acts_on_hot_voxels(monkeypatch):
    s = _session()
    d = s.current.dose_total_gy[PS.PlanSession.roi(s, 'ESOPHAGUS').idx]
    T = float(np.median(d))
    sub = s.hot_subset('ESOPHAGUS', T)
    assert np.array_equal(sub, np.flatnonzero(d >= T - PS.HOT_MARGIN_GY))
    monkeypatch.setattr(PS, 'HOT_SUBSET_ROIS', ('ESOPHAGUS',))
    s.add_objective('ESOPHAGUS', 'linear_overdose', 5000, T)
    v = s.solve('hot subset')
    assert v.ok and v.metrics['ESOPHAGUS']['D2'] < s.get_version(0).metrics['ESOPHAGUS']['D2']


def test_solve_cache_is_exact_and_skips_the_solver(tmp_path):
    s = _session()
    s.solve_cache = tmp_path
    s.set_objective_weight('quadratic:ESOPHAGUS', 80)
    v1 = s.solve('first')
    assert v1.ok and not v1.cached and len(list(tmp_path.glob('*.npz'))) == 1
    s.revert(0)
    s.set_objective_weight('quadratic:ESOPHAGUS', 80)          # identical problem again
    v2 = s.solve('again')
    assert v2.cached and v2.solve_seconds == 0.0
    assert np.array_equal(v1.fluence, v2.fluence) and v1.metrics == v2.metrics
    s.revert(0)
    s.set_objective_weight('quadratic:ESOPHAGUS', 81)          # any knob change is a different problem
    assert not s.solve('different').cached
