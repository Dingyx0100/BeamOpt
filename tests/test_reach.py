"""Step 2: reachability before solving (floor probe), the not_reachable outcome, and the rounding-slip snap."""
import pytest

from src.agent import feasibility as F
from src.agent.graph import build_graph, reach_options, run_turn
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetConstraint, SetWeight
from src.gpu_qp import gpu_available
from tests.test_graph import CFG, FakeLLM, _session


def _goal(roi, metric, value):
    return PhysicistGoal(intent='optimize', objectives=[GoalItem(roi=roi, metric=metric, change='reduce_to', value=value)])


def test_floor_probe_keeps_coverage_and_beats_the_current_plan():
    s = _session()
    p = s.current.metrics['PTV']
    f = s.floor_probe('ESOPHAGUS', 'Dmean', p['D95'] - 0.5, p['Dmax'] + 1.0)
    assert f['status'] == 'optimal'
    assert f['value'] < s.current.metrics['ESOPHAGUS']['Dmean']
    assert f['ptv_d95'] >= p['D95'] - 0.5 - 0.01 and f['ptv_dmax'] <= p['Dmax'] + 1.0 + 0.01
    assert s.current_id == 0 and not any(c.source == 'agent' for c in s.constraints.values())   # probe leaves no trace


def test_floor_probe_runs_for_every_metric_kind():
    s = _session()
    p = s.current.metrics['PTV']
    for metric in ('Dmax', 'D2', 'V20'):
        f = s.floor_probe('ESOPHAGUS', metric, p['D95'] - 0.5, p['Dmax'] + 1.0)
        assert f['status'] == 'optimal', (metric, f)
        assert f['value'] <= s.current.metrics['ESOPHAGUS'][metric] + 1e-6, (metric, f)
        assert f['ptv_d95'] >= p['D95'] - 0.5 - 0.05, (metric, f)


def test_unreachable_goal_stops_before_any_attempt():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [_goal('ESOPHAGUS', 'Dmean', 20.0)], 'Proposal': []})   # floor is ~28.7 Gy
    out = run_turn(build_graph(s, llm, CFG), 'get the esophagus mean down to 20 Gy')
    assert out['outcome'] == 'not_reachable'
    assert not out.get('attempts') and s.current_id == out['final_version'] == 0
    notes = out['resolved']['notes']
    assert 'is not reachable while protecting the PTV' in notes and 'aim for ESOPHAGUS Dmean <=' in notes
    assert [e['node'] for e in out['events']][:2] == ['understand', 'reach']


def test_small_reachable_goal_skips_the_upfront_probe():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    llm = FakeLLM({'PhysicistGoal': [_goal('ESOPHAGUS', 'Dmean', base - 1.0)],
                   'Proposal': [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=w)])
                                for w in (200, 400, 800)]})
    out = run_turn(build_graph(s, llm, CFG), 'lower the esophagus mean by 1 Gy')
    assert 'reach' not in [e['node'] for e in out['events']][:2]
    assert out['outcome'] != 'not_reachable'                           # floor ~28.7 Gy < goal: never called out
    assert all(not p['verdict'] == 'unreachable' for p in (out.get('feasibility') or {}).get('probes', []))


def test_verdict_margin_and_options():
    c = {'roi': 'ESOPHAGUS', 'metric': 'Dmax', 'threshold': 15.0}
    assert F.floor_verdict(c, {'value': 58.9}) == 'unreachable'
    assert F.floor_verdict(c, {'value': 15.4}) == 'reachable'          # inside the 3 % / 0.3 Gy margin
    assert F.floor_verdict(c, {}) == 'unknown'
    txt = reach_options({'probes': [{'roi': 'ESOPHAGUS', 'metric': 'Dmax', 'verdict': 'unreachable',
                                     'probe': {'value': 58.93}}]})
    assert 'aim for ESOPHAGUS Dmax <= 59.1 Gy' in txt


def test_rounding_slip_on_a_goal_limit_is_snapped_below_the_goal():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    goal = round(base - 0.513, 3)                                       # e.g. 33.487
    slip = SetConstraint(roi='ESOPHAGUS', kind='mean', limit_gy=round(goal + 0.003, 2))
    llm = FakeLLM({'PhysicistGoal': [_goal('ESOPHAGUS', 'Dmean', goal)],
                   'Proposal': [Proposal(reasoning='', set_constraints=[slip])] * 3})
    out = run_turn(build_graph(s, llm, CFG), f'esophagus mean to {goal} Gy')
    applied = out['attempts'][0]['applied']
    assert any('set to' in a and 'would not reliably meet' in a for a in applied), applied
    assert out['outcome'] == 'met'


@pytest.mark.skipif(not gpu_available(), reason='needs CUDA')
def test_limit_at_the_goal_gets_the_gpu_margin():
    """Round 4: a PTV Dmax limit exactly at a 62.0 goal came out at 62.009 on the GPU (accepted overshoot)."""
    s = _session()
    s.configure_gpu({'mode': 'on'})
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    goal = round(base - 0.5, 2)
    at_goal = SetConstraint(roi='ESOPHAGUS', kind='mean', limit_gy=goal)
    llm = FakeLLM({'PhysicistGoal': [_goal('ESOPHAGUS', 'Dmean', goal)],
                   'Proposal': [Proposal(reasoning='', set_constraints=[at_goal])] * 3})
    out = run_turn(build_graph(s, llm, CFG), f'esophagus mean to {goal} Gy')
    applied = out['attempts'][0]['applied']
    assert any(f'set to {round(goal - 0.01 - 0.05, 2):g}' in a for a in applied), applied
    assert out['outcome'] == 'met'
