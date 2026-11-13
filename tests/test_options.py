"""Step 3: trade-off options (solved alternative plans), the empty-proposal fallback and the protocol-weight floor."""
import pytest

from app.revisions import RevisionLog
from src.agent import options as O
from src.agent.graph import build_graph, run_turn
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetWeight
from tests.test_graph import CFG, FakeLLM, _session


def _reduce(roi, metric, by):
    return PhysicistGoal(intent='optimize', objectives=[GoalItem(roi=roi, metric=metric, change='reduce_by', value=by)])


def test_lever_and_cap_actions():
    s = _session()
    mean = {'roi': 'ESOPHAGUS', 'metric': 'Dmean', 'op': '<=', 'threshold': 33.0}
    assert O.lever_actions(mean, 33.0, s) == [{'action': 'set_constraint', 'roi': 'ESOPHAGUS', 'kind': 'mean',
                                               'limit_gy': 32.99}]
    assert O.lever_actions(mean, 33.0, s, margin=0.05)[0]['limit_gy'] == 32.94
    ptv = O.lever_actions({'roi': 'PTV', 'metric': 'Dmax', 'op': '<=', 'threshold': 55.0}, 55.0, s)
    assert ptv[0]['limit_gy'] == round(1.02 * s.rx_gy, 2)                 # never below 1.02 x Rx
    v20 = O.lever_actions({'roi': 'LUNGS_NOT_GTV', 'metric': 'V20', 'op': '<=', 'threshold': 20.0}, 20.0, s)
    assert v20 and v20[0]['kind'] == 'linear_overdose' and v20[0]['dose_gy'] == 18.0
    assert O.lever_actions({**mean, 'op': '>='}, 33.0, s) == []
    cap = O.cap_actions({'roi': 'CORD', 'metric': 'Dmax', 'threshold': 9.8})
    assert cap == [{'action': 'set_constraint', 'roi': 'CORD', 'kind': 'max', 'limit_gy': 9.79}]


def test_not_met_builds_options_and_promotes_one_that_meets_the_goal():
    s = _session()
    bad = Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:NOPE', weight=5)])   # rejected every time
    llm = FakeLLM({'PhysicistGoal': [_reduce('ESOPHAGUS', 'Dmean', 1.0)], 'Proposal': [bad] * 3})
    out = run_turn(build_graph(s, llm, CFG), 'lower the esophagus mean by 1 Gy')
    keys = [o['key'] for o in out['options']]
    assert keys == ['push', 'half'], keys
    assert all(o['version'] not in (None, 0) for o in out['options'])
    assert out['outcome'] == 'met' and out['resolved']['notes'].startswith('Met by the option')
    chosen = [o for o in out['options'] if o['chosen']]
    assert len(chosen) == 1 and chosen[0]['version'] == out['final_version'] == s.current_id
    assert 'options' in [e['node'] for e in out['events']]
    assert s.current.metrics['ESOPHAGUS']['Dmean'] <= 33.0 + 1e-6


def test_options_off_keeps_the_v7_result():
    s = _session()
    bad = Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:NOPE', weight=5)])
    llm = FakeLLM({'PhysicistGoal': [_reduce('ESOPHAGUS', 'Dmean', 1.0)], 'Proposal': [bad] * 3})
    cfg = {'agent': {**CFG['agent'], 'options': False}}
    out = run_turn(build_graph(s, llm, cfg), 'lower the esophagus mean by 1 Gy')
    assert out['outcome'] == 'not_met' and out['options'] == [] and s.current_id == 0


def test_needs_decision_offers_the_cost_and_a_capped_plan():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    cfg = {'agent': {'max_iterations': 3, 'ptv_d95_tolerance_gy': 0.5,
                     'oar_guard': {'tolerance': {'Dmax': -1.0}, 'mode': 'fixed'}}}   # every cord result breaks it
    llm = FakeLLM({'PhysicistGoal': [_reduce('ESOPHAGUS', 'Dmean', 0.05 * base)],
                   'Proposal': [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=w)])
                                for w in (100, 60, 150)]})
    out = run_turn(build_graph(s, llm, cfg), 'lower esophagus mean a bit')
    keys = [o['key'] for o in out['options']]
    assert keys[:2] == ['accept', 'capped'], keys
    capped = out['options'][1]
    assert 'cap CORD' in capped['title'] and any('CORD' in h for h in capped['how'])
    assert out['outcome'] in ('needs_decision', 'met')
    assert s.current_id == out['final_version'] and [o['chosen'] for o in out['options']].count(True) == 1


def test_empty_proposal_falls_back_to_the_direct_knob():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [_reduce('ESOPHAGUS', 'Dmean', 1.0)], 'Proposal': [Proposal(reasoning='')] * 6})
    out = run_turn(build_graph(s, llm, CFG), 'lower the esophagus mean by 1 Gy')
    a = out['attempts'][0]
    assert a['auto_lever'] and any('constraint' in x for x in a['applied']), a
    assert out['outcome'] == 'met'


def test_protocol_oar_weight_is_not_zeroed():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [_reduce('ESOPHAGUS', 'Dmean', 0.5)],
                   'Proposal': [Proposal(reasoning='', set_weights=[
                       SetWeight(term_id='quadratic:ESOPHAGUS', weight=100), SetWeight(term_id='quadratic:CORD',
                                                                                         weight=0)])] * 3})
    out = run_turn(build_graph(s, llm, CFG), 'lower the esophagus mean by 0.5 Gy')
    applied = out['attempts'][0]['applied']
    assert any('quadratic:CORD weight 0 raised to 2.5' in x for x in applied), applied
    assert s.get_version(out['attempts'][0]['version']).terms['quadratic:CORD'].weight == 2.5


def _two_option_turn(log, s):
    rev = log.begin('lower esophagus', s.current_id)
    s.set_objective_weight('quadratic:ESOPHAGUS', 60)
    a = s.solve('a')
    s.set_objective_weight('quadratic:ESOPHAGUS', 120)
    b = s.solve('b')
    s.revert(a.id)
    opts = [{'key': 'accept', 'title': 'A', 'version': a.id, 'chosen': True, 'checks': [{'roi': 'X'}], 'tradeoffs': []},
            {'key': 'capped', 'title': 'B', 'version': b.id, 'chosen': False, 'checks': [{'roi': 'Y'}],
             'tradeoffs': [{'roi': 'CORD'}]}]
    st = {'outcome': 'needs_decision', 'final_version': a.id, 'resolved': {'checks': [], 'notes': ''},
          'attempts': [], 'reply': 'r', 'options': opts}
    return log.finish(rev, st, s.current_id), a, b


def test_choose_option_switches_the_plan_and_waits_for_review():
    s, log = _session(), RevisionLog()
    rev, a, b = _two_option_turn(log, s)
    assert rev.status == 'review' and rev.options[0]['chosen']
    log.choose(rev.id, 'capped', s)
    assert s.current_id == b.id and rev.final_version == b.id and rev.status == 'review' and rev.chosen == 'capped'
    assert [o['chosen'] for o in rev.options] == [False, True] and rev.checks == [{'roi': 'Y'}]
    assert log.current_id == rev.id
    with pytest.raises(ValueError):
        log.choose(rev.id, 'nope', s)
    log.begin('next', s.current_id)
    log.items[-1].status = 'no_change'
    with pytest.raises(ValueError):
        log.choose(rev.id, 'accept', s)                                    # only the latest request
    restored = RevisionLog.from_state(log.state())
    assert restored.get(rev.id).chosen == 'capped' and len(restored.get(rev.id).options) == 2
