"""v5 agent fixes: data-availability judge, limit changes, attempt handling, reply checks."""
from src.agent import feasibility as F
from src.agent import reply_check as RC
from src.agent.graph import build_graph, run_turn
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetConstraint, SetWeight
from tests.test_graph import CFG, FakeLLM, _session


# ------------------------------------------------------------------ data-availability judge
def test_heart_out_of_box_explained_even_if_llm_says_out_of_scope():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='out_of_scope')]})
    out = run_turn(build_graph(s, llm, CFG), 'Reduce the heart mean dose by 3 Gy.')
    assert out['outcome'] == 'unavailable'
    assert 'heart' in out['reply'].lower() and 'dose calculation box' in out['reply']
    assert s.current_id == 0


def test_absent_contour_explained():
    s = _session()
    msg = F.explain_unavailable(s, 'LUNGS_NOT_GTV')
    assert 'no LUNGS_NOT_GTV contour' in msg
    assert F.explain_unavailable(s, 'ESOPHAGUS') is None
    assert 'cover only the 50%' in F.partial_note(s, 'CORD')


# ------------------------------------------------------------------ limit changes
def test_parse_limit_request():
    assert F.parse_limit_request('Relax the cord max limit to 55 Gy.') == {
        'direction': 'loosen', 'roi': 'CORD', 'kind': 'max', 'value': 55.0}
    assert F.parse_limit_request('allow the spinal cord to go up to 5200 cGy')['value'] == 52.0
    assert F.parse_limit_request('Tighten the esophagus max constraint to 50 Gy')['direction'] == 'tighten'
    for txt in ('Lower the cord max dose to 15 Gy.', 'Reduce the esophagus mean but stay under the protocol limit.',
                'Spinal cord max dose should be under 20 Gy.', 'Drop PTV D95 to 55 Gy.'):
        assert F.parse_limit_request(txt) is None, txt


def test_relax_protocol_limit_is_understood_and_explained():
    s = _session()
    # v4 failure mode: the LLM parses it as "cord <= 55" (already met). The deterministic parser overrides that.
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='CORD', metric='Dmax', change='reduce_to', value=55)])]})
    out = run_turn(build_graph(s, llm, CFG), 'Relax the cord max limit to 55 Gy.')
    assert out['outcome'] == 'limit_refused'
    assert 'raise the CORD max-dose limit from 50 Gy to 55 Gy' in out['reply'] and 'protocol' in out['reply']
    assert 'already' not in out['reply']
    assert s.current_id == 0


def test_plain_dose_request_misread_as_limit_change_becomes_a_goal():
    # round-2 failure mode: "make sure the cord max is below 55 Gy" (already met) was read as change_limit and
    # answered with "I cannot raise the protocol limit". Without limit wording it is an ordinary dose goal.
    from src.agent.schemas import LimitChange
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='change_limit', limit=LimitChange(
        roi='CORD', kind='max', direction='loosen', value=55))]})
    out = run_turn(build_graph(s, llm, CFG), 'Make sure the spinal cord max dose is below 55 Gy.')
    assert out['outcome'] == 'already_met', (out['outcome'], out['reply'])
    assert 'limit' not in out['reply'].lower().split('results (deterministic')[0] or 'protocol' not in out['reply']
    assert F.mentions_limit('Relax the cord constraint to 52 Gy') and not F.mentions_limit('keep cord below 20 Gy')


def test_relax_agent_limit_is_applied():
    s = _session()
    cur = s.current.metrics['CORD']['Dmax']
    s.set_dose_constraint('CORD', 'max', round(cur - 0.2, 2))
    s.solve()
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='change_limit')]})
    out = run_turn(build_graph(s, llm, CFG), 'Relax the cord max limit to 45 Gy.')
    assert out['outcome'] == 'limit_changed', out['reply']
    assert s.constraints['agent_max:CORD'].limit_gy == 45
    assert out['final_version'] == s.current_id == 2


# ------------------------------------------------------------------ attempt handling
def test_lowering_protected_weight_and_ptv_hold_rejected():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    llm = FakeLLM({
        'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
            GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [
            Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100),
                                                SetWeight(term_id='quadratic_underdose:PTV', weight=15000)]),
        ] + [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100)])] * 3,
    })
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    first = out['attempts'][0]
    assert any('holds PTV coverage' in e for e in first['errors'])
    assert s.get_version(out['final_version']).terms['quadratic_underdose:PTV'].weight == 100000


def test_repeat_setting_is_auto_adjusted():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    same = Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=400)])
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [same, same, same]})
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    a1, a2 = out['attempts'][0], out['attempts'][1]
    assert a1['rolled_back_to'] == 0                      # w=400 overshoots PTV (see test_graph)
    assert a2.get('auto_adjusted') and a2['actions'][0]['weight'] == 210   # halfway 20 -> 400
    assert a2['applied']


def test_protective_cap_clamped_to_start_value():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    cord = s.current.metrics['CORD']['Dmax']
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100)],
                              set_constraints=[SetConstraint(roi='CORD', kind='max', limit_gy=cord + 2)])]})
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    assert any('protective cap on CORD' in x for x in out['attempts'][0]['applied'])
    assert abs(s.constraints['agent_max:CORD'].limit_gy - round(cord, 2)) < 1e-9


# ------------------------------------------------------------------ reply checks
def test_reply_checks():
    facts = [18.68, 0.5, 17.2, 16.9, 61.2, 60.9]
    assert RC.unknown_numbers('Esophagus mean is 18.68 Gy, down 0.3 Gy.', facts) == []
    assert RC.unknown_numbers('Esophagus mean fell to 12.4 Gy.', facts) == ['12.4']
    assert RC.false_comparisons('The esophagus mean of 18.68 Gy is now below the 0.5 Gy target.')
    assert RC.false_comparisons('Cord max rose from 17.2 to 16.9 Gy.')
    assert not RC.false_comparisons('Cord max fell from 17.2 to 16.9 Gy.')
    assert RC.false_success('The goal was successfully achieved.', 'not_met')
    assert not RC.false_success('The goal was not met.', 'not_met')
    text, dropped = RC.strip_out_of_scope('Cord max is 16.9 Gy. Consider changing the beam angles to spare it.')
    assert text == 'Cord max is 16.9 Gy.' and len(dropped) == 1


def test_bad_report_falls_back_to_deterministic_reply():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']

    class LiarLLM(FakeLLM):
        def invoke(self, _messages):
            from types import SimpleNamespace
            return SimpleNamespace(content='Esophagus mean dropped to 1.234 Gy. Try new beam angles.')

    llm = LiarLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100)])]})
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    assert '1.234' not in out['reply'] and 'beam angles' not in out['reply']
    assert out['reply'].startswith('The requested goal was met.')
    assert 'used deterministic reply' in out['reply_checks']
