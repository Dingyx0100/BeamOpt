from types import SimpleNamespace

from src.agent.graph import build_graph, run_turn
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetWeight
from src.plan_session import PlanSession
from tests.fake_case import make_case


class FakeLLM:
    """Returns queued structured objects per schema; plain invoke returns a canned reply."""

    def __init__(self, queue):
        self.queue = {k: list(v) for k, v in queue.items()}

    def with_structured_output(self, schema, method=None):
        outer = self

        class _R:
            def invoke(self, _messages):
                return outer.queue[schema.__name__].pop(0)
        return _R()

    def invoke(self, _messages):
        return SimpleNamespace(content='<think>x</think>Report ok.')


CFG = {'agent': {'max_iterations': 3, 'ptv_d95_tolerance_gy': 0.5}}


def _session():
    s = PlanSession(make_case())
    s.solve()
    return s


def test_iterates_until_goal_met():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    llm = FakeLLM({
        'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
            GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [
            Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:NOPE', weight=5)]),   # invalid
            Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=400)]),  # overshoots PTV
            Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100)]),  # moderated
        ],
    })
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    assert out['outcome'] == 'met'
    assert out['action_errors'] == 1
    assert out['iteration'] == 3
    final = s.get_version(out['final_version'])
    assert final.terms['quadratic:ESOPHAGUS'].weight == 100   # the protection-respecting attempt was kept
    # the overshoot (w=400) was rolled back to baseline knobs before the moderated attempt
    assert final.parent == 0
    assert out['reply'].startswith('Report ok.') and 'deterministic check' in out['reply']
    assert s.current_id == out['final_version'] and s.current_id >= 1


def test_unavailable_roi_refused_honestly():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='HEART', metric='Dmean', change='reduce')])], 'Proposal': []})
    out = run_turn(build_graph(s, llm, CFG), 'reduce heart dose')
    assert out['outcome'] == 'unavailable' and 'dose calculation box' in out['reply']
    assert s.current_id == 0


def test_lowering_target_coverage_refused():
    s = _session()
    llm = FakeLLM({'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
        GoalItem(roi='PTV', metric='D95', change='reduce_to', value=50)])], 'Proposal': []})
    out = run_turn(build_graph(s, llm, CFG), 'drop PTV coverage to 50 Gy')
    assert out['outcome'] == 'refused'


def test_change_on_unrelated_organ_is_rejected():
    s = _session()
    base = s.current.metrics['ESOPHAGUS']['Dmean']
    other = next(t for t in s.terms if t.startswith('quadratic:') and not t.endswith(('ESOPHAGUS', 'PTV')))
    llm = FakeLLM({
        'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
            GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])],
        'Proposal': [
            Proposal(reasoning='', set_weights=[SetWeight(term_id=other, weight=500)]),               # wrong organ
            Proposal(reasoning='', set_weights=[SetWeight(term_id='quadratic:ESOPHAGUS', weight=100)]),
        ],
    })
    out = run_turn(build_graph(s, llm, CFG), 'lower esophagus mean a bit')
    first = out['attempts'][0]
    assert not first['applied'] and 'acts on ESOPHAGUS' in first['errors'][0]
    assert out['outcome'] == 'met'
    assert s.get_version(out['final_version']).terms[other].weight != 500
