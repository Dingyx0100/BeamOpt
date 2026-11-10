"""Web app API on the synthetic case: requests stream over SSE, revisions get Keep / Discard / Branch."""
import threading

import pytest
from fastapi.testclient import TestClient

from app import server
from app.revisions import RevisionLog
from app.workspace import Workspace
from src.agent.graph import build_graph
from src.agent.schemas import GoalItem, PhysicistGoal, Proposal, SetWeight
from tests.test_graph import CFG, FakeLLM, _session


def _workspace(llm):
    w = object.__new__(Workspace)            # skip patient loading: wire the synthetic case in directly
    w.session = _session()
    w.case = w.session.case
    w.graph = build_graph(w.session, llm, CFG)
    w.revlog, w.lock, w.cancel = RevisionLog(), threading.Lock(), threading.Event()
    w.history, w.turns, w.results = [], [], {0: 'baseline'}
    w.pid, w.reference = 'Fake_Patient', None
    w.sync = lambda: None
    return w


def _eso_llm(base, n=1):
    return FakeLLM({
        'PhysicistGoal': [PhysicistGoal(intent='optimize', objectives=[
            GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=0.05 * base)])] * n,
        'Proposal': [Proposal(reasoning='raise eso weight', set_weights=[
            SetWeight(term_id='quadratic:ESOPHAGUS', weight=w)]) for w in (100, 150, 200)] * n})


@pytest.fixture()
def client():
    w = _workspace(FakeLLM({}))
    server.S.ws, server.S.turn = w, None
    yield TestClient(server.app), w
    server.S.ws, server.S.turn = None, None


def _run(c, text):
    r = c.post('/api/turn', json={'text': text})
    assert r.status_code == 200, r.text
    body = c.get('/api/events').text           # the stream ends with 'done' once the turn finishes
    assert 'event: done' in body and 'event: step' in body
    return r.json()['revision']


def test_turn_streams_and_creates_review(client):
    c, w = client
    w.graph = build_graph(w.session, _eso_llm(w.session.current.metrics['ESOPHAGUS']['Dmean']), CFG)
    rid = _run(c, 'Reduce the esophagus mean dose a bit.')
    rev = c.get('/api/revisions').json()
    r = next(x for x in rev['items'] if x['id'] == rid)
    assert r['status'] == 'review' and rev['waiting'] == rid and rev['current'] == rid
    assert r['changes'] and r['checks'] and r['attempts']
    assert r['final_version'] == w.session.current_id != 0


def test_discard_and_branch_endpoints(client):
    c, w = client
    w.graph = build_graph(w.session, _eso_llm(w.session.current.metrics['ESOPHAGUS']['Dmean']), CFG)
    rid = _run(c, 'Reduce the esophagus mean dose a bit.')
    out = c.post(f'/api/revisions/{rid}/discard').json()
    assert out['current'] == 'R0' and w.session.current_id == 0
    assert c.post(f'/api/revisions/{rid}/keep').status_code == 400          # discarded: nothing to keep
    assert c.post('/api/revisions/R0/branch').status_code == 200


def test_locks_are_added_to_the_request(client):
    c, w = client
    seen = {}

    class Spy(FakeLLM):
        def with_structured_output(self, schema, **kw):
            inner = super().with_structured_output(schema, **kw)

            class W:
                def invoke(_, msgs):
                    seen.setdefault('human', msgs[-1][1])
                    return inner.invoke(msgs)
            return W()
    w.graph = build_graph(w.session, Spy({'PhysicistGoal': [PhysicistGoal(intent='question')]}), CFG)
    c.post('/api/turn', json={'text': 'What is the cord max?', 'locks': ['CORD', 'NOT_A_ROI']})
    c.get('/api/events')
    assert seen['human'].endswith('Keep spinal cord from getting worse.') or 'Keep' in seen['human']
    assert 'NOT_A_ROI' not in seen['human']


def test_busy_and_empty_requests_rejected(client):
    c, w = client
    assert c.post('/api/turn', json={'text': '  '}).status_code == 400
    w.lock.acquire()
    try:
        assert c.post('/api/turn', json={'text': 'hello'}).status_code == 409
    finally:
        w.lock.release()
