import json

import numpy as np
import pytest

from src.plan_export import PlanStore, list_sessions, restore_versions
from src.plan_session import PlanSession
from tests.fake_case import make_case


def _worked_session():
    s = PlanSession(make_case())
    s.solve('baseline (protocol defaults)')
    s.set_objective_weight('quadratic:ESOPHAGUS', 100)
    s.checkpoints.append(0)
    s.solve('lower esophagus [iter 1]')
    s.add_objective('CORD', 'mean', 50)
    s.solve('lower esophagus [iter 2]')
    s.revert(1)                                  # physicist picked iter 1 as working plan
    return s


def test_baseline_only_session_writes_nothing(tmp_path):
    s = PlanSession(make_case())
    s.solve()
    store = PlanStore(tmp_path, 'P1')
    assert store.sync(s) == [] and not store.dir.exists()
    assert list_sessions(tmp_path) == []


def test_save_and_restore_roundtrip(tmp_path):
    s = _worked_session()
    store = PlanStore(tmp_path, 'P1')
    store.sync(s, {'turns': [{'request': 'lower esophagus'}], 'results': {'1': 'lower esophagus'}})
    found = list_sessions(tmp_path)
    assert len(found) == 1 and found[0].patient_id == 'P1' and found[0].n_versions == 3
    assert found[0].last_request == 'lower esophagus'
    assert list_sessions(tmp_path / 'P1')[0].path == found[0].path      # patient folder works too
    assert list_sessions(found[0].path)[0].path == found[0].path         # and the session folder itself

    r = PlanSession(make_case())                 # "after a restart"
    r.solve()
    st = restore_versions(r, found[0].path)
    assert st['results'] == {'1': 'lower esophagus'}
    assert len(r.history) == 3 and r.current_id == 1 and r.checkpoints == [0]
    assert r.terms['quadratic:ESOPHAGUS'].weight == 100 and 'mean:CORD' not in r.terms
    for a, b in zip(s.history, r.history):
        assert a.label == b.label and a.parent == b.parent
        np.testing.assert_allclose(a.dose_total_gy, b.dose_total_gy)
        assert a.metrics == b.metrics
    # continuing the restored session appends to the same folder
    cont = PlanStore(tmp_path, 'P1', directory=found[0].path)
    r.set_objective_weight('quadratic:CORD', 50)
    r.solve('next request [iter 1]')
    assert [p.name for p in cont.sync(r)] == ['v3.json']
    assert json.loads((found[0].path / 'session.json').read_text())['n_versions'] == 4


def test_restore_rejects_other_patient(tmp_path):
    s = _worked_session()
    store = PlanStore(tmp_path, 'P1')
    store.sync(s)
    np.save(store.dir / 'v1_fluence.npy', np.ones(s.A_full.shape[1] + 3))
    st = json.loads((store.dir / 'session.json').read_text())
    st.pop('n_beamlets')
    (store.dir / 'session.json').write_text(json.dumps(st))
    r = PlanSession(make_case())
    r.solve()
    with pytest.raises(ValueError, match='wrong patient'):
        restore_versions(r, store.dir)
