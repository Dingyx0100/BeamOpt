import numpy as np
import pytest

from src.plan_session import PlanSession, ValidationError
from tests.fake_case import make_case


@pytest.fixture()
def sess():
    s = PlanSession(make_case())
    s.solve()
    return s


def test_protocol_knobs_mirror_portpy(sess):
    t = sess.protocol_terms
    assert t['quadratic_underdose:PTV'].dose_gy == 60
    assert t['linear_overdose:CORD'].weight == 0  # PortPy 1.2.0 ignores this type
    c = sess.protocol_constraints
    assert c['max:CORD'].limit_gy == 50 and c['mean:ESOPHAGUS'].limit_gy == 34


def test_gram_equals_sum_squares():
    import cvxpy as cp
    s = PlanSession(make_case())
    r = s.roi('ESOPHAGUS')
    x = np.random.default_rng(0).random(s.n_beamlets)
    assert np.isclose(x @ r.gram() @ x, np.sum((r.A @ x) ** 2), rtol=1e-5)
    assert cp is not None


def test_baseline_solves_and_respects_max(sess):
    v = sess.current
    assert v.ok
    assert v.dose_total_gy[sess.roi('CORD').idx].max() <= 50 + 1e-3
    assert v.metrics['HEART']['status'] == 'unavailable'


def test_esophagus_weight_lowers_mean(sess):
    base = sess.current.metrics['ESOPHAGUS']['Dmean']
    sess.set_objective_weight('quadratic:ESOPHAGUS', 2000)
    v = sess.solve()
    assert v.ok and v.metrics['ESOPHAGUS']['Dmean'] < base
    assert v.parent == 0 and 'weight quadratic:ESOPHAGUS' in v.changes[0]


def test_agent_mean_constraint_is_honored(sess):
    base = sess.current.metrics['ESOPHAGUS']['Dmean']
    sess.set_dose_constraint('ESOPHAGUS', 'mean', 0.8 * base)
    v = sess.solve()
    assert v.ok and v.metrics['ESOPHAGUS']['Dmean'] <= 0.8 * base + 1e-2


@pytest.mark.parametrize('call, msg', [
    (lambda s: s.set_objective_weight('quadratic:ESOPHAGUS', -1), 'Weight'),
    (lambda s: s.set_objective_weight('nope', 1), 'Unknown objective'),
    (lambda s: s.add_objective('HEART', 'mean', 10), 'no optimization voxels'),
    (lambda s: s.add_objective('LIVER', 'mean', 10), 'Unknown ROI'),
    (lambda s: s.add_objective('PTV', 'mean', 10), 'target'),
    (lambda s: s.set_dose_constraint('CORD', 'max', 55), 'relax'),
    (lambda s: s.set_objective_weight('quadratic_underdose:PTV', 10), 'coverage'),
    (lambda s: s.set_dose_constraint('PTV', 'max', 50), 'prescription'),
])
def test_validation_rejects(sess, call, msg):
    with pytest.raises(ValidationError, match=msg):
        call(sess)


def test_infeasible_rolls_back(sess):
    before = {k: v.weight for k, v in sess.terms.items()}
    sess.set_dose_constraint('ESOPHAGUS', 'max', 0.001)   # forces ~0 dose next to PTV
    sess.set_objective_weight('quadratic:ESOPHAGUS', 999)
    v = sess.solve()
    if not v.ok:   # infeasible or coverage-starved; either way current must not move
        assert sess.current_id == 0
        assert {k: t.weight for k, t in sess.terms.items()} == before
        assert 'agent_max:ESOPHAGUS' not in sess.constraints


def test_revert_and_compare(sess):
    sess.set_objective_weight('quadratic:ESOPHAGUS', 500)
    sess.solve()
    d = sess.compare(0, 1)
    assert d['ESOPHAGUS']['Dmean'] < 0
    sess.revert(0)
    assert sess.current_id == 0 and sess.terms['quadratic:ESOPHAGUS'].weight == 20
