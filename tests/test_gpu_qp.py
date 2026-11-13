"""GPU solver (src/gpu_qp.py): same answer as Clarabel, usable duals, and a safe fallback when its answer fails."""
import cvxpy as cp
import numpy as np
import pytest

from src.gpu_qp import PdhgGpu, gpu_available
from src.plan_session import PlanSession
from tests.fake_case import make_case

pytestmark = pytest.mark.skipif(not gpu_available(), reason='needs CUDA')


def _qp(seed=0):
    rng = np.random.default_rng(seed)
    n, m = 40, 60
    A = np.abs(rng.normal(size=(m, n)))
    x = cp.Variable(n, nonneg=True)
    lim = cp.Constant(np.full(m, 5.0))
    con = [A @ x <= lim, cp.sum(x) >= 3.0]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A[:10] @ x - 8.0) + 0.1 * cp.sum(x)), con)
    return prob, x, con


def test_matches_clarabel_including_duals():
    prob, x, con = _qp()
    prob.solve(solver='CLARABEL')
    x_ref, d_ref, f_ref = x.value.copy(), [np.atleast_1d(c.dual_value).copy() for c in con], prob.value
    prob.solve(solver=PdhgGpu())
    assert prob.status == 'optimal'
    assert abs(prob.value - f_ref) <= 1e-3 * abs(f_ref)
    assert np.linalg.norm(x.value - x_ref) <= 2e-2 * np.linalg.norm(x_ref)
    for d, r in zip([np.atleast_1d(c.dual_value) for c in con], d_ref):
        assert np.linalg.norm(d - r) <= 5e-2 * max(np.linalg.norm(r), 1e-3)


def test_session_gpu_plan_matches_cpu_plan():
    cpu, gpu = PlanSession(make_case()), PlanSession(make_case())
    gpu.configure_gpu({'mode': 'on'})
    a, b = cpu.solve('cpu'), gpu.solve('gpu')
    assert b.ok and b.solver_used == 'PDHG_GPU' and b.gpu_fallbacks == 0
    for roi, mets in a.metrics.items():
        for k, v in mets.items():
            if isinstance(v, (int, float)) and isinstance(b.metrics[roi].get(k), (int, float)):
                assert abs(b.metrics[roi][k] - v) <= 0.1, (roi, k, v, b.metrics[roi][k])


def test_rejected_gpu_answer_falls_back_to_cpu():
    s = PlanSession(make_case())
    s.configure_gpu({'mode': 'on'})
    s._gpu_solver.max_iter = 1                                 # stops unconverged -> every GPU answer is rejected
    v = s.solve('fallback')
    assert v.ok and v.solver_used == 'CLARABEL' and v.gpu_fallbacks >= 1


def test_mode_parsing_and_auto_threshold():
    s = PlanSession(make_case())
    s.configure_gpu({'mode': False})                         # YAML reads a bare `off` as False
    assert s.gpu == 'off' and not s.use_gpu()
    s.configure_gpu({'mode': 'auto', 'min_beamlets': s.n_beamlets + 1})
    assert not s.use_gpu()
    s.configure_gpu({'mode': 'auto', 'min_beamlets': 1})
    assert s.use_gpu()
    with pytest.raises(ValueError):
        s.configure_gpu({'mode': 'fast'})
