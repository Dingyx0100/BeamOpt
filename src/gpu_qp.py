"""GPU QP solver for the plan optimizer: restarted PDHG (PDQP-style) in PyTorch, plugged into cvxpy as a solver.

    min 1/2 x'Px + q'x   s.t.  A x = b,  F x <= g          (cvxpy's QP form)

Why: on the larger lung patients (24k QP variables) it solves an agent re-solve in ~5 s against ~44 s for Clarabel,
with dose metrics within 0.03 Gy (benchmark 2026-11-09). It only multiplies sparse matrices, so time grows slowly with
problem size. It is a first-order method (about 1e-4 relative accuracy, not 1e-8), so every answer is checked here:
  - it must converge and satisfy every constraint row to `viol_tol` (in the row's own units, per-fraction Gy for dose
    rows), otherwise the solve reports SOLVER_ERROR and PlanSession re-solves with Clarabel;
  - PlanSession's active-set loop then checks every max-dose voxel of the full problem, as for any solver.
Duals are returned in OSQP's convention, so per-constraint explanations (PlanSession._duals) keep working.
"""
from __future__ import annotations

import time
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP

NAME = 'PDHG_GPU'


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing or broken driver: no GPU path
        return False


# ------------------------------------------------------------------ PDHG core
def _split(A, l, u, n):
    """Rows with one nonzero become variable bounds (x >= 0 etc.); returns the rest and the bounds."""
    A = sp.csr_matrix(A)
    single = np.diff(A.indptr) == 1
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    for i in np.flatnonzero(single):
        j, a = A.indices[A.indptr[i]], A.data[A.indptr[i]]
        lo, hi = (l[i] / a, u[i] / a) if a > 0 else (u[i] / a, l[i] / a)
        lb[j], ub[j] = max(lb[j], lo), min(ub[j], hi)
    keep = ~single
    return A[keep], l[keep], u[keep], lb, ub, keep


def _ruiz(P, K, iters=10):
    n, m = P.shape[0], K.shape[0]
    D, E = np.ones(n), np.ones(m)
    Ps, Ks = P.copy().tocsc(), K.copy().tocsc()
    for _ in range(iters):
        cp_ = np.sqrt(np.maximum(abs(Ps).max(axis=0).toarray().ravel(), abs(Ks).max(axis=0).toarray().ravel()))
        cp_[cp_ < 1e-8] = 1.0
        rk = np.sqrt(abs(Ks).max(axis=1).toarray().ravel())
        rk[rk < 1e-8] = 1.0
        d, e = 1 / cp_, 1 / rk
        Ps = sp.diags(d) @ Ps @ sp.diags(d)
        Ks = sp.diags(e) @ Ks @ sp.diags(d)
        D, E = D * d, E * e
    return Ps.tocsr(), Ks.tocsr(), D, E


def _norm_est(mv, n, dev, torch, it=30):
    g = torch.Generator(device=dev).manual_seed(0)              # deterministic step sizes
    v = torch.randn(n, device=dev, generator=g)
    v /= v.norm()
    s_ = 0.0
    for _ in range(it):
        w = mv(v)
        s_ = float(w.norm())
        v = w / max(s_, 1e-30)
    return s_


def solve_pdhg(P, q, A, l, u, tol=1e-4, max_iter=100000, check_every=64, device='cuda'):
    """Returns (x, y, info): primal x, duals y for every row of A (OSQP sign convention: P x + q + A'y = 0)."""
    import torch
    dev = device
    n = P.shape[0]
    P = sp.csc_matrix(P)
    K, cl, cu, lb, ub, keep = _split(A, l, u, n)
    Ps, Ks, D, E = _ruiz(P, sp.csc_matrix(K))
    qs = q * D
    c_scale = 1.0 / max(1.0, float(np.abs(qs).max()))

    def t(M):
        M = sp.csr_matrix(M, dtype=np.float32)
        with warnings.catch_warnings():                         # "sparse CSR is beta" notices
            warnings.simplefilter('ignore', UserWarning)
            return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)),
                                           torch.from_numpy(M.indices.astype(np.int64)),
                                           torch.from_numpy(M.data), size=M.shape, device=dev)

    def f32(a):
        return torch.as_tensor(np.nan_to_num(a, posinf=3e38, neginf=-3e38), dtype=torch.float32, device=dev)

    Pt, Kt, KTt = t(Ps * c_scale), t(Ks), t(Ks.T.tocsr())
    qt, lbt, ubt, clt, cut = f32(qs * c_scale), f32(lb / D), f32(ub / D), f32(cl * E), f32(cu * E)
    LP = _norm_est(lambda v: Pt @ v, n, dev, torch)
    LK = _norm_est(lambda v: KTt @ (Kt @ v), n, dev, torch) ** 0.5
    x = torch.clamp(torch.zeros(n, device=dev), lbt, ubt)
    y = torch.zeros(K.shape[0], device=dev)
    w = 1.0                                                   # primal weight

    def steps(w):
        a = LP / w
        st = (-a + (a * a + 4 * 0.95 * LK * LK) ** 0.5) / (2 * LK * LK)
        return st / w, st * w

    def kkt(x, y):
        Kx = Kt @ x
        rp = (Kx - torch.clamp(Kx, clt, cut)).norm()
        g = Pt @ x + qt + KTt @ y
        r = g.clone()
        r[(x <= lbt + 1e-7) & (g > 0)] = 0
        r[(x >= ubt - 1e-7) & (g < 0)] = 0
        return float(rp) / (1 + max(float(Kx.norm()), 1.0)), float(r.norm()) / (1 + float(qt.norm())
                                                                                  + float((Pt @ x).norm()))

    tau, sig = steps(w)
    xs, ys, cnt = torch.zeros_like(x), torch.zeros_like(y), 0
    x_last, y_last = x.clone(), y.clone()
    e_last = max(kkt(x, y))
    status, it = 'max_iter', 0
    for it in range(1, max_iter + 1):
        g = Pt @ x + qt + KTt @ y
        xn = torch.clamp(x - tau * g, lbt, ubt)
        v = y + sig * (Kt @ (2 * xn - x))
        yn = v - sig * torch.clamp(v / sig, clt, cut)
        x, y = xn, yn
        xs += x
        ys += y
        cnt += 1
        if it % check_every == 0:
            xa, ya = xs / cnt, ys / cnt
            ec, ea = max(kkt(x, y)), max(kkt(xa, ya))
            bx, by, eb = (xa, ya, ea) if ea < ec else (x, y, ec)
            if eb <= tol:
                x, y, status = bx, by, 'converged'
                break
            if eb <= 0.2 * e_last or cnt >= 20 * check_every:     # adaptive / artificial restart
                dx, dy = float((bx - x_last).norm()), float((by - y_last).norm())
                if dx > 1e-10 and dy > 1e-10:
                    w = float(np.exp(0.5 * np.log(dy / dx) + 0.5 * np.log(w)))
                    tau, sig = steps(w)
                x, y = bx.clone(), by.clone()
                x_last, y_last, e_last = x.clone(), y.clone(), eb
                xs.zero_()
                ys.zero_()
                cnt = 0
    kk = max(kkt(x, y))
    xo = np.clip(x.double().cpu().numpy() * D, lb, ub)
    yk = y.double().cpu().numpy() * E / c_scale                # unscaled duals of the kept rows
    # Duals of the single-entry rows (bounds): the reduced gradient on the variable, given to its active row.
    A = sp.csr_matrix(A)
    yfull = np.zeros(A.shape[0])
    yfull[keep] = yk
    r = P @ xo + q + K.T @ yk
    for i in np.flatnonzero(~keep):
        j, a = A.indices[A.indptr[i]], A.data[A.indptr[i]]
        ax = a * xo[j]
        at_up = np.isfinite(u[i]) and abs(ax - u[i]) <= 1e-6 * max(1.0, abs(u[i]))
        at_lo = np.isfinite(l[i]) and abs(ax - l[i]) <= 1e-6 * max(1.0, abs(l[i]))
        if (at_up and -r[j] / a > 0) or (at_lo and -r[j] / a < 0):
            yfull[i] = -r[j] / a
            r[j] = 0.0                                        # one bound row per variable takes it
    return xo, yfull, {'iter': it, 'status': status, 'kkt': kk}


# ------------------------------------------------------------------ cvxpy interface
class PdhgGpu(OSQP):
    """cvxpy QP solver: reuses OSQP's problem form and dual mapping, solves with GPU PDHG."""
    STATUS_MAP = {1: s.OPTIMAL, 10: s.SOLVER_ERROR}

    def __init__(self, tol: float = 1e-4, viol_tol: float = 1e-3, max_iter: int = 100000):
        super().__init__()
        self.tol, self.viol_tol, self.max_iter = tol, viol_tol, max_iter
        self.last_info: dict = {}

    def name(self):
        return NAME

    def import_solver(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise ImportError('CUDA not available')

    def invert(self, solution, inverse_data):
        from cvxpy.reductions.solution import Solution, failure_solution
        import cvxpy.interface as intf
        attr = {s.SOLVE_TIME: solution.info.run_time, s.EXTRA_STATS: solution}
        if solution.info.status_val != 1:
            return failure_solution(s.SOLVER_ERROR, attr)
        attr[s.NUM_ITERS] = solution.info.iter
        primal = {OSQP.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))}
        duals = self.extract_duals_from_solution_attribute(solution.y, inverse_data)
        return Solution(s.OPTIMAL, solution.info.obj_val + inverse_data[s.OFFSET], primal, duals, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        t0 = time.time()
        P = sp.csc_matrix(data[s.P]) if data[s.P] is not None else sp.csc_matrix((len(data[s.Q]),) * 2)
        P = P + sp.triu(P, 1).T if sp.tril(P, -1).nnz == 0 and sp.triu(P, 1).nnz else P   # full symmetric
        q = np.asarray(data[s.Q], dtype=float)
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        u = np.concatenate((data[s.B], data[s.G]))
        l = np.concatenate([data[s.B], -np.inf * np.ones(data[s.G].shape)])
        info = {'status': 'error'}
        x = y = None
        try:
            x, y, info = solve_pdhg(P, q, A, l, u, tol=self.tol, max_iter=self.max_iter)
            Ax = A @ x
            info['max_viol'] = float(max(np.max(np.maximum(l - Ax, 0), initial=0.0),
                                         np.max(np.maximum(Ax - u, 0), initial=0.0)))
            ok = info['status'] == 'converged' and info['max_viol'] <= self.viol_tol and np.all(np.isfinite(x))
        except Exception as exc:  # noqa: BLE001 - CUDA out of memory, driver errors: fall back to the CPU solver
            info = {'status': f'error: {exc}'}
            ok = False
        info['seconds'] = round(time.time() - t0, 2)
        self.last_info = info
        obj = float(0.5 * x @ (P @ x) + q @ x) if ok else None
        res = SimpleNamespace(x=x, y=y, info=SimpleNamespace(status_val=1 if ok else 10, run_time=info['seconds'],
                                                             iter=info.get('iter', 0), obj_val=obj))
        return res
