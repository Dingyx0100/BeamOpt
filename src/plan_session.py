"""M3: a stateful, auditable plan-optimization session for the agent.

Mirrors PortPy's ``Optimization.create_cvxpy_problem`` (portpy 1.2.0) but in a
compact form that is cheap to re-solve many times:

* Every ``quadratic`` and ``smoothness-quadratic`` term is ``||M x||^2``, which
  equals ``x^T (M^T M) x``.  We precompute the dense 2558x2558 Gram matrices
  once, so the per-solve QP has only beamlet variables + PTV slacks instead of
  hundreds of thousands of voxel auxiliaries.  The objective is identical.
* Max-dose constraints use one precomputed sparse row-block whose right-hand
  side is recomputed from the current knobs.

All doses exposed to users/agents are TOTAL-course Gy.  Internally the
optimizer works per fraction (A @ x), exactly like PortPy.

Safety model (research prototype, not a TPS):
* Protocol constraints from clinical criteria are ALWAYS enforced and can only
  be tightened, never relaxed.
* The PTV prescription thresholds are not agent-editable.
* Every change is validated and recorded; failed solves roll back.
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .criteria_evaluator import evaluate_criteria, resolve_number
from .data_adapter import PatientCase, evaluate_dose, from_fluence
from .metrics import dose_at_volume_gy, volume_at_dose_percent, weighted_mean_gy
from .data_adapter import get_roi_dose

MAX_WEIGHT = 1e6
TARGET_ROIS = ('PTV', 'GTV', 'CTV')
REPORT_ROIS = ('PTV', 'GTV', 'ESOPHAGUS', 'CORD', 'HEART', 'LUNG_L', 'LUNG_R', 'LUNGS_NOT_GTV')
# OARs the agent may add terms / constraints to (filtered by available voxels).
AGENT_OARS = ('ESOPHAGUS', 'CORD', 'HEART', 'LUNG_L', 'LUNG_R', 'LUNGS_NOT_GTV')
# Linear-overdose and max-dose user terms create per-voxel rows; restrict to
# modest-size ROIs to keep the QP small.
VOXELWISE_ROIS = ('ESOPHAGUS', 'CORD', 'HEART', 'PTV')
# Large parallel organs: a linear_overdose term acts only on the voxels already near its threshold in the plan it
# starts from ("hot subset"). On Lung_Patient_36 the whole-lung term (130k voxels) took 70 s and barely moved V20;
# the hot subset (13k voxels, dose >= T - 5 Gy) took 36 s and lowered V20 7.4 -> 6.2 % at w=1000 (2026-11-09).
HOT_SUBSET_ROIS = ('LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R')
OVERDOSE_ROIS = VOXELWISE_ROIS + HOT_SUBSET_ROIS
HOT_MARGIN_GY = 5.0


class ValidationError(ValueError):
    """A requested change is not allowed; message is shown to agent and user."""


@dataclass
class ObjectiveTerm:
    id: str
    kind: str              # quadratic | quadratic_overdose | quadratic_underdose | smoothness | mean | linear_overdose
    roi: str | None
    weight: float
    dose_gy: float | None = None   # total-course threshold (overdose/underdose kinds)
    source: str = 'protocol'       # protocol | agent
    note: str = ''

    def describe(self) -> str:
        thr = f' @ {self.dose_gy:g} Gy' if self.dose_gy is not None else ''
        return f'{self.kind}({self.roi or "all beamlets"}{thr}) w={self.weight:g} [{self.source}]'


@dataclass
class DoseConstraint:
    id: str
    kind: str              # max | mean
    roi: str
    limit_gy: float        # total-course Gy
    source: str = 'protocol'
    scope: str = 'whole_organ'  # protocol mean limits are whole-organ (PortPy divides by coverage)

    def describe(self) -> str:
        return f'{self.kind}({self.roi}) <= {self.limit_gy:g} Gy [{self.source}, {self.scope}]'


@dataclass
class PlanVersion:
    id: int
    parent: int | None
    label: str
    changes: list[str]
    terms: dict[str, ObjectiveTerm]
    constraints: dict[str, DoseConstraint]
    status: str
    objective_value: float | None = None
    solve_seconds: float | None = None
    build_seconds: float | None = None
    fluence: np.ndarray | None = None
    dose_total_gy: np.ndarray | None = None
    metrics: dict | None = None
    criteria: Any = None   # pandas DataFrame
    rounds: int = 0        # constraint-generation rounds
    max_violation_gy: float | None = None   # worst max-dose excess over ALL voxels (accepted if <= tol)
    duals: dict | None = None  # constraint id -> {'dual': objective price, 'pull': elasticity} (src/explain.py)
    cached: bool = False       # result came from the solve cache (identical problem solved before)
    solver_used: str = ''      # solver of the last round ('PDHG_GPU', or the CPU solver incl. after a GPU fallback)
    gpu_fallbacks: int = 0     # rounds where the GPU answer failed its checks and the CPU solver re-solved

    @property
    def ok(self) -> bool:
        return str(self.status).startswith('optimal') and self.dose_total_gy is not None


@dataclass
class _ROIData:
    idx: np.ndarray
    vol_cc: np.ndarray
    coverage: float
    A: sp.csr_matrix                # rows of A for this ROI (per-fraction dose)
    mean_row: np.ndarray            # volume-weighted mean dose per unit x  (shape [n_beamlets])
    _gram: np.ndarray | None = field(default=None, repr=False)

    def gram(self) -> np.ndarray:
        if self._gram is None:
            G = (self.A.T @ self.A).toarray().astype(np.float64)
            self._gram = G
        return self._gram


def beamlet_grouping(beamlets_dict: list[dict], merge_x: int, n_beamlets: int) -> sp.csr_matrix:
    """Binary E [n_native x n_merged]: merge `merge_x` horizontally adjacent beamlets in the same MLC leaf row.

    Uses each beam's finest 2-D beamlet grid (2.5 mm cells). Merged beamlets are
    an exactly representable subset of native fluences (x_native = E x_merged).
    """
    group_of = np.full(n_beamlets, -1, dtype=np.int64)
    key_to_group: dict[tuple, int] = {}
    for b, beam in enumerate(beamlets_dict):
        grid = np.asarray(beam['beamlet_idx_2d_finest_grid'])
        rows, cols = np.nonzero(grid >= 0)
        ids = grid[rows, cols]
        for bid in np.unique(ids):
            sel = ids == bid
            r0, c0 = rows[sel].min(), cols[sel].min()
            key = (b, int(r0), int(c0) // merge_x)
            group_of[bid] = key_to_group.setdefault(key, len(key_to_group))
    missing = np.flatnonzero(group_of < 0)
    for bid in missing:   # beamlets absent from the grid stay unmerged
        group_of[bid] = len(key_to_group)
        key_to_group[('solo', int(bid))] = group_of[bid]
    return sp.csr_matrix((np.ones(n_beamlets), (np.arange(n_beamlets), group_of)),
                         shape=(n_beamlets, len(key_to_group)))


PROBE_GPU_VIOL_GY = 0.5     # floor probe: a GPU answer may exceed a row by this (total Gy) before a CPU re-solve; an
                            # overshoot only relaxes the probe, so the floor can only come out lower (safe side)
PROBE_GPU_MAX_ITER = 30000  # floor probe: give up on the GPU sooner (LP-like probes converge slowly), use the CPU
PROBE_PTV_START_GY = 1.0    # floor probe: PTV coverage rows start with voxels within this of the D95 floor
PROBE_SETTLE_ROWS = 20      # floor probe: stop when <= this many rows are left to add ...
PROBE_SETTLE_GY = 0.2       # ... and none is over its bound by more than this (total Gy)


class PlanSession:
    """Holds the patient case, current optimization knobs and version history."""

    def __init__(self, case: PatientCase, solver: str = 'CLARABEL', solver_opts: dict | None = None,
                 verbose: bool = False, log: bool = False, merge_x: int = 1):
        self.case = case
        self.solver = solver
        self.solver_opts = solver_opts or {}
        self.verbose = verbose   # solver output
        # GPU solver (src/gpu_qp.py): 'off' | 'auto' (only when there are >= gpu_min_beamlets beamlets) | 'on'.
        # Every GPU answer is checked; on failure the round is re-solved with `solver`.
        self.gpu, self.gpu_min_beamlets, self._gpu_solver = 'off', 1500, None
        self.log = log           # constraint-generation progress
        self._active = None
        inf = case.influence_matrix
        self.A_full = inf.A.tocsr() if sp.issparse(inf.A) else sp.csr_matrix(inf.A)
        # Optional beamlet merging: optimize x' with A' = A E, deliver fluence E x'.
        self.merge_x = int(merge_x)
        self.E = beamlet_grouping(inf.beamlets_dict, self.merge_x, self.A_full.shape[1]) if self.merge_x > 1 else None
        self.A = (self.A_full @ self.E).tocsr() if self.E is not None else self.A_full
        self.n_beamlets = self.A.shape[1]
        self.num_fx = int(case.plan.get_num_of_fractions())
        self.rx_gy = float(case.plan.get_prescription())
        self.structures = list(case.structures.get_structures())
        self._roi: dict[str, _ROIData] = {}
        self._smooth_gram: np.ndarray | None = None
        self.protocol_terms, self.protocol_constraints = self._protocol_knobs()
        self.terms = deepcopy(self.protocol_terms)
        self.constraints = deepcopy(self.protocol_constraints)
        self.history: list[PlanVersion] = []
        self.current_id: int | None = None
        self.pending_changes: list[str] = []
        self.checkpoints: list[int] = []   # plan version before each accepted agent turn (for undo)
        # Optional exact solve cache (a directory per patient): an identical problem is never solved twice.
        self.solve_cache = None

    # ------------------------------------------------------------------ setup
    def roi(self, name: str) -> _ROIData:
        if name not in self._roi:
            inf = self.case.influence_matrix
            if name not in self.structures:
                raise ValidationError(f'Unknown ROI {name!r}. Available: {self.structures}')
            idx = np.asarray(inf.get_opt_voxels_idx(name), dtype=np.int64).ravel()
            vol = np.asarray(inf.get_opt_voxels_volume_cc(name), dtype=np.float64).ravel()
            order = np.argsort(idx, kind='stable')
            idx, vol = idx[order], vol[order]
            cov = inf.get_fraction_of_vol_in_calc_box(name)
            A_roi = self.A[idx, :] if idx.size else sp.csr_matrix((0, self.n_beamlets))
            mean_row = (np.asarray(A_roi.T @ vol).ravel() / vol.sum()) if idx.size else np.zeros(self.n_beamlets)
            self._roi[name] = _ROIData(idx, vol, float(cov or 0.0), A_roi.tocsr(), mean_row)
        return self._roi[name]

    def has_voxels(self, name: str) -> bool:
        return name in self.structures and self.roi(name).idx.size > 0

    def optimizable_rois(self) -> dict[str, dict]:
        out = {}
        for name in self.structures:
            r = self.roi(name)
            out[name] = {'n_voxels': int(r.idx.size), 'coverage': round(r.coverage, 3),
                         'optimizable': bool(r.idx.size > 0),
                         'is_target': name in TARGET_ROIS}
        return out

    def _num(self, value) -> float:
        return resolve_number(value, self.rx_gy)

    def _protocol_knobs(self):
        """Translate PortPy opt_params + clinical criteria into explicit knobs."""
        terms: dict[str, ObjectiveTerm] = {}
        for i, f in enumerate(self.case.opt_params.get('objective_functions', [])):
            kind = f['type'].replace('-', '_')
            roi = f.get('structure_name')
            if kind == 'smoothness_quadratic':
                terms['smoothness'] = ObjectiveTerm('smoothness', 'smoothness', None, float(f['weight']))
                continue
            if roi not in self.structures:
                continue
            dose = None
            dose_key = next((k for k in f if 'dose' in k), None)
            if dose_key:
                dose = self._num(f[dose_key])
                if dose_key.endswith('perc'):
                    dose = dose * self.rx_gy / 100
            tid = f'{kind}:{roi}'
            note = ''
            w = float(f['weight'])
            if kind == 'linear_overdose':
                # PortPy 1.2.0 create_cvxpy_problem silently ignores this type.
                # Keep it visible but disabled so the baseline matches PortPy.
                note = f'declared in opt_params with weight {w:g} but ignored by PortPy baseline'
                w = 0.0
            if kind not in ('quadratic', 'quadratic_overdose', 'quadratic_underdose', 'linear_overdose'):
                continue
            terms[tid] = ObjectiveTerm(tid, kind, roi, w, dose, 'protocol', note)

        cons: dict[str, DoseConstraint] = {}
        crit = deepcopy(self.case.criteria.get_criteria())
        for oc in self.case.opt_params.get('constraints', []):
            if oc['parameters']['structure_name'] in self.structures:
                exists, ind = self.case.criteria.check_criterion_exists(oc, return_ind=True)
                if exists:
                    crit[ind] = oc
                else:
                    crit.append(oc)
        for c in crit:
            kind = {'max_dose': 'max', 'mean_dose': 'mean'}.get(c['type'])
            if kind is None:
                continue
            roi = c['parameters']['structure_name']
            lk = next((k for k in c['constraints'] if 'limit' in k), None)
            if not lk or roi not in self.structures:
                continue
            if kind == 'max' and roi in ('GTV', 'CTV'):
                continue  # PortPy skips GTV/CTV max constraints
            lim = self._num(c['constraints'][lk])
            if lk.endswith('perc'):
                lim = lim * self.rx_gy / 100
            cid = f'{kind}:{roi}'
            if cid in cons:
                lim = min(lim, cons[cid].limit_gy)
            cons[cid] = DoseConstraint(cid, kind, roi, lim, 'protocol',
                                       'whole_organ' if kind == 'mean' else 'voxel')
        return terms, cons

    # ------------------------------------------------------------- mutation
    def _record(self, msg: str) -> str:
        self.pending_changes.append(msg)
        return msg

    def _check_weight(self, weight) -> float:
        w = float(weight)
        if not np.isfinite(w) or not 0 <= w <= MAX_WEIGHT:
            raise ValidationError(f'Weight must be finite and in [0, {MAX_WEIGHT:g}], got {weight!r}')
        return w

    def _check_roi(self, roi: str, allowed=None, allow_target=False) -> str:
        roi = str(roi).strip().upper()
        if roi not in self.structures:
            raise ValidationError(f'Unknown ROI {roi!r}. Available ROIs: {", ".join(self.structures)}')
        if not self.has_voxels(roi):
            raise ValidationError(
                f'{roi} has no optimization voxels inside the dose calculation box for this patient, '
                f'so it cannot be optimized or evaluated here.')
        if roi in TARGET_ROIS and not allow_target:
            raise ValidationError(f'{roi} is a target; target objectives are protected in this prototype.')
        if allowed is not None and roi not in allowed:
            raise ValidationError(f'{roi} is not supported for this operation. Supported: {", ".join(allowed)}')
        return roi

    def set_objective_weight(self, term_id: str, weight: float) -> str:
        """Change the weight of an existing objective term (protocol or agent-added)."""
        if term_id not in self.terms:
            raise ValidationError(f'Unknown objective term {term_id!r}. Existing: {", ".join(self.terms)}')
        w = self._check_weight(weight)
        t = self.terms[term_id]
        if t.kind in ('quadratic_underdose',) and w < 1000:
            raise ValidationError('PTV underdose weight below 1000 would sacrifice target coverage; refused.')
        old = t.weight
        t.weight = w
        return self._record(f'weight {term_id}: {old:g} -> {w:g}')

    def add_objective(self, roi: str, kind: str, weight: float, dose_gy: float | None = None) -> str:
        """Add (or update) an agent objective term on an OAR.

        kind: 'mean' (minimize volume-weighted mean dose), 'quadratic' (minimize
        mean squared dose), or 'linear_overdose' (penalize dose above dose_gy).
        """
        kind = str(kind).strip().lower().replace('-', '_')
        if kind not in ('mean', 'quadratic', 'linear_overdose'):
            raise ValidationError("Objective kind must be one of: mean, quadratic, linear_overdose")
        allowed = OVERDOSE_ROIS if kind == 'linear_overdose' else AGENT_OARS
        roi = self._check_roi(roi, allowed=allowed)
        w = self._check_weight(weight)
        dose = None
        if kind == 'linear_overdose':
            if dose_gy is None:
                raise ValidationError('linear_overdose needs dose_gy (total-course Gy threshold)')
            dose = float(dose_gy)
            if not 0 < dose <= 1.2 * self.rx_gy:
                raise ValidationError(f'dose_gy must be in (0, {1.2 * self.rx_gy:g}] Gy')
        tid = f'{kind}:{roi}'
        if tid in self.terms:
            t = self.terms[tid]
            old = t.describe()
            t.weight, t.dose_gy = w, (dose if dose is not None else t.dose_gy)
            return self._record(f'update {old} -> {t.describe()}')
        self.terms[tid] = ObjectiveTerm(tid, kind, roi, w, dose, 'agent')
        return self._record(f'add {self.terms[tid].describe()}')

    def set_dose_constraint(self, roi: str, kind: str, limit_gy: float) -> str:
        """Add/tighten a hard constraint. Protocol limits can never be relaxed.

        Agent 'mean' limits refer to the represented (in-calc-box) sub-volume
        mean, i.e. exactly the Dmean the metrics report.
        """
        kind = str(kind).strip().lower()
        if kind not in ('max', 'mean'):
            raise ValidationError("Constraint kind must be 'max' or 'mean'")
        allowed = VOXELWISE_ROIS if kind == 'max' else AGENT_OARS
        roi = self._check_roi(roi, allowed=allowed, allow_target=(kind == 'max'))
        lim = float(limit_gy)
        if not np.isfinite(lim) or lim <= 0:
            raise ValidationError('limit_gy must be positive (total-course Gy)')
        if roi == 'PTV' and kind == 'max' and lim < 1.02 * self.rx_gy:
            raise ValidationError(f'PTV max below {1.02 * self.rx_gy:g} Gy conflicts with prescription coverage; refused.')
        pid = f'{kind}:{roi}'
        if pid in self.protocol_constraints and lim > self.protocol_constraints[pid].limit_gy and kind == 'max':
            raise ValidationError(f'Cannot relax protocol limit {self.protocol_constraints[pid].describe()}')
        cid = f'agent_{kind}:{roi}'
        old = self.constraints.get(cid)
        self.constraints[cid] = DoseConstraint(cid, kind, roi, lim, 'agent',
                                               'voxel' if kind == 'max' else 'represented_subvolume')
        return self._record(f'{"update" if old else "add"} constraint {self.constraints[cid].describe()}')

    def remove_agent_item(self, item_id: str) -> str:
        if item_id in self.terms and self.terms[item_id].source == 'agent':
            del self.terms[item_id]
            return self._record(f'remove objective {item_id}')
        if item_id in self.constraints and self.constraints[item_id].source == 'agent':
            del self.constraints[item_id]
            return self._record(f'remove constraint {item_id}')
        raise ValidationError(f'{item_id!r} is not an agent-added objective/constraint (protocol items cannot be removed)')

    def reset_to_protocol(self) -> str:
        self.terms = deepcopy(self.protocol_terms)
        self.constraints = deepcopy(self.protocol_constraints)
        return self._record('reset all knobs to protocol defaults')

    def revert(self, version_id: int) -> str:
        v = self.get_version(version_id)
        self.terms, self.constraints = deepcopy(v.terms), deepcopy(v.constraints)
        self.current_id = v.id
        self.pending_changes = []
        return f'reverted knobs and current plan to version {v.id} ({v.label})'

    # ------------------------------------------------------------- solving
    def _smoothness(self) -> np.ndarray:
        if self._smooth_gram is None:
            import portpy.photon as pp
            Qx, Qy, nr, nc = pp.Optimization.get_smoothness_matrix(self.case.influence_matrix.beamlets_dict)
            Qx = sp.csr_matrix(Qx, dtype=np.float64)
            Qy = sp.csr_matrix(Qy, dtype=np.float64)
            if self.E is not None:
                Qx, Qy = Qx @ self.E, Qy @ self.E
            self._smooth_gram = (0.6 / nc) * (Qx.T @ Qx).toarray() + (0.4 / nr) * (Qy.T @ Qy).toarray()
        return self._smooth_gram

    def _max_rhs(self) -> tuple[np.ndarray, np.ndarray]:
        """Voxel indices and per-fraction upper bounds for all active max constraints."""
        d_max = np.full(self.A.shape[0], np.inf)
        for c in self.constraints.values():
            if c.kind != 'max' or not self.has_voxels(c.roi):
                continue
            idx = self.roi(c.roi).idx
            d_max[idx] = np.minimum(d_max[idx], c.limit_gy / self.num_fx)
        rows = np.flatnonzero(np.isfinite(d_max))
        return rows, d_max[rows]

    def _max_owner(self, rows: np.ndarray) -> np.ndarray:
        """Id of the constraint that sets the (tightest) max-dose bound at each row, as in _max_rhs."""
        d_max = np.full(self.A.shape[0], np.inf)
        owner = np.full(self.A.shape[0], '', dtype=object)
        for c in self.constraints.values():
            if c.kind != 'max' or not self.has_voxels(c.roi):
                continue
            idx = self.roi(c.roi).idx
            lim = c.limit_gy / self.num_fx
            tighter = lim < d_max[idx]
            d_max[idx[tighter]] = lim
            owner[idx[tighter]] = c.id
        return owner[rows]

    def _duals(self, prob, tags) -> dict:
        """Sum the solver's dual values per hard constraint and turn them into a comparable elasticity."""
        out: dict[str, dict] = {}
        f = abs(float(prob.value)) if prob.value else 0.0
        for con, kind, payload in tags:
            dv = con.dual_value
            if dv is None:
                continue
            dv = np.asarray(dv, dtype=float).ravel()
            if kind == 'mean':
                cid, rhs = payload
                lam, scale = float(dv.sum()), float(rhs) * float(dv.sum())
                e = out.setdefault(cid, {'dual': 0.0, 'pull': 0.0})
                e['dual'] += lam
                e['pull'] += scale
                continue
            rows, rhs = payload
            owners = self._max_owner(rows)
            for cid in set(owners) - {''}:
                m = owners == cid
                e = out.setdefault(cid, {'dual': 0.0, 'pull': 0.0})
                e['dual'] += float(dv[m].sum())
                e['pull'] += float((dv[m] * rhs[m]).sum())
        for e in out.values():
            e['dual'] = round(e['dual'], 6)
            e['pull'] = round(e['pull'] / f, 6) if f > 0 else None
        return out

    def build_problem(self, active_rows: np.ndarray | None = None):
        """active_rows: subset of voxel rows to enforce max constraints on (None = all)."""
        import cvxpy as cp
        x = cp.Variable(self.n_beamlets, nonneg=True, name='x')
        P = np.zeros((self.n_beamlets, self.n_beamlets))
        lin = np.zeros(self.n_beamlets)
        obj, cons = [], []
        dose_vars: dict[str, Any] = {}
        tags = []                     # (constraint, 'max' | 'mean', payload) for dual read-out
        self._last_tags = tags

        def roi_dose(name):
            # One shared per-fraction dose variable per ROI: the dense A_roi block appears once
            # even when several voxel-wise terms (over/underdose, linear overdose) use it.
            if name not in dose_vars:
                r_ = self.roi(name)
                d_ = cp.Variable(r_.idx.size, name=f'd_{name}')
                cons.append(r_.A @ x == d_)
                dose_vars[name] = d_
            return dose_vars[name]

        for t in self.terms.values():
            if t.weight <= 0:
                continue
            if t.kind == 'smoothness':
                P += t.weight * self._smoothness()
                continue
            if not self.has_voxels(t.roi):
                continue
            r = self.roi(t.roi)
            n = r.idx.size
            if t.kind == 'quadratic':
                P += (t.weight / n) * r.gram()
            elif t.kind == 'mean':
                lin += t.weight * r.mean_row
            elif t.kind == 'quadratic_overdose':
                s = cp.Variable(n, nonneg=True)
                obj.append((t.weight / n) * cp.sum_squares(s))
                cons.append(roi_dose(t.roi) <= t.dose_gy / self.num_fx + s)
            elif t.kind == 'quadratic_underdose':
                s = cp.Variable(n, nonneg=True)
                obj.append((t.weight / n) * cp.sum_squares(s))
                cons.append(roi_dose(t.roi) >= t.dose_gy / self.num_fx - s)
            elif t.kind == 'linear_overdose':
                if t.roi in HOT_SUBSET_ROIS:
                    sub = self.hot_subset(t.roi, t.dose_gy)
                    if sub.size == 0:
                        continue
                    s = cp.Variable(sub.size, nonneg=True)
                    obj.append((t.weight / sub.size) * cp.sum(s))
                    cons.append(r.A[sub, :] @ x - t.dose_gy / self.num_fx <= s)
                    continue
                s = cp.Variable(n, nonneg=True)
                obj.append((t.weight / n) * cp.sum(s))
                cons.append(roi_dose(t.roi) - t.dose_gy / self.num_fx <= s)
        P = 0.5 * (P + P.T)
        obj.append(cp.quad_form(x, cp.psd_wrap(P)))
        if np.any(lin):
            obj.append(lin @ x)
        rows, rhs = self._max_rhs()
        if active_rows is not None:
            keep = np.isin(rows, active_rows)
            rows, rhs = rows[keep], rhs[keep]
        # Rows inside an ROI that already has a shared dose variable become cheap bounds on it.
        for name, d_ in dose_vars.items():
            ridx = self.roi(name).idx
            pos = np.searchsorted(ridx, rows)
            inside = (pos < ridx.size) & (ridx[np.minimum(pos, ridx.size - 1)] == rows)
            if inside.any():
                cons.append(d_[pos[inside]] <= rhs[inside])
                tags.append((cons[-1], 'max', (rows[inside], rhs[inside])))
                rows, rhs = rows[~inside], rhs[~inside]
        if rows.size:
            cons.append(self.A[rows, :] @ x <= rhs)
            tags.append((cons[-1], 'max', (rows, rhs)))
        for c in self.constraints.values():
            if c.kind != 'mean' or not self.has_voxels(c.roi):
                continue
            r = self.roi(c.roi)
            limit = c.limit_gy
            if c.scope == 'whole_organ':
                limit = limit / max(r.coverage, 1e-6)  # PortPy convention: out-of-box voxels assumed 0 Gy
            cons.append(r.mean_row @ x <= limit / self.num_fx)
            tags.append((cons[-1], 'mean', (c.id, limit / self.num_fx)))
        return cp.Problem(cp.Minimize(cp.sum(obj)), cons), x

    def floor_probe(self, roi: str, metric: str, ptv_d95_min: float, ptv_dmax_max: float,
                    max_rounds: int = 12, near_gy: float = 3.0, progress=None,
                    stop_above: float | None = None) -> dict:
        """Lowest `metric` of `roi` the optimizer can find while PTV coverage and every hard limit are kept.

        A probe, not a plan: the objective is ONLY this organ's metric (other organs are bounded by their hard
        protocol limits alone). PTV coverage is kept by a sufficient convex condition: the voxels that hold the
        hottest 95.5 % of the PTV volume in the current plan must stay >= ptv_d95_min, and all PTV voxels
        <= ptv_dmax_max. The result is achievable (an upper bound on the true floor), so "floor > goal" is strong
        evidence the goal cannot be met while protecting the PTV. Surrogate objectives: Dmax = max, D2 = CVaR of
        the hottest 2 %, Vx = mean overdose above x - 2 Gy; the returned value is always the real metric.

        Speed: every voxel set (organ objective voxels, PTV lower/upper rows, hard max-dose rows) starts with the
        voxels near their bound in the current plan and grows by constraint generation: after each solve, voxels
        that cross their bound (or would change the objective) are added and the problem is solved again. The
        answer equals the full problem's; the problems stay a few thousand rows instead of ~70k-190k.

        Geometry first: organ voxels inside the covered PTV must get >= ptv_d95_min, which bounds the floor from
        below with no solve (Dmax: any such voxel; Dx: >= x % of the organ inside; Vx with x <= ptv_d95_min: the
        inside share; Dmean: inside share x ptv_d95_min). When that bound already exceeds `stop_above` (the goal
        plus the verdict margin) the probe returns it at once with status 'bound'.
        """
        import cvxpy as cp
        t0 = time.time()
        say = progress or ((lambda m: print(m, flush=True)) if self.log else (lambda m: None))
        r, ptv, fx = self.roi(roi), self.roi('PTV'), self.num_fx
        dcur = np.asarray(self.current.dose_total_gy, dtype=float)
        w_all = r.vol_cc / r.vol_cc.sum()
        n_o = r.idx.size
        if metric == 'Dmean':
            kind = 'mean'
        elif metric == 'Dmax':
            kind = 'max'
        elif metric.startswith('D'):
            kind, p = 'cvar', float(metric[1:]) / 100
        elif metric.startswith('V'):
            kind, thr_v = 'hinge', (float(metric[1:]) - 2.0) / fx
        else:
            raise ValidationError(f'floor_probe: unsupported metric {metric}')
        do = dcur[r.idx]
        if kind == 'mean':
            o_set = np.zeros(0, dtype=np.int64)
        elif kind == 'hinge':
            o_set = np.flatnonzero(do >= thr_v * fx - near_gy)
        else:
            k = max(200, int(np.ceil((2 * p if kind == 'cvar' else 0.02) * n_o)))
            o_set = np.argsort(-do)[:min(k, n_o)]
        # PTV coverage: the hottest 95.5 % of the volume stays >= the D95 floor; every voxel <= the Dmax cap.
        pd_ = dcur[ptv.idx]
        order = np.argsort(-pd_)
        cum = np.cumsum(ptv.vol_cc[order]) / ptv.vol_cc.sum()
        hot = np.sort(order[:int(np.searchsorted(cum, 0.955)) + 1])
        f_in = float(w_all[np.isin(r.idx, ptv.idx[hot])].sum())          # organ volume share inside covered PTV
        if kind == 'mean':
            lb = f_in * ptv_d95_min
        elif kind == 'max':
            lb = ptv_d95_min if f_in > 0 else 0.0
        elif kind == 'cvar':
            lb = ptv_d95_min if f_in >= p else 0.0
        else:
            lb = 100 * f_in if float(metric[1:]) <= ptv_d95_min else 0.0
        if stop_above is not None and lb > stop_above:
            say(f'  probe {roi} {metric}: geometric bound {lb:.2f} > {stop_above:.2f} ({100 * f_in:.1f} % inside PTV)')
            return {'roi': roi, 'metric': metric, 'status': 'bound', 'bound': True, 'value': round(lb, 3),
                    'inside_pct': round(100 * f_in, 2), 'rounds': 0, 'seconds': round(time.time() - t0, 1)}
        lo_set = hot[pd_[hot] < ptv_d95_min + PROBE_PTV_START_GY]
        hi_set = np.flatnonzero(pd_ >= ptv_dmax_max - near_gy)
        # Hard limits stay, except the agent's own attempts on this organ.
        saved = self.constraints
        self.constraints = {k_: c for k_, c in saved.items() if not (c.roi == roi and c.source == 'agent')}
        mean_cons = []
        try:
            all_rows, all_rhs = self._max_rhs()
            for c in self.constraints.values():
                if c.kind == 'mean' and self.has_voxels(c.roi):
                    rr = self.roi(c.roi)
                    lim = c.limit_gy / max(rr.coverage, 1e-6) if c.scope == 'whole_organ' else c.limit_gy
                    mean_cons.append((rr.mean_row, lim / fx))
        finally:
            self.constraints = saved
        mx_set = np.flatnonzero(dcur[all_rows] / fx >= all_rhs - near_gy / fx)
        A_rows = self.A[all_rows, :].tocsr()
        tol = 0.05 / fx
        status, xval, rounds = 'error', None, 0
        for rounds in range(1, max_rounds + 1):
            x = cp.Variable(self.n_beamlets, nonneg=True, name='x')
            tt = None
            if kind == 'mean':
                obj = r.mean_row @ x
            elif kind == 'max':
                obj = cp.max(r.A[o_set, :] @ x)
            elif kind == 'cvar':
                tt = cp.Variable()
                obj = tt + (w_all[o_set] @ cp.pos(r.A[o_set, :] @ x - tt)) / p
            else:
                obj = w_all[o_set] @ cp.pos(r.A[o_set, :] @ x - thr_v)
            cons = [m_ @ x <= v_ for m_, v_ in mean_cons]
            if lo_set.size:
                cons.append(ptv.A[lo_set, :] @ x >= ptv_d95_min / fx)
            if hi_set.size:
                cons.append(ptv.A[hi_set, :] @ x <= ptv_dmax_max / fx)
            if mx_set.size:
                cons.append(A_rows[mx_set, :] @ x <= all_rhs[mx_set])
            prob = cp.Problem(cp.Minimize(obj + 1e-6 * cp.sum(x)), cons)
            ts = time.time()
            try:
                used, fb = self._solve_round(prob, viol_tol=PROBE_GPU_VIOL_GY / fx, max_iter=PROBE_GPU_MAX_ITER)
                status = str(prob.status)
                if fb:
                    gi = self._gpu_solver.last_info
                    used += f" (GPU rejected: {gi.get('status')}, iter {gi.get('iter')}, viol {gi.get('max_viol')})"
            except cp.error.SolverError as exc:
                used, status = '?', f'solver_error: {exc}'
            if status not in ('optimal', 'optimal_inaccurate') or x.value is None:
                say(f'  probe {roi} {metric} round {rounds}: {status} ({used}, {time.time() - ts:.1f}s)')
                xval = None
                break
            xval = np.maximum(np.asarray(x.value, dtype=float), 0.0)
            # Constraint generation: add every voxel that crossed its bound or would change the objective.
            dp = ptv.A @ xval
            add_lo = np.setdiff1d(hot[dp[hot] < ptv_d95_min / fx - tol], lo_set)
            add_hi = np.setdiff1d(np.flatnonzero(dp > ptv_dmax_max / fx + tol), hi_set)
            add_mx = np.setdiff1d(np.flatnonzero(A_rows @ xval > all_rhs + tol), mx_set)
            add_o, over_o = np.zeros(0, dtype=np.int64), 0.0
            if kind != 'mean':
                d_o = r.A @ xval
                if kind == 'max':
                    cut = float(d_o[o_set].max()) + 1e-6 if o_set.size else -np.inf
                elif kind == 'cvar':
                    cut = float(tt.value) if tt is not None and tt.value is not None else -np.inf
                else:
                    cut = thr_v
                add_o = np.setdiff1d(np.flatnonzero(d_o > cut + (tol if kind == 'max' else 0.0)), o_set)
                over_o = float((d_o[add_o] - cut).max()) * fx if add_o.size else 0.0
            say(f'  probe {roi} {metric} round {rounds}: rows obj {o_set.size} ptv>= {lo_set.size} ptv<= '
                f'{hi_set.size} max {mx_set.size} | {used} {time.time() - ts:.1f}s | add {add_o.size}/'
                f'{add_lo.size}/{add_hi.size}/{add_mx.size}')
            worst = max([0.0, over_o if kind != 'mean' else 0.0]
                        + ([float(ptv_d95_min / fx - dp[add_lo].min())] if add_lo.size else [])
                        + ([float(dp[add_hi].max() - ptv_dmax_max / fx)] if add_hi.size else [])
                        + ([float((A_rows[add_mx, :] @ xval - all_rhs[add_mx]).max())] if add_mx.size else [])) * fx
            n_add = add_o.size + add_lo.size + add_hi.size + add_mx.size
            if n_add == 0 or (n_add <= PROBE_SETTLE_ROWS and worst <= PROBE_SETTLE_GY):
                break
            o_set, lo_set = np.union1d(o_set, add_o), np.union1d(lo_set, add_lo)
            hi_set, mx_set = np.union1d(hi_set, add_hi), np.union1d(mx_set, add_mx)
        else:
            status = 'not_converged'                 # still adding rows after max_rounds: verdict 'unknown'
        out = {'roi': roi, 'metric': metric, 'status': status, 'rounds': rounds,
               'seconds': round(time.time() - t0, 1)}
        if xval is None or status == 'not_converged':
            return out
        fl = self.E @ xval if self.E is not None else xval
        m = self.key_metrics(from_fluence(self.case, fl), rois=(roi, 'PTV'))
        out.update({'value': m[roi].get(metric), 'ptv_d95': m['PTV']['D95'], 'ptv_dmax': m['PTV']['Dmax']})
        return out

    def hot_subset(self, roi: str, dose_gy: float) -> np.ndarray:
        """Positions (within the ROI's voxel list) whose dose in the current plan is >= dose_gy - HOT_MARGIN_GY."""
        r = self.roi(roi)
        cur = self.current
        if cur is None or cur.dose_total_gy is None:
            return np.arange(r.idx.size)
        d = np.asarray(cur.dose_total_gy, dtype=float)[r.idx]
        return np.flatnonzero(d >= float(dose_gy) - HOT_MARGIN_GY)

    def protocol_limits(self) -> dict[tuple[str, str], float]:
        """Protocol (clinical-criteria) limits by (roi, metric): max_dose -> Dmax and D2, mean_dose -> Dmean,
        dose_volume_V at X Gy -> VX (percent). Used to size side-effect guards by the room left to the limit."""
        out: dict[tuple[str, str], float] = {}
        try:
            crit = self.case.criteria.get_criteria()
        except Exception:  # noqa: BLE001 - test doubles without criteria
            return out
        for c in crit or []:
            roi = c.get('parameters', {}).get('structure_name')
            lk = next((k for k in c.get('constraints', {}) if k.startswith('limit')), None)
            if not roi or lk is None:
                continue
            try:
                lim = float(self._num(c['constraints'][lk]))
            except (TypeError, ValueError):
                continue
            if lk == 'limit_dose_perc':
                lim = lim * self.rx_gy / 100
            if c['type'] == 'max_dose':
                keys = [(roi, 'Dmax'), (roi, 'D2')]
            elif c['type'] == 'mean_dose':
                keys = [(roi, 'Dmean')]
            elif c['type'] == 'dose_volume_V' and 'dose_gy' in c['parameters'] and 'perc' in lk:
                keys = [(roi, f"V{float(self._num(c['parameters']['dose_gy'])):g}")]
            else:
                continue
            for k in keys:
                out[k] = min(out.get(k, float('inf')), lim)
        return out

    def _initial_active_rows(self) -> np.ndarray:
        """Rows likely to be active: voxels near the PTV and every non-RIND max-constrained ROI.

        Far-away RIND voxels (most of the ~430k rows) rarely approach their limits;
        constraint generation adds any that do.
        """
        rows = set()
        for c in self.constraints.values():
            if (c.kind == 'max' and self.has_voxels(c.roi) and c.roi not in ('RIND_2', 'RIND_3', 'RIND_4')
                    and self.roi(c.roi).idx.size <= 30000):
                rows.update(self.roi(c.roi).idx.tolist())
        return np.fromiter(rows, dtype=np.int64)

    # ------------------------------------------------------------- solve cache
    _CODE_HASH = None

    @classmethod
    def _code_hash(cls) -> str:
        """Any edit to this file (problem formulation, tolerances) invalidates every cached solve."""
        if cls._CODE_HASH is None:
            import hashlib
            cls._CODE_HASH = hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:10]
        return cls._CODE_HASH

    def configure_gpu(self, opts: dict | None):
        """opts = config optimization.gpu: {mode: off|auto|on, min_beamlets: int, tol: float}."""
        opts = opts or {}
        mode = opts.get('mode', 'off')
        mode = {False: 'off', True: 'on'}.get(mode, mode) if isinstance(mode, bool) else str(mode).lower()
        if mode not in ('off', 'auto', 'on'):
            raise ValueError(f'optimization.gpu.mode must be off, auto or on, not {mode!r}')
        self.gpu = mode
        self.gpu_min_beamlets = int(opts.get('min_beamlets', 1500))
        self._gpu_solver = None
        if self.use_gpu():
            from .gpu_qp import PdhgGpu
            self._gpu_solver = PdhgGpu(tol=float(opts.get('tol', 1e-4)), viol_tol=float(opts.get('viol_tol', 1e-3)))

    def use_gpu(self) -> bool:
        if self.gpu == 'off' or (self.gpu == 'auto' and self.n_beamlets < self.gpu_min_beamlets):
            return False
        from .gpu_qp import gpu_available
        return gpu_available()

    def _solve_round(self, prob, viol_tol: float | None = None, max_iter: int | None = None):
        """One QP solve: the GPU solver when enabled, else / on failure the configured CPU solver.
        viol_tol: per-fraction Gy a GPU answer may exceed a row by (default: the solver's own, 1e-3)."""
        import cvxpy as cp
        if self._gpu_solver is not None:
            keep_tol, keep_it = self._gpu_solver.viol_tol, self._gpu_solver.max_iter
            if viol_tol is not None:
                self._gpu_solver.viol_tol = viol_tol
            if max_iter is not None:
                self._gpu_solver.max_iter = max_iter
            try:
                prob.solve(solver=self._gpu_solver)
                if str(prob.status) == 'optimal':
                    return 'PDHG_GPU', False
            except cp.error.SolverError:
                pass
            finally:
                self._gpu_solver.viol_tol, self._gpu_solver.max_iter = keep_tol, keep_it
            if self.log:
                print(f'  GPU solve rejected ({self._gpu_solver.last_info.get("status")}, '
                      f'viol {self._gpu_solver.last_info.get("max_viol")}): re-solving with {self.solver}', flush=True)
            prob.solve(solver=self.solver, verbose=self.verbose, **self.solver_opts)
            return self.solver, True
        prob.solve(solver=self.solver, verbose=self.verbose, **self.solver_opts)
        return self.solver, False

    def _cache_key(self, max_rounds: int, tol_gy: float) -> str:
        import hashlib
        import json
        parts = [self._code_hash(), self.A.shape, int(self.A.nnz), self.solver, self.use_gpu(),
                 json.dumps(self.solver_opts, sort_keys=True), max_rounds, tol_gy, self.knobs_fingerprint()]
        # Hot-subset terms act on the voxels that are hot in the plan the solve starts from.
        if any(t.kind == 'linear_overdose' and t.roi in HOT_SUBSET_ROIS and t.weight > 0 for t in self.terms.values()):
            cur = self.current
            parts.append(hashlib.sha1(np.ascontiguousarray(cur.fluence).tobytes()).hexdigest()
                         if cur is not None and cur.fluence is not None else 'none')
        return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:20]

    def _cache_load(self, key: str):
        import json
        from pathlib import Path
        f = Path(self.solve_cache) / f'{key}.npz'
        if not f.exists():
            return None
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z['meta']))
            return np.asarray(z['fluence'], dtype=float), meta
        except Exception:  # noqa: BLE001 - a broken cache file is just a miss
            return None

    def _cache_save(self, key: str, v: 'PlanVersion'):
        import json
        from pathlib import Path
        d = Path(self.solve_cache)
        d.mkdir(parents=True, exist_ok=True)
        meta = {'status': v.status, 'objective_value': v.objective_value, 'rounds': v.rounds,
                'max_violation_gy': v.max_violation_gy, 'duals': v.duals, 'solve_seconds': v.solve_seconds}
        tmp = d / f'{key}.tmp.npz'
        np.savez_compressed(tmp, fluence=v.fluence, meta=json.dumps(meta))
        tmp.replace(d / f'{key}.npz')

    def solve(self, label: str | None = None, max_rounds: int = 8, tol_gy: float = 0.05) -> PlanVersion:
        """Solve with the current knobs via constraint generation, store a new version.

        The QP is solved with a subset of max-dose rows; after each solve all rows are
        checked and violated ones (plus a margin band) are added. Terminates when the
        solution satisfies every max-dose constraint, i.e. it is optimal for the full
        problem.  On failure, knobs roll back to the last successful version.
        """
        import cvxpy as cp
        t0 = time.time()
        key = self._cache_key(max_rounds, tol_gy) if self.solve_cache else None
        hit = self._cache_load(key) if key else None
        if hit is not None:
            fluence, meta = hit
            vid = len(self.history)
            v = PlanVersion(vid, self.current_id, label or f'v{vid}', list(self.pending_changes) or ['(no changes)'],
                            deepcopy(self.terms), deepcopy(self.constraints), meta['status'], meta['objective_value'],
                            0.0, round(time.time() - t0, 2))
            v.rounds, v.max_violation_gy, v.duals, v.cached = meta['rounds'], meta['max_violation_gy'], meta['duals'], True
            v.fluence = fluence
            v.dose_total_gy = from_fluence(self.case, v.fluence)
            v.metrics = self.key_metrics(v.dose_total_gy)
            v.criteria = evaluate_criteria(self.case, v.dose_total_gy)
            self.current_id = vid
            self.pending_changes = []
            self.history.append(v)
            self.seed_active_set(v.fluence)
            return v
        if self._active is None:
            self._active = self._initial_active_rows()
        all_rows, all_rhs = self._max_rhs()
        status, value, xval, rounds, build_s, solve_s = 'error', None, None, 0, 0.0, 0.0
        max_violation_gy, duals = None, None
        solver_used, fallbacks = '', 0
        for rounds in range(1, max_rounds + 1):
            tb = time.time()
            prob, x = self.build_problem(active_rows=self._active)
            ts = time.time()
            build_s += ts - tb
            try:
                solver_used, fell_back = self._solve_round(prob)
                fallbacks += fell_back
                status, value = str(prob.status), (float(prob.value) if prob.value is not None else None)
            except cp.error.SolverError as exc:
                status = f'solver_error: {exc}'
            solve_s += time.time() - ts
            if status not in ('optimal', 'optimal_inaccurate') or x.value is None:
                break
            xval = np.maximum(np.asarray(x.value, dtype=float), 0.0)
            try:
                duals = self._duals(prob, self._last_tags)
            except Exception:  # noqa: BLE001 - explanations must never break a solve
                duals = None
            d = self.A[all_rows, :] @ xval
            excess = d - all_rhs
            max_violation_gy = float(max(0.0, excess.max())) * self.num_fx
            viol = all_rows[excess > tol_gy / self.num_fx]
            if self.log:
                print(f'  round {rounds}: active={self._active.size} violated={viol.size} '
                      f'solve={time.time() - ts:.1f}s', flush=True)
            if viol.size == 0:
                break
            near = all_rows[d > 0.95 * all_rhs]   # add a margin band to avoid ping-pong
            self._active = np.union1d(self._active, np.union1d(viol, near))
        else:
            status = 'not_converged_constraint_generation'
        t2 = time.time()
        vid = len(self.history)
        changes = list(self.pending_changes) or ['(no changes)']
        v = PlanVersion(vid, self.current_id, label or (f'v{vid}' if vid else 'baseline (protocol defaults)'),
                        changes, deepcopy(self.terms), deepcopy(self.constraints), status, value,
                        round(solve_s, 2), round(t2 - t0 - solve_s, 2))
        v.rounds = rounds
        v.max_violation_gy = max_violation_gy
        v.duals = duals
        v.solver_used, v.gpu_fallbacks = solver_used, fallbacks
        if status in ('optimal', 'optimal_inaccurate') and xval is not None:
            v.fluence = self.E @ xval if self.E is not None else xval   # native-beamlet fluence
            v.dose_total_gy = from_fluence(self.case, v.fluence)
            v.metrics = self.key_metrics(v.dose_total_gy)
            v.criteria = evaluate_criteria(self.case, v.dose_total_gy)
            self.current_id = vid
            self.pending_changes = []
            if key:
                try:
                    self._cache_save(key, v)
                except OSError:
                    pass
        else:
            # roll back knobs to the last successful version
            if self.current_id is not None:
                last = self.get_version(self.current_id)
                self.terms, self.constraints = deepcopy(last.terms), deepcopy(last.constraints)
            self.pending_changes = []
        self.history.append(v)
        return v

    def add_version_from_fluence(self, fluence: np.ndarray, label: str, status: str = 'optimal (cached)') -> PlanVersion:
        """Register a precomputed solution for the CURRENT knobs (e.g. cached baseline)."""
        vid = len(self.history)
        v = PlanVersion(vid, self.current_id, label, list(self.pending_changes) or ['(cached)'],
                        deepcopy(self.terms), deepcopy(self.constraints), status)
        v.fluence = np.asarray(fluence, dtype=float)
        v.dose_total_gy = from_fluence(self.case, v.fluence)
        v.metrics = self.key_metrics(v.dose_total_gy)
        v.criteria = evaluate_criteria(self.case, v.dose_total_gy)
        self.history.append(v)
        self.current_id = vid
        self.pending_changes = []
        # seed the constraint-generation active set from this solution
        rows, rhs = self._max_rhs()
        d = self.A_full[rows, :] @ v.fluence
        seed = rows[d > 0.9 * rhs]
        base = self._initial_active_rows() if self._active is None else self._active
        self._active = np.union1d(base, seed)
        return v

    def seed_active_set(self, fluence: np.ndarray | None):
        """Seed constraint generation with the max-dose rows near their limit for `fluence` (after a restore)."""
        base = self._initial_active_rows() if self._active is None else self._active
        if fluence is None:
            self._active = base
            return
        rows, rhs = self._max_rhs()
        d = self.A_full[rows, :] @ np.asarray(fluence, dtype=float)
        self._active = np.union1d(base, rows[d > 0.9 * rhs])

    def knobs_fingerprint(self) -> str:
        import hashlib
        import json
        blob = json.dumps([self.merge_x] + self.knobs_table(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    # ------------------------------------------------------------- reading
    def get_version(self, version_id: int | None = None) -> PlanVersion:
        if version_id is None:
            version_id = self.current_id
        if version_id is None or not 0 <= int(version_id) < len(self.history):
            raise ValidationError(f'No plan version {version_id!r}')
        return self.history[int(version_id)]

    @property
    def current(self) -> PlanVersion | None:
        return None if self.current_id is None else self.history[self.current_id]

    def key_metrics(self, dose_total_gy: np.ndarray, rois=REPORT_ROIS) -> dict:
        """Metrics an agent/physicist reasons about. Partial ROIs are labeled, not hidden."""
        out = {}
        for name in rois:
            if name not in self.structures or not self.has_voxels(name):
                out[name] = {'status': 'unavailable',
                             'reason': 'no optimization voxels in calculation box' if name in self.structures else 'ROI absent'}
                continue
            roi = get_roi_dose(self.case, dose_total_gy, name)
            m = {'scope': 'whole_roi' if roi.calculation_coverage >= 0.999 else
                 f'represented_subvolume({roi.calculation_coverage:.0%} of organ)',
                 'Dmean': weighted_mean_gy(roi), 'Dmax': float(roi.dose_gy.max()),
                 'D2': dose_at_volume_gy(roi, 2), 'D95': dose_at_volume_gy(roi, 95)}
            if name in TARGET_ROIS:
                m.update({'D98': dose_at_volume_gy(roi, 98),
                          f'V{self.rx_gy:g}': volume_at_dose_percent(roi, self.rx_gy),
                          'V95%Rx': volume_at_dose_percent(roi, 0.95 * self.rx_gy)})
            else:
                m.update({'V20': volume_at_dose_percent(roi, 20), 'V30': volume_at_dose_percent(roi, 30),
                          'V60': volume_at_dose_percent(roi, 60)})
            out[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        return out

    def criteria_counts(self, version_id: int | None = None) -> dict:
        df = self.get_version(version_id).criteria
        return {} if df is None else df['status'].value_counts().to_dict()

    def knobs_table(self) -> list[dict]:
        rows = [{'id': t.id, 'type': 'objective', 'kind': t.kind, 'roi': t.roi, 'weight': t.weight,
                 'dose_gy': t.dose_gy, 'source': t.source, 'note': t.note} for t in self.terms.values()]
        rows += [{'id': c.id, 'type': 'constraint', 'kind': c.kind, 'roi': c.roi, 'limit_gy': c.limit_gy,
                  'source': c.source, 'scope': c.scope} for c in self.constraints.values()]
        return rows

    def compare(self, a: int, b: int | None = None) -> dict:
        """Metric deltas b - a (b defaults to current)."""
        va, vb = self.get_version(a), self.get_version(b)
        if not (va.ok and vb.ok):
            raise ValidationError('Both versions must be successful solves')
        out = {}
        for roi, mb in vb.metrics.items():
            ma = va.metrics.get(roi, {})
            if mb.get('status') == 'unavailable':
                continue
            out[roi] = {k: round(mb[k] - ma[k], 3) for k in mb if isinstance(mb[k], float) and k in ma}
        ca, cb = self.criteria_counts(a), self.criteria_counts(vb.id)
        out['_criteria'] = {s: cb.get(s, 0) - ca.get(s, 0) for s in set(ca) | set(cb)}
        return out
