"""Deterministic goal resolution and checking. The LLM never decides pass/fail."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .schemas import GoalItem, PhysicistGoal

HIGHER_IS_BETTER = {'D95', 'D98', 'V60', 'V95%Rx'}   # only meaningful for targets
TARGETS = {'PTV', 'GTV', 'CTV'}

# Side-effect guards: organs the physicist did NOT mention may not get worse by more than a tolerance.
# (metric per OAR chosen by organ type: serial organs -> near-max, parallel organs -> mean / V20)
OAR_GUARDS = {'ESOPHAGUS': ('Dmean', 'D2'), 'CORD': ('Dmax',), 'HEART': ('Dmean',),
              'LUNGS_NOT_GTV': ('Dmean', 'V20')}
DEFAULT_GUARD_TOL = {'Dmean': 1.0, 'Dmax': 2.0, 'D2': 2.0, 'V20': 2.0}   # Gy, Gy, Gy, percentage points
# Headroom rule (user decision 2026-11-09): an unmentioned organ may also use up to this fraction of the room left
# to its protocol limit, so a cord at 7.8 Gy (limit 50) is not blocked at 9.8 Gy. Never beyond the limit itself.
GUARD_HEADROOM = 0.5
AT_LIMIT_SLACK = {'Gy': 0.2, '%': 0.3}    # an organ already at / over its limit may only move within noise
# Ask cap (user decision 2026-11-10): however much room is left, an unmentioned organ may rise by at most this much
# without asking the physicist (round 3 accepted e.g. esophagus D2 +19.7 Gy silently). Gy; V metrics in %-points.
GUARD_MAX_RISE = {'Dmax': 5.0, 'D2': 5.0, 'Dmean': 3.0, 'V': 5.0}


def max_rise_for(metric: str, max_rise: dict | None) -> float | None:
    caps = GUARD_MAX_RISE if max_rise is None else max_rise
    return caps.get(metric, caps.get('V') if metric.startswith('V') else None)

# Trade-off table rows (roi, metric) shown after every change; PTV rows are higher-is-better for D95/D98.
TRADEOFF_ROWS = [('PTV', 'D95'), ('PTV', 'Dmax'), ('ESOPHAGUS', 'Dmean'), ('ESOPHAGUS', 'D2'), ('CORD', 'Dmax'),
                 ('HEART', 'Dmean'), ('LUNGS_NOT_GTV', 'Dmean'), ('LUNGS_NOT_GTV', 'V20'), ('LUNG_L', 'Dmean'),
                 ('LUNG_R', 'Dmean')]
NOISE = {'Gy': 0.2, '%': 0.3}


def _higher_is_better(roi: str, metric: str) -> bool:
    return roi in TARGETS and metric in HIGHER_IS_BETTER


def _default_step(metric: str, ref: float) -> float:
    if metric.startswith('V'):
        return 1.0                        # percentage points
    return max(0.5, 0.03 * abs(ref))      # Gy


@dataclass
class Check:
    roi: str
    metric: str
    op: str             # '<=' or '>='
    threshold: float
    ref: float          # value in the plan the request was made against
    role: str           # objective | protection
    text: str
    actual: float | None = None
    passed: bool | None = None
    limit: float | None = None      # protocol limit for this (roi, metric), when the protocol has one

    def evaluate(self, metrics: dict) -> 'Check':
        m = metrics.get(self.roi, {})
        v = m.get(self.metric)
        self.actual = None if v is None else float(v)
        if self.actual is None:
            self.passed = False
        else:
            self.passed = self.actual <= self.threshold + 1e-6 if self.op == '<=' else self.actual >= self.threshold - 1e-6
        return self

    def progress(self) -> float:
        """Fraction of the requested change achieved, clipped to [0, 1]."""
        if self.actual is None:
            return 0.0
        want = self.threshold - self.ref
        if abs(want) < 1e-9:
            return 1.0 if self.passed else 0.0
        return max(0.0, min(1.0, (self.actual - self.ref) / want))

    def as_dict(self):
        return asdict(self)


@dataclass
class ResolvedGoal:
    checks: list[Check]
    issues: list[str]         # ROIs/metrics that cannot be evaluated -> reported honestly

    @property
    def objectives(self):
        return [c for c in self.checks if c.role == 'objective']

    @property
    def protections(self):
        return [c for c in self.checks if c.role == 'protection']


def resolve_goal(goal: PhysicistGoal, metrics: dict, ptv_tol_gy: float = 0.5,
                 add_default_protections: bool = True, guard_tol: dict | None = None,
                 limits: dict | None = None, headroom: float = GUARD_HEADROOM,
                 max_rise: dict | None = None) -> ResolvedGoal:
    """guard_tol: per-metric tolerances for unmentioned OARs (None = DEFAULT_GUARD_TOL, {} = no guards).
    limits: protocol limits {(roi, metric): value}; with headroom > 0 a guard allows max(tol, headroom * room)."""
    checks, issues = [], []

    def ref_of(item: GoalItem):
        roi = item.roi.strip().upper()
        m = metrics.get(roi)
        if m is None:
            issues.append(f'{roi} is not a known structure for this patient.')
            return roi, None
        if m.get('status') == 'unavailable':
            issues.append(f'{roi} has no optimization voxels in the dose calculation box for this patient, '
                          f'so its dose cannot be evaluated or optimized here.')
            return roi, None
        if item.metric not in m:
            issues.append(f'{item.metric} is not computed for {roi}; available: '
                          f'{", ".join(k for k in m if k != "scope")}.')
            return roi, None
        return roi, float(m[item.metric])

    for item in goal.objectives:
        roi, ref = ref_of(item)
        if ref is None:
            continue
        unit = '%' if item.metric.startswith('V') else 'Gy'
        if item.direction == 'decrease':
            amt = item.amount if item.amount else _default_step(item.metric, ref)
            checks.append(Check(roi, item.metric, '<=', ref - abs(amt), ref, 'objective',
                                f'{roi} {item.metric} decrease by {abs(amt):g} {unit} (from {ref:.2f})'))
        elif item.direction == 'increase':
            amt = item.amount if item.amount else _default_step(item.metric, ref)
            checks.append(Check(roi, item.metric, '>=', ref + abs(amt), ref, 'objective',
                                f'{roi} {item.metric} increase by {abs(amt):g} {unit} (from {ref:.2f})'))
        elif item.direction in ('at_most', 'at_least'):
            if item.target is None:
                issues.append(f'No target value given for {roi} {item.metric} {item.direction}.')
                continue
            op = '<=' if item.direction == 'at_most' else '>='
            checks.append(Check(roi, item.metric, op, float(item.target), ref, 'objective',
                                f'{roi} {item.metric} {op} {item.target:g} {unit} (now {ref:.2f})'))
        else:  # keep given as objective -> treat as protection
            goal.protections.append(item)

    for item in goal.protections:
        roi, ref = ref_of(item)
        if ref is None:
            continue
        unit = '%' if item.metric.startswith('V') else 'Gy'
        if item.direction in ('at_most', 'at_least') and item.target is not None:
            op = '<=' if item.direction == 'at_most' else '>='
            checks.append(Check(roi, item.metric, op, float(item.target), ref, 'protection',
                                f'keep {roi} {item.metric} {op} {item.target:g} {unit}'))
            continue
        # A 0 / missing tolerance on 'keep' means "not specified" (small models emit 0.0); use the default.
        tol = abs(item.amount) if item.amount else (ptv_tol_gy if unit == 'Gy' else 1.0)
        if _higher_is_better(roi, item.metric):
            checks.append(Check(roi, item.metric, '>=', ref - tol, ref, 'protection',
                                f'keep {roi} {item.metric} >= {ref - tol:.2f} {unit} (now {ref:.2f}, tol {tol:g})'))
        else:
            checks.append(Check(roi, item.metric, '<=', ref + tol, ref, 'protection',
                                f'keep {roi} {item.metric} <= {ref + tol:.2f} {unit} (now {ref:.2f}, tol {tol:g})'))

    if add_default_protections and 'PTV' in metrics and metrics['PTV'].get('status') != 'unavailable':
        have = {(c.roi, c.metric) for c in checks if c.role == 'protection'}
        p = metrics['PTV']
        if ('PTV', 'D95') not in have:
            checks.append(Check('PTV', 'D95', '>=', p['D95'] - ptv_tol_gy, p['D95'], 'protection',
                                f'default: keep PTV D95 >= {p["D95"] - ptv_tol_gy:.2f} Gy'))
        if ('PTV', 'Dmax') not in have:
            checks.append(Check('PTV', 'Dmax', '<=', p['Dmax'] + 1.0, p['Dmax'], 'protection',
                                f'default: keep PTV Dmax <= {p["Dmax"] + 1.0:.2f} Gy'))
    if add_default_protections:
        checks += guard_checks(metrics, {c.roi for c in checks},
                               DEFAULT_GUARD_TOL if guard_tol is None else guard_tol, limits, headroom, max_rise)
    return ResolvedGoal(checks, issues)


def guard_allowance(ref: float, tol: float, limit: float | None, unit: str,
                    headroom: float = GUARD_HEADROOM, cap: float | None = None) -> float:
    """Threshold for an unmentioned organ: ref + max(tol, min(headroom * room to the protocol limit, cap)), never past
    the limit. cap = the largest rise accepted without asking (None = no cap)."""
    if limit is None or headroom <= 0:
        return ref + tol
    if ref >= limit - 1e-9:
        return ref + min(tol, AT_LIMIT_SLACK[unit])
    room = headroom * (limit - ref)
    if cap is not None:
        room = min(room, cap)
    return min(ref + max(tol, room), limit)


def guard_checks(metrics: dict, mentioned: set, tol: dict, limits: dict | None = None,
                 headroom: float = GUARD_HEADROOM, max_rise: dict | None = None) -> list[Check]:
    out = []
    for roi, mets in OAR_GUARDS.items():
        m = metrics.get(roi)
        if roi in mentioned or not m or m.get('status') == 'unavailable':
            continue
        for metric in mets:
            if metric not in tol or m.get(metric) is None:
                continue
            ref, t = float(m[metric]), float(tol[metric])
            unit = '%' if metric.startswith('V') else 'Gy'
            lim = (limits or {}).get((roi, metric))
            thr = guard_allowance(ref, t, lim, unit, headroom, max_rise_for(metric, max_rise))
            why = (f'now {ref:.2f}, protocol limit {lim:g}' if lim is not None and headroom > 0
                   else f'now {ref:.2f}, +{t:g}')
            out.append(Check(roi, metric, '<=', thr, ref, 'guard',
                             f'side-effect guard: {roi} {metric} <= {thr:.2f} {unit} ({why})', limit=lim))
    return out


def tradeoffs(before: dict, after: dict) -> list[dict]:
    """Deterministic before/after table across organs, with better/worse flags (noise-tolerant)."""
    rows = []
    for roi, metric in TRADEOFF_ROWS:
        a, b = before.get(roi, {}), after.get(roi, {})
        if a.get('status') == 'unavailable' or b.get('status') == 'unavailable':
            continue
        if a.get(metric) is None or b.get(metric) is None:
            continue
        unit = '%' if metric.startswith('V') else 'Gy'
        d = float(b[metric]) - float(a[metric])
        better_sign = 1 if _higher_is_better(roi, metric) else -1
        if abs(d) < NOISE[unit]:
            flag = 'same'
        else:
            flag = 'better' if d * better_sign > 0 else 'worse'
        rows.append({'roi': roi, 'metric': metric, 'unit': unit, 'before': float(a[metric]),
                     'after': float(b[metric]), 'delta': round(d, 3), 'flag': flag,
                     'partial': str(b.get('scope', '')).startswith('represented')})
    return rows


def evaluate(resolved: ResolvedGoal, metrics: dict) -> dict:
    checks = [Check(**{k: v for k, v in c.as_dict().items() if k not in ('actual', 'passed')}).evaluate(metrics)
              for c in resolved.checks]
    obj = [c for c in checks if c.role == 'objective']
    prot = [c for c in checks if c.role == 'protection']
    guard = [c for c in checks if c.role == 'guard']
    targets_ok = all(c.passed for c in prot)
    broken = [c for c in guard if not c.passed]
    return {
        'checks': [c.as_dict() for c in checks],
        'objectives_met': bool(obj) and all(c.passed for c in obj),
        'protections_ok': targets_ok and not broken,
        'guards_ok': not broken,
        'targets_ok': targets_ok,
        # Only side-effect guards broke, and no organ went past its protocol limit: the physicist can be asked.
        'askable': targets_ok and bool(broken) and all(c.actual is not None and (c.limit is None or
                                                                                 c.actual <= c.limit + 1e-6)
                                                       for c in broken),
        'violations': [f"{c.roi} {c.metric}" for c in prot + guard if not c.passed],
        'progress': (sum(c.progress() for c in obj) / len(obj)) if obj else 0.0,
    }


def score_tuple(ev: dict) -> tuple:
    """Ranking for choosing the best version in a turn."""
    return (ev['protections_ok'], ev['objectives_met'], round(ev['progress'], 4))
