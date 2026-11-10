"""M1-C: auditable PortPy clinical-criteria screening on an optimization dose grid.

Educational/research use only. `PASS` is NOT clinical plan approval.
Dose input is the TOTAL course dose in Gy, indexed by PortPy optimization voxels.

Supported PortPy criteria:
    max_dose, mean_dose, dose_volume_D, dose_volume_V.
Supports absolute/relative dose and percent-volume values, lower/upper bounds,
clinical Goal/Limit distinction and coldest-volume convention.

Absolute volume (cc) DVH constraints are intentionally UNDETERMINED until
full-structure geometry/volume is independently validated.
"""
from __future__ import annotations

import ast
import operator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def resolve_number(value: Any, prescription_gy: float) -> float:
    """Resolve numbers or simple expressions using 'prescription_gy' safely.

    Example: '1.1*prescription_gy' -> 66.0 when prescription is 60 Gy.
    Never use eval() on unknown protocol text.
    """
    if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool):
        answer = float(value)
    elif isinstance(value, str):
        tree = ast.parse(value.strip(), mode="eval")

        def walk(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id == 'prescription_gy':
                return float(prescription_gy)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                v = walk(node.operand)
                return -v if isinstance(node.op, ast.USub) else v
            if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
                return _ALLOWED_OPERATORS[type(node.op)](walk(node.left), walk(node.right))
            raise ValueError(f"Unsafe or unsupported numeric expression: {value}")

        answer = float(walk(tree.body))
    else:
        raise ValueError(f"Unsupported numeric value: {value!r}")
    if not np.isfinite(answer):
        raise ValueError("Non-finite numeric value")
    return answer


def dose_at_volume(doses_gy: np.ndarray, volumes_cc: np.ndarray, volume_perc: float) -> float:
    """Discrete cumulative DVH: dose received by at least `volume_perc` of volume.

    Percent 0 is defined as max dose. Percent 100 is min dose.
    """
    if not 0 <= volume_perc <= 100:
        raise ValueError("volume_perc must be in [0, 100]")
    if volume_perc == 0:
        return float(np.max(doses_gy))
    order = np.argsort(-doses_gy)
    d = doses_gy[order]
    v = volumes_cc[order]
    cum = np.cumsum(v) / v.sum() * 100
    idx = min(int(np.searchsorted(cum, volume_perc, side='left')), len(d) - 1)
    return float(d[idx])


def volume_at_dose_perc(doses_gy: np.ndarray, volumes_cc: np.ndarray, dose_gy: float) -> float:
    """Percent of represented ROI volume receiving >= dose_gy."""
    return float(100 * volumes_cc[doses_gy >= dose_gy].sum() / volumes_cc.sum())


@dataclass
class ROIDose:
    name: str
    doses_gy: np.ndarray
    volumes_cc: np.ndarray
    reported_coverage: float
    represented_volume_cc: float
    metadata_volume_cc: float | None


def read_roi(case: Any, total_dose_gy: np.ndarray, roi_name: str) -> ROIDose:
    """Extract a PortPy ROI from the optimization dose vector, preserving weights."""
    inf = case.influence_matrix
    names = case.structures.get_structures()
    if roi_name not in names:
        raise ValueError(f"ROI {roi_name!r} is not in patient structures")
    idx = np.asarray(inf.get_opt_voxels_idx(roi_name), dtype=np.int64).ravel()
    if len(idx) == 0:
        raise ValueError("No optimization voxels for ROI")
    v = np.asarray(inf.get_opt_voxels_volume_cc(roi_name), dtype=np.float64).ravel()
    if len(idx) != len(v) or np.any(v <= 0) or not np.all(np.isfinite(v)):
        raise ValueError("Invalid optimization voxel weights")
    if np.min(idx) < 0 or np.max(idx) >= len(total_dose_gy):
        raise ValueError("ROI dose voxel indices exceed dose vector")
    d = np.asarray(total_dose_gy[idx], dtype=np.float64)
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Invalid dose values")
    c = inf.get_fraction_of_vol_in_calc_box(roi_name)
    if c is None or not 0 <= float(c) <= 1:
        raise ValueError("Missing or invalid fraction_of_vol_in_calc_box")
    try:
        metadata_volume = float(case.structures.get_volume_cc(roi_name))
    except (AttributeError, TypeError, KeyError, ValueError):
        metadata_volume = None
    return ROIDose(roi_name, d, v, float(c), float(v.sum()), metadata_volume)


def _thresholds(criteria: dict, kind: str, unit: str, prescription_gy: float) -> dict:
    """Return {'goal': number, 'limit': number}; forbid ambiguous unit keys."""
    out = {}
    cs = criteria.get('constraints', {})
    for field in ('goal', 'limit'):
        options = [k for k in cs if k.startswith(field + '_') and (
            k.endswith('_dose_gy') or k.endswith('_dose_perc') or
            k.endswith('_volume_perc') or k.endswith('_volume_cc'))]
        if len(options) > 1:
            raise ValueError(f"Multiple {field} thresholds: {options}")
        if not options:
            continue
        key = options[0]
        value = resolve_number(cs[key], prescription_gy)
        if key.endswith('_dose_perc'):
            if unit != 'Gy':
                raise ValueError('Dose threshold not valid for volume metric')
            value *= prescription_gy / 100
        elif key.endswith('_dose_gy'):
            if unit != 'Gy':
                raise ValueError('Dose threshold not valid for volume metric')
        elif key.endswith('_volume_perc'):
            if unit != '%':
                raise ValueError('Volume % threshold not valid for dose metric')
        elif key.endswith('_volume_cc'):
            raise ValueError('Volume-in-cc constraints require independent full-ROI geometry validation')
        if value < 0:
            raise ValueError(f'Negative {field} threshold')
        if key.endswith('_volume_perc') and value > 100:
            raise ValueError(f'Volume percentage {value} exceeds 100')
        out[field] = value
    if not out:
        raise ValueError('Criterion has no supported goal/limit')
    return out


def _single_param(parameters: dict, variants: tuple[str, ...]):
    matches = [k for k in variants if k in parameters]
    if len(matches) != 1:
        raise ValueError(f"Expected one of {variants}, got {matches}")
    return matches[0], parameters[matches[0]]


def compute_criterion_value(roi: ROIDose, criterion: dict, prescription_gy: float):
    """Return (metric_label, actual_value, unit, thresholds, bound_direction)."""
    kind = criterion.get('type')
    params = criterion.get('parameters', {})
    cons = criterion.get('constraints', {})
    bound = str(cons.get('bound_type', params.get('bound_type', 'upper'))).lower()
    if bound not in ('upper', 'lower'):
        raise ValueError(f"Invalid bound_type: {bound}")
    # Target coverage is usually a lower bound. Never silently apply an upper
    # bound to a PTV/GTV dose-volume rule just because PortPy defaults to upper.
    if (kind in ('dose_volume_D', 'dose_volume_V')
            and str(roi.name).upper().startswith(('PTV', 'GTV'))
            and 'bound_type' not in cons and 'bound_type' not in params):
        raise ValueError('Ambiguous target DVH bound: specify bound_type=lower or upper explicitly')
    d, v = roi.doses_gy, roi.volumes_cc
    if kind == 'max_dose':
        metric, unit = 'Dmax', 'Gy'
        actual = float(np.max(d))
    elif kind == 'mean_dose':
        metric, unit = 'Dmean', 'Gy'
        actual = float(np.average(d, weights=v))
    elif kind == 'dose_volume_D':
        key, val = _single_param(params, ('volume_perc', 'volume_cc'))
        if key == 'volume_cc':
            raise ValueError('D(volume_cc) requires independent full-ROI geometry validation')
        p = resolve_number(val, prescription_gy)
        if params.get('coldest', False):
            p = 100 - p
        actual = dose_at_volume(d, v, p)
        metric, unit = f'D{p:g}', 'Gy'
    elif kind == 'dose_volume_V':
        key, val = _single_param(params, ('dose_gy', 'dose_perc'))
        dose = resolve_number(val, prescription_gy)
        if key == 'dose_perc':
            dose = dose * prescription_gy / 100
        actual = volume_at_dose_perc(d, v, dose)
        metric, unit = f'V{dose:g}Gy', '%'
    else:
        raise ValueError(f'Unsupported criterion type: {kind!r}')
    thresholds = _thresholds(criterion, kind, unit, prescription_gy)
    if kind == 'dose_volume_V' and params.get('coldest', False):
        # PortPy convention converts cold-volume to complementary hot volume.
        thresholds = {k: 100.0 - x for k, x in thresholds.items()}
        if any(not (0 <= x <= 100) for x in thresholds.values()):
            raise ValueError('Invalid complemented coldest-volume threshold')
    return metric, actual, unit, thresholds, bound


def judge(actual: float, thresholds: dict, direction: str, tolerance: float = 1e-6) -> str:
    """FAIL: limit violated, WARN: goal violated with limit met, PASS otherwise."""
    def meets(value):
        return actual <= value + tolerance if direction == 'upper' else actual >= value - tolerance

    if 'limit' in thresholds and not meets(thresholds['limit']):
        return 'FAIL'
    if 'goal' in thresholds and not meets(thresholds['goal']):
        return 'WARN'
    return 'PASS'


def evaluate_criteria(case: Any, total_dose_gy: np.ndarray,
                      criteria: list[dict] | None = None,
                      min_coverage: float = 0.999) -> pd.DataFrame:
    """Evaluate every rule without dropping unassessable rows.

    UNDETERMINED whenever a structure lacks dose coverage or a rule is
    unsupported/ambiguous. For 'complete' coverage, results are preliminary
    optimization-grid screening only; clinical_verified is always False.

    `criteria` should ideally be the raw `clinical_criteria_dict['criteria']`
    (copy first) so invalid rules remain visible in this audit.
    """
    dose = np.asarray(total_dose_gy, dtype=np.float64).ravel()
    A = case.influence_matrix.A
    if len(dose) != A.shape[0] or not np.all(np.isfinite(dose)) or np.any(dose < 0):
        raise ValueError('total_dose_gy must be a finite, nonnegative total-course Gy vector matching A rows')
    if criteria is None:
        criteria = case.criteria.clinical_criteria_dict['criteria']
    if not 0 < min_coverage <= 1:
        raise ValueError('min_coverage must be in (0, 1]')
    pres = float(case.plan.get_prescription())
    names = set(case.structures.get_structures())
    cache: dict[str, ROIDose | Exception] = {}
    rows = []
    for i, c in enumerate(deepcopy(criteria), start=1):
        kind = c.get('type', '?')
        params = c.get('parameters', {})
        roi_name = params.get('structure_name')
        cons = c.get('constraints', {})
        row = {
            'criterion_id': i, 'roi': roi_name, 'type': kind,
            'metric': None, 'actual': None, 'unit': None,
            'direction': cons.get('bound_type', params.get('bound_type', 'upper')),
            'goal': None, 'limit': None, 'status': 'UNDETERMINED',
            'coverage': None, 'represented_volume_cc': None, 'metadata_volume_cc': None,
            'metric_scope': 'optimization_grid_screening', 'clinical_verified': False,
            'reason': None,
        }
        if not roi_name or roi_name not in names:
            row['reason'] = f'Missing ROI: {roi_name!r}'
            rows.append(row)
            continue
        if roi_name not in cache:
            try:
                cache[roi_name] = read_roi(case, dose, roi_name)
            except Exception as exc:
                cache[roi_name] = exc
        roi = cache[roi_name]
        if isinstance(roi, Exception):
            row['reason'] = str(roi)
            rows.append(row)
            continue
        row['coverage'] = roi.reported_coverage
        row['represented_volume_cc'] = roi.represented_volume_cc
        row['metadata_volume_cc'] = roi.metadata_volume_cc
        if roi.reported_coverage < min_coverage:
            row['reason'] = f'Partial optimization ROI: coverage={roi.reported_coverage:.1%}'
            rows.append(row)
            continue
        try:
            metric, actual, unit, thresholds, bound = compute_criterion_value(roi, c, pres)
            row.update(metric=metric, actual=actual, unit=unit,
                       direction=bound, goal=thresholds.get('goal'), limit=thresholds.get('limit'),
                       status=judge(actual, thresholds, bound),
                       reason='Provisional optimization-grid result; full geometric/dosimetric QA not completed')
            if roi.metadata_volume_cc is not None and roi.metadata_volume_cc > 0:
                mismatch = abs(roi.represented_volume_cc / roi.metadata_volume_cc - 1)
                if mismatch > 0.03:
                    row['reason'] += f'; metadata-vs-represented volume differs by {mismatch:.1%}'
        except (TypeError, KeyError, ValueError, ZeroDivisionError, SyntaxError, OverflowError) as exc:
            row['reason'] = str(exc)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def print_screening_summary(df: pd.DataFrame) -> None:
    print('Criteria:', len(df))
    print(df['status'].value_counts().to_string())
    print('Undetermined:', (df['status'] == 'UNDETERMINED').sum())
    if df['status'].eq('UNDETERMINED').any():
        print('NOTE: A summary with undetermined rows cannot certify plan compliance.')
