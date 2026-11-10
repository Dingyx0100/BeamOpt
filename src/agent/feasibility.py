"""Deterministic data-availability judge: can this ROI be evaluated / optimized for this patient?

PortPy ships a precomputed dose-influence matrix only for voxels inside a calculation box around the
target. An organ outside that box has no rows in the matrix, so the optimizer cannot compute or change its
dose (PortPy 1.2.0 has no dose engine to compute new rows). This module turns that into an explicit,
numbers-backed reason the agent can give the physicist, before any LLM-driven optimization starts.

Status per ROI:
  ok        - fully inside the box
  partial   - partly inside; metrics cover only the represented sub-volume (caveat, still optimizable)
  no_voxels - contoured, but no voxel inside the box -> cannot evaluate or optimize
  absent    - no such contour for this patient
"""
from __future__ import annotations

import re

import numpy as np

# Words a physicist uses for each ROI (lowercase regexes). Used as a deterministic backup to the LLM parse.
ROI_PATTERNS = {
    'HEART': [r'\bheart\b', r'\bcardiac\b', r'\bpericard'],
    'ESOPHAGUS': [r'o?esophag'],
    'CORD': [r'\bcord\b', r'\bspinal\b'],
    'LUNGS_NOT_GTV': [r'\blungs?\b', r'\bmld\b'],
    'PTV': [r'\bptv\b'],
}
_NICE = {'HEART': 'heart', 'ESOPHAGUS': 'esophagus', 'CORD': 'spinal cord', 'LUNGS_NOT_GTV': 'lungs',
         'LUNG_L': 'left lung', 'LUNG_R': 'right lung', 'PTV': 'PTV'}


def mentioned_rois(text: str) -> list[str]:
    t = (text or '').lower()
    out = []
    for roi, pats in ROI_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            out.append(roi)
    if re.search(r'\bleft lung\b', t):
        out.append('LUNG_L')
    if re.search(r'\bright lung\b', t):
        out.append('LUNG_R')
    return out


def roi_status(session, roi: str) -> dict:
    roi = roi.upper()
    if roi not in session.structures:
        return {'roi': roi, 'status': 'absent', 'coverage': 0.0}
    r = session.roi(roi)
    if r.idx.size == 0:
        return {'roi': roi, 'status': 'no_voxels', 'coverage': float(r.coverage)}
    return {'roi': roi, 'status': 'ok' if r.coverage >= 0.999 else 'partial', 'coverage': float(r.coverage),
            'n_voxels': int(r.idx.size)}


def _geometry(session, roi: str) -> dict:
    """Where the ROI sits relative to the dose box / PTV, and its dose in the ECHO reference plan.

    Best effort: needs the real PortPy case (3-D masks, CT grid, RTDOSE). Returns {} if unavailable.
    """
    case = session.case
    try:
        plan, inf = case.plan, case.influence_matrix
        st = plan.structures
        dz = float(plan.ct.ct_dict['resolution_xyz_mm'][2])
        roi_m = np.asarray(st.get_structure_mask_3d(roi)).astype(bool)
        box = np.asarray(inf.dose_1d_to_3d(dose_1d=np.ones(inf.A.shape[0]))) > 0
        out = {}
        rz = np.flatnonzero(roi_m.any(axis=(1, 2)))
        bz = np.flatnonzero(box.any(axis=(1, 2)))
        if rz.size and bz.size:
            gap = max(bz.min() - rz.max(), rz.min() - bz.max(), 0)
            out['box_gap_cm'] = round(gap * dz / 10, 1)
            if gap == 0:
                out['box_gap_cm'] = 0.0      # overlaps axially but lies outside laterally
        if 'PTV' in session.structures:
            pz = np.flatnonzero(np.asarray(st.get_structure_mask_3d('PTV')).astype(bool).any(axis=(1, 2)))
            if rz.size and pz.size:
                gap = max(pz.min() - rz.max(), rz.min() - pz.max(), 0)
                out['ptv_gap_cm'] = round(gap * dz / 10, 1)
        ref = case.patient_dir / 'DicomFiles' / 'rt_dose_echo_imrt.dcm' if case.patient_dir else None
        if ref is not None and ref.is_file():
            import portpy.photon as pp
            d3 = np.asarray(pp.convert_dose_rt_dicom_to_portpy(my_plan=plan, dose_file_name=str(ref)))
            if d3.shape == roi_m.shape and roi_m.any():
                v = d3[roi_m]
                out['ref_mean_gy'] = round(float(v.mean()), 2)
                out['ref_max_gy'] = round(float(v.max()), 2)
        return out
    except Exception:   # noqa: BLE001 - explanation is optional context, never a failure
        return {}


def explain_unavailable(session, roi: str, cache: dict | None = None) -> str | None:
    """Explicit reason why ROI cannot be optimized, or None if it can (ok / partial)."""
    s = roi_status(session, roi)
    nice = _NICE.get(s['roi'], s['roi'])
    if s['status'] == 'absent':
        return (f'This patient has no {s["roi"]} contour in the structure set, so there is no {nice} dose to '
                f'evaluate or optimize.')
    if s['status'] != 'no_voxels':
        return None
    key = ('geom', s['roi'])
    if cache is not None and key in cache:
        g = cache[key]
    else:
        g = _geometry(session, s['roi'])
        if cache is not None:
            cache[key] = g
    msg = (f'The {nice} is contoured, but none of it lies inside the dose calculation box for this patient '
           f'(0% of its volume), so the dose-influence data has no {nice} voxels. Without that data the optimizer '
           f'cannot calculate or change the {nice} dose, so I did not change the plan.')
    where = []
    if g.get('ptv_gap_cm') is not None:
        where.append(f'it lies about {g["ptv_gap_cm"]:g} cm from the PTV along the patient axis')
    if g.get('ref_mean_gy') is not None:
        where.append(f'in the ECHO reference plan for this patient (same beams) it receives a mean of '
                     f'{g["ref_mean_gy"]:.2f} Gy and a maximum of {g["ref_max_gy"]:.2f} Gy')
    if where:
        msg += ' For context: ' + '; '.join(where) + '.'
        if g.get('ref_max_gy') is not None and g['ref_max_gy'] < 5:
            msg += f' So the {nice} is essentially out of field and there is little dose to reduce.'
    return msg


def partial_note(session, roi: str) -> str | None:
    s = roi_status(session, roi)
    if s['status'] == 'partial':
        return (f'{s["roi"]} metrics cover only the {s["coverage"]:.0%} of the organ inside the dose calculation box '
                f'(the rest has no dose data).')
    return None


# ------------------------------------------------------------------ limit changes ("relax the cord limit")
_LIMIT = r'(?:limit|constraint|cap|ceiling)s?'
_LOOSEN = r'(?:relax|loosen|raise|increase|lift|ease|bump|up|remove|drop|drop the)'
_TIGHTEN = r'(?:tighten|lower|reduce|decrease)'
_SET = r'(?:set|change|move|make|put|adjust)'


def _verb_on_limit(verbs: str, t: str) -> bool:
    # the verb must govern the limit itself: "relax the cord max limit", "raise the limit on the cord"
    return bool(re.search(rf'\b{verbs}\b\s+(?:the\s+|a\s+|its\s+|that\s+)?[\w\s%-]{{0,30}}?\b{_LIMIT}\b', t))


def mentions_limit(text: str) -> bool:
    """Does the request talk about a limit / constraint / cap at all (or 'allow ... up to')?"""
    t = (text or '').lower()
    return bool(re.search(rf'\b{_LIMIT}\b|\btolerance\b', t) or
                re.search(r'\b(?:allow|let|permit|accept)\b[^.]{0,40}\b(?:up to|as high as|as much as)\b', t))


def limit_as_goal(goal):
    """Turn an LLM 'change_limit' reading of a plain dose request into the equivalent optimize goal."""
    from src.agent.schemas import GoalItem
    lim = goal.limit
    if lim is None or not lim.roi:
        return goal.model_copy(update={'intent': 'optimize', 'limit': None})
    metric = 'Dmean' if lim.kind == 'mean' else 'Dmax'
    item = GoalItem(roi=lim.roi, metric=metric, change='reduce_to' if lim.value is not None else 'reduce',
                    value=lim.value)
    keep = [o for o in goal.objectives if not (o.roi == item.roi and o.metric == item.metric)]
    return goal.model_copy(update={'intent': 'optimize', 'limit': None, 'objectives': [item, *keep]})


def parse_limit_request(text: str) -> dict | None:
    """'Relax the cord max limit to 55 Gy' -> {'direction': 'loosen', 'roi': 'CORD', 'kind': 'max', 'value': 55}.

    Only fires when a verb acts on a LIMIT / constraint (not on the dose itself), so 'lower the cord max dose
    to 15 Gy' or 'reduce X while staying under the protocol limit' stay ordinary optimization requests.
    """
    t = (text or '').lower()
    if _verb_on_limit(_LOOSEN, t) or re.search(
            r'\b(?:allow|let|permit|accept)\b[^.]{0,40}\b(?:up to|as high as|as much as|to go up to|to reach)\b', t):
        direction = 'loosen'
    elif _verb_on_limit(_TIGHTEN, t):
        direction = 'tighten'
    elif _verb_on_limit(_SET, t):
        direction = 'set'
    else:
        return None
    rois = [r for r in mentioned_rois(t) if r != 'PTV'] or (['PTV'] if 'PTV' in mentioned_rois(t) else [])
    kind = 'mean' if re.search(r'\bmean\b|\bmld\b', t) else 'max'
    nums = re.findall(r'(\d+(?:\.\d+)?)\s*(c?gy)\b', t)
    value = None
    if nums:
        v, unit = nums[-1]
        value = float(v) / (100 if unit == 'cgy' else 1)
    return {'direction': direction, 'roi': rois[0] if rois else None, 'kind': kind, 'value': value}


# ------------------------------------------------------------------ reachability (step 2, 2026-11-10)
# Before spending attempts, ask the optimizer how low the requested metric can go at all while PTV coverage and every
# hard limit are kept (PlanSession.floor_probe). Round 3 spent 88 of 269 minutes on goals that ended not met or
# partial, e.g. "esophagus max below 15 Gy" on Lung_Patient_20: 5 attempts, 17 minutes, plan unchanged.
FLOOR_MARGIN = {'Gy': 0.3, '%': 0.3}      # floor must exceed the goal by more than this (and 3 %) to call it out
UPFRONT_REDUCTION = 0.25                  # probe before the first attempt when the goal asks for >= 25 % less


def ptv_overlap_pct(session, roi: str) -> float:
    """Share (% of volume) of the organ's optimization voxels that lie inside the PTV."""
    try:
        r, t = session.roi(roi), session.roi('PTV')
    except Exception:  # noqa: BLE001
        return 0.0
    inside = np.isin(r.idx, t.idx)
    return float(100 * r.vol_cc[inside].sum() / max(r.vol_cc.sum(), 1e-12))


def probe_targets(checks: list[dict]) -> list[dict]:
    """Objectives the floor probe can answer: '<=' goals on an organ (not the PTV) with a supported metric."""
    return [c for c in checks if c['role'] == 'objective' and c['op'] == '<=' and c['roi'] not in ('PTV', 'GTV', 'CTV')
            and (c['metric'] in ('Dmean', 'Dmax') or re.fullmatch(r'[DV]\d+(\.\d+)?', c['metric']))]


def wants_upfront(session, checks: list[dict]) -> bool:
    """Probe before the first attempt when a goal is a large cut, or the organ sits inside the PTV for a max goal."""
    for c in probe_targets(checks):
        ref = c.get('ref')
        if ref and ref > 0 and (ref - c['threshold']) / ref >= UPFRONT_REDUCTION:
            return True
        if c['metric'] in ('Dmax', 'D2') and ptv_overlap_pct(session, c['roi']) > 0:
            return True
    return False


def floor_margin(check: dict) -> float:
    """How far the floor must lie above the goal to call the goal unreachable."""
    unit = '%' if check['metric'].startswith('V') else 'Gy'
    return max(FLOOR_MARGIN[unit], 0.03 * abs(check['threshold']))


def floor_verdict(check: dict, probe: dict) -> str:
    """'unreachable' | 'reachable' | 'unknown' for one objective, from its probe result."""
    val = probe.get('value')
    if val is None:
        return 'unknown'
    return 'unreachable' if val > check['threshold'] + floor_margin(check) else 'reachable'


def _share(pct: float) -> str:
    return 'less than 0.1 %' if 0 < pct < 0.1 else f'{pct:.1f} %'


def floor_text(session, check: dict, probe: dict, d95_min: float) -> str:
    unit = '%' if check['metric'].startswith('V') else 'Gy'
    roi, met = check['roi'], check['metric']
    ov = ptv_overlap_pct(session, roi)
    where = (f' {_share(ov)} of the {_NICE.get(roi, roi)} lies inside the PTV, which must keep D95 >= {d95_min:.2f} Gy.'
             if ov > 0 else '')
    if floor_verdict(check, probe) == 'unreachable' and probe.get('bound'):
        return (f"{roi} {met} <= {check['threshold']:.2f} {unit} is not reachable while protecting the PTV: "
                f"{_share(probe.get('inside_pct', ov))} of the {_NICE.get(roi, roi)} lies inside the PTV, which must keep "
                f"D95 >= {d95_min:.2f} Gy, so {roi} {met} cannot go below about {probe['value']:.2f} {unit}.")
    if floor_verdict(check, probe) == 'unreachable':
        return (f"{roi} {met} <= {check['threshold']:.2f} {unit} is not reachable while protecting the PTV: the lowest "
                f"{roi} {met} the optimizer could find, with PTV D95 >= {d95_min:.2f} Gy and every hard limit kept and "
                f"no other organ considered, is {probe['value']:.2f} {unit}.{where}")
    return (f"{roi} {met} <= {check['threshold']:.2f} {unit} is reachable: the optimizer found {probe['value']:.2f} "
            f"{unit} with PTV coverage kept (when it cares only about this organ, so expect costs elsewhere).")
