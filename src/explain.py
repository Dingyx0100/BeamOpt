"""Exact, deterministic explanations of a plan: where each organ's dose comes from, and what holds the plan back.

Dose is linear in the controls (dose = A @ fluence), so the dose every beam gives every voxel is exact, not a model.
This is modality-agnostic: beamlets here, dwell positions / proton spots / control points elsewhere.

* beam_shares: per beam, how much of an organ metric it delivers. Dmean uses the whole (represented) organ; Dmax and
  D2 use the hottest 2 % of the organ; VX uses the voxels at or above X Gy.
* binding_limits: the hard limits the optimizer is pressed against, ranked by the solver's dual values. A dual is the
  price of a limit in objective units, so it ranks limits but is not a Gy trade-off by itself; `pull` is the
  elasticity (relative objective change per relative change of the limit), comparable across limits.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def beam_groups(session) -> list[dict]:
    """[{beam_id, gantry, cols}] for native beamlet columns."""
    inf = session.case.influence_matrix
    beams = getattr(session.case, 'beams', None)
    out = []
    for b in getattr(inf, 'beamlets_dict', None) or []:
        bid = b.get('beam_id')
        try:
            g = float(beams.get_gantry_angle(beam_id=bid)) if beams is not None else None
        except Exception:  # noqa: BLE001 - metadata shape differs between PortPy versions
            g = None
        out.append({'beam_id': bid, 'gantry': g,
                    'cols': np.arange(int(b['start_beamlet_idx']), int(b['end_beamlet_idx']) + 1)})
    if not out:                                   # synthetic cases: one pseudo-beam per beamlet
        out = [{'beam_id': j, 'gantry': None, 'cols': np.array([j])} for j in range(session.A_full.shape[1])]
    return out


def _region(roi_dose: np.ndarray, vol: np.ndarray, metric: str) -> tuple[np.ndarray, str]:
    if metric.startswith('V') and metric[1:].replace('.', '', 1).isdigit():
        x = float(metric[1:])
        return roi_dose >= x, f'voxels at or above {x:g} Gy'
    if metric in ('Dmax', 'D2'):
        order = np.argsort(-roi_dose, kind='stable')
        cum = np.cumsum(vol[order]) / max(vol.sum(), 1e-12)
        k = int(np.searchsorted(cum, 0.02)) + 1
        mask = np.zeros(roi_dose.size, bool)
        mask[order[:k]] = True
        return mask, 'hottest 2 % of the organ'
    return np.ones(roi_dose.size, bool), 'whole organ' if metric == 'Dmean' else 'whole organ'


def beam_shares(session, version_id: int, roi: str, metric: str = 'Dmean') -> dict:
    v = session.get_version(version_id)
    if not session.has_voxels(roi) or v.fluence is None:
        return {'roi': roi, 'metric': metric, 'available': False, 'beams': []}
    r = session.roi(roi)
    x = np.asarray(v.fluence, dtype=float)
    A = session.A_full[r.idx, :]
    fx = session.num_fx
    groups = beam_groups(session)
    B = sp.csr_matrix((np.ones(sum(g['cols'].size for g in groups)),
                       (np.concatenate([g['cols'] for g in groups]),
                        np.concatenate([[i] * g['cols'].size for i, g in enumerate(groups)]))),
                      shape=(A.shape[1], len(groups)))
    per_beam = np.asarray((A @ sp.diags(x) @ B).todense()) * fx          # [n_vox, n_beams], total-course Gy
    total = per_beam.sum(axis=1)
    mask, region = _region(total, r.vol_cc, metric)
    w = r.vol_cc * mask
    if w.sum() <= 0:
        return {'roi': roi, 'metric': metric, 'available': True, 'region': region, 'n_voxels': 0, 'beams': []}
    gy = (per_beam * w[:, None]).sum(axis=0) / w.sum()                   # volume-weighted mean dose per beam
    tot = float(gy.sum())
    rows = [{'beam_id': g['beam_id'], 'gantry': g['gantry'], 'gy': round(float(d), 3),
             'share': round(float(d) / tot, 4) if tot > 0 else 0.0} for g, d in zip(groups, gy)]
    return {'roi': roi, 'metric': metric, 'available': True, 'region': region, 'n_voxels': int(mask.sum()),
            'mean_gy': round(tot, 3), 'coverage': round(r.coverage, 3),
            'beams': sorted(rows, key=lambda b: -b['share'])}


def binding_limits(session, version_id: int, tol_gy: float = 0.1) -> list[dict]:
    """Hard limits at (within tol of) their bound in this plan, with dual-based ranking when available."""
    v = session.get_version(version_id)
    if v.dose_total_gy is None:
        return []
    duals = getattr(v, 'duals', None) or {}
    out = []
    for c in v.constraints.values():
        if not session.has_voxels(c.roi):
            continue
        r = session.roi(c.roi)
        d = np.asarray(v.dose_total_gy, dtype=float)[r.idx]
        if c.kind == 'max':
            value = float(d.max())
            limit = c.limit_gy
        else:
            value = float((d * r.vol_cc).sum() / max(r.vol_cc.sum(), 1e-12))
            limit = c.limit_gy / max(r.coverage, 1e-6) if c.scope == 'whole_organ' else c.limit_gy
        info = duals.get(c.id, {})
        at = value >= limit - tol_gy or info.get('dual', 0) > 1e-9
        if not at:
            continue
        out.append({'id': c.id, 'roi': c.roi, 'kind': c.kind, 'limit_gy': round(float(c.limit_gy), 3),
                    'value_gy': round(value, 3), 'source': c.source, 'dual': info.get('dual'),
                    'pull': info.get('pull')})
    tot = sum(o['pull'] or 0 for o in out)
    for o in out:
        o['pull_share'] = round((o['pull'] or 0) / tot, 3) if tot > 0 else None
    return sorted(out, key=lambda o: -(o['pull'] or 0))


def fmt_binding(rows: list[dict]) -> str:
    if not rows:
        return '(no hard limit is active: the plan is shaped by the objective weights only)'
    out = []
    for o in rows:
        what = 'max' if o['kind'] == 'max' else 'mean'
        share = f", {o['pull_share']:.0%} of the pull" if o.get('pull_share') is not None else ''
        out.append(f"{o['roi']} {what} <= {o['limit_gy']:g} Gy ({o['source']}): at {o['value_gy']:.2f} Gy{share}")
    return '\n'.join(out)


def fmt_shares(info: dict, top: int = 3) -> str:
    if not info.get('beams'):
        return ''
    parts = [f"{('gantry ' + format(b['gantry'], 'g') + '°') if b['gantry'] is not None else 'beam ' + str(b['beam_id'])}"
             f" {b['share']:.0%}" for b in info['beams'][:top]]
    return f"{info['roi']} {info['metric']} ({info['region']}): " + ', '.join(parts)
