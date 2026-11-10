"""Deliverability / plan-complexity checks for a fluence-map (step-and-shoot IMRT) plan.

The optimizer returns ideal beamlet intensities. A machine delivers them as a few MLC segments per beam.
We run PortPy's leaf sequencer (Siochi 1999, 40 intensity levels) and report:

* segments per beam and relative monitor units (sum of segment weights; same arbitrary unit as the
  fluence, so compare between plans of the SAME patient, e.g. vs. the protocol baseline),
* fluence modulation per beam (total variation / total intensity; 0 = flat, larger = spikier),
* sequencing fidelity: dose recomputed from the SEQUENCED fluence vs. the optimized one. If the PTV D95
  or an OAR max moves noticeably, the plan's numbers are not what the machine would deliver.

This is a screening check, not a machine-specific QA (no leaf speed, MLC transmission, min MU, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = 40
# What we flag (relative to the protocol baseline of the same patient)
FLAG_MU_RATIO = 1.3          # >30 % more MU than baseline
FLAG_SEG_RATIO = 1.3         # >30 % more segments
FLAG_FIDELITY_GY = 0.5       # sequencing moves PTV D95 / an OAR max by more than this


@dataclass
class Deliverability:
    beams: pd.DataFrame          # per beam: gantry, segments, MU, modulation
    total_segments: int
    total_mu: float
    mean_modulation: float
    fidelity: pd.DataFrame       # roi, metric, optimized, sequenced, delta
    max_voxel_diff_gy: float
    maps: dict                   # gantry -> 2-D fluence array (nan outside the field)

    def summary(self) -> dict:
        worst = float(self.fidelity['delta'].abs().max()) if len(self.fidelity) else 0.0
        return {'segments': self.total_segments, 'mu': self.total_mu, 'modulation': self.mean_modulation,
                'fidelity_worst_gy': worst, 'max_voxel_diff_gy': self.max_voxel_diff_gy}


def fluence_maps(case, fluence: np.ndarray) -> dict:
    inf = case.influence_matrix
    out = {}
    for bid in case.beams.get_all_beam_ids():
        grid = np.asarray(inf.get_bev_2d_grid(beam_id=bid, finest_grid=False))
        m = np.full(grid.shape, np.nan)
        inside = grid >= 0
        m[inside] = fluence[grid[inside]]
        out[int(case.beams.get_gantry_angle(beam_id=bid))] = m
    return out


def modulation(m: np.ndarray) -> float:
    """Total variation along and across leaf travel, divided by total intensity."""
    z = np.nan_to_num(m)
    tot = z.sum()
    if tot <= 0:
        return 0.0
    tv = np.abs(np.diff(z, axis=1)).sum() + np.abs(np.diff(z, axis=0)).sum()
    return float(tv / tot)


FIDELITY_ROWS = (('PTV', 'D95'), ('PTV', 'Dmax'), ('ESOPHAGUS', 'Dmax'), ('ESOPHAGUS', 'Dmean'),
                 ('CORD', 'Dmax'), ('HEART', 'Dmean'), ('LUNGS_NOT_GTV', 'Dmean'), ('LUNGS_NOT_GTV', 'V20'))


def assess(session, fluence: np.ndarray, metrics: dict | None = None) -> Deliverability:
    from portpy.photon.utils.leaf_sequencing_siochi import leaf_sequencing_siochi
    from .data_adapter import from_fluence
    case = session.case
    fluence = np.asarray(fluence, dtype=float)
    seq = leaf_sequencing_siochi(case.plan, {'optimal_intensity': fluence}, num_of_levels=LEVELS)
    maps = fluence_maps(case, fluence)
    rows = []
    for gantry, m in maps.items():
        s = seq.get(gantry, {})
        rows.append({'gantry': gantry, 'segments': len(s.get('MU', [])), 'MU (rel.)': float(np.sum(s.get('MU', []))),
                     'modulation': round(modulation(m), 3)})
    beams = pd.DataFrame(rows)
    d_opt = from_fluence(case, fluence)
    d_seq = from_fluence(case, np.asarray(seq['optimal_intensity'], dtype=float))
    m_opt = metrics or session.key_metrics(d_opt)
    m_seq = session.key_metrics(d_seq)
    fid = []
    for roi, k in FIDELITY_ROWS:
        a, b = m_opt.get(roi, {}), m_seq.get(roi, {})
        if k in a and k in b:
            fid.append({'roi': roi, 'metric': k, 'unit': '%' if k.startswith('V') else 'Gy',
                        'optimized': round(a[k], 2), 'after sequencing': round(b[k], 2),
                        'delta': round(b[k] - a[k], 2)})
    return Deliverability(beams, int(beams['segments'].sum()), float(beams['MU (rel.)'].sum()),
                          float(beams['modulation'].mean()), pd.DataFrame(fid),
                          float(np.abs(d_opt - d_seq).max()), maps)


def flags(d: Deliverability, base: Deliverability | None) -> list[str]:
    out = []
    if base is not None and base.total_mu > 0:
        r = d.total_mu / base.total_mu
        if r > FLAG_MU_RATIO:
            out.append(f'MU {r:.0%} of the baseline plan (more modulation, longer delivery, more leakage dose)')
        r = d.total_segments / max(base.total_segments, 1)
        if r > FLAG_SEG_RATIO:
            out.append(f'{r:.0%} of the baseline segment count')
    if len(d.fidelity):
        bad = d.fidelity[d.fidelity['delta'].abs() > FLAG_FIDELITY_GY]
        bad = bad[bad['unit'] == 'Gy']
        for _, r in bad.iterrows():
            out.append(f"leaf sequencing changes {r['roi']} {r['metric']} by {r['delta']:+.2f} Gy")
    return out
