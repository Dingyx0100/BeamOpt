"""Protocol plan-quality scorecard (PortPy ``compute_total_quality_score``) with per-item maxima.

A scorecard is a list of piecewise-linear score curves (e.g. PTV V60Gy: 94.5 % -> 0 pts ... 100 % -> 15 pts).
The total is the sum over items whose structure exists for this patient; items for missing structures
(LAD, brachial plexus, ...) are listed as "not scored" and excluded from the achievable maximum, so the
percentage is comparable between patients with different structure sets.

PortPy convention: DVH metrics of an organ only partly inside the dose box treat the out-of-box part as
0 Gy (whole-organ volume), i.e. they are optimistic for partial ROIs. Those items are labelled.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SC_DIR = Path(__file__).resolve().parents[1] / '.venv/Lib/site-packages/portpy/config_files/scorecard'
SCORECARDS = {                       # protocol -> scorecard json (only protocols we have a card for)
    'Lung_2Gy_30Fx': 'SC_Lung(60Gy)_2022MAAS_ExampleV2.json',
}
ALIAS = {'TOTAL LUNG - GTV': 'LUNGS_NOT_GTV', 'SPINAL_CORD': 'CORD'}


def scorecard_path(protocol: str) -> Path | None:
    name = SCORECARDS.get(protocol)
    if name is None:
        return None
    try:   # prefer the installed package location (works outside this venv layout too)
        import portpy
        p = Path(portpy.__file__).parent / 'config_files' / 'scorecard' / name
        if p.is_file():
            return p
    except ImportError:
        pass
    p = _SC_DIR / name
    return p if p.is_file() else None


def _metric_label(item: dict) -> str:
    t, v, u = item['MetricType'], item.get('InputValue'), item.get('InputUnit')
    if t == 'VolumeAtDose':
        return f'V{float(v):g}Gy (%)'
    if t == 'DoseAtVolume':
        return f'D{float(v):g}{"cc" if u == "CC" else "%"} (Gy)'
    if t == 'MeanDose':
        return 'Dmean (Gy)'
    if t == 'ConformationNumber':
        return f'CN @ {float(v):g} Gy'
    return t


@dataclass
class ScoreResult:
    total: float
    achievable: float           # sum of item maxima over scored items
    possible: float             # sum of item maxima over ALL items of the card
    items: pd.DataFrame         # structure, metric, value, score, max, status, note

    @property
    def percent(self) -> float:
        return 100.0 * self.total / self.achievable if self.achievable else float('nan')


class Scorecard:
    def __init__(self, case, protocol: str):
        import json
        self.case = case
        self.path = scorecard_path(protocol)
        self.card = json.loads(self.path.read_text(encoding='utf-8')) if self.path else None
        self.name = self.card.get('TemplateName', self.path.stem) if self.card else None
        inf = case.influence_matrix
        names = {s.upper(): s for s in case.structures.get_structures()}
        self._cov = {}
        for item in (self.card or {}).get('ScoreTemplates', []):
            sid = (item['Structure'].get('StructureId') or '').upper()
            real = names.get(ALIAS.get(sid, sid))
            if real is not None:
                try:
                    self._cov[sid] = float(inf.get_fraction_of_vol_in_calc_box(real) or 0.0)
                except Exception:  # noqa: BLE001
                    self._cov[sid] = None

    @property
    def available(self) -> bool:
        return self.card is not None

    def score(self, dose_total_gy: np.ndarray) -> ScoreResult:
        import portpy.photon as pp
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            total, df = pp.Evaluation.compute_total_quality_score(
                self.case.plan, np.asarray(dose_total_gy), self.card,
                prescription_gy=self.case.plan.get_prescription(), alias_map=ALIAS)
        rows, achievable, possible = [], 0.0, 0.0
        for item, (_, r) in zip(self.card['ScoreTemplates'], df.iterrows()):
            mx = max(float(p['Score']) for p in item['ScorePoints'])
            possible += mx
            sid = (item['Structure'].get('StructureId') or '').upper()
            ok = r['Status'] == 'ok' and r['Score'] == r['Score']
            cov = self._cov.get(sid)
            note = item.get('MetricComment') or ''
            if not ok:
                note = 'not scored: ' + ('structure not in this patient' if r['Status'] == 'missing_structure'
                                         else str(r['Status']))
            elif cov is not None and cov < 0.999:
                note = (note + '; ' if note else '') + f'{cov:.0%} of organ in dose box (rest counted as 0 Gy)'
            if ok:
                achievable += mx
            rows.append({'structure': item['Structure'].get('StructureId'), 'metric': _metric_label(item),
                         'value': round(float(r['MetricValue']), 2) if ok else None,
                         'score': round(float(r['Score']), 2) if ok else None, 'max': mx, 'note': note})
        return ScoreResult(float(total), achievable, possible, pd.DataFrame(rows))
