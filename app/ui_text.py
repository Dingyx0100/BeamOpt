"""Clinical wording and the shared visual language of the console.

Internal ids (LUNGS_NOT_GTV, quadratic_overdose:PTV, max_dose) stay in the data and in exports; only what the
user reads is translated here. Style: neutral greys, hairline borders, one ink-blue accent, tabular numbers,
no emoji in the interface chrome.
"""
from __future__ import annotations

import json

ROI_LABEL = {
    'PTV': 'PTV', 'GTV': 'GTV', 'CTV': 'CTV', 'ESOPHAGUS': 'Esophagus', 'CORD': 'Spinal cord', 'SPINAL_CORD': 'Spinal cord',
    'HEART': 'Heart', 'LUNGS_NOT_GTV': 'Total lung − GTV', 'TOTAL LUNG - GTV': 'Total lung − GTV',
    'LUNG_L': 'Left lung', 'LUNG_R': 'Right lung', 'LAD': 'LAD', 'BRACHIALPLEXUS': 'Brachial plexus',
    'GREATVESSELS': 'Great vessels', 'TRACHEA': 'Trachea', 'Ring_PTV': 'PTV ring', 'BODY_RING': 'Body rind',
    'SKIN': 'Skin', 'BODY': 'Body',
}

CRIT_TYPE = {'max_dose': 'Max dose', 'mean_dose': 'Mean dose', 'dose_volume_V': 'Volume at dose',
             'dose_volume_D': 'Dose at volume'}
STATUS_TEXT = {'PASS': 'Pass', 'WARN': 'Near limit', 'FAIL': 'FAIL', 'UNDETERMINED': 'Not checkable'}

# Shared palette
INK, SUB, LINE, ACCENT = '#1b2230', '#5d6676', '#dde1e7', '#2b59c3'
OK, WARN, BAD, NEUTRAL = '#1f8a5b', '#b7791f', '#c2412d', '#8a93a3'


def roi_label(name) -> str:
    if name is None:
        return ''
    s = str(name)
    if s.upper().startswith('RIND_'):
        return f'Rind {s.split("_", 1)[1]} (optimization ring)'
    return ROI_LABEL.get(s, ROI_LABEL.get(s.upper(), s.replace('_', ' ').title()))


def unit(metric: str) -> str:
    return '%' if str(metric).startswith('V') else 'Gy'


def metric_label(metric: str) -> str:
    m = str(metric)
    if m == 'V95%Rx':
        return 'V95%Rx (%)'
    return f'{m} ({unit(m)})'


def knob_text(t) -> str:
    """'Penalise dose above 60 Gy' instead of quadratic_overdose(PTV @ 60 Gy) w=10000."""
    kind = {'quadratic': 'Lower overall dose (quadratic)', 'quadratic_overdose': 'Penalise dose above threshold',
            'quadratic_underdose': 'Penalise dose below threshold', 'smoothness': 'Fluence smoothness',
            'mean': 'Lower mean dose', 'linear_overdose': 'Penalise dose above threshold (linear)'}.get(t.kind, t.kind)
    thr = f' {t.dose_gy:g} Gy' if t.dose_gy is not None else ''
    return f'{kind}{thr}'


def to_json(x) -> str:
    return json.dumps(x)
