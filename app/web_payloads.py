"""JSON-ready views of a Workspace for the web app (app/server.py). Mixed into app.workspace.Workspace."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import ui_text as UT
from app.viz import ROI_COLORS

DVH_ROIS = ['PTV', 'ESOPHAGUS', 'CORD', 'LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R']

CRIT_METRIC = {'max_dose': 'Dmax', 'mean_dose': 'Dmean'}
DVH_STEP_GY = 0.25
BOARD_SKIP = {'SKIN', 'BODY', 'GTV'}       # protocol rows the goals board leaves to the Details drawer


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _coverage(scope) -> float | None:
    """'represented_subvolume(56% of organ)' -> 0.56; 'whole_roi' -> 1.0."""
    s = str(scope or '')
    if s.startswith('whole'):
        return 1.0
    if '(' in s and '%' in s:
        try:
            return float(s.split('(')[1].split('%')[0]) / 100
        except ValueError:
            return None
    return None


class WebPayloads:
    """JSON-ready views of a Workspace for the web app. Mixed into Workspace."""

    def protocol_rows(self) -> list[dict]:
        """One row per protocol criterion with its limit / goal (the client scores any column against them)."""
        rows = []
        for c in (self.case.criteria.clinical_criteria_dict.get('criteria') or []):
            p, k = c.get('parameters', {}), c.get('constraints', {})
            roi, typ = p.get('structure_name'), c.get('type')
            if typ in CRIT_METRIC:
                metric, unit = CRIT_METRIC[typ], 'Gy'
                lim, goal = k.get('limit_dose_gy'), k.get('goal_dose_gy')
            elif typ == 'dose_volume_V':
                metric, unit = f"V{_num(p.get('dose_gy')):g}", '%'
                lim, goal = k.get('limit_volume_perc'), k.get('goal_volume_perc')
            else:
                continue
            if lim is None and goal is None:
                continue
            rows.append({'roi': roi, 'metric': metric, 'op': '<=', 'limit': _num(lim), 'goal': _num(goal),
                         'unit': unit, 'kind': 'protocol', 'board': roi not in BOARD_SKIP})
        merged = {}                 # HEART V30 has a limit row and a separate goal row: merge them
        for r in rows:
            key = (r['roi'], r['metric'])
            if key in merged:
                for f in ('limit', 'goal'):
                    merged[key][f] = merged[key][f] if merged[key][f] is not None else r[f]
            else:
                merged[key] = r
        return list(merged.values())

    def roi_info(self) -> dict:
        """Per ROI: coverage of the dose box, availability and a plain-language reason when unavailable."""
        out = {}
        m0 = self.session.get_version(0).metrics or {}
        for roi, m in m0.items():
            if m.get('status') == 'unavailable':
                why = m.get('reason', '')
                absent = 'absent' in why.lower()
                out[roi] = {'available': False, 'coverage': 0.0,
                            'short': 'not contoured' if absent else 'outside dose box',
                            'reason': 'this patient has no contour for it' if absent else
                            'none of it lies inside the dose calculation box, so it cannot be evaluated or optimized'}
            else:
                out[roi] = {'available': True, 'coverage': _coverage(m.get('scope'))}
        for roi in self.session.structures:
            out.setdefault(roi, {'available': self.session.has_voxels(roi), 'coverage': None})
        return out

    def header_info(self) -> dict:
        try:
            b = self.case.beams
            gantry = sorted(int(b.get_gantry_angle(beam_id=i)) for i in b.get_all_beam_ids())
        except Exception:  # noqa: BLE001
            gantry = []
        notes = []
        if not self.has_hints:
            notes.append('Not calibrated yet: the agent uses generic guidance')
        if self.restored_from:
            notes.append(f'Restored session {Path(self.restored_from).name}')
        return {'patient': self.pid, 'protocol': self.protocol, 'rx_gy': self.session.rx_gy,
                'fractions': int(self.case.plan.get_num_of_fractions()), 'gantry': gantry, 'model': self.model,
                'notes': notes, 'data_dir': self.cfg['data']['data_dir'], 'key': self.key,
                'has_echo': self.reference is not None}

    def knob_diff(self, a: int, b: int) -> list[dict]:
        """Optimizer settings that differ between plan versions a -> b (what a revision changed)."""
        va, vb = self.session.get_version(a), self.session.get_version(b)
        out = []
        for tid, t in vb.terms.items():
            old = va.terms.get(tid)
            if old is None and t.weight > 0:
                out.append({'what': f'{UT.roi_label(t.roi) if t.roi else "All beams"}: {UT.knob_text(t)}',
                            'detail': f'weight {t.weight:g}', 'tag': f'added by {t.source}'})
            elif old is not None and abs(old.weight - t.weight) > 1e-9:
                out.append({'what': f'{UT.roi_label(t.roi) if t.roi else "All beams"}: {UT.knob_text(t)}',
                            'detail': f'weight {old.weight:g} → {t.weight:g}', 'tag': ''})
        for cid, c in vb.constraints.items():
            old = va.constraints.get(cid)
            if old is None or abs(old.limit_gy - c.limit_gy) > 1e-9:
                out.append({'what': f'{UT.roi_label(c.roi)} {c.kind} dose ≤ {c.limit_gy:g} Gy',
                            'detail': '' if old is None else f'was {old.limit_gy:g} Gy',
                            'tag': f'hard limit · added by {c.source}'})
        for cid, c in va.constraints.items():
            if cid not in vb.constraints:
                out.append({'what': f'{UT.roi_label(c.roi)} {c.kind} dose limit removed', 'detail':
                            f'was {c.limit_gy:g} Gy', 'tag': ''})
        return out

    def dvh_curves(self, dose) -> dict:
        from src.data_adapter import get_roi_dose
        from src.metrics import cumulative_dvh
        top = max(66.0, float(np.max(dose)) * 1.02)
        bins = np.arange(0, top + DVH_STEP_GY, DVH_STEP_GY)
        out = {}
        for roi in DVH_ROIS + ['GTV', 'HEART']:
            try:
                r = get_roi_dose(self.case, dose, roi)
            except (KeyError, ValueError):
                continue
            _, ys = cumulative_dvh(r, bins)
            out[roi] = [round(float(y), 2) for y in ys]
        return {'step': DVH_STEP_GY, 'curves': out}

    def version_payload(self, vid: int) -> dict:
        v = self.session.get_version(int(vid))
        if not v.ok:
            return {'id': v.id, 'ok': False, 'status': v.status}
        metrics = {roi: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}
                   for roi, m in (v.metrics or {}).items()}
        crit = []
        if v.criteria is not None:
            for _, r in v.criteria.iterrows():
                crit.append({'roi': r['roi'], 'type': r['type'], 'metric': r.get('metric'),
                             'actual': _num(r.get('actual')), 'goal': _num(r.get('goal')), 'limit': _num(r.get('limit')),
                             'status': r['status'], 'coverage': _num(r.get('coverage')), 'reason': r.get('reason')})
        knobs = self.knobs_df(v).drop(columns=['id'], errors='ignore')
        rev = self.revlog.by_version(v.id)
        return {'id': v.id, 'ok': True, 'parent': v.parent, 'label': v.label, 'metrics': metrics,
                'criteria': crit, 'knobs': json.loads(knobs.to_json(orient='records')),
                'solve_s': v.solve_seconds, 'revision': rev.id if rev else None,
                'dvh': self.dvh_curves(v.dose_total_gy)}

    def echo_payload(self) -> dict | None:
        if self.reference is None:
            return None
        if not hasattr(self, '_echo_payload'):
            refm = self._ref_metrics()
            self._echo_payload = {'metrics': {roi: {k: (round(x, 3) if isinstance(x, float) else x)
                                                    for k, x in m.items()} for roi, m in refm.items()},
                                  'dvh': self.dvh_curves(self.reference)}
        return self._echo_payload

    def volume(self, kind: str, version):
        if kind == 'echo':
            if self.reference is None:
                return {'error': 'no ECHO reference dose for this patient'}
            return self.volumes.get_dose('echo', self.reference)
        return self.volumes.get(kind, None if version in (None, '') else int(version), self.session)

    def revision_payload(self) -> dict:
        return {'items': self.revlog.to_list(), 'current': self.revlog.current_id,
                'current_version': self.session.current_id,
                'waiting': self.revlog.waiting.id if self.revlog.waiting else None,
                'running': self.revlog.running.id if self.revlog.running else None}

    def state_payload(self) -> dict:
        return {'header': self.header_info(), 'protocol_rows': self.protocol_rows(), 'rois': self.roi_info(),
                'roi_colors': ROI_COLORS, 'roi_labels': {r: UT.roi_label(r) for r in self.session.structures},
                'echo': self.echo_payload(), 'revisions': self.revision_payload(),
                'versions': [{'id': v.id, 'ok': v.ok, 'parent': v.parent, 'label': v.label}
                             for v in self.session.history]}
