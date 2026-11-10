"""Per-patient workspace behind the web app (app/server.py).

Holds the loaded case, the PlanSession with its versions, the agent graph, caches (scorecard, deliverability,
slice-viewer volumes) and saving / restoring of sessions. No UI framework imports here.
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from app import ui_text as UT
from app.viz import SliceViewer
from app.revisions import RevisionLog
from app.web_payloads import WebPayloads
from app.volume_viewer import VolumeData
from src.agent.graph import build_graph
from src.agent.llm import get_llm
from src.config import load_config
from src.data_adapter import load_patient, load_reference_dose
from src import deliverability as DLV
from src.plan_export import PlanStore, restore_versions
from src.plan_session import PlanSession
from src.scorecard import Scorecard
from src.session_report import build_report

ROOT = Path(__file__).resolve().parents[1]

CFG = load_config()
UI = CFG.get('ui', {})
DVH_ROIS = ['PTV', 'ESOPHAGUS', 'CORD', 'LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R']
DVH_VISIBLE = ['PTV', 'ESOPHAGUS', 'CORD', 'LUNGS_NOT_GTV']
SLICE_ROIS = ['PTV', 'GTV', 'ESOPHAGUS', 'CORD', 'HEART', 'LUNG_L', 'LUNG_R']
TABLE_METRICS = {'PTV': ['D95', 'D98', 'Dmax', 'V60'], 'ESOPHAGUS': ['Dmean', 'D2', 'Dmax', 'V60'],
                 'CORD': ['Dmax', 'D2', 'Dmean'], 'LUNGS_NOT_GTV': ['Dmean', 'V20', 'V30'],
                 'LUNG_L': ['Dmean', 'V20'], 'LUNG_R': ['Dmean', 'V20']}
NODE_TITLES = {'understand': 'Understood request', 'propose': 'Proposed changes', 'apply': 'Validation',
               'optimize': 'Optimizer', 'finalize': 'Selection', 'restore': 'Restore'}
WORKING = -1          # "viewing" sentinel: follow the working plan
NONE = -1             # "compare" sentinel: no comparison
AUDIO_EXT = ('.wav', '.mp3', '.webm', '.ogg', '.m4a', '.flac')


class Workspace(WebPayloads):
    """Everything for ONE loaded patient (UI-independent). Switching patients builds a new one."""

    def __init__(self, data_dir: str | None = None, patient_id: str | None = None, protocol: str | None = None,
                 restore_dir: str | None = None):
        self.cfg = copy.deepcopy(CFG)
        d = self.cfg['data']
        if data_dir:
            d['data_dir'] = str(Path(data_dir).resolve())
        if patient_id:
            d['patient_id'] = patient_id
        if protocol:
            d['protocol'] = protocol
        self.pid, self.protocol = d['patient_id'], d['protocol']
        self.case = load_patient(d['data_dir'], d['patient_id'], d['protocol'])
        o = self.cfg['optimization']
        self.session = PlanSession(self.case, solver=o['solver'], solver_opts=o.get('solver_opts', {}), log=True,
                                   merge_x=int(o.get('merge_x', 1)))
        self.session.configure_gpu(o.get('gpu'))
        self.store = PlanStore(ROOT / UI.get('plans_dir', 'runs/plans'), d['patient_id'], int(o.get('merge_x', 1)))
        self._baseline()
        try:
            self.reference = load_reference_dose(self.case)
        except (FileNotFoundError, ValueError) as exc:
            print('No ECHO reference dose:', exc)
            self.reference = None
        vmax = max(66.0, float(self.session.current.dose_total_gy.max()),
                   float(self.reference.max()) if self.reference is not None else 0.0)
        self.viewer = SliceViewer(self.case, dose_vmax=vmax)
        self.volumes = VolumeData(self.case, self.viewer, SLICE_ROIS)
        self.model = CFG['llm']['model']
        self.graph = build_graph(self.session, get_llm(self.cfg), self.cfg)
        self.has_hints = (ROOT / 'runs' / 'cache' / f'hints_{self.pid}.txt').exists()
        self.turns: list[dict] = []                         # structured per-request log for the session report
        self.history: list[str] = []
        self.chat_log: list[dict] = []                      # restored when the page is reloaded
        self.results: dict[int, str] = {0: 'baseline'}     # version -> request it was the accepted result of
        self.lock = threading.Lock()
        self.cancel = threading.Event()                     # set by the Stop button
        self.scorecard = Scorecard(self.case, self.protocol)
        self._scores: dict = {}                             # version id / 'echo' -> ScoreResult
        self._deliv: dict = {}                              # version id -> Deliverability
        self.restored_from = None
        self.revlog = RevisionLog()                         # web app: one card per request (Keep / Discard / Branch)
        self.generation = 0
        if restore_dir:
            self.restore(restore_dir)
        # pre-pack the slice-viewer volumes so the first open of the Dose slice tab is fast
        threading.Thread(target=lambda: [self.volumes.get(k, v, self.session) for k, v in
                                         (('meta', None), ('ct', None), ('masks', None), ('dose', 0))],
                         daemon=True).start()

    def _baseline(self):
        cache = ROOT / 'runs' / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"baseline_{self.pid}_{self.session.knobs_fingerprint()}.npy"
        if f.exists():
            self.session.add_version_from_fluence(np.load(f), 'baseline (protocol defaults)')
            print('Loaded cached baseline', f.name)
        else:
            print('Solving baseline (first run only)...')
            v = self.session.solve('baseline (protocol defaults)')
            if not v.ok:
                raise RuntimeError(f'Baseline failed: {v.status}')
            np.save(f, v.fluence)

    def set_model(self, model):
        if model and model != self.model:
            with self.lock:
                self.model = model
                self.graph = build_graph(self.session, get_llm(self.cfg, model=model), self.cfg)

    # ------------------------------------------------------------ versions
    def resolve(self, view_id, compare_id):
        s = self.session
        vid = s.current_id if view_id in (None, WORKING) else int(view_id)
        view = s.get_version(vid)
        cmp = None
        if compare_id not in (None, NONE) and int(compare_id) != view.id:
            c = s.get_version(int(compare_id))
            cmp = c if c.ok else None
        return view, cmp

    # ------------------------------------------------------------ views
    def metrics_table(self, view, cmp):
        rows = []
        refm = self._ref_metrics()
        for roi, keys in TABLE_METRICS.items():
            m = view.metrics.get(roi, {})
            if m.get('status') == 'unavailable':
                continue
            for k in keys:
                if k not in m:
                    continue
                row = {'Structure': UT.roi_label(roi), 'Metric': UT.metric_label(k), f'v{view.id} (viewed)': round(m[k], 2)}
                if cmp is not None:
                    prev = cmp.metrics[roi][k]
                    row[f'v{cmp.id} (compared)'] = round(prev, 2)
                    row['Δ'] = round(m[k] - prev, 2)
                if refm is not None:
                    row['ECHO'] = round(refm[roi][k], 2)
                row['In dose box'] = '100%' if m['scope'] == 'whole_roi' else \
                    m['scope'].split('(')[-1].split(' of')[0]
                rows.append(row)
        return pd.DataFrame(rows)

    def _ref_metrics(self):
        if self.reference is None:
            return None
        if not hasattr(self, '_refm'):
            self._refm = self.session.key_metrics(self.reference)
        return self._refm

    def criteria_view(self, view):
        """(markdown note, table of checkable criteria). Not-checkable rows are summarised, not listed."""
        df = view.criteria
        if df is None or not len(df):
            return 'No protocol criteria.', pd.DataFrame()
        und = df[df['status'] == 'UNDETERMINED']
        ok = df[df['status'] != 'UNDETERMINED']

        def num(v):
            return None if v is None or pd.isna(v) else round(float(v), 2)
        out = pd.DataFrame({
            'Status': ok['status'].map(lambda s: UT.STATUS_TEXT.get(s, s)),
            'Structure': ok['roi'].map(UT.roi_label),
            'Criterion': ok['type'].map(lambda t: UT.CRIT_TYPE.get(t, t)) + ' ' + ok['metric'].fillna('').astype(str),
            'Value': ok['actual'].map(num), 'Goal': ok['goal'].map(num) if 'goal' in ok else None,
            'Limit': ok['limit'].map(num) if 'limit' in ok else None})
        md = f'**{len(ok)} of {len(df)} protocol criteria could be checked** on this plan.'
        if len(und):
            why = {}
            for _, r in und.iterrows():
                reason = str(r.get('reason') or '')
                cov = r.get('coverage')
                key = ('not contoured / no voxels in the dose box' if 'Missing' in reason or 'missing' in reason
                       else (f'only {float(cov):.0%} inside the dose box' if cov is not None and not pd.isna(cov)
                             else 'only partly inside the dose box'))
                why.setdefault(UT.roi_label(r['roi']), key)
            md += (f' {len(und)} could not, because the organ is ' + '; '.join(
                f'{k} ({v})' for k, v in why.items()) + '. Their metrics for the part inside the box are in the '
                'key metrics above.')
        return md, out.reset_index(drop=True)

    def knobs_df(self, view):
        rows = [{'Structure': UT.roi_label(t.roi) if t.roi else 'all beams', 'Objective': UT.knob_text(t),
                 'Weight': t.weight, 'Added by': t.source, 'id': t.id}
                for t in view.terms.values() if t.weight > 0 or t.source == 'agent']
        rows += [{'Structure': UT.roi_label(c.roi), 'Objective': f'Hard limit: {c.kind} dose ≤ {c.limit_gy:g} Gy',
                  'Weight': None, 'Added by': c.source, 'id': c.id} for c in view.constraints.values()]
        return pd.DataFrame(rows)

    @property
    def key(self) -> str:
        return f'{self.pid}@{id(self)}.{self.generation}'     # new App / restore -> viewer reloads everything

    def report_html(self) -> str:
        return build_report(self.session, self.turns, self.pid, self.protocol, self.model, self._ref_metrics())

    # ------------------------------------------------------------ save / restore
    def state(self) -> dict:
        """Everything besides the plan versions that a restored session needs (written to session.json)."""
        return {'data_dir': self.cfg['data']['data_dir'], 'protocol': self.protocol, 'model': self.model,
                'results': {str(k): v for k, v in self.results.items()}, 'turns': self.turns,
                'history': self.history, 'chat_log': self.chat_log, 'revisions': self.revlog.state()}

    def sync(self):
        self.store.sync(self.session, self.state())

    def restore(self, directory):
        st = restore_versions(self.session, Path(directory))
        o = self.cfg['optimization']
        self.store = PlanStore(ROOT / UI.get('plans_dir', 'runs/plans'), self.pid, int(o.get('merge_x', 1)),
                               directory=Path(directory))           # new versions continue in the same folder
        res = {int(k): v for k, v in (st.get('results') or {}).items() if str(k).isdigit()}
        self.results = res or {0: 'baseline'}
        if not st:   # older session without session.json: the last good version of each request is its result
            last = {}
            for v in self.session.history[1:]:
                if v.ok:
                    last[v.label.split(' [iter')[0]] = v.id
            self.results.update({vid: req for req, vid in last.items()})
        self.turns = st.get('turns') or []
        self.history = st.get('history') or []
        self.chat_log = st.get('chat_log') or []
        self.revlog = (RevisionLog.from_state(st['revisions']) if st.get('revisions')
                       else RevisionLog.from_turns(self.turns, self.session.current_id, self.results))
        self.chat_log.append({'role': 'assistant', 'content': (
            f'Restored saved session `{Path(directory).name}`: {len(self.session.history)} plan versions, '
            f'current plan v{self.session.current_id}. Doses and metrics were recomputed from the saved fluence.')})
        self._scores, self._deliv = {}, {}
        with self.volumes._lock:
            self.volumes._cache.clear()                   # cached dose volumes are keyed by version id
        self.generation += 1                              # -> new viewer key: the browser drops its caches
        self.restored_from = str(directory)

    # ------------------------------------------------------------ scorecard
    def score(self, key):
        if key not in self._scores:
            dose = self.reference if key == 'echo' else self.session.get_version(key).dose_total_gy
            self._scores[key] = self.scorecard.score(dose)
        return self._scores[key]

    def score_norm(self, key):
        """Score after scaling the dose so that PTV D95 = Rx (V100%Rx = 95 %), as clinics normalise plans."""
        nk = ('norm', key)
        if nk not in self._scores:
            dose = self.reference if key == 'echo' else self.session.get_version(key).dose_total_gy
            m = self._ref_metrics() if key == 'echo' else self.session.get_version(key).metrics
            d95 = m.get('PTV', {}).get('D95')
            f = self.session.rx_gy / d95 if d95 else 1.0
            self._scores[nk] = (self.scorecard.score(dose * f), f)
        return self._scores[nk]

    def scorecard_view(self, view, cmp):
        if not self.scorecard.available:
            return f'No plan-quality scorecard is configured for protocol {self.protocol}.', pd.DataFrame()
        r = self.score(view.id)
        cols = {('viewed', f'v{view.id}'): r}
        if cmp is not None:
            cols[('compare', f'v{cmp.id}')] = self.score(cmp.id)
        if view.id != 0 and (cmp is None or cmp.id != 0):
            cols[('baseline', 'v0')] = self.score(0)
        if self.reference is not None:
            cols[('ECHO', 'ECHO')] = self.score('echo')
        df = r.items[['structure', 'metric']].copy()
        df['structure'] = df['structure'].map(UT.roi_label)
        for (_, name), res in cols.items():
            df[f'{name} value'] = res.items['value']
            df[f'{name} pts'] = res.items['score']
        df['max pts'] = r.items['max']
        df['note'] = r.items['note']
        df = df[r.items['score'].notna() | (df.filter(like=' pts').notna().any(axis=1))]
        missing = r.items[r.items['score'].isna()]

        def line(role, name, res, key):
            nr, f = self.score_norm(key)
            return (f'**{role} {name}: {res.total:.1f} / {res.achievable:g} pts ({res.percent:.0f}%)** · normalised '
                    f'to PTV D95 = {self.session.rx_gy:g} Gy (dose ×{f:.3f}): {nr.total:.1f} pts')
        keys = {'viewed': view.id, 'compare': cmp.id if cmp else None, 'baseline': 0, 'ECHO': 'echo'}
        md = [f'**{self.scorecard.name}** plan-quality scorecard (PortPy, higher = better).']
        md += ['- ' + line(role, name if role != 'ECHO' else 'benchmark', res, keys[role])
               for (role, name), res in cols.items()]
        if len(missing):
            md.append(f'\n{len(missing)} items ({missing["max"].sum():g} pts) are not scored because the structure '
                      'does not exist for this patient: '
                      + ', '.join(sorted({UT.roi_label(x) for x in missing['structure']})) + '.')
        md.append('Organs only partly inside the dose box are scored as whole organs with 0 Gy outside the box '
                  '(PortPy convention, optimistic). Unnormalised plans can lose PTV V60 points that a simple '
                  'renormalisation would recover — compare the normalised totals for that.')
        return '\n'.join(md), df.reset_index(drop=True)

    # ------------------------------------------------------------ deliverability
    def deliver(self, vid):
        if vid not in self._deliv:
            v = self.session.get_version(vid)
            self._deliv[vid] = DLV.assess(self.session, v.fluence, v.metrics)
        return self._deliv[vid]

    def deliverability_view(self, view, cmp):
        d, base = self.deliver(view.id), self.deliver(0)
        c = self.deliver(cmp.id) if cmp is not None else None
        beams = d.beams.copy()
        ref = c if c is not None else base
        refname = f'v{cmp.id}' if c is not None else 'baseline'
        beams[f'segments {refname}'] = ref.beams['segments']
        beams[f'MU {refname}'] = ref.beams['MU (rel.)']
        beams['MU (rel.)'] = beams['MU (rel.)'].round(3)
        beams[f'MU {refname}'] = beams[f'MU {refname}'].round(3)

        def pct(a, b):
            return f'{a / b:.0%}' if b else 'n/a'
        fl = DLV.flags(d, base)
        md = [f'**v{view.id}: {d.total_segments} segments, relative MU {d.total_mu:.2f} '
              f'({pct(d.total_mu, base.total_mu)} of baseline), mean modulation {d.mean_modulation:.2f}** '
              f'(baseline: {base.total_segments} segments, modulation {base.mean_modulation:.2f}).',
              f'Leaf sequencing (step-and-shoot, {DLV.LEVELS} intensity levels) changes the dose by at most '
              f'{d.max_voxel_diff_gy:.2f} Gy in any voxel; key metrics below.',
              ('**Flags:** ' + ' · '.join(fl)) if fl else 'No deliverability flags (MU and segments within 30 % of the '
              'baseline, sequencing moves no key metric by more than 0.5 Gy).',
              '\n<span class="hint">Screening only: MU are in fluence units (compare plans of the same patient), '
              'not calibrated machine MU; no leaf-speed / min-MU / transmission modelling.</span>']
        fig = None                                   # fluence maps are drawn by the web app (js/drawers.js)
        fid = d.fidelity.copy()
        if len(fid):
            fid['roi'] = fid['roi'].map(UT.roi_label)
            fid = fid.rename(columns={'roi': 'Structure', 'metric': 'Metric', 'unit': 'Unit', 'optimized': 'Optimized',
                                      'after sequencing': 'After sequencing', 'delta': 'Δ'})
        beams = beams.rename(columns={'gantry': 'Gantry (°)', 'segments': 'Segments', 'modulation': 'Modulation'})
        return '\n\n'.join(md), beams, fid, fig

