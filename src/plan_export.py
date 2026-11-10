"""Plan storage and DICOM export.

Every solved PlanVersion is stored as a light-weight optimisation result
(fluence .npy + JSON with knobs, changes, metrics) so a session can be audited
or reloaded. DICOM is written only when the physicist explicitly exports a plan.

RTDOSE export: the optimisation-grid dose is mapped onto the CT grid with
PortPy's ct_to_dose_voxel_map, cropped to the calculation box, and written in
the CT patient coordinate system (same FrameOfReference/Study as the
dataset's ECHO RTDOSE, which is used as the header template).
An RTPLAN is not written: a deliverable plan needs leaf sequencing of the
fluence maps, which BeamOpt does not do.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.plan_session import DoseConstraint, ObjectiveTerm, PlanSession, PlanVersion


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def version_record(v: PlanVersion) -> dict:
    return {
        'id': v.id, 'parent': v.parent, 'label': v.label, 'status': v.status, 'changes': v.changes,
        'solve_seconds': v.solve_seconds, 'objective_value': v.objective_value,
        'max_violation_gy': v.max_violation_gy,
        'objectives': [asdict(t) for t in v.terms.values()],
        'constraints': [asdict(c) for c in v.constraints.values()],
        'metrics': v.metrics,
    }


class PlanStore:
    """Writes runs/plans/<patient>/<session>/v<id>.json + v<id>_fluence.npy (one per solved version)
    plus session.json (working plan, undo stack, chat, request log) so the session can be restored.

    Nothing is written until the session has more than the (cached) baseline, so opening the app
    does not leave empty session folders behind.
    """

    def __init__(self, root: Path, patient_id: str, merge_x: int = 1, directory: Path | None = None):
        self.dir = Path(directory) if directory else Path(root) / patient_id / time.strftime('%Y%m%d-%H%M%S')
        self.patient_id = patient_id
        self.merge_x = merge_x
        self.saved: set[int] = set()
        if directory:      # continuing a restored session: its files are already there
            self.saved = {int(p.stem[1:]) for p in self.dir.glob('v*.json') if p.stem[1:].isdigit()}

    def sync(self, session: PlanSession, state: dict | None = None) -> list[Path]:
        if len(session.history) <= 1 and not self.saved:
            return []
        self.dir.mkdir(parents=True, exist_ok=True)
        out = []
        for v in session.history:
            if v.id in self.saved:
                continue
            rec = version_record(v) | {'merge_x': self.merge_x}
            p = self.dir / f'v{v.id}.json'
            p.write_text(json.dumps(rec, indent=1, default=_jsonable), encoding='utf-8')
            if v.fluence is not None:
                np.save(self.dir / f'v{v.id}_fluence.npy', v.fluence)
            self.saved.add(v.id)
            out.append(p)
        st = {'format': 1, 'patient_id': self.patient_id, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
              'n_versions': len(session.history), 'current_id': session.current_id,
              'checkpoints': list(session.checkpoints), 'n_beamlets': int(session.A_full.shape[1]),
              **(state or {})}
        (self.dir / 'session.json').write_text(json.dumps(st, indent=1, default=_jsonable), encoding='utf-8')
        return out


# ---------------------------------------------------------------------- restore
@dataclass
class SavedSession:
    path: Path
    patient_id: str
    stamp: str
    n_versions: int
    n_requests: int
    last_request: str
    saved_at: str
    state: dict

    def label(self) -> str:
        when = self.saved_at or self.stamp
        req = self.last_request if len(self.last_request) <= 50 else self.last_request[:49] + '…'
        return (f'{self.patient_id} · {when} · {self.n_versions} versions · {self.n_requests} requests'
                + (f' · last: “{req}”' if req else ''))


def _read_session(d: Path) -> SavedSession | None:
    vs = sorted((p for p in d.glob('v*.json') if p.stem[1:].isdigit()), key=lambda p: int(p.stem[1:]))
    if not vs:
        return None
    st = {}
    if (d / 'session.json').is_file():
        try:
            st = json.loads((d / 'session.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            st = {}
    turns = st.get('turns') or []
    if turns:
        last = turns[-1].get('request', '')
    else:   # older sessions: no session.json -> take the newest request label
        try:
            last = json.loads(vs[-1].read_text(encoding='utf-8')).get('label', '').split(' [iter')[0]
        except (OSError, ValueError):
            last = ''
        last = '' if last.startswith('baseline') else last
    n_req = len(turns) if st else len({json.loads(p.read_text(encoding='utf-8')).get('label', '').split(' [iter')[0]
                                       for p in vs[1:]})
    return SavedSession(d, st.get('patient_id') or d.parent.name, d.name, len(vs), n_req, last,
                        st.get('saved_at', ''), st)


def list_sessions(path, patient_id: str | None = None, include_empty: bool = False) -> list[SavedSession]:
    """Saved sessions under `path`: a plans root (<pid>/<stamp>/), one patient's folder, or a session folder.

    Newest first. Sessions with only the baseline are skipped unless include_empty.
    """
    p = Path(path or '').expanduser()
    if not p.is_dir():
        return []
    if (p / 'v0.json').is_file():
        dirs = [p]
    elif any(c.is_dir() and (c / 'v0.json').is_file() for c in p.iterdir()):
        dirs = [c for c in p.iterdir() if c.is_dir()]
    else:
        dirs = [s for c in p.iterdir() if c.is_dir() for s in c.iterdir() if s.is_dir()]
    out = []
    for d in dirs:
        s = _read_session(d)
        if s is None or (patient_id and s.patient_id != patient_id) or (s.n_versions <= 1 and not include_empty):
            continue
        out.append(s)
    return sorted(out, key=lambda s: s.stamp, reverse=True)


def restore_versions(session: PlanSession, directory: Path) -> dict:
    """Replace session.history with the saved versions (doses/metrics/criteria recomputed from the fluence).

    Returns the saved session state (session.json; {} for older sessions). Raises ValueError when the
    saved plans do not belong to this patient (beamlet count mismatch).
    """
    from src.criteria_evaluator import evaluate_criteria
    from src.data_adapter import from_fluence
    d = Path(directory)
    files = {int(p.stem[1:]): p for p in d.glob('v*.json') if p.stem[1:].isdigit()}
    if not files:
        raise ValueError(f'No saved plan versions in {d}')
    st = _read_session(d).state
    nb = int(session.A_full.shape[1])
    if st.get('n_beamlets') not in (None, nb):
        raise ValueError(f'Saved session has {st["n_beamlets"]} beamlets, this patient has {nb}: wrong patient?')
    history = []
    for vid in range(max(files) + 1):
        if vid not in files:      # gap (should not happen): keep ids == list index
            history.append(PlanVersion(vid, None, f'v{vid} (missing file)', [], {}, {}, 'missing'))
            continue
        r = json.loads(files[vid].read_text(encoding='utf-8'))
        terms = {t['id']: ObjectiveTerm(**t) for t in r.get('objectives', [])}
        cons = {c['id']: DoseConstraint(**c) for c in r.get('constraints', [])}
        v = PlanVersion(vid, r.get('parent'), r.get('label', f'v{vid}'), r.get('changes', []), terms, cons,
                        r.get('status', 'unknown'), r.get('objective_value'), r.get('solve_seconds'))
        v.max_violation_gy = r.get('max_violation_gy')
        flu = d / f'v{vid}_fluence.npy'
        if flu.is_file():
            x = np.load(flu)
            if x.shape != (nb,):
                raise ValueError(f'{flu.name}: {x.shape[0]} beamlets, this patient has {nb}: wrong patient?')
            v.fluence = x
            v.dose_total_gy = from_fluence(session.case, x)
            v.metrics = session.key_metrics(v.dose_total_gy)
            v.criteria = evaluate_criteria(session.case, v.dose_total_gy)
        history.append(v)
    ok = [v.id for v in history if v.ok]
    if not ok:
        raise ValueError('No successfully solved version in the saved session')
    cur = st.get('current_id')
    cur = cur if isinstance(cur, int) and cur < len(history) and history[cur].ok else ok[-1]
    session.history = history
    session.current_id = cur
    session.terms = deepcopy(history[cur].terms)
    session.constraints = deepcopy(history[cur].constraints)
    session.pending_changes = []
    session.checkpoints = [c for c in st.get('checkpoints', []) if isinstance(c, int) and c < len(history)]
    session.seed_active_set(history[cur].fluence)
    return st


def dose_on_ct_grid(case, dose_1d: np.ndarray) -> np.ndarray:
    """[z,y,x] total Gy on the CT grid (0 outside the calculation box)."""
    vm = case.influence_matrix.opt_voxels_dict['ct_to_dose_voxel_map']
    vm = np.asarray(vm[0] if isinstance(vm, list) else vm)
    out = np.zeros(vm.shape, dtype=np.float32)
    m = vm >= 0
    out[m] = np.asarray(dose_1d, dtype=np.float32)[vm[m]]
    return out


def export_rtdose(case, dose_1d: np.ndarray, out_file: Path, description: str,
                  template: Path | None = None) -> Path:
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ct = case.ct.ct_dict
    direction = np.asarray(ct['direction'], dtype=float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-4):
        raise ValueError('RTDOSE export supports axial CT orientation only')
    origin = np.asarray(ct['origin_xyz_mm'], dtype=float)
    sx, sy, sz = (float(s) for s in ct['resolution_xyz_mm'])

    dose3d = dose_on_ct_grid(case, dose_1d)
    nz_idx = np.argwhere(dose3d > 0) if np.any(dose3d > 0) else np.argwhere(np.ones_like(dose3d))
    pad = 2   # keep a zero rim so interpolating readers do not clip edge voxels
    lo = np.maximum(nz_idx.min(0) - pad, 0)
    hi = np.minimum(nz_idx.max(0) + 1 + pad, dose3d.shape)
    crop = dose3d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    template = template or (case.patient_dir / 'DicomFiles' / 'rt_dose_echo_imrt.dcm')
    ds = pydicom.dcmread(str(template))
    for tag in ('ReferencedRTPlanSequence', 'DVHSequence', 'ReferencedStructureSetSequence'):
        if tag in ds:
            delattr(ds, tag)
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = f'BeamOpt {description}'[:64]
    ds.SeriesNumber = 901
    ds.InstanceNumber = 1
    ds.Manufacturer = 'BeamOpt research prototype (PortPy) - NOT FOR CLINICAL USE'
    ds.ContentDate = time.strftime('%Y%m%d')
    ds.ContentTime = time.strftime('%H%M%S')
    ds.DoseUnits, ds.DoseType, ds.DoseSummationType = 'GY', 'PHYSICAL', 'PLAN'

    nz, ny, nx = crop.shape
    ds.ImagePositionPatient = [float(origin[0] + lo[2] * sx), float(origin[1] + lo[1] * sy),
                               float(origin[2] + lo[0] * sz)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [sy, sx]            # [row spacing, column spacing]
    ds.SliceThickness = sz
    ds.GridFrameOffsetVector = [float(k * sz) for k in range(nz)]
    ds.NumberOfFrames = nz
    ds.Rows, ds.Columns = ny, nx
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 32
    ds.HighBit, ds.PixelRepresentation = 31, 0
    scale = max(float(crop.max()), 1e-6) / (2 ** 31)   # headroom: float rounding must not wrap uint32
    ds.DoseGridScaling = f'{scale:.10g}'
    scale = float(ds.DoseGridScaling)
    ds.PixelData = np.clip(np.round(crop.astype(np.float64) / scale), 0, 2 ** 32 - 1).astype(np.uint32).tobytes()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_file), enforce_file_format=True)
    return out_file


def export_version(session: PlanSession, case, version_id: int, out_dir: Path) -> list[Path]:
    """RTDOSE + JSON summary + fluence for one plan version."""
    v = session.get_version(version_id)
    if not v.ok:
        raise ValueError(f'Version {version_id} has no valid dose ({v.status})')
    out_dir = Path(out_dir)
    stem = f'{Path(case.patient_dir).name}_v{v.id}'
    files = [export_rtdose(case, v.dose_total_gy, out_dir / f'{stem}_RTDOSE.dcm', f'v{v.id} {v.label}')]
    summary = out_dir / f'{stem}_plan.json'
    summary.write_text(json.dumps(version_record(v), indent=1, default=_jsonable), encoding='utf-8')
    files.append(summary)
    if v.fluence is not None:
        flu = out_dir / f'{stem}_fluence.npy'
        np.save(flu, v.fluence)
        files.append(flu)
    return files
