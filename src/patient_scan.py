"""Find and validate PortPy-format patient folders in a user-chosen directory.

Accepted inputs (whatever the user picks in the UI):
  * a folder that directly contains patient folders     .../data/Lung_Patient_3, ...
  * the Hugging Face download root that has data/<pid>  .../portpy_downloads
  * a single patient folder                             .../data/Lung_Patient_3

A patient folder is usable when it has the PortPy HDF5/JSON files and the
influence-matrix files for every planner beam. The ECHO RTDOSE/RTPLAN are
optional (benchmark + protocol detection). Nothing large is read: only JSON
metadata and the RTPLAN header.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REQUIRED = ('CT_Data.h5', 'CT_MetaData.json', 'StructureSet_Data.h5', 'StructureSet_MetaData.json',
            'OptimizationVoxels_Data.h5', 'OptimizationVoxels_MetaData.json', 'PlannerBeams.json')
KEY_ROIS = ('PTV', 'GTV', 'ESOPHAGUS', 'CORD', 'HEART', 'LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R')
# (fractions, total Rx Gy) -> PortPy protocol name the app knows how to optimize
PROTOCOLS = {(30, 60.0): 'Lung_2Gy_30Fx'}
SUPPORTED = set(PROTOCOLS.values())


@dataclass
class PatientInfo:
    patient_id: str
    path: str
    usable: bool
    problems: list[str] = field(default_factory=list)
    n_beams: int = 0
    beam_ids: list[int] = field(default_factory=list)
    size_gb: float = 0.0
    has_echo_dose: bool = False
    fractions: int | None = None
    rx_gy: float | None = None
    protocol: str | None = None
    rois: dict = field(default_factory=dict)      # name -> {'volume_cc', 'in_calc_box'}

    def as_dict(self):
        return asdict(self)

    def label(self) -> str:
        ptv = self.rois.get('PTV', {}).get('volume_cc')
        bits = [self.patient_id]
        if ptv:
            bits.append(f'PTV {ptv:.0f} cc')
        heart = self.rois.get('HEART')
        bits.append('heart ✓' if heart and heart['in_calc_box'] and heart['in_calc_box'] > 0.05 else 'no heart')
        if not self.usable:
            bits.append('⚠ ' + self.problems[0])
        return ' · '.join(bits)


def _rtplan_rx(path: Path):
    try:
        import pydicom
        d = pydicom.dcmread(str(path), stop_before_pixels=True)
        fx = int(d.FractionGroupSequence[0].NumberOfFractionsPlanned)
        rx = None
        for r in getattr(d, 'DoseReferenceSequence', []):
            if getattr(r, 'TargetPrescriptionDose', None) is not None:
                rx = float(r.TargetPrescriptionDose)
                break
        return fx, rx
    except Exception:  # noqa: BLE001 - header missing / unreadable -> unknown
        return None, None


def inspect_patient(folder: Path, default_protocol: str = 'Lung_2Gy_30Fx') -> PatientInfo:
    folder = Path(folder)
    info = PatientInfo(folder.name, str(folder), usable=False)
    missing = [f for f in REQUIRED if not (folder / f).is_file()]
    if missing:
        info.problems.append('missing ' + ', '.join(missing))
        return info
    try:
        info.beam_ids = [int(b) for b in json.loads((folder / 'PlannerBeams.json').read_text())['IDs']]
    except Exception as exc:  # noqa: BLE001
        info.problems.append(f'unreadable PlannerBeams.json ({exc})')
        return info
    info.n_beams = len(info.beam_ids)
    nobeam = [b for b in info.beam_ids if not (folder / 'Beams' / f'Beam_{b}_Data.h5').is_file()
              or not (folder / 'Beams' / f'Beam_{b}_MetaData.json').is_file()]
    if nobeam:
        info.problems.append(f'influence matrix missing for planner beams {nobeam}')
    try:
        for s in json.loads((folder / 'StructureSet_MetaData.json').read_text()):
            if s.get('name') in KEY_ROIS:
                info.rois[s['name']] = {'volume_cc': s.get('volume_cc'),
                                        'in_calc_box': s.get('fraction_of_vol_in_calc_box')}
    except Exception as exc:  # noqa: BLE001
        info.problems.append(f'unreadable StructureSet_MetaData.json ({exc})')
    if 'PTV' not in info.rois:
        info.problems.append('no PTV structure')
    dicom = folder / 'DicomFiles'
    info.has_echo_dose = (dicom / 'rt_dose_echo_imrt.dcm').is_file()
    if (dicom / 'rt_plan_echo_imrt.dcm').is_file():
        info.fractions, info.rx_gy = _rtplan_rx(dicom / 'rt_plan_echo_imrt.dcm')
    if info.fractions and info.rx_gy:
        info.protocol = PROTOCOLS.get((info.fractions, round(info.rx_gy, 1)))
        if info.protocol is None:
            info.problems.append(f'unsupported prescription {info.rx_gy:g} Gy / {info.fractions} fx')
    else:
        info.protocol = default_protocol        # no RTPLAN: assume the configured protocol
    info.size_gb = round(sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) / 1e9, 2)
    info.usable = not info.problems
    return info


def scan_data_dir(path: str | Path, default_protocol: str = 'Lung_2Gy_30Fx') -> tuple[Path | None, list[PatientInfo]]:
    """Return (data_dir to pass to PortPy, patients found). data_dir is the parent of the patient folders."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return None, []
    if (p / 'PlannerBeams.json').is_file() or (p / 'CT_MetaData.json').is_file():
        return p.parent, [inspect_patient(p, default_protocol)]
    for root in (p, p / 'data'):
        kids = sorted((d for d in root.iterdir() if d.is_dir() and (d / 'CT_MetaData.json').is_file()),
                      key=_natural) if root.is_dir() else []
        if kids:
            return root, [inspect_patient(d, default_protocol) for d in kids]
    return None, []


def _natural(d: Path):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d.name)]


if __name__ == '__main__':
    import sys
    root, pts = scan_data_dir(sys.argv[1] if len(sys.argv) > 1 else '.')
    print('data_dir:', root)
    for pi in pts:
        print(('OK  ' if pi.usable else 'BAD ') + pi.label(), pi.protocol, f'{pi.size_gb} GB', pi.problems or '')
