"""M1: isolate PortPy-specific code in one small, inspectable adapter."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .metrics import ROIDose, summarize_roi


@dataclass
class PatientCase:
    data: object
    ct: object
    structures: object
    beams: object
    influence_matrix: object
    criteria: object
    plan: object
    opt_params: dict
    patient_dir: Path


def load_patient(data_dir: str, patient_id='Lung_Patient_3', protocol='Lung_2Gy_30Fx',
                 beamlet_mm: tuple[float, float] | None = None) -> PatientCase:
    """Load ONE local PortPy patient. Does not download anything automatically.

    data_dir must contain subfolder Lung_Patient_3 (not the Hugging Face
    outer directory); see README for the quicklook notebook directory layout.

    beamlet_mm: optional (width, height) to merge beamlets via PortPy's own
    down-sampling (e.g. (5, 5) from native 2.5 mm). The optimization VOXEL grid
    is unchanged, so all dose metrics stay directly comparable.
    """
    import portpy.photon as pp
    root = Path(data_dir).expanduser().resolve()
    patient_dir = root / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f'{patient_dir} not found. Set data_dir to the parent of patient folder.')
    data = pp.DataExplorer(data_dir=str(root))
    data.patient_id = patient_id
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)  # requires downloaded planner beam metadata AND influence files
    criteria = pp.ClinicalCriteria(data, protocol_name=protocol)
    opt_params = data.load_config_opt_params(protocol_name=protocol)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=criteria)
    if beamlet_mm:
        inf = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams,
                                 beamlet_width_mm=float(beamlet_mm[0]), beamlet_height_mm=float(beamlet_mm[1]))
    else:
        inf = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)
    plan = pp.Plan(ct=ct, structs=structs, beams=beams,
                   inf_matrix=inf, clinical_criteria=criteria)
    print('Loaded:', patient_id, '| voxels x beamlets:', inf.A.shape)
    print('Fractions:', plan.get_num_of_fractions(), '| Rx total Gy:', plan.get_prescription())
    return PatientCase(data, ct, structs, beams, inf, criteria, plan, opt_params, patient_dir)


def load_reference_dose(case: PatientCase, technique='imrt') -> np.ndarray:
    """Resample ECHO reference RTDOSE to PortPy CT grid, then optimization grid.

    Returns [n_opt_voxels], total-course dose in Gy (never per-fraction).
    """
    import portpy.photon as pp
    path = case.patient_dir / 'DicomFiles' / f'rt_dose_echo_{technique}.dcm'
    if not path.is_file():
        raise FileNotFoundError(f'Missing reference RTDOSE: {path}')
    converted = np.asarray(pp.convert_dose_rt_dicom_to_portpy(
        my_plan=case.plan, dose_file_name=str(path)))
    expected_n = case.influence_matrix.A.shape[0]
    if converted.ndim == 3:
        dose_1d = case.influence_matrix.dose_3d_to_1d(converted)
    elif converted.ndim == 1 and converted.size == expected_n:
        dose_1d = converted
    else:
        raise ValueError(f'Unexpected converted reference dose shape {converted.shape}; check PortPy version')
    dose_1d = np.asarray(dose_1d, dtype=float).reshape(-1)
    if dose_1d.size != expected_n or not np.all(np.isfinite(dose_1d)):
        raise ValueError('Invalid reference dose array')
    if dose_1d.min() < 0:
        raise ValueError('Negative dose in reference data')
    return dose_1d


def from_fluence(case: PatientCase, intensity_per_fraction: np.ndarray) -> np.ndarray:
    """Total Gy = fractions * (A @ beamlet intensities). No arbitrary reshape."""
    x = np.asarray(intensity_per_fraction, dtype=float).reshape(-1)
    if len(x) != case.influence_matrix.A.shape[1] or np.any(x < -1e-7):
        raise ValueError('Fluence has wrong dimensions or negative values')
    return np.asarray(case.influence_matrix.A @ x, dtype=float).reshape(-1) * case.plan.get_num_of_fractions()


def get_roi_dose(case: PatientCase, dose_total_gy: np.ndarray, roi_name: str) -> ROIDose:
    """Map ROI to dose rows and physical voxel volumes; reject missing ROIs."""
    inf = case.influence_matrix
    if roi_name not in case.structures.get_structures():
        raise KeyError(f'Unknown ROI: {roi_name}')
    idx = np.asarray(inf.get_opt_voxels_idx(roi_name), dtype=int)
    if idx.size == 0:
        raise ValueError(f'No optimization voxels for {roi_name}')
    vols = np.asarray(inf.get_opt_voxels_volume_cc(roi_name), dtype=float)
    fraction = inf.get_fraction_of_vol_in_calc_box(roi_name)
    if fraction is None:
        raise ValueError(f'Missing calculation-box coverage fraction for ROI: {roi_name}')
    d = np.asarray(dose_total_gy, dtype=float).reshape(-1)
    if d.size != inf.A.shape[0] or idx.min() < 0 or idx.max() >= len(d):
        raise ValueError('Incompatible dose vector and ROI voxel map')
    return ROIDose(d[idx], vols, float(fraction))


def evaluate_dose(case: PatientCase, dose_total_gy, roi_names) -> dict:
    """Return transparent ROI metrics; partial ROIs produce no valid metrics."""
    rows = {}
    for name in roi_names:
        try:
            rows[name] = summarize_roi(get_roi_dose(case, dose_total_gy, name), strict=False)
        except (KeyError, ValueError) as exc:
            rows[name] = {'status':'unavailable', 'reason':str(exc)}
    return rows


def official_criteria_table(case: PatientCase, dose_total_gy):
    """Return PortPy's protocol scorecard; review its limits/goals before use.

    This table is for comparison/inspection; it is not a blanket certificate
    of plan safety or machine deliverability.
    """
    import portpy.photon as pp
    return pp.Evaluation.display_clinical_criteria(
        my_plan=case.plan, dose_1d=np.asarray(dose_total_gy),
        clinical_criteria=case.criteria, return_df=True)


def plot_dvh(case: PatientCase, dose_total_gy, roi_names, ax=None):
    """Volume-weighted cumulative DVH with explicit dose unit and coverage caveat."""
    import matplotlib.pyplot as plt
    from .metrics import cumulative_dvh
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = np.asarray(dose_total_gy)
    bins = np.linspace(0, max(1.0, float(np.max(d)) * 1.02), 251)
    for name in roi_names:
        try:
            roi = get_roi_dose(case, d, name)
        except (KeyError, ValueError) as exc:
            print(f'Skipping {name}: {exc}')
            continue
        xs, ys = cumulative_dvh(roi, bins)
        label = f'{name} (grid={roi.calculation_coverage:.1%})'
        ax.plot(xs, ys, label=label)
    ax.set(xlabel='Total dose (Gy)', ylabel='Represented ROI volume receiving >= dose (%)',
           title='Research DVH on PortPy optimization grid', ylim=(0, 105))
    ax.legend()
    ax.grid(alpha=0.25)
    return ax
