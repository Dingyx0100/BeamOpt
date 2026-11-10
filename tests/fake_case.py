"""A tiny synthetic PortPy-like case so PlanSession/agent logic is testable without patient data.

Geometry: 1-D row of 60 voxels, 12 beamlets with Gaussian-ish dose kernels.
PTV = voxels 25..34, ESOPHAGUS = 36..41 (adjacent, competes with PTV),
CORD = 10..15, HEART has zero optimization voxels (outside calc box).
"""
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from src.data_adapter import PatientCase

N_VOX, N_BEAMLETS = 60, 12
ROIS = {
    'PTV': np.arange(25, 35), 'ESOPHAGUS': np.arange(36, 42), 'CORD': np.arange(10, 16),
    'HEART': np.array([], dtype=int), 'BODY': np.arange(0, 60),
}


def _A():
    centers = np.linspace(5, 55, N_BEAMLETS)
    v = np.arange(N_VOX)[:, None]
    A = np.exp(-0.5 * ((v - centers[None, :]) / 4.0) ** 2) * 0.1
    A[A < 1e-4] = 0
    return sp.csr_matrix(A.astype(np.float32))


class _Inf:
    def __init__(self):
        self.A = _A()

    def get_opt_voxels_idx(self, name):
        return ROIS[name]

    def get_opt_voxels_volume_cc(self, name):
        return np.ones(len(ROIS[name]))

    def get_fraction_of_vol_in_calc_box(self, name):
        return {'HEART': 0.0, 'CORD': 0.5}.get(name, 1.0)


class _Criteria:
    def __init__(self, crit):
        self.clinical_criteria_dict = {'criteria': crit}

    def get_criteria(self):
        return self.clinical_criteria_dict['criteria']

    def check_criterion_exists(self, c, return_ind=False):
        for i, k in enumerate(self.get_criteria()):
            if k['type'] == c['type'] and k['parameters']['structure_name'] == c['parameters']['structure_name']:
                return (True, i) if return_ind else True
        return (False, None) if return_ind else False


def make_case():
    criteria = [
        {'type': 'max_dose', 'parameters': {'structure_name': 'PTV'}, 'constraints': {'limit_dose_gy': 69}},
        {'type': 'max_dose', 'parameters': {'structure_name': 'CORD'}, 'constraints': {'limit_dose_gy': 50}},
        {'type': 'mean_dose', 'parameters': {'structure_name': 'ESOPHAGUS'}, 'constraints': {'limit_dose_gy': 34}},
        {'type': 'max_dose', 'parameters': {'structure_name': 'ESOPHAGUS'}, 'constraints': {'limit_dose_gy': 66}},
    ]
    opt_params = {'objective_functions': [
        {'type': 'quadratic-overdose', 'structure_name': 'PTV', 'weight': 10000, 'dose_gy': 'prescription_gy'},
        {'type': 'quadratic-underdose', 'structure_name': 'PTV', 'weight': 100000, 'dose_gy': 'prescription_gy'},
        {'type': 'quadratic', 'structure_name': 'ESOPHAGUS', 'weight': 20},
        {'type': 'quadratic', 'structure_name': 'CORD', 'weight': 10},
        {'type': 'quadratic', 'structure_name': 'HEART', 'weight': 20},
        {'type': 'linear-overdose', 'structure_name': 'CORD', 'weight': 100, 'dose_gy': 50},
    ], 'constraints': []}
    structs = SimpleNamespace(get_structures=lambda: list(ROIS), get_volume_cc=lambda n: float(max(len(ROIS[n]), 1)))
    plan = SimpleNamespace(get_num_of_fractions=lambda: 30, get_prescription=lambda: 60.0)
    return PatientCase(data=None, ct=None, structures=structs, beams=None, influence_matrix=_Inf(),
                       criteria=_Criteria(criteria), plan=plan, opt_params=opt_params, patient_dir=None)
