"""M2: deliberately small, bounded *research* IMRT fluence optimization study.

This is NOT a treatment planning system; clinical and delivery verification
are required before considering any clinical application.
"""
from dataclasses import dataclass
import numpy as np
from .data_adapter import PatientCase, from_fluence, evaluate_dose


@dataclass
class Trial:
    esophagus_weight: float
    solver_status: str
    objective_value: float | None
    dose_total_gy: np.ndarray | None
    metrics: dict | None


def solve_trial(case: PatientCase, esophagus_weight: float, *,
                target_roi='ESOPHAGUS', solver='SCS', roi_names=None) -> Trial:
    """Solve baseline PortPy objective + lambda * esophagus mean dose per Fx.

    Setting lambda=0 reproduces the *PortPy research baseline* optimization.
    The extra term is a convex surrogate objective; it is not a new hard constraint.
    We recreate the PortPy Optimization object for every trial to avoid
    accidentally accumulating constraints/objectives across runs.
    """
    import cvxpy as cp
    import portpy.photon as pp
    from copy import deepcopy

    if not np.isfinite(esophagus_weight) or not 0 <= esophagus_weight <= 100:
        raise ValueError('Choose a finite research weight in [0,100]')
    if target_roi not in case.structures.get_structures():
        raise KeyError(f'Missing target ROI {target_roi}; list available ROIs first.')
    inf = case.influence_matrix
    roi_indices = np.asarray(inf.get_opt_voxels_idx(target_roi), dtype=int)
    if roi_indices.size == 0:
        raise ValueError(f'No optimization voxels for {target_roi}')

    opt = pp.Optimization(case.plan, opt_params=deepcopy(case.opt_params),
                          clinical_criteria=case.criteria)
    opt.create_cvxpy_problem()
    # For PortPy, A @ x is dose PER fraction, while evaluation uses total Gy.
    # This objective minimizes average ROI dose at the optimizer grid.
    if esophagus_weight > 0:
        roi_per_fraction = inf.A[roi_indices, :] @ opt.vars['x']
        opt.add_objective(float(esophagus_weight) * cp.sum(roi_per_fraction) / len(roi_indices))
    solution, problem = opt.solve(return_cvxpy_prob=True, solver=solver, verbose=False)
    status = str(problem.status)
    if status not in ('optimal', 'optimal_inaccurate') or solution['optimal_intensity'] is None:
        return Trial(float(esophagus_weight), status, None, None, None)
    dose = from_fluence(case, solution['optimal_intensity'])
    if roi_names is None:
        roi_names = ['PTV', target_roi, 'CORD', 'HEART', 'LUNGS_NOT_GTV']
    metrics = evaluate_dose(case, dose, roi_names)
    obj_value = float(problem.value) if problem.value is not None else None
    return Trial(float(esophagus_weight), status, obj_value, dose, metrics)


def compare_trials(trials: list[Trial], *, preserve_ptv_d95_within_gy=0.5):
    """Educational candidate screening; does NOT certify hard protocol limits.

    The first trial MUST be weight=0 and becomes our model-consistent baseline.
    'screen_pass' only means dose objective improves and PTV D95 stays close
    to baseline. It is NOT a clinical acceptance criterion.
    """
    if not trials or trials[0].esophagus_weight != 0 or trials[0].metrics is None:
        raise ValueError('First trial must be successful baseline with weight 0')
    baseline = trials[0].metrics
    if baseline.get('PTV', {}).get('status') != 'complete' or baseline.get('ESOPHAGUS', {}).get('status') != 'complete':
        raise ValueError('Cannot compare: PTV or ESOPHAGUS missing/incompletely covered')
    base_ptv = baseline['PTV']['D95_Gy']
    base_eso = baseline['ESOPHAGUS']['Dmean_Gy']
    rows = []
    for trial in trials:
        row = {'weight': trial.esophagus_weight, 'solver_status': trial.solver_status}
        if trial.metrics is None:
            row['screen_pass'] = False
            rows.append(row)
            continue
        ptv = trial.metrics.get('PTV', {})
        eso = trial.metrics.get('ESOPHAGUS', {})
        if ptv.get('status') != 'complete' or eso.get('status') != 'complete':
            row['screen_pass'] = False
            row['note'] = 'missing or partial ROI'
            rows.append(row)
            continue
        ptv_d95 = ptv['D95_Gy']
        eso_dmean = eso['Dmean_Gy']
        row.update({'PTV_D95_Gy':ptv_d95, 'ESOPHAGUS_Dmean_Gy':eso_dmean,
                    'delta_PTV_D95_Gy':ptv_d95-base_ptv,
                    'delta_ESOPHAGUS_Dmean_Gy':eso_dmean-base_eso,
                    'screen_pass': (ptv_d95>=base_ptv-preserve_ptv_d95_within_gy
                                    and eso_dmean < base_eso-1e-6),
                    'note':'Exploratory screen only: does not test all clinical criteria'})
        rows.append(row)
    return rows
