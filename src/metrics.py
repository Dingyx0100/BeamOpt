"""M1: transparent, testable dose metrics on a *total-course Gy* dose vector.

All functions are independent of PortPy. They deliberately report dose coverage:
a partial ROI in the calculation box must never silently pass evaluation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ROIDose:
    """Values sampled at the ROI's *optimization* voxels, not a CT slice.

    dose_gy: full-course Gy, shape [N]
    volumes_cc: volume represented by each optimization voxel, shape [N]
    calculation_coverage: fraction of the whole ROI represented by these voxels
    """
    dose_gy: np.ndarray
    volumes_cc: np.ndarray
    calculation_coverage: float

    def __post_init__(self):
        d = np.asarray(self.dose_gy, dtype=float).reshape(-1)
        w = np.asarray(self.volumes_cc, dtype=float).reshape(-1)
        if d.size == 0 or d.shape != w.shape:
            raise ValueError('ROI must contain equally sized, nonempty dose and volume arrays')
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError('ROI dose contains invalid values')
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError('ROI voxel volumes must be positive and finite')
        if not 0 < self.calculation_coverage <= 1.000001:
            raise ValueError('ROI calculation coverage must be in (0, 1]')
        object.__setattr__(self, 'dose_gy', d)
        object.__setattr__(self, 'volumes_cc', w)


def weighted_mean_gy(roi: ROIDose) -> float:
    """Dmean = sum(dose_i * volume_i)/sum(volume_i), in Gy."""
    return float(np.average(roi.dose_gy, weights=roi.volumes_cc))


def dose_at_volume_gy(roi: ROIDose, volume_percent: float) -> float:
    """D_v: dose received by at least v% ROI volume (step-function DVH).

    This is a conservative discrete definition without interpolation; some
    TPS/PortPy interpolated DVH values may differ by a small amount.
    """
    if not 0 < volume_percent <= 100:
        raise ValueError('volume_percent must be in (0,100]')
    order = np.argsort(-roi.dose_gy)
    sorted_doses = roi.dose_gy[order]
    sorted_weights = roi.volumes_cc[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum() * 100
    index = int(np.searchsorted(cumulative, volume_percent, side='left'))
    return float(sorted_doses[min(index, len(sorted_doses)-1)])


def volume_at_dose_percent(roi: ROIDose, threshold_gy: float) -> float:
    """V_x: volume (%) receiving at least x Gy."""
    if threshold_gy < 0:
        raise ValueError('Dose threshold must be nonnegative')
    hit = roi.dose_gy >= threshold_gy
    return float(100 * roi.volumes_cc[hit].sum() / roi.volumes_cc.sum())


def cumulative_dvh(roi: ROIDose, bins_gy: np.ndarray):
    """Return (dose_bins_Gy, volume_percent_at_or_above_each_bin)."""
    bins = np.asarray(bins_gy, dtype=float).reshape(-1)
    if np.any(bins < 0) or not np.all(np.isfinite(bins)):
        raise ValueError('Invalid DVH dose bins')
    order = np.argsort(roi.dose_gy)
    sorted_doses, sorted_weights = roi.dose_gy[order], roi.volumes_cc[order]
    weighted_cumsum = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    first_ge = np.searchsorted(sorted_doses, bins, side='left')
    remaining = sorted_weights.sum() - weighted_cumsum[first_ge]
    return bins, 100 * remaining / sorted_weights.sum()


def summarize_roi(roi: ROIDose, *, v_threshold_gy: float = 20, strict: bool = True) -> dict:
    """Only publish clinically interpretable metrics if ROI coverage is complete.

    If an ROI extends outside the calculation box, its full-volume metrics are
    NOT determinable from these optimization voxels alone. This prototype
    fails closed instead of quietly labeling them full-ROI metrics.
    """
    coverage = min(float(roi.calculation_coverage), 1.0)
    is_complete = (np.isfinite(coverage) and coverage >= 0.999)
    result = {'coverage_fraction': coverage, 
              'represented_volume_cc': float(roi.volumes_cc.sum()),
              "status": ("complete" if is_complete else "partial_roi"),
              "metric_scope": ("optimization_roi" if is_complete else "represented_subvolume")
    }
    if strict and coverage < 0.999:
        result.update({'Dmean_Gy': None, 'Dmax_Gy': None,
                       'D95_Gy':None, f'V{v_threshold_gy:g}_pct':None})
        return result
    result.update({'Dmean_Gy': weighted_mean_gy(roi),
                   'Dmax_Gy': float(roi.dose_gy.max()),
                   'D95_Gy': dose_at_volume_gy(roi, 95),
                   f'V{v_threshold_gy:g}_pct':volume_at_dose_percent(roi, v_threshold_gy)})
    return result