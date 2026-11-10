import numpy as np
import pytest

from src.metrics import ROIDose, summarize_roi


def test_four_voxel_example():
    roi = ROIDose(np.array([10., 20., 30., 40.]), np.ones(4), 1.0)
    s = summarize_roi(roi)
    assert s['Dmean_Gy'] == 25 and s['Dmax_Gy'] == 40 and s['D95_Gy'] == 10 and s['V20_pct'] == 75


def test_partial_roi_fails_closed():
    roi = ROIDose(np.array([10., 20.]), np.ones(2), 0.5)
    s = summarize_roi(roi, strict=True)
    assert s['status'] == 'partial_roi' and s['Dmean_Gy'] is None


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ROIDose(np.array([-1.0]), np.ones(1), 1.0)
