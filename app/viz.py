"""Viewer geometry for the web app.

- SliceViewer: CT / mask / dose-map geometry (body crop, PTV-centred default
  slice) used by app/volume_viewer.py. Slices themselves are drawn in the
  browser by app/web/js/viewer3.js, following PortPy_Single_Patient_Quicklook.ipynb:
  numeric HU window (soft tissue default), turbo dose wash with adjustable
  opacity, dose < 0.1 Gy transparent, contours, optional isodose lines.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


ROI_COLORS = {'PTV': '#d62728', 'GTV': '#ff7f0e', 'CTV': '#8c564b', 'ESOPHAGUS': '#1f77b4', 'CORD': '#2ca02c',
              'LUNGS_NOT_GTV': '#9467bd', 'LUNG_L': '#17becf', 'LUNG_R': '#bcbd22', 'HEART': '#e377c2'}

# Numeric HU window (center, width), as in the quicklook notebook.
CT_WINDOWS = {
    'Soft tissue (40 / 400)': (40.0, 400.0),
    'Mediastinum (50 / 350)': (50.0, 350.0),
    'Lung (-600 / 1500)': (-600.0, 1500.0),
    'Bone (300 / 1600)': (300.0, 1600.0),
}
DEFAULT_WINDOW = 'Soft tissue (40 / 400)'
ISODOSE_GY = (20, 40, 57, 60)        # 57 Gy = 95% of the 60 Gy prescription (legacy)
# Isodose lines in % of the prescription: (percent, colour, on by default)
ISODOSE_LEVELS = ((107, '#ff4fd8', False), (100, '#ff3b30', True), (95, '#ff9f0a', True), (80, '#ffd60a', False),
                  (50, '#30d158', True), (30, '#40c8ff', False))



class SliceViewer:
    def __init__(self, case, dose_vmax: float = 66.0):
        self.case = case
        ct = case.ct.ct_dict['ct_hu_3d']
        self.ct = np.asarray(ct[0] if isinstance(ct, list) else ct)
        vm = case.influence_matrix.opt_voxels_dict['ct_to_dose_voxel_map']
        self.vox_map = np.asarray(vm[0] if isinstance(vm, list) else vm)
        self.dose_vmax = float(dose_vmax)
        self._masks = {}
        ptv = self.mask('PTV')
        zs = np.flatnonzero(ptv.any(axis=(1, 2))) if ptv is not None else []
        self.default_z = int(zs[len(zs) // 2]) if len(zs) else self.ct.shape[0] // 2
        self.n_slices = self.ct.shape[0]
        # Crop away air and couch: bounding box of the body (HU > -500) over all slices, plus margin.
        body = (self.ct > -500).any(axis=0)
        body = ndimage.binary_opening(body, iterations=3)
        lab, n = ndimage.label(body)              # keep the patient, drop the couch (separate component)
        if n > 1:
            body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        ys, xs = np.nonzero(body)
        pad = 12
        if len(ys):
            self.crop = (slice(max(0, ys.min() - pad), min(body.shape[0], ys.max() + pad)),
                         slice(max(0, xs.min() - pad), min(body.shape[1], xs.max() + pad)))
        else:
            self.crop = (slice(None), slice(None))

    def mask(self, name):
        if name not in self._masks:
            try:
                self._masks[name] = np.asarray(self.case.structures.get_structure_mask_3d(name), dtype=bool)
            except (ValueError, KeyError, IndexError, AttributeError):
                self._masks[name] = None
        return self._masks[name]

