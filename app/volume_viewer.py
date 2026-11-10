"""Data side of the in-browser 3D slice viewer (see app/web/js/viewer3.js).

The browser fetches the cropped CT (int16 HU), a bit-packed ROI mask volume
and one dose volume per plan version ONCE, then renders every slice, window,
opacity, contour and difference change locally on a <canvas>. No server round
trip per slider step, so scrolling is smooth.

Everything is gzip-compressed and base64-encoded for transport through a
server endpoint (GET /api/volume); the browser decompresses with DecompressionStream.
"""
from __future__ import annotations

import base64
import gzip
import threading

import matplotlib
import numpy as np
from scipy import ndimage

from app.viz import CT_WINDOWS, DEFAULT_WINDOW, ISODOSE_LEVELS, ROI_COLORS, SliceViewer
from app.ui_text import roi_label
from src.plan_export import dose_on_ct_grid


def _pack(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), compresslevel=3)).decode('ascii')


def _lut(name: str) -> list[int]:
    rgb = matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]
    return np.round(rgb * 255).astype(int).ravel().tolist()


class VolumeData:
    def __init__(self, case, viewer: SliceViewer, rois: list[str]):
        self.case = case
        self.viewer = viewer
        ys, xs = viewer.crop
        self.ys, self.xs = ys, xs
        ct = viewer.ct[:, ys, xs]
        self.shape = ct.shape                      # (nz, ny, nx)
        self._ct = np.clip(ct, -1024, 3071).astype('<i2')
        self.rois = []
        masks = np.zeros(self.shape, dtype=np.uint8)
        for name in rois:
            m = viewer.mask(name)
            if m is None or len(self.rois) >= 8:
                continue
            masks |= (m[:, ys, xs].astype(np.uint8) << len(self.rois))
            self.rois.append(name)
        self._masks = masks
        inside = viewer.vox_map[:, ys, xs] >= 0
        self._inside = inside
        self._w = ndimage.uniform_filter(inside.astype(np.float32), size=(1, 3, 3))
        self._cache: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        sp = case.ct.ct_dict['resolution_xyz_mm']
        self.meta = {
            'nz': int(self.shape[0]), 'ny': int(self.shape[1]), 'nx': int(self.shape[2]),
            'spacing': [float(s) for s in sp], 'default_z': int(viewer.default_z),
            'rois': [{'name': n, 'label': roi_label(n), 'bit': i, 'color': ROI_COLORS.get(n, '#ffffff')}
                     for i, n in enumerate(self.rois)],
            'default_rois': [n for n in ('PTV', 'ESOPHAGUS', 'CORD') if n in self.rois],
            'windows': {k: list(v) for k, v in CT_WINDOWS.items()}, 'default_window': DEFAULT_WINDOW,
            'vmax': float(viewer.dose_vmax), 'rx': float(case.plan.get_prescription()),
            'isodose': [{'pct': p, 'gy': round(p / 100 * float(case.plan.get_prescription()), 2), 'color': c,
                         'on': on} for p, c, on in ISODOSE_LEVELS],
            'turbo': _lut('turbo'), 'rdbu': _lut('RdBu_r'),
        }
        zz, yy, xx = np.nonzero(inside)                 # dose calculation box in cropped-volume voxel indices
        if len(zz):
            self.meta['box'] = {'z0': int(zz.min()), 'z1': int(zz.max()), 'y0': int(yy.min()), 'y1': int(yy.max()),
                                'x0': int(xx.min()), 'x1': int(xx.max())}

    def dose_volume(self, dose_1d: np.ndarray) -> np.ndarray:
        """Cropped CT-grid dose, box-smoothed over one dose voxel in-plane (same as the 2D renderer)."""
        d = dose_on_ct_grid(self.case, dose_1d)[:, self.ys, self.xs] * self._inside
        num = ndimage.uniform_filter(d, size=(1, 3, 3))
        return np.where(self._inside, num / np.maximum(self._w, 1e-6), 0.0).astype(np.float32)

    def get(self, kind: str, version=None, session=None) -> dict:
        key = (kind, version)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if kind == 'meta':
            out = self.meta
        elif kind == 'ct':
            out = {'data': _pack(self._ct)}
        elif kind == 'masks':
            out = {'data': _pack(self._masks)}
        elif kind == 'dose':
            v = session.get_version(int(version))
            if not v.ok:
                return {'error': f'version {version} has no dose'}
            out = self._pack_dose(v.dose_total_gy)
        else:
            return {'error': f'unknown volume {kind}'}
        with self._lock:
            self._cache[key] = out
        return out

    def _pack_dose(self, dose_1d) -> dict:
        d = self.dose_volume(dose_1d)
        scale = max(float(d.max()), 1e-6) / 65000.0
        return {'data': _pack(np.round(d / scale).astype('<u2')), 'scale': scale, 'max': float(d.max())}

    def get_dose(self, key: str, dose_1d) -> dict:
        """A dose volume that is not a plan version (e.g. the ECHO reference), cached under `key`."""
        k = ('dose', key)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        out = self._pack_dose(dose_1d)
        with self._lock:
            self._cache[k] = out
        return out


VIEWER_HTML = """
<div class="bv">
  <div class="tg-row rois"><span class="tg-lbl">Structures</span></div>
  <div class="bv-bar">
    <div class="grp"><span class="gl">Display</span>
      <div class="segc" data-k="mode"><button type="button" data-v="dose" class="on">Dose</button><button
        type="button" data-v="difference">Difference</button></div></div>
    <div class="grp"><span class="gl">Layout</span>
      <div class="segc" data-k="layout"><button type="button" data-v="side" class="on">CT + dose</button><button
        type="button" data-v="single">Dose only</button></div></div>
    <div class="grp"><span class="gl">CT window</span><select class="win"></select></div>
    <div class="grp"><span class="gl">Dose opacity</span><input type="range" class="op" min="0" max="1" step="0.05"
      value="0.5"><b class="opv">50%</b></div>
    <div class="grp"><button type="button" class="tgl iso" aria-pressed="false">Isodose lines</button></div>
  </div>
  <div class="bv-iso" hidden><span class="gl">Isodose</span><span class="isorx"></span><span class="isolv"></span></div>
  <div class="bv-status">Loading…</div>
  <div class="bv-stage">
    <figure class="pane pane-ct"><figcaption class="cap1"></figcaption><canvas class="img c1"></canvas></figure>
    <figure class="pane pane-dose"><figcaption class="cap2"></figcaption><canvas class="img c2"></canvas></figure>
    <div class="cbar"><canvas class="cb"></canvas><div class="ticks"></div><div class="cblabel"></div></div>
  </div>
  <div class="bv-scrub"><span class="gl">Slice</span><input type="range" class="z" min="0" max="1" step="1">
    <b class="zval"></b><span class="hintw">mouse wheel over the image also scrolls</span></div>
  <div class="bv-readout">&nbsp;</div>
</div>
"""

VIEWER_CSS = """
.bv { display: flex; flex-direction: column; gap: 0; font-size: 13px; }
.bv-bar, .bv-scrub { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 22px; padding: 8px 12px;
  background: #16191f; color: #d5d9e0; }
.bv-bar { border-radius: 6px 6px 0 0; border-bottom: 1px solid #262b34; }
.bv-scrub { border-radius: 0 0 6px 6px; border-top: 1px solid #262b34; gap: 10px; }
.bv-scrub .z { flex: 1 1 auto; accent-color: #7ea2ff; }
.bv-scrub b { min-width: 64px; font-weight: 600; color: #fff; font-variant-numeric: tabular-nums; }
.bv-scrub .hintw { font-size: 11.5px; color: #8b93a1; }
.grp { display: inline-flex; align-items: center; gap: 8px; }
.gl { font-size: 10.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: #8b93a1; }
.segc { display: inline-flex; align-items: stretch; height: 28px; box-sizing: border-box; border: 1px solid #3a414d;
  border-radius: 4px; overflow: hidden; }
.segc button { font: inherit; font-size: 12.5px; line-height: 1; height: auto !important; min-height: 0 !important;
  margin: 0 !important; padding: 0 11px; border: none; border-radius: 0; background: transparent; color: #c3c9d3;
  cursor: pointer; display: inline-flex; align-items: center; box-sizing: border-box; }
.segc button + button { border-left: 1px solid #3a414d; }
.segc button:hover { background: #222833; }
.segc button.on { background: #e9edf5; color: #10141b; font-weight: 600; }
.segc button, .tgl { display: block !important; line-height: 26px !important; padding-top: 0 !important;
  padding-bottom: 0 !important; text-align: center; }
.tgl { line-height: 26px !important; }
.segc button.dis { color: #5c6470; cursor: not-allowed; }
.tgl { font: inherit; font-size: 12.5px; line-height: 1; height: 28px; margin: 0; padding: 0 11px;
  border: 1px solid #3a414d; border-radius: 4px; background: transparent; color: #c3c9d3; cursor: pointer;
  box-sizing: border-box; }
.tgl[aria-pressed="true"] { background: #e9edf5; color: #10141b; font-weight: 600; }
.bv-bar select { font: inherit; font-size: 12.5px; height: 28px; padding: 0 6px; border-radius: 4px;
  border: 1px solid #3a414d; background: #1f242c; color: #e6e9ee; color-scheme: dark; }
.bv-bar select option { background: #1f242c; color: #e6e9ee; }
.bv-iso { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; padding: 6px 12px;
  background: #16191f; border-bottom: 1px solid #262b34; color: #c3c9d3; }
.bv-iso[hidden] { display: none; }
.isorx { font-size: 12px; color: #8b93a1; }
.isolv { display: inline-flex; flex-wrap: wrap; gap: 2px 4px; }
.isolv button { display: inline-flex; align-items: center; gap: 7px; height: 24px; margin: 0; padding: 0 8px;
  border: 1px solid transparent; border-radius: 4px; background: transparent; font: inherit; font-size: 12.5px;
  line-height: 1; color: #e6e9ee; cursor: pointer; font-variant-numeric: tabular-nums; }
.isolv button:hover { background: #222833; }
.isolv button i { width: 16px; height: 0; border-top: 2px solid var(--c); flex: none; }
.isolv button span { color: #e6e9ee !important; }
.isolv button small { color: #8b93a1 !important; font-size: 11.5px; }
.isolv button.off span { color: #6b7380 !important; }
.isolv button.off { color: #6b7380; } .isolv button.off i { border-top-style: dotted; opacity: .5; }
.bv-bar input[type=range] { width: 110px; accent-color: #7ea2ff; }
.bv-bar b { font-weight: 600; color: #fff; min-width: 34px; font-variant-numeric: tabular-nums; }
.bv-status { padding: 8px 12px; color: #8b93a1; background: #000; }
.bv-stage { display: flex; gap: 8px; align-items: flex-start; justify-content: center; background: #000;
  padding: 8px; }
.pane { margin: 0; flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; align-items: center; }
.pane figcaption { color: #c3c9d3; font-size: 12px; padding: 2px 0 6px; text-align: center; }
.pane canvas { width: 100%; height: auto; max-height: max(360px, calc(100vh - 500px)); object-fit: contain;
  image-rendering: auto; cursor: crosshair; }
.pane-ct[hidden] + .pane-dose canvas { width: auto; max-width: 100%; }
.cbar { position: relative; width: 64px; flex: 0 0 64px; align-self: stretch; margin: 26px 0 8px; }
.cbar canvas { position: absolute; left: 0; top: 0; width: 12px; height: 100%; image-rendering: auto; }
.cbar .ticks span { position: absolute; left: 17px; transform: translateY(-50%); color: #c3c9d3; font-size: 11px; }
.cbar .cblabel { position: absolute; left: 44px; top: 50%; transform: translate(-50%, -50%) rotate(-90deg);
  white-space: nowrap; color: #c3c9d3; font-size: 11px; transform-origin: center; }
.rois { padding: 0 2px 8px; }
.bv-readout { font-size: 12px; color: #5d6676; padding: 4px 4px 0; min-height: 18px; font-variant-numeric: tabular-nums; }
"""

EXPORT_HTML = """<div class="exp"><button class="exp-btn" type="button">Export shown plan · DICOM (.zip)</button>
<div class="exp-msg"></div></div>"""

REPORT_HTML = """<div class="exp"><button class="exp-btn" type="button">Session report (.html)</button>
<div class="exp-msg"></div></div>"""

EXPORT_CSS = """
.exp { display: flex; flex-direction: column; gap: 4px; }
.exp-btn { width: 100%; height: 38px; padding: 0 10px; border: none; border-radius: 8px; box-sizing: border-box;
  background: var(--button-primary-background-fill); color: var(--button-primary-text-color);
  font-weight: 600; font-size: 13px; cursor: pointer; }
.exp-btn:hover { background: var(--button-primary-background-fill-hover); }
.exp-btn:disabled { opacity: .6; cursor: progress; }
.exp-msg { font-size: 11px; color: var(--body-text-color-subdued); min-height: 14px; overflow-wrap: anywhere; }
"""

# Save dialog first (needs the click's user activation), then build the zip on the server and write it.
EXPORT_JS = r"""
element.addEventListener('click', async (e) => {
  const btn = e.target.closest('.exp-btn');
  if (!btn) return;
  const msg = element.querySelector('.exp-msg');
  let info = {};
  try { info = JSON.parse(props.value || '{}'); } catch (err) {}
  const name = info.name || 'BeamOpt_export.zip';
  const fn = info.fn || 'export_zip', mime = info.mime || 'application/zip', ext = info.ext || '.zip';
  let handle = null;
  if (window.showSaveFilePicker) {
    try {
      handle = await window.showSaveFilePicker({suggestedName: name,
        types: [{description: info.desc || 'Zip archive', accept: {[mime]: [ext]}}]});
    } catch (err) {
      if (err && err.name === 'AbortError') { msg.textContent = 'Export cancelled'; return; }
      handle = null;
    }
  }
  btn.disabled = true; msg.textContent = 'Exporting…';
  try {
    const r = await server[fn](info.view === undefined ? -1 : info.view);
    if (r.error) throw new Error(r.error);
    const bytes = Uint8Array.from(atob(r.data), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], {type: mime});
    if (handle) {
      const w = await handle.createWritable(); await w.write(blob); await w.close();
      msg.textContent = `Saved ${handle.name}`;
    } else {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = r.name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      msg.textContent = `Downloaded ${r.name}`;
    }
  } catch (err) {
    msg.textContent = 'Export failed: ' + err.message;
  } finally { btn.disabled = false; }
});
"""
