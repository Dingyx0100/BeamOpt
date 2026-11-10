// Linked axial / coronal / sagittal viewer. CT, ROI masks and dose volumes are fetched once (gzip + base64) and
// every slice is drawn locally. All three panes are square and share ONE mm-per-pixel scale (1:1:1), the
// shorter side of each image padded with black.
import {$, api, esc, hex, store, toast, unpack} from './api.js';
import {S, label, on, parentVersion, rev, shownVersion} from './state.js';

const V = {key: null, meta: null, ct: null, masks: null, doses: new Map(), loading: new Map(), diffLim: new Map(),
           cross: null, mode: 'dose', win: null, op: store.get('op', 0.5), iso: store.get('iso', false),
           isoOn: new Set(), roisOff: new Set(store.get('roisOff', [])), cent: new Map(), raf: 0, hover: null};
const GROUP = {LUNGS_NOT_GTV: ['LUNG_L', 'LUNG_R']};
const off = document.createElement('canvas');

// Slice index 0 is the most inferior CT slice, so superior is drawn at the top of coronal / sagittal.
const PANES = {
  ax: {cv: 'cAx', lab: 'lAx', msg: 'mAx', dims: (m) => [m.nx, m.ny, m.spacing[0], m.spacing[1]],
       idx: (m, c, i, j) => c.z * m.ny * m.nx + j * m.nx + i, hv: (m, c) => [c.x, c.y],
       set: (c, i, j) => { c.x = i; c.y = j; }, wheel: (m, c, s) => { c.z = clamp(c.z + s, 0, m.nz - 1); },
       txt: (m, c) => `slice ${c.z + 1} / ${m.nz}`, letters: ['A', 'P', 'R', 'L']},
  co: {cv: 'cCo', lab: 'lCo', msg: 'mCo', dims: (m) => [m.nx, m.nz, m.spacing[0], m.spacing[2]],
       idx: (m, c, i, j) => (m.nz - 1 - j) * m.ny * m.nx + c.y * m.nx + i, hv: (m, c) => [c.x, m.nz - 1 - c.z],
       set: (c, i, j, m) => { c.x = i; c.z = m.nz - 1 - j; }, wheel: (m, c, s) => { c.y = clamp(c.y + s, 0, m.ny - 1); },
       txt: (m, c) => `row ${c.y + 1} / ${m.ny}`, letters: ['S', 'I', 'R', 'L']},
  sa: {cv: 'cSa', lab: 'lSa', msg: 'mSa', dims: (m) => [m.ny, m.nz, m.spacing[1], m.spacing[2]],
       idx: (m, c, i, j) => (m.nz - 1 - j) * m.ny * m.nx + i * m.nx + c.x, hv: (m, c) => [c.y, m.nz - 1 - c.z],
       set: (c, i, j, m) => { c.y = i; c.z = m.nz - 1 - j; }, wheel: (m, c, s) => { c.x = clamp(c.x + s, 0, m.nx - 1); },
       txt: (m, c) => `col ${c.x + 1} / ${m.nx}`, letters: ['S', 'I', 'A', 'P']},
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------- data
async function loadPatient(key) {
  V.key = key; V.meta = V.ct = V.masks = null; V.doses = new Map(); V.loading = new Map(); V.diffLim = new Map();
  msg('ax', 'Loading CT…');
  const meta = await api('/api/volume/meta');
  const [ct, masks] = await Promise.all([api('/api/volume/ct'), api('/api/volume/masks')]);
  if (V.key !== key) return;
  V.ct = await unpack(ct.data, Int16Array);
  V.masks = await unpack(masks.data, Uint8Array);
  V.meta = meta;
  V.win = store.get('win', meta.default_window);
  if (!meta.windows[V.win]) V.win = meta.default_window;
  V.isoOn = new Set(meta.isodose.map((L, k) => (L.on ? k : -1)).filter((k) => k >= 0));
  V.cross = {x: Math.floor(meta.nx / 2), y: Math.floor(meta.ny / 2), z: meta.default_z};
  centroids();
  const sel = $('win');
  sel.innerHTML = Object.keys(meta.windows).map((k) => `<option ${k === V.win ? 'selected' : ''}>${esc(k)}</option>`).join('');
  msg('ax', '');
  await ensure();
}

function centroids() {
  const m = V.meta, n = m.nx * m.ny, acc = m.rois.map(() => [0, 0, 0, 0]);
  for (let p = 0; p < V.masks.length; p += 2) {
    const v = V.masks[p];
    if (!v) continue;
    const z = Math.floor(p / n), r = p - z * n, y = Math.floor(r / m.nx), x = r - y * m.nx;
    m.rois.forEach((roi, k) => { if (v & (1 << roi.bit)) { const a = acc[k]; a[0] += x; a[1] += y; a[2] += z; a[3]++; } });
  }
  m.rois.forEach((roi, k) => { const a = acc[k]; if (a[3]) V.cent.set(roi.name, {x: Math.round(a[0] / a[3]), y: Math.round(a[1] / a[3]), z: Math.round(a[2] / a[3])}); });
}

async function dose(id) {
  if (id === null || id === undefined) return null;
  if (V.doses.has(id)) return V.doses.get(id);
  if (!V.loading.has(id)) {
    V.loading.set(id, (async () => {
      const r = await api(id === 'echo' ? '/api/volume/echo' : `/api/volume/dose?version=${id}`);
      const d = {arr: await unpack(r.data, Uint16Array), scale: r.scale, max: r.max};
      V.doses.set(id, d);
      return d;
    })().catch((e) => { V.loading.delete(id); toast('Could not load the dose: ' + e.message, true); return null; }));
  }
  return V.loading.get(id);
}

function ids() {
  const r = rev(S.sel);
  return {view: shownVersion(r), cmp: parentVersion(r), r};
}

async function ensure() {
  if (!V.meta) return;
  const {view, cmp} = ids();
  schedule();
  await dose(view);
  if (V.mode === 'diff' && cmp !== null) await dose(cmp);
  schedule();
}

// ---------------------------------------------------------------- drawing
function diffLimit(a, b, key) {
  if (V.diffLim.has(key)) return V.diffLim.get(key);
  const vals = [];
  for (let i = 0; i < a.arr.length; i += 7) {
    const d = Math.abs(a.arr[i] * a.scale - b.arr[i] * b.scale);
    if (d > 0.05) vals.push(d);
  }
  vals.sort((x, y) => x - y);
  const lim = Math.max(1, vals.length ? vals[Math.floor(vals.length * 0.995)] : 1);
  V.diffLim.set(key, lim);
  return lim;
}

function fov() {
  const m = V.meta;
  return Math.max(...Object.values(PANES).flatMap((P) => { const [c, r, pu, pv] = P.dims(m); return [c * pu, r * pv]; }));
}

function selectedBits() {
  if (!S.organ) return 0;
  const names = GROUP[S.organ] || [S.organ];
  return V.meta.rois.filter((r) => names.includes(r.name)).reduce((b, r) => b | (1 << r.bit), 0);
}

function drawPane(key, ctx) {
  const P = PANES[key], m = V.meta, c = V.cross;
  const [cols, rows, pu, pv] = P.dims(m);
  off.width = cols; off.height = rows;
  const g = off.getContext('2d'), im = g.createImageData(cols, rows), D = im.data;
  const [wc, ww] = m.windows[V.win], lo = wc - ww / 2;
  const {A, B, diff, lim, lut} = ctx;
  const n = cols * rows, dslice = new Float32Array(n), mslice = new Uint8Array(n);
  const op = V.op;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const p = j * cols + i, vi = P.idx(m, c, i, j);
    let gr = (V.ct[vi] - lo) / ww * 255;
    gr = gr < 0 ? 0 : gr > 255 ? 255 : gr;
    let r = gr, gg = gr, b = gr;
    mslice[p] = V.masks[vi];
    if (A) {
      const da = A.arr[vi] * A.scale;
      dslice[p] = da;
      let t = -1;
      if (diff) { const dv = da - B.arr[vi] * B.scale; if (Math.abs(dv) >= 0.1) t = Math.max(0, Math.min(1, (dv / lim + 1) / 2)); }
      else if (da > 0.1) t = Math.min(1, da / lim);
      if (t >= 0) { const k = Math.round(t * 255) * 3; r = gr + (lut[k] - gr) * op; gg = gr + (lut[k + 1] - gr) * op; b = gr + (lut[k + 2] - gr) * op; }
    }
    const q = p * 4; D[q] = r; D[q + 1] = gg; D[q + 2] = b; D[q + 3] = 255;
  }
  if (A && !diff && V.iso) {
    const lv = m.isodose.map((L, k) => ({...L, k})).filter((L) => V.isoOn.has(L.k)).sort((a, b) => a.gy - b.gy);
    for (const L of lv) {
      const [cr, cg, cb] = hex(L.color), g0 = L.gy;
      for (let j = 1; j < rows - 1; j++) for (let i = 1; i < cols - 1; i++) {
        const p = j * cols + i;
        if (dslice[p] >= g0 && (dslice[p - 1] < g0 || dslice[p + 1] < g0 || dslice[p - cols] < g0 || dslice[p + cols] < g0)) {
          const q = p * 4; D[q] = cr; D[q + 1] = cg; D[q + 2] = cb;
        }
      }
    }
  }
  const selBits = selectedBits();
  for (const roi of m.rois) {
    if (V.roisOff.has(roi.name)) continue;
    const bit = 1 << roi.bit, [cr, cg, cb] = hex(roi.color);
    const dim = selBits && !(selBits & bit), isSel = selBits & bit;
    for (let j = 1; j < rows - 1; j++) for (let i = 1; i < cols - 1; i++) {
      const p = j * cols + i;
      if (!(mslice[p] & bit)) continue;
      const q = p * 4;
      const edge = !((mslice[p - 1] & bit) && (mslice[p + 1] & bit) && (mslice[p - cols] & bit) && (mslice[p + cols] & bit));
      if (edge) {
        const a = dim ? 0.4 : 1;
        D[q] = D[q] + (cr - D[q]) * a; D[q + 1] = D[q + 1] + (cg - D[q + 1]) * a; D[q + 2] = D[q + 2] + (cb - D[q + 2]) * a;
      } else if (isSel) {
        D[q] += (cr - D[q]) * 0.16; D[q + 1] += (cg - D[q + 1]) * 0.16; D[q + 2] += (cb - D[q + 2]) * 0.16;
      }
    }
  }
  g.putImageData(im, 0, 0);

  const cv = $(P.cv), dpr = window.devicePixelRatio || 1, side = cv.clientWidth || 300, N = Math.round(side * dpr);
  if (cv.width !== N) { cv.width = N; cv.height = N; }
  const s = N / fov(), iw = cols * pu * s, ih = rows * pv * s, ox = (N - iw) / 2, oy = (N - ih) / 2;
  P.geo = {ox, oy, iw, ih, cols, rows, N};
  const x = cv.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, N, N);
  x.imageSmoothingEnabled = true; x.drawImage(off, ox, oy, iw, ih);
  const U = (i) => ox + (i + 0.5) / cols * iw, Vv = (j) => oy + (j + 0.5) / rows * ih;
  x.save(); x.beginPath(); x.rect(ox, oy, iw, ih); x.clip();
  if (m.box) {                                     // dose calculation box
    const bx = m.box;
    const [u0, u1, v0, v1] = key === 'ax' ? [bx.x0, bx.x1, bx.y0, bx.y1] : key === 'co'
      ? [bx.x0, bx.x1, m.nz - 1 - bx.z1, m.nz - 1 - bx.z0] : [bx.y0, bx.y1, m.nz - 1 - bx.z1, m.nz - 1 - bx.z0];
    x.setLineDash([5 * dpr, 4 * dpr]); x.strokeStyle = 'rgba(240,196,106,.75)'; x.lineWidth = dpr;
    x.strokeRect(U(u0) - 0.5 * iw / cols, Vv(v0) - 0.5 * ih / rows, (u1 - u0 + 1) * iw / cols, (v1 - v0 + 1) * ih / rows);
    x.setLineDash([]);
    if (key !== 'ax') {
      x.fillStyle = 'rgba(240,196,106,.9)'; x.font = `${10 * dpr}px "IBM Plex Sans Condensed", sans-serif`;
      x.fillText('dose calculation box', U(u0) + 4 * dpr, Vv(v1) + 0.5 * ih / rows - 4 * dpr);
    }
  }
  const [hu, hv] = P.hv(m, c);
  x.strokeStyle = 'rgba(94,163,212,.5)'; x.lineWidth = dpr;
  x.beginPath(); x.moveTo(U(hu), oy); x.lineTo(U(hu), oy + ih); x.moveTo(ox, Vv(hv)); x.lineTo(ox + iw, Vv(hv)); x.stroke();
  x.restore();
  x.fillStyle = 'rgba(201,209,220,.75)'; x.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
  const [top, bot, left, right] = P.letters, pad = 4 * dpr;
  x.textAlign = 'center';
  x.fillText(top, ox + iw / 2, oy > 16 * dpr ? oy - pad : oy + 12 * dpr);
  x.fillText(bot, ox + iw / 2, oy + ih + (N - oy - ih > 16 * dpr ? 12 * dpr : -pad));
  x.textAlign = 'left';
  x.fillText(left, ox > 16 * dpr ? ox - 11 * dpr : ox + pad, oy + ih / 2);
  x.textAlign = 'right';
  x.fillText(right, ox + iw + (N - ox - iw > 16 * dpr ? 11 * dpr : -pad), oy + ih / 2);
  x.textAlign = 'left';
  $(P.lab).textContent = P.txt(m, c);
}

function msg(key, t) { const el = $(PANES[key].msg); if (el) el.textContent = t || ''; }

function draw() {
  V.raf = 0;
  if (!V.meta || !V.ct) return;
  const m = V.meta, {view, cmp, r} = ids();
  const A = V.doses.get(view) || null, B = cmp !== null ? V.doses.get(cmp) || null : null;
  const diff = V.mode === 'diff' && A && B;
  const lim = diff ? diffLimit(A, B, `${view}-${cmp}`) : m.vmax;
  const ctx = {A, B, diff, lim, lut: diff ? m.rdbu : m.turbo};
  for (const k of Object.keys(PANES)) drawPane(k, ctx);
  const outside = m.box && (V.cross.z < m.box.z0 || V.cross.z > m.box.z1);
  msg('ax', !A ? 'Loading dose…' : outside ? 'This slice is outside the dose calculation box: no dose data here.' : '');
  msg('co', ''); msg('sa', '');
  $('mDiff').disabled = cmp === null;
  const curV = S.revs.current_version;
  $('vsub').textContent = `${r ? r.id : ''} · plan v${view}${diff ? ` − v${cmp} (parent)` : ''}${S.organ ? ' · ' + label(S.organ) : ''}`;
  const bn = $('vbanner');
  if (r && view !== curV && r.status !== 'working') {
    const curR = S.revs.current;
    bn.hidden = false;
    bn.textContent = `You are viewing ${r.id}. New requests start from the current plan (${curR}, v${curV})`
      + (['kept', 'review', 'set_aside'].includes(r.status) ? `; use “Continue from ${r.id}” on its card to work on this one.` : '.');
  } else bn.hidden = true;
  foot(ctx);
}

export function schedule() { if (!V.raf) V.raf = requestAnimationFrame(draw); }

function foot({diff, lim, lut}) {
  const m = V.meta;
  let h = `<div class="cbar"><span class="lbl">${diff ? 'Δ dose vs parent, Gy · blue = lower' : 'Total dose, Gy'}</span>
    <canvas id="cbar" width="256" height="1"></canvas><div class="ticks">${[0, 0.25, 0.5, 0.75, 1].map((f) => {
      const v = diff ? -lim + 2 * lim * f : lim * f; return `<span>${Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)}</span>`; }).join('')}</div></div>`;
  if (V.iso && !diff) {
    h += `<div class="isolv"><span class="lbl">Isodose</span>${m.isodose.map((L, k) =>
      `<button data-iso="${k}" class="${V.isoOn.has(k) ? '' : 'off'}" title="${L.pct}% of ${m.rx} Gy"><i style="background:${L.color}"></i>${L.gy % 1 ? L.gy.toFixed(1) : L.gy} Gy</button>`).join('')}</div>`;
  }
  h += `<div class="organs"><span class="lbl">Contours</span>${m.rois.map((r) =>
    `<button data-roi="${esc(r.name)}" class="${V.roisOff.has(r.name) ? 'off' : ''}"><i style="background:${r.color}"></i>${esc(r.label || r.name)}</button>`).join('')}</div>`;
  $('vfoot').innerHTML = h;
  const cb = $('cbar'), g = cb.getContext('2d'), im = g.createImageData(256, 1);
  for (let i = 0; i < 256; i++) { im.data[i * 4] = lut[i * 3]; im.data[i * 4 + 1] = lut[i * 3 + 1]; im.data[i * 4 + 2] = lut[i * 3 + 2]; im.data[i * 4 + 3] = 255; }
  g.putImageData(im, 0, 0);
}

function readout(key, e) {
  const P = PANES[key], g = P.geo, m = V.meta;
  if (!g || !m || !V.ct) return;
  const cv = $(P.cv), rc = cv.getBoundingClientRect();
  const px = (e.clientX - rc.left) / rc.width * g.N, py = (e.clientY - rc.top) / rc.height * g.N;
  const i = Math.floor((px - g.ox) / g.iw * g.cols), j = Math.floor((py - g.oy) / g.ih * g.rows);
  if (i < 0 || j < 0 || i >= g.cols || j >= g.rows) { $('readout').innerHTML = '&nbsp;'; return null; }
  const vi = P.idx(m, V.cross, i, j);
  const {view, cmp} = ids();
  const A = V.doses.get(view), B = cmp !== null ? V.doses.get(cmp) : null;
  const parts = [`HU ${V.ct[vi]}`];
  if (A) parts.push(`dose ${(A.arr[vi] * A.scale).toFixed(2)} Gy`);
  if (A && B) parts.push(`Δ ${(A.arr[vi] * A.scale - B.arr[vi] * B.scale).toFixed(2)} Gy`);
  const inside = m.rois.filter((r) => V.masks[vi] & (1 << r.bit)).map((r) => r.label || r.name);
  if (inside.length) parts.push(inside.join(', '));
  $('readout').textContent = parts.join(' · ');
  return [i, j];
}

// ---------------------------------------------------------------- wiring
export function initViewer() {
  for (const [key, P] of Object.entries(PANES)) {
    const cv = $(P.cv);
    cv.addEventListener('click', (e) => {
      const ij = readout(key, e);
      if (!ij) return;
      P.set(V.cross, ij[0], ij[1], V.meta); schedule();
    });
    cv.addEventListener('mousemove', (e) => readout(key, e));
    cv.addEventListener('mouseleave', () => { $('readout').innerHTML = '&nbsp;'; });
    cv.addEventListener('wheel', (e) => {
      if (!V.meta) return;
      e.preventDefault(); P.wheel(V.meta, V.cross, e.deltaY > 0 ? -1 : 1); schedule();
    }, {passive: false});
  }
  $('mDose').addEventListener('click', () => { V.mode = 'dose'; $('mDose').classList.add('on'); $('mDiff').classList.remove('on'); ensure(); });
  $('mDiff').addEventListener('click', () => { V.mode = 'diff'; $('mDiff').classList.add('on'); $('mDose').classList.remove('on'); ensure(); });
  $('win').addEventListener('change', (e) => { V.win = e.target.value; store.set('win', V.win); schedule(); });
  $('op').value = V.op;
  $('op').addEventListener('input', (e) => { V.op = +e.target.value; store.set('op', V.op); schedule(); });
  $('isoBtn').setAttribute('aria-pressed', String(V.iso));
  $('isoBtn').classList.toggle('on', V.iso);
  $('isoBtn').addEventListener('click', () => {
    V.iso = !V.iso; store.set('iso', V.iso);
    $('isoBtn').setAttribute('aria-pressed', String(V.iso)); $('isoBtn').classList.toggle('on', V.iso); schedule();
  });
  $('vfoot').addEventListener('click', (e) => {
    const iso = e.target.closest('[data-iso]'), roi = e.target.closest('[data-roi]');
    if (iso) { const k = +iso.dataset.iso; V.isoOn.has(k) ? V.isoOn.delete(k) : V.isoOn.add(k); schedule(); }
    if (roi) { const n = roi.dataset.roi; V.roisOff.has(n) ? V.roisOff.delete(n) : V.roisOff.add(n); store.set('roisOff', [...V.roisOff]); schedule(); }
  });
  on('state', () => { if (S.st.header.key !== V.key) loadPatient(S.st.header.key).catch((e) => toast('Could not load the images: ' + e.message, true)); });
  on('select', ensure);
  on('revisions', ensure);
  on('organ', (o) => {
    if (o?.jump && S.organ && V.meta) {
      const names = GROUP[S.organ] || [S.organ];
      const c = names.map((n) => V.cent.get(n)).filter(Boolean);
      if (c.length) {
        const pick = c.length === 1 ? c[0] : c.find((p) => S.organ === 'LUNGS_NOT_GTV' && p) || c[0];
        V.cross = {...pick};
      }
    }
    schedule();
  });
  if (window.ResizeObserver) new ResizeObserver(schedule).observe(document.querySelector('.vgrid'));
}
