// Interactive DVH (SVG): real curves for the shown plan (solid), its parent (dashed) and ECHO (dotted).
// Hover readout, click a curve to select the organ, legend to show / hide, drag to zoom, double-click to
// reset, and goal / limit markers that can be dragged to fill in an instruction (never sent automatically).
import {$, esc, fmt, opTxt, status, store, toast, unit} from './api.js';
import {S, color, emit, label, on, parentVersion, rev, shownVersion, version} from './state.js';
import {targetText} from './board.js';

const DEFAULT = ['PTV', 'ESOPHAGUS', 'CORD', 'LUNGS_NOT_GTV'];
const G = {L: 46, T: 30, R: 390, B: 362};           // plot area in viewBox units (400 x 400)
const D = {shown: null, parent: null, echo: null, dom: null, full: null, drag: null, plans: store.get('dvhPlans', {parent: true, echo: false}),
           markers: []};
let shownRois = new Set(store.get('dvhRois', DEFAULT));

const X = (d) => G.L + (d - D.dom.x0) / (D.dom.x1 - D.dom.x0) * (G.R - G.L);
const Y = (v) => G.B - (v - D.dom.y0) / (D.dom.y1 - D.dom.y0) * (G.B - G.T);
const Xi = (px) => D.dom.x0 + (px - G.L) / (G.R - G.L) * (D.dom.x1 - D.dom.x0);
const Yi = (py) => D.dom.y0 + (G.B - py) / (G.B - G.T) * (D.dom.y1 - D.dom.y0);

function ticks(a, b, n = 6) {
  const span = b - a, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(6));
  return out;
}

function vAt(dvh, roi, d) {
  const ys = dvh?.curves?.[roi];
  if (!ys) return null;
  const f = d / dvh.step, i = Math.floor(f);
  if (i < 0) return ys[0];
  if (i >= ys.length - 1) return ys[ys.length - 1];
  return ys[i] + (ys[i + 1] - ys[i]) * (f - i);
}

function path(dvh, roi) {
  const ys = dvh?.curves?.[roi];
  if (!ys) return '';
  let s = '', started = false;
  for (let i = 0; i < ys.length; i++) {
    const d = i * dvh.step;
    if (d < D.dom.x0 - dvh.step || d > D.dom.x1 + dvh.step) continue;
    s += `${started ? 'L' : 'M'}${X(d).toFixed(1)},${Y(ys[i]).toFixed(1)}`;
    started = true;
  }
  return s;
}

// goal / limit markers: where on the DVH a criterion lives (dose, volume) and which axis a drag moves
function markerFor(row) {
  const m = row.metric, lim = row.limit;
  if (lim === null || lim === undefined) return null;
  if (m === 'Dmax') return {d: lim, v: 0, axis: 'x'};
  if (/^D\d+$/.test(m)) return {d: lim, v: +m.slice(1), axis: 'x'};
  if (/^V\d+/.test(m)) return {d: parseFloat(m.slice(1)), v: lim, axis: 'y'};
  if (m === 'Dmean') return {d: lim, v: null, axis: 'x', mean: true};
  return null;
}

function collectMarkers(r) {
  const rows = [];
  for (const c of r?.checks || []) rows.push({roi: c.roi, metric: c.metric, op: c.op, limit: c.threshold, goal: null,
    who: c.role === 'objective' ? `Your request (${r.id})` : c.role === 'guard' ? 'Guard added by the agent' : 'Protection'});
  for (const p of S.st.protocol_rows) if (p.board) rows.push({...p, who: 'Protocol'});
  return rows.filter((x) => shownRois.has(x.roi) && S.st.rois[x.roi]?.available !== false)
    .map((x) => ({...x, pos: markerFor(x)})).filter((x) => x.pos);
}

async function load() {
  const r = rev(S.sel);
  if (!r) return;
  const [a, b] = await Promise.all([version(shownVersion(r)), version(parentVersion(r))]);
  if (rev(S.sel) !== r) return;
  D.shown = a?.dvh || null; D.parent = b?.dvh || null; D.echo = S.st.echo?.dvh || null;
  D.vshown = a; D.rev = r;
  const n = Math.max(...Object.values(D.shown?.curves || {PTV: [0]}).map((ys) => ys.length));
  const x1 = Math.ceil(n * (D.shown?.step || 0.25) / 5) * 5;
  D.full = {x0: 0, x1: Math.max(10, x1), y0: 0, y1: 102};
  if (!D.dom) D.dom = {...D.full};
  render();
}

export function render() {
  if (!D.shown || !D.dom) return;
  const svg = $('dvh'), cs = getComputedStyle(document.documentElement);
  const line = cs.getPropertyValue('--line').trim(), mut = cs.getPropertyValue('--muted').trim();
  const r = D.rev, hasParent = D.parent && r && r.id !== 'R0' && D.plans.parent;
  let s = `<defs><clipPath id="plotclip"><rect x="${G.L}" y="${G.T - 2}" width="${G.R - G.L}" height="${G.B - G.T + 4}"/></clipPath></defs>`;
  for (const t of ticks(D.dom.x0, D.dom.x1, 7)) s += `<line x1="${X(t)}" y1="${G.T}" x2="${X(t)}" y2="${G.B}" stroke="${line}"/><text x="${X(t)}" y="${G.B + 13}" font-size="10" text-anchor="middle" fill="${mut}" font-family="IBM Plex Mono, monospace">${t}</text>`;
  for (const t of ticks(D.dom.y0, Math.min(100, D.dom.y1), 5)) s += `<line x1="${G.L}" y1="${Y(t)}" x2="${G.R}" y2="${Y(t)}" stroke="${line}"/><text x="${G.L - 5}" y="${Y(t) + 3}" font-size="10" text-anchor="end" fill="${mut}" font-family="IBM Plex Mono, monospace">${t}</text>`;
  s += `<text x="${G.R}" y="${G.B + 27}" font-size="10" text-anchor="end" fill="${mut}">Dose (Gy)</text>`;
  s += `<text x="11" y="${(G.T + G.B) / 2}" font-size="10" fill="${mut}" transform="rotate(-90 11 ${(G.T + G.B) / 2})" text-anchor="middle">Volume (%)</text>`;
  s += `<g clip-path="url(#plotclip)">`;
  const rois = Object.keys(D.shown.curves).filter((x) => shownRois.has(x));
  for (const roi of rois) {
    const sel = S.organ === roi || (S.organ === 'LUNGS_NOT_GTV' && roi === 'LUNGS_NOT_GTV');
    const op = !S.organ || sel ? 1 : 0.3, w = sel ? 2.6 : 1.6, c = color(roi);
    if (D.plans.echo && D.echo) s += `<path d="${path(D.echo, roi)}" fill="none" stroke="${c}" stroke-width="1.2" stroke-dasharray="1.5 3" opacity="${op * 0.9}"/>`;
    if (hasParent) s += `<path d="${path(D.parent, roi)}" fill="none" stroke="${c}" stroke-width="1.2" stroke-dasharray="5 3" opacity="${op * 0.85}"/>`;
    s += `<path d="${path(D.shown, roi)}" fill="none" stroke="${c}" stroke-width="${w}" opacity="${op}" data-roi="${esc(roi)}"/>`;
  }
  // markers
  D.markers = collectMarkers(r);
  D.markers.forEach((mk, k) => {
    const p = mk.pos, now = D.vshown?.metrics?.[mk.roi]?.[mk.metric];
    const st = status(typeof now === 'number' ? now : null, mk.op, mk.limit, mk.goal);
    const fill = st === 'na' ? 'var(--faint)' : `var(--${st})`;
    const x = X(p.d), y = p.mean ? G.B - 1 : Y(p.v);
    const op = !S.organ || S.organ === mk.roi ? 1 : 0.3;
    const shape = p.axis === 'y' ? `M${x - 6},${y} L${x + 1},${y - 5} L${x + 1},${y + 5} Z` : `M${x},${y + 1} L${x - 5},${y - 8} L${x + 5},${y - 8} Z`;
    const tip = `${mk.who}: ${label(mk.roi)} ${mk.metric} ${opTxt(mk.op)} ${+(+mk.limit).toFixed(2)}${mk.goal != null ? ` (goal ${mk.goal})` : ''} ${unit(mk.metric)} · now ${fmt(now, mk.metric)} · drag to set a new target`;
    s += `<g class="mk" data-mk="${k}" opacity="${op}" style="cursor:${p.axis === 'x' ? 'ew-resize' : 'ns-resize'}"><path d="${shape}" fill="${fill}" stroke="${color(mk.roi)}" stroke-width="1.4"/>${p.mean ? `<text x="${x}" y="${y - 10}" font-size="8.5" text-anchor="middle" fill="${mut}">mean</text>` : ''}<circle cx="${x}" cy="${y - 3}" r="9" fill="transparent"/><title>${esc(tip)}</title></g>`;
  });
  s += `</g><line id="dvhCursor" x1="0" y1="${G.T}" x2="0" y2="${G.B}" stroke="${mut}" stroke-dasharray="2 3" visibility="hidden"/>
    <rect id="dvhZoom" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1" visibility="hidden"/>
    <g id="dvhDragLbl" visibility="hidden"><rect rx="2" fill="var(--panel)" stroke="var(--line-2)"/><text font-size="10.5" fill="var(--text)"></text></g>`;
  svg.innerHTML = s;
  legend(rois, hasParent);
}

function legend(rois, hasParent) {
  const all = Object.keys(D.shown.curves);
  const r = D.rev;
  let h = all.map((roi) => `<button data-leg="${esc(roi)}" class="${shownRois.has(roi) ? '' : 'off'}${S.organ === roi ? ' sel' : ''}" title="Show or hide ${esc(label(roi))}"><i style="background:${color(roi)}"></i>${esc(label(roi))}${S.st.rois[roi]?.coverage < 0.999 ? ' <span class="muted">(part)</span>' : ''}</button>`).join('');
  h += '<div class="sep"></div>';
  h += `<button data-plan="solid" class="" title="The plan you are viewing"><i style="background:var(--text)"></i>${esc(r?.id || '')} solid</button>`;
  if (r && r.id !== 'R0') h += `<button data-plan="parent" class="${D.plans.parent ? '' : 'off'}" title="The plan this request started from"><i style="background:repeating-linear-gradient(90deg,var(--text) 0 4px,transparent 4px 6px)"></i>parent dashed</button>`;
  if (D.echo) h += `<button data-plan="echo" class="${D.plans.echo ? '' : 'off'}" title="ECHO benchmark plan (same beams)"><i style="background:repeating-linear-gradient(90deg,var(--text) 0 1.5px,transparent 1.5px 4px)"></i>ECHO dotted</button>`;
  $('dvhLeg').innerHTML = h;
}

// ---------------------------------------------------------------- interaction
function pt(e) {
  const svg = $('dvh'), p = svg.createSVGPoint();
  p.x = e.clientX; p.y = e.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  return {x: q.x, y: q.y};
}
const inPlot = (p) => p.x >= G.L && p.x <= G.R && p.y >= G.T && p.y <= G.B;

function tip(e, p) {
  const t = $('dvhTip'), cur = $('dvhCursor');
  if (!inPlot(p) || !D.shown) { t.hidden = true; cur?.setAttribute('visibility', 'hidden'); return; }
  const d = Xi(p.x), r = D.rev, hasParent = D.parent && r && r.id !== 'R0' && D.plans.parent;
  cur.setAttribute('x1', p.x); cur.setAttribute('x2', p.x); cur.setAttribute('visibility', 'visible');
  const rows = Object.keys(D.shown.curves).filter((x) => shownRois.has(x)).map((roi) => {
    const a = vAt(D.shown, roi, d), b = hasParent ? vAt(D.parent, roi, d) : null;
    const dd = b !== null ? a - b : null;
    return `<tr><td><i style="display:inline-block;width:9px;height:2px;background:${color(roi)};vertical-align:middle"></i> ${esc(label(roi))}</td><td class="num">${a.toFixed(1)}%</td>${hasParent ? `<td class="num" style="color:${Math.abs(dd) < 0.05 ? 'var(--muted)' : dd < 0 ? 'var(--pass)' : 'var(--fail)'}">${dd >= 0 ? '+' : ''}${dd.toFixed(1)}</td>` : ''}</tr>`;
  }).join('');
  t.innerHTML = `<div class="lbl" style="margin-bottom:3px">${d.toFixed(1)} Gy${hasParent ? ' · Δ vs parent' : ''}</div><table>${rows}</table>`;
  t.hidden = false;
  const pane = $('dvhPane').getBoundingClientRect();
  let lx = e.clientX - pane.left + 14, ly = e.clientY - pane.top + 12;
  if (lx + t.offsetWidth > pane.width - 4) lx = e.clientX - pane.left - t.offsetWidth - 14;
  if (ly + t.offsetHeight > pane.height - 4) ly = pane.height - t.offsetHeight - 4;
  t.style.left = lx + 'px'; t.style.top = ly + 'px';
}

function nearestCurve(p) {
  const d = Xi(p.x);
  let best = null, bd = 9;
  for (const roi of Object.keys(D.shown.curves)) {
    if (!shownRois.has(roi)) continue;
    const v = vAt(D.shown, roi, d);
    const dist = Math.abs(Y(v) - p.y);
    if (dist < bd) { bd = dist; best = roi; }
  }
  return best;
}

function dragLabel(p, text) {
  const g = $('dvhDragLbl');
  if (!g) return;
  const tx = g.querySelector('text'), rc = g.querySelector('rect');
  tx.textContent = text;
  const x = Math.min(G.R - 120, Math.max(G.L + 2, p.x + 8)), y = Math.max(G.T + 12, p.y - 10);
  tx.setAttribute('x', x + 4); tx.setAttribute('y', y);
  rc.setAttribute('x', x); rc.setAttribute('y', y - 11); rc.setAttribute('width', text.length * 5.9 + 8); rc.setAttribute('height', 15);
  g.setAttribute('visibility', 'visible');
}

export function initDvh() {
  const svg = $('dvh');
  svg.addEventListener('pointerdown', (e) => {
    if (!D.shown) return;
    const p = pt(e), mk = e.target.closest('.mk');
    svg.setPointerCapture(e.pointerId);
    if (mk) { D.drag = {kind: 'marker', k: +mk.dataset.mk, start: p, moved: false}; return; }
    if (inPlot(p)) D.drag = {kind: 'zoom', start: p, moved: false};
  });
  svg.addEventListener('pointermove', (e) => {
    const p = pt(e);
    const dr = D.drag;
    if (!dr) { tip(e, p); return; }
    $('dvhTip').hidden = true;
    if (Math.abs(p.x - dr.start.x) + Math.abs(p.y - dr.start.y) > 4) dr.moved = true;
    if (dr.kind === 'zoom' && dr.moved) {
      const z = $('dvhZoom');
      const x0 = Math.max(G.L, Math.min(dr.start.x, p.x)), x1 = Math.min(G.R, Math.max(dr.start.x, p.x));
      const y0 = Math.max(G.T, Math.min(dr.start.y, p.y)), y1 = Math.min(G.B, Math.max(dr.start.y, p.y));
      z.setAttribute('x', x0); z.setAttribute('y', y0); z.setAttribute('width', x1 - x0); z.setAttribute('height', y1 - y0);
      z.setAttribute('visibility', 'visible');
    }
    if (dr.kind === 'marker' && dr.moved) {
      const mk = D.markers[dr.k], g = svg.querySelector(`.mk[data-mk="${dr.k}"]`);
      const v = mk.pos.axis === 'x' ? Math.max(0, Xi(p.x)) : Math.max(0, Math.min(100, Yi(p.y)));
      dr.value = Math.round(v * 10) / 10;
      const dx = mk.pos.axis === 'x' ? p.x - dr.start.x : 0, dy = mk.pos.axis === 'y' ? p.y - dr.start.y : 0;
      g.setAttribute('transform', `translate(${dx},${dy})`);
      dragLabel(p, `${label(mk.roi)} ${mk.metric} ${opTxt(mk.op)} ${dr.value} ${unit(mk.metric)}`);
    }
  });
  svg.addEventListener('pointerup', (e) => {
    const dr = D.drag, p = pt(e);
    D.drag = null;
    if (!dr) return;
    if (dr.kind === 'zoom') {
      $('dvhZoom').setAttribute('visibility', 'hidden');
      if (dr.moved && Math.abs(p.x - dr.start.x) > 8 && Math.abs(p.y - dr.start.y) > 8) {
        const a = Xi(Math.min(dr.start.x, p.x)), b = Xi(Math.max(dr.start.x, p.x));
        const c = Yi(Math.max(dr.start.y, p.y)), d = Yi(Math.min(dr.start.y, p.y));
        D.dom = {x0: Math.max(0, a), x1: b, y0: Math.max(0, c), y1: Math.min(102, d)};
        render();
      } else if (!dr.moved) {
        const roi = nearestCurve(p);
        if (roi) { S.organ = S.organ === roi ? null : roi; emit('organ', {jump: true}); }
      }
    }
    if (dr.kind === 'marker') {
      if (dr.moved && dr.value !== undefined) {
        const mk = D.markers[dr.k];
        emit('fill', targetText(mk.roi, mk.metric, mk.op, dr.value));
        toast('Instruction filled in below. Press Send to ask the agent.');
      }
      render();
    }
  });
  svg.addEventListener('pointerleave', () => { if (!D.drag) { $('dvhTip').hidden = true; $('dvhCursor')?.setAttribute('visibility', 'hidden'); } });
  svg.addEventListener('dblclick', () => { D.dom = {...D.full}; render(); });
  $('dvhLeg').addEventListener('click', (e) => {
    const b = e.target.closest('[data-leg]'), pl = e.target.closest('[data-plan]');
    if (b) {
      const roi = b.dataset.leg;
      shownRois.has(roi) ? shownRois.delete(roi) : shownRois.add(roi);
      store.set('dvhRois', [...shownRois]); render();
    }
    if (pl && pl.dataset.plan !== 'solid') {
      D.plans[pl.dataset.plan] = !D.plans[pl.dataset.plan]; store.set('dvhPlans', D.plans); render();
    }
  });
  on('select', load);
  on('revisions', load);
  on('organ', () => { if (S.organ && !shownRois.has(S.organ) && D.shown?.curves?.[S.organ]) { shownRois.add(S.organ); } render(); });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', render);
  on('theme', render);
}
