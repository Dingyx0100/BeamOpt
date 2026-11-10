// Goals board: the request, what is kept from the start, and the protocol, for Baseline / ECHO / shown revision.
import {$, esc, fmt, opTxt, status, unit} from './api.js';
import {S, emit, label, on, rev, shownVersion, version} from './state.js';

const ICON_LOCK = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="7" width="10" height="7" rx="1"/><path d="M5.5 7V5a2.5 2.5 0 015 0v2"/></svg>';
const ICON_PEN = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 13l1-3.5L11 2.5 13.5 5 6.5 12z"/></svg>';
let form = null;          // key of the row whose "set target" form is open
let lastRows = [];

const val = (p, roi, metric) => {
  const m = p?.metrics?.[roi];
  const v = m ? m[metric] : undefined;
  return typeof v === 'number' ? v : null;
};
const info = (roi) => S.st.rois[roi] || {available: true, coverage: 1};

export function boardRows(r) {
  const groups = [];
  const checks = r?.checks || [];
  const obj = checks.filter((c) => c.role === 'objective');
  const keep = checks.filter((c) => c.role !== 'objective');
  const asRow = (c) => ({roi: c.roi, metric: c.metric, op: c.op, limit: c.threshold, goal: null, kind: c.role,
                         who: c.role === 'objective' ? `your request (${r.id})` : c.role === 'guard'
                           ? 'guard added by the agent' : 'protection'});
  if (obj.length) groups.push([`Request · ${r.id}`, obj.map(asRow)]);
  if (keep.length) groups.push(['Kept from start', keep.map(asRow)]);
  groups.push(['Protocol · ' + S.st.header.protocol.replaceAll('_', ' '),
               S.st.protocol_rows.filter((x) => x.board).map((x) => ({...x, who: 'protocol'}))]);
  return groups;
}

function dot(v, row, roi) {
  if (row.limit === null && row.goal === null) return '';
  const s = status(v, row.op, row.limit, row.goal);
  if (s === 'na') return '';
  const cov = info(roi).coverage;
  return cov !== null && cov < 0.999
    ? `<i class="st half" style="--c:var(--${s})" title="${s} · only ${Math.round(cov * 100)}% of the organ is in the dose box"></i>`
    : `<i class="st ${s}" title="${s}"></i>`;
}

export async function renderBoard() {
  const r = rev(S.sel);
  if (!S.st || !r) return;
  const [base, shown] = await Promise.all([version(0), version(shownVersion(r))]);
  if (rev(S.sel) !== r) return;                  // selection changed while loading
  const echo = S.st.echo;
  const showCol = r.id !== 'R0';
  $('boardSub').textContent = showCol ? `Baseline · ECHO · ${r.id}` : 'Baseline · ECHO reference';
  let h = `<thead><tr><th>Goal</th><th>Base</th>${echo ? '<th>ECHO</th>' : ''}${showCol ? `<th class="shown">${r.id}</th>` : ''}</tr></thead><tbody>`;
  const ncol = 2 + (echo ? 1 : 0) + (showCol ? 1 : 0);
  lastRows = [];
  for (const [title, rows] of boardRows(r)) {
    h += `<tr class="grp"><td colspan="${ncol}"><span class="lbl">${esc(title)}</span></td></tr>`;
    rows.forEach((x, i) => {
      const key = `${x.kind}|${x.roi}|${x.metric}|${i}`;
      lastRows.push({...x, key});
      const inf = info(x.roi);
      const limTxt = (x.limit !== null && x.limit !== undefined) || (x.goal !== null && x.goal !== undefined)
        ? `${opTxt(x.op)} ${x.limit !== null && x.limit !== undefined ? +x.limit.toFixed(2) : ''}${x.goal !== null && x.goal !== undefined ? ` (goal ${x.goal})` : ''} ${unit(x.metric)}` : '';
      const cov = inf.coverage !== null && inf.coverage < 0.999 && inf.available
        ? `<span class="cov" title="Only ${Math.round(inf.coverage * 100)}% of this organ lies in the dose box; values cover that part.">${Math.round(inf.coverage * 100)}% in box</span>` : '';
      const tag = (x.kind === 'guard' ? '<span class="tag">guard</span>' : '')
        + (!inf.available ? `<span class="tag" title="${esc(inf.reason || '')}">${esc(inf.short || 'no dose data')}</span>` : '');
      const locked = S.locks.has(x.roi);
      const on_ = S.organ === x.roi ? ' on' : '';
      h += `<tr class="row${on_}" data-roi="${esc(x.roi)}" data-key="${esc(key)}"><td><div class="gname">
        <span class="roi">${esc(label(x.roi))}</span><span class="met">${esc(x.metric)}</span><span class="lim">${esc(limTxt)}</span>${cov}${tag}
        <span class="ract${locked ? ' has' : ''}">${inf.available ? `<button class="ib${locked ? ' on' : ''}" data-lock="${esc(x.roi)}" title="${locked ? 'Unlock' : 'Lock'} ${esc(label(x.roi))}: the next request must not make it worse" aria-label="Lock ${esc(label(x.roi))}">${ICON_LOCK}</button>
        <button class="ib" data-pen="${esc(key)}" title="Set a target for ${esc(label(x.roi))} ${esc(x.metric)}" aria-label="Set target">${ICON_PEN}</button>` : ''}</span></div></td>`;
      if (!inf.available) {
        h += `<td colspan="${ncol - 1}" class="num cell-na" title="No dose data: ${esc(inf.reason || '')}">no dose data<i class="st na"></i></td></tr>`;
      } else {
        const b = val(base, x.roi, x.metric), e = echo ? val(echo, x.roi, x.metric) : null, v = val(shown, x.roi, x.metric);
        let d = '';
        if (showCol && v !== null && b !== null && Math.abs(v - b) >= 0.005) {
          const better = x.op === '>=' ? v > b : v < b;
          d = `<span class="delta ${better ? 'better' : 'worse'}">${v > b ? '+' : ''}${(v - b).toFixed(2)}</span>`;
        }
        h += `<td class="num">${fmt(b, x.metric)}${dot(b, x, x.roi)}</td>`;
        if (echo) h += `<td class="num">${e === null ? '<span class="cell-na">—</span>' : fmt(e, x.metric) + dot(e, x, x.roi)}</td>`;
        if (showCol) h += `<td class="num">${fmt(v, x.metric)}${dot(v, x, x.roi)}${d}</td>`;
        h += '</tr>';
      }
      if (form === key) {
        const cur = val(shown, x.roi, x.metric);
        const def = cur !== null ? (x.op === '<=' ? Math.max(0, cur - 1) : cur + 0.5).toFixed(1) : '';
        h += `<tr><td colspan="${ncol}" class="tform"><span>${esc(label(x.roi))} ${esc(x.metric)} ${opTxt(x.op)}</span>
          <input id="tval" type="number" step="0.1" value="${def}" aria-label="Target value"><span>${unit(x.metric)}</span>
          <button class="btn primary" data-addtarget="${esc(x.roi)}|${esc(x.metric)}|${esc(x.op)}">Put in the request</button>
          <button class="btn ghost" data-cancel="1">Cancel</button></td></tr>`;
      }
    });
  }
  $('board').innerHTML = h + '</tbody>';
}

export const visibleRows = () => lastRows;

export function targetText(roi, metric, op, v) {
  return `${op === '<=' ? 'Lower' : 'Raise'} the ${label(roi)} ${metric} to ${v} ${unit(metric)}.`;
}

export function initBoard() {
  $('board').addEventListener('click', (e) => {
    const lock = e.target.closest('[data-lock]'), pen = e.target.closest('[data-pen]');
    const add = e.target.closest('[data-addtarget]');
    if (lock) {
      const roi = lock.dataset.lock;
      S.locks.has(roi) ? S.locks.delete(roi) : S.locks.add(roi);
      emit('locks'); renderBoard(); return;
    }
    if (pen) { form = form === pen.dataset.pen ? null : pen.dataset.pen; renderBoard().then(() => $('tval')?.focus()); return; }
    if (add) {
      const [roi, metric, op] = add.dataset.addtarget.split('|');
      emit('fill', targetText(roi, metric, op, $('tval').value));
      form = null; renderBoard(); return;
    }
    if (e.target.closest('[data-cancel]')) { form = null; renderBoard(); return; }
    if (e.target.closest('.tform')) return;
    const row = e.target.closest('tr.row');
    if (!row) return;
    S.organ = S.organ === row.dataset.roi ? null : row.dataset.roi;
    emit('organ', {jump: true});
  });
  $('board').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.id === 'tval') $('board').querySelector('[data-addtarget]')?.click();
    if (e.key === 'Escape' && form) { form = null; renderBoard(); }
  });
  on('select', renderBoard);
  on('revisions', renderBoard);
  on('organ', renderBoard);
}
