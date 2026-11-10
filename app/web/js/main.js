// BeamOpt Workspace: boot, header, resizable columns.
import {$, api, store, toast} from './api.js';
import {S, emit, on, rev} from './state.js';
import {initBoard} from './board.js';
import {initTimeline} from './timeline.js';
import {initViewer} from './viewer3.js';
import {initDvh} from './dvh.js';
import {follow, initCommand} from './command.js';
import {initDrawers} from './drawers.js';

function header() {
  const h = S.st.header;
  $('hPatient').textContent = h.patient;
  const g = h.gantry.length ? `${h.gantry.length} beams, gantry ${h.gantry[0]}–${h.gantry[h.gantry.length - 1]}°` : '';
  $('hMeta').textContent = [h.protocol.replaceAll('_', ' '), `${h.rx_gy} Gy in ${h.fractions} fractions`, g, ...h.notes]
    .filter(Boolean).join(' · ');
  document.title = `${h.patient} · BeamOpt`;
}

function chip() {
  const cur = rev(S.revs.current), w = S.revs.waiting;
  $('curTxt').textContent = `Current plan: ${cur ? cur.id : '?'}${cur?.id === 'R0' ? ' baseline' : ''}`
    + (w && w === S.revs.current ? ' · awaiting review' : '');
  $('curChip').classList.toggle('review', !!w && w === S.revs.current);
}

async function boot() {
  $('loading').hidden = false;
  try {
    S.st = await api('/api/state');
  } catch (e) {
    $('loading').textContent = 'Could not reach the BeamOpt server: ' + e.message;
    return;
  }
  S.versions.clear(); S.live.clear(); S.organ = null;
  S.revs = S.st.revisions;
  S.sel = S.revs.waiting || S.revs.current;
  header();
  emit('state');
  emit('revisions');
  emit('select');
  $('loading').hidden = true;
  if (S.revs.running) follow(S.revs.running);
}

// ---------------------------------------------------------------- resizable columns (remembered per browser)
function splitters() {
  const main = $('main'), MIN = [280, 300, 340], SPL = 14, W = {w1: 0, w2: 0};
  const defaults = () => { const t = main.clientWidth - SPL; return {w1: Math.round(t * 0.29), w2: Math.round(t * 0.3)}; };
  const apply = () => {
    const t = main.clientWidth - SPL;
    W.w1 = Math.max(MIN[0], Math.min(W.w1, t - MIN[1] - MIN[2]));
    W.w2 = Math.max(MIN[1], Math.min(W.w2, t - W.w1 - MIN[2]));
    main.style.setProperty('--w1', W.w1 + 'px'); main.style.setProperty('--w2', W.w2 + 'px');
  };
  const save = () => store.set('cols', [W.w1, W.w2, main.clientWidth]);
  Object.assign(W, defaults());
  const v = store.get('cols', null);
  if (v && v[2]) { const k = main.clientWidth / v[2]; W.w1 = v[0] * k; W.w2 = v[1] * k; }
  apply();
  ['sp1', 'sp2'].forEach((id, i) => {
    const sp = $(id), k = i ? 'w2' : 'w1';
    sp.addEventListener('pointerdown', (e) => {
      e.preventDefault(); sp.setPointerCapture(e.pointerId); sp.classList.add('drag'); document.body.classList.add('resizing');
      const x0 = e.clientX, w0 = W[k];
      const mv = (ev) => { W[k] = w0 + (ev.clientX - x0); apply(); };
      const up = () => { sp.classList.remove('drag'); document.body.classList.remove('resizing');
        sp.removeEventListener('pointermove', mv); sp.removeEventListener('pointerup', up); save(); };
      sp.addEventListener('pointermove', mv); sp.addEventListener('pointerup', up);
    });
    sp.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault(); W[k] += e.key === 'ArrowLeft' ? -24 : 24; apply(); save();
    });
    sp.addEventListener('dblclick', () => { Object.assign(W, defaults()); apply(); save(); });
  });
  window.addEventListener('resize', apply);
}

initBoard();
initTimeline();
initViewer();
initDvh();
initCommand();
initDrawers();
splitters();
on('revisions', chip);
on('reload', boot);
window.addEventListener('unhandledrejection', (e) => toast(String(e.reason?.message || e.reason), true));
boot();
