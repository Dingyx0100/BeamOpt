// Toolbar drawers: Details (metrics, protocol, scorecard), Deliverability, Export, Settings.
import {$, api, esc, store, toast} from './api.js';
import {S, emit, label, on, parentVersion, rev, shownVersion} from './state.js';

let open = null;
const TITLES = {why: 'Why this dose', details: 'Details', deliv: 'Deliverability', export: 'Export', settings: 'Settings'};

function md(s) {          // the few markdown bits the backend notes use: **bold**, `code`, "- " lists, line breaks
  const lines = esc(s || '').replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>').split('\n');
  let out = '', inList = false;
  for (const l of lines) {
    if (l.startsWith('- ')) { if (!inList) { out += '<ul>'; inList = true; } out += `<li>${l.slice(2)}</li>`; continue; }
    if (inList) { out += '</ul>'; inList = false; }
    if (l.trim()) out += `<p>${l.replace(/&lt;span class=&quot;hint&quot;&gt;|&lt;\/span&gt;/g, '')}</p>`;
  }
  return out + (inList ? '</ul>' : '');
}

function table(rows) {
  if (!rows?.length) return '<p class="muted">Nothing to show.</p>';
  const cols = Object.keys(rows[0]);
  const cell = (v) => (typeof v === 'number' ? `<td class="n">${Number.isInteger(v) ? v : v.toFixed(2)}</td>`
    : `<td>${esc(v ?? '')}</td>`);
  return `<div class="tbl"><table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${
    rows.map((r) => `<tr>${cols.map((c) => cell(r[c])).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function ids() {
  const r = rev(S.sel);
  return {r, vid: shownVersion(r), cmp: parentVersion(r)};
}

function whyOrgan(r) {
  if (S.organ && S.st.rois[S.organ]?.available) return S.organ;
  const obj = (r?.checks || []).find((c) => c.role === 'objective' && S.st.rois[c.roi]?.available);
  if (obj) return obj.roi;
  return ['LUNGS_NOT_GTV', 'ESOPHAGUS', 'CORD', 'PTV'].find((x) => S.st.rois[x]?.available) || 'PTV';
}

function beamBars(m, hasParent) {
  if (!m?.beams?.length) return '<p class="muted">No dose data for this organ.</p>';
  const rows = [...m.beams].sort((a, b) => (a.gantry ?? a.beam_id) - (b.gantry ?? b.beam_id));
  const hi = Math.max(...rows.map((b) => Math.max(b.gy, b.parent_gy ?? 0)), 1e-6);
  const top = Math.max(...rows.map((b) => b.share));
  return `<div class="bars">${rows.map((b) => {
    const d = hasParent && b.parent_gy != null ? b.gy - b.parent_gy : null;
    const dTxt = d === null ? '' : `<span class="num ${d > 0.05 ? 'worse' : d < -0.05 ? 'better' : 'muted'}">${d > 0 ? '+' : ''}${d.toFixed(2)}</span>`;
    return `<div class="bar-row${b.share === top ? ' lead' : ''}"><span class="bl">${b.gantry != null ? `Gantry ${b.gantry}°` : `Beam ${esc(b.beam_id)}`}</span>
      <span class="bt"><i style="width:${(100 * b.gy / hi).toFixed(1)}%"></i>${b.parent_gy != null ? `<s style="left:${(100 * b.parent_gy / hi).toFixed(1)}%" title="Parent plan: ${b.parent_gy.toFixed(2)} Gy"></s>` : ''}</span>
      <span class="num">${b.gy.toFixed(2)} Gy</span><span class="num muted">${Math.round(b.share * 100)}%</span>${dTxt}</div>`;
  }).join('')}</div>`;
}

async function why() {
  const {r, vid, cmp} = ids();
  const roi = whyOrgan(r);
  const d = await api(`/api/explain/${vid}?roi=${encodeURIComponent(roi)}${cmp !== null ? `&cmp=${cmp}` : ''}`);
  const organs = Object.keys(S.st.rois).filter((x) => S.st.rois[x].available && !['BODY', 'SKIN'].includes(x));
  const mean = d.metrics.Dmean, hot = d.metrics.D2;
  const part = mean.coverage < 0.999 ? ` Only ${Math.round(mean.coverage * 100)}% of this organ is in the dose box; the numbers cover that part.` : '';
  const pull = d.binding.some((b) => b.pull_share != null);
  const bind = d.binding.length ? `<div class="tbl"><table><thead><tr><th>Limit</th><th>Set by</th><th class="n">Now</th>${pull ? '<th class="n">Pull</th>' : ''}</tr></thead><tbody>${
    d.binding.map((b) => `<tr><td>${esc(label(b.roi))} ${b.kind === 'max' ? 'max' : 'mean'} ≤ ${b.limit_gy} Gy</td><td>${b.source === 'protocol' ? 'Protocol' : 'Agent'}</td><td class="n">${b.value_gy.toFixed(2)} Gy</td>${pull ? `<td class="n">${b.pull_share != null ? Math.round(b.pull_share * 100) + '%' : '—'}</td>` : ''}</tr>`).join('')}</tbody></table></div>${
    pull ? '' : '<p class="muted">These limits are at their bound. Their pull appears once the plan is re-optimized in this session (a cached plan has no solver data).</p>'}`
    : '<p class="muted">No hard limit is active. The plan is shaped by the objective weights alone.</p>';
  return `<div class="form"><label for="wSel">Organ</label><select id="wSel">${organs.map((x) => `<option value="${esc(x)}" ${x === roi ? 'selected' : ''}>${esc(label(x))}</option>`).join('')}</select><span></span></div>
    <p class="muted">${esc(r.id)} (plan v${vid})${cmp !== null ? `; the tick on each bar is the parent plan v${cmp}` : ''}.${part}</p>
    <h4>Mean dose by beam <span class="muted">· ${mean.mean_gy?.toFixed(2) ?? '—'} Gy over the ${esc(mean.region || '')}</span></h4>${beamBars(mean, cmp !== null)}
    <h4>Hot spot by beam <span class="muted">· mean ${hot.mean_gy?.toFixed(2) ?? '—'} Gy over the ${esc(hot.region || '')}</span></h4>${beamBars(hot, cmp !== null)}
    <h4>Limits holding the plan back</h4>${bind}
    <p class="muted">Beam doses are exact: the dose matrix gives what each beam delivers to every voxel. Pull comes from the optimizer's dual values. It ranks which limit costs the plan most, but it is not a dose trade-off in Gy.</p>`;
}

async function details() {
  const {r, vid, cmp} = ids();
  const d = await api(`/api/details/${vid}${cmp !== null ? `?cmp=${cmp}` : ''}`);
  return `<p class="muted">${esc(r.id)} (plan v${vid})${cmp !== null ? ` compared with its parent v${cmp}` : ''}. ECHO is the benchmark plan with the same beams.</p>
    <h4>Key metrics</h4>${table(d.metrics)}
    <h4>Protocol criteria</h4>${md(d.criteria_note)}${table(d.criteria)}
    <h4>Plan-quality scorecard</h4>${md(d.scorecard_note)}${table(d.scorecard)}`;
}

function fluence(maps) {
  const out = [];
  for (const [g, m] of Object.entries(maps)) {
    let hi = 0;
    for (const row of m) for (const v of row) if (v !== null && v > hi) hi = v;
    out.push({g, m, hi: hi || 1});
  }
  return out;
}

async function deliv() {
  const {r, vid, cmp} = ids();
  const d = await api(`/api/deliverability/${vid}${cmp !== null ? `?cmp=${cmp}` : ''}`);
  const maps = fluence(d.maps);
  setTimeout(() => {
    for (const {g, m, hi} of maps) {
      const cv = document.getElementById('fl' + g);
      if (!cv) continue;
      const h = m.length, w = m[0]?.length || 1;
      cv.width = w; cv.height = h;
      const x = cv.getContext('2d'), im = x.createImageData(w, h);
      m.forEach((row, j) => row.forEach((v, i) => {
        const q = (j * w + i) * 4;
        const t = v === null ? 0 : Math.min(1, v / hi);
        im.data[q] = 255 * t; im.data[q + 1] = 210 * t; im.data[q + 2] = 120 * t; im.data[q + 3] = v === null ? 0 : 255;
      }));
      x.putImageData(im, 0, 0);
    }
  });
  return `<p class="muted">${esc(r.id)} (plan v${vid}).</p>${md(d.note)}
    <h4>Per beam</h4>${table(d.beams)}<h4>Key metrics: optimized vs after leaf sequencing</h4>${table(d.fidelity)}
    <h4>Optimized fluence, beam's-eye view</h4><div class="fluence">${maps.map(({g}) => `<figure><canvas id="fl${g}"></canvas><figcaption>Gantry ${g}°</figcaption></figure>`).join('')}</div>`;
}

function exportPanel() {
  const {r, vid} = ids();
  return `<p>Export writes the plan you are viewing: RTDOSE on the CT grid, the optimizer settings, the fluence and the session report. A copy is kept under <code>runs/exports</code>.</p>
    <div class="acts"><a class="btn primary" href="/api/export/${vid}" download>Download ${esc(r.id)} (plan v${vid}) as DICOM .zip</a>
    <a class="btn" href="/api/report" download>Download the session report (HTML)</a></div>
    <p class="muted">The session report lists every request, optimizer change, check and trade-off. No RTPLAN is included: a deliverable plan needs leaf sequencing. Research prototype, not for clinical use.</p>`;
}

async function settings() {
  const [pts, ss, models] = await Promise.all([api('/api/patients'), api('/api/sessions'), api('/api/models')]);
  const theme = store.get('theme', 'system');
  return `<h4>Patient</h4><div class="form">
      <label for="pDir">Data folder</label><input id="pDir" type="text" value="${esc(pts.dir)}"><button class="btn" data-browse="pDir">Browse…</button>
      <label for="pSel">Patient</label><select id="pSel">${pts.patients.map((p) => `<option value="${esc(p.id)}" ${p.id === pts.current ? 'selected' : ''} ${p.usable && p.supported ? '' : 'disabled'}>${esc(p.label)}${p.problem ? ' · ' + esc(p.problem) : ''}</option>`).join('')}</select><button class="btn primary" id="pLoad">Load</button>
      <span class="msg" id="pMsg">Loading a patient reads its influence matrix; the first load also solves the protocol baseline (about a minute).</span></div>
    <h4>Saved sessions</h4><div class="form">
      <label for="sDir">Sessions folder</label><input id="sDir" type="text" value="${esc(ss.dir)}"><button class="btn" data-browse="sDir">Browse…</button>
      <label for="sSel">Session</label><select id="sSel">${ss.sessions.map((s) => `<option value="${esc(s.path)}">${esc(s.label)}</option>`).join('') || '<option disabled>No saved sessions</option>'}</select><button class="btn primary" id="sLoad" ${ss.sessions.length ? '' : 'disabled'}>Restore</button>
      <span class="msg" id="sMsg">Every solved plan is saved automatically. Restoring recomputes doses from the saved fluence; new requests continue in the same session.</span></div>
    <h4>Language model</h4><div class="form"><label for="mSel">Model</label><select id="mSel">${models.models.map((m) => `<option ${m === models.current ? 'selected' : ''}>${esc(m)}</option>`).join('')}</select><span></span>
      <span class="msg">The agent runs on this machine through Ollama. ${esc(models.validated)} is the validated default; others are for experiments.</span></div>
    <h4>Appearance</h4><div class="form"><label for="tSel">Theme</label><select id="tSel">${['system', 'dark', 'light'].map((t) => `<option ${t === theme ? 'selected' : ''}>${t}</option>`).join('')}</select><span></span></div>`;
}

async function show(kind) {
  open = kind;
  document.querySelectorAll('#toolbar .tbtn').forEach((b) => b.classList.toggle('on', b.dataset.drawer === kind));
  $('drawer').hidden = false;
  $('dTitle').textContent = TITLES[kind];
  $('dBody').innerHTML = '<p class="muted">Loading…</p>';
  try {
    const html = kind === 'why' ? await why() : kind === 'details' ? await details() : kind === 'deliv' ? await deliv()
      : kind === 'export' ? exportPanel() : await settings();
    if (open === kind) $('dBody').innerHTML = html;
  } catch (e) { $('dBody').innerHTML = `<p>${esc(e.message)}</p>`; }
}

function close() {
  open = null; $('drawer').hidden = true;
  document.querySelectorAll('#toolbar .tbtn').forEach((b) => b.classList.remove('on'));
}

async function busyCall(btn, msgEl, text, fn) {
  btn.disabled = true; $(msgEl).textContent = text;
  try { const out = await fn(); $(msgEl).textContent = out.message || 'Done.'; return out; }
  catch (e) { $(msgEl).textContent = e.message; toast(e.message, true); return null; }
  finally { btn.disabled = false; }
}

export function initDrawers() {
  $('toolbar').addEventListener('click', (e) => {
    const b = e.target.closest('[data-drawer]');
    if (!b) return;
    open === b.dataset.drawer ? close() : show(b.dataset.drawer);
  });
  $('dClose').addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && open) close(); });
  $('dBody').addEventListener('click', async (e) => {
    const br = e.target.closest('[data-browse]');
    if (br) {
      const inp = $(br.dataset.browse);
      try {
        const out = await api('/api/browse', {body: {current: inp.value, title: br.dataset.browse === 'pDir' ? 'Choose a PortPy data folder' : 'Choose a saved-sessions folder'}});
        if (out.path) { inp.value = out.path; inp.dispatchEvent(new Event('change')); }
      } catch (err) { toast(err.message, true); }
    }
    if (e.target.id === 'pLoad') {
      const out = await busyCall(e.target, 'pMsg', 'Loading… (about a minute the first time)', () =>
        api('/api/load', {body: {patient: $('pSel').value, dir: $('pDir').value}}));
      if (out) { close(); emit('reload'); }
    }
    if (e.target.id === 'sLoad') {
      const out = await busyCall(e.target, 'sMsg', 'Restoring…', () =>
        api('/api/restore', {body: {path: $('sSel').value, data_dir: $('pDir').value}}));
      if (out) { close(); emit('reload'); }
    }
  });
  $('dBody').addEventListener('change', async (e) => {
    if (e.target.id === 'pDir') {
      const pts = await api(`/api/patients?dir=${encodeURIComponent(e.target.value)}`).catch((err) => toast(err.message, true));
      if (pts) $('pSel').innerHTML = pts.patients.map((p) => `<option value="${esc(p.id)}" ${p.usable && p.supported ? '' : 'disabled'}>${esc(p.label)}</option>`).join('');
    }
    if (e.target.id === 'sDir') {
      const ss = await api(`/api/sessions?dir=${encodeURIComponent(e.target.value)}`).catch((err) => toast(err.message, true));
      if (ss) { $('sSel').innerHTML = ss.sessions.map((s) => `<option value="${esc(s.path)}">${esc(s.label)}</option>`).join(''); $('sLoad').disabled = !ss.sessions.length; }
    }
    if (e.target.id === 'wSel') { S.organ = e.target.value; emit('organ', {jump: true}); return; }
    if (e.target.id === 'mSel') {
      try { await api('/api/model', {body: {model: e.target.value}}); toast(`The agent now uses ${e.target.value}.`); }
      catch (err) { toast(err.message, true); }
    }
    if (e.target.id === 'tSel') {
      const t = e.target.value;
      store.set('theme', t);
      if (t === 'system') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
      emit('theme');
    }
  });
  on('select', () => { if (['why', 'details', 'deliv', 'export'].includes(open)) show(open); });
  on('organ', () => { if (open === 'why') show('why'); });
}
