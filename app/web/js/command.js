// Command bar: instruction input, context chips (locks, starting plan), suggestions, Send / Stop, dictation,
// and following a running request over server-sent events.
import {$, api, esc, toast} from './api.js';
import {S, emit, label, on, refreshRevisions, rev, select} from './state.js';

const BASE_SUGG = ['What are the current esophagus mean dose and PTV D95?', 'Reduce the esophagus mean dose by 1.5 Gy.',
                   'Lower the cord max dose by 2 Gy.', 'Undo the last change.'];
let es = null;

function busy() { return !!S.revs?.running; }

export function renderCommand() {
  const run = busy();
  $('send').hidden = run; $('stop').hidden = !run;
  $('send').disabled = run;
  const cur = rev(S.revs?.current);
  const chips = [];
  if (cur) chips.push(`<span class="ctxchip plain">Starts from ${esc(cur.id)}${cur.id === 'R0' ? ' (baseline)' : ''} · v${S.revs.current_version}</span>`);
  for (const r of S.locks) chips.push(`<span class="ctxchip">Locked: ${esc(label(r))}<button data-unlock="${esc(r)}" aria-label="Unlock ${esc(label(r))}">×</button></span>`);
  $('ctx').innerHTML = chips.join('');
  const r = rev(S.sel), sugg = [...BASE_SUGG];
  if (r && ['not_met', 'partial'].includes(r.outcome)) sugg.unshift(`Why did ${r.id} not reach the goal?`);
  const aim = r && r.outcome === 'not_reachable' && /aim for (\S+ \S+ <= [\d.]+ (?:Gy|%))/.exec(r.notes || '');
  if (aim) sugg.unshift(`Lower ${aim[1].replace('<=', 'to')}.`);
  $('sugg').innerHTML = `<span class="lbl">Try</span>${sugg.slice(0, 4).map((s) => `<button data-s="${esc(s)}">${esc(s)}</button>`).join('')}<span class="hint">Each optimizer solve takes about 30–60 s</span>`;
}

async function send() {
  const text = $('cmd').value.trim();
  if (!text || busy()) return;
  $('send').disabled = true;
  try {
    const out = await api('/api/turn', {body: {text, locks: [...S.locks]}});
    $('cmd').value = '';
    S.live.set(out.revision, {steps: [], elapsed: 0});
    await refreshRevisions();
    select(out.revision);
    follow(out.revision);
  } catch (e) { toast(e.message, true); }
  renderCommand();
}

export function follow(rid) {
  if (es) es.close();
  const live = S.live.get(rid) || {steps: [], elapsed: 0};
  S.live.set(rid, live);
  es = new EventSource(`/api/events?since=${live.steps.length}`);
  es.addEventListener('step', (e) => { live.steps.push(JSON.parse(e.data)); emit('live', rid); });
  es.addEventListener('beat', (e) => { live.elapsed = JSON.parse(e.data).elapsed; emit('live', rid); });
  es.addEventListener('done', async () => { es.close(); es = null; await finished(rid); });
  es.addEventListener('idle', async () => { es.close(); es = null; await finished(rid); });
  es.onerror = () => {
    if (!es) return;
    es.close(); es = null;
    setTimeout(async () => { await refreshRevisions(); if (S.revs.running) follow(S.revs.running); else finished(rid); }, 1500);
  };
}

async function finished(rid) {
  await refreshRevisions();
  const r = rev(rid);
  if (r) {
    select(rid);
    const t = {review: `${rid} is ready for review: Keep or Discard it, or just continue with a follow-up.`,
               no_change: `${rid}: the plan is unchanged.`, stopped: `${rid} stopped. The plan is unchanged.`,
               error: `${rid} failed. The plan is unchanged.`, kept: `${rid} done.`}[r.status];
    if (t) toast(t, r.status === 'error');
  }
  renderCommand();
}

// ---------------------------------------------------------------- dictation
let rec = null, chunks = [];
async function toggleMic() {
  const b = $('mic');
  if (rec) { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    rec = new MediaRecorder(stream);
    chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      b.classList.remove('rec'); rec = null;
      const fd = new FormData();
      fd.append('audio', new Blob(chunks, {type: 'audio/webm'}), 'dictation.webm');
      toast('Transcribing…', false, 2500);
      try {
        const out = await api('/api/transcribe', {body: fd});
        const cur = $('cmd').value.trim();
        $('cmd').value = [cur, out.text].filter(Boolean).join(' ');
        $('cmd').focus();
      } catch (e) { toast(e.message, true); }
    };
    rec.start();
    b.classList.add('rec');
  } catch (e) { toast('Microphone unavailable: ' + e.message, true); }
}

export function initCommand() {
  $('send').addEventListener('click', send);
  $('cmd').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
  $('stop').addEventListener('click', async () => {
    try { toast((await api('/api/stop', {method: 'POST'})).message); } catch (e) { toast(e.message, true); }
  });
  $('mic').addEventListener('click', toggleMic);
  $('sugg').addEventListener('click', (e) => { const b = e.target.closest('[data-s]'); if (b) { $('cmd').value = b.dataset.s; $('cmd').focus(); } });
  $('ctx').addEventListener('click', (e) => {
    const b = e.target.closest('[data-unlock]');
    if (b) { S.locks.delete(b.dataset.unlock); emit('locks'); }
  });
  on('fill', (text) => { $('cmd').value = text; $('cmd').focus(); });
  on('locks', renderCommand);
  on('revisions', renderCommand);
  on('select', renderCommand);
}
