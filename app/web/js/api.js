// Server calls and small shared helpers.

export async function api(path, opts = {}) {
  const init = {method: opts.method || (opts.body ? 'POST' : 'GET'), headers: {}};
  if (opts.body instanceof FormData) init.body = opts.body;
  else if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers['Content-Type'] = 'application/json'; }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

// Volumes arrive gzip + base64 (see app/volume_viewer.py); data: URL decoding is the fast path in browsers.
export async function unpack(b64, Kind) {
  const blob = await (await fetch('data:application/octet-stream;base64,' + b64)).blob();
  const stream = blob.stream().pipeThrough(new DecompressionStream('gzip'));
  return new Kind(await new Response(stream).arrayBuffer());
}

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
export const $ = (id) => document.getElementById(id);
export const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));

export function toast(text, err = false, ms = 4200) {
  const el = document.createElement('div');
  el.className = 'toast' + (err ? ' err' : '');
  el.textContent = text;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

export function fmt(v, metric) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return metric && String(metric).startsWith('V') ? v.toFixed(1) : v.toFixed(2);
}
export const unit = (m) => (String(m).startsWith('V') ? '%' : 'Gy');
export const opTxt = (o) => (o === '<=' ? '≤' : '≥');

export function status(v, op, lim, goal) {
  if (v === null || v === undefined) return 'na';
  const ok = (t) => (op === '<=' ? v <= t + 1e-6 : v >= t - 1e-6);
  if (goal !== null && goal !== undefined) { if (ok(goal)) return 'pass'; return lim !== null && lim !== undefined && ok(lim) ? 'warn' : 'fail'; }
  if (lim === null || lim === undefined) return 'na';
  return ok(lim) ? 'pass' : 'fail';
}

export function timeOf(t) {
  if (!t) return '';
  const d = new Date(t * 1000);
  return d.toTimeString().slice(0, 5);
}

export const store = {
  get(k, d) { try { const v = localStorage.getItem('beamopt.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('beamopt.' + k, JSON.stringify(v)); } catch (e) { /* storage off */ } },
};
