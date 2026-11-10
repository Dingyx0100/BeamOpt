// Shared app state and a tiny event bus. Modules read S and subscribe with on('name', fn).
import {api} from './api.js';

export const S = {
  st: null,                 // /api/state payload
  revs: null,               // revision payload {items, current, waiting, running, current_version}
  sel: null,                // id of the revision being shown
  organ: null,              // selected ROI (board row / DVH curve / viewer)
  locks: new Set(),         // ROIs the next request must not make worse
  versions: new Map(),      // version id -> /api/version payload (cached)
  live: new Map(),          // revision id -> {steps: [], elapsed}
  dvhHidden: new Set(),     // DVH curves switched off
};

const subs = new Map();
export function on(name, fn) { (subs.get(name) || subs.set(name, []).get(name)).push(fn); }
export function emit(name, arg) { for (const fn of subs.get(name) || []) fn(arg); }

export const rev = (id) => S.revs?.items.find((r) => r.id === id) || null;
export const label = (roi) => S.st?.roi_labels?.[roi] || roi;
export const color = (roi) => S.st?.roi_colors?.[roi] || '#9aa3b2';

// Which plan a revision shows, and which it is compared with ("parent").
export function shownVersion(r) {
  if (!r) return 0;
  if (r.status === 'working') return r.start_version;
  return r.final_version ?? r.start_version;
}
export function parentVersion(r) {
  if (!r || r.id === 'R0') return null;
  return r.start_version;
}

export async function version(vid) {
  if (vid === null || vid === undefined) return null;
  if (!S.versions.has(vid)) S.versions.set(vid, api(`/api/version/${vid}`));
  try { return await S.versions.get(vid); } catch (e) { S.versions.delete(vid); throw e; }
}

export async function refreshRevisions() {
  S.revs = await api('/api/revisions');
  if (!rev(S.sel)) S.sel = S.revs.waiting || S.revs.current;
  emit('revisions');
}

export async function select(rid) {
  S.sel = rid;
  emit('select');
}
