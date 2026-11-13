// Revision timeline: one card per request with its result, cost, attempts and Keep / Discard / Branch.
import {$, api, esc, fmt, opTxt, timeOf, toast, unit} from './api.js';
import {S, emit, label, on, refreshRevisions, rev, select} from './state.js';

const PILL = {working: 'Working', review: 'Awaiting review', kept: 'Kept', discarded: 'Discarded',
              set_aside: 'Set aside', no_change: 'No change', stopped: 'Stopped', error: 'Error'};
const OUTCOME = {
  met: 'Goal met.',
  needs_decision: 'Goal met, but at a cost to an organ you did not mention. Keep it to accept the cost, or Discard.',
  partial: 'Partly met. The best attempt is kept for your review.',
  not_met: 'Not met. The plan is unchanged.',
  not_reachable: 'Not reachable while protecting the PTV. The plan is unchanged; see the lowest value found below.', limit_changed: 'Limit changed and the plan re-optimized.',
  limit_failed: 'The new limit broke a protection, so the plan was put back.',
  limit_refused: 'No change.', refused: 'No change.', unavailable: 'No change: no dose data for this organ.',
  answered: 'Answered. The plan is unchanged.', already_met: 'Already met by the current plan. Nothing to change.',
  revert: 'Plan restored.', reset: 'Plan reset to the protocol settings.', undo: 'Last change undone.',
  clarify: 'The agent needs a bit more detail.', restored: 'Plan restored from a saved session.', no_change: 'No change.', error: 'Something went wrong.',
};
const REPLY_FIRST = new Set(['answered', 'refused', 'unavailable', 'limit_refused', 'clarify', 'already_met',
                             'limit_failed', null, undefined]);

function checkRow(c) {
  const a = c.ref, b = c.actual;
  const s = c.passed === true ? 'pass' : c.passed === false ? 'fail' : null;
  return `<tr><td>${esc(label(c.roi))} <span class="k">${esc(c.metric)} ${opTxt(c.op)} ${+(+c.threshold).toFixed(2)}${c.role === 'guard' ? ' · guard' : ''}</span></td>
    <td class="num">${fmt(a, c.metric)} → ${fmt(b, c.metric)}</td><td>${s ? `<i class="st ${s}"></i>` : ''}</td></tr>`;
}

function attemptsHtml(r) {
  if (!r.attempts?.length) return '';
  const li = r.attempts.map((a) => {
    const what = a.applied.length ? a.applied.filter((x) => !x.startsWith('(note')).join('; ')
      : (a.errors[0] ? 'rejected: ' + a.errors[0] : 'no change applied');
    let v;
    if (!a.applied.length) v = '<span class="v bad">not solved</span>';
    else if (a.violations.length) v = `<span class="v bad">broke ${esc(a.violations.join(', '))}</span>`;
    else if (a.objectives_met) v = '<span class="v ok">met the goal, held all limits</span>';
    else v = `<span class="v ok">held all limits${a.progress != null ? `, reached ${Math.round(a.progress * 100)}% of the change` : ''}</span>`;
    return `<li><span class="n">${a.n}</span><span>${esc(what)}${a.auto_adjusted ? ' <span class="tag">auto-adjusted repeat</span>' : ''}<br>${v}</span></li>`;
  }).join('');
  return `<details><summary>Attempts (${r.attempts.length})</summary><ul class="att">${li}</ul></details>`;
}

// Engine text names organs by their structure id (ESOPHAGUS); show the names used everywhere else in the app.
function plain(t) {
  const names = S.st?.roi_labels || {};
  let out = String(t || '');
  for (const id of Object.keys(names).sort((a, b) => b.length - a.length)) out = out.replaceAll(id, names[id]);
  return out.replaceAll('<=', '≤').replaceAll('>=', '≥');
}

const aimOf = (p) => Math.ceil((p.floor + 0.1) * 10) / 10;

function reachHtml(r) {
  const un = (r.reach || []).filter((p) => p.verdict === 'unreachable' && p.floor !== null && p.floor !== undefined);
  if (!un.length) return '';
  const now = (p) => (r.checks || []).find((c) => c.roi === p.roi && c.metric === p.metric)?.actual;
  const rows = un.map((p) => `<tr><td>${esc(label(p.roi))} <span class="k">${esc(p.metric)}</span></td>
    <td class="num">${fmt(p.goal, p.metric)}</td><td class="num floor">${p.bound ? '≈ ' : ''}${fmt(p.floor, p.metric)}</td>
    <td class="num">${fmt(now(p), p.metric)}</td></tr>`).join('');
  const why = un.map((p) => plain(p.text)).join(' ');
  const cav = /Caveat: (.*)$/.exec(r.notes || '');
  const tries = un.map((p) => {
    const t = `Lower ${label(p.roi)} ${p.metric} to ${aimOf(p)} ${unit(p.metric)}.`;
    return `<button class="btn" data-fill="${esc(t)}" title="Puts this instruction in the command bar; it is not sent">Try ${esc(p.metric)} ${opTxt('<=')} ${aimOf(p)} ${unit(p.metric)}${p.bound ? ' or higher' : ''}</button>`;
  }).join('');
  return `<div class="block"><span class="lbl">What is reachable</span><table class="res reach">
    <tr class="hd"><td></td><td>Your goal</td><td>Lowest possible</td><td>Now</td></tr>${rows}</table></div>
    <div class="why note">${esc(why)}${cav ? `<br><span class="muted">${esc(plain(cav[1]))}</span>` : ''}</div>
    <div class="acts">${tries}</div>`;
}

function optState(o) {
  const orgs = [...new Set((o.violations || []).map((v) => v.split(' ')[0]))].map((x) => label(x));
  const brk = orgs.length > 2 ? `${orgs.length} organ limits` : orgs.join(' and ') + (orgs.length > 1 ? ' limits' : ' limit');
  if (o.objectives_met && o.protections_ok) return ['pass', 'Meets the goal, holds every limit'];
  if (o.objectives_met) return ['warn', `Meets the goal, breaks ${brk}`];
  if (o.protections_ok) return ['warn', `${Math.round(o.progress * 100)}% of the change, holds every limit`];
  return ['fail', `${Math.round(o.progress * 100)}% of the change, breaks ${brk}`];
}

function optionsHtml(r) {
  const os = r.options || [];
  if (os.length < 2) return '';
  const keys = [];
  const add = (roi, metric) => { if (!keys.some((k) => k[0] === roi && k[1] === metric)) keys.push([roi, metric]); };
  for (const c of os[0].rows) if (c.role === 'objective') add(c.roi, c.metric);
  add('PTV', 'D95');
  for (const o of os) for (const c of o.cost) add(c.roi, c.metric);
  for (const o of os) for (const c of o.rows) if (c.role !== 'objective' && !c.passed) add(c.roi, c.metric);
  const cell = (o, roi, metric) => {
    const c = o.checks.find((x) => x.roi === roi && x.metric === metric);
    const t = o.tradeoffs.find((x) => x.roi === roi && x.metric === metric);
    const v = c?.actual ?? t?.after;
    let cls = '';
    if (c && c.role === 'objective') cls = c.passed ? 'better' : '';
    else if ((c && c.passed === false) || t?.flag === 'worse') cls = 'worse';
    return `<td class="num ${cls}">${fmt(v, metric)}</td>`;
  };
  const start = (roi, metric) => {
    for (const o of os) {
      const t = o.tradeoffs.find((x) => x.roi === roi && x.metric === metric);
      if (t) return t.before;
      const c = o.checks.find((x) => x.roi === roi && x.metric === metric);
      if (c && c.ref !== undefined) return c.ref;
    }
    return null;
  };
  const goal = (roi, metric) => os[0].rows.find((c) => c.role === 'objective' && c.roi === roi && c.metric === metric);
  const busy = !!S.revs.running;
  const last = S.revs.items[S.revs.items.length - 1]?.id === r.id;
  const head = os.map((o, i) => `<th><span class="on">${i + 1}</span>${esc(plain(o.title))}</th>`).join('');
  const state = os.map((o) => { const [k, t] = optState(o); return `<td><span class="ost ${k}">${esc(t)}</span></td>`; }).join('');
  const body = keys.map(([roi, metric]) => {
    const g = goal(roi, metric);
    return `<tr><td>${esc(label(roi))} <span class="k">${esc(metric)}${g ? ` ${opTxt(g.op)} ${+(+g.threshold).toFixed(2)}` : ''}</span></td>
      <td class="num muted">${fmt(start(roi, metric), metric)}</td>${os.map((o) => cell(o, roi, metric)).join('')}</tr>`;
  }).join('');
  const use = os.map((o, i) => `<td>${o.chosen ? '<span class="tag">In use</span>'
    : last ? `<button class="btn" data-opt="${esc(o.key)}" data-n="${i + 1}" data-rid="${r.id}" ${busy ? 'disabled' : ''}>Use ${i + 1}</button>` : ''}</td>`).join('');
  return `<div class="block"><span class="lbl">Options · each one is a solved plan</span><div class="optwrap"><table class="opts">
    <tr><th></th><th class="muted">Start</th>${head}</tr><tr class="ostrow"><td></td><td></td>${state}</tr>${body}
    <tr class="userow"><td></td><td></td>${use}</tr></table></div></div>`;
}

function steps(r) {
  const live = S.live.get(r.id) || {steps: []};
  const seen = [];
  for (const s of live.steps) if (!seen.includes(s.stage)) seen.push(s.stage);
  if (!seen.length) seen.push('Reading request');
  const solves = live.steps.filter((s) => s.node === 'optimize').length;
  return `<div class="steps">${seen.map((s, i) => `<span class="step ${i < seen.length - 1 ? 'done' : 'now'}">${esc(s)}</span>`).join('')}
    <span class="elapsed">${solves ? `${solves} solve${solves > 1 ? 's' : ''} · ` : ''}${live.elapsed ? `${live.elapsed} s` : ''}</span></div>`;
}

function card(r) {
  const cur = S.revs.current === r.id;
  let pill = PILL[r.status] || r.status;
  if (r.id === 'R0') pill = 'Baseline';
  const sel = S.sel === r.id ? ' sel' : '';
  const dim = ['discarded', 'set_aside', 'stopped', 'error'].includes(r.status) ? ' dim' : '';
  let b = '<div class="cb">';
  if (r.id === 'R0') {
    const h = S.st.header;
    b += `<div class="muted">Protocol objectives for ${esc(h.protocol.replaceAll('_', ' '))}, ${h.gantry.length || ''} planner beams, solved by the optimizer without the agent.</div>`;
  } else {
    b += `<div class="req">${esc(r.request)}</div>`;
    if (r.locks?.length) b += `<div class="muted">Locked: ${r.locks.map((x) => esc(label(x))).join(', ')}</div>`;
  }
  if (r.status === 'working') b += steps(r);
  else if (r.id !== 'R0') {
    let out = OUTCOME[r.outcome] || OUTCOME[r.status] || '';
    const n = r.attempts?.length || 0;
    if (r.outcome === 'met' && n) out = `Goal met in ${n} attempt${n > 1 ? 's' : ''}.`;
    const byOpt = r.outcome === 'met' && /^Met by the option/.test(r.notes || '');
    if (byOpt) {
      const i = (r.options || []).findIndex((o) => o.chosen);
      out = `Goal met by option ${i + 1} after ${n} attempt${n > 1 ? 's' : ''} fell short. Keep it, pick another option, or Discard.`;
    }
    if (r.status === 'discarded') out = 'Discarded. The plan went back to where this request started.';
    if (r.status === 'set_aside') out = 'Set aside when you branched from another plan. You can still branch from it.';
    if (r.kept_reason === 'continued') out += ' Kept because you continued from it.';
    if (r.status === 'stopped' || r.status === 'error') out = r.reply;
    if (r.chosen) {
      const i = (r.options || []).findIndex((o) => o.key === r.chosen);
      if (i >= 0) out = `You picked option ${i + 1}: ${plain(r.options[i].title)}.` + (r.status === 'review' ? ' Keep it or Discard.' : '');
    }
    if (out) b += `<div>${esc(out)}</div>`;
    const replyFirst = REPLY_FIRST.has(r.outcome) && r.reply && r.status !== 'stopped' && r.status !== 'error';
    if (replyFirst) b += `<div class="why note">${esc(r.reply)}</div>`;
    if (r.changes?.length) {
      b += `<div class="block"><span class="lbl">Changes</span><ul class="chg">${r.changes.map((c) =>
        `<li><span>${esc(c.what)}${c.detail ? ` <span class="muted">${esc(c.detail)}</span>` : ''}${c.tag ? ` <span class="tag">${esc(c.tag)}</span>` : ''}</span></li>`).join('')}</ul></div>`;
    }
    const unreachable = r.outcome === 'not_reachable' && (r.reach || []).some((p) => p.verdict === 'unreachable');
    const checks = (r.checks || []).filter((c) => c.actual !== null && c.actual !== undefined);
    if (unreachable) b += reachHtml(r);
    else if (checks.length && !replyFirst) {
      b += `<div class="block"><span class="lbl">Result vs ${esc(r.parent || 'start')}</span><table class="res">${checks.map(checkRow).join('')}</table></div>`;
    }
    const worse = (r.tradeoffs || []).filter((t) => t.flag === 'worse');
    if (worse.length) {
      b += `<div class="block"><span class="lbl">Cost</span><table class="res">${worse.map((t) =>
        `<tr><td>${esc(label(t.roi))} <span class="k">${esc(t.metric)}${t.partial ? ' · part in box' : ''}</span></td><td class="num">${fmt(t.before, t.metric)} → ${fmt(t.after, t.metric)}</td><td class="num worse">${t.delta > 0 ? '+' : ''}${t.delta.toFixed(2)}</td></tr>`).join('')}</table></div>`;
    }
    if (['not_met', 'not_reachable', 'partial', 'needs_decision'].includes(r.outcome) && r.notes && !unreachable) b += `<div class="why">${esc(plain(r.notes))}</div>`;
    if (byOpt) b += `<div class="why note">${esc(plain(r.notes))}</div>`;
    b += optionsHtml(r);
    b += attemptsHtml(r);
    if (!replyFirst && r.reply && !['stopped', 'error'].includes(r.status)) {
      b += `<details><summary>Agent's explanation</summary><div class="reply">${esc(r.reply)}</div></details>`;
    }
  }
  const acts = [];
  const busy = !!S.revs.running;
  if (r.status === 'review') {
    acts.push(`<button class="btn primary" data-act="keep" data-rid="${r.id}" ${busy ? 'disabled' : ''}>Keep</button>`,
              `<button class="btn" data-act="discard" data-rid="${r.id}" ${busy ? 'disabled' : ''}>Discard</button>`);
  }
  if (['kept', 'review', 'set_aside'].includes(r.status) && !cur) {
    acts.push(`<button class="btn ghost" data-act="branch" data-rid="${r.id}" ${busy ? 'disabled' : ''} title="Make this the plan the next request starts from">Continue from ${r.id}</button>`);
  }
  if (acts.length) b += `<div class="acts">${acts.join('')}</div>`;
  b += '</div>';
  return `<article class="card${sel}${dim}" data-rev="${r.id}" tabindex="0"><div class="ch"><span class="rid">${r.id}</span>
    <span class="pill ${r.id === 'R0' ? 'kept' : r.status}">${esc(pill)}</span>${cur ? '<span class="cur">current plan</span>' : ''}
    <span class="when">${timeOf(r.finished || r.created)}</span></div>${b}</article>`;
}

export function renderTimeline() {
  if (!S.revs) return;
  const tl = $('tl');
  const atBottom = tl.parentElement.scrollHeight - tl.parentElement.scrollTop - tl.parentElement.clientHeight < 40;
  tl.innerHTML = S.revs.items.map(card).join('');
  if (atBottom) tl.parentElement.scrollTop = tl.parentElement.scrollHeight;
}

export function renderCard(rid) {
  const el = document.querySelector(`.card[data-rev="${rid}"]`);
  const r = rev(rid);
  if (!el || !r) return renderTimeline();
  el.outerHTML = card(r);
}

export function initTimeline() {
  $('tl').addEventListener('click', async (e) => {
    const f = e.target.closest('[data-fill]');
    if (f) {
      e.stopPropagation();
      $('cmd').value = f.dataset.fill;
      $('cmd').focus();
      return;
    }
    const o = e.target.closest('[data-opt]');
    if (o) {
      e.stopPropagation();
      o.disabled = true;
      try {
        S.revs = await api(`/api/revisions/${o.dataset.rid}/options/${o.dataset.opt}`, {method: 'POST'});
        toast(`${o.dataset.rid} now uses option ${o.dataset.n}. Keep it or Discard.`);
        select(o.dataset.rid);
        emit('revisions');
      } catch (err) { toast(err.message, true); o.disabled = false; }
      return;
    }
    const b = e.target.closest('[data-act]');
    if (b) {
      e.stopPropagation();
      b.disabled = true;
      try {
        S.revs = await api(`/api/revisions/${b.dataset.rid}/${b.dataset.act}`, {method: 'POST'});
        const msg = {keep: `${b.dataset.rid} is kept.`, discard: `${b.dataset.rid} discarded. The plan is back where the request started.`,
                     branch: `The next request starts from ${b.dataset.rid}.`}[b.dataset.act];
        toast(msg);
        if (b.dataset.act !== 'keep') S.sel = S.revs.current;
        emit('revisions');
      } catch (err) { toast(err.message, true); b.disabled = false; }
      return;
    }
    if (e.target.closest('details')) return;
    const c = e.target.closest('.card');
    if (c && c.dataset.rev !== S.sel) select(c.dataset.rev);
  });
  $('tl').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.classList.contains('card')) select(e.target.dataset.rev);
  });
  on('revisions', renderTimeline);
  on('select', renderTimeline);
  on('live', renderCard);
}

export {refreshRevisions};
