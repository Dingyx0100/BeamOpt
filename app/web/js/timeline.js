// Revision timeline: one card per request with its result, cost, attempts and Keep / Discard / Branch.
import {$, api, esc, fmt, opTxt, timeOf, toast} from './api.js';
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
    if (r.status === 'discarded') out = 'Discarded. The plan went back to where this request started.';
    if (r.status === 'set_aside') out = 'Set aside when you branched from another plan. You can still branch from it.';
    if (r.kept_reason === 'continued') out += ' Kept because you continued from it.';
    if (r.status === 'stopped' || r.status === 'error') out = r.reply;
    if (out) b += `<div>${esc(out)}</div>`;
    const replyFirst = REPLY_FIRST.has(r.outcome) && r.reply && r.status !== 'stopped' && r.status !== 'error';
    if (replyFirst) b += `<div class="why note">${esc(r.reply)}</div>`;
    if (r.changes?.length) {
      b += `<div class="block"><span class="lbl">Changes</span><ul class="chg">${r.changes.map((c) =>
        `<li><span>${esc(c.what)}${c.detail ? ` <span class="muted">${esc(c.detail)}</span>` : ''}${c.tag ? ` <span class="tag">${esc(c.tag)}</span>` : ''}</span></li>`).join('')}</ul></div>`;
    }
    const checks = (r.checks || []).filter((c) => c.actual !== null && c.actual !== undefined);
    if (checks.length && !replyFirst) {
      b += `<div class="block"><span class="lbl">Result vs ${esc(r.parent || 'start')}</span><table class="res">${checks.map(checkRow).join('')}</table></div>`;
    }
    const worse = (r.tradeoffs || []).filter((t) => t.flag === 'worse');
    if (worse.length) {
      b += `<div class="block"><span class="lbl">Cost</span><table class="res">${worse.map((t) =>
        `<tr><td>${esc(label(t.roi))} <span class="k">${esc(t.metric)}${t.partial ? ' · part in box' : ''}</span></td><td class="num">${fmt(t.before, t.metric)} → ${fmt(t.after, t.metric)}</td><td class="num worse">${t.delta > 0 ? '+' : ''}${t.delta.toFixed(2)}</td></tr>`).join('')}</table></div>`;
    }
    if (['not_met', 'not_reachable', 'partial', 'needs_decision'].includes(r.outcome) && r.notes) b += `<div class="why">${esc(r.notes)}</div>`;
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
