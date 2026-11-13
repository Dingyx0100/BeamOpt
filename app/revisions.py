"""Revision log: one card per physicist request, with Keep / Discard / Branch review.

A revision records what one request did: the plan version it started from, the version the agent settled on,
the deterministic checks, every attempt, the trade-offs and the agent's explanation.

Review flow (decided with the user, 2026-11-07):
- A result that changed the plan waits for review (status 'review'). The agent has already made it the
  session's current plan, so a follow-up request builds on it; sending one marks it kept ("continued from").
- Keep: mark it kept. Discard (only the waiting review): put the plan back to where the request started.
- Branch from any revision with a plan: make that plan current; a waiting review is set aside (still viewable).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

REVIEW_OUTCOMES = {'met', 'partial', 'limit_changed', 'needs_decision'}
USER_MOVES = {'revert', 'reset', 'undo'}                 # the request itself asked to move the plan
HAS_PLAN = {'kept', 'review', 'set_aside'}
REPLY_CUT = '\n\nResults (deterministic check'


@dataclass
class Revision:
    id: str
    request: str
    parent: str | None                    # revision whose plan this request started from
    start_version: int
    final_version: int | None = None
    status: str = 'working'               # working | review | kept | discarded | set_aside | no_change | stopped | error
    outcome: str | None = None
    kept_reason: str | None = None        # 'kept' (button) | 'continued' (a follow-up built on it) | 'request'
    checks: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    tradeoffs: list = field(default_factory=list)
    changes: list = field(default_factory=list)      # optimizer settings changed between start and final plan
    notes: str = ''
    reply: str = ''
    events: list = field(default_factory=list)
    locks: list = field(default_factory=list)
    options: list = field(default_factory=list)      # solved alternative plans (src/agent/options.py)
    chosen: str | None = None                        # key of the option the physicist picked
    reach: list = field(default_factory=list)        # floor probes: [{roi, metric, goal, floor, verdict, bound}]
    created: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def has_plan(self) -> bool:
        return self.status in HAS_PLAN and self.final_version is not None

    def to_dict(self) -> dict:
        return asdict(self)


def _attempts(raw: list) -> list[dict]:
    out = []
    for i, a in enumerate(raw or []):
        ev = a.get('eval') or {}
        out.append({'n': i + 1, 'applied': list(a.get('applied') or []), 'errors': list(a.get('errors') or []),
                    'version': a.get('version'), 'auto_adjusted': bool(a.get('auto_adjusted')),
                    'violations': list(ev.get('violations') or []),
                    'progress': ev.get('progress'), 'objectives_met': ev.get('objectives_met')})
    return out


def _reach(feas: dict | None) -> list[dict]:
    out = []
    for o in (feas or {}).get('probes') or []:
        pr = o.get('probe') or {}
        out.append({'roi': o['roi'], 'metric': o['metric'], 'goal': o.get('goal'), 'floor': pr.get('value'),
                    'verdict': o.get('verdict'), 'bound': bool(pr.get('bound')), 'text': o.get('text', '')})
    return out


def status_for(outcome: str | None, start: int, final: int | None) -> str:
    if outcome in USER_MOVES:
        return 'kept'
    if outcome in REVIEW_OUTCOMES and final is not None and final != start:
        return 'review'
    if outcome == 'error':
        return 'error'
    return 'no_change'


class RevisionLog:
    def __init__(self):
        self.items: list[Revision] = [Revision('R0', 'Baseline', None, 0, 0, status='kept', outcome='baseline',
                                               kept_reason='baseline', finished=time.time())]
        self.current_id = 'R0'            # revision whose plan is the session's current plan

    # ------------------------------------------------------------ lookup
    def get(self, rid: str) -> Revision:
        for r in self.items:
            if r.id == rid:
                return r
        raise KeyError(f'No revision {rid}')

    @property
    def waiting(self) -> Revision | None:
        return next((r for r in reversed(self.items) if r.status == 'review'), None)

    @property
    def running(self) -> Revision | None:
        return next((r for r in reversed(self.items) if r.status == 'working'), None)

    def by_version(self, version: int) -> Revision | None:
        return next((r for r in reversed(self.items) if r.final_version == version and r.has_plan), None)

    # ------------------------------------------------------------ a request
    def begin(self, request: str, start_version: int, locks: list | None = None) -> Revision:
        if self.running is not None:
            raise RuntimeError(f'{self.running.id} is still running')
        w = self.waiting
        if w is not None and w.final_version == start_version:
            w.status, w.kept_reason = 'kept', 'continued'
        rev = Revision(f'R{len(self.items)}', request, self.current_id, start_version, locks=list(locks or []))
        self.items.append(rev)
        return rev

    def finish(self, rev: Revision, state: dict, current_version: int, tradeoffs: list | None = None,
               changes: list | None = None) -> Revision:
        resolved = state.get('resolved') or {}
        rev.outcome = state.get('outcome')
        rev.final_version = state.get('final_version', current_version)
        if rev.final_version is None:
            rev.final_version = current_version
        rev.status = status_for(rev.outcome, rev.start_version, rev.final_version)
        if rev.status == 'kept':
            rev.kept_reason = 'request'
        rev.checks = (resolved.get('final_eval') or {}).get('checks') or resolved.get('checks') or []
        rev.attempts = _attempts(state.get('attempts'))
        rev.tradeoffs = list(tradeoffs if tradeoffs is not None else (state.get('tradeoffs') or []))
        rev.changes = list(changes or [])
        rev.notes = resolved.get('notes', '') or ''
        rev.reply = (state.get('reply') or '').split(REPLY_CUT)[0].strip()
        rev.events = [{k: e.get(k) for k in ('node', 'text', 't')} for e in state.get('events') or []]
        rev.options = list(state.get('options') or [])
        rev.reach = _reach(state.get('feasibility'))
        rev.finished = time.time()
        if rev.status in ('review', 'kept'):
            found = self.by_version(current_version)
            self.current_id = found.id if found is not None else rev.id
        return rev

    def fail(self, rev: Revision, status: str, message: str) -> Revision:
        rev.status, rev.reply, rev.final_version, rev.finished = status, message, rev.start_version, time.time()
        return rev

    # ------------------------------------------------------------ review actions
    def keep(self, rid: str) -> Revision:
        r = self.get(rid)
        if r.status != 'review':
            raise ValueError(f'{rid} is not waiting for review')
        r.status, r.kept_reason = 'kept', 'kept'
        return r

    def discard(self, rid: str, session) -> Revision:
        r = self.get(rid)
        if r.status != 'review':
            raise ValueError(f'Only a result that is waiting for review can be discarded ({rid} is {r.status}).')
        if self.running is not None:
            raise RuntimeError('Wait for the running request to finish.')
        session.revert(r.start_version)
        r.status = 'discarded'
        self.current_id = r.parent or 'R0'
        return r

    def choose(self, rid: str, key: str, session) -> Revision:
        """Use one of the latest request's options as its result (it then waits for review like any result)."""
        r = self.get(rid)
        if self.running is not None:
            raise RuntimeError('Wait for the running request to finish.')
        if r is not self.items[-1]:
            raise ValueError('Options can only be picked on the latest request.')
        opt = next((o for o in r.options if o.get('key') == key), None)
        if opt is None:
            raise ValueError(f'{rid} has no option {key!r}.')
        session.revert(opt['version'])
        r.final_version, r.status, r.kept_reason, r.chosen = opt['version'], 'review', None, key
        r.checks, r.tradeoffs = list(opt.get('checks') or r.checks), list(opt.get('tradeoffs') or [])
        for o in r.options:
            o['chosen'] = o is opt
        self.current_id = rid
        return r

    def branch(self, rid: str, session) -> Revision:
        r = self.get(rid)
        if not r.has_plan:
            raise ValueError(f'{rid} has no plan to branch from ({r.status}).')
        if self.running is not None:
            raise RuntimeError('Wait for the running request to finish.')
        w = self.waiting
        if w is not None and w.id != rid:
            w.status = 'set_aside'
        session.revert(r.final_version)
        self.current_id = rid
        return r

    # ------------------------------------------------------------ persistence
    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.items]

    def state(self) -> dict:
        return {'items': self.to_list(), 'current_id': self.current_id}

    @classmethod
    def from_state(cls, st: dict | None) -> 'RevisionLog':
        log = cls()
        if st and st.get('items'):
            log.items = [Revision(**d) for d in st['items']]
            for r in log.items:                      # a request that was running when the app stopped
                if r.status == 'working':
                    r.status, r.reply = 'stopped', r.reply or 'The app stopped before this request finished.'
            log.current_id = st.get('current_id') or 'R0'
        return log

    @classmethod
    def from_turns(cls, turns: list[dict], current_version: int, results: dict | None = None) -> 'RevisionLog':
        """Older saved sessions have no revision log: rebuild one from their turn list (all results kept).

        Sessions saved before turns were recorded only have `results` ({version: request}); if even that does not
        cover the current plan, a 'restored plan' card stands in for it so the timeline matches the plan in use.
        """
        log = cls()
        if not turns and results:
            prev = 0
            for vid in sorted(int(v) for v in results if int(v) != 0):
                turns = (turns or []) + [{'request': results.get(vid, results.get(str(vid), '')), 'outcome': 'met',
                                          'start': prev, 'final': vid}]
                prev = vid
        for t in turns or []:
            start, final = int(t.get('start') or 0), t.get('final')
            final = start if final is None else int(final)
            st = 'kept' if final != start else ('stopped' if t.get('outcome') == 'stopped' else 'no_change')
            parent = log.by_version(start)
            log.items.append(Revision(f'R{len(log.items)}', t.get('request', ''), parent.id if parent else None,
                                      start, final, status=st, outcome=t.get('outcome'),
                                      kept_reason='restored' if st == 'kept' else None,
                                      checks=t.get('checks') or [], reply=t.get('reply_text', ''),
                                      finished=time.time()))
        found = log.by_version(current_version)
        if found is None and current_version != 0:
            found = Revision(f'R{len(log.items)}', f'Restored plan (version {current_version})', log.current_id,
                             current_version, current_version, status='kept', outcome='restored',
                             kept_reason='restored', finished=time.time())
            log.items.append(found)
        log.current_id = found.id if found else 'R0'
        return log
