import pytest

from app.revisions import RevisionLog, status_for
from tests.test_graph import _session


def _turn(log, s, request, outcome, solve=True):
    """Simulate one agent turn: optionally solve a new version and finish the revision."""
    rev = log.begin(request, s.current_id)
    if solve:
        s.set_objective_weight(next(iter(s.terms)), 2.0 + len(log.items))
        s.solve(request)
    state = {'outcome': outcome, 'final_version': s.current_id,
             'resolved': {'checks': [{'roi': 'CORD', 'passed': True}], 'notes': 'n'},
             'attempts': [{'applied': ['w'], 'eval': {'violations': ['CORD Dmax'], 'progress': 0.5}}],
             'reply': 'done.\n\nResults (deterministic check, version 1): ...'}
    return log.finish(rev, state, s.current_id)


def test_status_mapping():
    assert status_for('met', 0, 3) == 'review'
    assert status_for('partial', 0, 3) == 'review'
    assert status_for('limit_changed', 0, 2) == 'review'
    assert status_for('met', 2, 2) == 'no_change'
    for o in ('not_met', 'unavailable', 'refused', 'limit_refused', 'answered', 'already_met'):
        assert status_for(o, 0, 0) == 'no_change'
    assert status_for('revert', 3, 0) == 'kept'


def test_finish_records_card_fields():
    s, log = _session(), RevisionLog()
    r = _turn(log, s, 'lower eso', 'met')
    assert r.id == 'R1' and r.parent == 'R0' and r.status == 'review'
    assert r.reply == 'done.' and r.notes == 'n'
    assert r.attempts[0]['violations'] == ['CORD Dmax'] and r.attempts[0]['n'] == 1
    assert log.current_id == 'R1'


def test_follow_up_keeps_waiting_review():
    s, log = _session(), RevisionLog()
    r1 = _turn(log, s, 'lower eso', 'met')
    r2 = _turn(log, s, 'now also cord', 'met')
    assert r1.status == 'kept' and r1.kept_reason == 'continued'
    assert r2.parent == 'R1' and r2.status == 'review'


def test_question_does_not_change_or_keep():
    s, log = _session(), RevisionLog()
    r1 = _turn(log, s, 'lower eso', 'met')
    v = s.current_id
    q = _turn(log, s, 'what is the cord max?', 'answered', solve=False)
    assert q.status == 'no_change' and s.current_id == v
    assert r1.status == 'kept'          # continued from it (the question started on its plan)


def test_discard_reverts_and_only_for_waiting():
    s, log = _session(), RevisionLog()
    r1 = _turn(log, s, 'lower eso', 'met')
    log.keep('R1')
    with pytest.raises(ValueError):
        log.discard('R1', s)
    r2 = _turn(log, s, 'now cord', 'met')
    log.discard('R2', s)
    assert r2.status == 'discarded' and s.current_id == r1.final_version and log.current_id == 'R1'


def test_branch_sets_aside_waiting_review():
    s, log = _session(), RevisionLog()
    _turn(log, s, 'lower eso', 'met')
    log.keep('R1')
    r2 = _turn(log, s, 'now cord', 'met')
    log.branch('R0', s)
    assert r2.status == 'set_aside' and s.current_id == 0 and log.current_id == 'R0'
    r3 = _turn(log, s, 'lung instead', 'met')
    assert r3.parent == 'R0' and r2.status == 'set_aside'
    with pytest.raises(ValueError):          # a kept revision cannot be discarded, only branched from
        log.discard('R1', s)


def test_round_trip_and_rebuild_from_turns():
    s, log = _session(), RevisionLog()
    _turn(log, s, 'lower eso', 'met')
    back = RevisionLog.from_state(log.state())
    assert [r.id for r in back.items] == ['R0', 'R1'] and back.get('R1').status == 'review'
    assert back.current_id == 'R1'
    old = RevisionLog.from_turns([{'request': 'x', 'outcome': 'met', 'start': 0, 'final': 1},
                                  {'request': 'q', 'outcome': 'answered', 'start': 1, 'final': 1}], 1)
    assert [r.status for r in old.items] == ['kept', 'kept', 'no_change'] and old.current_id == 'R1'
    # sessions saved before turns were recorded: rebuilt from results, or a stand-in card for the current plan
    res = RevisionLog.from_turns([], 3, {0: 'baseline', 3: 'lower eso'})
    assert [(r.request, r.start_version, r.final_version) for r in res.items[1:]] == [('lower eso', 0, 3)]
    assert res.current_id == 'R1'
    bare = RevisionLog.from_turns([], 2, {0: 'baseline'})
    assert bare.items[-1].outcome == 'restored' and bare.current_id == 'R1'


def test_running_request_blocks_review_actions():
    s, log = _session(), RevisionLog()
    _turn(log, s, 'lower eso', 'met')
    log.begin('another', s.current_id)
    with pytest.raises(RuntimeError):
        log.branch('R0', s)
    with pytest.raises(RuntimeError):
        log.begin('third', s.current_id)
