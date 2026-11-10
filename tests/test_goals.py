from src.agent.goals import evaluate, resolve_goal
from src.agent.schemas import GoalItem, PhysicistGoal

METRICS = {
    'PTV': {'scope': 'whole_roi', 'Dmean': 60.4, 'Dmax': 63.9, 'D95': 59.7, 'D98': 59.6},
    'ESOPHAGUS': {'scope': 'sub', 'Dmean': 6.0, 'Dmax': 60.1, 'D2': 32.8, 'V20': 5.6},
    'HEART': {'status': 'unavailable', 'reason': 'no voxels'},
}


def test_decrease_with_default_protections():
    g = PhysicistGoal(intent='optimize', objectives=[GoalItem(roi='esophagus', metric='Dmean', change='reduce_by', value=2)])
    r = resolve_goal(g, METRICS)
    obj = r.objectives[0]
    assert obj.op == '<=' and abs(obj.threshold - 4.0) < 1e-9
    prot = {(c.roi, c.metric): c for c in r.protections}
    assert prot[('PTV', 'D95')].op == '>=' and abs(prot[('PTV', 'D95')].threshold - 59.2) < 1e-9
    new = {**METRICS, 'ESOPHAGUS': {**METRICS['ESOPHAGUS'], 'Dmean': 3.9},
           'PTV': {**METRICS['PTV'], 'D95': 59.5}}
    ev = evaluate(r, new)
    assert ev['objectives_met'] and ev['protections_ok'] and ev['progress'] == 1.0


def test_unavailable_roi_is_issue():
    g = PhysicistGoal(intent='optimize', objectives=[GoalItem(roi='HEART', metric='Dmean', change='reduce')])
    r = resolve_goal(g, METRICS)
    assert not r.objectives and 'no optimization voxels' in r.issues[0]


def test_protection_violation_detected():
    g = PhysicistGoal(intent='optimize', objectives=[GoalItem(roi='ESOPHAGUS', metric='D2', change='reduce_to', value=30)])
    r = resolve_goal(g, METRICS)
    new = {**METRICS, 'ESOPHAGUS': {**METRICS['ESOPHAGUS'], 'D2': 29},
           'PTV': {**METRICS['PTV'], 'D95': 58.0}}
    ev = evaluate(r, new)
    assert ev['objectives_met'] and not ev['protections_ok']


GUARD_METRICS = {**METRICS, 'CORD': {'scope': 'sub', 'Dmean': 5.0, 'Dmax': 20.0, 'D2': 18.0, 'V20': 0.0}}


def test_unmentioned_oar_guard():
    g = PhysicistGoal(intent='optimize', objectives=[GoalItem(roi='ESOPHAGUS', metric='Dmean', change='reduce_by', value=2)])
    r = resolve_goal(g, GUARD_METRICS)
    guards = {(c.roi, c.metric): c for c in r.checks if c.role == 'guard'}
    assert ('CORD', 'Dmax') in guards and abs(guards[('CORD', 'Dmax')].threshold - 22.0) < 1e-9
    assert not any(roi in ('ESOPHAGUS', 'HEART', 'PTV') for roi, _ in guards)   # mentioned / unavailable / target
    bad = {**GUARD_METRICS, 'ESOPHAGUS': {**METRICS['ESOPHAGUS'], 'Dmean': 3.9},
           'CORD': {**GUARD_METRICS['CORD'], 'Dmax': 27.3}}
    ev = evaluate(r, bad)
    assert ev['objectives_met'] and not ev['protections_ok'] and not ev['guards_ok']
    assert ev['violations'] == ['CORD Dmax']
    assert not any(c.role == 'guard' for c in resolve_goal(g, GUARD_METRICS, guard_tol={}).checks)


def test_tradeoff_flags():
    from src.agent.goals import tradeoffs
    after = {**GUARD_METRICS, 'ESOPHAGUS': {**METRICS['ESOPHAGUS'], 'Dmean': 4.0},
             'CORD': {**GUARD_METRICS['CORD'], 'Dmax': 27.3}, 'PTV': {**METRICS['PTV'], 'D95': 59.65}}
    rows = {(r['roi'], r['metric']): r['flag'] for r in tradeoffs(GUARD_METRICS, after)}
    assert rows[('ESOPHAGUS', 'Dmean')] == 'better' and rows[('CORD', 'Dmax')] == 'worse'
    assert rows[('PTV', 'D95')] == 'same' and ('HEART', 'Dmean') not in rows
