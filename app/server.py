"""BeamOpt Workspace: local web app (goals board · revisions · linked viewer) on top of app.workspace.Workspace.

Run:  .venv/Scripts/python.exe app/server.py        then open http://127.0.0.1:7861  (Ollama must be up)

One patient Workspace per process. Agent requests run in a worker thread; the browser follows them over
server-sent events (GET /api/events).
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import math
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import ui_text as UT  # noqa: E402
from app.workspace import AUDIO_EXT, CFG, UI, Workspace  # noqa: E402
from src.agent.goals import tradeoffs  # noqa: E402
from src.agent.graph import run_turn  # noqa: E402
from src.patient_scan import SUPPORTED, scan_data_dir  # noqa: E402
from src.plan_export import export_version, list_sessions  # noqa: E402

WEB = ROOT / 'app' / 'web'
PORT = int(UI.get('port', 7861))
STAGES = {'understand': 'Reading request', 'propose': 'Proposing changes', 'apply': 'Checking the change',
          'optimize': 'Solving', 'finalize': 'Choosing the best attempt', 'report': 'Writing the summary',
          'answer': 'Answering', 'limit': 'Changing the limit', 'refuse': 'Answering', 'restore': 'Restoring'}


class Stopped(Exception):
    pass


class State:
    ws: Workspace | None = None
    switching = threading.Lock()          # held while a patient loads or a session restores
    turn: dict | None = None              # {'rev', 'events': [...], 'done': bool}


def clean(o):
    """JSON has no NaN / inf (pandas and numpy produce them for missing values): send null instead."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, 'item') and not isinstance(o, (str, bytes)):          # numpy scalars
        try:
            return clean(o.item())
        except (TypeError, ValueError):
            return o
    return o


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


S = State()
app = FastAPI(title='BeamOpt Workspace', docs_url=None, redoc_url=None, default_response_class=SafeJSON)


def ws() -> Workspace:
    if S.ws is None:
        raise HTTPException(503, 'No patient is loaded yet.')
    return S.ws


def _busy(w: Workspace) -> bool:
    return w.lock.locked() or (S.turn is not None and not S.turn['done'])


# ====================================================================== state & data
@app.get('/api/state')
def get_state():
    return SafeJSON(ws().state_payload())


@app.get('/api/revisions')
def get_revisions():
    return ws().revision_payload()


@app.get('/api/version/{vid}')
def get_version(vid: int):
    w = ws()
    try:
        return SafeJSON(w.version_payload(vid))
    except KeyError:
        raise HTTPException(404, f'No plan version {vid}')


@app.get('/api/volume/{kind}')
def get_volume(kind: str, version: str | None = None):
    out = ws().volume(kind, version)
    if 'error' in out:
        raise HTTPException(404, out['error'])
    return SafeJSON(out)


@app.get('/api/details/{vid}')
def get_details(vid: int, cmp: int | None = None):
    """Key-metrics table, full protocol table and scorecard for the Details drawer."""
    w = ws()
    view = w.session.get_version(vid)
    other = w.session.get_version(cmp) if cmp is not None and cmp != vid else None
    md, crit = w.criteria_view(view)
    out = {'metrics': json.loads(w.metrics_table(view, other).to_json(orient='records')),
           'criteria_note': md, 'criteria': json.loads(crit.to_json(orient='records'))}
    try:
        smd, sdf = w.scorecard_view(view, other)
        out.update(scorecard_note=smd, scorecard=json.loads(sdf.to_json(orient='records')))
    except Exception as exc:  # noqa: BLE001
        out.update(scorecard_note=f'Scorecard failed: {exc}', scorecard=[])
    return SafeJSON(out)


@app.get('/api/explain/{vid}')
def get_explain(vid: int, roi: str, cmp: int | None = None):
    """Where an organ's dose comes from (exact per-beam shares) and which hard limits hold the plan back."""
    from src import explain as X
    w = ws()
    roi = roi.upper()
    try:
        w.session.get_version(vid)
        out = {'roi': roi, 'version': vid, 'cmp': cmp, 'binding': X.binding_limits(w.session, vid), 'metrics': {}}
        for metric in ('Dmean', 'D2'):
            cur = X.beam_shares(w.session, vid, roi, metric)
            if cmp is not None:
                par = {b['beam_id']: b for b in X.beam_shares(w.session, cmp, roi, metric).get('beams', [])}
                for b in cur.get('beams', []):
                    b['parent_gy'] = par.get(b['beam_id'], {}).get('gy')
            out['metrics'][metric] = cur
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f'Explanation failed: {exc}')
    return SafeJSON(clean(out))


@app.get('/api/deliverability/{vid}')
def get_deliverability(vid: int, cmp: int | None = None):
    w = ws()
    try:
        md, beams, fid, _ = w.deliverability_view(w.session.get_version(vid),
                                                  w.session.get_version(cmp) if cmp is not None else None)
        d = w.deliver(vid)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f'Deliverability check failed: {exc}')
    maps = {str(g): [[None if x != x else round(float(x), 4) for x in row] for row in m] for g, m in d.maps.items()}
    return SafeJSON({'note': md, 'beams': json.loads(beams.to_json(orient='records')),
                         'fidelity': json.loads(fid.to_json(orient='records')), 'maps': maps})


# ====================================================================== agent requests
def _lock_text(locks: list[str]) -> str:
    return ''.join(f' Keep {UT.roi_label(r)} from getting worse.' for r in locks)


@app.post('/api/turn')
def post_turn(body: dict = Body(...)):
    w = ws()
    text = str(body.get('text') or '').strip()
    locks = [str(x) for x in body.get('locks') or [] if x in w.session.structures]
    if not text:
        raise HTTPException(400, 'Type an instruction or a question first.')
    if _busy(w) or S.switching.locked():
        raise HTTPException(409, 'The agent is still working on the previous request.')
    start = w.session.current_id
    rev = w.revlog.begin(text, start, locks)
    turn = {'rev': rev.id, 'events': [], 'done': False, 't0': time.time()}
    S.turn = turn
    request = text + _lock_text(locks)

    def on_event(e):
        turn['events'].append({'node': e.get('node'), 'stage': STAGES.get(e.get('node'), e.get('node')),
                               'text': e.get('text', ''), 't': e.get('t'), 'version': e.get('version')})
        if w.cancel.is_set():
            raise Stopped()

    def work():
        try:
            with w.lock:
                w.cancel.clear()
                try:
                    st = run_turn(w.graph, request, w.history, on_event=on_event)
                except Stopped:
                    w.session.revert(start)
                    w.revlog.fail(rev, 'stopped', 'Stopped by you. The plan is unchanged; versions solved before '
                                                  'the stop are still listed under Details.')
                    w.turns.append({'time': time.strftime('%H:%M:%S'), 'request': text, 'outcome': 'stopped',
                                    'start': start, 'final': w.session.current_id, 'checks': None,
                                    'reply_text': 'Stopped by the user.'})
                    w.history += [f'Physicist: {request}', '(Stopped by the physicist before the agent finished.)']
                    return
                cur = w.session.current_id
                final = st.get('final_version', cur)
                final = cur if final is None else final
                a, b = w.session.get_version(start).metrics, w.session.get_version(final).metrics
                w.revlog.finish(rev, st, cur, tradeoffs=tradeoffs(a, b) if final != start else [],
                                changes=w.knob_diff(start, final) if final != start else [])
                reply = st.get('reply', '')
                if final != start and st.get('outcome') not in ('revert', 'reset', 'undo'):
                    w.results[final] = text
                w.turns.append({'time': time.strftime('%H:%M:%S'), 'request': text, 'outcome': st.get('outcome'),
                                'start': start, 'final': cur,
                                'checks': (st.get('resolved') or {}).get('final_eval', {}).get('checks')
                                or (st.get('resolved') or {}).get('checks'),
                                'reply_text': reply.split('\n\nResults (deterministic check')[0]})
                w.history += [f'Physicist: {request}', f'Assistant: {reply}']
        except Exception as exc:  # noqa: BLE001
            w.revlog.fail(rev, 'error', f'Something went wrong: {exc!r}. The plan is unchanged.')
            try:
                w.session.revert(start)
            except Exception:  # noqa: BLE001
                pass
        finally:
            w.cancel.clear()
            try:
                w.sync()
            except Exception as exc:  # noqa: BLE001
                print('Could not save the session:', exc)
            turn['done'] = True

    threading.Thread(target=work, daemon=True).start()
    return {'revision': rev.id}


@app.get('/api/events')
async def get_events(since: int = 0):
    """Server-sent events for the running request: stage updates, then a final 'done' with the revision."""
    turn = S.turn

    async def gen():
        if turn is None:
            yield 'event: idle\ndata: {}\n\n'
            return
        i = since
        last_beat = time.time()
        while True:
            while i < len(turn['events']):
                yield f"event: step\ndata: {json.dumps({'i': i, 'rev': turn['rev'], **turn['events'][i]})}\n\n"
                i += 1
            if turn['done'] and i >= len(turn['events']):
                yield f"event: done\ndata: {json.dumps({'rev': turn['rev']})}\n\n"
                return
            if time.time() - last_beat > 5:
                yield f"event: beat\ndata: {json.dumps({'elapsed': round(time.time() - turn['t0'])})}\n\n"
                last_beat = time.time()
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})


@app.post('/api/stop')
def post_stop():
    w = ws()
    if _busy(w):
        w.cancel.set()
        return {'ok': True, 'message': 'Stopping after the current optimizer step…'}
    return {'ok': False, 'message': 'Nothing is running.'}


@app.post('/api/revisions/{rid}/{action}')
def post_revision(rid: str, action: str):
    w = ws()
    if _busy(w):
        raise HTTPException(409, 'Wait for the running request to finish.')
    try:
        with w.lock:
            if action == 'keep':
                w.revlog.keep(rid)
            elif action == 'discard':
                r = w.revlog.discard(rid, w.session)
                w.history.append(f'(Physicist discarded the result of "{r.request}"; the plan is back at '
                                 f'version {w.session.current_id}.)')
            elif action == 'branch':
                r = w.revlog.branch(rid, w.session)
                w.history.append(f'(Physicist switched the current plan to {rid}, version {r.final_version}.)')
            else:
                raise HTTPException(404, f'Unknown action {action}')
            w.sync()
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return w.revision_payload()


# ====================================================================== downloads
@app.get('/api/export/{vid}')
def get_export(vid: int):
    w = ws()
    view = w.session.get_version(vid)
    out = ROOT / UI.get('exports_dir', 'runs/exports') / time.strftime('%Y%m%d-%H%M%S')
    files = export_version(w.session, w.case, view.id, out)
    rev = w.revlog.by_version(view.id)
    name = f"{w.pid}_v{view.id}{'_' + rev.id if rev else ''}_BeamOpt.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
        z.writestr('README.txt', f"BeamOpt export of plan v{view.id} ({view.label})"
                   + (f", revision {rev.id} ({rev.status})" if rev else '') + ".\n"
                   "RTDOSE: total-course physical dose (Gy) on the CT grid, same frame of reference as the dataset "
                   "CT.\nNo RTPLAN is included (a deliverable plan needs leaf sequencing).\n"
                   "session_report.html: every request, optimizer change, check and trade-off of this session.\n"
                   "Research prototype - NOT FOR CLINICAL USE.\n")
        z.writestr('session_report.html', w.report_html())
    (out / name).write_bytes(buf.getvalue())
    return Response(buf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})


@app.get('/api/report')
def get_report():
    w = ws()
    data = w.report_html().encode('utf-8')
    name = f"{w.pid}_session_{time.strftime('%Y%m%d-%H%M')}.html"
    out = ROOT / UI.get('exports_dir', 'runs/exports') / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(data)
    return Response(data, media_type='text/html', headers={'Content-Disposition': f'attachment; filename="{name}"'})


# ====================================================================== patients, sessions, settings
def _ask_dir(current: str, title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(initialdir=current or str(ROOT), title=title)
    root.destroy()
    return path or ''


@app.post('/api/browse')
def post_browse(body: dict = Body(...)):
    try:
        return {'path': _ask_dir(body.get('current') or '', body.get('title') or 'Choose a folder')}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f'Folder dialog unavailable ({exc}); type the path instead.')


@app.get('/api/patients')
def get_patients(dir: str = ''):
    path = dir or (S.ws.cfg['data']['data_dir'] if S.ws else CFG['data']['data_dir'])
    root, pts = scan_data_dir(path)
    return {'dir': str(root or path), 'current': S.ws.pid if S.ws else None,
            'patients': [{'id': p.patient_id, 'label': p.label(), 'usable': p.usable,
                          'supported': p.protocol in SUPPORTED, 'protocol': p.protocol,
                          'problem': p.problems[0] if p.problems else None} for p in pts]}


def _replace_workspace(make):
    old = (S.ws.cfg['data']['data_dir'], S.ws.pid, S.ws.protocol) if S.ws else None
    S.ws = None
    S.turn = None
    gc.collect()
    try:
        S.ws = make()
    except Exception:
        S.ws = Workspace(*old) if old else None
        raise


@app.post('/api/load')
def post_load(body: dict = Body(...)):
    pid, path = body.get('patient'), body.get('dir') or ''
    root, pts = scan_data_dir(path)
    info = {p.patient_id: p for p in pts}.get(pid)
    if info is None or not info.usable:
        raise HTTPException(400, 'Choose a usable patient first.')
    if info.protocol not in SUPPORTED:
        raise HTTPException(400, f'{pid}: protocol {info.protocol} is not supported yet.')
    if S.ws is not None and _busy(S.ws):
        raise HTTPException(409, 'The agent is still working on a request; wait for it to finish.')
    with S.switching:
        try:
            _replace_workspace(lambda: Workspace(str(root), pid, info.protocol))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f'Could not load {pid}: {exc}')
    return {'ok': True, 'message': f'Loaded {pid}.'}


@app.get('/api/sessions')
def get_sessions(dir: str = ''):
    path = dir or str(ROOT / UI.get('plans_dir', 'runs/plans'))
    ss = list_sessions(path)
    cur = S.ws.pid if S.ws else None
    ss = sorted(ss, key=lambda s: s.patient_id != cur)
    return {'dir': path, 'sessions': [{'path': str(s.path), 'label': s.label(), 'patient': s.patient_id}
                                      for s in ss]}


@app.post('/api/restore')
def post_restore(body: dict = Body(...)):
    sess_path, data_path = body.get('path'), body.get('data_dir') or ''
    found = list_sessions(sess_path, include_empty=True) if sess_path else []
    if not found:
        raise HTTPException(400, f'{sess_path!r} is not a saved session.')
    s = found[0]
    if S.ws is not None and _busy(S.ws):
        raise HTTPException(409, 'The agent is still working on a request; wait for it to finish.')
    data_dir = s.state.get('data_dir') or data_path or (S.ws.cfg['data']['data_dir'] if S.ws else '')
    root, pts = scan_data_dir(data_dir)
    info = {p.patient_id: p for p in pts}.get(s.patient_id)
    if info is None and data_path and data_path != data_dir:
        root, pts = scan_data_dir(data_path)
        info = {p.patient_id: p for p in pts}.get(s.patient_id)
    if info is None or not info.usable:
        raise HTTPException(400, f'Patient {s.patient_id} of this session was not found in {data_dir!r}. '
                                 'Choose the data folder that contains it.')
    with S.switching:
        same = S.ws is not None and S.ws.pid == s.patient_id and \
            Path(S.ws.cfg['data']['data_dir']) == Path(root).resolve()
        try:
            if same:
                with S.ws.lock:
                    S.ws.restore(s.path)
            else:
                _replace_workspace(lambda: Workspace(str(root), s.patient_id, info.protocol, restore_dir=str(s.path)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f'Could not restore: {exc}')
    return {'ok': True, 'message': f'Restored {s.patient_id} session {s.stamp}.'}


@app.get('/api/models')
def get_models():
    names = [CFG['llm']['model']]
    try:
        import requests
        tags = requests.get(CFG['llm']['base_url'] + '/api/tags', timeout=3).json()['models']
        names = [t['name'] for t in tags]
        if CFG['llm']['model'] not in names:
            names.insert(0, CFG['llm']['model'])
    except Exception:  # noqa: BLE001
        pass
    return {'models': names, 'current': S.ws.model if S.ws else CFG['llm']['model'],
            'validated': CFG['llm']['model']}


@app.post('/api/model')
def post_model(body: dict = Body(...)):
    w = ws()
    if _busy(w):
        raise HTTPException(409, 'Wait for the running request to finish.')
    w.set_model(str(body.get('model') or ''))
    return {'current': w.model}


@app.post('/api/transcribe')
async def post_transcribe(audio: UploadFile = File(...)):
    from src.transcribe import transcribe
    suffix = Path(audio.filename or 'rec.webm').suffix.lower()
    if suffix not in AUDIO_EXT:
        suffix = '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        tmp = f.name
    try:
        text = await asyncio.to_thread(transcribe, tmp, CFG.get('asr', {}))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f'Transcription failed: {exc}')
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'text': text}


# ====================================================================== static frontend
@app.get('/favicon.ico')
def favicon():
    return Response(status_code=204)


@app.get('/')
def index():
    return FileResponse(WEB / 'index.html', headers={'Cache-Control': 'no-cache'})


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers['Cache-Control'] = 'no-cache'
        return r


if WEB.is_dir():
    app.mount('/static', NoCacheStatic(directory=WEB), name='static')


def main():
    import uvicorn
    t = time.time()
    S.ws = Workspace()
    print(f'Loaded {S.ws.pid} in {time.time() - t:.0f} s. Open http://127.0.0.1:{PORT}')
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')


if __name__ == '__main__':
    main()
