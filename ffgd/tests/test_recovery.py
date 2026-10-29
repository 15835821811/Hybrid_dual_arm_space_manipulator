import platform
import psutil
from ffgd.recovery import live_identity


def test_atomic_json_numeric_scalars_preserve_finite_evidence(tmp_path):
    import json
    import numpy as np
    import pytest
    from ffgd.orchestration import atomic_json
    path=tmp_path/'result.json'
    atomic_json(path,dict(passed=np.bool_(False),count=np.int32(17),values=[np.float32(.5),np.int64(2**60)]))
    assert json.loads(path.read_text())==dict(passed=False,count=17,values=[.5,2**60])
    saved=path.read_bytes()
    for value in (np.float32(np.nan),np.float64(np.inf),np.longdouble(-np.inf)):
        with pytest.raises(ValueError):atomic_json(path,dict(value=value))
        assert path.read_bytes()==saved
    with pytest.raises(TypeError):atomic_json(path,dict(large_array=np.ones(3)))
    assert path.read_bytes()==saved


def test_process_identity_requires_pid_creation_time_host_and_command():
    p=psutil.Process();identity=dict(pid=p.pid,create_time=p.create_time(),host=platform.node(),command=p.cmdline())
    assert live_identity(identity)
    assert not live_identity(dict(identity,create_time=identity['create_time']-1))
    assert not live_identity(dict(identity,command=['different-worker']))
    assert not live_identity(dict(identity,host='different-host'))


def test_stale_worker_recovery_preserves_live_process_and_never_marks_success(tmp_path,monkeypatch):
    import hashlib
    import json
    import ffgd.recovery as recovery
    from ffgd.orchestration import atomic_json
    root=tmp_path;orch=root/'artifacts/ffgd/orchestration';run=root/'artifacts/ffgd/runs/fixture'
    orch.mkdir(parents=True);run.mkdir(parents=True);master=root/'docs/ffgd/MASTER.md';master.parent.mkdir(parents=True);master.write_text('fixture')
    p=psutil.Process();identity=dict(pid=p.pid,create_time=p.create_time()-1,host=platform.node(),command=p.cmdline())
    manifest=dict(id='fixture',status='RUNNING',process_identity=identity);atomic_json(run/'manifest.json',manifest)
    atomic_json(orch/'heavy_worker.lock',identity)
    state=dict(active_unit=manifest,owned_processes=[identity],workspace=dict(branch='fixture',head='abc'),
               source_contract=dict(master_sha256=hashlib.sha256(master.read_bytes()).hexdigest()),
               milestones=dict(M0=dict(status='PASSED'),M4=dict(status='RUNNING')),overall_status='ACTIVE')
    atomic_json(orch/'state.json',state)
    monkeypatch.setattr(recovery,'ROOT',root);monkeypatch.setattr(recovery,'ORCH',orch)
    monkeypatch.setattr(recovery,'read_state',lambda:json.loads((orch/'state.json').read_text()))
    monkeypatch.setattr(recovery,'event',lambda *args,**kwargs:None)
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda args,**kwargs:str(root) if '--show-toplevel' in args else 'fixture' if '--show-current' in args else 'abc')
    report=recovery.resume()
    assert report['ready'] and report['recovered']['status']=='INTERRUPTED_RECOVERABLE'
    assert psutil.pid_exists(p.pid) and not (orch/'heavy_worker.lock').exists()
    assert json.loads((orch/'state.json').read_text())['overall_status']=='ACTIVE'
    assert json.loads((run/'manifest.json').read_text())['status']!='PASSED'


def test_new_bundle_initialization_never_inherits_or_resets_progress(tmp_path,monkeypatch):
    import json
    import pytest
    import ffgd.orchestration as orchestration
    root=tmp_path/'bundle';master=root/'docs/ffgd/MASTER.md';master.parent.mkdir(parents=True);master.write_text('frozen contract')
    orch=root/'artifacts/ffgd/orchestration'
    monkeypatch.setattr(orchestration,'ROOT',root);monkeypatch.setattr(orchestration,'ORCH',orch)
    # An enclosing checkout must not become this portable bundle's identity.
    monkeypatch.setattr(orchestration.subprocess,'check_output',lambda *a,**kw:str(tmp_path))
    result=orchestration.initialize();assert result['repository']['kind']=='SOURCE_BUNDLE'
    state=json.loads((orch/'state.json').read_text());assert state['active_unit'] is None and state['owned_processes']==[]
    assert all(m['status'] in ('NOT_STARTED','NOT_RUN') for m in state['milestones'].values())
    before=(orch/'state.json').read_bytes()
    with pytest.raises(FileExistsError,match='STATE_ALREADY_EXISTS'):orchestration.initialize()
    assert (orch/'state.json').read_bytes()==before
