import platform
import psutil
from ffgd.recovery import live_identity


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
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda args,**kwargs:'fixture' if '--show-current' in args else 'abc')
    report=recovery.resume()
    assert report['ready'] and report['recovered']['status']=='INTERRUPTED_RECOVERABLE'
    assert psutil.pid_exists(p.pid) and not (orch/'heavy_worker.lock').exists()
    assert json.loads((orch/'state.json').read_text())['overall_status']=='ACTIVE'
    assert json.loads((run/'manifest.json').read_text())['status']!='PASSED'
