"""Verify durable evidence and actual process identities before resuming work.

This command recovers bookkeeping and returns the next agent action. It never
retries a failed experiment blindly or claims to start an autonomous agent.
"""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import psutil
from .orchestration import ROOT,ORCH,read_state,atomic_json,event,utc


def live_identity(identity):
    try:
        process=psutil.Process(identity['pid'])
        return identity['host']==platform.node() and abs(process.create_time()-identity['create_time'])<.01 and process.cmdline()==identity['command']
    except psutil.NoSuchProcess:return False
    except psutil.AccessDenied as error:raise RuntimeError('PROCESS_IDENTITY_UNVERIFIABLE') from error


def resume():
    s=read_state();active=s.get('active_unit');checks={};recovered=None
    checks['master_hash']=hashlib.sha256((ROOT/'docs/ffgd/MASTER.md').read_bytes()).hexdigest()==s['source_contract']['master_sha256']
    checks['branch']=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==s['workspace']['branch']
    actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if active:checks['worker_host']=active['process_identity']['host']==platform.node()
    checkpoint=s.get('latest_verified_checkpoint')
    checks['checkpoint_hash']=None if checkpoint is None else Path(checkpoint['path']).exists() and hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==checkpoint['sha256']
    if active and all(v is not False for v in checks.values()) and not live_identity(active['process_identity']):
        path=ROOT/'artifacts/ffgd/runs'/active['id'];manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        # A completed manifest may precede final state cleanup by milliseconds.
        # Preserve its real status; only unfinished units become interrupted.
        if manifest['status']=='RUNNING':
            manifest.update(status='INTERRUPTED_RECOVERABLE',recovered_at=utc(),recovery_reason='Recorded PID/create-time/command is no longer live; no completion inferred.')
            atomic_json(path/'manifest.json',manifest)
            with (ORCH/'work_units.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(manifest)+'\n')
        recovered=dict(id=active['id'],status=manifest['status'])
        lock=ORCH/'heavy_worker.lock'
        if lock.exists():
            owner=json.loads(lock.read_text(encoding='utf-8'))
            if owner==active['process_identity'] and not live_identity(owner):lock.unlink()
        # Do not overwrite a newer worker's state if another process started.
        current=read_state()
        if (current.get('active_unit') or {}).get('id')==active['id']:
            current['active_unit']=None;current['owned_processes']=[];current['last_update_utc']=utc()
            atomic_json(ORCH/'state.json',current);s=current
        event('resume_recovered_stale_worker',**recovered)
    active=s.get('active_unit');live=bool(active and live_identity(active['process_identity']))
    earliest=next((name for name,m in s['milestones'].items() if m['status']!='PASSED'),None)
    result=dict(ready=all(v is not False for v in checks.values()),checks=checks,actual_head=actual_head,recorded_head=s['workspace']['head'],
                earliest_unpassed=earliest,active_worker=active['process_identity'] if live else None,recovered=recovered,
                checkpoint=checkpoint,next_action=s.get('next_action'),overall_status=s['overall_status'],
                action='WAIT_FOR_OWNED_WORKER' if live else 'AGENT_REVIEW_EVIDENCE_AND_CONTINUE_EARLIEST_GATE',
                background_agent_started=False)
    atomic_json(ORCH/'last_recovery.json',result)
    return result
