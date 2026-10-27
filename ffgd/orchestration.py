"""Atomic status and append-only evidence; unfinished work never becomes PASS."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
ORCH=ROOT/'artifacts/ffgd/orchestration'


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_state(): return json.loads((ORCH/'state.json').read_text(encoding='utf-8-sig'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def event(kind,**details):
    with (ORCH/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=utc(),kind=kind,**details),ensure_ascii=False)+'\n')


def source_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'ffgd').rglob('*.py')):
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def unit(name, callback, budget):
    ident=f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}_{name}_{uuid.uuid4().hex[:6]}"
    path=ROOT/'artifacts/ffgd/runs'/ident; path.mkdir(parents=True)
    manifest=dict(id=ident, status='RUNNING',start=utc(),pid=os.getpid(),host=platform.node(),
                  command=[sys.executable,'-m','ffgd.cli',*sys.argv[1:]],source_hash=source_hash(),budget=budget)
    atomic_json(path/'manifest.json',manifest)
    s=read_state();s['active_unit']=manifest;s['overall_status']='ACTIVE';atomic_json(ORCH/'state.json',s)
    start=time.perf_counter(); error=None
    try:
        result=callback(path)
        manifest['status']='PASSED' if result['passed'] else 'FAILED_NEEDS_FIX'
        atomic_json(path/'result.json',result)
    except Exception as ex:
        import traceback
        (path/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        manifest['status']='FAILED_NEEDS_FIX';error=ex
    manifest.update(end=utc(),wall_seconds=time.perf_counter()-start,exit_code=0 if manifest['status']=='PASSED' else 1)
    atomic_json(path/'manifest.json',manifest)
    with (ORCH/'work_units.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(manifest)+'\n')
    event('work_unit_end',id=ident,status=manifest['status'])
    s=read_state();s['active_unit']=None;s['last_update_utc']=utc();s['latest_verified_command']=manifest['command'];atomic_json(ORCH/'state.json',s)
    print(json.dumps(dict(run=str(path),**manifest),indent=2))
    if error is not None:raise error
    return manifest['exit_code']
