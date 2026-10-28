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
import shutil
import psutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ORCH=ROOT/'artifacts/ffgd/orchestration'
_deadline=None
_memory_limit=None


def check_budget():
    if _deadline is not None and time.perf_counter()>_deadline:raise RuntimeError('RESOURCE_WALL_BUDGET_EXHAUSTED')
    if _memory_limit is not None and psutil.Process().memory_info().rss>_memory_limit:raise RuntimeError('RESOURCE_MEMORY_BUDGET_EXHAUSTED')


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_state(): return json.loads((ORCH/'state.json').read_text(encoding='utf-8-sig'))


def repository_identity(root=ROOT):
    """Do not mistake an enclosing repository for a standalone source bundle."""
    try:
        top=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()).resolve()
        if top!=Path(root).resolve():return dict(kind='SOURCE_BUNDLE',head=None,branch=None)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        return dict(kind='GIT_CHECKOUT',head=head,branch=branch)
    except (subprocess.CalledProcessError,FileNotFoundError):return dict(kind='SOURCE_BUNDLE',head=None,branch=None)


def initialize():
    """Bootstrap a fresh delivery only; never reset an existing work ledger."""
    path=ORCH/'state.json'
    if path.exists():raise FileExistsError('FFGD_STATE_ALREADY_EXISTS_USE_RESUME')
    master=ROOT/'docs/ffgd/MASTER.md'
    if not master.is_file():raise FileNotFoundError('DELIVERY_MASTER_MISSING')
    identity=repository_identity(ROOT)
    state=dict(schema='ffgd_long_running_state_v1',template_only=False,goal_id='free_floating_guided_diffusion_core',
        overall_status='NOT_STARTED',architecture_status='NOT_RUN',engineering_status='NOT_STARTED',learning_benefit_status='NOT_RUN',current_milestone='M0',
        workspace=dict(path=str(ROOT),branch=identity['branch'],head=identity['head'],kind=identity['kind'],user_changes_reviewed=False),
        source_contract=dict(master_sha256=hashlib.sha256(master.read_bytes()).hexdigest()),
        milestones={f'M{i}':dict(status='NOT_STARTED' if i==0 else 'NOT_RUN',evidence=[],open_failures=[]) for i in range(9)},
        active_unit=None,owned_processes=[],latest_verified_checkpoint=None,external_blockers=[],completion_evidence=None,
        next_action='Run doctor, verify declared assets/config/checkpoint, then the explicit engineering commands. No milestone is pre-certified.',
        last_update_utc=utc())
    ORCH.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects against concurrent initializers as well.
    with path.open('x',encoding='utf-8') as stream:json.dump(state,stream,indent=2,ensure_ascii=False)
    event('initialized_fresh_delivery',repository=identity,stages='NOT_STARTED_OR_NOT_RUN')
    return dict(initialized=True,state=str(path),repository=identity,existing_state_modified=False)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def event(kind,**details):
    with (ORCH/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=utc(),kind=kind,**details),ensure_ascii=False)+'\n')


def resource_event(kind,**details):
    with (ORCH/'resource_ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=utc(),kind=kind,**details))+'\n')


def source_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'ffgd').rglob('*.py')):
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def unit(name, callback, budget):
    global _deadline,_memory_limit
    heavy=name not in ('software','evidence_report','execution_state_audit');lock=ORCH/'heavy_worker.lock'
    process=psutil.Process();identity=dict(pid=os.getpid(),host=platform.node(),create_time=process.create_time(),command=process.cmdline())
    if heavy:
        if lock.exists():
            owner=json.loads(lock.read_text())
            if owner['host']!=platform.node():raise RuntimeError('HEAVY_WORKER_HOST_UNVERIFIABLE')
            try:
                other=psutil.Process(owner['pid'])
                live=owner['host']==platform.node() and abs(other.create_time()-owner['create_time'])<.01 and other.cmdline()==owner['command']
            except psutil.NoSuchProcess:live=False
            except psutil.AccessDenied as error:raise RuntimeError('HEAVY_WORKER_IDENTITY_UNVERIFIABLE') from error
            if live:raise RuntimeError(f"HEAVY_WORKER_ALREADY_RUNNING {owner['pid']}")
            event('stale_worker_lock',owner=owner);lock.unlink()
        with lock.open('x',encoding='utf-8') as f:json.dump(identity,f)
    ident=f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}_{name}_{uuid.uuid4().hex[:6]}"
    path=ROOT/'artifacts/ffgd/runs'/ident; path.mkdir(parents=True)
    manifest=dict(id=ident, status='RUNNING',start=utc(),pid=os.getpid(),host=platform.node(),
                  process_identity=identity,command=[sys.executable,'-m','ffgd.cli',*sys.argv[1:]],source_hash=source_hash(),budget=budget)
    manifest['repository']=repository_identity(ROOT);manifest['git_head']=manifest['repository']['head']
    memory=psutil.virtual_memory();disk=shutil.disk_usage(ROOT)
    manifest['resources']=dict(available_ram=memory.available,max_rss=int(.8*memory.available),free_disk=disk.free,logical_cpus=os.cpu_count(),reserved_cpus=budget.get('reserved_cpus',1))
    if disk.free<10*1024**3:
        if heavy:lock.unlink()
        raise RuntimeError('RESOURCE_DISK_BELOW_10_GIB')
    _deadline=time.perf_counter()+budget.get('max_wall_seconds',600);_memory_limit=int(.8*memory.available)
    # Small immutable code/config snapshot makes uncommitted experiment inputs recoverable.
    with zipfile.ZipFile(path/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for base,pattern in ((ROOT/'ffgd','*.py'),(ROOT/'configs/ffgd','*')):
            for source in sorted(base.rglob(pattern)):
                if source.is_file():archive.write(source,str(source.relative_to(ROOT)))
    atomic_json(path/'manifest.json',manifest)
    resource_event('start',id=ident,process_identity=identity,budget=budget,resources=manifest['resources'])
    s=read_state();s['overall_status']='ACTIVE'
    s['workspace']['head']=manifest['git_head']
    if heavy:s['active_unit']=manifest;s['owned_processes']=[identity]
    atomic_json(ORCH/'state.json',s)
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
    cpu=process.cpu_times();memory=process.memory_info()
    resource_event('end',id=ident,status=manifest['status'],wall_seconds=manifest['wall_seconds'],cpu_user_seconds=cpu.user,
                   cpu_system_seconds=cpu.system,rss_bytes=memory.rss,peak_rss_bytes=getattr(memory,'peak_wset',None))
    atomic_json(path/'manifest.json',manifest)
    with (ORCH/'work_units.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(manifest)+'\n')
    event('work_unit_end',id=ident,status=manifest['status'])
    s=read_state()
    if heavy:s['active_unit']=None;s['owned_processes']=[];lock.unlink()
    s['last_update_utc']=utc();s['latest_verified_command']=manifest['command'];atomic_json(ORCH/'state.json',s)
    print(json.dumps(dict(run=str(path),**manifest),indent=2))
    if error is not None:raise error
    return manifest['exit_code']
