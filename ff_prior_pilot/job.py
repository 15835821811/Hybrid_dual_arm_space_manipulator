"""Local detached job, cheap terminal status, explicit resume and automatic report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from ff_prior.storage import atomic_json, read_json
from ff_prior.recovery import process_is_alive, recover_dead_lock

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE='artifacts/v7_1/smoke_002'
DEFAULT_OUT='artifacts/v7_1/qualified_pilot_001'
TERMINAL={'FINISHED','FINISHED_WITH_BLOCKED_ITEMS','FAILED'}


def status(out):
    out=Path(out)
    if not (out/'status.json').exists(): return {'status':'NOT_STARTED_OR_STARTING','output':str(out)}
    state=read_json(out/'status.json')
    if state['status'] not in TERMINAL:
        try: alive=process_is_alive(state['pid'])
        except RuntimeError: alive=None
        state['owner_alive']=alive
        state['heartbeat_age_s']=round(time.time()-state['heartbeat'],1)
        if alive is False: state['status']='INTERRUPTED_REQUIRES_EXPLICIT_RESUME'
        elif state['heartbeat_age_s']>30: state['status']='HEARTBEAT_STALE_CHECK_LOG'
    return state


class Monitor:
    def __init__(self,out,previous=0.):
        self.out=Path(out); self.started=time.monotonic(); self.previous=previous
        self.lock=threading.Lock(); self.stop=threading.Event()
        self.state={'status':'RUNNING','stage':'preflight','pid':os.getpid(),'started_at':time.time()}
        self.thread=threading.Thread(target=self._loop,daemon=True)
    def _save(self):
        self.state.update(heartbeat=time.time(),elapsed_s=self.previous+time.monotonic()-self.started)
        atomic_json(self.out/'status.json',self.state)
    def _loop(self):
        while not self.stop.wait(5):
            with self.lock: self._save()
    def __enter__(self):
        self.update('preflight'); self.thread.start(); return self
    def update(self,stage,**details):
        with self.lock:
            self.state.update(stage=stage,progress=details); self._save()
            event=dict(timestamp=time.time(),stage=stage,**details)
            with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(event,ensure_ascii=False)+'\n')
        print(json.dumps(event,ensure_ascii=True),flush=True)
    def finish(self,result):
        with self.lock:
            self.state.update(status=result['status'],stage='complete',report=str(self.out/'REPORT.md'))
            self._save()
    def __exit__(self,*_):
        self.stop.set(); self.thread.join(timeout=6)


def worker(source,out,resume=False):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'run.lock').exists():
        raise RuntimeError('worker lock already exists; use explicit resume after dead-owner verification')
    previous=read_json(out/'status.json').get('elapsed_s',0.) if resume and (out/'status.json').exists() else 0.
    with Monitor(out,previous) as monitor:
        try:
            from .experiment import run
            result=run(source,out,monitor.update,resume,previous)
            monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc()
            failure={'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'failure.json',failure)
            if not (out/'REPORT.md').exists():
                (out/'REPORT.md').write_text('# 后台验证未完成\n\n'+repr(error)+'\n\n已有证据保留；依赖环节不得记为通过。详见 failure.json 和 worker.stderr.log。\n',encoding='utf-8')
            monitor.finish(failure); return 1


def launch(source,out,resume=False):
    source,out=Path(source).resolve(),Path(out).resolve()
    if not resume: out.mkdir(parents=True,exist_ok=False)
    elif not (out/'run.json').exists(): raise ValueError('resume requires an existing frozen run.json')
    # The guard serializes launches; a run lock is separately owned by the worker.
    guard=out/'launch.guard'; fd=os.open(guard,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    try:
        old=status(out)
        if old.get('status')=='RUNNING' or old.get('owner_alive') is True:
            raise RuntimeError('worker is still alive; refusing duplicate launch')
        if resume and old.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):
            from .experiment import inputs,sources,PROTOCOL
            from ff_prior.storage import digest,file_digest
            _,_,tasks,_,files=inputs(source)
            expected={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(tasks),'source_hashes':sources()}
            if read_json(out/'run.json')['identity']!=expected: raise ValueError('completed job identity changed')
            return dict(old,resume='NO_WORK_ALREADY_COMPLETE')
        if resume and old.get('status') not in TERMINAL:
            atomic_json(out/'interruption.json',{'previous_status':old,'unknown_uncheckpointed_work':True,
                'policy':'do not replay reserved candidates or executions; unknown calls not zero'})
        if (out/'run.lock').exists(): recover_dead_lock(out)
        argv=[sys.executable,'-u','-m','ff_prior_pilot.job','worker','--source',str(source),'--output',str(out)]
        if resume: argv.append('--resume')
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
        with (out/'worker.stdout.log').open('ab') as stdout,(out/'worker.stderr.log').open('ab') as stderr:
            proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,**kwargs)
        atomic_json(out/'launcher.json',{'pid':proc.pid,'argv':argv,'started':time.time(),'detached':True})
        for _ in range(100):
            if (out/'run.lock').exists() or proc.poll() is not None: break
            time.sleep(.1)
        return {'status':'LAUNCHED','launcher_pid':proc.pid,'worker':status(out),'output':str(out)}
    finally:
        os.close(fd); guard.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','resume','status','report','worker'))
    parser.add_argument('--source',default=DEFAULT_SOURCE); parser.add_argument('--output',default=DEFAULT_OUT)
    parser.add_argument('--resume',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(); os.chdir(ROOT)
    if args.action=='worker': return worker(args.source,args.output,args.resume)
    if args.action in ('start','resume'): result=launch(args.source,args.output,args.action=='resume')
    elif args.action=='status': result=status(args.output)
    else:
        path=Path(args.output)/'REPORT.md'
        if path.exists(): print(path.read_text(encoding='utf-8')); return 0
        result={'status':'REPORT_NOT_READY','progress':status(args.output)}
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': sys.exit(main())
