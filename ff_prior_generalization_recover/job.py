"""Detached recovery of diagnostics only, retaining all original experiment evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor,status
from ff_prior.recovery import recover_dead_lock

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/independent_new_validation_001'
OUT=ROOT/'artifacts/v7_1/independent_new_validation_recovery_001'


def worker(source,out,resume=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    previous=read_json(out/'status.json')['elapsed_s'] if resume and (out/'status.json').exists() else read_json(Path(source)/'status.json')['elapsed_s']
    with Monitor(out,previous) as monitor:
        try:
            from .experiment import run
            result=run(source,out,monitor.update,resume,previous);monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc();r={'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'failure.json',r)
            from ff_prior_generalization.reporting import report
            report(out,'FAILED');monitor.finish(r);return 1


def launch(source,out,resume=False):
    from .experiment import inspect_source,sources
    source,out=Path(source).resolve(),Path(out).resolve();old,manifest,zero=inspect_source(source)
    if not resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'recovery_run.json').exists():raise ValueError('frozen recovery identity required')
    guard=out/'launch.guard';fd=os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        state=status(out)
        if state.get('owner_alive') or state.get('status')=='RUNNING':raise ValueError('recovery worker is alive')
        if resume and state.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):
            ident=read_json(out/'recovery_run.json')['identity']
            if ident['source']!=str(source) or ident['source_manifest']!=manifest or ident['source_hashes']!=sources():
                raise ValueError('completed recovery identity changed')
            return dict(state,resume='NO_WORK_ALREADY_COMPLETE')
        if (out/'run.lock').exists():recover_dead_lock(out)
        argv=[sys.executable,'-u','-m','ff_prior_generalization_recover.job','worker','--source',str(source),'--output',str(out)]
        if resume:argv.append('--resume')
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
        with (out/'worker.stdout.log').open('ab') as stdout,(out/'worker.stderr.log').open('ab') as stderr:
            proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,**flags)
        atomic_json(out/'launcher.json',{'pid':proc.pid,'argv':argv,'started':time.time(),'detached':True})
        for _ in range(100):
            if (out/'run.lock').exists() or proc.poll() is not None:break
            time.sleep(.1)
        return {'status':'LAUNCHED','launcher_pid':proc.pid,'worker':status(out),'output':str(out)}
    finally:os.close(fd);guard.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','resume','worker','status','report'))
    p.add_argument('--source',default=str(SOURCE));p.add_argument('--output',default=str(OUT))
    p.add_argument('--resume',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();os.chdir(ROOT)
    if a.action=='worker':return worker(a.source,a.output,a.resume)
    if a.action in ('start','resume'):r=launch(a.source,a.output,a.action=='resume')
    elif a.action=='status':r=status(a.output)
    else:
        path=Path(a.output)/'ANALYSIS.md'
        if path.exists():print(path.read_text(encoding='utf-8'));return 0
        r={'status':'REPORT_NOT_READY','progress':status(a.output)}
    print(json.dumps(r,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(main())
