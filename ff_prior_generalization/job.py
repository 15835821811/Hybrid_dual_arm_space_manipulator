"""Detached bounded validation; query status/report without chat polling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor, status, TERMINAL
from ff_prior.recovery import recover_dead_lock
from .protocol import ROOT, SOURCE

DEFAULT_OUT=ROOT/'artifacts/v7_1/independent_new_validation_001'


def worker(source,out,resume=False):
    from .experiment import run
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.lock').exists():raise RuntimeError('worker lock exists; explicit dead-owner recovery required')
    previous=read_json(out/'status.json').get('elapsed_s',0.) if resume and (out/'status.json').exists() else 0.
    with Monitor(out,previous) as monitor:
        try:
            result=run(source,out,monitor.update,resume,previous);monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc()
            failure={'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'failure.json',failure)
            # Preserve partial results and write an explicitly failed report.
            from .reporting import report
            report(out,'FAILED');monitor.finish(failure);return 1


def launch(source,out,resume=False):
    from .experiment import require_parent, completed_result
    source,out=Path(source).resolve(),Path(out).resolve();require_parent(source)
    if not resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'run.json').exists():raise ValueError('resume requires frozen identity')
    guard=out/'launch.guard';fd=os.open(guard,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    try:
        old=status(out)
        if old.get('status')=='RUNNING' or old.get('owner_alive') is True:
            raise RuntimeError('worker is alive; duplicate launch refused')
        if resume and old.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):
            if Path(read_json(out/'run.json')['identity']['source']).resolve()!=source:
                raise ValueError('completed job source changed')
            completed_result(out);return dict(old,resume='NO_WORK_ALREADY_COMPLETE')
        if resume and old.get('status') not in TERMINAL:
            atomic_json(out/'interruption.json',{'previous_status':old,'unknown_uncheckpointed_work':True,
                       'policy':'reserved work is not repeated; unknown calls never counted as zero'})
        if (out/'run.lock').exists():recover_dead_lock(out)
        argv=[sys.executable,'-u','-m','ff_prior_generalization.job','worker','--source',str(source),'--output',str(out)]
        if resume:argv.append('--resume')
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
        with (out/'worker.stdout.log').open('ab') as stdout,(out/'worker.stderr.log').open('ab') as stderr:
            proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,**kwargs)
        atomic_json(out/'launcher.json',{'pid':proc.pid,'argv':argv,'started':time.time(),'detached':True})
        for _ in range(100):
            if (out/'run.lock').exists() or proc.poll() is not None:break
            time.sleep(.1)
        return {'status':'LAUNCHED','launcher_pid':proc.pid,'worker':status(out),'output':str(out)}
    finally:os.close(fd);guard.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','resume','worker','status','report'))
    p.add_argument('--source',default=str(SOURCE));p.add_argument('--output',default=str(DEFAULT_OUT))
    p.add_argument('--resume',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();os.chdir(ROOT)
    if a.action=='worker':return worker(a.source,a.output,a.resume)
    if a.action in ('start','resume'):result=launch(a.source,a.output,a.action=='resume')
    elif a.action=='status':result=status(a.output)
    else:
        path=Path(a.output)/'ANALYSIS.md'
        if path.exists():print(path.read_text(encoding='utf-8'));return 0
        result={'status':'REPORT_NOT_READY','progress':status(a.output)}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(main())
