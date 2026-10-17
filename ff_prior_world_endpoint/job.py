"""Hidden detached local worker; terminal status/report; conservative resume."""
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
from ff_prior_coverage_train.job import Monitor,status,TERMINAL
from ff_prior.recovery import recover_dead_lock
from .protocol import ROOT,SOURCE,DEFAULT_OUT


def worker(source,out,resume=False):
    out=Path(out).resolve()
    previous=read_json(out/'status.json')['elapsed_s'] if resume and (out/'status.json').exists() else 0.
    with Monitor(out,previous) as monitor:
        try:
            from .experiment import run
            result=run(source,out,monitor.update,resume,previous);monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc()
            result={'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'failure.json',result)
            prior=(out/'REPORT.md').read_text(encoding='utf-8') if (out/'REPORT.md').exists() else ''
            (out/'REPORT.md').write_text('# 新覆盖实际训练与smoke未完成\n\n'+repr(error)+'\n\n状态FAILED；原结果保留，不能记为通过。\n\n此前进度快照：\n\n'+prior,encoding='utf-8')
            monitor.finish(result);return 1


def launch(source,out,resume=False):
    source,out=Path(source).resolve(),Path(out).resolve()
    if not resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'run.json').exists():raise ValueError('resume requires frozen run identity')
    guard=out/'launch.guard';fd=os.open(guard,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    try:
        state=status(out)
        if state.get('owner_alive') or state.get('status')=='RUNNING':raise RuntimeError('worker is alive; duplicate refused')
        if resume and state.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):
            from .experiment import identity_for
            identity,_=identity_for(source)
            if read_json(out/'run.json')['identity']!=identity:raise ValueError('completed identity changed')
            return dict(state,resume='NO_WORK_ALREADY_COMPLETE')
        if resume and state.get('status') not in TERMINAL:
            atomic_json(out/'interruption.json',{'previous_status':state,'unknown_uncheckpointed_work':True,
                'policy':'consume reserved work; never replay unknown candidates or native executions'})
        if (out/'run.lock').exists():recover_dead_lock(out)
        argv=[sys.executable,'-u','-m','ff_prior_world_endpoint.job','worker','--source',str(source),'--output',str(out)]
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','resume','status','report','worker'))
    parser.add_argument('--source',default=str(SOURCE));parser.add_argument('--output',default=str(DEFAULT_OUT))
    parser.add_argument('--resume',action='store_true',help=argparse.SUPPRESS)
    a=parser.parse_args();os.chdir(ROOT)
    if a.action=='worker':return worker(a.source,a.output,a.resume)
    if a.action in ('start','resume'):result=launch(a.source,a.output,a.action=='resume')
    elif a.action=='status':result=status(a.output)
    else:
        path=Path(a.output)/'REPORT.md'
        if path.exists():print(path.read_text(encoding='utf-8'));return 0
        result={'status':'REPORT_NOT_READY','progress':status(a.output)}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(main())
