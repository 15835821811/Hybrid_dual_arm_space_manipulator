"""Detached bounded worker. Resume never repeats uncertain physics."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor,status
from .protocol import ROOT,DEFAULT_OUT

def completed_no_work(out):
    from .inputs import identity
    if read_json(out/'run.json')['identity']!=identity():raise ValueError('completed identity changed')
    if read_json(out/'summary.json')['status'] not in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):raise ValueError('not completed')
    return dict(status(out),resume='NO_WORK_ALREADY_COMPLETE')

def worker(out):
    with Monitor(out,0.) as monitor:
        try:
            from .experiment import run
            result=run(out,monitor.update);monitor.finish(result)
            return 0 if result['status']=='FINISHED' else 2
        except Exception as error:
            traceback.print_exc()
            failure={'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'failure.json',failure)
            (out/'REPORT.md').write_text('# 有界验证未完成\n\n'+repr(error)+'\n已有证据保留，不自动重跑，不写为通过。\n',encoding='utf-8')
            monitor.finish(failure);return 1

def launch(out):
    if out.resolve()!=DEFAULT_OUT.resolve():raise ValueError('only frozen output allowed')
    out.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-u','-m','ff_prior_endpoint_precision.job','worker']
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    with (out/'worker.stdout.log').open('ab') as stdout,(out/'worker.stderr.log').open('ab') as stderr:
        proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,**flags)
    atomic_json(out/'launcher.json',{'pid':proc.pid,'argv':argv,'started':time.time(),'detached':True})
    for _ in range(100):
        if (out/'status.json').exists() or proc.poll() is not None:break
        time.sleep(.1)
    return {'status':'LAUNCHED','launcher_pid':proc.pid,'worker':status(out),'output':str(out)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('start','worker','status','report','resume'))
    action=parser.parse_args().action;os.chdir(ROOT);out=DEFAULT_OUT
    if action=='worker':return worker(out)
    if action=='start':result=launch(out)
    elif action=='status':result=status(out)
    elif action=='report' and (out/'REPORT.md').exists():print((out/'REPORT.md').read_text(encoding='utf-8'));return 0
    elif action=='resume':
        current=status(out)
        if current.get('owner_alive'):result=dict(current,resume='NO_WORK_OWNER_ALIVE')
        elif current.get('status') in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS'):result=completed_no_work(out)
        else:result=dict(current,resume='BLOCKED_UNKNOWN_WORK_NO_REPLAY')
    else:result={'status':'REPORT_NOT_READY','progress':status(out)}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':sys.exit(main())
