"""Bounded local continuation: wait for actual data-worker exit, then gated launch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from ff_prior.storage import read_json,file_digest,RunLock
from .storage import atomic_json
from ff_prior.recovery import process_is_alive
from .job import Monitor,status,launch,ROOT,DEFAULT_SOURCE,DEFAULT_OUT

DEFAULT_QUEUE='artifacts/v7_1/coverage_training_queue_001'


def worker(source,output,queue):
    from .experiment import sources,require_data
    source,output,queue=map(lambda p:Path(p).resolve(),(source,output,queue))
    plan=read_json(queue/'queue_plan.json');started=time.monotonic()
    with RunLock(queue),Monitor(queue) as monitor:
        try:
            while True:
                state=read_json(source/'status.json')
                if state['pid']!=plan['parent_pid']:raise ValueError('data owner changed; explicit continuation required')
                if not process_is_alive(state['pid']):break
                if time.monotonic()-started>=plan['maximum_wait_s']:raise RuntimeError('QUEUE_WAIT_BUDGET_EXHAUSTED')
                monitor.update('waiting_for_verified_data_exit',parent_pid=state['pid'],parent_stage=state['stage'])
                time.sleep(10)
            require_data(source)
            if sources()!=plan['source_hashes'] or file_digest(source/'run.json')!=plan['parent_run_sha256']:raise ValueError('queued source identity changed')
            result=launch(source,output,False)
            atomic_json(queue/'training_launch.json',result)
            summary={'status':'FINISHED','meaning':'training job launched, not training completed','training_output':str(output),'launch':result}
            atomic_json(queue/'summary.json',summary)
            (queue/'REPORT.md').write_text('# 后续训练已启动\n\n数据前置验收通过后已启动独立训练worker。此队列完成不表示训练或DEV完成。请使用 ff_prior_coverage_train/job.ps1 status/report 查看实际结果。\n',encoding='utf-8')
            monitor.finish(summary);return 0
        except Exception as error:
            result={'status':'FAILED','reason':repr(error),'training_completed':False}
            atomic_json(queue/'failure.json',result)
            (queue/'REPORT.md').write_text('# 后续训练未启动\n\n'+repr(error)+'\n\n原数据与旧作业保留；没有绕过数据验收门槛。\n',encoding='utf-8')
            monitor.finish(result);return 1


def start(source,output,queue):
    from .experiment import sources
    source,output,queue=map(lambda p:Path(p).resolve(),(source,output,queue))
    if output.exists():raise ValueError('training output already exists; inspect and use explicit job resume')
    state=read_json(source/'status.json')
    if state['status']!='RUNNING' or not process_is_alive(state['pid']):raise ValueError('queue requires currently verified live data worker')
    proof=read_json(ROOT/'ff_prior_coverage_train/evidence/software_tests.json')
    if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_coverage_train/evidence/tests.xml'):raise ValueError('software proof stale')
    queue.mkdir(parents=True,exist_ok=False)
    atomic_json(queue/'queue_plan.json',{'parent_pid':state['pid'],'parent_run_sha256':file_digest(source/'run.json'),
        'source_hashes':sources(),'source':str(source),'output':str(output),'maximum_wait_s':6000,
        'launch_only_after_data_gate_and_actual_exit':True,'training_wall_budget_s':5400,'final_TEST':False})
    args=[sys.executable,'-u','-m','ff_prior_coverage_train.queue','worker','--source',str(source),'--output',str(output),'--queue',str(queue)]
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    with (queue/'stdout.log').open('ab') as stdout,(queue/'stderr.log').open('ab') as stderr:
        proc=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,**flags)
    result={'status':'LAUNCHED','launcher_pid':proc.pid,'queue':str(queue)};atomic_json(queue/'launcher.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('start','status','worker'))
    parser.add_argument('--source',default=DEFAULT_SOURCE);parser.add_argument('--output',default=DEFAULT_OUT);parser.add_argument('--queue',default=DEFAULT_QUEUE)
    args=parser.parse_args();os.chdir(ROOT)
    if args.action=='worker':return worker(args.source,args.output,args.queue)
    import json
    result=start(args.source,args.output,args.queue) if args.action=='start' else status(args.queue)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(main())
