"""Detached dependency watcher: cheap status reads, one gated training launch."""
import json
import os
import subprocess
import sys
import time
import traceback
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor,status
from ff_prior.recovery import process_is_alive
from .protocol import ROOT,SOURCE,DEFAULT_OUT,CHAIN_OUT


def verify_source_resume_once():
    proof=CHAIN_OUT/'source_completed_resume.json'
    if proof.exists():return read_json(proof)
    from ff_prior_expand256.job import completed_no_work
    def inventory():return {p.relative_to(SOURCE).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in SOURCE.rglob('*') if p.is_file()}
    core=['run.json','summary.json','ledger.json','independent_verification.json','training_index.json','qualified_increment.json','executed_source.zip']
    before=inventory();hashes={n:file_digest(SOURCE/n) for n in core}
    response=completed_no_work(SOURCE)
    after=inventory();final={n:file_digest(SOURCE/n) for n in core}
    if response['resume']!='NO_WORK_ALREADY_COMPLETE' or before!=after or hashes!=final:raise ValueError('source completion verification changed output')
    result={'status':'PASS','response':response,'files':len(before),'inventory_before_hash':digest(before),
        'inventory_after_hash':digest(after),'core_before':hashes,'core_after':final,'physical_or_training_repeated':False}
    atomic_json(proof,result);return result


def run():
    from .inputs import identity_for,sources
    from .job import launch
    with RunLock(CHAIN_OUT),Monitor(CHAIN_OUT) as monitor:
        try:
            frozen=read_json(CHAIN_OUT/'watch_identity.json')
            if frozen['source_hashes']!=sources():raise ValueError('watcher source changed after freeze')
            monitor.update('waiting_for_expansion_gate',source=str(SOURCE))
            started=time.time()
            while True:
                state=status(SOURCE)
                if state['status']=='RUNNING':
                    if time.time()-started>18000:raise RuntimeError('DEPENDENCY_WAIT_LIMIT; source not stopped')
                    time.sleep(30);continue
                if state['status']=='FINISHED':
                    if process_is_alive(state['pid']):time.sleep(2);continue
                    break
                raise RuntimeError('DEPENDENCY_BLOCKED: '+json.dumps(state))
            monitor.update('verify_completed_expansion_gate_and_inputs')
            identity,_=identity_for()
            atomic_json(CHAIN_OUT/'admission.json',{'status':'PASS','source_identity_hash':digest(identity),
                'actual_TRAIN':identity['actual_TRAIN'],'actual_increment':identity['actual_increment'],
                'source_summary_sha256':file_digest(SOURCE/'summary.json'),'source_audit_sha256':file_digest(SOURCE/'independent_verification.json')})
            verify_source_resume_once()
            if DEFAULT_OUT.exists():raise RuntimeError('training output already exists; no duplicate or automatic resume')
            response=launch()
            atomic_json(CHAIN_OUT/'training_launch.json',response)
            (CHAIN_OUT/'REPORT.md').write_text('# 依赖门通过，训练已启动\n\n'+json.dumps(response,ensure_ascii=False,indent=2),encoding='utf-8')
            monitor.finish({'status':'FINISHED'})
        except Exception as error:
            atomic_json(CHAIN_OUT/'failure.json',{'status':'FAILED','reason':repr(error),'traceback':traceback.format_exc()})
            (CHAIN_OUT/'REPORT.md').write_text('# 后续训练未启动/受阻\n\n'+repr(error)+'\n保留源运行；未完成不可写通过。',encoding='utf-8')
            monitor.finish({'status':'FAILED'})


def start():
    from .inputs import sources
    CHAIN_OUT.mkdir(parents=True,exist_ok=False)
    atomic_json(CHAIN_OUT/'watch_identity.json',{'source_hashes':sources(),'upstream':str(SOURCE),'training_output':str(DEFAULT_OUT),
        'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'no_training_before_gate':True})
    flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (CHAIN_OUT/'stdout.log').open('ab') as stdout,(CHAIN_OUT/'stderr.log').open('ab') as stderr:
        proc=subprocess.Popen([sys.executable,'-u','-m','ff_prior_expand256_train.chain'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**flags)
    atomic_json(CHAIN_OUT/'launcher.json',{'pid':proc.pid,'hidden':True,'started':time.time()})
    return {'status':'DEPENDENCY_WATCHER_LAUNCHED','pid':proc.pid,'source':str(SOURCE),'training_output':str(DEFAULT_OUT)}


if __name__=='__main__':run()
