"""After worker exit, verify terminal resume performs no additional work."""
from pathlib import Path
import sys,subprocess,time,os,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ff_prior.storage import read_json,file_digest
from ff_prior.recovery import process_is_alive
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_paired_objectives.protocol import ROOT,DEFAULT_OUT,ARMS

if __name__=='__main__':
    state=read_json(DEFAULT_OUT/'status.json')
    assert state['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS') and not process_is_alive(state['pid'])
    paths=['status.json','summary.json','ledger.json','launcher.json','TRAIN_selection.json']
    paths += [f'learning/{a}/model.pt' for a in ARMS]
    before={p:file_digest(DEFAULT_OUT/p) for p in paths};started=time.time()
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    proc=subprocess.run([sys.executable,'-m','ff_prior_paired_objectives.job','resume'],cwd=ROOT,env=env,capture_output=True)
    stdout=proc.stdout.decode('utf-8');stderr=proc.stderr.decode('utf-8');response=json.loads(stdout)
    after={p:file_digest(DEFAULT_OUT/p) for p in paths}
    assert proc.returncode==0 and response['resume']=='NO_WORK_ALREADY_COMPLETE' and before==after
    proof={'status':'PASS','before':before,'after':after,'response':response,'stderr':stderr,
        'wall_s':time.time()-started,'new_training_or_physics_calls':0,'old_worker_exited':True}
    atomic_json(ROOT/'ff_prior_paired_objectives/evidence/completed_001/resume_noop.json',proof)
    print(json.dumps({'status':'PASS','resume':response['resume'],'unchanged_files':len(paths),'wall_s':proof['wall_s']}))
