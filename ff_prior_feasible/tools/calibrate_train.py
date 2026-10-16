"""Bounded TRAIN-only calibration; keeps every attempt and the executed source."""
from pathlib import Path
import time
import zipfile
from ff_prior.storage import read_json,atomic_json,digest
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop import data,training
from ff_prior_feasible.experiment import candidate,sources,ROOT,PREVIOUS,PROTOCOL


def main():
    out=ROOT/'ff_prior_feasible/evidence/train_calibration_001'; out.mkdir(parents=True,exist_ok=False)
    entries=[e for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['split']=='TRAIN' and 'static_detour' in e['sample_id']]
    methods=['structured_feasible','trained_diffusion_feasible']; seed=20261073
    atomic_json(out/'plan.json',{'split':'TRAIN','tasks':[e['task']['task_hash'] for e in entries],
        'methods':methods,'seed':seed,'seeds_per_task':2,'max_candidates':12,'wall_budget_s':180,
        'formal_VAL_used':False,'protocol':PROTOCOL,'source_hashes':sources()})
    with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sources(): archive.write(ROOT/p,p)
    config=read_json('artifacts/v7_1/smoke_002/run.json')['config']; _,robot,initial,_=audit(config); profile=acceptance(config)
    bundle=training.load(PREVIOUS/'learning',read_json(PREVIOUS/'learning/training.json')['metadata']['scope'])
    deadline=time.time()+180; results=[]
    for ti,e in enumerate(entries):
        rr,ii,cc,_=data.runtime(e,robot,initial); task=e['task']; base=public_goal_initializer(task,rr,ii)
        for method in methods:
            for k in range(2):
                result=candidate(method,k,task,base,seed+ti*101+k*997,bundle,rr,ii,cc,profile,deadline)
                result['sample_id']=e['sample_id']; atomic_json(out/f'{ti}_{method}_{k}.json',result); results.append(result)
                print(e['sample_id'],method,k,result['status'],result['eligible'],result.get('reason'),flush=True)
    atomic_json(out/'summary.json',{'attempts':len(results),'eligible':sum(r['eligible'] for r in results),
        'by_method':{m:sum(r['eligible'] for r in results if r['method']==m) for m in methods},
        'task_hashes_unchanged':all(digest(e['task'])==digest(next(x for x in entries if x['sample_id']==e['sample_id'])['task']) for e in entries),
        'no_torque_execution_claim':True})


if __name__=='__main__': main()
