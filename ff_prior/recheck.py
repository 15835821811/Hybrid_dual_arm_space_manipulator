"""Explicit smoke-only learning/comparison rerun on an immutable prior dataset.

No generation, TEST evaluation, pilot or study entry point is provided here.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from .pipeline import audit,acceptance,source_manifest
from .storage import read_json,atomic_json,digest,file_digest,Ledger,RunLock,atomic_npz
from .contracts import assert_task_compatible
from .diffusion import train,DIFFUSION_CONTRACT
from .benchmark import benchmark

BUDGET={"training_updates_max":200,"evaluation_tasks_max":4,"benchmark_selected_execution_max":12,
        "candidates_per_task":5,"ddim_steps":16,"guidance_updates_per_candidate":2}


def run_recheck(source,output,dry_run=False,resume=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError("comparison output must be separate from the source run")
    old_run=read_json(source/'run.json')
    if old_run['identity']['profile']!='smoke':
        raise ValueError("only an existing smoke source is accepted")
    config=old_run['config']
    tasks=read_json(source/'benchmark_tasks.json')
    dataset=read_json(source/'dataset.json')
    if len(tasks)!=4 or len(dataset['eligible']['TRAIN'])!=150 or len(dataset['eligible']['VAL'])!=30:
        raise ValueError("this repair protocol freezes the original 4-task / 150-TRAIN / 30-VAL smoke")
    source_files={name:file_digest(source/name) for name in ('run.json','dataset.json','splits.json',
        'publication.json','initial_state.npz','pair_policy.json','benchmark_tasks.json','audit_selection.json','benchmark.json')}
    rows=[]
    publication=read_json(source/'publication.json')['files']
    # Only training/validation arrays are read or linked. TEST remains sealed.
    for sample in dataset['eligible']['TRAIN']+dataset['eligible']['VAL']:
        name=f'records/{sample}.json'
        row=read_json(source/name)
        for rel in (name,row['arrays']):
            if file_digest(source/rel)!=publication[rel]:
                raise ValueError("source published data changed: "+rel)
            source_files[rel]=publication[rel]
        row['arrays']=str(source/row['arrays'])
        rows.append(row)
    identity={'kind':'smoke_comparison_recheck','profile':'smoke','source':str(source),
        'source_files':source_files,'source_hash':digest(source_manifest()),'budget':BUDGET,
        'tasks_hash':digest(tasks),'dataset_hash':digest(dataset),'config_hash':digest(config),
        'diffusion_contract':DIFFUSION_CONTRACT,'clean_prior_mode':'train_temporal'}
    if dry_run:
        return {'status':'DRY_RUN','identity':identity}
    if (output/'run.json').exists() and not resume:
        raise ValueError("output exists; explicit --resume required")
    with RunLock(output):
        ledger=Ledger(output,identity)
        audit_report,robot,initial,collision=audit(config)
        if initial.model_hash!=old_run['identity']['robot_hash'] or initial.state_id!=old_run['identity']['initial_state_id']:
            raise ValueError("physical model/initial state changed")
        if digest(read_json(source/'pair_policy.json'))!=collision.policy_hash:
            raise ValueError("collision pair policy changed")
        profile=acceptance(config)
        for entry in tasks:
            assert_task_compatible(entry['task'],robot,initial,collision,profile)
        manifest={'identity':identity,'config':config,'scope':'DEVELOPMENT_ONLY','source_hashes':source_manifest(),
            'producer_head':audit_report['head'],'physical_acceptance':audit_report['acceptance'],
            'robot_hash':initial.model_hash,'initial_state_id':initial.state_id,'TEST':'NOT_RUN'}
        atomic_json(output/'run.json',manifest)
        for name,value in (('dataset.json',dataset),('benchmark_tasks.json',tasks),
            ('scene.json',robot.scene.to_dict()),('pair_policy.json',collision.policy),
            ('audit_selection.json',read_json(source/'audit_selection.json'))):
            atomic_json(output/name,value)
        atomic_npz(output/'initial_state.npz',values=initial.values)
        training=train(output,rows,dataset,initial.model_hash,initial.state_id,updates=200)
        if training['status']!='PASS':
            raise RuntimeError("training/reload smoke failed")
        compared=benchmark(output,[],BUDGET,config,robot,initial,collision,ledger)
        reports=[read_json(p) for p in (output/'benchmark').glob('*/execution/report.json')]
        summary={'status':'FINISHED','scope':'SMOKE_DEV_RECHECK','source':str(source),'budget':BUDGET,
            'same_frozen_tasks':read_json(output/'benchmark_tasks.json')==tasks,
            'physical_profile_unchanged':audit_report['acceptance']==read_json(source/'baseline_audit.json')['acceptance'],
            'training':{k:v for k,v in training.items() if k not in ('history','metadata')},
            'benchmark':compared,'physical_execution_count':len(reports),
            'independent_replay_pass':sum(r.get('replay',{}).get('passed') is True for r in reports),
            'source_unchanged':all(file_digest(source/name)==h for name,h in source_files.items()),
            'calls':ledger.data['calls'],'wall_s':ledger.data['wall_s'],'cpu_s':ledger.data['cpu_s'],
            'SCALEUP_ELIGIBLE':False,'pilot':'NOT_RUN','final_TEST':'NOT_RUN'}
        assert summary['same_frozen_tasks'] and summary['physical_profile_unchanged'] and summary['source_unchanged']
        atomic_json(output/'summary.json',summary)
        return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    result=run_recheck(args.source,args.output,args.dry_run,args.resume)
    print({k:v for k,v in result.items() if k not in ('identity','calls')})
