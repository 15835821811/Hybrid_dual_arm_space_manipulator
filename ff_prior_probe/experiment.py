"""Frozen TRAIN-only mechanism smoke; no task generation or checkpoint update."""
from pathlib import Path
import os
import subprocess
import time
import zipfile
import torch
from ff_prior.storage import read_json,atomic_json,atomic_npz,file_digest,digest,Ledger,RunLock
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop.experiment import execute
from ff_prior_loop import data as old_data
from ff_prior_modes import learning,data
from ff_prior_modes.experiment import inputs as previous_inputs,sources as previous_sources,FROZEN
from ff_prior_latent import learning as frozen_learning
from ff_prior_feasible.guidance import CONTRACT as GUIDE
from ff_prior_feasible.validation import CONTRACT as VALIDATE
from .sampling import candidate,SPECS

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'artifacts/v7_1/qualified_modes_smoke_001'
METHODS=tuple(SPECS)
PROTOCOL={'version':'TRAIN_residual_amplitude_smoke_1','profile':'smoke','study':False,
    'methods':list(METHODS),'specs':{k:list(v) for k,v in SPECS.items()},'guidance':GUIDE,'validation':VALIDATE,
    'TRAIN_tasks':8,'task_selection':'all eight preexisting qualified_modes generation parents; label controls removed from inference entries',
    'candidate_seed':20261113,'candidates_per_task':5,'max_candidates':400,
    'witness_task_slots':[0,6],'max_selected_executions':20,
    'candidate_query_ceiling':113,'candidate_integration_interval_ceiling':15872,'candidate_wall_ceiling_s':15.,
    'wall_budget_s':5400,'threads':2,'amplitudes':[0.,.25,.5,1.],'checkpoint_updates':0,'new_data':0,
    'method_order':'rotate by (task_index+candidate_index) before each candidate block; no cross-method proposal cache',
    'selection':'minimum normalized goal error among eligible; one execution on each preselected TRAIN task; no replacement',
    'mean_zero':'five identical proposals intentionally retained; report one distinct path, charge all five; strong mean-plus-bow baseline retained',
    'stability':'at least36/40 eligible, each task>=4/5, preselected2/2 actual execution/replay',
    'diversity':'greedy candidate-order packing of eligible control matrices at RMS>=0.005rad; also mean pairwise RMS; not homotopy',
    'attenuation_signal':'no less eligible than full amplitude; stable; queries AND time <=90% full; mean pairwise eligible RMS >=80% full; distinct eligible count >=80% full',
    'selection_scope':'TRAIN mechanism evidence only; no generalization/superiority/expansion declaration',
    'original_DEV_executed':False,'VAL_executed':False,'TEST':False,'pilot':False,'strict_multimodal':False,
    'production_expansion':False,'physical_gates_changed':False,'frozen_goal_relabel':False}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_probe').rglob('*.py'))})


def inputs(source):
    old,ds,dev,rows,files=previous_inputs(source);source=Path(source).resolve()
    paths=[PREVIOUS/'frozen_tasks.json',PREVIOUS/'training_index.json',PREVIOUS/'independent_verification.json']
    for mode in ('v_only','goal_loss'):
        paths.extend(PREVIOUS/'learning'/mode/name for name in ('model.pt','codec.json','training.json'))
    for p in paths:files[os.path.relpath(p,source)]=file_digest(p)
    return old,ds,dev,rows,files


def targets():
    old=read_json(PREVIOUS/'frozen_tasks.json')['generation_entries'];result=[]
    index=read_json(PREVIOUS/'training_index.json')['records'];ids={r['sample_id'] for r in index if r['split']=='TRAIN'}
    for e in old:
        if e['split']!='TRAIN' or e['parent_sample_id'] not in ids:raise ValueError('non-TRAIN target')
        result.append({k:v for k,v in e.items() if k not in ('Z','source_arrays','source_record')})
    if len(result)!=8 or len({e['task']['task_hash'] for e in result})!=8:raise ValueError('unexpected TRAIN task set')
    return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('run identity changed')
        proof=read_json(ROOT/'ff_prior_probe/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_probe/evidence/tests.xml'):
            raise ValueError('software proof stale')
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        if physical['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']:raise ValueError('physical gates changed')
        entries=targets();frozen={'evaluation_entries':entries,'split':'TRAIN','DEV_hash':digest(dev),'VAL_executed':False,'TEST_executed':False}
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',frozen);atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        if read_json(out/'frozen_tasks.json')!=frozen:raise ValueError('frozen targets changed')
        ledger=Ledger(out,identity);deadline=time.time()+max(0,5400-previous_elapsed);torch.set_num_threads(2)
        bundles={'structured':None,'frozen':frozen_learning.load(FROZEN),'v_only':learning.load(PREVIOUS/'learning/v_only'),'goal_loss':learning.load(PREVIOUS/'learning/goal_loss')}
        bundles['mean_bow']=bundles['mean']=bundles['v_only']
        if digest(bundles['v_only'][1].state)!=digest(bundles['goal_loss'][1].state):raise ValueError('matched codec required')
        cache=data.RuntimeCache(canonical,initial,profile)
        for ti,e in enumerate(entries):
            if time.time()>=deadline:raise RuntimeError('RUN_WALL_BUDGET_EXHAUSTED')
            task=e['task'];sid=e['sample_id'];rr,ii,cc,transfer=cache.get(e)
            atomic_npz(out/'tasks'/sid/'initial_state.npz',values=ii.values)
            start=time.perf_counter();before=rr.calls.copy();base=public_goal_initializer(task,rr,ii)
            atomic_json(out/'transfer_audits'/f'{sid}.json',dict(transfer,public_base=base,shared_public_base_hash=digest(base),
                shared_setup_wall_s=time.perf_counter()-start,shared_initializer_calls={n:rr.calls[n]-before[n] for n in before}))
            for k in range(5):
                shift=(ti+k)%len(METHODS)
                for method in METHODS[shift:]+METHODS[:shift]:
                    key=f'candidate_{sid}_{method}_{k}';path=out/'candidates'/sid/method/f'{k:02d}.json'
                    if key in ledger.data['completed']:continue
                    ledger.begin(key,'candidate',5400,400);progress('TRAIN_residual_probe',task=ti+1,total=8,candidate=k+1,method=method)
                    r=candidate(method,k,task,base,20261113+ti*101+k*997,bundles[SPECS[method][0]],rr,ii,cc,profile,deadline)
                    r.update(sample_id=sid,split='TRAIN');atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
            if ti in PROTOCOL['witness_task_slots']:
                for method in METHODS:
                    key=f'execution_{sid}_{method}';d=out/'executions'/sid/method
                    if key in ledger.data['completed']:continue
                    records=[read_json(p) for p in sorted((out/'candidates'/sid/method).glob('*.json'))]
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(d/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                    ledger.begin(key,'execution',5400,20);before=rr.calls.copy();progress('preselected_TRAIN_torque_replay',task=ti+1,method=method)
                    if chosen:r=execute(d,task,chosen['Z'],rr,ii,cc,profile,deadline)
                    else:r={'status':'NO_CANDIDATE'};atomic_json(d/'outcome.json',r)
                    ledger.finish(r,{n:rr.calls[n]-before[n] for n in before})
            report(out,ledger,'RUNNING')
        if sources()!=identity['source_hashes'] or any(file_digest(source/p)!=h for p,h in files.items()):raise ValueError('input/runtime changed')
        incomplete=ledger.data.get('unresolved_call_accounting') or any(r['status'] in ('ERROR','INTERRUPTED','BLOCKED') for r in ledger.data['completed'].values())
        return report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
