"""Small qualified TRAIN-mode loop and same-budget frozen VAL comparison."""
from pathlib import Path
import os
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,Ledger,RunLock
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop import data as old_data
from ff_prior_loop.experiment import execute
from ff_prior_latent import learning as frozen_learning
from ff_prior_latent.experiment import inputs as old_inputs,sources as previous_sources,candidate as previous_candidate
from ff_prior_feasible.guidance import CONTRACT as GUIDE
from ff_prior_feasible.validation import CONTRACT as VALIDATE
from . import data,learning,sensitivity

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'artifacts/v7_1/qualified_guidance_loop_001'
FROZEN=ROOT/'artifacts/v7_1/latent_residual_smoke_001/learning'
METHODS=('structured_feasible','frozen_latent_diffusion','augmented_mean_spline','augmented_diffusion','goal_loss_diffusion')
PROTOCOL={'version':'qualified_modes_smoke_1','profile':'smoke','data':data.CONTRACT,'sensitivity':sensitivity.CONTRACT,
    'training':learning.CONTRACT,'guidance':GUIDE,'validation':VALIDATE,'methods':list(METHODS),
    'seed':20261067,'candidates_per_task':5,'same_tasks_seeds_as_previous':True,'VAL_tasks':10,
    'max_candidates':250,'max_selected_executions':50,'max_witness_executions':4,'max_new_training_rows':48,
    'candidate_coupled_ceiling':113,'candidate_integration_interval_ceiling':15872,'candidate_wall_ceiling_s':15.,
    'sensitivity_query_ceiling':7020,'sensitivity_interval_ceiling':898560,'wall_budget_s':5400,
    'selection':'minimum normalized world-goal error among eligible; one torque execution and replay, no replacement',
    'data_gate':'at least 80% of 48 admitted; each obstacle target >=2 distinct labels; 4/4 preselected execution/replay',
    'stability_gate':'each group >=90% eligible; each task >=4/5 eligible; 10/10 selected execution/replay',
    'advantage_gate':'stable; eligible/physical noninferior to BOTH strong spline and augmented mean; more physical tasks OR >=10% lower actual queries AND wall',
    'scope':'small TRAIN enrichment, exploratory original VAL; not strict multimodal certification',
    'production_expansion':False,'pilot':False,'study':False,'TEST':False,'DEV_executed':False,
    'physical_gates_changed':False,'frozen_goal_relabel':False}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_modes').rglob('*.py'))})


def inputs(source):
    old,ds,dev,rows,files=old_inputs(source);source=Path(source).resolve()
    for name in ('model.pt','codec.json','training.json'):
        p=FROZEN/name;files[os.path.relpath(p,source)]=file_digest(p)
    return old,ds,dev,rows,files


def candidate(method,k,task,base,seed,bundle,rr,ii,cc,profile,deadline):
    mapping={'structured_feasible':'structured_feasible','augmented_mean_spline':'learned_mean_spline'}
    if method not in METHODS:raise ValueError('unknown comparison method')
    if bundle is None and method!='structured_feasible':
        return {'status':'BLOCKED','eligible':False,'method':method,'candidate_id':k,'seed':seed,'task_hash':task['task_hash'],
            'public_base_hash':digest(base),'private_witness_used':False,'reason':'TRAINING_DEPENDENCY_BLOCKED',
            'model_calls':dict.fromkeys(rr.calls,0),'actual_network_calls':0,'repair_coupled_calls':0,
            'validation_coupled_calls':0,'integration_intervals':0,'wall_s':0.,'cpu_s':0.}
    r=previous_candidate(mapping.get(method,'latent_diffusion_feasible'),k,task,base,seed,bundle,rr,ii,cc,profile,deadline)
    r['method']=method;return r


def enrich(entries,cache,out,profile,ledger,deadline,progress):
    for e in entries:
        for k in range(6):
            key=e['sample_id']+f'_v{k}';path=out/'generation'/f'{key}.json'
            if key in ledger.data['completed']:continue
            ledger.begin(key,'generation',5400,48);progress('fixed_TRAIN_world_modes',target=e['slot']+1,variant=k+1,total=48)
            result=data.generate(e,k,cache,out,profile,deadline);atomic_json(path,result)
            ledger.finish({'status':result['status']},result['calls'])
    for slot,k in data.CONTRACT['witness_slots']:
        e=entries[slot];sid=e['sample_id']+f'_v{k}';key='witness_'+sid;directory=out/'witness_executions'/sid
        if key in ledger.data['completed']:continue
        ledger.begin(key,'witness',5400,4);progress('preselected_TRAIN_torque_replay',sample=sid)
        path=out/'records'/f'{sid}.json';calls={}
        if path.exists():
            row=read_json(path);rr,ii,cc,_=cache.get(e);before=rr.calls.copy()
            with np.load(out/row['arrays']) as a:z=a['Z'].copy()
            result=execute(directory,e['task'],z,rr,ii,cc,profile,deadline)
            calls={n:rr.calls[n]-before[n] for n in before};row['execution_status']='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL'
            row['execution_report']=f'witness_executions/{sid}/report.json'
            if row['execution_status']=='PASS':row['validation_level']='EXECUTION_VERIFIED'
            atomic_json(path,row)
        else:
            result={'status':'BLOCKED','reason':'PRESELECTED_CANDIDATE_NOT_ADMITTED'};atomic_json(directory/'outcome.json',result)
        ledger.finish(result,calls)


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('run identity changed')
        proof=read_json(ROOT/'ff_prior_modes/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_modes/evidence/tests.xml'):
            raise ValueError('software proof missing or stale')
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        if physical['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']:raise ValueError('physical gates changed')
        original=read_json(PREVIOUS/'training_index.json')['records'];entries=data.select(original)
        targets=read_json(PREVIOUS/'frozen_tasks.json')['evaluation_entries']
        frozen={'generation_entries':entries,'evaluation_entries':targets,'frozen_DEV_hash':digest(dev),
            'all_targets_preexisting':True,'goal_relabelled':False,'frozen_before_generation_training_comparison':True}
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',frozen);atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in sources():archive.write(ROOT/p,p)
        if digest(read_json(out/'frozen_tasks.json'))!=digest(frozen):raise ValueError('frozen targets changed')
        ledger=Ledger(out,identity);deadline=time.time()+max(0,5400-previous_elapsed)
        for e in entries:
            neighbours=old_data.neighbours(e['task'],'TRAIN',original,dev)
            if neighbours:raise ValueError('existing TRAIN target has cross-split goal neighbours: '+str(neighbours))
        cache=data.RuntimeCache(canonical,initial,profile)
        enrich(entries,cache,out,profile,ledger,deadline,progress)
        index=data.combined(out);atomic_json(out/'training_index.json',{'records':index,'actual_loader_input':True,'TEST_read':False})
        arrays,rows=data.arrays(index,cache,out,progress);codec=learning.fit_codec(out/'learning',arrays)
        matrices,sensitivity_report=sensitivity.prepare(out,arrays,rows,codec,cache,ledger,deadline,progress)
        for mode in ('v_only','goal_loss'):
            learning.train(out/'learning'/mode,arrays,codec,matrices,mode,sensitivity_report,progress,deadline)
        bundles={'frozen_latent_diffusion':frozen_learning.load(FROZEN)}
        for mode,method in [('v_only','augmented_diffusion'),('goal_loss','goal_loss_diffusion')]:
            bundles[method]=learning.load(out/'learning'/mode) if (out/'learning'/mode/'training.json').exists() else None
        bundles['augmented_mean_spline']=bundles['augmented_diffusion'];bundles['structured_feasible']=None
        for ti,e in enumerate(targets):
            rr,ii,cc,transfer=old_data.runtime(e,canonical,initial);task=e['task'];sid=e['sample_id']
            setup=time.perf_counter();before=rr.calls.copy();base=public_goal_initializer(task,rr,ii)
            atomic_json(out/'transfer_audits'/f'{sid}.json',dict(transfer,shared_public_base_hash=digest(base),
                shared_setup_wall_s=time.perf_counter()-setup,shared_initializer_calls={n:rr.calls[n]-before[n] for n in before}))
            for method in METHODS[ti%5:]+METHODS[:ti%5]:
                results=[]
                for k in range(5):
                    key=f'candidate_{sid}_{method}_{k}';path=out/'candidates'/sid/method/f'{k:02d}.json'
                    if key not in ledger.data['completed']:
                        ledger.begin(key,'candidate',5400,250);progress('paired_VAL_candidates',task=ti+1,total=10,method=method,candidate=k+1)
                        result=candidate(method,k,task,base,20261067+ti*101+k*997,bundles[method],rr,ii,cc,profile,deadline)
                        result.update(sample_id=sid,split='VAL',group=e['group']);atomic_json(path,result)
                        ledger.finish({'status':result['status']},result['model_calls'])
                    if path.exists():results.append(read_json(path))
                key=f'execution_{sid}_{method}';directory=out/'executions'/sid/method
                if key in ledger.data['completed']:continue
                chosen=min((r for r in results if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,'task_hash':task['task_hash'],
                    'selection_before_execution':True,'rule':PROTOCOL['selection']})
                ledger.begin(key,'execution',5400,50);before=rr.calls.copy();progress('selected_torque_replay',task=ti+1,method=method)
                if chosen:result=execute(directory,task,chosen['Z'],rr,ii,cc,profile,deadline)
                else:
                    result={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',result)
                ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
            report(out,ledger,'RUNNING')
        if any(file_digest(source/p)!=h for p,h in files.items()) or sources()!=identity['source_hashes']:raise ValueError('source/runtime changed')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,
            'original_DEV_unchanged':True,'TEST_arrays_read':False,'physical_gates_unchanged':True,'goal_relabelled':False,'production_expansion':False})
        incomplete=ledger.data.get('unresolved_call_accounting') or any(r['status'] in ('ERROR','INTERRUPTED','BLOCKED') for r in ledger.data['completed'].values())
        return report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
