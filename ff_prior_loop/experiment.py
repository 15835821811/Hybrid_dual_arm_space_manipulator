"""Bounded generator -> learning -> collision-guided sampling -> torque/replay loop."""
from pathlib import Path
import os
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,Ledger,RunLock
from ff_prior.pipeline import audit,acceptance
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.physical_audit import audit_execution
from ff_prior_next import learning
from ff_prior_cost.experiment import inputs as old_inputs,validate,CHECKPOINT
from ff_prior_cost.sampling import sample
from ff_prior_complex.experiment import sources as previous_sources
from . import data,training
from .guidance import JointProjector,CONTRACT as GUIDE

ROOT=Path(__file__).resolve().parents[1]
METHODS=('structured_joint','frozen_diffusion_joint','trained_diffusion_joint','trained_diffusion_goal_only')
PROTOCOL={'version':'qualified_guidance_loop_smoke_1','profile':'smoke','data':data.CONTRACT,'guidance':GUIDE,
    'training':training.CONTRACT,'methods':list(METHODS),'candidate_seeds':5,'max_candidates':200,
    'task_groups':{'old_complex_VAL':6,'new_origin_VAL':4},'candidate_coupled_ceiling':113,
    'candidate_wall_ceiling_s':15.,'ddim_steps':16,'structured_bow_sigma_rad':.03,
    'max_witness_executions':8,'max_selected_executions':40,'wall_budget_s':3600,
    'seed':20261067,'selection':'minimum goal error among eligible; one execution; no replacement',
    'expansion_gate':{'data_eligible_fraction':.85,'trained_eligible_each_group':.9,'trained_eligible_each_task':4,
        'witness_executions':8,'trained_selected_executions':10,'training_all_rows_seen':True},
    'conditional_expansion':{'new_TRAIN_families':4,'max_attempts':12,'max_executions':4,'retrain':False},
    'primary_claim':'qualified data actually trains a stable specified-world-goal guided sampler in exploratory VAL',
    'supporting_claim':'collision guidance contribution isolated with same weights and same proposal budget',
    'benefit':'report paired wins/losses and actual costs; stable loop does not imply diffusion advantage',
    'pilot':False,'study':False,'TEST':False,'DEV_executed':False,'strict_multimodal':False,
    'physical_gates_changed':False,'frozen_goal_relabel':False,'old_results_changed':False}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_loop').glob('*.py'))})


def inputs(source):
    old,ds,dev,rows,files=old_inputs(source); source=Path(source).resolve()
    paths=[data.OLD/'combined_train_val_index.json',data.OLD/'frozen_complex_tasks.json']
    for e in data.old_index(): paths.extend([ROOT/e['record'],ROOT/e['arrays']])
    for p in paths: files[os.path.relpath(p,source)]=file_digest(p)
    return old,ds,dev,rows,files


def candidate(method,k,task,base,seed,bundle,rr,ii,cc,profile,deadline):
    assert_task_compatible(task,rr,ii,cc,profile); before=rr.calls.copy(); start=time.perf_counter(); cpu=time.process_time()
    deadline=min(deadline,time.time()+15.)
    projector=JointProjector(task,rr,ii,cc,profile,deadline,collision_enabled=method!='trained_diffusion_goal_only')
    result={'status':'ERROR','eligible':False,'method':method,'candidate_id':k,'seed':seed,'task_hash':task['task_hash'],
        'public_base_hash':digest(base),'private_witness_used':False,'shared_query_ceiling':113,'shared_wall_ceiling_s':15.}
    try:
        if method=='structured_joint':
            s=np.r_[np.linspace(0,1,20)[1:-1],1.]
            z=base+.03*np.sin(np.pi*s)[:,None]*np.random.default_rng(seed).normal(size=(1,17))
            z,info=projector(z,15); sampling={'actual_network_calls':0,'guidance':[info]}
        else:
            if bundle is None: raise RuntimeError('TRAINED_CHECKPOINT_BLOCKED')
            z,sampling=sample(bundle,base,task,seed,projector)
        result.update(Z=z,sampling=sampling,**validate(z,task,rr,ii,cc,profile,deadline)); result['status']='DONE'
        if time.time()>deadline: raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error: result.update(status='ERROR',eligible=False,reason=repr(error))
    calls={n:rr.calls[n]-before[n] for n in before}
    result.update(model_calls=calls,repair_coupled_calls=projector.used,validation_coupled_calls=calls['coupled']-projector.used,
        wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)
    if calls['coupled']>113 or projector.used>110 or calls['coupled']-projector.used>3:
        result.update(status='ERROR',eligible=False,reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task); return result


def execute(directory,task,z,rr,ii,cc,profile,deadline):
    try:
        p=audit_execution(task,profile,rr,ii,Reference(z,task['condition']['initial_features'],task['condition']['duration']),directory,deadline)
        result={'status':p['level']}
    except Exception as error: result={'status':'ERROR','reason':repr(error)}
    atomic_json(directory/'outcome.json',result); return result


def freeze(out,parents,canonical,profile,index,dev,ledger,deadline,progress,expansion=False):
    entries=[]
    for j,parent in enumerate(parents):
        key=('expansion_' if expansion else '')+f'freeze_{j:02d}'; path=out/'generation'/f'{key}.json'
        if key not in ledger.data['completed']:
            ledger.begin(key,'freeze',3600,20); progress('freeze_world_tasks',slot=j,split=parent['split'],expansion=expansion)
            calls={}
            try:
                result,calls=data.freeze_parent(parent,j,canonical,profile,index,dev,out,deadline,expansion)
                if result['status']=='FROZEN':
                    from ff_prior.coupled_predictor import pose_errors
                    for earlier in entries:
                        if earlier['status']!='FROZEN' or earlier['split']==result['split']: continue
                        p,r=pose_errors(earlier['task']['condition']['goals'],result['task']['condition']['goals'])
                        if max(p)<.01 and max(r)<np.deg2rad(1):
                            result.update(status='BLOCKED',reason='NEW_CROSS_SPLIT_GOAL_NEIGHBOUR',neighbour=earlier['sample_id'])
                            break
            except Exception as error: result={'status':'ERROR','reason':repr(error),'split':parent['split'],'sample_id':key}
            atomic_json(path,result); ledger.finish({'status':result['status']},calls)
        entries.append(read_json(path) if path.exists() else {'status':'BLOCKED','sample_id':key,'split':parent['split']})
    return entries


def generate(out,entries,canonical,initial,profile,ledger,deadline,progress):
    for e in entries:
        if e['status']!='FROZEN': continue
        rr,ii,cc,_=data.runtime(e,canonical,initial)
        for k in range(3):
            key='generate_'+e['sample_id']+f'_v{k}'; path=out/'generation'/f'{key}.json'
            if key in ledger.data['completed']: continue
            ledger.begin(key,'generation',3600,60); before=rr.calls.copy(); progress('fixed_world_goal_labels',sample=e['sample_id'],variant=k)
            try:
                result=data.generate_variant(e,k,rr,ii,cc,profile,out,deadline)
                result['status']='MODEL_VALID' if result['eligible'] else 'REJECTED'
            except Exception as error: result={'status':'ERROR','reason':repr(error),'eligible':False}
            atomic_json(path,result); ledger.finish({'status':result['status']},{n:rr.calls[n]-before[n] for n in before})


def witnesses(out,entries,canonical,initial,profile,ledger,deadline,progress):
    for e in entries:
        sid=e['sample_id']+'_v0'; key='witness_'+sid; directory=out/'witness_executions'/sid
        if key in ledger.data['completed']: continue
        ledger.begin(key,'witness',3600,12); progress('preselected_witness_torque_replay',sample=sid); calls={}
        path=out/'records'/f'{sid}.json'
        if path.exists():
            row=read_json(path); rr,ii,cc,_=data.runtime(e,canonical,initial); before=rr.calls.copy()
            with np.load(out/row['arrays']) as a: z=a['Z'].copy()
            result=execute(directory,e['task'],z,rr,ii,cc,profile,deadline)
            calls={n:rr.calls[n]-before[n] for n in before}
            row['execution_report']=f'witness_executions/{sid}/report.json'
            row['execution_status']='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL'
            if row['execution_status']=='PASS': row['validation_level']='EXECUTION_VERIFIED'
            atomic_json(path,row)
        else:
            result={'status':'BLOCKED','reason':'PRESELECTED_WITNESS_NOT_MODEL_VALID'}; atomic_json(directory/'outcome.json',result)
        ledger.finish(result,calls)


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report,decision
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity): raise ValueError('run identity changed')
        proof=read_json(ROOT/'ff_prior_loop/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_loop/evidence/tests.xml'):
            raise ValueError('software proof missing or stale')
        physical,canonical,initial,collision=audit(old['config']); profile=acceptance(old['config'])
        if physical['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical gates changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in sources(): archive.write(ROOT/p,p)
        ledger=Ledger(out,identity); deadline=time.time()+max(0,3600-previous_elapsed); index=data.old_index()
        parents=data.select(index); entries=freeze(out,parents,canonical,profile,index,dev,ledger,deadline,progress)
        old_tasks=[dict(e,group='old_complex') for e in read_json(data.OLD/'frozen_complex_tasks.json')['entries'] if e['split']=='VAL']
        targets=old_tasks+[e for e in entries if e['split']=='VAL']
        frozen={'training_generation_entries':entries,'evaluation_entries':targets,'before_new_training_and_comparison':True}
        if (out/'frozen_tasks.json').exists():
            if read_json(out/'frozen_tasks.json')!=frozen: raise ValueError('task freeze changed')
        else: atomic_json(out/'frozen_tasks.json',frozen)
        generate(out,entries,canonical,initial,profile,ledger,deadline,progress)
        audited=[e for e in entries if e['split']=='TRAIN'][:4]+[e for e in entries if e['split']=='VAL']
        witnesses(out,audited,canonical,initial,profile,ledger,deadline,progress)
        train_index=data.combined(out,include_expansion=False)
        if (out/'training_index.json').exists():
            if read_json(out/'training_index.json')['records']!=train_index: raise ValueError('training index changed')
        else: atomic_json(out/'training_index.json',{'records':train_index,'actual_loader_input':True,'TEST_read':False})
        trained=None
        if not (out/'training_failure.json').exists():
            try:
                arrays=data.training_arrays(train_index,canonical,initial,profile,out,progress)
                result=training.train(out/'learning',arrays,initial.model_hash,progress,deadline)
                if result['status']=='PASS': trained=training.load(out/'learning',result['metadata']['scope'])
            except Exception as error: atomic_json(out/'training_failure.json',{'status':'BLOCKED','reason':repr(error)})
        frozen_bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
        for ti,e in enumerate(targets):
            if e['status']!='FROZEN': continue
            rr,ii,cc,transfer=data.runtime(e,canonical,initial); task=e['task']; assert_task_compatible(task,rr,ii,cc,profile)
            sid=e['sample_id']; atomic_json(out/'transfer_audits'/f'{sid}.json',transfer)
            base=public_goal_initializer(task,rr,ii)
            for method in METHODS[ti%4:]+METHODS[:ti%4]:
                results=[]
                for k in range(5):
                    key=f'candidate_{sid}_{method}_{k}'; path=out/'candidates'/sid/method/f'{k:02d}.json'
                    if key not in ledger.data['completed']:
                        ledger.begin(key,'candidate',3600,200); progress('paired_VAL_candidates',task=ti+1,total=10,method=method,candidate=k+1)
                        bundle=frozen_bundle if method=='frozen_diffusion_joint' else trained
                        result=candidate(method,k,task,base,20261067+ti*101+k*997,bundle,rr,ii,cc,profile,deadline)
                        result.update(sample_id=sid,split='VAL',group=e['group']); atomic_json(path,result)
                        ledger.finish({'status':result['status']},result['model_calls'])
                    if path.exists(): results.append(read_json(path))
                key=f'execution_{sid}_{method}'; directory=out/'executions'/sid/method
                if key in ledger.data['completed']: continue
                selected=min((r for r in results if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                atomic_json(directory/'selection.json',{'selected_id':selected['candidate_id'] if selected else None,'task_hash':task['task_hash'],
                    'selection_before_execution':True,'rule':PROTOCOL['selection']})
                ledger.begin(key,'execution',3600,40); before=rr.calls.copy(); progress('selected_torque_replay',task=ti+1,method=method)
                if selected: result=execute(directory,task,selected['Z'],rr,ii,cc,profile,deadline)
                else:
                    result={'status':'NO_CANDIDATE'}; atomic_json(directory/'outcome.json',result)
                ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
            report(out,ledger,'RUNNING')
        gate=decision(out); atomic_json(out/'expansion_decision.json',gate)
        if gate['passed']:
            progress('conditional_bounded_expansion',attempts=12,split='TRAIN')
            more=freeze(out,data.select(index,True),canonical,profile,index,dev,ledger,deadline,progress,True)
            atomic_json(out/'expansion_tasks.json',more)
            generate(out,more,canonical,initial,profile,ledger,deadline,progress)
            witnesses(out,more,canonical,initial,profile,ledger,deadline,progress)
        else: progress('expansion_BLOCKED_by_frozen_gate',checks=gate['checks'])
        atomic_json(out/'final_dataset_index.json',{'records':data.combined(out),'expanded':gate['passed'],
            'no_batch_execution_upgrade':True,'expanded_rows_used_by_this_checkpoint':False})
        if any(file_digest(source/p)!=h for p,h in files.items()) or sources()!=identity['source_hashes']: raise ValueError('input/runtime source changed')
        if read_json(out/'frozen_tasks.json')!=frozen: raise ValueError('frozen tasks changed')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,'original_DEV_unchanged':True,
            'DEV_executed':False,'TEST_arrays_read':False,'physical_gates_unchanged':True,'frozen_goal_relabelled':False})
        return report(out,ledger,'FINISHED' if gate['passed'] else 'FINISHED_WITH_BLOCKED_ITEMS')
