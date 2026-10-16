"""Small, frozen complex TRAIN/VAL comparison; original DEV is read-only."""
from pathlib import Path
from collections import Counter
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.pipeline import audit, acceptance
from ff_prior.storage import atomic_json, atomic_npz, read_json, file_digest, digest, Ledger, RunLock
from ff_prior.spline import Reference
from ff_prior.scenes import SceneSpec
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.guidance import public_goal_initializer
from ff_prior.physical_audit import audit_execution
from ff_prior_next import learning
from ff_prior_cost.experiment import inputs, sources as previous_sources, validate, CHECKPOINT
from ff_prior_cost.projection import Projector, CONTRACT as PROJECTION
from ff_prior_cost.sampling import sample
from . import tasks

ROOT=Path(__file__).resolve().parents[1]
METHODS=('structured_narrow','structured_wide','frozen_diffusion')
PROTOCOL={'version':'complex_train_val_smoke_1','profile':'smoke','generation':tasks.CONTRACT,
    'methods':list(METHODS),'candidates_per_task':5,'max_tasks':12,'max_candidates':180,
    'max_executions_including_witnesses':48,'wall_budget_s':1800,'candidate_wall_ceiling_s':8.,
    'candidate_total_coupled_ceiling':113,'projection':PROJECTION,'ddim_steps':16,
    'narrow_sigma_rad':.015,'wide_first_harmonic_sigma_rad':.10,'wide_second_harmonic_sigma_rad':.05,
    'seed':20261047,'retraining':False,'checkpoint':'fixed_train_consistency_001/learning',
    'witness_required':'MODEL_VALID plus actual torque execution and independent replay',
    'selection':'minimum frozen-goal error among eligible candidates; one execution, no replacement',
    'primary_claim':'at least one VAL task fails both spline budgets but diffusion passes actual execution and replay',
    'supporting_claim':'same paired tasks/candidates/query/wall ceilings; report all consumed work and failures',
    'scope':'exploratory TRAIN/VAL; existing weights frozen; not a new training-benefit or TEST claim',
    'collision_optimization':False,'no_private_witness_for_planner':True,
    'DEV_rerun':False,'TEST':False,'pilot':False,'study':False,'strict_multimodal':False,
    'physical_gates_changed':False,'failure_replacement':False,'frozen_goal_relabel':False}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
        for p in sorted((ROOT/'ff_prior_complex').glob('*.py'))})


def proposal(base,method,seed):
    rng=np.random.default_rng(seed)
    if method=='structured_narrow':
        return base+.015*np.sin(np.linspace(0,np.pi,19))[:,None]*rng.normal(size=(1,17))
    if method!='structured_wide': raise ValueError('unknown structured method')
    s=np.r_[np.linspace(0,1,20)[1:-1],1.]
    return base+.10*np.sin(np.pi*s)[:,None]*rng.normal(size=(1,17))+.05*np.sin(2*np.pi*s)[:,None]*rng.normal(size=(1,17))


def candidate(method,k,task,base,seed,bundle,robot,initial,collision,profile,deadline):
    assert_frozen(task); start=time.perf_counter(); cpu=time.process_time(); before=robot.calls.copy()
    deadline=min(deadline,time.time()+PROTOCOL['candidate_wall_ceiling_s'])
    result={'status':'ERROR','eligible':False,'method':method,'candidate_id':k,'seed':seed,
        'task_hash':task['task_hash'],'public_base_hash':digest(base),'private_witness_fields_accessed':False,
        'shared_coupled_ceiling':113,'shared_wall_ceiling_s':PROTOCOL['candidate_wall_ceiling_s']}
    projector=Projector(task,robot,initial,collision,profile,deadline,schedule=(15,))
    try:
        if method=='frozen_diffusion': z,info=sample(bundle,base,task,seed,projector)
        else:
            z=proposal(base,method,seed); raw=digest(z); z,guide=projector(z,15)
            info={'actual_network_calls':0,'raw_proposal_hash':raw,'guidance':[guide]}
        result.update(Z=z,sampling=info,repair_coupled_calls=projector.used)
        result.update(validate(z,task,robot,initial,collision,profile,deadline))
        result['status']='DONE'
        if time.time()>deadline: raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error:
        result.update(status='ERROR',eligible=False,reason=repr(error))
    calls={n:robot.calls[n]-before[n] for n in before}
    result.update(model_calls=calls,repair_coupled_calls=projector.used,
        validation_coupled_calls=calls['coupled']-projector.used,
        wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)
    if calls['coupled']>113 or projector.used>110 or calls['coupled']-projector.used>3:
        result.update(status='ERROR',eligible=False,reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task)
    return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import write_report
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    old,_,dev,rows,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity): raise ValueError('run identity conflict')
        proof=read_json(ROOT/'ff_prior_complex/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources(): raise ValueError('software proof stale')
        if proof['junit_sha256']!=file_digest(ROOT/'ff_prior_complex/evidence/tests.xml'): raise ValueError('test proof changed')
        report,robot,initial,collision=audit(old['config']); profile=acceptance(old['config'])
        if report['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical gates changed')
        selected=tasks.select_parents(rows)
        parents=[{'sample_id':r['sample_id'],'family_id':r['family_id'],'split':s} for r,s in selected]
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':report['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'parent_selection.json',parents); atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in identity['source_hashes']: archive.write(ROOT/p,p)
        if read_json(out/'parent_selection.json')!=parents: raise ValueError('parent selection changed')
        ledger=Ledger(out,identity); deadline=time.time()+max(0,1800-previous_elapsed)
        bundle=learning.load(CHECKPOINT,initial.model_hash,initial.state_id)
        # Generate and freeze every slot before any competing planner is sampled.
        entries=[]
        for i,(row,split) in enumerate(selected):
            for kind in tasks.CONTRACT['kinds']:
                key=f'generation_{split}_{i}_{kind}'; path=out/'generation'/f'{split}_{i}_{kind}.json'
                if key not in ledger.data['completed']:
                    ledger.begin(key,'generation',1800,12); before=robot.calls.copy()
                    progress('generate_and_freeze',split=split,parent=i,kind=kind)
                    extra_calls=Counter()
                    try:
                        result=tasks.generate(row,split,kind,i,source,robot,initial,collision,profile,deadline)
                        if result['status']=='GENERATED':
                            scene=SceneSpec.from_dict(result['scene'])
                            rr,ii,cc=tasks.scene_runtime(scene) if scene.objects else (robot,initial,collision)
                            result['transfer']=tasks.transfer_audit(robot,initial,rr,ii)
                            task=tasks.task_for(rr,ii,cc,profile,result['goals'],result['duration'])
                            result['task']=task; result['frozen_before_comparison']=True
                            ref=Reference(result['Z'],result['initial_features'],result['duration'])
                            validation=validate(result['Z'],task,rr,ii,cc,profile,deadline)
                            if rr is not robot: extra_calls.update(rr.calls)
                            result['witness_model']=validation
                            result['status']='FROZEN' if validation['eligible'] else 'BLOCKED'
                            if not validation['eligible']: result['reason']='WITNESS_FAILED_IMMUTABLE_SCENE_GATES'
                            directory=out/'tasks'/result['sample_id']
                            atomic_json(directory/'task.json',task); atomic_json(directory/'scene.json',result['scene'])
                            atomic_json(directory/'pair_policy.json',cc.policy)
                            atomic_npz(directory/'initial_state.npz',values=ii.values)
                            atomic_npz(directory/'witness.npz',**ref.arrays())
                    except Exception as error:
                        result={'status':'ERROR','reason':repr(error),'split':split,'kind':kind,'parent_sample_id':row['sample_id']}
                    atomic_json(path,result)
                    calls={n:robot.calls[n]-before[n] for n in before}
                    extra_calls.update(result.get('screen',{}).get('scratch_calls',{}))
                    calls={n:v+extra_calls[n] for n,v in calls.items()}
                    ledger.finish({'status':result['status']},calls)
                if path.exists(): entries.append(read_json(path))
                else: entries.append({'status':'BLOCKED','reason':'INTERRUPTED_RESERVED_GENERATION','split':split,'kind':kind})
        freeze={'entries':entries,'digest':digest(entries),'all_slots':12,'before_any_comparison':True}
        if (out/'frozen_complex_tasks.json').exists():
            if read_json(out/'frozen_complex_tasks.json')!=freeze: raise ValueError('frozen tasks mutated')
        else: atomic_json(out/'frozen_complex_tasks.json',freeze)
        progress('all_tasks_frozen',usable=sum(e['status']=='FROZEN' for e in entries),slots=12)
        for ti,e in enumerate(entries):
            if e['status']!='FROZEN': continue
            split=e['split']; sid=e['sample_id']; task=e['task']; scene=SceneSpec.from_dict(e['scene'])
            rr,ii,cc=tasks.scene_runtime(scene) if scene.objects else (robot,initial,collision)
            tasks.transfer_audit(robot,initial,rr,ii); assert_task_compatible(task,rr,ii,cc,profile)
            key='witness_'+sid; directory=out/'witness_executions'/sid
            if key not in ledger.data['completed']:
                ledger.begin(key,'witness_execution',1800,12); before=rr.calls.copy()
                progress('independent_witness_torque_and_replay',split=split,task=ti+1,kind=e['kind'])
                try:
                    p=audit_execution(task,profile,rr,ii,Reference(e['Z'],e['initial_features'],e['duration']),directory,deadline)
                    result={'status':p['level']}
                except Exception as error: result={'status':'ERROR','reason':repr(error)}
                atomic_json(directory/'outcome.json',result)
                ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
            p=read_json(directory/'outcome.json') if (directory/'outcome.json').exists() else {'status':'BLOCKED'}
            if p['status']!='EXECUTION_VERIFIED':
                progress('dependent_comparison_BLOCKED',sample_id=sid,reason='WITNESS_NOT_EXECUTION_VERIFIED'); continue
            augmented={'sample_id':sid,'parent_sample_id':e['parent_sample_id'],'family_id':e['family_id'],
                'split':split,'kind':e['kind'],'condition':task['condition'],'task':task,'scene':e['scene'],
                'arrays':f'tasks/{sid}/witness.npz','validation_level':'EXECUTION_VERIFIED','execution_status':'PASS',
                'training_eligible':split=='TRAIN','data_role':'COMPLEX_WITNESS_AUGMENTATION',
                'checkpoint_used_this_new_row':False,'execution_report':f'witness_executions/{sid}/report.json'}
            atomic_json(out/'records'/f'{sid}.json',augmented)
            base=public_goal_initializer(task,rr,ii)
            # Rotate method order by task; fixed seeds are paired across all methods.
            order=METHODS[ti%3:]+METHODS[:ti%3]
            for method in order:
                candidates=[]
                for k in range(5):
                    key=f'candidate_{sid}_{method}_{k}'; path=out/'candidates'/sid/method/f'{k:02d}.json'
                    if key not in ledger.data['completed']:
                        ledger.begin(key,'candidates',1800,180)
                        progress('paired_candidates',split=split,task=ti+1,kind=e['kind'],method=method,candidate=k+1)
                        result=candidate(method,k,task,base,20261047+ti*101+k*997,bundle,rr,ii,cc,profile,deadline)
                        result.update(split=split,kind=e['kind'],sample_id=sid)
                        atomic_json(path,result); ledger.finish({'status':result['status']},result['model_calls'])
                    if path.exists(): candidates.append(read_json(path))
                key=f'execution_{sid}_{method}'; directory=out/'executions'/sid/method
                if key in ledger.data['completed']: continue
                eligible=[r for r in candidates if r.get('eligible')]
                chosen=min(eligible,key=lambda r:(r['score'],r['candidate_id'])) if eligible else None
                atomic_json(directory/'selection.json',{'task_hash':task['task_hash'],
                    'selected_id':chosen['candidate_id'] if chosen else None,'selection_before_execution':True,
                    'rule':PROTOCOL['selection']})
                ledger.begin(key,'execution',1800,36); before=rr.calls.copy()
                progress('selected_torque_and_replay',split=split,task=ti+1,method=method,selected=chosen['candidate_id'] if chosen else None)
                try:
                    if chosen:
                        p=audit_execution(task,profile,rr,ii,Reference(chosen['Z'],task['condition']['initial_features'],task['condition']['duration']),directory,deadline)
                        result={'status':p['level']}
                    else: result={'status':'NO_CANDIDATE'}
                except Exception as error: result={'status':'ERROR','reason':repr(error)}
                atomic_json(directory/'outcome.json',result)
                ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
            write_report(out,ledger,'RUNNING')
        if any(file_digest(source/p)!=h for p,h in files.items()): raise ValueError('input mutated')
        if sources()!=identity['source_hashes']: raise ValueError('runtime source mutated')
        if read_json(out/'frozen_complex_tasks.json')!=freeze: raise ValueError('task freeze changed')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,
            'original_DEV_unchanged':True,'DEV_executed':False,'TEST_arrays_read':False,
            'all_comparison_tasks_frozen_before_candidates':True,'physical_gates_unchanged':True})
        incomplete=any(e['status']!='FROZEN' for e in entries) or ledger.data.get('unresolved_call_accounting') or any(
            r['status'] in ('ERROR','BLOCKED','INTERRUPTED','EXECUTION_FAILED') for r in ledger.data['completed'].values())
        return write_report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
