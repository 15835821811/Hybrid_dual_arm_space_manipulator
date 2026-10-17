from pathlib import Path
import subprocess,time,traceback,zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_coverage_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_origin_learn_bounded.mean import fit as fit_mean
from ff_prior_origin_learn_bounded.learning import load as old_load
from ff_prior_generalization.experiment import execute
from ff_prior_generalization.sampling import candidate
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from .protocol import ROOT,SOURCE,METHODS,PROTOCOL as P
from .identity import sources,identity_for
from . import data,learning

def capacity(out,deadline):
    if time.time()>=deadline:raise RuntimeError('TOTAL_WALL_BUDGET_EXHAUSTED')
    if sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())>=P['max_output_bytes']:
        raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')

def blocked_candidate(method,k,seed,task,base,reason):
    return {'status':'BLOCKED','eligible':False,'method':method,'candidate_id':k,'seed':seed,
            'task_hash':task['task_hash'],'public_base_hash':digest(base),'private_witness_used':False,
            'reason':reason,'model_calls':{},'actual_network_calls':0,'wall_s':0.}

def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    from .audit_results import verify,audit_data
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    deadline=time.time()+max(0,P['wall_budget_s']-previous_elapsed)
    identity,(config,old_index,entries,targets)=identity_for(source)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_world_endpoint/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_world_endpoint/evidence/tests.xml'):
            raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:raise ValueError('physical gates changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'joint_lower':canonical.lower,'joint_upper':canonical.upper,
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',{'entries':entries,'before_training':True,'before_comparison':True,
                'new_validation_role':'REUSED_DEV; no independent generalization claim'})
            atomic_json(out/'inherited_accounting.json',{'parent_status':read_json(source/'status.json'),
                'parent_summary':read_json(source/'summary.json'),'parent_ledger_sha256':file_digest(source/'ledger.json'),
                'prior_history':read_json(source/'inherited_accounting.json')})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'identity_hash':digest(identity)})
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=P['runtime_cache_entries'],
            max_scene_models=P['runtime_scene_models'],max_private_bytes=P['max_process_private_bytes'],
            min_available_bytes=P['minimum_available_memory_bytes'])
        index,new_ids,data_report=data.generate(out,targets,old_index,cache,profile,ledger,
            min(deadline-P['training_reserve_for_smoke_s'],time.time()+1800),progress)
        data_proof=audit_data(out,targets,ledger.data)
        atomic_json(out/'data_verification.json',data_proof)
        arrays,rows=data.load_arrays(out,index,cache,progress)
        old_bundle=old_load(source/'learning/goal_loss');codec=old_bundle[1]
        mean_codec=fit_mean(out/'learning',arrays,codec,new_ids)
        try:
            if data_report['status']!='PASS' or data_proof['status']!='PASS':raise ValueError('fixed TRAIN qualification blocked')
            training=learning.train(out/'learning/goal_loss',arrays,rows,targets,new_ids,cache,ledger,progress,
                deadline-P['training_reserve_for_smoke_s'])
        except Exception as error:
            training={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
            if ledger.data['inflight']:
                ledger.data.setdefault('unresolved_call_accounting',[]).append(ledger.data['inflight']['id'])
                ledger.finish({'status':'BLOCKED','reason':repr(error)},{})
            atomic_json(out/'training_failure.json',training)
        new_bundle=learning.load(out/'learning/goal_loss') if training['status']=='PASS' else None
        atomic_json(out/'checkpoint_binding.json',{'old':file_digest(source/'learning/goal_loss/model.pt'),
            'new':file_digest(out/'learning/goal_loss/model.pt') if new_bundle else None,
            'fixed_codec':digest(codec.state),'mean_codec_sha256':file_digest(out/'learning/weighted_mean.json'),
            'validation_checkpoint_selection':False})
        smoke_deadline=deadline-P['diagnostic_reserve_s']
        for sj,seed in enumerate(P['sampling_seeds']):
            for entry in entries:
                ti=entry['comparison_slot'];sid=entry['sample_id'];task=entry['task']
                rr,ii,cc,proof=cache.get(entry);base=public_goal_initializer(task,rr,ii)
                atomic_npz(out/'comparison_tasks'/sid/'initial_state.npz',values=ii.values)
                atomic_json(out/'transfer_audits'/f'{sid}.json',dict(proof,public_base_hash=digest(base)))
                for method in METHODS[ti%4:]+METHODS[:ti%4]:
                    bundle=(None if method=='structured' else old_bundle if method=='frozen_diffusion'
                            else (old_bundle[0],mean_codec) if method=='learned_mean' else new_bundle)
                    records=[]
                    for k in range(5):
                        key=f'candidate_{sj}_{sid}_{method}_{k}';path=out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                        if key not in ledger.data['completed']:
                            capacity(out,smoke_deadline);ledger.begin(key,'candidate',P['wall_budget_s'],720)
                            progress('frozen_smoke_candidates',seed=sj+1,task=ti+1,split=entry['split'],method=method,candidate=k+1)
                            sample_seed=seed+ti*101+k*997
                            r=(blocked_candidate(method,k,sample_seed,task,base,'TRAINING_DEPENDENCY_BLOCKED') if method=='expanded_diffusion' and bundle is None
                               else candidate(method,k,task,base,sample_seed,bundle,rr,ii,cc,profile,smoke_deadline))
                            r.update(sample_id=sid,split=entry['split'],seed_group=sj)
                            atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
                        if path.exists():records.append(read_json(path))
                    key=f'execution_{sj}_{sid}_{method}';directory=out/'executions'/str(sj)/sid/method
                    if key in ledger.data['completed']:continue
                    capacity(out,smoke_deadline)
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,
                        'task_hash':task['task_hash'],'selection_before_execution':True,'rule':P['selection']})
                    ledger.begin(key,'execution',P['wall_budget_s'],144);before=rr.calls.copy()
                    progress('smoke_native_torque_replay',seed=sj+1,task=ti+1,split=entry['split'],method=method)
                    if chosen:r=execute(directory,task,chosen['Z'],rr,ii,cc,profile,smoke_deadline)
                    else:r={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',r)
                    ledger.finish(r,{k:rr.calls[k]-before[k] for k in before})
                report(out,'RUNNING')
        diagnostic_deadline=min(deadline,time.time()+P['diagnostic_reserve_s'])
        for entry in entries:
            task=entry['task'];sid=entry['sample_id'];rr,ii,cc,_=cache.get(entry)
            for method in METHODS:
                original=out/'candidates'/'0'/sid/method/'00.json';record=read_json(original)
                for stage in ('raw','smoothed_clipped'):
                    key=f'diagnostic_{sid}_{method}_{stage}';path=out/'proposal_diagnostics'/sid/method/f'{stage}.json'
                    if key in ledger.data['completed']:continue
                    capacity(out,diagnostic_deadline);ledger.begin(key,'diagnostic',600,96);before=rr.calls.copy()
                    progress('independent_proposal_diagnostic',task=sid,method=method,proposal_stage=stage)
                    try:
                        z=np.asarray(record['sampling']['guidance'][-1]['raw_Z'])
                        if stage=='smoothed_clipped':z=np.clip(smooth(z,task['condition']['initial_features']),rr.lower,rr.upper)
                        r=dict(validate(z,task,rr,ii,cc,profile,diagnostic_deadline),status='DONE',stage=stage,
                            task_hash=task['task_hash'],proposal_Z_hash=digest(z),affects_selection=False,
                            source_candidate_sha256=file_digest(original))
                    except Exception as error:r={'status':'BLOCKED','reason':repr(error),'eligible':False,'stage':stage}
                    r['calls']={k:rr.calls[k]-before[k] for k in before}
                    atomic_json(path,r);ledger.finish({'status':r['status']},r['calls'])
        atomic_json(out/'runtime_cache_audit.json',cache.stats())
        progress('independent_training_and_native_raw_audit')
        try:proof=verify(out,identity,index,rows,codec,arrays)
        except Exception as error:proof={'status':'BLOCKED','problems':[repr(error)],'traceback':traceback.format_exc()}
        atomic_json(out/'independent_verification.json',proof)
        capacity(out,deadline)
        return report(out,'FINISHED' if proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
