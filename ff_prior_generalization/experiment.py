"""Freeze new tasks, compare fixed models, native torque/replay, then read-only audit."""
from pathlib import Path
import json
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit, acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.physical_audit import audit_execution
from ff_prior_modes.data import RuntimeCache
from ff_prior_coverage_train.experiment import sources as parent_sources
from ff_prior_coverage_train.learning import load as current_load
from ff_prior_pilot_train.learning import load as old_load
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from .protocol import ROOT, BASE, SOURCE, OLD, METHODS, PROTOCOL
from . import data
from .sampling import candidate


def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
                                  for p in sorted((ROOT/'ff_prior_generalization').rglob('*.py'))})


def require_parent(source):
    source=Path(source).resolve();state=read_json(source/'status.json');s=read_json(source/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):
        raise ValueError('parent must be completed and exited')
    if not s['coverage_training_closure_passed'] or read_json(source/'independent_verification.json')['status']!='PASS':
        raise ValueError('verified parent closure is required')


def inputs(source):
    source=Path(source).resolve();require_parent(source)
    parent_run=read_json(source/'run.json');paths={}
    for p,h in parent_run['identity']['source_files'].items():
        actual=(Path(parent_run['identity']['source'])/p).resolve()
        if file_digest(actual)!=h: raise ValueError('inherited source changed')
        paths[actual.relative_to(ROOT).as_posix()]=h
    for directory in (source,OLD):
        for name in ('run.json','status.json','summary.json','independent_verification.json',
                     'frozen_tasks.json','training_index.json','learning/goal_loss/model.pt',
                     'learning/goal_loss/training.json','learning/goal_loss/codec.json'):
            p=directory/name;paths[p.relative_to(ROOT).as_posix()]=file_digest(p)
    for name in ('run.json','baseline_audit.json','splits.json'):
        p=BASE/name;paths[p.relative_to(ROOT).as_posix()]=file_digest(p)
    index=read_json(source/'training_index.json')['records']
    assert sum(e['split']=='TRAIN' for e in index)==12144 and sum(e['split']=='VAL' for e in index)==48
    for e in index:
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']: raise ValueError('protected data changed')
            paths[e[k]]=e[k+'_sha256']
    dev=read_json(source/'frozen_tasks.json')['DEV']
    guard=data.guard_population(index,dev)
    # Reserved metadata was already read solely by the isolation guard.
    for e in guard:
        if e['split']=='TEST_METADATA_ONLY':
            p=BASE/'records'/f"{e['sample_id'].removeprefix('RESERVED_')}.json"
            paths[p.relative_to(ROOT).as_posix()]=file_digest(p)
    return read_json(BASE/'run.json')['config'],dev,index,guard,paths


def identity_for(source):
    config,dev,index,guard,files=inputs(source)
    return {'protocol':PROTOCOL,'source':str(Path(source).resolve()),'source_files':files,
            'source_hashes':sources(),'original_DEV_hash':digest(dev),'guard_hash':digest(guard)},(config,dev,index,guard)


def execute(directory,task,z,rr,ii,cc,profile,deadline):
    try:
        result=audit_execution(task,profile,rr,ii,Reference(z,task['condition']['initial_features'],task['condition']['duration']),directory,deadline)
        result={'status':result['level']}
    except Exception as error:
        import traceback
        result={'status':'ERROR','reason':repr(error),'traceback':traceback.format_exc()}
    atomic_json(directory/'outcome.json',result)
    return result


def completed_result(out):
    """Identity must be verified before returning a completed no-work resume."""
    frozen=read_json(Path(out)/'run.json')['identity']
    expected,_=identity_for(frozen['source'])
    if expected!=frozen: raise ValueError('completed job identity changed')
    return read_json(Path(out)/'summary.json')


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    from .audit_results import verify
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    started=time.time();identity,(config,dev,index,guard)=identity_for(source)
    deadline=started+max(0,PROTOCOL['wall_budget_s']-previous_elapsed)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_generalization/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_generalization/evidence/tests.xml'):
            raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):
            raise ValueError('frozen run identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:
            raise ValueError('physical acceptance changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                                      'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
            atomic_json(out/'original_DEV.json',dev);atomic_json(out/'protected_goal_metadata.json',guard)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'identity_hash':digest(identity)})
        frozen_path=out/'frozen_tasks.json'
        if not frozen_path.exists():
            entries=[];generation_deadline=min(deadline-5400,time.time()+1800)
            for slot in range(12):
                chosen=None
                for attempt in range(8):
                    key=f'freeze_{slot:02d}_{attempt}';path=out/'generation'/f'{key}.json'
                    if key not in ledger.data['completed']:
                        if time.time()>=generation_deadline:break
                        ledger.begin(key,'generation',1800,96);progress('new_world_task_generation',slot=slot+1,attempt=attempt+1)
                        extra=[{'sample_id':e['sample_id'],'family_id':e['family_id'],'split':'NEW_VAL',
                                'goals':e['task']['condition']['goals']} for e in entries if e['status']=='FROZEN']
                        r=data.generate(slot,attempt,canonical,initial,profile,guard+extra,out,generation_deadline)
                        atomic_json(path,r);ledger.finish({'status':r['status']},r['calls'])
                        if r.get('unknown_screen_call_accounting'):
                            ledger.data.setdefault('unresolved_call_accounting',[]).append(key);ledger.save()
                    if path.exists() and read_json(path)['status']=='FROZEN' and ledger.data['completed'][key]['status']=='FROZEN':
                        chosen=read_json(path);break
                entries.append(chosen or {'sample_id':f'newval_{slot:02d}','slot':slot,'status':'BLOCKED',
                                          'reason':'NO_QUALIFIED_TEACHER_IN_FROZEN_BUDGET'})
            atomic_json(frozen_path,{'entries':entries,'digest':digest(entries),'before_any_comparison':True})
            progress('all_new_tasks_frozen',qualified=sum(e['status']=='FROZEN' for e in entries),total=12)
        frozen=read_json(frozen_path);assert digest(frozen['entries'])==frozen['digest']
        events=out/'events.jsonl'
        if not any(json.loads(s).get('stage')=='all_new_tasks_frozen' for s in events.read_text(encoding='utf-8').splitlines()):
            progress('all_new_tasks_frozen',qualified=sum(e['status']=='FROZEN' for e in frozen['entries']),total=12)
        entries=[e for e in frozen['entries'] if e['status']=='FROZEN']
        cache=RuntimeCache(canonical,initial,profile)
        # All task slots have been frozen before teacher execution or model inference.
        for e in entries:
            sid=e['sample_id'];key='witness_'+sid;directory=out/'witness_executions'/sid
            if key in ledger.data['completed']:continue
            if time.time()>=deadline-600:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
            rr,ii,cc,proof=cache.get(e);before=rr.calls.copy()
            with np.load(out/'tasks'/sid/'witness.npz') as a:z=a['Z'].copy()
            ledger.begin(key,'witness',7200,12);progress('new_validation_teacher_torque_replay',task=sid)
            r=execute(directory,e['task'],z,rr,ii,cc,profile,deadline-600)
            ledger.finish(r,{k:rr.calls[k]-before[k] for k in before})
        import torch
        torch.set_num_threads(2)
        old_bundle=old_load(OLD/'learning/goal_loss');new_bundle=current_load(source/'learning/goal_loss')
        if digest(old_bundle[1].state)!=digest(new_bundle[1].state):raise ValueError('comparison codec changed')
        atomic_json(out/'checkpoint_binding.json',{'old':file_digest(OLD/'learning/goal_loss/model.pt'),
                   'new':file_digest(source/'learning/goal_loss/model.pt'),'codec':digest(new_bundle[1].state),
                   'training_updates':0,'validation_used_for_checkpoint_selection':False})
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            for e in entries:
                ti=e['slot'];sid=e['sample_id'];task=e['task'];rr,ii,cc,proof=cache.get(e)
                base=public_goal_initializer(task,rr,ii)
                atomic_json(out/'transfer_audits'/f'{sid}.json',dict(proof,public_base_hash=digest(base)))
                for method in METHODS[ti%4:]+METHODS[:ti%4]:
                    bundle=None if method=='structured' else new_bundle if method=='expanded_diffusion' else old_bundle
                    records=[]
                    for k in range(5):
                        key=f'candidate_{sj}_{sid}_{method}_{k}';path=out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                        if key not in ledger.data['completed']:
                            if time.time()>=deadline-600:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
                            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']:
                                raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')
                            ledger.begin(key,'candidate',7200,720);progress('new_validation_candidates',seed=sj+1,task=ti+1,method=method,candidate=k+1)
                            r=candidate(method,k,task,base,seed+ti*101+k*997,bundle,rr,ii,cc,profile,deadline-600)
                            r.update(sample_id=sid,split='NEW_VAL',seed_group=sj)
                            atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
                        if path.exists():records.append(read_json(path))
                    key=f'execution_{sj}_{sid}_{method}';directory=out/'executions'/str(sj)/sid/method
                    if key in ledger.data['completed']:continue
                    if time.time()>=deadline-600:raise RuntimeError('TOTAL_BUDGET_EXHAUSTED')
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,
                                'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                    ledger.begin(key,'execution',7200,144);before=rr.calls.copy()
                    progress('new_validation_torque_replay',seed=sj+1,task=ti+1,method=method)
                    if chosen:r=execute(directory,task,chosen['Z'],rr,ii,cc,profile,deadline-600)
                    else:r={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',r)
                    ledger.finish(r,{k:rr.calls[k]-before[k] for k in before})
                report(out,'RUNNING')
        # Independent diagnostics run after all comparisons and cannot change
        # candidate selection. All methods have the same two extra validations.
        diagnostic_deadline=min(deadline,time.time()+600)
        for e in entries:
            sid=e['sample_id'];rr,ii,cc,_=cache.get(e);task=e['task']
            for method in METHODS:
                path=out/'candidates'/'0'/sid/method/'00.json'
                r=read_json(path)
                for stage in ('raw','smoothed_clipped'):
                    key=f'diagnostic_{sid}_{method}_{stage}';p=out/'proposal_diagnostics'/sid/method/f'{stage}.json'
                    if key in ledger.data['completed']:continue
                    if time.time()>=diagnostic_deadline:raise RuntimeError('PROPOSAL_DIAGNOSTIC_BUDGET_EXHAUSTED')
                    ledger.begin(key,'diagnostic',600,96);before=rr.calls.copy()
                    progress('independent_pre_guidance_validation',task=sid,method=method,stage=stage)
                    try:
                        z=np.asarray(r['sampling']['guidance'][-1]['raw_Z'])
                        if stage=='smoothed_clipped':z=np.clip(smooth(z,task['condition']['initial_features']),rr.lower,rr.upper)
                        check=validate(z,task,rr,ii,cc,profile,diagnostic_deadline)
                        v=dict(check,status='DONE',stage=stage,task_hash=task['task_hash'],source_candidate_sha256=file_digest(path),
                               proposal_Z_hash=digest(z),affects_selection=False)
                    except Exception as error:v={'status':'ERROR','reason':repr(error),'eligible':False,'stage':stage}
                    v['calls']={k:rr.calls[k]-before[k] for k in before}
                    atomic_json(p,v);ledger.finish({'status':v['status']},v['calls'])
        progress('independent_new_validation_audit')
        proof=verify(out,identity,guard);atomic_json(out/'independent_verification.json',proof)
        return report(out,'FINISHED' if proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
