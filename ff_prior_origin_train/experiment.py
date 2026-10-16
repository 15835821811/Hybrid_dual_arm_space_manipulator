"""Actual balanced ingestion, fixed final checkpoint, and equal-budget smoke."""
from pathlib import Path
import subprocess
import time
import traceback
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_coverage_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.recovery import process_is_alive
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_modes import data as loader
from ff_prior_origin_scene import TaskOriginSceneCache
from ff_prior_origin_fill.experiment import sources as parent_sources,identity_for as parent_identity
from ff_prior_generalization.experiment import execute
from ff_prior_generalization.sampling import candidate
from ff_prior_coverage_train.learning import load as old_load
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from .protocol import ROOT,SOURCE,OLD,BASE,METHODS,PRODUCERS,PROTOCOL as P
from . import learning,sensitivity,mean


def sources():
    return dict(parent_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p)
                                  for p in sorted((ROOT/'ff_prior_origin_train').rglob('*.py'))})


def require_data(source):
    source=Path(source).resolve();state=read_json(source/'status.json');summary=read_json(source/'summary.json')
    if state['status']!='FINISHED' or process_is_alive(state['pid']):raise ValueError('merged data worker must complete and exit')
    if not summary['ready_for_training_stage'] or not summary['merged_data_gate_passed']:
        raise ValueError('new merged qualification required; old failed gate cannot be used')
    if read_json(source/'independent_verification.json')['status']!='PASS':raise ValueError('merged raw audit missing')
    index=read_json(source/'training_index.json')['records']
    if sum(e['split']=='TRAIN' for e in index)!=12688 or sum(e['split']=='VAL' for e in index)!=48:
        raise ValueError('complete merged TRAIN and original VAL required')
    if any(e['execution_status']=='FAIL' for e in index):raise ValueError('failed execution in training')
    coverage=[e['sample_id'] for e in index if e['split']=='TRAIN' and e.get('producer_run_id') in PRODUCERS]
    if len(coverage)!=544:raise ValueError('full 544-origin coverage producer group required')
    validation=read_json(source/'frozen_validation.json')
    if len(validation['entries'])!=8 or {e['sample_id'] for e in validation['entries']}&{e['sample_id'] for e in index}:
        raise ValueError('independent NEW_VAL isolation violated')
    if not validation['before_TRAIN_generation'] or any(e['training_eligible'] or e['method_results_used'] for e in validation['entries']):
        raise ValueError('validation used for training or method selection')
    loader.old_data.assert_disjoint(index)
    return index,coverage,validation


def identity_for(source):
    source=Path(source).resolve()
    if source!=SOURCE.resolve():raise ValueError('exact new merged parent required')
    index,coverage,validation=require_data(source)
    parent=read_json(source/'run.json')['identity'];expected,_=parent_identity(parent['source'])
    if parent!=expected:raise ValueError('merged parent identity changed')
    files=dict(parent['source_files'])
    for directory in (source,OLD):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.name not in ('run.lock','launch.guard') and p.suffix!='.tmp':
                files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    oldstate=read_json(OLD/'status.json')
    if oldstate['status']!='FINISHED' or process_is_alive(oldstate['pid']) or not read_json(OLD/'summary.json')['coverage_training_closure_passed']:
        raise ValueError('verified latest 12144 model required')
    for e in index:
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']:raise ValueError('training source changed')
    dev=read_json(OLD/'frozen_tasks.json')['DEV']
    entries=[dict(e,sample_id=f'DEV_{i:02d}',split='DEV',comparison_slot=i) for i,e in enumerate(dev)]
    entries += [dict(e,comparison_slot=4+i) for i,e in enumerate(validation['entries'])]
    if len(entries)!=12:raise ValueError('original DEV4 and frozen NEW_VAL8 required')
    identity={'protocol':P,'source':str(source),'source_files':files,'source_hashes':sources(),
              'original_DEV_hash':digest(dev),'validation_hash':digest(validation),
              'coverage_ids':sorted(coverage),'comparison_entries_hash':digest(entries)}
    return identity,(read_json(BASE/'run.json')['config'],index,coverage,validation,entries)


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
    from .audit_results import verify
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    deadline=time.time()+max(0,P['wall_budget_s']-previous_elapsed)
    identity,(config,index,coverage,validation,entries)=identity_for(source)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_origin_train/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_origin_train/evidence/tests.xml'):
            raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen identity changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:raise ValueError('physical gates changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'joint_lower':canonical.lower,'joint_upper':canonical.upper,
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'training_index.json',{'records':index})
            atomic_json(out/'frozen_validation.json',validation)
            atomic_json(out/'frozen_tasks.json',{'entries':entries,'before_training':True,'before_comparison':True})
            atomic_json(out/'inherited_accounting.json',{'merged_source_status':read_json(source/'status.json'),
                'merged_source_summary':read_json(source/'summary.json'),
                'merged_ledger_sha256':file_digest(source/'ledger.json'),
                'origin_history':read_json(source/'inherited_accounting.json'),
                'previous_training_status':read_json(OLD/'status.json'),
                'previous_training_summary':read_json(OLD/'summary.json'),
                'previous_training_ledger_sha256':file_digest(OLD/'ledger.json')})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'identity_hash':digest(identity)})
        cache=TaskOriginSceneCache(canonical,initial,profile)
        arrays,rows=loader.arrays(index,cache,out,progress)
        codec=learning.fit_codec(out/'learning',arrays)
        mean_codec=mean.fit(out/'learning',arrays,codec,coverage)
        training_deadline=deadline-P['training_reserve_for_smoke_s']
        try:
            matrices,sreport=sensitivity.prepare(out,arrays,rows,codec,cache,ledger,training_deadline,progress)
        except Exception as error:
            matrices=None;sreport={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
            if ledger.data['inflight']:
                ledger.data.setdefault('unresolved_call_accounting',[]).append(ledger.data['inflight']['id'])
                ledger.finish({'status':'BLOCKED','reason':repr(error)},{})
            atomic_json(out/'sensitivity_failure.json',sreport)
        try:
            training=learning.train(out/'learning/goal_loss',arrays,codec,matrices,'goal_loss',sreport,progress,training_deadline,coverage)
        except Exception as error:
            training={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'training_failure.json',training)
        new_bundle=learning.load(out/'learning/goal_loss') if training['status']=='PASS' else None
        old_bundle=old_load(OLD/'learning/goal_loss')
        atomic_json(out/'checkpoint_binding.json',{'old':file_digest(OLD/'learning/goal_loss/model.pt'),
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
        progress('independent_training_and_native_raw_audit')
        try:proof=verify(out,identity,index,rows,codec,arrays)
        except Exception as error:proof={'status':'BLOCKED','problems':[repr(error)],'traceback':traceback.format_exc()}
        atomic_json(out/'independent_verification.json',proof)
        return report(out,'FINISHED' if proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
