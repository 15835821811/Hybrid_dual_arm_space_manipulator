"""Qualified forward prior expansion -> training -> frozen DEV execution smoke."""
from pathlib import Path
import os
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,atomic_npz,file_digest,digest,Ledger,RunLock
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_modes.experiment import inputs as old_inputs,sources as old_sources
from ff_prior_modes import data as previous_data,learning as previous_learning
from ff_prior_loop.experiment import execute
from ff_prior_feasible.guidance import CONTRACT as GUIDE
from ff_prior_feasible.validation import CONTRACT as VALIDATE
from . import data,learning,sensitivity
from .sampling import candidate

ROOT=Path(__file__).resolve().parents[1]
METHODS=('structured','frozen_diffusion','learned_mean','expanded_diffusion')
PROTOCOL={'version':'bounded_expansion_full_loop_1','profile':'bounded_TRAIN_expansion_plus_smoke',
    'data':data.CONTRACT,'training':learning.CONTRACT,'sensitivity':sensitivity.CONTRACT,
    'guidance':GUIDE,'validation':VALIDATE,'methods':list(METHODS),'candidates_per_task':5,
    'sampling_seeds':[20261219,20261223,20261229],'DEV_tasks':4,'max_candidates':240,
    'max_selected_executions':48,'max_witness_executions':16,'max_mechanism_candidates':18,
    'wall_budget_s':7200,'max_output_bytes':20*1024**3,'cpu_threads':2,
    'candidate_wall_s':15.,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
    'mechanism_seeds':[20261203,20261207,20261211],
    'mechanism_tasks':'all three preexisting TRAIN static_detour targets; no tested-method selection',
    'selection':'minimum normalized world-goal error among eligible; exactly one execution/replay, no replacement',
    'stability_gate':'expanded model each seed >=18/20 eligible; every task >=1; 12/12 selected executions and replay; finite valid numerics',
    'data_gate':'cumulative TRAIN >=1000; >=150 new empty-scene families; 16/16 preselected torque/replay; no unresolved accounting',
    'efficiency_advantage_required':False,'old_results_rewritten':False,'TEST':False,'study':False,
    'pilot_run_in_this_profile':False,'pilot_after_success_authorized':True,
    'pilot_gate':'data + stability + actual training ingestion + independent audit; existing scale/physics capability limits checked before launch',
    'final_TEST_auto_launch':False,'strict_multimodal':False,'goal_relabelled':False,'physical_gates_changed':False}


def sources():
    return dict(old_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_expand').rglob('*.py'))})


def inputs(source):
    old,ds,dev,rows,files=old_inputs(source);source=Path(source).resolve()
    for p in [data.PREVIOUS/'training_index.json']+[data.PREVIOUS/'learning/goal_loss'/n for n in ('model.pt','codec.json','training.json')]:
        files[os.path.relpath(p,source)]=file_digest(p)
    for e in data.old_index():
        for k in ('record','arrays'):files[os.path.relpath(ROOT/e[k],source)]=e[k+'_sha256']
    for sid,r in read_json(source/'splits.json')['records'].items():
        if r['split']=='TEST':files[f'records/{sid}.json']=file_digest(source/'records'/f'{sid}.json')
    return old,ds,dev,rows,files


def budget(out,deadline):
    if time.time()>=deadline:raise RuntimeError('TOTAL_WALL_BUDGET_EXHAUSTED')
    if sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']:
        raise RuntimeError('STORAGE_BUDGET_EXHAUSTED')


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    from .audit_results import verify
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('run identity changed')
        tests=read_json(ROOT/'ff_prior_expand/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_expand/evidence/tests.xml'):
            raise ValueError('software proof stale')
        physical,canonical,initial,_=audit(old['config']);profile=acceptance(old['config'])
        if physical['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']:raise ValueError('physical gates changed')
        original=data.old_index();contexts=data.contexts(canonical,initial,profile,original)
        frozen={'DEV':dev,'mechanism_entries':[read_json(ROOT/e['record']) for e in original if e['sample_id'].startswith('complex_train_') and e['sample_id'].endswith('static_detour') and e['split']=='TRAIN'],
            'contexts':[dict(meta,initial_state_id=ii.state_id) for meta,rr,ii,cc in contexts]}
        # Teacher control points/arrays are not exposed to inference entries.
        frozen['mechanism_entries']=[{k:r[k] for k in ('sample_id','family_id','split','task','scene')} for r in frozen['mechanism_entries']]
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',frozen);atomic_json(out/'software_tests.json',tests)
            atomic_json(out/'baseline_audit.json',physical)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        if digest(read_json(out/'frozen_tasks.json'))!=digest(frozen):raise ValueError('frozen tasks changed')
        ledger=Ledger(out,identity);deadline=time.time()+max(0,7200-previous_elapsed)
        cache=previous_data.RuntimeCache(canonical,initial,profile)
        frozen_bundle=previous_learning.load(data.PREVIOUS/'learning/goal_loss')
        # Paired TRAIN-only mechanism diagnostic, not a cost superiority gate.
        for ti,e in enumerate(frozen['mechanism_entries']):
            rr,ii,cc,_=cache.get(e);base=public_goal_initializer(e['task'],rr,ii)
            for sj,seed in enumerate(PROTOCOL['mechanism_seeds']):
                for method in ('frozen_diffusion','diffusion_goal_only'):
                    key=f'mechanism_{ti}_{sj}_{method}';path=out/'mechanism'/f'{key}.json'
                    if key in ledger.data['completed']:continue
                    budget(out,deadline);ledger.begin(key,'mechanism',7200,18)
                    progress('TRAIN_collision_mechanism',task=ti+1,seed=sj+1,method=method)
                    r=candidate(method,0,e['task'],base,seed+ti*101,frozen_bundle,rr,ii,cc,profile,deadline)
                    r.update(sample_id=e['sample_id'],split='TRAIN');atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
        heldout=data.heldout_goals(original,dev,source)
        hashes={read_json(ROOT/e['record'])['reference_hash'] for e in original}
        admitted=0
        for path in sorted((out/'records').glob('*.json')):
            row=data.check_new_record(out,path,ledger);hashes.add(row['reference_hash']);admitted+=row['execution_status']!='FAIL'
        for attempt in range(1200):
            sid=f'expand_train_{attempt:05d}'
            if sid in ledger.data['completed']:continue
            if admitted+256>=1000:break
            budget(out,deadline);ledger.begin(sid,'generation',7200,1200)
            progress('forward_TRAIN_generation',attempt=attempt+1,maximum=1200,admitted=admitted,total_TRAIN=256+admitted)
            result=data.generate(attempt,contexts[(attempt//2)%9],out,profile,heldout,hashes,deadline)
            atomic_json(out/'generation'/f'{sid}.json',result);ledger.finish({'status':result['status']},result['calls'])
            if result['admitted']:admitted+=1;hashes.add(result['reference_hash'])
            if (attempt+1)%50==0:report(out,ledger,'RUNNING')
        selection=data.select_audits(out,ledger)
        for sid in selection['random_stratified']+selection['risk']:
            key='witness_'+sid;directory=out/'witness_executions'/sid
            if key in ledger.data['completed']:continue
            budget(out,deadline);ledger.begin(key,'witness',7200,16);progress('TRAIN_torque_replay',sample=sid)
            path=out/'records'/f'{sid}.json';row=data.check_new_record(out,path,ledger)
            rr,ii,cc,_=cache.get(row);before=rr.calls.copy()
            with np.load(out/row['arrays']) as a:z=a['Z'].copy()
            result=execute(directory,row['task'],z,rr,ii,cc,profile,deadline)
            row['execution_status']='PASS' if result['status']=='EXECUTION_VERIFIED' else 'FAIL'
            row['training_eligible']=row['execution_status']=='PASS'
            if row['execution_status']=='PASS':row['validation_level']='EXECUTION_VERIFIED'
            row['execution_report']=f'witness_executions/{sid}/report.json'
            atomic_json(path,row);ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
        index=data.combined(out,ledger)
        atomic_json(out/'training_index.json',{'records':index,'actual_loader_input':True,'TEST_read':False})
        budget(out,deadline);arrays,rows=previous_data.arrays(index,cache,out,progress)
        codec=learning.fit_codec(out/'learning',arrays)
        matrices,sensitivity_report=sensitivity.prepare(out,arrays,rows,codec,cache,ledger,deadline,progress)
        training=learning.train(out/'learning/goal_loss',arrays,codec,matrices,'goal_loss',sensitivity_report,progress,deadline)
        bundle=learning.load(out/'learning/goal_loss') if training['status']=='PASS' else None
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            for ti,e in enumerate(dev):
                task=e['task'];sid=f'DEV_{ti:02d}';entry={'task':task}
                rr,ii,cc,transfer=cache.get(entry)
                atomic_npz(out/'DEV_tasks'/sid/'initial_state.npz',values=ii.values)
                base=public_goal_initializer(task,rr,ii)
                atomic_json(out/'transfer_audits'/f'{sj}_{sid}.json',dict(transfer,public_base=base,shared_public_base_hash=digest(base)))
                for method in METHODS[ti%4:]+METHODS[:ti%4]:
                    b=None if method=='structured' else frozen_bundle if method=='frozen_diffusion' else bundle
                    records=[]
                    for k in range(5):
                        key=f'candidate_{sj}_{sid}_{method}_{k}';path=out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                        if key not in ledger.data['completed']:
                            budget(out,deadline);ledger.begin(key,'candidate',7200,240)
                            progress('frozen_DEV_candidates',seed=sj+1,task=ti+1,method=method,candidate=k+1)
                            if method in ('learned_mean','expanded_diffusion') and bundle is None:
                                r={'status':'BLOCKED','eligible':False,'reason':'TRAINING_DEPENDENCY_BLOCKED','model_calls':{},'actual_network_calls':0,'candidate_id':k,'method':method}
                            else:r=candidate(method,k,task,base,seed+ti*101+k*997,b,rr,ii,cc,profile,deadline)
                            r.update(sample_id=sid,split='DEV',seed_group=sj);atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
                        if path.exists():records.append(read_json(path))
                    key=f'execution_{sj}_{sid}_{method}';directory=out/'executions'/str(sj)/sid/method
                    if key in ledger.data['completed']:continue
                    budget(out,deadline)
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                    ledger.begin(key,'execution',7200,48);before=rr.calls.copy();progress('DEV_torque_replay',seed=sj+1,task=ti+1,method=method)
                    if chosen:r=execute(directory,task,chosen['Z'],rr,ii,cc,profile,deadline)
                    else:r={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',r)
                    ledger.finish(r,{n:rr.calls[n]-before[n] for n in before})
                report(out,ledger,'RUNNING')
        if any(file_digest(source/p)!=h for p,h in files.items()) or sources()!=identity['source_hashes']:raise ValueError('source/input changed')
        ledger.index();result=report(out,ledger,'FINISHED')
        progress('independent_raw_evidence_audit')
        verification=verify(out)
        atomic_json(out/'independent_verification.json',verification)
        result=report(out,ledger,'FINISHED' if verification['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
        return result
