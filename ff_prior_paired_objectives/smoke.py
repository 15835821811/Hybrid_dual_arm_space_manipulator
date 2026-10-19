from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior.guidance import public_goal_initializer
from ff_prior_generalization.sampling import candidate
from ff_prior_generalization.experiment import execute
from ff_prior_world_endpoint.experiment import capacity,blocked_candidate
from ff_prior_goal.projection import smooth
from ff_prior_feasible.validation import validate
from .protocol import METHODS,PROTOCOL as P
from .reporting import report

def run(out,entries,cache,profile,ledger,deadline,progress,old_bundle,new_bundle,mean_codec):
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
