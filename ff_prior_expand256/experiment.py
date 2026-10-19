"""Bounded attempts, irreversible slot accounting, and frozen native audit units."""
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, atomic_npz, Ledger
from ff_prior.pipeline import audit, acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_generalization.experiment import execute
from ff_prior_delivery_train.learning import load
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_delivery_guidance.sampling import candidate
from .protocol import ROOT, MODEL_DIR, BASE, DEFAULT_OUT, PROTOCOL, proposal_id, proposal_seed
from .inputs import identity_for
from . import data


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .audit_results import verify
    from .reporting import report
    out=Path(out).resolve()
    if out!=DEFAULT_OUT.resolve():
        raise ValueError('independent frozen E256 output required')
    identity,(config,selection,index,inherited_hashes)=identity_for(source)
    deadline=time.time()+max(0.,PROTOCOL['wall_budget_s']-previous_elapsed)
    with RunLock(out):
        if (out/'run.json').exists():
            if not resume or read_json(out/'run.json')['identity']!=identity:
                raise ValueError('frozen run identity changed')
        else:
            atomic_json(out/'run.json',{'identity':identity,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                     'TEST':False,'training_updates':0})
            atomic_json(out/'frozen_tasks.json',selection)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in identity['source_hashes']:
                    z.write(ROOT/p,p)
            progress('E256_targets_and_execution_units_frozen',targets=len(selection['entries']),execution_units=32,max_attempts=512)
        tests=read_json(ROOT/'ff_prior_expand256/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=identity['source_hashes'] or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_expand256/evidence/tests.xml'):
            raise ValueError('software proof stale')
        physical,canonical,initial,_=audit(config)
        if physical['acceptance']!=read_json(BASE/'baseline_audit.json')['acceptance']:
            raise ValueError('physical gate changed')
        atomic_json(out/'physical_contract.json',physical)
        profile=acceptance(config)
        ledger=Ledger(out,{'identity_hash':digest(identity)})
        import torch
        torch.set_num_threads(2)
        bundle=load(MODEL_DIR)
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=8,max_scene_models=4)
        known=set(inherited_hashes)|{r['reference_hash'] for r in data.population(out,ledger)}
        limit_reason=None
        def within_budget():
            nonlocal limit_reason
            if time.time()>=deadline-180:
                limit_reason='TOTAL_WALL_BUDGET';return False
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=PROTOCOL['max_output_bytes']-32*1024**2:
                limit_reason='STORAGE_BUDGET';return False
            return True
        for e in selection['entries']:
            ti=e['target_slot'];target=e['target_id'];task=e['task']
            runtime=None
            for slot in range(4):
                slot_path=out/'slots'/f'{target}_s{slot}.json'
                if slot_path.exists():
                    continue
                attempts=[];chosen=None
                for attempt in range(2):
                    sid=proposal_id(ti,slot,attempt);p=out/'generation'/f'{sid}.json'
                    if sid not in ledger.data['completed']:
                        if not within_budget():
                            break
                        if runtime is None:
                            runtime=cache.get(e)
                            rr,ii,cc,transfer=runtime
                            base=public_goal_initializer(task,rr,ii)
                            atomic_json(out/'transfer_audits'/f'{target}.json',dict(transfer,public_base_hash=digest(base)))
                            atomic_json(out/'tasks'/target/'task.json',task)
                            state_path=out/'tasks'/target/'initial_state.npz'
                            if not state_path.exists():
                                atomic_npz(state_path,values=ii.values)
                            else:
                                with np.load(state_path) as a:
                                    if not np.array_equal(a['values'],ii.values):
                                        raise ValueError('complete target state changed on resume')
                        rr,ii,cc,_=runtime
                        ledger.begin(sid,'generation',PROTOCOL['wall_budget_s'],512)
                        progress('fixed_TRAIN_diffusion_generation',target=ti+1,slot=slot+1,attempt=attempt+1)
                        r=candidate(selection['sampler'],slot,task,base,proposal_seed(ti,slot,attempt),bundle,rr,ii,cc,profile,deadline-180)
                        proof=data.admit(out,e,sid,r,ii,known)
                        proof.update(target_slot=ti,slot=slot,attempt=attempt)
                        atomic_json(p,proof)
                        ledger.finish({'status':proof['status']},proof['calls'])
                    attempts.append(sid)
                    if ledger.data['completed'][sid]['status']=='ADMITTED':
                        if not p.exists() or not read_json(p)['admitted']:
                            raise ValueError('committed admission record missing')
                        chosen=sid;break
                atomic_json(slot_path,{'target_id':target,'slot':slot,'attempt_ids':attempts,'selected_id':chosen,
                                      'status':'ADMITTED' if chosen else 'BUDGET_LIMIT' if limit_reason else 'UNQUALIFIED',
                                      'limit_reason':limit_reason})
            report(out,'RUNNING',ledger)
        rows=data.population(out,ledger)
        for target in selection['execution_target_ids']:
            key='execution_'+target;directory=out/'executions'/target
            if key not in ledger.data['completed']:
                e=next(v for v in selection['entries'] if v['target_id']==target)
                candidates=[r for r in rows if r['target_id']==target]
                chosen=min(candidates,key=lambda r:r['sample_id'],default=None)
                ledger.begin(key,'execution',PROTOCOL['wall_budget_s'],32)
                atomic_json(directory/'selection.json',{'selected_id':chosen['sample_id'] if chosen else None,
                    'task_hash':e['task']['task_hash'],'target_id':target,'selection_before_execution':True,
                    'rule':PROTOCOL['execution_selection'],'replacement_after_failure':False})
                progress('preselected_TRAIN_native_torque_and_replay',target=target)
                if chosen and within_budget():
                    rr,ii,cc,_=cache.get(e)
                    with np.load(out/chosen['arrays']) as a:
                        z=a['Z'].copy()
                    before=rr.calls.copy()
                    result=execute(directory,e['task'],z,rr,ii,cc,profile,deadline-120)
                    calls={k:rr.calls[k]-before[k] for k in before}
                else:
                    result={'status':'BLOCKED_BUDGET' if chosen else 'NO_CANDIDATE','reason':limit_reason}
                    calls={};atomic_json(directory/'outcome.json',result)
                ledger.finish(result,calls)
            data.apply_execution_grade(out,ledger,target)
        merged,new=data.index_for(out,index,ledger)
        atomic_json(out/'training_index.json',{'records':merged,'actual_training_consumed':False,'TEST_arrays_read':False})
        atomic_json(out/'qualified_increment.json',{'records':new,'validation_levels_are_individual':True,'failed_or_unknown_execution_excluded':True})
        atomic_json(out/'diversity.json',data.diversity(out,data.population(out,ledger)))
        atomic_json(out/'runtime_cache.json',cache.stats())
        atomic_json(out/'budget_end.json',{'limit_reason':limit_reason,'automatic_budget_extension':False})
        progress('E256_read_only_raw_audit')
        proof=verify(out,identity,ledger)
        atomic_json(out/'independent_verification.json',proof)
        return report(out,'FINISHED',ledger)
