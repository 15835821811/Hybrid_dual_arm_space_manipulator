"""One training run followed by original frozen DEV with matched candidate caps."""
from pathlib import Path
import subprocess
import time
import traceback
import zipfile
from ff_prior.storage import read_json,file_digest,digest,RunLock
from ff_prior_pilot_train.storage import atomic_json,atomic_npz,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_generalization.experiment import execute
from ff_prior_delivery_train.learning import load as parent_load
from ff_prior_origin_learn_bounded.mean import fit as fit_mean
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_delivery_guidance.sampling import candidate
from .protocol import ROOT,SOURCE,OLD,BASE,DEFAULT_OUT,METHODS,PROTOCOL
from .inputs import identity_for
from . import data,learning


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .audit_results import verify
    from .reporting import report
    out=Path(out).resolve()
    if out!=DEFAULT_OUT.resolve():raise ValueError('frozen independent E256T output required')
    identity,(config,index,entries,new_ids)=identity_for(source)
    deadline=time.time()+max(0.,14400.-previous_elapsed)
    with RunLock(out):
        if (out/'run.json').exists():
            if not resume or read_json(out/'run.json')['identity']!=identity:raise ValueError('run identity changed')
        else:
            atomic_json(out/'run.json',{'identity':identity,'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'TEST':False})
            atomic_json(out/'training_index.json',{'records':index,'TEST_arrays_read':False})
            atomic_json(out/'frozen_tasks.json',{'entries':entries,'before_training':True,'before_comparison':True,'used_8_tasks_are_not_unseen':True})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in identity['source_hashes']:archive.write(ROOT/p,p)
            progress('E256T_data_model_and_original_DEV_frozen',TRAIN=sum(e['split']=='TRAIN' for e in index),VAL=48,comparison_tasks=12)
        proof=read_json(ROOT/'ff_prior_expand256_train/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=identity['source_hashes'] or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_expand256_train/evidence/tests.xml'):
            raise ValueError('software proof stale')
        physical,canonical,initial,_=audit(config)
        if physical['acceptance']!=read_json(BASE/'baseline_audit.json')['acceptance']:raise ValueError('physical acceptance changed')
        atomic_json(out/'physical_contract.json',physical)
        profile=acceptance(config);ledger=Ledger(out,{'identity_hash':digest(identity)})
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=16,max_scene_models=4)
        old_bundle=parent_load(OLD/'learning/goal_loss');codec=old_bundle[1]
        training_deadline=deadline-PROTOCOL['smoke_reserve_s']
        arrays,rows,G,sreport=data.load(out,index,codec,cache,ledger,training_deadline,progress)
        mean_codec=fit_mean(out/'learning',arrays,codec,new_ids)
        try:
            training=learning.train(out/'learning/goal_loss',arrays,codec,G,sreport,new_ids,progress,training_deadline)
        except Exception as error:
            training={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
            atomic_json(out/'training_failure.json',training)
        new_bundle=learning.load(out/'learning/goal_loss') if training['status']=='PASS' else None
        atomic_json(out/'checkpoint_binding.json',{'old':file_digest(OLD/'learning/goal_loss/model.pt'),
            'new':file_digest(out/'learning/goal_loss/model.pt') if new_bundle else None,
            'old_and_new_codec_hash':digest(codec.state),'mean_codec_sha256':file_digest(out/'learning/weighted_mean.json'),
            'old_and_new_diffusion_sampler':'late4_diffusion','DEV_checkpoint_selection':False,
            'delivery_model_choice_uses_development_gate':True,'new_weights_reloaded':new_bundle is not None})
        def capacity():
            if time.time()>=deadline-180:raise RuntimeError('E256T_WALL_BUDGET_EXHAUSTED')
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=20*1024**3:raise RuntimeError('E256T_STORAGE_BUDGET_EXHAUSTED')
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            for ti,e in enumerate(entries):
                task=e['task'];sid=e['sample_id'];rr,ii,cc,transfer=cache.get(e)
                base=public_goal_initializer(task,rr,ii)
                atomic_json(out/'transfer_audits'/f'{sid}.json',dict(transfer,public_base_hash=digest(base)))
                atomic_npz(out/'comparison_tasks'/sid/'initial_state.npz',values=ii.values)
                for method in METHODS[ti%4:]+METHODS[:ti%4]:
                    bundle=None if method=='structured' else (old_bundle[0],mean_codec) if method=='learned_mean' else old_bundle if method=='frozen_diffusion' else new_bundle
                    actual_method='late4_diffusion' if method.endswith('diffusion') else method
                    records=[]
                    for k in range(5):
                        key=f'candidate_{sj}_{sid}_{method}_{k}';path=out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                        if key not in ledger.data['completed']:
                            capacity();ledger.begin(key,'candidate',14400.,720)
                            progress('original_frozen_DEV_candidates',seed=sj+1,task=ti+1,method=method,candidate=k+1)
                            sample_seed=seed+ti*101+k*997
                            if method=='expanded_diffusion' and bundle is None:
                                r={'status':'BLOCKED','eligible':False,'method':method,'candidate_id':k,'seed':sample_seed,
                                   'task_hash':task['task_hash'],'public_base_hash':digest(base),'private_witness_used':False,
                                   'reason':'TRAINING_DEPENDENCY_BLOCKED','model_calls':{},'actual_network_calls':0,'wall_s':0.}
                            else:
                                r=candidate(actual_method,k,task,base,sample_seed,bundle,rr,ii,cc,profile,deadline-180)
                            r.update(method=method,sampling_method=actual_method,sample_id=sid,split=e['split'],seed_group=sj,
                                     development_role='ORIGINAL_DEV' if ti<4 else 'ALREADY_USED_DEVELOPMENT')
                            atomic_json(path,r);ledger.finish({'status':r['status']},r['model_calls'])
                        if path.exists() and ledger.data['completed'][key]['status']!='INTERRUPTED':records.append(read_json(path))
                    key=f'execution_{sj}_{sid}_{method}';directory=out/'executions'/str(sj)/sid/method
                    if key in ledger.data['completed']:continue
                    capacity();ledger.begin(key,'execution',14400.,144)
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,
                        'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                    before=rr.calls.copy();progress('original_frozen_DEV_native_replay',seed=sj+1,task=ti+1,method=method)
                    if chosen:result=execute(directory,task,chosen['Z'],rr,ii,cc,profile,deadline-180)
                    else:result={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',result)
                    ledger.finish(result,{k:rr.calls[k]-before[k] for k in before})
                report(out,'RUNNING')
        atomic_json(out/'runtime_cache.json',cache.stats())
        progress('E256T_read_only_training_and_raw_physics_audit')
        proof=verify(out,identity,arrays,rows,G,ledger)
        atomic_json(out/'independent_verification.json',proof)
        return report(out,'FINISHED')
