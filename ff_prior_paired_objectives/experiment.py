from pathlib import Path
import shutil,subprocess,time,traceback,zipfile
import torch
from ff_prior.storage import read_json,digest,file_digest,RunLock
from ff_prior_coverage_train.storage import atomic_json,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_origin_learn_bounded.learning import load as old_load
from ff_prior_origin_learn_bounded.mean import fit as fit_mean
from .protocol import ROOT,SOURCE,WARM,PROTOCOL as P
from .identity import identity_for,sources
from . import inputs,objectives,diagnostic,learning,smoke,audit_results
from .reporting import report

def run(source,out,progress,resume=False,previous_elapsed=0.):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('independent output required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    deadline=time.time()+max(0,P['wall_budget_s']-previous_elapsed)
    identity,(config,index,entries,targets)=identity_for(source)
    with RunLock(out):
        tests=read_json(ROOT/'ff_prior_paired_objectives/evidence/software_tests.json')
        if tests['status']!='PASS' or tests['source_hashes']!=sources() or tests['junit_sha256']!=file_digest(ROOT/'ff_prior_paired_objectives/evidence/tests.xml'):raise ValueError('software proof stale')
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity):raise ValueError('frozen run changed')
        physical,canonical,initial,_=audit(config);profile=acceptance(config)
        if physical['acceptance']!=read_json(source/'run.json')['acceptance']:raise ValueError('physical gates changed')
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'joint_lower':canonical.lower,'joint_upper':canonical.upper,
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',{'entries':entries,'before_training':True,'before_comparison':True,'new_validation_role':'REUSED_DEV'})
            atomic_json(out/'training_index.json',{'records':index,'TEST_arrays_read':False})
            atomic_json(out/'fixed_TRAIN_targets.json',{'entries':targets,'reused':True})
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sources():z.write(ROOT/p,p)
        ledger=Ledger(out,{'identity_hash':digest(identity)})
        if ledger.data.get('unresolved_call_accounting'):raise RuntimeError('unknown interrupted physical work; review before continuation')
        cache=TaskOriginSceneCache(canonical,initial,profile,max_entries=64,max_scene_models=4,
            max_private_bytes=P['max_process_private_bytes'],min_available_bytes=P['minimum_available_memory_bytes'])
        old_bundle=old_load(WARM/'learning/goal_loss');codec=old_bundle[1]
        arrays,rows,G=inputs.load(out,index,codec,progress);G=torch.tensor(G,dtype=torch.float32)
        new_ids={e['sample_id'] for e in index if e.get('producer_run_id')==SOURCE.name}
        data=objectives.prepare(arrays,codec,rows,targets,new_ids)
        single=diagnostic.run(out,old_bundle[0],codec,data,arrays,rows,G,cache,ledger,deadline,progress)
        trained=compared=False
        if single['status']=='PASS':
            selection=learning.train(out,codec,data,arrays,rows,G,cache,ledger,deadline-P['training_reserve_for_smoke_s'],progress)
            progress('audit_paired_training_before_DEV')
            proof=audit_results.verify_training(out,codec,data,arrays,rows);atomic_json(out/'training_verification.json',proof);trained=True
            if selection['status']=='PASS':
                chosen=out/'learning'/selection['selected_arm'];new_bundle=learning.load(chosen)
                mean_codec=fit_mean(out/'learning',arrays,codec,new_ids)
                atomic_json(out/'checkpoint_binding.json',{'old':file_digest(WARM/'learning/goal_loss/model.pt'),
                    'new':file_digest(chosen/'model.pt'),'selected_arm':selection['selected_arm'],
                    'fixed_codec':digest(codec.state),'mean_codec_sha256':file_digest(out/'learning/weighted_mean.json'),
                    'validation_checkpoint_selection':False})
                smoke.run(out,entries,cache,profile,ledger,deadline,progress,old_bundle,new_bundle,mean_codec);compared=True
        atomic_json(out/'runtime_cache_audit.json',cache.stats());progress('raw_results_audit')
        try:proof=audit_results.verify(out,identity,codec,data,arrays,rows,trained,compared)
        except Exception as error:proof={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc()}
        atomic_json(out/'independent_verification.json',proof)
        status='FINISHED' if compared and proof['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS'
        return report(out,status)
