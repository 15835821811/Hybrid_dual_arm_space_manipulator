"""Frozen paired smoke: strong spline, old diffusion, learned mean, latent diffusion."""
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
from ff_prior_loop import data,training
from ff_prior_loop.experiment import execute
from ff_prior_cost.sampling import sample as old_sample
from ff_prior_feasible.experiment import inputs as old_inputs,sources as previous_sources,PREVIOUS
from ff_prior_feasible.guidance import FeasibleProjector,CONTRACT as GUIDE
from ff_prior_feasible.validation import validate,CONTRACT as VALIDATE
from . import learning

ROOT=Path(__file__).resolve().parents[1]
METHODS=('structured_feasible','frozen_diffusion_feasible','learned_mean_spline','latent_diffusion_feasible')
PROTOCOL={'version':'latent_residual_smoke_1','profile':'smoke','guidance':GUIDE,'validation':VALIDATE,
    'training':learning.CONTRACT,'methods':list(METHODS),'task_groups':{'old_complex_VAL':6,'new_origin_VAL':4},
    'candidates_per_task':5,'seed':20261067,'same_tasks_seeds_as_previous':True,
    'candidate_coupled_ceiling':113,'candidate_integration_interval_ceiling':15872,'candidate_wall_ceiling_s':15.,
    'ddim_steps':16,'max_candidates':200,'max_selected_executions':40,'wall_budget_s':3600,
    'structured_bow_sigma_rad':.03,'mean_spline':'ridge center; candidate0 exact mean; candidates1-4 same 0.03rad bow as spline',
    'selection':'lowest normalized goal error among eligible; one torque execution plus replay; no replacement',
    'stability_gate':{'old_complex_eligible':27,'new_origin_eligible':18,'each_task_eligible':4,'execution_verified':10},
    'advantage_gate':'stable + no lower eligible/physical success + more physical tasks OR >=10% lower queries AND wall than BOTH spline and learned_mean_spline',
    'representation_claim':'low-rank noise and clean trajectory share TRAIN-only basis; not claimed until evidence',
    'expansion':False,'pilot':False,'study':False,'TEST':False,'DEV_executed':False,'physical_gates_changed':False,'frozen_goal_relabel':False}


def sources():
    return dict(previous_sources(),**{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted((ROOT/'ff_prior_latent').rglob('*.py'))})


def inputs(source):
    old,ds,dev,rows,files=old_inputs(source); source=Path(source).resolve()
    for e in read_json(PREVIOUS/'training_index.json')['records']:
        for k in ('record','arrays'):
            p=ROOT/e[k]
            if file_digest(p)!=e[k+'_sha256']: raise ValueError('training file changed')
            files[os.path.relpath(p,source)]=file_digest(p)
    return old,ds,dev,rows,files


def candidate(method,k,task,base,seed,bundle,rr,ii,cc,profile,deadline):
    assert_task_compatible(task,rr,ii,cc,profile); before=rr.calls.copy(); start=time.perf_counter(); cpu=time.process_time()
    deadline=min(deadline,time.time()+15.)
    network_before=bundle[0].calls if bundle is not None else 0
    projector=FeasibleProjector(task,rr,ii,cc,profile,deadline,collision_enabled=True)
    result={'status':'ERROR','eligible':False,'method':method,'candidate_id':k,'seed':seed,'task_hash':task['task_hash'],
        'public_base_hash':digest(base),'private_witness_used':False,'shared_query_ceiling':113,
        'shared_wall_ceiling_s':15.,'shared_integration_interval_ceiling':15872}
    try:
        if method in ('structured_feasible','learned_mean_spline'):
            s=np.r_[np.linspace(0,1,20)[1:-1],1.]
            z=base.copy()
            if method=='learned_mean_spline':
                codec=bundle[1]; z=codec.decode(codec.center(task['condition'],base),base)
            if method=='structured_feasible' or k>0:
                z=z+.03*np.sin(np.pi*s)[:,None]*np.random.default_rng(seed).normal(size=(1,17))
            z,info=projector(z,15); sampling={'actual_network_calls':0,'guidance':[info]}
        elif method=='frozen_diffusion_feasible': z,sampling=old_sample(bundle,base,task,seed,projector)
        elif method=='latent_diffusion_feasible': z,sampling=learning.sample(bundle,base,task,seed,projector)
        else: raise ValueError('unknown method')
        result.update(Z=z,sampling=sampling)
        result.update(validate(z,task,rr,ii,cc,profile,deadline)); result['status']='DONE'
        if time.time()>deadline: raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error: result.update(status='ERROR',eligible=False,reason=repr(error))
    calls={n:rr.calls[n]-before[n] for n in before}
    result.update(model_calls=calls,repair_coupled_calls=projector.used,validation_coupled_calls=calls['coupled']-projector.used,
        actual_network_calls=(bundle[0].calls-network_before) if bundle is not None else 0,
        integration_intervals=(128*projector.used+result['validation_integration_intervals']) if result['status']=='DONE' else None,
        wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)
    if calls['coupled']>113 or projector.used>110 or calls['coupled']-projector.used>3 or (result['integration_intervals'] or 0)>15872:
        result.update(status='ERROR',eligible=False,reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task); return result


def run(source,out,progress,resume=False,previous_elapsed=0.):
    from .reporting import report
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents: raise ValueError('independent output required')
    old,_,dev,_,files=inputs(source)
    identity={'protocol':PROTOCOL,'source':str(source),'source_files':files,'tasks_hash':digest(dev),'source_hashes':sources()}
    with RunLock(out):
        if (out/'run.json').exists() and (not resume or read_json(out/'run.json')['identity']!=identity): raise ValueError('run identity changed')
        proof=read_json(ROOT/'ff_prior_latent/evidence/software_tests.json')
        if proof['status']!='PASS' or proof['source_hashes']!=sources() or proof['junit_sha256']!=file_digest(ROOT/'ff_prior_latent/evidence/tests.xml'):
            raise ValueError('software proof missing or stale')
        physical,canonical,initial,_=audit(old['config']); profile=acceptance(old['config'])
        if physical['acceptance']!=read_json(source/'baseline_audit.json')['acceptance']: raise ValueError('physical gates changed')
        targets=read_json(PREVIOUS/'frozen_tasks.json')['evaluation_entries']
        frozen={'evaluation_entries':targets,'frozen_DEV_hash':digest(dev),'DEV_executed':False,'frozen_before_new_training':True}
        if not (out/'run.json').exists():
            atomic_json(out/'run.json',{'identity':identity,'acceptance':physical['acceptance'],
                'producer_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
            atomic_json(out/'frozen_tasks.json',frozen); atomic_json(out/'software_tests.json',proof)
            with zipfile.ZipFile(out/'executed_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in sources(): archive.write(ROOT/p,p)
        if read_json(out/'frozen_tasks.json')!=frozen: raise ValueError('frozen tasks changed')
        ledger=Ledger(out,identity); deadline=time.time()+max(0,3600-previous_elapsed)
        frozen_bundle=training.load(PREVIOUS/'learning',read_json(PREVIOUS/'learning/training.json')['metadata']['scope'])
        index=read_json(PREVIOUS/'training_index.json')['records']; data.assert_disjoint(index)
        arrays=data.training_arrays(index,canonical,initial,profile,out,progress)
        trained_report=learning.train(out/'learning',arrays,progress,deadline)
        if trained_report['status']!='PASS': raise ValueError('new training not verified')
        trained=learning.load(out/'learning')
        for ti,e in enumerate(targets):
            setup_start=time.perf_counter(); rr,ii,cc,transfer=data.runtime(e,canonical,initial); task=e['task']; sid=e['sample_id']
            setup_before=rr.calls.copy(); base=public_goal_initializer(task,rr,ii)
            atomic_json(out/'transfer_audits'/f'{sid}.json',dict(transfer,shared_setup_wall_s=time.perf_counter()-setup_start,
                shared_initializer_calls={n:rr.calls[n]-setup_before[n] for n in setup_before},shared_public_base_hash=digest(base)))
            for method in METHODS[ti%4:]+METHODS[:ti%4]:
                results=[]
                for k in range(5):
                    key=f'candidate_{sid}_{method}_{k}'; path=out/'candidates'/sid/method/f'{k:02d}.json'
                    if key not in ledger.data['completed']:
                        ledger.begin(key,'candidate',3600,200); progress('paired_VAL_candidates',task=ti+1,total=10,method=method,candidate=k+1)
                        bundle=None if method=='structured_feasible' else frozen_bundle if method=='frozen_diffusion_feasible' else trained
                        result=candidate(method,k,task,base,20261067+ti*101+k*997,bundle,rr,ii,cc,profile,deadline)
                        result.update(sample_id=sid,split='VAL',group=e['group']); atomic_json(path,result)
                        ledger.finish({'status':result['status']},result['model_calls'])
                    if path.exists(): results.append(read_json(path))
                key=f'execution_{sid}_{method}'; directory=out/'executions'/sid/method
                if key in ledger.data['completed']: continue
                selected=min((r for r in results if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                atomic_json(directory/'selection.json',{'selected_id':selected['candidate_id'] if selected else None,
                    'task_hash':task['task_hash'],'selection_before_execution':True,'rule':PROTOCOL['selection']})
                ledger.begin(key,'execution',3600,40); before=rr.calls.copy(); progress('selected_torque_replay',task=ti+1,method=method)
                if selected: result=execute(directory,task,selected['Z'],rr,ii,cc,profile,deadline)
                else:
                    result={'status':'NO_CANDIDATE'}; atomic_json(directory/'outcome.json',result)
                ledger.finish(result,{n:rr.calls[n]-before[n] for n in before})
            report(out,ledger,'RUNNING')
        if any(file_digest(source/p)!=h for p,h in files.items()) or sources()!=identity['source_hashes']: raise ValueError('input/runtime source changed')
        atomic_json(out/'verification.json',{'source_inputs_unchanged':True,'source_code_unchanged':True,'original_DEV_unchanged':True,
            'DEV_executed':False,'TEST_arrays_read':False,'physical_gates_unchanged':True,'frozen_goal_relabelled':False,'expansion':False})
        incomplete=ledger.data.get('unresolved_call_accounting') or any(r['status'] in ('ERROR','INTERRUPTED','BLOCKED') for r in ledger.data['completed'].values())
        return report(out,ledger,'FINISHED_WITH_BLOCKED_ITEMS' if incomplete else 'FINISHED')
