"""Independent saved-evidence review. No training, optimizer.step or physics calls."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json, hashlib, zipfile, time, traceback
from collections import Counter
import numpy as np
import torch
from ff_prior.storage import read_json, digest, file_digest
from ff_prior_full_control.protocol import ROOT,OUT,P,DATA,SOURCE
from ff_prior_full_control.data import load as load_data
from ff_prior_full_control.inputs import sources
from ff_prior_full_control.learning import load as load_model
from ff_prior_latent.learning import LatentDenoiser
from ff_prior.diffusion import schedule, weight_hash
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior.contracts import MODEL_GATES,EXECUTION_GATES
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_delivery_guidance.audit_results import check_candidate_validation

DEST=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_full_control_run01'
DEST.mkdir(parents=True,exist_ok=True)
RESULT={'scope':'Independent read-only raw evidence and neural math audit; no new experiment/training/physics/optimizer.step','checks':{},'errors':[]}
INPUTS={}
def bind(p):
    p=Path(p);INPUTS[p.relative_to(ROOT).as_posix()]={'sha256':file_digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
def save():
    (DEST/'checks.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'inputs.json').write_text(json.dumps(INPUTS,ensure_ascii=False,indent=2),encoding='utf-8')
def done(k,v):
    RESULT['checks'][k]=v;save();print(k+': PASS',flush=True)
def delta(a,b):return float(torch.max(torch.abs(a-b)).item()) if a.numel() else 0.
def vector(g):return torch.cat([v.reshape(-1) for v in g])

def main():
    torch.set_num_threads(2)
    # Guard against accidental stepping even if a reused function changes in future.
    def forbidden(*a,**kw):raise RuntimeError('Reviewer audit forbids optimizer.step and physics execution')
    torch.optim.AdamW.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    mujoco.mj_forward=forbidden
    run=read_json(OUT/'run.json');ident=run['identity'];selection=read_json(OUT/'frozen_tasks.json')
    assert len(ident['source_hashes'])==165 and len(ident['binding']['files'])==15411
    assert sources()==ident['source_hashes']
    with zipfile.ZipFile(OUT/'executed_source.zip') as z:
        for p,h in ident['source_hashes'].items():
            assert hashlib.sha256(z.read(p)).hexdigest()==h
            assert file_digest(ROOT/p)==h;bind(ROOT/p)
    for p,h in ident['binding']['files'].items():
        assert file_digest(ROOT/p)==h,p;bind(ROOT/p)
    assert selection==ident['selection']
    done('identity',{'source_files':165,'bound_input_files':15411,'producer_head':run['producer_head'],'archive_and_live_match':True})
    # Mathematical reconstruction of the TRAIN-only codec, no model-world call.
    codec,arrays,rows,data=load_data();saved_codec=read_json(OUT/'codec.json')
    assert digest(codec.state)==digest(saved_codec)
    z,bases,conditions,ids,splits,states=arrays;tr=data['tr'];va=data['va']
    assert not {rows[i]['family_id'] for i in tr}&{rows[i]['family_id'] for i in va}
    assert not {rows[i]['task']['task_hash'] for i in tr}&set(selection['forbidden_TRAIN_task_hashes'])
    assert set(codec.state['fit_ids'])=={ids[i] for i in tr}
    assert not set(codec.state['fit_ids'])&{ids[i] for i in va}
    done('data_codec',{'TRAIN':len(tr),'VAL':len(va),'TRAIN_VAL_family_disjoint':True,'TRAIN_DEV_task_disjoint':True,'codec_recomputed_TRAIN_only_exact':True,'all_rows_float64_roundtrip_max_rad':data['reconstruction_max_rad'],'qualification_levels':dict(Counter(r['validation_level'] for r in rows))})
    reconstruction=[]
    for row in selection['reconstruction']:
        d=OUT/'reconstruction'/row['sample_id'];r=read_json(d/'validation.json');prep=read_json(d/'preparation.json')
        check_candidate_validation(r,row['task']);assert r['eligible']
        with np.load(d/'representation.npz') as a:
            assert np.array_equal(a['model_normalized'],a['normalized'].astype(np.float32).astype(np.float64))
            decoded=codec.decode(codec.coefficients(a['model_normalized'],codec.center(row['condition'],a['base'])),a['base'])
            assert np.array_equal(decoded,a['decoded']) and np.array_equal(a['prepared'],r['Z'])
            assert np.array_equal(a['prepared'],a['decoded'])
            err=float(abs(decoded-a['teacher']).max());assert err<=1e-6 and err==prep['decoded_error_rad']==prep['prepared_error_rad']
            assert digest({'values':a['initial_values'],'state_mask':prep['state_mask'],'model_hash':prep['model_hash']})==row['task']['contract']['initial_state_id']
        reconstruction.append({'id':row['sample_id'],'float32_target_max_rad':err,'eligible':r['eligible'],'grids':[v['intervals'] for v in r['validation_grids']]})
    done('six_reconstructions',reconstruction)
    directory=OUT/'learning/full';training=read_json(directory/'training.json');progress=torch.load(directory/'progress.pt',weights_only=False,map_location='cpu')
    model,loaded_codec=load_model(directory)
    assert progress['updates']==training['updates']==6000 and progress['rollouts']==training['full16_teacher_batches']==1200
    assert {int(v['step']) for v in progress['optimizer']['state'].values()}=={6000}
    assert weight_hash(model)==training['trained_weight_hash']!=training['initial_weight_hash']
    assert all(torch.equal(progress['weights'][k],v) for k,v in model.state_dict().items())
    torch.manual_seed(P['training_seed']);initial_model=LatentDenoiser(323,65)
    assert weight_hash(initial_model)==training['initial_weight_hash']
    assert all(torch.equal(v,torch.load(directory/'initial.pt',weights_only=False,map_location='cpu')['weights'][k]) for k,v in initial_model.state_dict().items())
    sampler=CoverageSampler(data['keys'],P['training_seed']+2);rng=torch.Generator().manual_seed(P['training_seed'])
    counts=np.zeros(len(tr),dtype=np.int64);proofs=[];saved_steps={}
    for step in range(6000):
        batch=sampler.take(64);np.add.at(counts,batch,1)
        t=torch.randint(0,1000,(64,),generator=rng);noise=torch.randn((64,323),generator=rng)
        proofs.append(digest({'step':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
        if step+1 in P['audit_updates']:
            s=torch.load(directory/'updates'/f'{step+1:04d}'/'before.pt',weights_only=False,map_location='cpu')
            assert np.array_equal(s['indices'],tr[batch]) and torch.equal(s['t'],t) and torch.equal(s['noise'],noise)
            assert s['full_seeds']==[P['rollout_seed']+step*101+k for k in range(8)]
            saved_steps[step+1]=s
        if (step+1)%20==0:
            expected={'start':step-18,'end':step+1,'draws':1280,'digest':digest(proofs)}
            assert training['segments'][step//20]==expected
            proofs=[]
    assert training['segments']==progress['segments']
    assert np.array_equal(counts,progress['counts'])
    assert training['train_draw_counts']==dict(zip([ids[i] for i in tr],counts.tolist()))
    assert counts.min()>0 and counts.sum()==384000
    assert digest(sampler.state())==digest(progress['sampler']) and torch.equal(rng.get_state(),progress['rng'])
    done('training_consumption',{'updates_in_all_optimizer_states':6000,'committed_segments_recreated':300,'all_batch_t_noise_segment_digests_recreated':True,'TRAIN_rows_seen':int((counts>0).sum()),'draws':int(counts.sum()),'min_row_draws':int(counts.min()),'max_row_draws':int(counts.max()),'full16_teacher_batches_from_source_and_counter':1200,'graph_paths_from_batch8':9600,'neural_calls_per_teacher_batch':16,'note':'All segment inputs/counters reconstructed; retained gradient checks cover four fixed updates, not all 6000.'})
    updates=[]
    for step,before in saved_steps.items():
        d=directory/'updates'/f'{step:04d}';after=torch.load(d/'after.pt',weights_only=False,map_location='cpu');record=read_json(d/'update.json')
        model=LatentDenoiser(323,65);model.load_state_dict(before['weights']);params=tuple(model.parameters())
        idx=before['indices'];t=before['t'];noise=before['noise'];x=data['x'][idx];c=data['c'][idx];a=data['alpha'][t,None]
        v=model(a.sqrt()*x+(1-a).sqrt()*noise,t,c)
        loss=(v-(a.sqrt()*noise-(1-a).sqrt()*x)).square().mean()
        base=vector(torch.autograd.grad(loss,params)).detach();extra=None;coef=0.;tv=None
        calls=0
        if step%5==0:
            # Independently implement the complete differentiable 16-step v-DDIM equations.
            full=torch.cat([torch.randn((1,323),generator=torch.Generator().manual_seed(s)) for s in before['full_seeds']]).double()
            alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int)
            for j,tau in enumerate(times):
                aa=alpha[tau];pred=model(full.float(),torch.full((8,),int(tau)),c[:8]).double();calls+=1
                clean=aa.sqrt()*full-(1-aa).sqrt()*pred
                eps=(full-aa.sqrt()*clean)/(1-aa).sqrt()
                nxt=alpha[times[j+1]] if j<15 else full.new_tensor(1.)
                full=nxt.sqrt()*clean+(1-nxt).sqrt()*eps
            scale=torch.tensor(codec.latent_scale,dtype=torch.float64)
            tv=((full-data['x'][idx[:8]].double())*scale).square().mean()/scale.square().mean()
            extra=vector(torch.autograd.grad(tv,params)).detach()
            coef=min(1.,.5*float(base.norm())/max(float(extra.norm()),1e-30))
        mixed=base if extra is None else base+coef*extra
        with np.load(d/'gradients.npz') as g:
            errors={'base':delta(base,torch.from_numpy(g['base'])),'teacher':0. if extra is None else delta(extra,torch.from_numpy(g['teacher'])),'mixed':delta(mixed,torch.from_numpy(g['mixed']))}
            assert max(errors.values())<=1e-7,errors
        assert abs(float(loss)-record['v_loss'])<=1e-7 and abs(coef-record['teacher_scale'])<=1e-7
        if tv is not None:assert abs(float(tv)-record['full16_loss'])<=1e-7
        pieces=list(mixed.split([p.numel() for p in params]));norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in pieces]))
        assert abs(float(norm)-record['clip_norm'])<=1e-7
        clip=min(1.,1./(float(norm)+1e-6));group=before['optimizer']['param_groups'][0]
        assert group['lr']==2e-4 and group['weight_decay']==.01 and not group['amsgrad']
        b1,b2=group['betas'];lr=group['lr'];eps=group['eps'];weight_errors=[];moment_errors=[]
        for (name,param),pid,gradient in zip(model.named_parameters(),group['params'],pieces):
            old=before['optimizer']['state'].get(pid,{})
            m=old.get('exp_avg',torch.zeros_like(param)).clone();vv=old.get('exp_avg_sq',torch.zeros_like(param)).clone()
            n=int(old.get('step',0))+1;assert n==step
            grad=gradient.reshape_as(param)*clip
            # Same arithmetic order as scalar AdamW, explicitly without optimizer construction/step.
            m.lerp_(grad,1-b1);vv.mul_(b2).addcmul_(grad,grad,value=1-b2)
            predicted=param.detach().clone().mul_(1-lr*group['weight_decay'])
            denom=vv.sqrt()/(1-b2**n)**.5;denom.add_(eps)
            predicted.addcdiv_(m,denom,value=-lr/(1-b1**n))
            actual=after['optimizer']['state'][pid]
            weight_errors.append(delta(predicted,after['weights'][name]));moment_errors += [delta(m,actual['exp_avg']),delta(vv,actual['exp_avg_sq'])]
            assert int(actual['step'])==step
        assert max(weight_errors)<=2e-7 and max(moment_errors)<=2e-7
        updates.append({'update':step,'v_loss':float(loss.detach()),'teacher_loss':None if tv is None else float(tv.detach()),'teacher_norm_scale':coef,'gradient_max_abs_errors':errors,'manual_AdamW_weight_max_abs_error':max(weight_errors),'manual_AdamW_moments_max_abs_error':max(moment_errors),'full_DDIM_forward_calls_for_VJP':calls,'optimizer_step_called':False})
        done('four_update_neural_VJP_manual_AdamW',updates)
    records=[];failures=Counter();execunits=Counter();physics=[]
    for scope in ('DEV','APPLICATION_DEV'):
        paths=sorted((OUT/'comparison'/scope).glob('*/*/candidates/*.json'))
        scope_records=[]
        for p in paths:
            r=read_json(p);task=next(e['task'] for e in selection['entries']+selection['application'] if e['task']['task_hash']==r['task_hash'])
            check_candidate_validation(r,task);assert r['status']=='DONE'
            assert r['model_calls']['coupled']<=113 and r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
            assert r['wall_s']<=15 and r['integration_intervals']<=15872
            assert r['actual_network_calls']==16 and len(r['sampling']['steps'])==16
            assert not r['private_witness_used']
            for k in MODEL_GATES:
                if not r['validation']['gates'].get(k,False):failures[scope+':'+k]+=1
            records.append(r);scope_records.append(r)
        for d in sorted((OUT/'comparison'/scope).glob('*/*/execution')):
            rs=[read_json(p) for p in sorted((d.parent/'candidates').glob('*.json'))]
            chosen=min((r for r in rs if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
            assert read_json(d/'selection.json')['selected_id']==(None if chosen is None else chosen['candidate_id'])
            outcome=read_json(d/'outcome.json')['status'];execunits[scope+':'+outcome]+=1
            if chosen is None:assert outcome=='NO_CANDIDATE' and not (d/'report.json').exists()
        done(scope+'_candidates',{'total':len(scope_records),'eligible':sum(r['eligible'] for r in scope_records),'model_valid':sum(r['validation']['validation_level']=='MODEL_VALID' for r in scope_records),'frozen_goal_pass':sum(r['frozen_goal_pass'] for r in scope_records),'max_queries':max(r['model_calls']['coupled'] for r in scope_records),'max_intervals':max(r['integration_intervals'] for r in scope_records),'max_wall_s':max(r['wall_s'] for r in scope_records)})
    assert len(records)==190
    for p in sorted(OUT.rglob('execution/report.json')):
        d=p.parent;r=read_json(p);task=read_json(d/'frozen_task.json');r,n=audit_physics(d,task,r['metadata']['reference_hash'])
        assert r['level']=='EXECUTION_VERIFIED' and all(r['gates'][k] for k in EXECUTION_GATES)
        with np.load(d/'native_trace.npz') as a,np.load(d/'replay_trace.npz') as b:
            assert np.array_equal(a['ctrl'],b['ctrl']) and a['ctrl'].shape[1]==67
            for k in ('qpos','qvel','act','time','qacc_warmstart'):
                assert float(np.max(abs(a[k]-b[k]),initial=0))==r['replay']['maximum_errors'][k]<=1e-10
            assert not r['replay']['copies_actual_state_each_step']
        with np.load(d/'geometry_scans.npz') as a:
            assert float(a['minima'].min())==r['minimum_clearance_m']>=0
            assert len(a['minima'])==r['checked_states']
        physics.append({'directory':d.relative_to(OUT).as_posix(),'steps':n,'min_clearance_m':r['minimum_clearance_m'],'max_hold_position_m':max(r['hold_max_position_error_m']),'max_hold_rotation_rad':max(r['hold_max_rotation_error_rad']),'gates':r['gates'],'native_and_replay_equal':True})
    assert len(physics)==16
    done('new_executions',{'total':16,'native_plus_replay_steps':sum(r['steps'] for r in physics),'execution_unit_outcomes':dict(execunits),'raw_reports':physics,'historical_physics_rerun':False})
    done('new_failed_gates',dict(failures))
    unguided=[]
    for sj in range(2):
        for j in range(48):
            r=read_json(OUT/'unguided'/str(sj)/f'{j:02d}.json');old=read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json')
            assert r['seed']==old['seed'] and r['task_hash']==old['task_hash'] and len(r['steps'])==16 and r['physics_guidance_calls']==0
            task=rows[va[j]]['task'];check_candidate_validation(dict(r,Z=r['prepared_Z']),task)
            unguided.append({'seed_group':sj,'task_hash':r['task_hash'],'new_eligible':r['eligible'],'old_eligible':old['eligible'],'new_model_valid':r['validation']['validation_level']=='MODEL_VALID','new_frozen_goal_pass':r['frozen_goal_pass']})
    done('unguided_full_denominator',{'total':len(unguided),'new_eligible':sum(r['new_eligible'] for r in unguided),'old_eligible':sum(r['old_eligible'] for r in unguided),'new_model_valid':sum(r['new_model_valid'] for r in unguided),'new_frozen_goal_pass':sum(r['new_frozen_goal_pass'] for r in unguided),'same_task_and_seed_all':True})
    historical={}
    for scope in ('DEV','APPLICATION_DEV'):
        for method in ('old_aligned','structured','learned_mean'):
            es=[e for e in selection['reuse'] if e['scope']==scope and e['method']==method]
            rs=[read_json(ROOT/p) for e in es for p in e['candidates']]
            outcomes=Counter(read_json(ROOT/e['execution']/'outcome.json')['status'] for e in es)
            historical[scope+':'+method]={'candidate_denominator':len(rs),'eligible':sum(r['eligible'] for r in rs),'execution_unit_denominator':len(es),'saved_historical_outcomes':dict(outcomes),'identity_only_reuse':True}
    done('historical_full_denominators',historical)
    ledger=read_json(OUT/'ledger.json');assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert read_json(directory/'journal.json')['status']=='COMMITTED'
    assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==sum(r['steps'] for r in physics)
    done('ledger',{'no_unknown_work':True,'completed_units':len(ledger['completed']),'calls':ledger['calls'],'reserved_coupled_upper':read_json(OUT/'budget.json')['reserved_coupled_upper']})
    audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json'
    done('paper_claim_audit',read_json(audit) if audit.exists() else {'status':'unavailable','verdict_label':'provisional — no paper claim audit available'})
    # Bind every new artifact actually available at review time; immutable run only read.
    for p in OUT.rglob('*'):
        if p.is_file():bind(p)
    bind(ROOT/'ff_prior_full_control/docs/EXPERIMENT_PLAN.md');bind(Path(__file__).resolve())
    RESULT['status']='PASS';save()

if __name__=='__main__':
    try:main()
    except Exception:
        RESULT['status']='FAIL';RESULT['errors'].append(traceback.format_exc());save();raise
