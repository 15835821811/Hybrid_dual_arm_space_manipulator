"""Fresh read-only review: saved arrays and neural VJP; never train or simulate."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, hashlib, zipfile, traceback
from collections import Counter
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_control_transfer.protocol import ROOT,OUT,SOURCE,DATA,P
from ff_prior_control_transfer.data import load
from ff_prior_control_transfer.inputs import sources
from ff_prior_control_transfer.model import TransferDenoiser,initial_noise
from ff_prior_control_transfer.sampling import unguided_sample
from ff_prior_sample_alignment.learning import load as old_load
from ff_prior_endpoint_jacobian.comparison import unguided_sample as old_sample
from ff_prior.diffusion import schedule,weight_hash
from ff_prior.coupled_predictor import pose_errors
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics

DEST=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01'
DEST.mkdir(parents=True,exist_ok=True)
R={'status':'RUNNING','scope':'Fresh saved-evidence and neural-only review; no optimizer.step, simulation, old audit execution, or resume','checks':{},'errors':[]}
INPUTS={}
def bind(p):
    p=Path(p);INPUTS[p.relative_to(ROOT).as_posix()]={'sha256':file_digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
def save():
    (DEST/'checks.json').write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding='utf8')
    (DEST/'inputs.json').write_text(json.dumps(INPUTS,ensure_ascii=False,indent=2),encoding='utf8')
def done(k,v):R['checks'][k]=v;save();print(k+': PASS',flush=True)
def delta(a,b):return float(abs(a-b).max()) if a.numel() else 0.
def independent_loss(model,codec,c,target,seeds):
    # Spell out full DDIM and both objective terms independently of production rollout/loss.
    x=torch.cat([torch.cat([torch.randn((1,102),generator=torch.Generator().manual_seed(s)),torch.randn((1,221),generator=torch.Generator().manual_seed(s+971003))],1) for s in seeds]).double()
    alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int)
    for j,t in enumerate(times):
        tau=torch.full((len(seeds),),int(t));aa=alpha[t]
        v=model.low(x[:,:102].float(),tau,c).double()
        freq=torch.exp(torch.arange(16)*(-np.log(10000)/15));phase=tau[:,None]*freq[None]
        head=model.detail(torch.cat((x.float(),c,phase.sin(),phase.cos()),1)).double()
        detail=model.detail_limit*torch.tanh(head/model.detail_limit)
        clean=torch.cat((aa.sqrt()*x[:,:102]-(1-aa).sqrt()*v,detail),1)
        eps=(x-aa.sqrt()*clean)/(1-aa).sqrt()
        nxt=alpha[times[j+1]] if j<15 else x.new_tensor(1.)
        x=nxt.sqrt()*clean+(1-nxt).sqrt()*eps
    scale=torch.tensor(codec.latent_scale,dtype=torch.float64)
    err=(x-target.double())*scale
    physical=torch.einsum('tk,nkj->ntj',torch.tensor(codec.basis),err.reshape(-1,19,17))
    return (physical.square().mean()+.25*torch.diff(physical,n=2,dim=1).square().mean())/scale.square().mean()

def physical_saved(d,task):
    r=read_json(d/'validation.json');check_candidate_validation(r,task)
    with np.load(d/'prediction.npz') as pred:
        assert all(np.isfinite(pred[k]).all() for k in pred.files)
        pos,rot=pose_errors(pred['ee'][-1],task['condition']['goals'])
        assert np.max(abs(pos-r['validation']['frozen_goal_position_error_m']))<1e-12
        assert np.max(abs(rot-r['validation']['frozen_goal_rotation_error_rad']))<1e-12
        final=read_json(d/'validation.json')['validation_grids'][-1]['intervals']
        with np.load(d/f'grid_{final}.npz') as grid:
            assert all(np.array_equal(pred[k],grid[k]) for k in pred.files)
    for g in r['validation_grids']:
        n=g['intervals'];geom=g['geometry']
        with np.load(d/f'grid_{n}_geometry_scans.npz') as a:
            rows=read_json(d/f'grid_{n}_geometry_scans.json')
            assert np.array_equal(a['minima'],[v['minimum_clearance_m'] for v in rows])
            assert float(a['minima'].min())==geom['minimum_clearance_m']
            assert g['gates']['geometry']==bool(a['minima'].min()>=0)
            assert g['gates']['geometry_refinement']==bool(geom['refinement_difference_m']<=.001 and geom['maximum_geometry_step_m']<=.002)
    return r

def main():
    torch.set_num_threads(2)
    def forbidden(*a,**k):raise RuntimeError('No optimizer update or physics allowed in reviewer')
    torch.optim.AdamW.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden;mujoco.mj_forward=forbidden
    run=read_json(OUT/'run.json');ident=run['identity'];sel=read_json(OUT/'frozen_tasks.json')
    assert len(ident['source_hashes'])==170 and len(ident['binding']['files'])==15412
    assert sources()==ident['source_hashes'] and sel==ident['selection']
    with zipfile.ZipFile(OUT/'executed_source.zip') as archive:
        for p,h in ident['source_hashes'].items():
            assert hashlib.sha256(archive.read(p)).hexdigest()==h==file_digest(ROOT/p);bind(ROOT/p)
    for p,h in ident['binding']['files'].items():assert file_digest(ROOT/p)==h,p;bind(ROOT/p)
    done('identity',dict(source_files=170,bound_files=15412,producer_head=run.get('producer_head'),frozen_archive_live_match=True))
    codec,arrays,rows,data=load();z,bases,conditions,ids,splits,states=arrays;tr=data['tr'];va=data['va']
    assert digest(codec.state)==digest(read_json(OUT/'codec.json'))
    assert not {rows[i]['family_id'] for i in tr}&{rows[i]['family_id'] for i in va}
    assert not {rows[i]['task']['task_hash'] for i in tr}&set(sel['forbidden_TRAIN_task_hashes'])
    assert set(codec.state['fit_ids'])=={ids[i] for i in tr}
    oldmodel,oldcodec=old_load(SOURCE/'learning/aligned')
    assert np.array_equal(codec.basis[:,:6],oldcodec.basis)
    for key in ('feature_mean','feature_scale'):assert np.array_equal(getattr(codec,key),getattr(oldcodec,key))
    for key in ('latent_mean','latent_scale'):assert np.array_equal(getattr(codec,key)[:102],getattr(oldcodec,key))
    assert np.array_equal(codec.ridge[:,:102],oldcodec.ridge) and np.count_nonzero(codec.ridge[:,102:])==0
    det=np.einsum('tk,ntj->nkj',codec.basis[:,6:],z[tr]-bases[tr]).reshape(len(tr),221)
    scales=np.maximum(det.std(0),1e-4);limits=np.maximum(3,abs(det/scales).max(0)+1)
    assert np.array_equal(scales,codec.latent_scale[102:]) and np.array_equal(limits,codec.state['detail_limit'])
    assert np.all(abs(det/scales)<limits)
    assert np.linalg.matrix_rank(codec.basis)==19
    expected=sorted([r for r in rows if r['split']=='TRAIN' and r['sample_id'].startswith('complex_train_')],key=lambda r:r['sample_id'])+sorted([r for r in rows if r['split']=='TRAIN' and not r['sample_id'].startswith('complex_train_')],key=lambda r:digest(r['sample_id']))[:2]
    assert sel['diagnostic']==expected
    done('codec_split_and_selection',dict(TRAIN=len(tr),VAL=len(va),rank=19,dimensions=323,all_rows_roundtrip_max_rad=data['reconstruction_max_rad'],orthogonality_max=float(abs(codec.basis.T@codec.basis-np.eye(19)).max()),all_TRAIN_inside_detail_bounds=True,min_limit=float(limits.min()),max_limit=float(limits.max()),TRAIN_VAL_family_disjoint=True,TRAIN_DEV_task_disjoint=True,fixed_diagnostic_ids=[r['sample_id'] for r in expected],qualification_levels=dict(Counter(r['validation_level'] for r in rows))))
    recon=[]
    for row in sel['reconstruction']:
        d=OUT/'reconstruction'/row['sample_id'];r=physical_saved(d,row['task']);assert r['eligible'];prep=read_json(d/'preparation.json')
        with np.load(d/'representation.npz') as a:
            assert np.array_equal(a['model_normalized'],a['normalized'].astype(np.float32).astype(np.float64))
            decoded=codec.decode(codec.coefficients(a['model_normalized'],codec.center(row['condition'],a['base'])),a['base'])
            assert np.array_equal(decoded,a['decoded']) and np.array_equal(decoded,a['prepared']) and np.array_equal(a['prepared'],r['Z'])
            error=float(abs(decoded-a['teacher']).max());assert error==prep['decoded_error_rad']<=1e-6
            assert digest({'values':a['initial_values'],'state_mask':prep['state_mask'],'model_hash':prep['model_hash']})==row['task']['contract']['initial_state_id']
        recon.append(dict(id=row['sample_id'],float32_error_rad=error,eligible=True))
    done('reconstruction',recon)
    diag=OUT/'TRAIN_diagnostic';gate=read_json(diag/'gate.json')
    before=torch.load(diag/'before.pt',map_location='cpu',weights_only=False);after=torch.load(diag/'after.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(P['training_seed']);model=TransferDenoiser(codec)
    assert all(torch.equal(v,before['weights'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,oldmodel.state_dict()[k]) for k,v in model.low.state_dict().items())
    assert torch.count_nonzero(model.detail[-1].weight)==0 and torch.count_nonzero(model.detail[-1].bias)==0
    assert weight_hash(model)==gate['initial_hash'];initial_hash=weight_hash(model)
    fixed=[(r,seed+j*997) for j,r in enumerate(sel['diagnostic']) for seed in P['diagnostic_seeds']]
    idx=np.array([data['ids'][r['sample_id']] for r,s in fixed]);seeds=[s for r,s in fixed]
    assert np.array_equal(idx,before['indices']) and seeds==before['seeds']==after['seeds']
    for j,s in enumerate(seeds):assert torch.equal(initial_noise([s])[0,:102].float(),torch.randn((1,102),generator=torch.Generator().manual_seed(s))[0])
    migrations=[]
    for j,(r,s) in enumerate(fixed):
        new,_=unguided_sample((model,codec),bases[idx[j]],r['task'],s);old,_=old_sample((oldmodel,oldcodec),bases[idx[j]],r['task'],s)
        with np.load(diag/'migration'/f'{j:02d}.npz') as a:assert np.array_equal(new,a['new']) and np.array_equal(old,a['old'])
        migrations.append(float(abs(new-old).max()))
    assert max(migrations)<=1e-12 and migrations==read_json(diag/'migration.json')['errors']
    done('migration',dict(cases=16,max_raw_rad=max(migrations),old_low_weights_equal=True,detail_clean_zero=True,old_RNG_preserved=True,initial_hash=initial_hash))
    c=data['c'][idx];target=data['x'][idx];losses={};phase_records={};details=[]
    for phase,state in [('before',before),('after',after)]:
        model.load_state_dict(state['weights'])
        with torch.no_grad():losses[phase]=float(independent_loss(model,codec,c,target,seeds))
        assert abs(losses[phase]-gate[phase+'_loss'])<1e-12
        phase_records[phase]=[]
        for j,(row,s) in enumerate(fixed):
            d=diag/phase/f'{j:02d}';r=physical_saved(d,row['task']);identity=read_json(d/'identity.json')
            assert identity['sample_id']==row['sample_id'] and identity['seed']==s and identity['weight_hash']==weight_hash(model)
            raw,_=unguided_sample((model,codec),bases[idx[j]],row['task'],s)
            with np.load(d/'controls.npz') as a:
                assert np.array_equal(raw,a['raw']) and np.array_equal(a['prepared'],r['Z']) and np.array_equal(a['teacher'],z[idx[j]])
                shrink=identity['preparation']['coefficient_shrink'];prepared=raw if shrink==1 else bases[idx[j]]+shrink*(raw-bases[idx[j]])
                assert np.array_equal(prepared,a['prepared'])
            v=r['validation'];p=max(v['frozen_goal_position_error_m'])*1000;a=np.rad2deg(max(v['frozen_goal_rotation_error_rad']))
            phase_records[phase].append(r)
            details.append(dict(phase=phase,index=j,id=row['sample_id'],seed=s,eligible=r['eligible'],position_mm=float(p),rotation_deg=float(a),Q=float(max(p/5,a)),failed_gates=[k for k,vv in v['gates'].items() if not vv],shrink=shrink))
    aggregates={}
    for phase,rr in phase_records.items():
        ds=[v for v in details if v['phase']==phase];q=[v['Q'] for v in ds]
        rough=float(np.sqrt(np.mean([np.mean(np.diff(np.asarray(r['Z']),n=2,axis=0)**2) for r in rr])))
        assert abs(rough-gate[phase]['roughness_rms_rad'])<1e-15 and np.allclose(q,gate[phase]['Q'],rtol=0,atol=1e-12)
        def dist(x):return dict(mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(max(x)))
        aggregates[phase]=dict(eligible=sum(r['eligible'] for r in rr),model_valid=sum(r['validation']['validation_level']=='MODEL_VALID' for r in rr),median_Q=float(np.median(q)),roughness_rms_rad=rough,position_mm=dist([v['position_mm'] for v in ds]),rotation_deg=dist([v['rotation_deg'] for v in ds]),missing=0,loss=losses[phase])
    regressions=[j for j in range(16) if phase_records['before'][j]['eligible'] and not phase_records['after'][j]['eligible']]
    teacher_rough=float(np.sqrt(np.mean(np.diff(z[idx],n=2,axis=1)**2)));a=aggregates['before'];b=aggregates['after']
    checks=dict(migration=True,loss_decreased=b['loss']<=.9*a['loss'],qualified=b['eligible']>=12,qualified_nonregression=b['eligible']>=a['eligible'],old_qualified_preserved=not regressions,median_Q_improved=b['median_Q']<=.9*a['median_Q'],controlled_time_detail=b['roughness_rms_rad']<=1.1*max(a['roughness_rms_rad'],teacher_rough),weights_changed=weight_hash(model)!=initial_hash)
    model.load_state_dict(before['weights']);checks['initial_weight_restorable']=weight_hash(model)==initial_hash
    assert checks==gate['checks'] and gate['status']=='BLOCKED' and not all(checks.values())
    done('diagnostic_32_raw_physics_records',dict(aggregates=aggregates,gate_checks=checks,regressed_indices=regressions,teacher_roughness_rms_rad=teacher_rough,per_case=details))
    update_checks=[]
    for step in (1,256):
        d=diag/'updates'/f'{step:04d}';a=torch.load(d/'before.pt',map_location='cpu',weights_only=False);b=torch.load(d/'after.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(a['indices'],idx) and a['seeds']==seeds
        model.load_state_dict(a['weights']);params=tuple(model.parameters())
        loss=independent_loss(model,codec,c,target,seeds);grads=list(torch.autograd.grad(loss,params))
        norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]));coef=min(1.,1./(float(norm)+1e-6));grads=[g*coef for g in grads]
        gerror=max(delta(g,v) for g,v in zip(grads,a['gradients']));assert gerror<1e-7
        group=a['optimizer']['param_groups'][0];assert group['lr']==1e-4 and group['weight_decay']==.01
        b1,b2=group['betas'];we=[];me=[]
        for (name,param),pid,g in zip(model.named_parameters(),group['params'],grads):
            prev=a['optimizer']['state'].get(pid,{});n=int(prev.get('step',0))+1;assert n==step
            m=prev.get('exp_avg',torch.zeros_like(param)).clone();v=prev.get('exp_avg_sq',torch.zeros_like(param)).clone()
            m.lerp_(g,1-b1);v.mul_(b2).addcmul_(g,g,value=1-b2)
            pred=param.detach().clone().mul_(1-group['lr']*group['weight_decay']);denom=v.sqrt()/(1-b2**n)**.5;denom.add_(group['eps']);pred.addcdiv_(m,denom,value=-group['lr']/(1-b1**n))
            actual=b['optimizer']['state'][pid];assert int(actual['step'])==step
            we.append(delta(pred,b['weights'][name]));me.extend([delta(m,actual['exp_avg']),delta(v,actual['exp_avg_sq'])])
        assert max(we)<2e-7 and max(me)<2e-7
        if step==1:assert all(torch.equal(v,before['weights'][k]) for k,v in a['weights'].items())
        if step==256:assert all(torch.equal(v,after['weights'][k]) for k,v in b['weights'].items())
        update_checks.append(dict(step=step,independent_loss=float(loss.detach()),gradient_max_abs_error=gerror,manual_AdamW_weight_max_abs_error=max(we),manual_AdamW_moment_max_abs_error=max(me),full16_batch_size=16))
    assert {int(v['step']) for v in after['optimizer']['state'].values()}=={256}
    assert [v['step'] for v in gate['history']]==list(range(32,257,32))
    done('first_last_full16_VJP_manual_AdamW',update_checks)
    executions=[]
    for p in sorted(OUT.rglob('execution/report.json')):
        d=p.parent;rr=read_json(p);task=read_json(d/'frozen_task.json');rr,n=audit_physics(d,task,rr['metadata']['reference_hash'])
        assert rr['level']=='EXECUTION_VERIFIED' and not rr['replay']['copies_actual_state_each_step']
        with np.load(d/'native_trace.npz') as a,np.load(d/'replay_trace.npz') as b:
            assert np.array_equal(a['ctrl'],b['ctrl']) and a['ctrl'].shape[1]==67
            assert all(np.isfinite(a[k]).all() for k in a.files)
            for k in ('qpos','qvel','act','time','qacc_warmstart'):assert float(np.max(abs(a[k]-b[k]),initial=0))==rr['replay']['maximum_errors'][k]<=1e-10
        with np.load(d/'geometry_scans.npz') as a:assert float(a['minima'].min())==rr['minimum_clearance_m']>=0
        executions.append(dict(id=d.parent.name,steps=n,min_clearance_m=rr['minimum_clearance_m'],hold_position_m=max(rr['hold_max_position_error_m']),hold_rotation_rad=max(rr['hold_max_rotation_error_rad'])))
    assert len(executions)==3 and {r['id'] for r in executions}==set(sel['reconstruction_execution_ids'])
    done('teacher_native_replay',dict(count=3,native_plus_replay_steps=sum(r['steps'] for r in executions),records=executions,network_generated_executions=0))
    ledger=read_json(OUT/'ledger.json');budget=read_json(OUT/'budget.json')
    assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==sum(r['steps'] for r in executions)
    assert ledger['attempts']=={'TRAIN_diagnostic_physics':32,'TRAIN_diagnostic_updates':1,'TRAIN_execution':3,'TRAIN_reconstruction':6}
    assert not (OUT/'learning').exists() and not (OUT/'comparison').exists() and not (OUT/'unguided').exists()
    assert not gate['retained_for_candidate'] and budget['reserved_coupled_upper']==266<=33000
    done('blocked_and_ledger',dict(no_unknown_work=True,attempts=ledger['attempts'],actual_coupled=sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled')),reserved_coupled=266,native_plus_replay_steps=46500,temporary_updates=256,formal_updates=0,M2='BLOCKED',M3='BLOCKED',M4='BLOCKED',new_B_DEV_success_rate=None,new_B_evaluation='not_evaluated'))
    audit=ROOT/'paper/PAPER_CLAIM_AUDIT.json';done('paper_claim_audit',read_json(audit) if audit.exists() else {'status':'unavailable','label':'provisional — no paper claim audit available'})
    for p in OUT.rglob('*'):
        if p.is_file():bind(p)
    bind(ROOT/'ff_prior_control_transfer/docs/EXPERIMENT_PLAN.md');bind(Path(__file__).resolve())
    R['status']='PASS';save()

if __name__=='__main__':
    try:main()
    except Exception:R['status']='FAIL';R['errors'].append(traceback.format_exc());save();raise
