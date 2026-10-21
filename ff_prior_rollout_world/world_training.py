"""Bounded refreshed world supervision with saved real-physics acceptance.

64-step local surrogate proposals are temporary. Only an explicitly accepted
parameter interpolation is retained. AdamW starts fresh per trust round;
discarded or scaled trial moments are never used for later rounds.
"""
import copy
import numpy as np
import torch
from ff_prior.storage import read_json, digest, file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_backbone_v2.learning import save_torch
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from ff_prior_rep_fidelity.validation import validate as dense_validate
from ff_prior_control_transfer.diagnostic import metrics
from ff_prior_control_transfer.model import TransferDenoiser,rollout,teacher_loss
from ff_prior_control_transfer.sampling import unguided_sample,FullProjector
from .world_endpoint import jacobian
from .protocol import P,REUSE


def exact_rollout(model,c,seeds):
    """Match deployment's batch-one float32 kernels before aggregating loss."""
    return torch.cat([rollout(model,c[j:j+1],[seed]) for j,seed in enumerate(seeds)],dim=0)


def world_loss(full,x0,residual,J,protected):
    # The current physical residual is exact at x0. Elsewhere this is a local
    # surrogate, checked by actual propagation before any proposal is retained.
    estimated=residual+torch.einsum('nij,nj->ni',J,full-x0)
    q=estimated.reshape(-1,4,3).square().sum(-1)
    world=q.mean()+P['world_violation_weight']*torch.relu(q-1).square().mean()
    mask=torch.tensor(protected,dtype=torch.bool)
    guard=torch.relu(q[mask]-P['world_protection_margin']**2).square().mean() if mask.any() else world*0
    trust=(full-x0).square().mean()
    return world+P['world_protection_weight']*guard+P['world_trust_weight']*trust


def physical_teacher_loss(full,codec,target):
    scale=torch.tensor(codec.latent_scale,dtype=torch.float64)
    basis=torch.tensor(codec.basis,dtype=torch.float64)
    physical=torch.einsum('tk,nkj->ntj',basis,((full-target.double())*scale).reshape(-1,19,17))
    return (physical.square().mean()+P['temporal_error_weight']*torch.diff(physical,n=2,dim=1).square().mean())/scale.square().mean()


def accept(before,after,protected,rough_limit):
    a,b=metrics(before),metrics(after)
    checks=dict(finite=b['missing']==0 and all(np.isfinite(b['Q'])),
        old_qualified_preserved=all(not keep or r['eligible'] for keep,r in zip(protected,after)),
        no_new_model_failure=all(x['validation'].get('validation_level')!='MODEL_VALID' or y['validation'].get('validation_level')=='MODEL_VALID' for x,y in zip(before,after)),
        controlled_time_detail=b['roughness_rms_rad']<=rough_limit,
        mean_Q_improved=a['missing']==0 and b['missing']==0 and float(np.mean(b['Q']))<=P['world_accept_mean_ratio']*float(np.mean(a['Q'])),
        median_Q_nonregression=a['median_Q'] is not None and b['median_Q'] is not None and b['median_Q']<=a['median_Q'])
    return all(checks.values()),checks


def refine(model,dest,selection,codec,arrays,data,cache,profile,ledger,budget,progress,reuse_initial=False):
    if dest.exists():raise ValueError('PARTIAL_WORLD_REFINEMENT_NOT_REPLAYABLE')
    dest.mkdir(parents=True);initial=weight_hash(model)
    fixed=[(r,s+j*997) for j,r in enumerate(selection['diagnostic']) for s in P['diagnostic_seeds']]
    indices=np.array([data['ids'][r['sample_id']] for r,s in fixed]);seeds=[s for r,s in fixed]
    c=data['c'][indices];target=data['x'][indices]
    save_torch(dest/'initial.pt',dict(weights=model.state_dict(),indices=indices,seeds=seeds))

    def evaluate(phase):
        records=[]
        for j,(row,seed) in enumerate(fixed):
            i=indices[j];rr,ii,cc,_=cache.get(row);directory=dest/phase/f'{j:02d}'
            ledger.begin(dest.name+'_'+phase.replace('/','_')+f'_{j}','world_TRAIN_validation',P['wall_s'],1000)
            start=rr.calls.copy();budget.reserve(7)
            raw,steps=unguided_sample((model,codec),arrays[1][i],row['task'],seed)
            z,prep=FullProjector(row['task'],rr,ii,cc,profile,budget.deadline,codec=codec,base=arrays[1][i]).prepare(raw)
            atomic_npz(directory/'controls.npz',raw=raw,prepared=z,teacher=arrays[0][i])
            r=dense_validate(directory,z,row['task'],rr,ii,cc,profile,budget)
            atomic_json(directory/'identity.json',dict(sample_id=row['sample_id'],seed=seed,task_hash=row['task']['task_hash'],
                        preparation=prep,network_calls=16,weight_hash=weight_hash(model)))
            # World derivatives are on raw full output; do not attribute a
            # successful changed preparation to the supervised raw network.
            r=dict(r,raw_equal_prepared=bool(np.array_equal(raw,z)))
            records.append(r)
            ledger.finish({'status':'DONE'},{k:rr.calls[k]-start[k] for k in start})
        return records

    with torch.no_grad():initial_x=exact_rollout(model,c,seeds).numpy()
    if reuse_initial:
        old=torch.load(REUSE/'TRAIN_diagnostic/before.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,old['weights'][k]) for k,v in model.state_dict().items())
        before=[]
        for j,(row,seed) in enumerate(fixed):
            raw=codec.decode(codec.coefficients(initial_x[j],data['centers'][indices[j]]),arrays[1][indices[j]])
            with np.load(REUSE/'TRAIN_diagnostic/before'/f'{j:02d}'/'controls.npz') as saved:
                assert np.array_equal(raw,saved['raw']) and np.array_equal(raw,saved['prepared'])
            before.append(dict(read_json(REUSE/'TRAIN_diagnostic/before'/f'{j:02d}'/'validation.json'),raw_equal_prepared=True))
        atomic_json(dest/'before_reuse.json',dict(source=str(REUSE),count=16,initial_hash=initial,not_new_physics=True))
    else:before=evaluate('before')
    current=before
    atomic_json(dest/'initial_records.json',before)
    teacher_rough=float(np.sqrt(np.mean(np.diff(arrays[0][indices],n=2,axis=1)**2)))
    rough_limit=P['learnability_roughness_ratio']*max(metrics(before)['roughness_rms_rad'],teacher_rough)
    original=[bool(r['eligible']) or j in selection['original_protected'] for j,r in enumerate(before)]
    with torch.no_grad():before_loss=float(physical_teacher_loss(torch.tensor(initial_x),codec,target))
    rounds=[];updates=0;gradient_count=0;gradient_failed=False
    for number in range(P['world_rounds']):
        budget.check();directory=dest/'rounds'/f'{number:02d}'
        state=copy.deepcopy(model.state_dict());protected=[a or bool(r['eligible']) for a,r in zip(original,current)]
        with torch.no_grad():x0=exact_rollout(model,c,seeds).detach()
        residuals=[];jacobians=[];checks=[]
        for j,(row,seed) in enumerate(fixed):
            progress('world_'+dest.name,round=number+1,gradient=j+1,total=16)
            rr,ii,cc,_=cache.get(row);i=indices[j]
            ledger.begin(dest.name+f'_J_{number}_{j}','current_full323_world_J',P['wall_s'],1000)
            start=rr.calls.copy();budget.reserve(P['world_queries_per_jacobian'])
            r,J,proof,arrays_proof=jacobian(x0[j].numpy(),codec,data['centers'][i],arrays[1][i],row['task'],rr,ii,budget.deadline)
            atomic_npz(directory/'jacobians'/f'{j:02d}.npz',**arrays_proof)
            atomic_json(directory/'jacobians'/f'{j:02d}.json',proof)
            ledger.finish({'status':proof['status']},{k:rr.calls[k]-start[k] for k in start})
            residuals.append(r);jacobians.append(J);checks.append(proof['status']=='PASS');gradient_count+=1
        save_torch(directory/'before.pt',dict(weights=state,indices=indices,seeds=seeds,x0=x0,protected=protected))
        if not all(checks):
            gradient_failed=True
            rounds.append(dict(round=number,accepted=False,status='GRADIENT_BLOCKED',gradient_pass=sum(checks),gradient_total=16))
            break
        R=torch.tensor(np.stack(residuals),dtype=torch.float64);J=torch.tensor(np.stack(jacobians),dtype=torch.float64)
        opt=torch.optim.AdamW(model.parameters(),lr=P['world_lr'],weight_decay=P['weight_decay'])
        ledger.begin(dest.name+f'_updates_{number}','world_local_surrogate_updates',P['wall_s'],P['world_inner_updates'])
        history=[]
        for step in range(P['world_inner_updates']):
            budget.check();opt.zero_grad();full=exact_rollout(model,c,seeds)
            w=world_loss(full,x0,R,J,protected);t=physical_teacher_loss(full,codec,target)
            loss=w+P['world_teacher_weight']*t
            if not torch.isfinite(loss):raise ValueError('NONFINITE_WORLD_TRAIN_LOSS')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('NONFINITE_WORLD_TRAIN_GRADIENT')
            if step in (0,P['world_inner_updates']-1):
                save_torch(directory/'updates'/f'{step+1:03d}'/'before.pt',dict(weights=model.state_dict(),optimizer=opt.state_dict(),
                    gradients=[p.grad.detach().clone() for p in model.parameters()]))
            opt.step();updates+=1
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('NONFINITE_WORLD_WEIGHTS')
            if step in (0,P['world_inner_updates']-1):save_torch(directory/'updates'/f'{step+1:03d}'/'after.pt',dict(weights=model.state_dict(),optimizer=opt.state_dict()))
            if (step+1)%16==0:history.append(dict(step=step+1,world=float(w.detach()),teacher=float(t.detach()),clip_norm=float(norm)))
        proposed=copy.deepcopy(model.state_dict())
        save_torch(directory/'proposed.pt',dict(weights=proposed,optimizer=opt.state_dict(),history=history))
        ledger.finish({'status':'DONE','updates':64},{})
        trials=[];accepted=False
        for trial,scale in enumerate(P['world_trial_scales']):
            trial_state={k:(state[k]+scale*(proposed[k]-state[k])) if state[k].is_floating_point() else state[k] for k in state}
            model.load_state_dict(trial_state)
            after=evaluate(f'rounds/{number:02d}/trial_{trial}')
            ok,reasons=accept(current,after,protected,rough_limit)
            reasons['raw_unchanged_by_prepare']=all(r['raw_equal_prepared'] for r in after)
            ok=ok and reasons['raw_unchanged_by_prepare']
            trials.append(dict(scale=scale,accepted=ok,checks=reasons,metrics=metrics(after)))
            if ok:current=after;accepted=True;break
        if not accepted:model.load_state_dict(state)
        save_torch(directory/'retained.pt',dict(weights=model.state_dict(),optimizer_policy='fresh each round; trial moments discarded'))
        result=dict(round=number,accepted=accepted,status='ACCEPTED' if accepted else 'NO_ACCEPTED_PROPOSAL',
            trials=trials,protected=protected,gradient_pass=sum(checks),gradient_total=16,history=history)
        atomic_json(directory/'result.json',result);rounds.append(result)
        # No unchanged-state repeat of identical gradients/optimization.
        if not accepted:break
    with torch.no_grad():after_loss=float(physical_teacher_loss(exact_rollout(model,c,seeds),codec,target))
    save_torch(dest/'final.pt',dict(weights=model.state_dict(),indices=indices,seeds=seeds))
    a,b=metrics(before),metrics(current)
    criteria=dict(numerical_gradients=not gradient_failed and gradient_count>0,
        qualified=b['eligible']>=P['learnability_qualified_min'],
        old_qualified_preserved=all(not keep or r['eligible'] for keep,r in zip(original,current)),
        median_Q_improved=a['median_Q'] is not None and b['median_Q'] is not None and b['median_Q']<=P['learnability_q_ratio']*a['median_Q'],
        controlled_time_detail=b['roughness_rms_rad']<=rough_limit,
        accepted_world_updates=any(r['accepted'] for r in rounds),finite=all(torch.isfinite(p).all() for p in model.parameters()))
    result=dict(status='PASS' if all(criteria.values()) else 'BLOCKED',checks=criteria,before=a,after=b,
        rounds=rounds,actual_proposed_updates=updates,gradient_count=gradient_count,
        original_protected=original,rough_limit=rough_limit,teacher_roughness_rms_rad=teacher_rough,
        before_loss=before_loss,after_loss=after_loss,initial_hash=initial,final_hash=weight_hash(model),
        final_records=current,initial_records=before,retained_for_candidate=False,
        exact_world_values_at_refresh=True,inner_loss_is_local_linear_surrogate=True)
    atomic_json(dest/'gate.json',result)
    return result


def diagnostic(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress):
    dest=out/'TRAIN_world'
    if (dest/'gate.json').exists():return read_json(dest/'gate.json')
    torch.manual_seed(P['training_seed']);model=TransferDenoiser(codec)
    return refine(model,dest,selection,codec,arrays,data,cache,profile,ledger,budget,progress,True)


def refine_candidate(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress):
    from .learning import load
    final=out/'learning/full'
    if (final/'training.json').exists():load(final);return
    model,_=load(out/'learning/ordinary')
    result=refine(model,out/'candidate_world',selection,codec,arrays,data,cache,profile,ledger,budget,progress)
    # Invalid numerical derivative is a stopped candidate, never fallback training.
    if not result['checks']['numerical_gradients']:raise ValueError('FORMAL_WORLD_GRADIENT_BLOCKED')
    final.mkdir(parents=True,exist_ok=True)
    ordinary=read_json(out/'learning/ordinary/training.json')
    save_torch(final/'model.pt',dict(weights=model.state_dict(),metadata=ordinary['metadata']))
    atomic_json(final/'codec.json',codec.state)
    ordinary.update(ordinary_checkpoint=str(out/'learning/ordinary/model.pt'),
        trained_weight_hash=weight_hash(model),checkpoint_sha256=file_digest(final/'model.pt'),
        world_refinement=result,world_proposed_updates=result['actual_proposed_updates'],
        candidate_count=1,VAL_gradient_updates=0)
    atomic_json(final/'training.json',ordinary);load(final)
