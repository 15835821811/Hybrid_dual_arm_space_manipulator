"""Fit-only trust refinement and one adaptation-held-out TRAIN check."""
import copy
import numpy as np
import torch
from ff_prior.storage import read_json
from ff_prior.diffusion import weight_hash
from ff_prior_backbone_v2.learning import save_torch
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior_rep_fidelity.validation import validate as dense_validate
from ff_prior_control_transfer.diagnostic import metrics
from ff_prior_rollout_world.world_training import exact_rollout,physical_teacher_loss,world_loss,accept
from .model import TransferDenoiser
from .sampling import FullProjector,unguided_sample
from .world_endpoint import jacobian
from .protocol import P


def fixed_pool(selection,role):
    return [(row,seed) for row,seeds in zip(selection[role+'_pool'],selection[role+'_seed_pairs']) for seed in seeds]


def independent_gate(before,after,strata):
    a,b=metrics(before),metrics(after)
    checks=dict(original_qualified_preserved=all(not x['eligible'] or y['eligible'] for x,y in zip(before,after)),
        no_new_model_failure=all(x['validation'].get('validation_level')!='MODEL_VALID' or y['validation'].get('validation_level')=='MODEL_VALID' for x,y in zip(before,after)),
        median_nonregression=b['median_Q']<=P['check_median_ratio']*a['median_Q'],
        tail_nonregression=float(np.percentile(b['Q'],95))<=P['check_p95_ratio']*float(np.percentile(a['Q'],95)),
        raw_unchanged=all(x['raw_equal_prepared'] for x in after),finite=b['missing']==0,
        qualification_nonvacuous=a['eligible']>=P['protected_min_each_pool'])
    per={}
    for name in sorted(set(strata)):
        ids=[i for i,x in enumerate(strata) if x==name]
        old=sum(before[i]['eligible'] for i in ids);new=sum(after[i]['eligible'] for i in ids)
        per[name]=dict(before=old,after=new,total=len(ids));checks['stratum_'+name]=new>=old
    return dict(status='PASS' if all(checks.values()) else 'BLOCKED',checks=checks,before=a,after=b,strata=per)


def evaluate(model,codec,arrays,data,selection,role,dest,cache,profile,ledger,budget,progress):
    records=[];fixed=fixed_pool(selection,role)
    for j,(row,seed) in enumerate(fixed):
        progress('TRAIN_'+role+'_evaluation',phase=dest.name,output=j+1,total=len(fixed))
        rr,ii,cc,_=cache.get(row);i=data['ids'][row['sample_id']];d=dest/f'{j:02d}'
        ledger.begin('_'.join(dest.parts[-3:])+f'_{role}_{j}','TRAIN_'+role+'_validation',P['wall_s'],1000)
        start=rr.calls.copy();budget.reserve(7)
        raw,steps=unguided_sample((model,codec),arrays[1][i],row['task'],seed)
        z,prep=FullProjector(row['task'],rr,ii,cc,profile,budget.deadline,codec=codec,base=arrays[1][i]).prepare(raw)
        atomic_npz(d/'controls.npz',raw=raw,prepared=z,teacher=arrays[0][i])
        r=dense_validate(d,z,row['task'],rr,ii,cc,profile,budget)
        r.update(raw_equal_prepared=bool(np.array_equal(raw,z)),stratum=row['frozen_stratum'],sample_id=row['sample_id'],seed=seed)
        atomic_json(d/'identity.json',dict(sample_id=row['sample_id'],seed=seed,task_hash=row['task']['task_hash'],preparation=prep,network_calls=16,weight_hash=weight_hash(model)))
        records.append(r);ledger.finish({'status':'DONE'},{k:rr.calls[k]-start[k] for k in start})
    atomic_json(dest/'records.json',records)
    return records


def refine(model,dest,selection,codec,arrays,data,cache,profile,ledger,budget,progress,baseline_fit):
    if dest.exists():raise ValueError('PARTIAL_REFINEMENT_NOT_REPLAYABLE')
    dest.mkdir(parents=True);low_hash=model.low_hash()
    fixed=fixed_pool(selection,'fit');indices=np.array([data['ids'][r['sample_id']] for r,s in fixed]);seeds=[s for r,s in fixed]
    c=data['c'][indices];target=data['x'][indices]
    if dest.name=='TRAIN_fit':
        current=copy.deepcopy(baseline_fit)
        atomic_json(dest/'before_reused.json',dict(source='../TRAIN_baseline/fit/records.json',same_model_no_update=True,new_physics=False))
    else:current=evaluate(model,codec,arrays,data,selection,'fit',dest/'before',cache,profile,ledger,budget,progress)
    before=copy.deepcopy(current);original=[bool(x['eligible']) for x in baseline_fit]
    # Frozen old functionality supplies a TRAIN-only replay target, not the independent check pool.
    with torch.no_grad():
        reference=TransferDenoiser(codec);old_x=exact_rollout(reference,c,seeds).detach()
    initial=weight_hash(model);save_torch(dest/'initial.pt',dict(weights=model.state_dict(),indices=indices,seeds=seeds,old_x=old_x,original_protected=original))
    teacher_rough=float(np.sqrt(np.mean(np.diff(arrays[0][indices],n=2,axis=1)**2)))
    rough_limit=P['learnability_roughness_ratio']*max(metrics(baseline_fit)['roughness_rms_rad'],teacher_rough)
    rounds=[];gradient_count=0;updates=0;numeric_ok=True;low_errors=[]
    for number in range(P['world_rounds']):
        d=dest/'rounds'/f'{number:02d}';state=copy.deepcopy(model.state_dict());protected=[v or bool(r['eligible']) for v,r in zip(original,current)]
        with torch.no_grad():x0=exact_rollout(model,c,seeds).detach()
        low_error=float(abs(x0[:,:102]-old_x[:,:102]).max());assert low_error==0;low_errors.append(low_error)
        residuals=[];js=[];passed=[]
        for j,(row,seed) in enumerate(fixed):
            progress('world_'+dest.name,round=number+1,gradient=j+1,total=32)
            rr,ii,cc,_=cache.get(row);i=indices[j];ledger.begin(dest.name+f'_J_{number}_{j}','fit_full323_J',P['wall_s'],1000)
            start=rr.calls.copy();budget.reserve(P['world_queries_per_jacobian'])
            r,J,proof,arr=jacobian(x0[j].numpy(),codec,data['centers'][i],arrays[1][i],row['task'],rr,ii,budget.deadline)
            atomic_npz(d/'jacobians'/f'{j:02d}.npz',**arr);atomic_json(d/'jacobians'/f'{j:02d}.json',proof)
            ledger.finish({'status':proof['status']},{k:rr.calls[k]-start[k] for k in start})
            residuals.append(r);js.append(J);passed.append(proof['status']=='PASS');gradient_count+=1
        save_torch(d/'before.pt',dict(weights=state,indices=indices,seeds=seeds,x0=x0,protected=protected,old_x=old_x))
        if not all(passed):
            numeric_ok=False;rounds.append(dict(round=number,status='GRADIENT_BLOCKED',accepted=False,gradient_pass=sum(passed),gradient_total=32));break
        R=torch.tensor(np.stack(residuals),dtype=torch.float64);J=torch.tensor(np.stack(js),dtype=torch.float64)
        optimizer=torch.optim.AdamW(model.trainable(),lr=P['world_lr'],weight_decay=.01)
        ledger.begin(dest.name+f'_updates_{number}','fit_only_world_proposal',P['wall_s'],64);history=[]
        for step in range(64):
            budget.check();optimizer.zero_grad();full=exact_rollout(model,c,seeds)
            w=world_loss(full,x0,R,J,protected)
            mask=torch.tensor(original,dtype=torch.bool)
            replay=(full[mask]-old_x[mask]).square().mean() if mask.any() else full.sum()*0
            t=physical_teacher_loss(full,codec,target)
            loss=w+P['world_replay_weight']*replay+P['world_teacher_weight']*t
            if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.trainable(),1.)
            assert all(p.grad is None for p in model.low.parameters())
            if not torch.isfinite(norm):raise ValueError('NONFINITE_GRADIENT')
            if step in (0,63):save_torch(d/'updates'/f'{step+1:03d}'/'before.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),gradients=[p.grad.clone() for p in model.trainable()]))
            optimizer.step();updates+=1
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('NONFINITE_WEIGHTS')
            assert model.low_hash()==low_hash
            if step in (0,63):save_torch(d/'updates'/f'{step+1:03d}'/'after.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict()))
            if (step+1)%16==0:history.append(dict(step=step+1,world=float(w.detach()),replay=float(replay.detach()),teacher=float(t.detach()),clip_norm=float(norm)))
        proposed=copy.deepcopy(model.state_dict());save_torch(d/'proposed.pt',dict(weights=proposed,optimizer=optimizer.state_dict(),history=history));ledger.finish({'status':'DONE','updates':64},{})
        trials=[];accepted=False
        for k,scale in enumerate(P['world_trial_scales']):
            # Never interpolate frozen parameters: even x + 0 may lose bitwise identity.
            model.load_state_dict({n:state[n]+scale*(proposed[n]-state[n]) if n.startswith('detail.') else state[n] for n in state})
            after=evaluate(model,codec,arrays,data,selection,'fit',d/f'trial_{k}',cache,profile,ledger,budget,progress)
            ok,checks=accept(current,after,protected,rough_limit)
            checks['raw_unchanged_by_prepare']=all(x['raw_equal_prepared'] for x in after)
            checks['frozen_low']=model.low_hash()==low_hash
            ok=all(checks.values());trials.append(dict(scale=scale,accepted=ok,checks=checks,metrics=metrics(after)))
            if ok:current=after;accepted=True;break
        if not accepted:model.load_state_dict(state)
        save_torch(d/'retained.pt',dict(weights=model.state_dict(),optimizer_policy='fresh per round; discard proposal moments'))
        result=dict(round=number,status='ACCEPTED' if accepted else 'NO_ACCEPTED_PROPOSAL',accepted=accepted,gradient_pass=sum(passed),gradient_total=32,trials=trials,history=history)
        atomic_json(d/'result.json',result);rounds.append(result)
        if not accepted:break
    with torch.no_grad():final_x=exact_rollout(model,c,seeds);assert torch.equal(final_x[:,:102],old_x[:,:102])
    save_torch(dest/'final.pt',dict(weights=model.state_dict(),indices=indices,seeds=seeds))
    a,b=metrics(baseline_fit),metrics(current)
    checks=dict(numerical_gradients=numeric_ok and gradient_count>0,frozen_low=model.low_hash()==low_hash,
        qualified=b['eligible']>=max(P['qualified_fit_min'],a['eligible']),
        protected=all(not keep or r['eligible'] for keep,r in zip(original,current)),
        qualified_nonvacuous=sum(original)>=P['protected_min_each_pool'],
        improved=b['median_Q']<=.9*a['median_Q'],accepted=any(r['accepted'] for r in rounds),
        controlled_detail=b['roughness_rms_rad']<=rough_limit,raw_unchanged=all(x['raw_equal_prepared'] for x in current))
    result=dict(status='PASS' if all(checks.values()) else 'BLOCKED',checks=checks,initial_hash=initial,final_hash=weight_hash(model),
        low_hash=low_hash,before=metrics(before),baseline=metrics(baseline_fit),after=b,initial_records=before,final_records=current,
        gradient_count=gradient_count,actual_proposed_updates=updates,rounds=rounds,low_max_errors=low_errors,
        original_protected=original,rough_limit=rough_limit,check_pool_used_for_update_or_scale=False)
    atomic_json(dest/'gate.json',result);return result


def diagnostic(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress):
    if (out/'TRAIN_gate.json').exists():return read_json(out/'TRAIN_gate.json')
    if (out/'TRAIN_baseline').exists():raise ValueError('PARTIAL_DIAGNOSTIC_NOT_REPLAYABLE')
    torch.manual_seed(P['training_seed']);model=TransferDenoiser(codec)
    fit=evaluate(model,codec,arrays,data,selection,'fit',out/'TRAIN_baseline/fit',cache,profile,ledger,budget,progress)
    check=evaluate(model,codec,arrays,data,selection,'check',out/'TRAIN_baseline/check',cache,profile,ledger,budget,progress)
    coverage=dict(fit_qualified=sum(x['eligible'] for x in fit),check_qualified=sum(x['eligible'] for x in check))
    if min(coverage.values())<P['protected_min_each_pool']:
        result=dict(status='BLOCKED',reason='INSUFFICIENT_INITIAL_QUALIFIED_PROTECTION',coverage=coverage)
    else:
        result=refine(model,out/'TRAIN_fit',selection,codec,arrays,data,cache,profile,ledger,budget,progress,fit)
        after=evaluate(model,codec,arrays,data,selection,'check',out/'TRAIN_check_final',cache,profile,ledger,budget,progress)
        chk=independent_gate(check,after,[r['stratum'] for r in check]);atomic_json(out/'TRAIN_check_final/gate.json',chk)
        result=dict(status='PASS' if result['status']=='PASS' and chk['status']=='PASS' else 'BLOCKED',fit=result,check=chk,coverage=coverage)
    atomic_json(out/'TRAIN_gate.json',result);return result


def refine_candidate(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress):
    from .learning import load
    from ff_prior.storage import file_digest
    if (out/'learning/full/training.json').exists():load(out/'learning/full');return
    model,_=load(out/'learning/ordinary');fit=read_json(out/'TRAIN_baseline/fit/records.json')
    result=refine(model,out/'candidate_fit',selection,codec,arrays,data,cache,profile,ledger,budget,progress,fit)
    if not result['checks']['numerical_gradients']:raise ValueError('FORMAL_GRADIENT_BLOCKED')
    after=evaluate(model,codec,arrays,data,selection,'check',out/'candidate_check_final',cache,profile,ledger,budget,progress)
    before=read_json(out/'TRAIN_baseline/check/records.json')
    chk=independent_gate(before,after,[r['stratum'] for r in before]);atomic_json(out/'candidate_check_final/gate.json',chk)
    # No selection or retry based on independent check; the one fixed candidate is evaluated as-is.
    d=out/'learning/full';ordinary=read_json(out/'learning/ordinary/training.json')
    save_torch(d/'model.pt',dict(weights=model.state_dict(),metadata=ordinary['metadata']))
    atomic_json(d/'codec.json',codec.state)
    ordinary.update(trained_weight_hash=weight_hash(model),checkpoint_sha256=file_digest(d/'model.pt'),world_refinement=result,
        independent_check=chk,world_proposed_updates=result['actual_proposed_updates'],candidate_count=1)
    atomic_json(d/'training.json',ordinary);load(d)
