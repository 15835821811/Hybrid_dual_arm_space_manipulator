"""Frozen TRAIN-only temporary overfit and raw functional migration checks."""
import numpy as np
import torch
from ff_prior.storage import read_json
from ff_prior.diffusion import weight_hash
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior_backbone_v2.learning import save_torch
from ff_prior_sample_alignment.learning import load as old_load
from ff_prior_endpoint_jacobian.comparison import unguided_sample as old_sample
from ff_prior_rep_fidelity.validation import validate as dense_validate
from .model import TransferDenoiser,teacher_loss
from .sampling import unguided_sample,FullProjector
from .protocol import SOURCE,P


def metrics(records):
    q=[];rough=[]
    for r in records:
        v=r['validation'];q.append(max(max(v['frozen_goal_position_error_m'])/.005,max(v['frozen_goal_rotation_error_rad'])/np.deg2rad(1.)) if 'frozen_goal_position_error_m' in v else None)
        rough.append(float(np.mean(np.diff(np.asarray(r['Z']),n=2,axis=0)**2)))
    return dict(eligible=sum(r['eligible'] for r in records),Q=q,missing=sum(v is None for v in q),median_Q=float(np.median(q)) if all(v is not None for v in q) else None,
        roughness_rms_rad=float(np.sqrt(np.mean(rough))))


def run(out,selection,codec,arrays,rows,data,cache,profile,ledger,budget,progress):
    dest=out/'TRAIN_diagnostic'
    if (dest/'gate.json').exists():return read_json(dest/'gate.json')
    # A partial diagnostic is unknown work. It cannot silently be retrained.
    if dest.exists():raise ValueError('PARTIAL_TRAIN_DIAGNOSTIC_NOT_REPLAYABLE')
    dest.mkdir();torch.manual_seed(P['training_seed']);model=TransferDenoiser(codec);initial=weight_hash(model)
    old=old_load(SOURCE/'learning/aligned')
    fixed=[(r,seed+j*997) for j,r in enumerate(selection['diagnostic']) for seed in P['diagnostic_seeds']]
    indices=np.array([data['ids'][r['sample_id']] for r,seed in fixed]);seeds=[seed for r,seed in fixed]
    migration=[]
    for j,(r,seed) in enumerate(fixed):
        i=indices[j];z,steps=unguided_sample((model,codec),arrays[1][i],r['task'],seed)
        oz,_=old_sample(old,arrays[1][i],r['task'],seed)
        error=float(abs(z-oz).max());migration.append(error)
        atomic_npz(dest/'migration'/f'{j:02d}.npz',old=oz,new=z)
    migrated=max(migration)<=P['migration_error_rad']
    atomic_json(dest/'migration.json',dict(status='PASS' if migrated else 'FAIL',max_control_error_rad=max(migration),errors=migration,
        scope='Raw unguided full16 only; guided prepare and full method equivalence not asserted',low_weight_preserved=True))
    if not migrated:
        gate=dict(status='BLOCKED',reason='FUNCTIONAL_MIGRATION_FAILED',temporary_updates=0)
        atomic_json(dest/'gate.json',gate);return gate
    def evaluate(stage):
        records=[]
        for j,(row,seed) in enumerate(fixed):
            rr,ii,cc,_=cache.get(row);i=indices[j];directory=dest/stage/f'{j:02d}'
            key=f'diagnostic_{stage}_{j}';ledger.begin(key,'TRAIN_diagnostic_physics',P['wall_s'],1000);before=rr.calls.copy()
            budget.reserve(7);progress('M1_TRAIN_full16_'+stage,case=j+1,total=16)
            raw,steps=unguided_sample((model,codec),arrays[1][i],row['task'],seed)
            z,prep=FullProjector(row['task'],rr,ii,cc,profile,budget.deadline,codec=codec,base=arrays[1][i]).prepare(raw)
            atomic_npz(directory/'controls.npz',raw=raw,prepared=z,teacher=arrays[0][i])
            dense_validate(directory,z,row['task'],rr,ii,cc,profile,budget)
            atomic_json(directory/'identity.json',dict(sample_id=row['sample_id'],seed=seed,task_hash=row['task']['task_hash'],preparation=prep,network_calls=16,weight_hash=weight_hash(model)))
            records.append(read_json(directory/'validation.json'))
            ledger.finish({'status':'DONE'},{k:rr.calls[k]-before[k] for k in before})
        return records
    before_records=evaluate('before')
    c=data['c'][indices];target=data['x'][indices]
    optimizer=torch.optim.AdamW(model.parameters(),lr=P['diagnostic_lr'],weight_decay=P['weight_decay'])
    save_torch(dest/'before.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),indices=indices,seeds=seeds))
    with torch.no_grad():before_loss=float(teacher_loss(model,codec,c,target,seeds))
    ledger.begin('TRAIN_diagnostic_updates','TRAIN_diagnostic_updates',P['wall_s'],256)
    history=[]
    for step in range(P['diagnostic_updates']):
        budget.check();optimizer.zero_grad();loss=teacher_loss(model,codec,c,target,seeds)
        if not torch.isfinite(loss):raise ValueError('nonfinite diagnostic loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite diagnostic gradient')
        if step in (0,P['diagnostic_updates']-1):
            save_torch(dest/'updates'/f'{step+1:04d}'/'before.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),indices=indices,seeds=seeds,
                gradients=[p.grad.detach().clone() for p in model.parameters()]))
        optimizer.step()
        if step in (0,P['diagnostic_updates']-1):save_torch(dest/'updates'/f'{step+1:04d}'/'after.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict()))
        if (step+1)%32==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),clip_norm=float(norm)));progress('M1_TRAIN_learnability',update=step+1,total=P['diagnostic_updates'])
    save_torch(dest/'after.pt',dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),indices=indices,seeds=seeds))
    with torch.no_grad():after_loss=float(teacher_loss(model,codec,c,target,seeds))
    ledger.finish({'status':'DONE','updates':P['diagnostic_updates']},{})
    after_records=evaluate('after');a,b=metrics(before_records),metrics(after_records)
    teacher_rough=float(np.sqrt(np.mean(np.diff(arrays[0][indices],n=2,axis=1)**2)))
    checks=dict(migration=migrated,loss_decreased=after_loss<=P['learnability_loss_ratio']*before_loss,
        qualified=b['eligible']>=P['learnability_qualified_min'],qualified_nonregression=b['eligible']>=a['eligible'],
        old_qualified_preserved=all(not x['eligible'] or y['eligible'] for x,y in zip(before_records,after_records)),
        median_Q_improved=a['median_Q'] is not None and b['median_Q'] is not None and b['median_Q']<=P['learnability_q_ratio']*a['median_Q'],
        controlled_time_detail=b['roughness_rms_rad']<=P['learnability_roughness_ratio']*max(a['roughness_rms_rad'],teacher_rough),
        weights_changed=weight_hash(model)!=initial)
    reset=TransferDenoiser(codec);saved=torch.load(dest/'before.pt',map_location='cpu',weights_only=False);reset.load_state_dict(saved['weights'])
    checks['initial_weight_restorable']=weight_hash(reset)==initial
    gate=dict(status='PASS' if all(checks.values()) else 'BLOCKED',checks=checks,before=a,after=b,
        before_loss=before_loss,after_loss=after_loss,teacher_roughness_rms_rad=teacher_rough,history=history,
        temporary_updates=P['diagnostic_updates'],retained_for_candidate=False,initial_hash=initial,temporary_final_hash=weight_hash(model))
    atomic_json(dest/'gate.json',gate);return gate
