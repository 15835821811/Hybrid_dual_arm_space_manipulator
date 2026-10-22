"""One candidate and one sequence comparator; full-sampler TRAIN supervision."""
import copy
import hashlib
import time
import numpy as np
import torch
from ff_prior.diffusion import schedule,weight_hash,save_torch
from ff_prior_pilot_train.storage import atomic_json
from .models import SequenceNetwork,sample
from .protocol import P

def objective(prediction,target):
    delta=prediction-target
    return (delta.square().mean()+P['detail_weight']*torch.diff(delta,dim=1).square().mean())

def tensor_data(arrays):
    return {k:torch.tensor(arrays[k],dtype=torch.float32) for k in ['x','global_condition','local_condition']}

def full_probe(model,data,method):
    seeds=[20264200+i for i in range(len(data['x']))]
    with torch.no_grad():
        x,records=sample(model,data['global_condition'],data['local_condition'],seeds,method)
        loss=objective(x,data['x']).item()
    return dict(loss=loss,finite=bool(torch.isfinite(x).all()),seeds=seeds,
                maximum_normalized_error=float(abs(x-data['x']).max()),ddim=records)

def base_loss(model,data,method,step):
    order=np.random.default_rng(P['seed']+step).permutation(8)
    idx=torch.tensor(order);target=data['x'][idx];g=data['global_condition'][idx];local=data['local_condition'][idx]
    rng=torch.Generator().manual_seed(P['seed']+100000+step)
    t=torch.randint(0,1000,(8,),generator=rng);noise=torch.randn(target.shape,generator=rng)
    if method=='deterministic_sequence':
        pred,_=sample(model,g,local,[0]*8,method);loss=objective(pred,target)
    else:
        a=schedule()[t,None,None];xt=a.sqrt()*target+(1-a).sqrt()*noise
        vtarget=a.sqrt()*noise-(1-a).sqrt()*target
        loss=(model(xt,t.float(),g,local)-vtarget).square().mean()
    h=hashlib.sha256()
    for value in (order,t.numpy(),noise.numpy()):h.update(np.asarray(value).tobytes())
    return loss,dict(ids=order.tolist(),noise_seed=P['seed']+100000+step,batch_noise_sha256=h.hexdigest())

def finite_gradient(model):
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    return bool(gradients and all(torch.isfinite(g).all() for g in gradients))

def train(out,arrays,progress,budget):
    start=time.monotonic();data=tensor_data(arrays);torch.set_num_threads(P['threads'])
    assert len(data['x'])==8 and arrays['split']==['train']*8
    result={};diagnostics={};models={}
    for method in P['methods']:
        torch.manual_seed(P['seed']);model=SequenceNetwork(data['global_condition'].shape[1])
        models[method]=model;initial=copy.deepcopy(model.state_dict());initial_hash=weight_hash(model)
        path=out/method;path.mkdir();save_torch(path/'initial.pt',dict(weights=initial,condition_dim=model.condition_dim,method=method))
        before=full_probe(model,data,method);optim=torch.optim.AdamW(model.parameters(),lr=P['lr'],weight_decay=P['weight_decay'])
        losses=[]
        for k in range(P['diagnostic_updates']):
            budget.check();optim.zero_grad(set_to_none=True)
            pred,_=sample(model,data['global_condition'],data['local_condition'],before['seeds'],method)
            loss=objective(pred,data['x']);loss.backward()
            if not finite_gradient(model):raise RuntimeError('DIAGNOSTIC_GRADIENT_NONFINITE')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),P['clip_grad']);optim.step()
            losses.append(dict(step=k+1,loss=float(loss),gradient_norm=float(norm)))
            if k%8==0:progress('TRAIN_full_sample_diagnostic',method=method,updates=k+1)
        after=full_probe(model,data,method)
        save_torch(path/'diagnostic_after.pt',dict(weights=model.state_dict(),optimizer=optim.state_dict(),updates=P['diagnostic_updates']))
        model.load_state_dict(initial);restored=weight_hash(model)==initial_hash
        gate=bool(before['finite'] and after['finite'] and after['loss']<before['loss'] and restored)
        diagnostics[method]=dict(before=before,after=after,history=losses,passed=gate,weights_restored=restored,
            actual_temporary_updates=P['diagnostic_updates'],retained_updates=0)
        atomic_json(path/'diagnostic.json',diagnostics[method])
        del optim
    passed=all(x['passed'] for x in diagnostics.values());atomic_json(out/'diagnostic_gate.json',dict(passed=passed,methods=diagnostics))
    if not passed:return dict(status='TRAIN_DIAGNOSTIC_BLOCKED',formal_updates=0,diagnostics=diagnostics),models
    for method in P['methods']:
        model=models[method];optim=torch.optim.AdamW(model.parameters(),lr=P['lr'],weight_decay=P['weight_decay'])
        calls_before=model.calls;initial_hash=weight_hash(model)
        path=out/method;history=[];digests=[];consumed=np.zeros(8,dtype=np.int64);rollouts=0
        for step in range(1,P['updates']+1):
            budget.check()
            if time.monotonic()-start>P['training_wall_s']:raise TimeoutError('TRAINING_WALL_BUDGET')
            optim.zero_grad(set_to_none=True);loss,batch=base_loss(model,data,method,step)
            consumed[batch['ids']]+=1;rollout_loss=None
            if method=='temporal_diffusion' and step%P['rollout_every']==0:
                ids=torch.tensor([(2*(step//4)+j)%8 for j in range(2)])
                seeds=[20265000+step*2+j for j in range(2)]
                pred,_=sample(model,data['global_condition'][ids],data['local_condition'][ids],seeds,method)
                rollout_loss=objective(pred,data['x'][ids]);loss=loss+P['rollout_weight']*rollout_loss;rollouts+=2
            loss.backward()
            if not finite_gradient(model) or not torch.isfinite(loss):raise RuntimeError('FORMAL_GRADIENT_NONFINITE')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),P['clip_grad']);optim.step()
            digests.append(batch['batch_noise_sha256'])
            if step%50==0:
                history.append(dict(step=step,loss=float(loss),gradient_norm=float(norm),
                    rollout_loss=float(rollout_loss) if rollout_loss is not None else None,
                    batch_noise_sha256=hashlib.sha256(''.join(digests[-50:]).encode()).hexdigest(),
                    accumulated_consumption=consumed.copy(),network_calls=model.calls))
                atomic_json(path/'progress.json',dict(actual_updates=step,history=history))
                progress('formal_training',method=method,updates=step,total=P['updates'])
        probe=full_probe(model,data,method)
        payload=dict(weights=model.state_dict(),optimizer=optim.state_dict(),updates=P['updates'],method=method,
                     condition_dim=model.condition_dim,seed=P['seed'],consumed=consumed,batch_noise_digests=digests)
        save_torch(path/'model.pt',payload)
        loaded=torch.load(path/'model.pt',map_location='cpu',weights_only=False)
        check=SequenceNetwork(model.condition_dim);check.load_state_dict(loaded['weights'])
        same=weight_hash(check)==weight_hash(model)
        steps=[int(v['step']) for v in loaded['optimizer']['state'].values()]
        if not same or not steps or set(steps)!={P['updates']}:raise RuntimeError('RELOAD_OR_ACTUAL_OPTIMIZER_UPDATES')
        result[method]=dict(actual_updates=P['updates'],data_consumption=consumed.tolist(),TRAIN_rows=8,VAL_rows_in_gradient=0,
            full_sampling_training_examples=rollouts if method=='temporal_diffusion' else 8*P['updates'],
            full_sampling_training_updates=P['updates']//4 if method=='temporal_diffusion' else P['updates'],
            final_TRAIN_probe=probe,weight_hash=weight_hash(model),reload_exact=same,optimizer_steps=sorted(set(steps)),
            network_calls_including_diagnostic=model.calls,formal_and_final_probe_calls=model.calls-calls_before,
            initial_weight_hash=initial_hash,weights_changed=weight_hash(model)!=initial_hash,
            parameters=sum(p.numel() for p in model.parameters()),history=history)
        atomic_json(path/'training.json',result[method])
    return dict(status='TRAINED',formal_updates=2*P['updates'],methods=result,diagnostics=diagnostics,
                elapsed_s=time.monotonic()-start,loss_scope='qualified full temporal coefficient reconstruction; not nonlinear current-world endpoint gradient'),models
