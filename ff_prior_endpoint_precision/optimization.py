"""True-current endpoint gradients and transactional one-step AdamW proposals."""
import copy,time
import numpy as np
import torch
from ff_prior.storage import digest,file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior_paired_objectives.objectives import predict,terms,vector_gradient,get_vector,set_vector
from ff_prior_origin_learn_bounded.learning import save_torch
from ff_prior.spline import Reference
from . import endpoint
from .protocol import PROTOCOL as P


def assign_grad(model,vector):
    n=0
    for v in model.parameters():v.grad=vector[n:n+v.numel()].view_as(v).clone();n+=v.numel()

def groups(record):return np.r_[np.asarray(record['position_m'])/.005,np.asarray(record['rotation_rad'])/np.deg2rad(1)]

def admissible(before,after):
    a=groups(before);b=groups(after)
    return bool(np.isfinite(b).all() and max(b)<max(a)-1e-9 and np.all(b[a<=1.]<=1.))

def propose(model,optimizer,gradient):
    optimizer.zero_grad();assign_grad(model,gradient)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not np.isfinite(norm):raise ValueError('nonfinite parameter gradient')
    optimizer.step();return get_vector(model),copy.deepcopy(optimizer.state_dict()),norm

def endpoint_record(model,codec,data,arrays,rows,cache,spec,deadline,budget):
    i,t,seed=spec
    with torch.no_grad():x=predict(model,data,i,t,seed).numpy().astype(float)
    rr,ii,_,_=cache.get(rows[i]);budget.reserve(1)
    value,record=endpoint.evaluate(x,codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline,512)
    z=codec.decode(codec.coefficients(x,data['centers'][i]),arrays[1][i]);ref=Reference(z,rows[i]['task']['condition']['initial_features'],rows[i]['task']['condition']['duration'])
    # Domain means joint range/finite for gradient probes, not MODEL_VALID certification.
    domain=bool(np.isfinite(z).all())
    # B-spline controls in range are a sufficient convex-hull joint-domain condition.
    domain=domain and bool(np.all(z>=rr.lower) and np.all(z<=rr.upper))
    return dict(record,loss=value,x_hash=digest(x),domain=domain)

def controlled_step(dest,model,optimizer,basegrad,codec,data,arrays,rows,cache,specs,controls,deadline,budget,retain):
    dest.mkdir(parents=True,exist_ok=False)
    original=get_vector(model);state=copy.deepcopy(optimizer.state_dict());w0=weight_hash(model)
    save_torch(dest/'before.pt',{'weights':model.state_dict(),'optimizer':state,'specs':specs,'controls':controls})
    rawgrads=[];proofs=[]
    for j,(i,t,seed) in enumerate(specs):
        pred=predict(model,data,i,t,seed);rr,ii,_,_=cache.get(rows[i]);budget.reserve(212)
        value,g,proof,arr=endpoint.gradient(pred.detach().numpy().astype(float),codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline)
        atomic_npz(dest/f'gradient_{j}.npz',**arr)
        proof.update(sample_id=arrays[3][i],t=t,noise_seed=seed,input_weight_hash=w0,arrays_sha256=file_digest(dest/f'gradient_{j}.npz'))
        atomic_json(dest/f'gradient_{j}.json',proof);proofs.append(proof['status'])
        rawgrads.append(vector_gradient(endpoint.attach_gradient(pred,value,g),model).detach())
    world=torch.stack(rawgrads).mean(0);bn=float(basegrad.norm());wn=float(world.norm())
    scale=.1*bn/wn if bn>1e-12 and wn>1e-12 else 0.
    mixed=basegrad+scale*world
    base_vec,base_state,base_clip=propose(model,optimizer,basegrad)
    set_vector(model,original);optimizer.load_state_dict(copy.deepcopy(state))
    mixed_vec,mixed_state,mixed_clip=propose(model,optimizer,mixed)
    set_vector(model,original);optimizer.load_state_dict(copy.deepcopy(state))
    checkspec=specs+controls
    before=[endpoint_record(model,codec,data,arrays,rows,cache,s,deadline,budget) for s in checkspec]
    trials=[];accepted=None
    if all(x=='PASS' for x in proofs) and scale>0:
        for fraction in P['line_scales']:
            set_vector(model,original+fraction*(mixed_vec-original))
            after=[endpoint_record(model,codec,data,arrays,rows,cache,s,deadline,budget) for s in checkspec]
            ok=all(r['domain'] and admissible(a,r) for a,r in zip(before,after))
            trials.append({'fraction':fraction,'after':after,'accepted':ok})
            if ok:accepted=fraction;break
    if retain:
        if accepted is None:set_vector(model,base_vec);optimizer.load_state_dict(base_state)
        else:set_vector(model,original+accepted*(mixed_vec-original));optimizer.load_state_dict(mixed_state)
    else:set_vector(model,original);optimizer.load_state_dict(state)
    atomic_npz(dest/'parameter_proposals.npz',base_gradient=basegrad.numpy(),world_gradient=world.numpy(),
       base_delta=(base_vec-original).numpy(),combined_delta=(mixed_vec-original).numpy())
    result={'gradient_status':'PASS' if all(x=='PASS' for x in proofs) else 'BLOCKED','accepted':accepted is not None,
      'fraction':accepted,'base_norm':bn,'world_norm':wn,'world_coefficient':scale,'weighted_world_norm':scale*wn,
      'cosine':float(torch.dot(basegrad,world)/max(bn*wn,1e-30)),'base_clip_norm':base_clip,'mixed_clip_norm':mixed_clip,
      'input_weight_hash':w0,'output_weight_hash':weight_hash(model),'before':before,'trials':trials,
      'updates_retained':retain,'committed_choice':'combined' if retain and accepted is not None else 'base' if retain else 'NONE',
      'ordinary_updates_committed':int(retain),'optimizer_restored':not retain,'world_samples':len(specs),
      'actual_coupled_calls':212*len(specs)+len(before)+sum(len(t['after']) for t in trials)}
    if not retain and weight_hash(model)!=w0:raise ValueError('diagnostic changed weights')
    atomic_json(dest/'result.json',result);return result
