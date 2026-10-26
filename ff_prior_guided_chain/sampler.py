"""Frozen v-prediction DDIM; corrected clean estimates feed the next DDIM state."""
import numpy as np
import torch
from ff_prior.diffusion import schedule,weight_hash
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .models import load_static
from ff_prior_temporal_learning.models import SequenceNetwork
from ff_prior_temporal_learning.features import encode_conditions
from ff_prior.guidance import public_goal_initializer
from .protocol import P,ROOT


def generate(rep,guide,task,progress):
    directory=guide.out;directory.mkdir(parents=True,exist_ok=True)
    budget=guide.budget;phase=rep.phase;seed=P['phase_seeds'][phase]
    torch.set_num_threads(P['cpu_threads']);torch.set_grad_enabled(False)
    if phase=='A':
        model,codec=load_static(ROOT/P['static_model'])
        base=public_goal_initializer(task,guide.r,guide.pred.initial)
        center=codec.center(task['condition'],base)
        condition=torch.tensor(codec.condition(task['condition'],base)[None],dtype=torch.float32)
        detail=np.zeros((19,17));shape=(1,codec.dim)
        def forward(x,t):return model(x,t,condition)
        def decode(clean):return codec.decode(codec.coefficients(clean[0],center),base)+detail
        def encode(free):return codec.normalized(codec.encode(free,base),center)[None]
        atomic_npz(directory/'public_initializer.npz',base=base,center=center,condition=condition.numpy())
    else:
        checkpoint=ROOT/P['temporal_model']/'model.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=SequenceNetwork(payload['condition_dim']);model.load_state_dict(payload['weights']);model.eval()
        codec=read_json(ROOT/P['temporal_codec']);features=guide.targets.features(rep)
        gc,lc=encode_conditions(features,codec)
        condition=torch.tensor(gc[None],dtype=torch.float32);local=torch.tensor(lc[None],dtype=torch.float32)
        mean,scale=np.asarray(codec['mean']),np.asarray(codec['scale']);shape=(1,271,17)
        def forward(x,t):return model(x,t,condition,local)
        def decode(clean):
            control=np.vstack([rep.fixed[:2],clean[0]*scale+mean+rep.q0]);control[:3]=rep.fixed
            return control
        def encode(free):return ((np.vstack([rep.fixed,free])[2:]-rep.q0-mean)/scale)[None]
        atomic_npz(directory/'public_conditions.npz',global_condition=gc,local_condition=lc,
            initial=guide.pred.initial.values,coefficient_times=features['coefficient_times'])
        atomic_json(directory/'condition_schema.json',dict(obstacle_slots=features['obstacle_slot_ids'],
            neural_distribution_limitations='free-floating seven-obstacle moving-satellite conditions are out of the frozen training distribution',
            all_seven_constraints=True,phase=phase))
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=weight_hash(model);model.eval()
    x=torch.randn(shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    alpha=schedule().double();times=torch.linspace(999,0,16).round().long();records=[]
    for j,t in enumerate(times):
        progress('frozen_diffusion_ddim',phase=phase,step=j+1,total=16)
        budget.reserve('neural');v=forward(x.float(),t.repeat(1).float()).double();budget.complete('neural')
        a=alpha[t];clean=a.sqrt()*x-(1-a).sqrt()*v
        if not torch.isfinite(clean).all():raise ValueError('NONFINITE_NEURAL_CLEAN')
        decoded=decode(clean.numpy());free,qp=rep.project(decoded,budget.deadline)
        before_guidance=free.copy()
        if j in P['guidance_steps']:free=guide.update(free,j)
        if phase=='A':
            low=codec.basis@(codec.basis.T@(free-base));detail=free-base-low
        corrected=torch.tensor(encode(free),dtype=torch.float64)
        # Re-derive epsilon from the same x_t, so corrected x0 participates in
        # the reverse process instead of being an unrelated postprocessor.
        epsilon=(x-a.sqrt()*corrected)/(1-a).sqrt()
        decomposition=float(abs(a.sqrt()*corrected+(1-a).sqrt()*epsilon-x).max())
        if decomposition>1e-10:raise ValueError('GUIDED_DDIM_DECOMPOSITION')
        next_a=alpha[times[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
        next_x=next_a.sqrt()*corrected+(1-next_a).sqrt()*epsilon
        budget.ensure_output(x.numel()*8*6+free.nbytes*3+1024**2)
        atomic_npz(directory/f'ddim_{j:02d}.npz',xt=x.numpy(),v=v.numpy(),clean=clean.numpy(),decoded=decoded,
            projected=before_guidance,guided=free,encoded_corrected=corrected.numpy(),epsilon=epsilon.numpy(),next_xt=next_x.numpy(),
            deterministic_detail=detail if phase=='A' else np.empty(0))
        records.append(dict(step=j,t=int(t),alpha=float(a),next_alpha=float(next_a),projection=qp,
            guided=j in P['guidance_steps'],decomposition_error=decomposition,
            maximum_guidance_change=float(abs(free-before_guidance).max())))
        x=next_x;budget.save()
    after=weight_hash(model)
    if after!=before:raise ValueError('FROZEN_WEIGHT_CHANGED')
    atomic_json(directory/'sampler.json',dict(phase=phase,seed=seed,network_calls=model.calls,
        before_weight_hash=before,after_weight_hash=after,steps=records,new_neural_sample=True,
        low_rank_static_details_neural=False,optimizer_updates=0,training=False))
    final=guide.dense(free,directory/'qualification')
    atomic_json(directory/'qualification_result.json',dict(level='READY_FOR_FULL_QUALIFICATION' if final['metrics']['original_model_gates_pass'] else 'MODEL_PRECHECK_REJECTED',
        metrics=final['metrics'],new_neural_sample=True,physical_execution=False))
    return rep.ref(free),final
