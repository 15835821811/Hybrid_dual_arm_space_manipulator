"""M3 learned checkpoint checks. No model-valid planning success is inferred."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from .model_io import load_model,normalize_condition
from .model import ConditionalDenoiser
from .diffusion import sample,SamplingBudget
from .trajectory import Boundary,BoundaryCodec
from .contracts import Snapshot
from .robot import Robot
from .physics import DifferentiableTree
from .propagation import Propagator
from .orchestration import atomic_json,check_budget


def run(path,config):
    torch.set_num_threads(1);device=config.get('device','cpu')
    model,checkpoint,identity=load_model(config['checkpoint'],device)
    other,_,_=load_model(config['checkpoint'],device)
    dataset=Path(config['dataset'])
    with np.load(dataset/'prior.npz') as stored:data={k:stored[k] for k in stored.files}
    ids=np.flatnonzero(~data['is_train'])[:5]
    if len(ids)<5:raise ValueError('five VAL conditions required')
    r=Robot();tree=DifferentiableTree(r);rows=[];outputs=[];budget=SamplingBudget(steps=60,guide_steps=0)
    for index,item in enumerate(ids):
        check_budget();seed=54200+index
        snapshot=Snapshot(r.model_hash,int(checkpoint['dataset_manifest']['state_mask']),data['initial_state'][item])
        prop=Propagator(tree,snapshot);T=float(data['duration'][item]);n=int(checkpoint['dataset_manifest']['control_points'])
        bc=[Boundary(0,j,tuple(range(17)),tuple(prop.initial_q if j==0 else prop.initial_v if j==1 else np.zeros(17))) for j in (0,1,2)]
        bc.extend(Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2));codec=BoundaryCodec(n,17,T,bc)
        condition=normalize_condition(data['tokens'][item:item+1],data['padding'][item:item+1],checkpoint,device)
        result=sample(model,condition,codec.free_dim,seed,budget,mean=checkpoint['mean'],scale=checkpoint['scale'])
        C=codec.decode(result['free']);error=float(np.max(abs(codec.A@C.ravel()-codec.b)))
        np.savez_compressed(path/f'sample_{index}.npz',control=C,free=result['free'],seed=seed,duration=T,
                            initial_noise=result['trace'][0]['input'],task_tokens=data['tokens'][item],initial_state=snapshot.state)
        probe=torch.tensor(result['trace'][20]['input'][None],device=device);t=torch.tensor([500],device=device)
        with torch.no_grad():
            prediction=model(probe,t,condition)
            loaded=other(probe,t,condition)
            changed=dict(tokens=condition['tokens'].clone(),padding_mask=condition['padding_mask'])
            # Change only public world target positions, keeping type and masks.
            active=(data['tokens'][item,:,3]>0)&(~data['padding'][item])
            changed['tokens'][0,active,9]+= .03/float(checkpoint['condition_scale'][9])
            conditioned=model(probe,t,changed)
            original_bias=model.output.bias.detach().clone();model.output.bias.add_(.1)
            intervened=model(probe,t,condition);model.output.bias.copy_(original_bias)
        repeat=None
        if index==0:
            again=sample(other,condition,codec.free_dim,seed,budget,mean=checkpoint['mean'],scale=checkpoint['scale'])
            repeat=float(np.max(abs(again['free']-result['free'])))
        xs,vs=prop.native_rollout(codec,result['free'],128);d=mujoco.MjData(r.model);d.qpos[:]=xs[-1];mujoco.mj_forward(r.model,d)
        task_error=[]
        for token in data['tokens'][item]:
            if token[3] and token[7]==1:
                tip=0 if token[5] else 1;body=r.tips[tip];position=d.xpos[body].copy()
                if tip==0:position+=d.xmat[body].reshape(3,3)@np.array([.0475,0,0])
                task_error.append(float(np.linalg.norm(position-token[9:12])))
        row=dict(index=index,seed=seed,free_dim=codec.free_dim,network_calls=result['network_calls'],boundary_residual=error,
                 finite=bool(np.isfinite(C).all()),reload_prediction_error=float((prediction-loaded).abs().max()),
                 condition_intervention_norm=float(torch.linalg.vector_norm(prediction-conditioned)),
                 weight_intervention_norm=float(torch.linalg.vector_norm(prediction-intervened)),
                 repeat_sampling_error=repeat,endpoint_position_errors_m=task_error,
                 basic_q17_bounds=bool(np.all(C>=r.lower)&np.all(C<=r.upper)),model_valid_status='NOT_RUN')
        row['passed']=row['finite'] and error<1e-10 and row['reload_prediction_error']==0 and (repeat is None or repeat==0) and row['condition_intervention_norm']>1e-8 and row['weight_intervention_norm']>1e-8
        rows.append(row);outputs.append(result['free']);atomic_json(path/'progress.json',rows)
    variation=np.std(outputs,axis=0)
    return dict(passed=all(x['passed'] for x in rows),checkpoint=identity,samples=rows,minimum_sample_coordinate_std=float(variation.min()),
                dataset_accepted=checkpoint['dataset_manifest']['accepted'],full_free_dim=len(checkpoint['mean']),
                architecture_status='NOT_RUN',engineering_status='NOT_RUN',learning_benefit_status='NOT_RUN',
                scope='trained full-variable model, full-noise finite boundary decoding and dependency checks only')
