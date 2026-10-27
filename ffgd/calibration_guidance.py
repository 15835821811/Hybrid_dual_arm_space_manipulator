"""Fixed failed-sample step-size experiment, never a planner or repaired plan."""
import json
from pathlib import Path
import numpy as np
import torch
from .problem_io import load_problem
from .model_io import load_model
from .api import boundary_codec
from .physics import DifferentiableTree,tensor
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .orchestration import atomic_json,check_budget


def run(path,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);folder=root/'candidate_000'
    manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('DIAGNOSTIC_CHECKPOINT_MISMATCH')
    r,snapshot,task,problem=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,24)
    p=Propagator(DifferentiableTree(r),snapshot);energy=MultiSpaceEnergy(p,codec,Geometry(r),task,manifest['budget']['intervals'])
    with np.load(folder/'denoising.npz') as data:clean=data['clean'][-1].copy()
    mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale']);x=tensor(clean).requires_grad_()
    terms=energy.evaluate(mean+scale*x);gradient=torch.autograd.grad(terms['total'],x)[0].numpy();norm=float(np.linalg.norm(gradient));direction=gradient/norm
    before={k:float(v.detach()) for k,v in terms.items()};rows=[];proposals=[]
    from .guidance_metric import acceleration_metric
    metric=acceleration_metric(codec,checkpoint['scale'],r.spec.planner_acceleration_limits,energy.grid,config.get('metric_weight',0))
    if config.get('metric_weight',0):
        direction=torch.linalg.solve(metric,torch.tensor(gradient)).numpy();direction/=np.linalg.norm(direction)
    for length in config.get('lengths',[.1,.25,.5,1.,2.]):
        check_budget();proposal=clean-length*direction;z=checkpoint['mean']+checkpoint['scale']*proposal
        values=energy.native_evaluate(z);C=codec.decode(z);base=codec.decode(checkpoint['mean']+checkpoint['scale']*clean)
        times=np.linspace(0,task.duration,1025)
        row=dict(normalized_length=length,maximum_control_change=float(np.max(abs(C-base))),energies=values,
                 maximum_velocity_ratio=float(np.max(abs(codec.evaluate(z,times,1))/r.spec.planner_velocity_limits)),
                 maximum_acceleration_ratio=float(np.max(abs(codec.evaluate(z,times,2))/r.spec.planner_acceleration_limits)))
        rows.append(row);proposals.append(proposal);atomic_json(path/'rays.json',rows)
    trace=json.loads((folder/'denoising.json').read_text(encoding='utf-8'));alpha=np.cumprod(1-np.linspace(.0001,.02,1000));coupling=[]
    for i,row in enumerate(trace[:-1]):
        if not row['active']:continue
        k=row['noise_level'];nextk=trace[i+1]['noise_level'];gain=float(np.sqrt(alpha[nextk])-np.sqrt(1-alpha[nextk])*np.sqrt(alpha[k]/(1-alpha[k])))
        coupling.append(dict(step=i,clean_to_next_state_gain=gain,guide_reduction=row['updates'][-1]['before']-row['updates'][-1]['after'],
                             next_network_energy_increase=trace[i+1]['energies_before_guidance']['total']-row['updates'][-1]['after']))
    np.savez_compressed(path/'diagnostic_inputs.npz',original_clean=clean,gradient=gradient,one_step_proposals=np.asarray(proposals))
    return dict(passed=True,scope='bounded root-cause experiment, no plan returned or executed',source_run=str(root),before=before,gradient_norm=norm,
                metric_weight=config.get('metric_weight',0),metric_condition=float(torch.linalg.cond(metric)),positive_descent_inner_product=float(gradient@direction),
                one_step_rays=rows,denoising_coupling=coupling,main_candidate_count_added=0,
                finding='Quantify the physical-step cap and the exact DDIM clean-to-next-state gain; acceptance gates are unchanged.')
