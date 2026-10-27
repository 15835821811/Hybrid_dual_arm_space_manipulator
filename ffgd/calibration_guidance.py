"""Fixed failed-sample step-size experiment, never a planner or repaired plan."""
import json
import time
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
    tree=DifferentiableTree(r)
    if config.get('compiled_backend'):
        from .compiled_physics import compile_tree
        compile_tree(tree)
    p=Propagator(tree,snapshot);energy=MultiSpaceEnergy(p,codec,Geometry(r),task,manifest['budget']['intervals'],collect_task_residual=config.get('task_metric',False))
    with np.load(folder/'denoising.npz') as data:clean=data['clean'][-1].copy()
    mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale']);x=tensor(clean).requires_grad_()
    start=time.perf_counter();terms=energy.evaluate(mean+scale*x)
    J=None;residual_energy_error=None
    if config.get('task_metric'):
        residual=energy.last_task_residual
        residual_energy_error=float(abs(residual.square().sum()-terms['task']).detach())
        from .guidance_metric import residual_jacobian
        J=residual_jacobian(residual,x,lambda count,total:atomic_json(path/'derivative_progress.json',dict(rows=count,total=total,seconds=time.perf_counter()-start)))
    gradient=torch.autograd.grad(terms['total'],x)[0].numpy();norm=float(np.linalg.norm(gradient));direction=gradient/norm
    derivative_seconds=time.perf_counter()-start
    verification=[];jacobian_verified=None
    if config.get('verify_jacobian'):
        rng=np.random.default_rng(78321)
        value_error=float(np.max(abs(residual.detach().numpy()-energy.native_task_residual((mean+scale*x).detach().numpy()))))
        for i in range(8):
            direction_fd=rng.normal(size=len(clean));direction_fd/=np.linalg.norm(direction_fd)
            ad=J.numpy()@direction_fd
            for epsilon in (1e-4,1e-5,1e-6):
                check_budget()
                plus=energy.native_task_residual(checkpoint['mean']+checkpoint['scale']*(clean+epsilon*direction_fd))
                minus=energy.native_task_residual(checkpoint['mean']+checkpoint['scale']*(clean-epsilon*direction_fd))
                fd=(plus-minus)/(2*epsilon)
                error=float(np.linalg.norm(ad-fd)/max(1.,np.linalg.norm(fd)))
                verification.append(dict(direction=i,epsilon=epsilon,relative_error=error))
        jacobian_verified=value_error<1e-8 and max(row['relative_error'] for row in verification)<1e-5
        atomic_json(path/'jacobian_check.json',dict(passed=jacobian_verified,native_value_max_error=value_error,rows=verification))
    before={k:float(v.detach()) for k,v in terms.items()};rows=[];proposals=[]
    from .guidance_metric import acceleration_metric
    metric=acceleration_metric(codec,checkpoint['scale'],r.spec.planner_acceleration_limits,energy.grid,config.get('metric_weight',0))
    if J is not None:metric=metric+J.T@J
    if config.get('metric_weight',0) or J is not None:
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
    return dict(passed=jacobian_verified is not False,scope='bounded root-cause experiment, no plan returned or executed',source_run=str(root),before=before,gradient_norm=norm,jacobian_verified=jacobian_verified,
                metric_weight=config.get('metric_weight',0),metric_condition=float(torch.linalg.cond(metric)),positive_descent_inner_product=float(gradient@direction),
                task_metric=J is not None,task_jacobian_singular_values=torch.linalg.svdvals(J).tolist() if J is not None else None,
                residual_energy_error=residual_energy_error,derivative_seconds=derivative_seconds,
                one_step_rays=rows,denoising_coupling=coupling,main_candidate_count_added=0,
                finding='Quantify the physical-step cap and the exact DDIM clean-to-next-state gain; acceptance gates are unchanged.')
