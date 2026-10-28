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
    # Isolated diagnostic of the previous compiled-Jacobian stall. Keep this
    # executor choice out of the main sampler until whole-path AD is verified.
    with torch.jit.optimized_execution(config.get('jit_optimize',True)):
        return _run(path,config)


def _run(path,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);folder=root/'candidate_000'
    manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'))
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('DIAGNOSTIC_CHECKPOINT_MISMATCH')
    r,snapshot,task,problem=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points'],origin=manifest['budget'].get('boundary_origin','pivot_zero'))
    from .model_io import bind_free_coordinates
    checkpoint=bind_free_coordinates(checkpoint,task.duration)
    tree=DifferentiableTree(r)
    tree.use_momentum_mass_rows=config.get('mass_rows',False)
    if config.get('compiled_backend'):
        from .compiled_physics import compile_tree
        compile_tree(tree)
    aggregation=config.get('collision_aggregation',manifest.get('collision_aggregation','pair_mean'))
    joint_peak=config.get('joint_peak_penalty',manifest.get('joint_peak_penalty',False))
    p=Propagator(tree,snapshot);energy=MultiSpaceEnergy(p,codec,Geometry(r),task,config.get('intervals',manifest['budget']['intervals']),collect_task_residual=config.get('task_metric',False),collision_aggregation=aggregation,joint_peak_penalty=joint_peak,task_objective=config.get('task_objective',manifest.get('task_objective','center_squared')),planning_margins=config.get('planning_margins',manifest.get('planning_margins')))
    energy.weights.update(manifest.get('guide_weights',{}))
    stage=config.get('stage','clean');metric_stage=config.get('metric_stage',stage)
    if stage not in ('clean','guided') or metric_stage not in ('clean','guided'):raise ValueError('invalid saved diagnostic stage')
    if stage!=metric_stage and config.get('verify_jacobian'):raise ValueError('Jacobian verification requires same diagnostic and metric input')
    if stage!=metric_stage and not config.get('task_metric'):raise ValueError('different metric input requires task metric')
    with np.load(folder/'denoising.npz') as data:
        clean=data[stage][-1].copy();metric_input=data[metric_stage][-1].copy()
    mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale']);x=tensor(clean).requires_grad_()
    start=time.perf_counter()
    if stage!=metric_stage:x=tensor(metric_input).requires_grad_()
    terms=energy.evaluate(mean+scale*x)
    J=None;residual_energy_error=None
    if config.get('task_metric'):
        residual=energy.last_task_residual
        # The raw pose Jacobian remains the metric in a tolerance interior;
        # only the exact-center objective equals its squared residual norm.
        if energy.task_objective=='center_squared':residual_energy_error=float(abs(residual.square().sum()-terms['task']).detach())
        from .guidance_metric import residual_jacobian
        J=residual_jacobian(residual,x,lambda count,total:atomic_json(path/'derivative_progress.json',dict(rows=count,total=total,seconds=time.perf_counter()-start)),config.get('jacobian_batch_size',1))
    if stage!=metric_stage:
        del terms,residual
        x=tensor(clean).requires_grad_();terms=energy.evaluate(mean+scale*x)
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
        direction=torch.linalg.solve(metric,torch.tensor(gradient)).numpy()
        if 'coefficients' not in config:direction/=np.linalg.norm(direction)
    for length in config.get('coefficients',config.get('lengths',[.1,.25,.5,1.,2.])):
        check_budget();delta=length*direction
        if 'coefficients' in config:
            limits=manifest['budget']['sampling'];delta*=min(1.,limits['max_update_norm']/max(np.linalg.norm(delta),1e-30))
            change=np.max(abs((codec.N*checkpoint['scale'][None,:])@delta))
            delta*=min(1.,limits['max_control_change']/max(change,1e-30))
        proposal=clean-delta;z=checkpoint['mean']+checkpoint['scale']*proposal
        values=energy.native_evaluate(z);C=codec.decode(z);base=codec.decode(checkpoint['mean']+checkpoint['scale']*clean)
        times=np.linspace(0,task.duration,1025)
        row=dict(normalized_length=float(np.linalg.norm(delta)),step_coefficient=length if 'coefficients' in config else None,
                 accepted=bool(values['total']<=before['total']),maximum_control_change=float(np.max(abs(C-base))),energies=values,
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
                jacobian_batch_size=config.get('jacobian_batch_size',1),
                stage=stage,metric_stage=metric_stage,
                intervals=config.get('intervals',manifest['budget']['intervals']),mass_rows=config.get('mass_rows',False),
                jit_optimize=config.get('jit_optimize',True),compiled_backend=config.get('compiled_backend',False),
                collision_aggregation=aggregation,
                joint_peak_penalty=joint_peak,
                task_objective=energy.task_objective,metric_uses_raw_pose_residual=True,
                one_step_rays=rows,denoising_coupling=coupling,main_candidate_count_added=0,
                finding='Quantify the physical-step cap and the exact DDIM clean-to-next-state gain; acceptance gates are unchanged.')
