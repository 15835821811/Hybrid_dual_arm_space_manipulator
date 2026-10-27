"""One-step multi-space weight sensitivity at a saved failed sample, not a solver."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from .model_io import load_model
from .problem_io import load_problem
from .api import boundary_codec
from .physics import DifferentiableTree,tensor
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .guidance_metric import acceleration_metric,residual_jacobian
from .trajectory import knots
from .orchestration import atomic_json,check_budget


def run(output,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);manifest=json.loads((root/'planning_manifest.json').read_text())
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('WEIGHT_DIAGNOSTIC_CHECKPOINT_MISMATCH')
    r,snapshot,task,_=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points'],origin=manifest['budget'].get('boundary_origin','pivot_zero'))
    aggregation=config.get('collision_aggregation',manifest.get('collision_aggregation','pair_mean'))
    tree=DifferentiableTree(r);tree.use_momentum_mass_rows=manifest.get('physics_backend') in ('eager_mass_rows','batched_bodyframe','batched_full_path')
    tree.use_batched_bodyframe_momentum=manifest.get('physics_backend') in ('batched_bodyframe','batched_full_path')
    tree.use_batched_path_physics=manifest.get('physics_backend')=='batched_full_path'
    peak=config.get('joint_peak_penalty',manifest.get('joint_peak_penalty',False))
    energy=MultiSpaceEnergy(Propagator(tree,snapshot),codec,Geometry(r),task,manifest['budget']['intervals'],collect_task_residual=True,collision_aggregation=aggregation,joint_peak_penalty=peak,task_objective=config.get('task_objective',manifest.get('task_objective','center_squared')))
    energy.weights.update(manifest.get('guide_weights',{}))
    energy.weights.update(config.get('guide_weights',{}))
    candidate_index=int(config.get('candidate_index',0))
    with np.load(root/f'candidate_{candidate_index:03d}'/'denoising.npz') as data:x0=data[config.get('stage','clean')][-1].copy()
    x=tensor(x0).requires_grad_();mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale']);terms=energy.evaluate(mean+scale*x)
    J=residual_jacobian(energy.last_task_residual,x,batch_size=4)
    gradients={name:torch.autograd.grad(terms[name],x,retain_graph=True)[0].detach() for name in energy.weights}
    values={name:float(value.detach()) for name,value in terms.items()}
    oracle=energy.native_evaluate((mean+scale*x).detach().numpy())
    value_error=float(max(abs(values[k]-oracle[k])/max(1.,abs(oracle[k])) for k in oracle))
    metric=acceleration_metric(codec,checkpoint['scale'],r.spec.planner_acceleration_limits,energy.grid,manifest['budget']['sampling']['acceleration_metric_weight'])+J.T@J
    factor=torch.linalg.cholesky(metric);jac=tensor(codec.N*checkpoint['scale'][None,:]);budget=manifest['budget']['sampling'];rows=[];proposals=[]
    for weight in config.get('constraint_weights',[1.,10.,100.,1000.]):
        if not np.isfinite(weight) or weight<=0:raise ValueError('invalid diagnostic weight')
        check_budget();energy.weights.update(collision=weight,joint=weight,actuator=weight)
        gradient=sum(energy.weights[k]*g for k,g in gradients.items());before=sum(energy.weights[k]*values[k] for k in energy.weights)
        direction=torch.cholesky_solve(gradient[:,None],factor).squeeze(1)
        coefficient=budget['step_size']*before/max(float(gradient@direction),1e-30)
        if budget.get('max_step_coefficient') is not None:coefficient=min(coefficient,budget['max_step_coefficient'])
        delta=coefficient*direction;delta*=min(1.,budget['max_update_norm']/max(float(torch.linalg.vector_norm(delta)),1e-30))
        delta*=min(1.,budget['max_control_change']/max(float(torch.max(abs(jac@delta))),1e-30))
        trials=[]
        for trial in range(budget['line_trials']):
            check_budget();proposal=x0-(2.**(-trial))*delta.numpy();z=checkpoint['mean']+checkpoint['scale']*proposal
            after=energy.native_evaluate(z);C=codec.decode(z);spline=BSpline(knots(codec.n),C,3)
            trials.append(dict(line_scale=2.**(-trial),energies=after,weighted_before=before,accepted=bool(after['total']<=before),
                               planner_position_hull_violation_rad=float(np.maximum(C-r.upper,r.lower-C).max()),
                               velocity_hull_ratio=float(np.max(abs(spline.derivative(1).c)/task.duration/r.spec.planner_velocity_limits)),
                               acceleration_hull_ratio=float(np.max(abs(spline.derivative(2).c)/task.duration**2/r.spec.planner_acceleration_limits))))
            if after['total']<=before:break
        rows.append(dict(constraint_weight=weight,trials=trials,step_coefficient=coefficient));proposals.append(proposal)
        atomic_json(output/'weight_rays.json',rows)
    np.savez_compressed(output/'diagnostic_inputs.npz',original_clean=x0,one_step_proposals=proposals,**{name:g.numpy() for name,g in gradients.items()})
    return dict(passed=value_error<1e-6,scope='bounded weight-sensitivity diagnostic; no plan returned, repaired or executed',
                source_run=str(root),native_value_relative_error=value_error,original_terms=values,
                collision_aggregation=aggregation,
                task_objective=energy.task_objective,
                joint_peak_penalty=peak,candidate_index=candidate_index,
                gradient_norms={name:float(torch.linalg.vector_norm(g)) for name,g in gradients.items()},
                preconditioned_gradient_norms={name:float(torch.linalg.vector_norm(torch.cholesky_solve(g[:,None],factor))) for name,g in gradients.items()},
                one_step_rays=rows,main_candidate_count_added=0)
