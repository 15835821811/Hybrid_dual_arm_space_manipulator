"""Compare call-audit backends with identical real physical energy/gradients."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .architecture_guard import LegacySentinel
from .model_io import load_model
from .problem_io import load_problem
from .api import boundary_codec
from .physics import DifferentiableTree,tensor
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .orchestration import atomic_json,check_budget
from .guidance_metric import residual_jacobian


def run(output,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);manifest=json.loads((root/'planning_manifest.json').read_text())
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('GUARD_CALIBRATION_CHECKPOINT_MISMATCH')
    r,snapshot,task,_=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points'],origin=manifest['budget'].get('boundary_origin','pivot_zero'))
    energy=MultiSpaceEnergy(Propagator(DifferentiableTree(r),snapshot),codec,Geometry(r),task,manifest['budget']['intervals'],collect_task_residual=True,
                           collision_aggregation=manifest.get('collision_aggregation','pair_mean'),joint_peak_penalty=manifest.get('joint_peak_penalty',False),task_objective=manifest.get('task_objective','center_squared'),planning_margins=manifest.get('planning_margins'))
    energy.weights.update(manifest.get('guide_weights',{}))
    with np.load(root/'candidate_000/denoising.npz') as data:z=checkpoint['mean']+checkpoint['scale']*data['guided'][-1]
    rows=[];gradients=[];jacobians=[]
    # Two passes with reversed order expose warm-up/order effects.
    for backend in ('profile','trace','trace','profile'):
        check_budget();x=tensor(z).requires_grad_();start=time.perf_counter()
        with LegacySentinel([],call_backend=backend) as guard:
            terms=energy.evaluate(x);forward=time.perf_counter()-start
            gradient=torch.autograd.grad(terms['total'],x,retain_graph=True)[0].numpy()
            jacobian_start=time.perf_counter()
            jacobian=residual_jacobian(energy.last_task_residual,x,batch_size=4).numpy()
            jacobian_seconds=time.perf_counter()-jacobian_start
        elapsed=time.perf_counter()-start;report=guard.report();gradients.append(gradient)
        jacobians.append(jacobian)
        atomic_json(output/f'guard_{len(rows)}.json',report)
        rows.append(dict(backend=backend,forward_seconds=forward,jacobian_seconds=jacobian_seconds,total_seconds=elapsed,calls=report['observed_ffgd_calls'],
                         methods=len(report['actual_methods']),terms={k:float(v.detach()) for k,v in terms.items()},forbidden_calls=report['forbidden_calls']))
    maximum_gradient_difference=float(np.max(abs(np.asarray(gradients)-gradients[0])))
    maximum_jacobian_difference=float(np.max(abs(np.asarray(jacobians)-jacobians[0])))
    same_values=all(row['terms']==rows[0]['terms'] for row in rows)
    return dict(passed=same_values and maximum_gradient_difference<1e-12 and maximum_jacobian_difference<1e-12 and all(not row['forbidden_calls'] for row in rows),
                scope='real full-energy call-audit performance and equivalence; no planner candidate added',
                same_values=same_values,maximum_gradient_difference=maximum_gradient_difference,
                maximum_jacobian_difference=maximum_jacobian_difference,jacobian_batch_size=4,rows=rows)
