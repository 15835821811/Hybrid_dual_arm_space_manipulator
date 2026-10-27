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


def run(output,config):
    torch.set_num_threads(1);root=Path(config['planning_run']);manifest=json.loads((root/'planning_manifest.json').read_text())
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('GUARD_CALIBRATION_CHECKPOINT_MISMATCH')
    r,snapshot,task,_=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points'])
    energy=MultiSpaceEnergy(Propagator(DifferentiableTree(r),snapshot),codec,Geometry(r),task,manifest['budget']['intervals'])
    with np.load(root/'candidate_000/denoising.npz') as data:z=checkpoint['mean']+checkpoint['scale']*data['guided'][-1]
    rows=[];gradients=[]
    # Two passes with reversed order expose warm-up/order effects.
    for backend in ('profile','trace','trace','profile'):
        check_budget();x=tensor(z).requires_grad_();start=time.perf_counter()
        with LegacySentinel([],call_backend=backend) as guard:
            terms=energy.evaluate(x);forward=time.perf_counter()-start
            gradient=torch.autograd.grad(terms['total'],x)[0].numpy()
        elapsed=time.perf_counter()-start;report=guard.report();gradients.append(gradient)
        atomic_json(output/f'guard_{len(rows)}.json',report)
        rows.append(dict(backend=backend,forward_seconds=forward,total_seconds=elapsed,calls=report['observed_ffgd_calls'],
                         methods=len(report['actual_methods']),terms={k:float(v.detach()) for k,v in terms.items()},forbidden_calls=report['forbidden_calls']))
    maximum_gradient_difference=float(np.max(abs(np.asarray(gradients)-gradients[0])))
    same_values=all(row['terms']==rows[0]['terms'] for row in rows)
    return dict(passed=same_values and maximum_gradient_difference<1e-12 and all(not row['forbidden_calls'] for row in rows),
                scope='real full-energy call-audit performance and equivalence; no planner candidate added',
                same_values=same_values,maximum_gradient_difference=maximum_gradient_difference,rows=rows)
