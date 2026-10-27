"""Full-matrix versus exact base-row whole-path values, gradients and Jacobians."""
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
from .guidance_metric import residual_jacobian
from .architecture_guard import LegacySentinel
from .orchestration import atomic_json,check_budget


def run(output,config):
    torch.set_num_threads(1);root=Path(config['planning_run'])
    manifest=json.loads((root/'planning_manifest.json').read_text())
    model,checkpoint,identity=load_model(config['checkpoint'],'cpu')
    if identity['sha256']!=manifest['checkpoint']['sha256']:raise ValueError('MASS_ROWS_CHECKPOINT_MISMATCH')
    r,snapshot,task,_=load_problem(root/'problem.json');codec=boundary_codec(r,snapshot,task.duration,manifest['budget']['control_points'])
    with np.load(root/'candidate_000/denoising.npz') as data:initial=data['clean'][-1].copy()
    mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale']);rows=[];reference=None
    for active in (False,True,True,False):
        check_budget();tree=DifferentiableTree(r);tree.use_momentum_mass_rows=active
        energy=MultiSpaceEnergy(Propagator(tree,snapshot),codec,Geometry(r),task,manifest['budget']['intervals'],collect_task_residual=True)
        energy.weights.update(manifest.get('guide_weights',{}));x=tensor(initial).requires_grad_();start=time.perf_counter()
        with LegacySentinel([]) as sentinel:
            terms=energy.evaluate(mean+scale*x);forward=time.perf_counter()-start
            J=residual_jacobian(energy.last_task_residual,x,batch_size=4).numpy()
            gradient=torch.autograd.grad(terms['total'],x)[0].numpy()
        elapsed=time.perf_counter()-start;values=np.array([float(v.detach()) for v in terms.values()])
        if reference is None:reference=(values.copy(),gradient.copy(),J.copy())
        value_error=float(np.max(abs(values-reference[0])/np.maximum(1.,abs(reference[0]))))
        gradient_error=float(np.linalg.norm(gradient-reference[1])/max(1.,np.linalg.norm(reference[1])))
        jacobian_error=float(np.linalg.norm(J-reference[2])/max(1.,np.linalg.norm(reference[2])))
        rows.append(dict(base_rows_only=active,forward_seconds=forward,total_seconds=elapsed,normalized_value_error=value_error,
                         gradient_relative_error=gradient_error,jacobian_relative_error=jacobian_error,
                         terms={k:float(v.detach()) for k,v in terms.items()},full_free_dimension=len(x),task_residual_dimension=len(J)))
        atomic_json(output/'progress.json',rows);atomic_json(output/f'audit_{len(rows)}.json',sentinel.report())
        del energy,terms,x,tree
    return dict(passed=all(row['normalized_value_error']<1e-10 and row['gradient_relative_error']<1e-10 and row['jacobian_relative_error']<1e-10 for row in rows),
                source_run=str(root),rows=rows,scope='exact base mass-row optimization equivalence; all 67 actuator demands still use full mass matrix',
                candidate_count_added=0,main_backend_enabled=False)
