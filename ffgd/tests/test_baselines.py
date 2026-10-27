"""Software budget and isolation tests; these are not robot success evidence."""
import numpy as np
import pytest
import torch
from ffgd.baselines.compare import optimize, nonlearned_initial
from ffgd.diffusion import SamplingBudget
from ffgd.trajectory import Boundary, BoundaryCodec
from ffgd.architecture_guard import LegacySentinel, ArchitectureViolation


def quadratic(x):
    quadratic.task_residual=x
    return dict(total=(x*x).sum())


def test_baseline_optimizer_enforces_physics_budget_and_keeps_partial():
    sampling=SamplingBudget(task_metric=True,steps=3,guide_steps=3,updates_per_step=1,
                            step_size=.2,max_update_norm=2.,max_control_change=2.)
    result,report=optimize(np.array([1.,-1.]),quadratic,torch.eye(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64),
                          sampling,dict(energy_evaluations=3,gradient_evaluations=3,jacobian_refreshes=3))
    assert report['status']=='INCOMPLETE_PHYSICS_BUDGET'
    assert report['energy_evaluations']==3
    assert report['gradient_evaluations']==2
    assert len(report['trace'])==1
    assert np.linalg.norm(result)<np.sqrt(2.)


def test_nonlearned_initializations_preserve_only_public_boundaries():
    codec=BoundaryCodec(10,2,3.,[Boundary(0.,0,(0,1),(.2,-.3)),Boundary(0.,1,(0,1),(0.,0.)),Boundary(3.,1,(0,1),(0.,0.))])
    values=[nonlearned_initial(codec,np.array([.2,-.3]),np.array([2.,4.]),500+i,i) for i in range(5)]
    assert len({v.tobytes() for v in values})==5
    for z in values:assert np.max(abs(codec.A@codec.decode(z).ravel()-codec.b))<1e-10


def test_main_guard_rejects_already_imported_comparison_entry():
    with pytest.raises(ArchitectureViolation):
        with LegacySentinel([]):
            nonlearned_initial(None,None,None,0,0)
