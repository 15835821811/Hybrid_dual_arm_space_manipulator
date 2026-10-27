"""Task cost must distinguish feasible tolerance interiors from exact centers."""
import numpy as np
import torch
import pytest
from ffgd.robot import Robot
from ffgd.api import boundary_codec
from ffgd.physics import DifferentiableTree,tensor
from ffgd.propagation import Propagator
from ffgd.geometry import Geometry
from ffgd.contracts import TaskSpec,PoseConstraint
from ffgd.costs import MultiSpaceEnergy


def test_tolerance_loss_native_agreement_and_interior_gradient():
    torch.set_num_threads(1)
    robot=Robot();data=robot.new_data();snapshot=robot.snapshot(data)
    tree=DifferentiableTree(robot);prop=Propagator(tree,snapshot)
    codec=boundary_codec(robot,snapshot,4.,8,origin='initial_position')
    z0=np.zeros(codec.free_dim);kin=tree.kinematics(tensor(data.qpos))
    body=robot.tips[0];rotation=kin['rotation'][body].numpy()
    position=kin['position'][body].numpy()+rotation@np.array([.0475,0,0])
    geometry=Geometry(robot)
    for displacement in (0.,.002,.008):
        target=PoseConstraint(4.,0,position+np.array([displacement,0,0]),rotation,position_tolerance=.005,rotation_enabled=False)
        task=TaskSpec(4.,(target,))
        energy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=2,task_objective='tolerance_squared_hinge',collect_task_residual=True)
        z=tensor(z0).requires_grad_();value=energy.evaluate(z)['task']
        gradient=torch.autograd.grad(value,z)[0]
        expected=max(displacement/.005-1,0)**2
        assert abs(float(value.detach())-expected)<1e-9
        assert abs(energy.native_evaluate(z0)['task']-expected)<1e-9
        if displacement<.005:assert torch.count_nonzero(gradient)==0
        else:assert torch.linalg.vector_norm(gradient)>1e-6
        if displacement==.002:assert torch.linalg.vector_norm(energy.last_task_residual)>.39
        legacy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=2)
        assert abs(legacy.native_evaluate(z0)['task']-(displacement/.005)**2)<1e-9


def test_planning_reserve_activates_before_hard_boundary_without_mutating_task():
    torch.set_num_threads(1)
    robot=Robot();data=robot.new_data();snapshot=robot.snapshot(data)
    tree=DifferentiableTree(robot);prop=Propagator(tree,snapshot);geometry=Geometry(robot)
    codec=boundary_codec(robot,snapshot,.5,8,origin='initial_position')
    kin=tree.kinematics(tensor(data.qpos));body=robot.tips[0]
    rotation=kin['rotation'][body].numpy();position=kin['position'][body].numpy()+rotation@np.array([.0475,0,0])
    target=PoseConstraint(.5,0,position+[.0045,0,0],rotation,rotation_enabled=False)
    task=TaskSpec(.5,(target,));saved=target.position.copy();limits=robot.spec.planner_acceleration_limits.copy()
    energy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=4,task_objective='tolerance_squared_hinge',joint_peak_penalty=True,
                            planning_margins=dict(task_tolerance_fraction=.8,derivative_limit_fraction=.95))
    z=tensor(np.zeros(codec.free_dim)).requires_grad_();value=energy.evaluate(z)['task']
    assert abs(float(value.detach())-.01)<1e-9
    assert torch.linalg.vector_norm(torch.autograd.grad(value,z)[0])>1e-6
    assert abs(energy.native_evaluate(z.detach().numpy())['task']-.01)<1e-9
    direction=np.random.default_rng(738).normal(size=codec.free_dim)
    scale=np.max(abs(codec.evaluate(direction,energy.grid,2))/limits)
    direction*=.975/scale
    z=tensor(direction).requires_grad_();joint=energy.evaluate(z)['joint']
    assert float(joint.detach())>0 and torch.linalg.vector_norm(torch.autograd.grad(joint,z)[0])>1e-6
    assert abs(float(joint.detach())-energy.native_evaluate(direction)['joint'])<1e-9
    energy.set_planning_margins()
    assert energy.native_evaluate(direction)['joint']==0
    np.testing.assert_array_equal(target.position,saved)
    np.testing.assert_array_equal(robot.spec.planner_acceleration_limits,limits)
    assert target.position_tolerance==.005 and target.time==.5
    for invalid in ({'unknown':.8},{'task_tolerance_fraction':1.01},{'derivative_limit_fraction':0},{'task_tolerance_fraction':float('nan')}):
        with pytest.raises(ValueError):energy.set_planning_margins(invalid)
