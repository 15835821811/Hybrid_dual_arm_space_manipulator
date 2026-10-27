"""Task cost must distinguish feasible tolerance interiors from exact centers."""
import numpy as np
import torch
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
        energy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=2,task_objective='tolerance_squared_hinge')
        z=tensor(z0).requires_grad_();value=energy.evaluate(z)['task']
        gradient=torch.autograd.grad(value,z)[0]
        expected=max(displacement/.005-1,0)**2
        assert abs(float(value.detach())-expected)<1e-9
        assert abs(energy.native_evaluate(z0)['task']-expected)<1e-9
        if displacement<.005:assert torch.count_nonzero(gradient)==0
        else:assert torch.linalg.vector_norm(gradient)>1e-6
        legacy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=2)
        assert abs(legacy.native_evaluate(z0)['task']-(displacement/.005)**2)<1e-9
