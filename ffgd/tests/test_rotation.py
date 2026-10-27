import torch
import pytest
from ffgd.physics import rotation,tensor
from ffgd.costs import rotation_angle,rotation_log


def test_pi_domain_is_explicit_not_zero_gradient():
    with pytest.raises(ValueError,match='SO3_PI'):rotation_angle(rotation(tensor([1.,0,0]),tensor(torch.pi)))
    for theta in (0.,.2,torch.pi-1e-5):
        angle=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
        cost=rotation_angle(rotation(tensor([1.,0,0]),angle))**2
        grad=torch.autograd.grad(cost,angle)[0]
        torch.testing.assert_close(grad,2*angle,atol=1e-8,rtol=1e-8)


def test_so3_residual_vector_is_stable_and_has_correct_jacobian():
    axis=tensor([1.,2.,3.]);axis/=torch.linalg.vector_norm(axis)
    for theta in (0.,1e-8,.2,torch.pi-1e-5):
        angle=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
        residual=rotation_log(rotation(axis,angle))
        torch.testing.assert_close(residual,angle*axis,atol=1e-9,rtol=1e-9)
        derivative=torch.stack([torch.autograd.grad(v,angle,retain_graph=True)[0] for v in residual])
        torch.testing.assert_close(derivative,axis,atol=1e-8,rtol=1e-8)
