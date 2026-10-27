import numpy as np
import pytest
import torch
from ffgd.trajectory import Boundary, BoundaryCodec, basis, AffineScaler
from ffgd.contracts import Candidate


def default_codec(duration=4.):
    joints=tuple(range(17)); zeros=(0.,)*17
    return BoundaryCodec(24,17,duration,[Boundary(0.,d,joints,zeros) for d in (0,1,2)]+[Boundary(duration,d,joints,zeros) for d in (1,2)])


def test_full_323_and_derivatives():
    c=default_codec(); assert c.free_dim == 323
    z=np.random.default_rng(1).normal(size=c.free_dim)
    C=c.decode(z)
    assert np.max(abs(c.A@C.ravel()-c.b)) < 1e-10
    assert np.max(abs(c.A@c.N)) < 1e-10
    np.testing.assert_allclose(c.encode(C),z,atol=1e-12)
    zt=torch.tensor(z,requires_grad=True)
    c.evaluate(zt,np.array([.3,1.3])).square().sum().backward()
    assert torch.linalg.vector_norm(zt.grad)>0
    for d in (1,2):
        np.testing.assert_allclose(basis(24,[.8,2.4],4,d),basis(24,[1.6,4.8],8,d)*2**d,atol=1e-10)
    h=1e-5;t=1.1
    np.testing.assert_allclose((c.evaluate(z,t+h)-c.evaluate(z,t-h))/(2*h),c.evaluate(z,t,1),atol=1e-7)


def test_masks_nonzero_and_unfixed_endpoint():
    c=BoundaryCodec(24,3,2.,[Boundary(0,0,(0,1),(1.,2.)),Boundary(0,1,(1,),(0.3,)),Boundary(2,2,(2,),(-.2,))])
    assert c.free_dim==68
    C=c.decode(np.ones(c.free_dim))
    np.testing.assert_allclose(c.A@C.ravel(),c.b,atol=1e-10)
    d=BoundaryCodec(24,3,2.,[Boundary(0,0,(0,1),(1.,2.)),Boundary(0,1,(1,),(0.3,)),Boundary(2,2,(2,),(-.2,))])
    np.testing.assert_array_equal(c.N,d.N)
    assert np.linalg.norm(basis(24,2.,2.)@c.decode(np.ones(c.free_dim))-basis(24,2.,2.)@c.decode(np.zeros(c.free_dim)))>0


def test_invalid_and_readonly():
    with pytest.raises(ValueError): BoundaryCodec(8,1,2.,[Boundary(0,0,(0,),(0.,)),Boundary(0,0,(0,),(1.,))])
    with pytest.raises(ValueError): basis(24,1.,0.)
    c=Candidate(np.zeros((24,17)),4.,3); identity=c.identity
    with pytest.raises(ValueError): c.control[0,0]=1
    with pytest.raises(ValueError): c.control.setflags(write=True)
    assert c.identity==identity
    with pytest.raises(ValueError): AffineScaler().fit(np.ones((2,3)),"TEST")
    x=np.random.default_rng(0).normal(size=(10,7)); s=AffineScaler().fit(x,"TRAIN")
    np.testing.assert_allclose(s.decode(s.encode(x)),x,atol=1e-15)
