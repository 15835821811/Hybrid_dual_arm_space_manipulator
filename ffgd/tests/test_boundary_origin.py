import numpy as np
import pytest
from ffgd.trajectory import Boundary,BoundaryCodec


def boundaries(q,v,T):
    n=len(q)
    return [Boundary(0,k,tuple(range(n)),tuple(q if k==0 else v if k==1 else np.zeros(n))) for k in (0,1,2)]+[Boundary(T,k,tuple(range(n)),(0.,)*n) for k in (1,2)]


def test_origin_preserves_full_space_and_nonzero_initial_derivatives():
    rng=np.random.default_rng(914);q=rng.uniform(-2,2,17);v=rng.normal(0,.02,17)
    for T in (.8,5.):
        bc=boundaries(q,v,T);old=BoundaryCodec(24,17,T,bc);new=BoundaryCodec(24,17,T,bc,origin='initial_position')
        assert old.free_dim==new.free_dim==323
        np.testing.assert_array_equal(old.N,new.N)
        C=old.decode(rng.normal(size=323));z=new.encode(C)
        np.testing.assert_allclose(new.decode(z),C,atol=2e-14,rtol=0)
        np.testing.assert_allclose(new.A@new.offset,new.b,atol=1e-10,rtol=0)
        np.testing.assert_allclose(new.evaluate(z,0,1),v,atol=1e-12,rtol=0)


def test_relative_coordinates_are_initial_position_translation_invariant():
    rng=np.random.default_rng(915);q=rng.normal(size=17);delta=rng.uniform(-2,2,17);v=np.zeros(17)
    a=BoundaryCodec(24,17,5.,boundaries(q,v,5.),origin='initial_position')
    b=BoundaryCodec(24,17,5.,boundaries(q+delta,v,5.),origin='initial_position')
    np.testing.assert_allclose(a.decode(np.zeros(323)),np.tile(q,(24,1)),atol=2e-14,rtol=0)
    z=rng.normal(0,.1,323)
    np.testing.assert_allclose(b.decode(z)-a.decode(z),np.tile(delta,(24,1)),atol=2e-14,rtol=0)
    np.testing.assert_allclose(b.evaluate(z,[.2,1.,4.],2),a.evaluate(z,[.2,1.,4.],2),atol=2e-12,rtol=0)


def test_new_origin_cannot_silently_fill_missing_initial_state():
    with pytest.raises(ValueError,match='complete initial position'):
        BoundaryCodec(8,2,2.,[Boundary(0,0,(0,),(1.,))],origin='initial_position')


def test_experimental_origin_weights_cannot_enter_unmigrated_main(tmp_path,monkeypatch):
    from ffgd import api
    monkeypatch.setattr(api,'load_model',lambda *args:(None,{'dataset_manifest':{'boundary_origin':'initial_position'}},{}))
    with pytest.raises(ValueError,match='BOUNDARY_ORIGIN_NOT_ENABLED_IN_MAIN'):
        api.plan(None,None,None,'unused',api.PlanningBudget(),{},tmp_path)
