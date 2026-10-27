import numpy as np
from ffgd.robot import Robot
from ffgd.propagation import native_momentum,native_subtree_momentum


def test_subtree_equals_independent_body_sum():
    r=Robot();d=r.new_data();rng=np.random.default_rng(78)
    for i in range(32):
        d.qpos[r.qpos_ids]=r.B@(r.spec.planner_zero+rng.normal(0,.1,17))
        d.qvel[:]=rng.normal(0,.1,r.model.nv)
        np.testing.assert_allclose(native_subtree_momentum(r,d),native_momentum(r,d),atol=1e-11,rtol=0)
