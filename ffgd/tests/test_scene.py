import numpy as np
import mujoco
from ffgd.robot import Robot
from ffgd.contracts import BoxObstacle,SceneSpec
from ffgd.geometry import Geometry
from ffgd.physics import DifferentiableTree,tensor
from ffgd.convex_guide import ConvexSupportGuide


def test_scene_order_and_full_geometry():
    a=BoxObstacle([2,2,2],[.2,.3,.4],np.eye(3));b=BoxObstacle([-2,2,2],[.1,.2,.3],np.eye(3))
    assert SceneSpec((a,b)).identity==SceneSpec((b,a)).identity
    base=Robot();r=Robot(scene=SceneSpec((a,b)));rr=Robot(scene=SceneSpec((b,a)))
    assert r.model_hash==rr.model_hash
    assert base.physical_hash==r.physical_hash
    assert r.model.nv==base.model.nv and r.model.nu==67
    g=Geometry(r);g0=Geometry(base);assert len(g.ids)==len(g0.ids)+2
    d=r.new_data();tree=DifferentiableTree(r);kin=tree.kinematics(tensor(d.qpos))
    guide=ConvexSupportGuide(g)
    np.testing.assert_allclose(guide.gaps(kin).numpy(),guide.numpy_gaps(d.geom_xpos[g.ids],d.geom_xmat[g.ids].reshape(-1,3,3)),atol=1e-10)


def test_scene_is_physical_guidance_input_not_an_untrained_network_feature():
    from ffgd.conditioning import encode_condition
    from ffgd.contracts import TaskSpec,PoseConstraint
    base=Robot();other=Robot(scene=SceneSpec((BoxObstacle([2,2,2],[.2,.3,.4],np.eye(3)),)))
    task=TaskSpec(4.,(PoseConstraint(4.,0,[1,2,3],np.eye(3)),))
    left=encode_condition(base,base.snapshot(base.new_data()),task,np.zeros(6))
    right=encode_condition(other,other.snapshot(other.new_data()),task,np.zeros(6))
    np.testing.assert_array_equal(left,right)
    assert len(Geometry(other).pairs)>len(Geometry(base).pairs)
