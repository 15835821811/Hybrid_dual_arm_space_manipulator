import numpy as np
import mujoco
import torch
from types import SimpleNamespace
from ffgd.contracts import SphereObstacle, SceneSpec
from ffgd.convex_guide import ConvexSupportGuide
from ffgd.physics import DifferentiableTree, tensor
from ffgd.robot import Robot
from ffgd.geometry import Geometry


def test_sphere_support_is_analytic_with_correct_gradient():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom type="sphere" size=".2"/></body><body pos=".51 .03 .02"><geom type="sphere" size=".3"/></body></worldbody></mujoco>')
    geom=SimpleNamespace(robot=SimpleNamespace(model=model),ids=np.array([0,1]),bodies=np.array([1,2]),rot=torch.eye(3,dtype=torch.float64).repeat(2,1,1),geom_pos=torch.zeros((2,3),dtype=torch.float64),pairs=np.array([[0,1]]))
    guide=ConvexSupportGuide(geom)
    pos=tensor([[0,0,0],[0,0,0],[.51,.03,.02]]).requires_grad_()
    rotations=torch.eye(3,dtype=torch.float64).repeat(3,1,1)
    gap=guide.gaps(dict(position=pos,rotation=rotations),cap=.1)[0]
    expected=np.linalg.norm([.51,.03,.02])-.5
    np.testing.assert_allclose(float(gap.detach()),expected,atol=1e-14)
    gradient=torch.autograd.grad(gap,pos)[0].numpy()
    np.testing.assert_allclose(gradient[2],np.array([.51,.03,.02])/np.linalg.norm([.51,.03,.02]),atol=1e-12)
    np.testing.assert_allclose(guide.numpy_gaps(pos.detach().numpy()[[1,2]],rotations.numpy()[[1,2]],cap=.1),[expected],atol=1e-14)


def test_sphere_robot_scene_native_support_and_order():
    torch.set_num_threads(1)
    base=Robot();d=base.new_data();mujoco.mj_forward(base.model,d)
    point=d.xpos[base.tips[0]].copy()
    a=SphereObstacle(point+[.015,.02,.01],.025);b=SphereObstacle([3,2,2],.04)
    r=Robot(scene=SceneSpec(spheres=(a,b)));other=Robot(scene=SceneSpec(spheres=(b,a)))
    assert r.model_hash==other.model_hash and r.physical_hash==base.physical_hash
    g=Geometry(r);guide=ConvexSupportGuide(g);d=r.new_data();mujoco.mj_forward(r.model,d)
    tree=DifferentiableTree(r);x=tensor(d.qpos).requires_grad_()
    gap=guide.gaps(tree.kinematics(x));native=guide.numpy_gaps(d.geom_xpos[g.ids],d.geom_xmat[g.ids].reshape(-1,3,3))
    np.testing.assert_allclose(gap.detach().numpy(),native,atol=1e-10)
    pair=[i for i,(a,b) in enumerate(g.pairs) if max(g.ids[a],g.ids[b])>=base.model.ngeom and native[i]<.019]
    assert pair, 'must exercise active sphere/robot support, not only broadphase'
    i=pair[int(np.argmin(native[pair]))]
    grad=torch.autograd.grad(gap[i],x)[0].numpy()
    direction=np.zeros(r.model.nq);direction[r.base_qpos[:3]]=[.3,-.4,.5]
    values=[]
    for sign in (-1,1):
        d.qpos[:]=x.detach().numpy()+sign*1e-6*direction;mujoco.mj_kinematics(r.model,d)
        values.append(guide.numpy_gaps(d.geom_xpos[g.ids],d.geom_xmat[g.ids].reshape(-1,3,3))[i])
    np.testing.assert_allclose(grad@direction,(values[1]-values[0])/2e-6,rtol=1e-5,atol=1e-7)
    d.qpos[:]=x.detach().numpy();mujoco.mj_kinematics(r.model,d)
    exact=g.native_distances(d.qpos)
    # Support is a conservative separating-distance lower bound.
    assert np.all(native[pair]<=exact[pair]+1e-7)
