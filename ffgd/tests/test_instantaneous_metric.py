"""Native local-map checks; not a claim of full-path derivative equivalence."""
import numpy as np
import mujoco
import torch
import pytest
from scipy.spatial.transform import Rotation
from ffgd.robot import Robot
from ffgd.contracts import TaskSpec, PoseConstraint, PoseInterval
from ffgd.api import boundary_codec
from ffgd.guidance_metric import instantaneous_tip_jacobians, instantaneous_task_metric


@pytest.mark.parametrize('nonzero',[False,True])
def test_instantaneous_map_preserves_local_base_momentum_and_matches_pose_fd(nonzero):
    r=Robot();d=r.new_data()
    if nonzero:
        d.qpos[r.qpos_ids]+=r.B@np.random.default_rng(701).normal(0.,.02,17)
        d.qpos[r.base_qpos[3:]]=Rotation.from_euler('xyz',[.2,-.1,.3]).as_quat()[[3,0,1,2]]
    saved=d.qpos.copy();J,G=instantaneous_tip_jacobians(r,d)
    np.testing.assert_array_equal(d.qpos,saved)
    np.testing.assert_allclose(r.mass(d)[r.base_dofs]@G,0.,atol=2e-13)
    direction=np.random.default_rng(706).normal(size=17);epsilon=1e-6;observed=[]
    for sign in (-1,1):
        probe=mujoco.MjData(r.model);probe.qpos[:]=saved
        mujoco.mj_integratePos(r.model,probe.qpos,G@direction,sign*epsilon)
        mujoco.mj_kinematics(r.model,probe)
        P=probe.xpos[r.tips].copy();R=probe.xmat[r.tips].reshape(2,3,3).copy()
        P[0]+=R[0]@np.array([.0475,0,0]);observed.append((P,R))
    for tip in range(2):
        position=(observed[1][0][tip]-observed[0][0][tip])/(2*epsilon)
        rotation=Rotation.from_matrix(observed[1][1][tip]@observed[0][1][tip].T).as_rotvec()/(2*epsilon)
        np.testing.assert_allclose(J[tip]@direction,np.r_[position,rotation],atol=2e-9,rtol=2e-8)


def test_interval_metric_is_psd_on_all_free_coordinates_and_keeps_inputs():
    torch.set_num_threads(1)
    r=Robot();d=r.new_data();snapshot=r.snapshot(d);T=.2
    P=d.xpos[r.tips].copy();R=d.xmat[r.tips].reshape(2,3,3).copy();P[0]+=R[0]@np.array([.0475,0,0])
    task=TaskSpec(T,(PoseConstraint(T,0,P[0],R[0]),),(PoseInterval(0.,T,1,P[1],R[1]),),terminal_policy='free')
    codec=boundary_codec(r,snapshot,T,8,terminal_policy='free');grid=np.array([0.,.1,.2]);xs=np.tile(d.qpos,(3,1));before=xs.copy()
    H,info=instantaneous_task_metric(r,codec,np.ones(codec.free_dim),task,grid,xs)
    assert H.shape==(codec.free_dim,codec.free_dim) and info['full_free_dimension']==codec.free_dim
    np.testing.assert_array_equal(xs,before)
    assert float(torch.linalg.eigvalsh(H).min())>-1e-8
    full=H+torch.eye(codec.free_dim,dtype=torch.float64);g=torch.linspace(-1.,1.,codec.free_dim,dtype=torch.float64)
    direction=torch.linalg.solve(full,g)
    assert float(g@direction)>0 and info['replaces_path_gradient'] is False
    assert len(info['interval_peak_times'])==1 and info['native_pose_states']==3
    zero,_=instantaneous_task_metric(r,codec,np.ones(codec.free_dim),task,grid,xs,task_weight=0.)
    np.testing.assert_array_equal(zero.numpy(),np.zeros_like(zero.numpy()))


def test_active_collision_rows_match_native_spatial_fd():
    from ffgd.contracts import SceneSpec,SphereObstacle
    from ffgd.geometry import Geometry
    from ffgd.convex_guide import ConvexSupportGuide
    from ffgd.guidance_metric import instantaneous_collision_rows
    torch.set_num_threads(1)
    root=Robot();base=root.new_data();body=root.tips[0]
    center=base.xpos[body]+base.xmat[body].reshape(3,3)@np.array([.0475,0,0])+np.array([.001,.003,.005])
    r=Robot(scene=SceneSpec(spheres=(SphereObstacle(center,.02),)));d=r.new_data()
    _,G=instantaneous_tip_jacobians(r,d);geometry=Geometry(r);guide=ConvexSupportGuide(geometry)
    ids,J,_=instantaneous_collision_rows(r,d,geometry,guide,G,.002)
    assert len(ids)>0
    direction=np.random.default_rng(818).normal(size=17);epsilon=1e-7;values=[]
    for sign in (-1,1):
        probe=mujoco.MjData(r.model);probe.qpos[:]=d.qpos
        mujoco.mj_integratePos(r.model,probe.qpos,G@direction,sign*epsilon);mujoco.mj_kinematics(r.model,probe)
        values.append(guide.numpy_gaps(probe.geom_xpos[geometry.ids],probe.geom_xmat[geometry.ids].reshape(-1,3,3))[ids])
    np.testing.assert_allclose(J@direction,(values[1]-values[0])/(2*epsilon),atol=2e-7,rtol=2e-5)
