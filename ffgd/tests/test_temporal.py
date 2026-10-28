import numpy as np
import mujoco
import pytest
from ffgd.contracts import PoseInterval,TaskSpec
from ffgd.temporal import inspect_intervals,condition_poses,target_body
from ffgd.robot import Robot
from ffgd.api import boundary_codec


@pytest.fixture(scope='module')
def robot():return Robot()


def test_interval_checks_interior_and_complete_time_range(robot):
    r=robot;d=r.new_data();body=r.tips[1];R=d.xmat[body].reshape(3,3).copy();P=d.xpos[body].copy()
    task=TaskSpec(.2,(),(PoseInterval(0.,.2,1,P,R),))
    xs=np.tile(d.qpos,(3,1));xs[1,r.base_qpos[0]]+=.01
    report=inspect_intervals(r,task,[0.,.1,.2],xs)[0]
    import json
    json.dumps(report,allow_nan=False)
    assert not report['passed'] and report['worst_position_time']==.1 and report['samples']==3
    assert not inspect_intervals(r,task,[0.,.1],xs[:2])[0]['passed']


def test_relative_tracking_uses_actual_target_pose_and_predictive_tokens(robot):
    r=robot;d=r.new_data();b=target_body(r);tip=r.tips[1]
    R=d.xmat[b].reshape(3,3);Rt=d.xmat[tip].reshape(3,3)
    offset=R.T@(d.xpos[tip]-d.xpos[b]);relative=R.T@Rt
    interval=PoseInterval(0.,.2,1,offset,relative,frame='target')
    task=TaskSpec(.2,(),(interval,));target=r.model.joint(r.spec.target_free_joint_name);qa=int(target.qposadr[0]);va=int(target.dofadr[0])
    d.qvel[va]=.1;before=r.snapshot(d).state.copy();poses=condition_poses(r,r.snapshot(d),task)
    np.testing.assert_allclose(poses[-1].position-poses[0].position,[.02,0,0],atol=1e-12)
    np.testing.assert_array_equal(r.snapshot(d).state,before)
    xs=np.tile(d.qpos,(3,1));xs[:,qa]+=np.array([0,.01,.02])
    report=inspect_intervals(r,task,[0.,.1,.2],xs)[0]
    assert not report['passed'];np.testing.assert_allclose(report['maximum_position_m'],.02,atol=1e-12)


def test_control_reference_c2_does_not_overwrite_actual_state(robot):
    r=robot;d=r.new_data();snapshot=r.snapshot(d);original=snapshot.state.copy()
    reference=np.stack((r.Bplus@d.qpos[r.qpos_ids],np.zeros(17),np.zeros(17)))
    reference[0,0]+=.001;reference[1,1]=.002;reference[2,2]=.003
    codec=boundary_codec(r,snapshot,.8,origin='initial_position',reference_start=reference)
    z=np.random.default_rng(31).normal(size=codec.free_dim)*1e-4
    for k in (0,1,2):np.testing.assert_allclose(codec.evaluate(z,0,k),reference[k],atol=1e-10)
    np.testing.assert_array_equal(snapshot.state,original)
    assert np.max(abs(codec.evaluate(z,0)-r.Bplus@r.restore(snapshot).qpos[r.qpos_ids]))>0


def test_irregular_timestamp_slerp_and_readonly(robot):
    from ffgd.temporal import interpolate_qpos
    from scipy.spatial.transform import Rotation
    r=robot;d=r.new_data();xs=np.tile(d.qpos,(2,1));adr=r.base_qpos[3]
    xs[0,adr:adr+4]=[1,0,0,0]
    xs[1,adr:adr+4]=-Rotation.from_rotvec([0,0,.1]).as_quat()[[3,0,1,2]]
    xs[1,r.base_qpos[0]]+=.002;before=xs.copy()
    x,bracket=interpolate_qpos(r,np.array([0.,.002]),xs,.00073)
    np.testing.assert_allclose(Rotation.from_quat(x[adr:adr+4][[1,2,3,0]]).as_rotvec(),[0,0,.0365],atol=1e-12)
    assert bracket==.002
    np.testing.assert_array_equal(before,xs)


def test_interval_metadata_is_validated_and_canonical():
    a=PoseInterval(.02,.1,0,[0,0,0],np.eye(3));b=PoseInterval(0,.2,1,[1,0,0],np.eye(3),frame='target')
    assert TaskSpec(.2,(),(a,b)).intervals[0] is b
    with pytest.raises(ValueError):TaskSpec(.1,(),(b,))
    with pytest.raises(ValueError):PoseInterval(0,.1,0,[0,0,0],np.eye(3),frame='scenario_specific')
