import numpy as np
import mujoco
from ffgd.robot import Robot


def test_mapping_actual_limits_snapshot_and_forward_response():
    r=Robot();m=r.model;d=r.new_data()
    # Independent coordinate convention: alternating Y/X modules, six per section.
    expected=np.zeros((67,17))
    for section in range(5):
        for k,column in enumerate([1,0,0,1,1,0,0,1,1,0,0,1]):expected[12*section+k,2*section+column]=1/6
    expected[60:,10:]=np.eye(7)
    np.testing.assert_array_equal(r.B,expected)
    np.testing.assert_allclose(r.Bplus@r.B,np.eye(17),atol=1e-14)
    np.testing.assert_array_equal(m.actuator_trnid[r.actuator_ids,0],r.joints)
    np.testing.assert_array_equal(m.actuator_gear[r.actuator_ids,0],np.ones(67))
    np.testing.assert_array_equal(m.actuator_ctrlrange[r.actuator_ids,1],r.spec.torque_limits)
    assert m.jnt_type[r.base_joint]==mujoco.mjtJoint.mjJNT_FREE
    rng=np.random.default_rng(15)
    d.qvel[:]=rng.normal(0,.01,m.nv);d.ctrl[:]=rng.normal(0,.01,m.nu)
    d.qacc_warmstart[:]=rng.normal(0,.1,m.nv);d.time=1.2
    snap=r.snapshot(d);e=r.restore(snap)
    np.testing.assert_array_equal(r.snapshot(e).state,snap.state)
    mujoco.mj_step(m,d);mujoco.mj_step(m,e)
    np.testing.assert_array_equal(r.snapshot(e).state,r.snapshot(d).state)
    # Instantaneous forward response includes contacts through native force ledger.
    mujoco.mj_forward(m,d)
    force=d.qfrc_actuator+d.qfrc_passive+d.qfrc_applied+d.qfrc_constraint-d.qfrc_bias
    np.testing.assert_allclose(r.mass(d)@d.qacc,force,atol=1e-8,rtol=1e-8)
