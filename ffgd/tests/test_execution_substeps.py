import numpy as np
import pytest
from ffgd.robot import Robot
from ffgd.contracts import Candidate
import ffgd.execution as execution


def test_refinement_preserves_full_state_and_rejects_other_model_changes():
    r=Robot();d=r.new_data();d.qvel[:]=.001;d.ctrl[:]=.002;d.qacc_warmstart[:]=.003;d.time=.7
    snap=r.snapshot(d);rr,ss=execution.refine_physics(r,snap,2)
    assert rr.model.opt.timestep==.001 and rr.physical_hash==r.physical_hash
    np.testing.assert_array_equal(rr.snapshot(rr.restore(ss)).state,snap.state)
    assert execution.refine_physics(r,snap,1)==(r,snap)
    r.model.body_mass[r.base_body]*=1.01;r._hash=None;changed=r.snapshot(d)
    with pytest.raises(ValueError,match='BEYOND_TIMESTEP'):execution.refine_physics(r,changed,2)


def test_native_substeps_hold_control_and_replay_without_state_injection(tmp_path,monkeypatch):
    r=Robot();initial=r.snapshot(r.new_data());r,initial=execution.refine_physics(r,initial,2)
    C=np.tile(r.spec.planner_zero,(24,1));C[3:,0]+=np.linspace(0,.0001,21)
    c=Candidate(C,.2,1);calls=[];original=execution.command
    def counted(*args):
        calls.append(float(args[1].time));return original(*args)
    monkeypatch.setattr(execution,'command',counted)
    bundle,report=execution.track(r,initial,c,tmp_path,dict(physical_geometry_clearance_m=0.),control_substeps=2)
    assert report['status']=='FINISHED' and report['actual_steps']==200
    assert len(calls)==report['controller_updates']==100
    np.testing.assert_allclose(np.diff(calls),.002,atol=1e-15)
    np.testing.assert_array_equal(bundle['ctrl'][::2],bundle['ctrl'][1::2])
    assert np.max(abs(bundle['ctrl']))>0
    assert execution.replay(r,bundle,initial)['max_state_abs']==0
    assert report['state_writes_after_restore']==0 and report['control_period']==.002
