import numpy as np
import pytest
from ffgd.contracts import ForcePulse,TaskSpec,PoseConstraint
from ffgd.forcing import target_force,target_displacement,generalized_forces,apply_native_target_force,target_indices
from ffgd.robot import Robot


def test_force_pulse_units_integrals_and_canonical_contract():
    a=ForcePulse(0.,2.,[0,0,.04]);b=ForcePulse(3.,4.,[.1,0,0])
    np.testing.assert_allclose(target_force((a,),[0,1,2,3]),[[0,0,0],[0,0,.04],[0,0,0],[0,0,0]],atol=1e-15)
    r=Robot();body,_=target_indices(r);mass=r.model.body_mass[body]
    np.testing.assert_allclose(target_displacement(r,(a,),2),[0,0,.04/mass],atol=1e-14)
    np.testing.assert_allclose(target_displacement(r,(a,),29),[0,0,1.12/mass],atol=1e-14)
    pose=PoseConstraint(4.,0,[0,0,0],np.eye(3));task=TaskSpec(4.,(pose,),target_force_pulses=(b,a))
    assert task.target_force_pulses==(a,b)
    with pytest.raises(ValueError):TaskSpec(1.,(PoseConstraint(1.,0,[0,0,0],np.eye(3)),),target_force_pulses=(a,))


def test_native_force_is_input_only_and_has_generalized_equivalence():
    import mujoco
    r=Robot();d=r.new_data();pulses=(ForcePulse(.03,.17,[.012,-.021,.034]),)
    before_x=d.qpos.copy();before_v=d.qvel.copy();body,adr=target_indices(r)
    apply_native_target_force(r,d,pulses,.1)
    np.testing.assert_array_equal(d.qpos,before_x);np.testing.assert_array_equal(d.qvel,before_v)
    result=np.zeros(r.model.nv)
    mujoco.mj_applyFT(r.model,d,d.xfrc_applied[body,:3],np.zeros(3),d.xipos[body],body,result)
    np.testing.assert_allclose(result,generalized_forces(r,pulses,.1),atol=1e-14)
    np.testing.assert_array_equal(result[r.dof_ids],np.zeros(67))
    np.testing.assert_array_equal(result[r.base_dofs],np.zeros(6))


def test_native_force_trace_replays_and_impulse_matches_public_pulse(tmp_path):
    from ffgd.contracts import Candidate
    from ffgd.execution import track,replay
    r=Robot();snapshot=r.snapshot(r.new_data());d=r.restore(snapshot)
    pulses=(ForcePulse(0.,.1,[0,0,.04]),);duration=.2
    candidate=Candidate(np.tile(r.Bplus@d.qpos[r.qpos_ids],(24,1)),duration,0)
    bundle,report=track(r,snapshot,candidate,tmp_path,dict(physical_geometry_clearance_m=0.),target_force_pulses=pulses)
    assert report['status']=='FINISHED' and report['state_writes_after_restore']==0
    body,adr=target_indices(r);dt=float(r.model.opt.timestep)
    np.testing.assert_allclose(bundle['xfrc_applied'][:,body,:3],target_force(pulses,(np.arange(100)+.5)*dt),atol=1e-15)
    impulse=bundle['xfrc_applied'][:,body,:3].sum(0)*dt
    np.testing.assert_allclose(impulse,[0,0,.002],atol=1e-14)
    from ffgd.contracts import Snapshot
    final=r.restore(Snapshot(r.model_hash,snapshot.mask,bundle['states'][-1]))
    np.testing.assert_allclose(final.qvel[adr:adr+3]-d.qvel[adr:adr+3],impulse/r.model.body_mass[body],atol=1e-12)
    assert replay(r,bundle,snapshot)['max_state_abs']==0
    from ffgd.forcing import inspect_target_inputs
    report=inspect_target_inputs(r,bundle,pulses)
    assert report['passed'],report
