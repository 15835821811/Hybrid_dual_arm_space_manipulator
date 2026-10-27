"""All-state acceptance regressions; synthetic records are not execution proof."""
import numpy as np
from ffgd.robot import Robot
from ffgd.execution_acceptance import inspect_state_bounds


def test_final_state_limits_and_unprojected_velocity_are_checked_readonly():
    r=Robot();d=r.new_data();initial=r.snapshot(d)
    d.qpos[r.qpos_ids]=r.B@np.r_[1.01,np.zeros(9),r.spec.planner_zero[10:]]
    # A high continuum velocity component cannot be hidden by averaging q67.
    envelope=abs(r.B)@r.spec.planner_velocity_limits
    d.qvel[r.dof_ids[0]]=envelope[0]*2.
    final=r.snapshot(d)
    bundle=dict(states=np.stack([initial.state,final.state]),mask=initial.mask,
                ctrl=np.zeros((1,r.model.nu)),qfrc_applied=np.zeros((1,r.model.nv)),xfrc_applied=np.zeros((1,r.model.nbody,6)))
    before={k:v.copy() for k,v in bundle.items() if isinstance(v,np.ndarray)}
    report=inspect_state_bounds(r,bundle)
    assert report['includes_final_state'] and report['states_checked']==2
    assert not report['gates']['actual_q17']
    assert not report['gates']['mapped_q67_velocity_ratio']
    assert report['manifold_residual_max']['velocity_rad_s']>0
    assert report['reference_tracking_max'] is None
    for key,value in before.items():np.testing.assert_array_equal(bundle[key],value)
