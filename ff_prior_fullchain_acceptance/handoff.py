"""A phase switch on the same live simulation object; never sets physical state.

This interface has software checks. A successful actual composed trajectory is
still required before any physical continuity claim is made.
"""
from dataclasses import dataclass
import numpy as np
from ff_prior.storage import digest
from ff_prior.coupled_predictor import pose_errors
from ff_prior_static_goal_seed.tracker import integration_state


def check_boundary(before,after,model_before,model_after,previous_level,actual_pose,next_pose):
    if previous_level!='EXECUTION_VERIFIED':raise ValueError('PREVIOUS_PHASE_NOT_VERIFIED')
    if model_before!=model_after:raise ValueError('MODEL_OR_SCENE_CHANGED')
    a,b=np.asarray(before),np.asarray(after)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('NONFINITE_OR_STATE_SCHEMA_CHANGED')
    if not np.array_equal(a,b):raise ValueError('PHYSICAL_STATE_CHANGED_AT_HANDOFF')
    pos,rot=pose_errors(np.asarray(actual_pose),np.asarray(next_pose))
    if not np.isfinite(pos).all() or not np.isfinite(rot).all() or max(pos)>.005 or max(rot)>np.pi/180:
        raise ValueError('NEXT_TARGET_DISCONTINUITY')
    return dict(full_state_exact=True,state_reset=False,position_m=pos,rotation_rad=rot)


@dataclass
class PhaseSession:
    robot: object
    data: object
    phase: str
    origin_time: float

    def switch(self,name,previous_receipt,next_start_pose,build_policy):
        # The supplied builder must use this live data; a fresh environment is
        # never accepted as a substitute for physical continuity.
        before=integration_state(self.robot,self.data)
        model=digest(self.robot.identity());owner=id(self.data)
        if previous_receipt.get('end_state_hash')!=digest(before):
            raise ValueError('UNBOUND_PREVIOUS_END_STATE')
        if previous_receipt.get('model_hash')!=model:raise ValueError('UNBOUND_PREVIOUS_MODEL')
        actual=self.robot.ee(self.data).copy()
        check_boundary(before,before,model,model,previous_receipt.get('level'),actual,next_start_pose)
        policy=build_policy(self.robot,self.data,float(self.data.time))
        if id(self.data)!=owner or getattr(policy,'data',None) is not self.data:
            raise ValueError('NEW_ENVIRONMENT_OR_DATA_AT_HANDOFF')
        checked=check_boundary(before,integration_state(self.robot,self.data),model,
            digest(self.robot.identity()),previous_receipt.get('level'),actual,next_start_pose)
        self.phase=name;self.origin_time=float(self.data.time)
        return policy,dict(checked,phase=name,absolute_time=self.origin_time,
            end_state_hash=digest(before),new_state_writes=0)

    def local_time(self):
        value=float(self.data.time)-self.origin_time
        if value<0:raise ValueError('CLOCK_RESET')
        return value
