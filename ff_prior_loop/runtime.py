"""Explicit planning-origin view; physical home and compiled model are unchanged."""
import copy
import numpy as np
from ff_prior.initial_state import InitialState
from ff_prior.geometry import CollisionBackend
from ff_prior.storage import digest
from ff_prior_complex.tasks import task_for


class PlanningOriginSpec:
    """Compatibility view for legacy validators whose boundary slot is planner_zero.

    This changes neither the physical spec nor a model parameter. The task's
    origin is separately captured in the complete integration-state snapshot.
    """
    def __init__(self,physical_spec,q0):
        self.physical_spec=physical_spec
        self.planner_zero=np.asarray(q0,dtype=np.float64).copy()
        self.planner_zero.setflags(write=False)
    def __getattr__(self,name): return getattr(self.physical_spec,name)


def at_origin(canonical,q0):
    q0=np.asarray(q0,dtype=np.float64)
    if q0.shape!=(17,) or not np.isfinite(q0).all() or np.any(q0<canonical.lower) or np.any(q0>canonical.upper):
        raise ValueError('invalid planning initial configuration')
    canonical.identity()
    robot=copy.copy(canonical); robot.calls=dict.fromkeys(canonical.calls,0)
    physical=canonical.spec.physical_spec if isinstance(canonical.spec,PlanningOriginSpec) else canonical.spec
    robot.spec=PlanningOriginSpec(physical,q0)
    data=robot.new_data(); initial=InitialState.capture(robot,data)
    assert np.array_equal(data.qpos[robot.qpos_ids],robot.B@q0)
    assert not np.any(data.qvel) and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
    return robot,initial,CollisionBackend(robot)


def origin_audit(canonical,robot,initial):
    physical=robot.spec.physical_spec
    data=initial.reset(robot)
    checks={'same_compiled_model_object':canonical.model is robot.model,
        'same_physical_identity':canonical.identity()['scene_model_hash']==initial.model_hash,
        'same_physical_spec':canonical.spec.identity().to_dict()==physical.identity().to_dict(),
        'mapping_unchanged':np.array_equal(canonical.B,robot.B),
        'q0_matches_complete_state':np.array_equal(data.qpos[robot.qpos_ids],robot.B@robot.spec.planner_zero),
        'stationary':not np.any(data.qvel),'no_external_force':not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)}
    if not all(checks.values()): raise ValueError('initial-origin audit failed')
    return {'status':'PASS','checks':checks,'physical_home':physical.planner_zero,
        'planning_origin':robot.spec.planner_zero,'initial_state_id':initial.state_id,
        'complete_state_hash':digest(initial.values),'scope':'stationary zero-momentum origins; static scenes only'}
