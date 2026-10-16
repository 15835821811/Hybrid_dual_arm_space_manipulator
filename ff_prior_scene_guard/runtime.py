"""Reject colliding initial scenes explicitly; retain every full-state check."""
import numpy as np
from ff_prior.scenes import SceneSpec
from ff_prior_complex.tasks import scene_runtime,transfer_audit
from ff_prior_loop.runtime import at_origin,origin_audit


class InitialGeometryRejected(ValueError):
    def __init__(self,geometry):
        super().__init__('actual task initial state penetrates whole-machine geometry')
        self.geometry=geometry


def at_scene_origin(canonical,canonical_initial,scene,q0):
    """Only check order changes; colliding inputs never reach state transfer.

    This version is deliberately outside frozen running source. A successful
    input still passes exact integration-state, mapping, model/geometry,
    dynamics, gravity and solver transfer checks, including warmstart.
    """
    scene=scene if isinstance(scene,SceneSpec) else SceneSpec.from_dict(scene)
    if canonical_initial.model_hash!=canonical.identity()['scene_model_hash']:
        raise ValueError('canonical full initial-state identity mismatch')
    parent,pi,pc=at_origin(canonical,np.asarray(q0,dtype=np.float64))
    if scene.objects:
        base,_,_=scene_runtime(scene);robot,initial,collision=at_origin(base,q0)
    else:base=canonical;robot,initial,collision=parent,pi,pc
    geometry=collision.scan(initial.reset(robot),0.)
    if not geometry['passed']:raise InitialGeometryRejected(geometry)
    transfer=transfer_audit(parent,pi,robot,initial) if scene.objects else {'status':'SAME_CANONICAL_MODEL'}
    origin=origin_audit(base,robot,initial)
    if not np.array_equal(pi.values,initial.values):raise ValueError('actual-origin complete integration state differs')
    proof={'scene_transfer':transfer,'origin':origin,'initial_geometry':geometry,
        'comparison_origin':'actual_task_initial_state','complete_state_fields_preserved':True,
        'canonical_input_state_id':canonical_initial.state_id,
        'parent_task_origin_state_id':pi.state_id,'scene_task_origin_state_id':initial.state_id,
        'initial_geometry_checked_before_transfer':True,
        'scope':'initialization only; no trajectory validation or execution grade'}
    return robot,initial,collision,proof
