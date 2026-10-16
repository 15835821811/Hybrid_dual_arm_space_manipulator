"""Compare complete task-origin state, preserving all existing transfer gates."""
import numpy as np
from ff_prior.storage import digest
from ff_prior.scenes import SceneSpec
from ff_prior.contracts import assert_task_compatible
from ff_prior_complex.tasks import scene_runtime, transfer_audit
from ff_prior_loop.runtime import at_origin, origin_audit


def at_scene_origin(canonical,canonical_initial,scene,q0):
    """Audit unchanged robot dynamics and full integration state at task q0.

    Added world geometry may contact the unused default-home configuration.
    We do not remove warmstart or any other integration-state field: the parent
    and child states must still agree exactly at the actual planning origin.
    Initial whole-machine nonpenetration is independently checked at q0.
    This proves initialization compatibility only, never trajectory validity.
    """
    scene=scene if isinstance(scene,SceneSpec) else SceneSpec.from_dict(scene)
    if canonical_initial.model_hash!=canonical.identity()['scene_model_hash']:
        raise ValueError('canonical full initial-state identity mismatch')
    q0=np.asarray(q0,dtype=np.float64)
    parent,pi,pc=at_origin(canonical,q0)
    if scene.objects:
        child,_,_=scene_runtime(scene)
        robot,initial,collision=at_origin(child,q0)
        transfer=transfer_audit(parent,pi,robot,initial)
        origin=origin_audit(child,robot,initial)
    else:
        robot,initial,collision=parent,pi,pc
        transfer={'status':'SAME_CANONICAL_MODEL'}
        origin=origin_audit(canonical,robot,initial)
    # The exact equality in transfer_audit includes every integration-state
    # component. Geometry is a separate hard check, never inferred from it.
    geometry=collision.scan(initial.reset(robot),0.)
    if not geometry['passed']:
        raise ValueError('actual task initial state penetrates whole-machine geometry')
    if not np.array_equal(pi.values,initial.values):
        raise ValueError('actual-origin complete integration state differs')
    proof={'scene_transfer':transfer,'origin':origin,'initial_geometry':geometry,
           'comparison_origin':'actual_task_initial_state','complete_state_fields_preserved':True,
           'canonical_input_state_id':canonical_initial.state_id,
           'parent_task_origin_state_id':pi.state_id,'scene_task_origin_state_id':initial.state_id,
           'scope':'initialization only; no trajectory MODEL_VALID or execution grade implied'}
    return robot,initial,collision,proof


class TaskOriginSceneCache:
    """Keep every scene/model/state contract bound to its exact task origin."""
    def __init__(self,canonical,initial,profile):
        self.canonical,self.initial,self.profile=canonical,initial,profile
        self.cache={}

    def get(self,row):
        task=row['task'];scene=SceneSpec.from_dict(row.get('scene',{}))
        key=digest({'initial':task['contract']['initial_state_id'],'robot':task['contract']['robot_hash'],
                    'scene':scene.to_dict(),'q0':task['condition']['initial_features']})
        if key not in self.cache:
            runtime=at_scene_origin(self.canonical,self.initial,scene,task['condition']['initial_features'])
            rr,ii,cc,_=runtime
            assert_task_compatible(task,rr,ii,cc,self.profile)
            self.cache[key]=runtime
        rr,ii,cc,proof=self.cache[key]
        assert_task_compatible(task,rr,ii,cc,self.profile)
        return rr,ii,cc,proof
