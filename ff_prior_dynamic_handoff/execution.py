"""Unchanged feedback controller with observational full-state/geometry recording."""
from ff_prior_waypoint_domain.adapter import Environment as Controller
from ff_prior_dynamic_reference.execution import RecordedCollision,integration,evaluate

class Environment(Controller):
    def __init__(self,robot,moving,seed):
        self.integration_history=[]
        self.filter_history=[]
        super().__init__(robot,moving,seed)

    def velocity_filter(self,desired,g,drift,hessian=None):
        result=super().velocity_filter(desired,g,drift,hessian)
        self.filter_history.append(dict(time=float(self.data.time),desired=desired.copy(),
                                        filtered=result.copy(),delta=float(self.last_filter_delta)))
        return result

    def record(self):
        if not isinstance(self.collision,RecordedCollision):
            self.collision=RecordedCollision(self.collision)
        try:
            super().record()
        finally:
            if len(self.history['time'])>len(self.integration_history):
                self.integration_history.append(integration(self.robot,self.data))

def task_identity(env,scene,seed):
    return dict(initial=env.initial.values,initial_id=env.initial.state_id,
                waypoints=env.path.to_dict(),grasp_local=env.grasp_local,
                target_velocity=env.data.qvel[env.tv],seed=seed,scene=scene)
