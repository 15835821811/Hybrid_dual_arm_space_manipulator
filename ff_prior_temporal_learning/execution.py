"""Common public-world feedback applied equally to both learned sequence proposals."""
import numpy as np
from ff_prior_dynamic_handoff.execution import Environment as ObservedController
from .protocol import P

def guided_velocity(env,spline):
    g,drift=env.generalized_map();j,offset=env.task_jacobian(g,drift)
    q,_=env.state();preferred=spline(env.data.time,1)+P['world_feedback_tracking_gain']*(spline(env.data.time)-q)
    weights=np.repeat([1.,2.,1.4,2.4],3);jw=weights[:,None]*j
    h=jw.T@jw+P['world_feedback_regularizer']*np.eye(17)
    world=weights*(env.desired_velocity(True)-offset)
    correction=np.linalg.solve(h,jw.T@(world-jw@preferred))
    desired=preferred+correction
    filtered=env.velocity_filter(desired,g,drift,h)
    record=dict(time=float(env.data.time),sequence_preferred=preferred,world_correction=correction,
                requested=desired,filtered=filtered,world_correction_norm=float(np.linalg.norm(correction)),
                constraint_correction_norm=float(np.linalg.norm(filtered-desired)))
    return filtered,record
