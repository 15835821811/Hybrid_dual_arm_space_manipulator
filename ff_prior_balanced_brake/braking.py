"""Conditional, exact discrete joint braking bounds; no robot evaluation.

Let T=.02, h=.002, N=10 and A=a*T. A command ramp from v to u
integrates the reference by alpha*v + beta*u, where alpha=.009 and
beta=.011. These are right-endpoint sums, not the continuous trapezoid.

After the candidate tick, an emergency stopping policy is
v_j = sign(u)*max(abs(u)-j*A, 0). At future tick j its reference is
r_j = r + alpha*v + beta*u + sum(i=1..j)(alpha*v_(i-1)+beta*v_i).
With the existing tracking gate |q-r| <= E=.02 and the existing barrier
coefficient k=4, the next emergency command satisfies BOTH measured-state
barriers for all admissible errors iff
    lower+.04+E <= r_j + v_(j+1)/k <= upper-.04-E.
All future ticks through stop are tested. Velocity and acceleration bounds
hold by construction. Future reference substeps lie between the monotone
braking endpoints, and the final stop is inside the reference bounds.

Every expression r_j+v_(j+1)/k is nondecreasing in candidate u. Thus their
minimum and maximum are nondecreasing, and bisection returns one interval
to intersect with ALL unchanged current-tick velocity_bounds constraints.
This is conditional reference/joint braking viability, not a guarantee of
future tracking error, geometry, task error, torque, or full robot feasibility.
No state, reference, waypoint, timing, or acceptance threshold is modified.
"""
import numpy as np
from ff_prior_waypoint_domain.adapter import velocity_bounds as original_velocity_bounds


TICK_S = .02
SUBSTEP_S = .002
SUBSTEPS = 10
ALPHA = SUBSTEP_S*(SUBSTEPS-1)/2
BETA = SUBSTEP_S*(SUBSTEPS+1)/2
DOMAIN_INSET_RAD = .04
BARRIER_GAIN = 4.
TRACKING_ENVELOPE_RAD = .02  # Existing acceptance threshold, not tuned margin.
BISECTION_STEPS = 64


def stopping_distance(velocity, acceleration):
    """Exact signed reference displacement after this velocity starts braking.

For speed s>0 and m=ceil(s/A), the unsigned sum is
 [m*alpha+(m-1)*beta]*s - T*A*m*(m-1)/2.
This excludes the current candidate ramp from previous to velocity.
"""
    velocity,acceleration=np.broadcast_arrays(np.asarray(velocity,dtype=float),
                                             np.asarray(acceleration,dtype=float))
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all() or np.any(acceleration<=0):
        raise ValueError('BRAKING_REQUIRES_FINITE_VELOCITY_AND_POSITIVE_ACCELERATION')
    speed=np.abs(velocity);step=acceleration*TICK_S
    ticks=np.ceil(speed/step)
    distance=(ticks*ALPHA+(ticks-1)*BETA)*speed-TICK_S*step*ticks*(ticks-1)/2
    return np.sign(velocity)*np.maximum(distance,0.)


def emergency_profile(reference, previous, candidate, acceleration, max_ticks):
    """Pure affine reference arithmetic, not a dynamics/robot integration.

Rows j=0..max_ticks are future control instants after the candidate tick.
The extra velocity row supplies the next command at every tested instant.
"""
    reference,previous,candidate,acceleration=np.broadcast_arrays(
        *[np.asarray(x,dtype=float) for x in (reference,previous,candidate,acceleration)])
    steps=np.arange(max_ticks+2,dtype=float)[:,None]
    velocity=np.sign(candidate)*np.maximum(np.abs(candidate)-steps*acceleration*TICK_S,0.)
    delta=ALPHA*velocity[:-1]+BETA*velocity[1:]
    offset=np.vstack((np.zeros_like(candidate),np.cumsum(delta[:max_ticks],axis=0)))
    position=reference+ALPHA*previous+BETA*candidate+offset
    barrier_coordinate=position+velocity[1:]/BARRIER_GAIN
    return position,velocity,barrier_coordinate


def _envelope(reference,previous,candidate,acceleration,max_ticks):
    position,velocity,barrier=emergency_profile(reference,previous,candidate,acceleration,max_ticks)
    return (np.min(barrier,axis=0),np.max(barrier,axis=0),
            np.argmin(barrier,axis=0),np.argmax(barrier,axis=0),position[-1])


def braking_velocity_bounds(q,reference,previous,lower,upper,vmax,acc):
    """Return tightened lo,hi and all numerical terms for the audit record.

An empty intersection remains empty. In particular this cannot repair an
already infeasible original velocity box at the failed terminal state.
Geometry constraints must still intersect the returned box in the existing QP.
"""
    values=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in
        (q,reference,previous,lower,upper,vmax,acc)])
    q,reference,previous,lower,upper,vmax,acc=[x.copy() for x in values]
    if q.ndim!=1 or not q.size or not all(np.isfinite(x).all() for x in values):
        raise ValueError('BRAKING_REQUIRES_FINITE_ONE_DIMENSIONAL_ARRAYS')
    if np.any(lower>=upper) or np.any(vmax<=0) or np.any(acc<=0):
        raise ValueError('BRAKING_REQUIRES_VALID_DOMAINS_AND_POSITIVE_LIMITS')
    original_lo,original_hi=original_velocity_bounds(q,reference,previous,lower,upper,vmax,acc)
    L,U=lower+DOMAIN_INSET_RAD,upper-DOMAIN_INSET_RAD
    robust_L,robust_U=L+TRACKING_ENVELOPE_RAD,U-TRACKING_ENVELOPE_RAD
    # One extra zero-velocity row avoids relying on floating-point equality
    # when a speed is an exact multiple of a*T. No physical horizon is changed.
    max_ticks=int(np.ceil(np.max(vmax/(acc*TICK_S))))+1

    def envelope(candidate):return _envelope(reference,previous,candidate,acc,max_ticks)
    low_end=envelope(-vmax);high_end=envelope(vmax)
    left=-vmax.copy();right=vmax.copy()
    for _ in range(BISECTION_STEPS):
        mid=left+(right-left)/2
        ok=envelope(mid)[0]>=robust_L
        right=np.where(ok,mid,right);left=np.where(ok,left,mid)
    brake_lo=np.where(low_end[0]>=robust_L,-vmax,
        np.where(high_end[0]<robust_L,vmax+1.,right))

    left=-vmax.copy();right=vmax.copy()
    for _ in range(BISECTION_STEPS):
        mid=left+(right-left)/2
        ok=envelope(mid)[1]<=robust_U
        left=np.where(ok,mid,left);right=np.where(ok,right,mid)
    brake_hi=np.where(high_end[1]<=robust_U,vmax,
        np.where(low_end[1]>robust_U,-vmax-1.,left))
    lo=np.maximum(original_lo,brake_lo);hi=np.minimum(original_hi,brake_hi)
    tracking_error=q-reference
    tracking_ok=np.abs(tracking_error)<=TRACKING_ENVELOPE_RAD
    # Fail closed on an already violated prerequisite, using a finite empty
    # interval so the caller can save the diagnostic before rejecting its QP.
    lo=np.where(tracking_ok,lo,np.maximum(lo,hi+1.))

    k=np.arange(1,SUBSTEPS+1,dtype=float)[:,None]
    weights=SUBSTEP_S*k*(k+1)/(2*SUBSTEPS)
    offsets=reference+SUBSTEP_S*(k-k*(k+1)/(2*SUBSTEPS))*previous
    endpoint_details={}
    for name,candidate in (('braking_lower',brake_lo),('braking_upper',brake_hi)):
        in_range=np.abs(candidate)<=vmax
        # Sentinel empty bounds are not policy commands. Evaluate the nearest
        # search endpoint only for their diagnostic; never apply it to a state.
        probe=np.minimum(np.maximum(candidate,-vmax),vmax)
        minimum,maximum,jmin,jmax,stop=envelope(probe)
        endpoint_details[name]=dict(candidate=candidate,within_search_range=in_range,
            minimum_barrier_coordinate=minimum,maximum_barrier_coordinate=maximum,
            lower_limiting_future_tick=jmin,upper_limiting_future_tick=jmax,
            stop_reference=stop,braking_displacement=stopping_distance(probe,acc),
            stopping_ticks=np.ceil(np.abs(probe)/(acc*TICK_S)).astype(int))
    diagnostic=dict(schema='exact_discrete_joint_braking_intersection_v1',
        q=q,reference=reference,previous=previous,lower=lower,upper=upper,vmax=vmax,acceleration=acc,
        tick_s=TICK_S,substep_s=SUBSTEP_S,substeps=SUBSTEPS,alpha=ALPHA,beta=BETA,
        barrier_gain=BARRIER_GAIN,domain_inset_rad=DOMAIN_INSET_RAD,
        tracking_envelope_rad=TRACKING_ENVELOPE_RAD,tracking_error=tracking_error,
        current_tracking_envelope_satisfied=tracking_ok,
        original_components=dict(velocity_lower=-vmax,velocity_upper=vmax,
            measured_barrier_lower=BARRIER_GAIN*(L-q),measured_barrier_upper=BARRIER_GAIN*(U-q),
            acceleration_lower=previous-acc*TICK_S,acceleration_upper=previous+acc*TICK_S,
            reference_substep_lower=(L-offsets)/weights,reference_substep_upper=(U-offsets)/weights),
        original_lower=original_lo,original_upper=original_hi,
        robust_future_barrier_lower=robust_L,robust_future_barrier_upper=robust_U,
        braking_lower=brake_lo,braking_upper=brake_hi,final_lower=lo,final_upper=hi,
        empty_intersection_indices=np.flatnonzero(lo>hi),
        tightened_lower=lo>original_lo,tightened_upper=hi<original_hi,
        future_ticks_checked=max_ticks+1,bisection_steps=BISECTION_STEPS,
        endpoint_details=endpoint_details,
        certificate_scope='conditional joint/reference emergency-braking feasibility while future abs(q-reference)<=existing .02 tracking gate',
        geometry_torque_task_feasibility_guaranteed=False,original_constraints_relaxed=False,
        physical_state_modified=False,new_physics_steps=0,new_geometry_queries=0)
    return lo,hi,diagnostic
