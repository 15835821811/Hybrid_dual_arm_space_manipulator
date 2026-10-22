"""Qualification-preserving prepare: identity if feasible, convex shrinking otherwise."""
import numpy as np
from scipy.interpolate import BSpline
from ff_prior_dynamic_margin.representation import design,polynomial_bounds

def prepare(coefficients,q0,v0,lower,upper,vmax,amax):
    c=np.asarray(coefficients,dtype=np.float64).copy()
    assert c.shape==(273,17)
    if not np.isfinite(c).all():raise ValueError('NONFINITE_REFERENCE')
    knots,_,d1,d2=design(27.)
    if np.max(abs(v0))>1e-12:raise ValueError('PREPARE_NONZERO_PLANNER_INITIAL_VELOCITY_UNSUPPORTED')
    # Only the reference coefficients are transformed, never an actual physical state.
    c[:2]=q0
    centre=np.tile(q0,(273,1));delta=c-centre
    if np.any(q0<lower) or np.any(q0>upper):raise ValueError('INITIAL_DOMAIN')
    positive=delta>0;negative=delta<0;ratios=[1.]
    if positive.any():ratios.append(float(np.min(np.broadcast_to(upper-q0,delta.shape)[positive]/delta[positive])))
    if negative.any():ratios.append(float(np.min(np.broadcast_to(lower-q0,delta.shape)[negative]/delta[negative])))
    for D,limit in [(d1,vmax),(d2,amax)]:
        derivative=abs(D@delta);mask=derivative>0
        if mask.any():ratios.append(float(np.min(np.broadcast_to(limit,derivative.shape)[mask]/derivative[mask])))
    alpha=max(0.,min(ratios));changed=alpha<1.
    if changed:alpha*=1-1e-7;c=centre+alpha*delta
    spline=BSpline(knots,c,3,extrapolate=False)
    bounds=polynomial_bounds(spline)
    gates=dict(finite=bool(np.isfinite(c).all()),position=bool(np.all(bounds[0][0]>=lower) and np.all(bounds[0][1]<=upper)),
        velocity=bool(np.all(np.maximum(abs(bounds[1][0]),abs(bounds[1][1]))<=vmax)),
        acceleration=bool(np.all(np.maximum(abs(bounds[2][0]),abs(bounds[2][1]))<=amax)),
        initial_q=bool(np.max(abs(spline(0)-q0))<1e-10),initial_v=bool(np.max(abs(spline(0,1)-v0))<1e-10))
    return spline,dict(gates=gates,scale=alpha,changed=changed,maximum_change_rad=float(abs(c-coefficients).max()),
                       physics_validated=False,actual_state_changed=False)
