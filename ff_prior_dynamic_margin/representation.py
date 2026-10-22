"""C2 cubic B-spline fit with coefficient-domain and derivative hull constraints."""
import time
import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import minimize, LinearConstraint, Bounds
from .protocol import P

def design(end):
    knots=np.r_[np.zeros(4),np.arange(P['knot_spacing'],end-1e-9,P['knot_spacing']),np.full(4,end)]
    n=len(knots)-4
    basis=BSpline(knots,np.eye(n),3,extrapolate=False)
    d1=np.zeros((n-1,n))
    for i in range(n-1):d1[i,i:i+2]=np.array([-3.,3.])/(knots[i+4]-knots[i+1])
    d2=np.zeros((n-2,n))
    for i in range(n-2):d2[i]=2*(d1[i+1]-d1[i])/(knots[i+4]-knots[i+2])
    return knots,basis,d1,d2

def polynomial_bounds(spline):
    """Independent extrema of each scalar polynomial, including interval boundaries."""
    coeff=np.asarray(spline.c); result=[]
    for order in range(3):
        lo=np.full(coeff.shape[1],np.inf);hi=-lo.copy()
        for j in range(coeff.shape[1]):
            poly=PPoly.from_spline(BSpline(spline.t,coeff[:,j],spline.k)).derivative(order)
            for i,w in enumerate(np.diff(poly.x)):
                if w<=0:continue
                c=poly.c[:,i];roots=np.roots(np.polyder(c))
                xx=[0.,w]+[float(z.real) for z in roots if abs(z.imag)<1e-10 and 0<z.real<w]
                yy=np.polyval(c,xx);lo[j]=min(lo[j],min(yy));hi[j]=max(hi[j],max(yy))
        result.append((lo,hi))
    return result

def fit(t,q,dq,lower,upper,vmax,amax,progress,deadline):
    assert np.isclose(t[0],0) and np.isclose(t[-1],P['duration'])
    # Native accumulated time can be 27 - 4e-12. The contractual endpoint is
    # exactly 27; otherwise extrapolate=False would return NaN on the last step.
    knots,basis,d1,d2=design(P['duration']); b=basis(t); v=basis(t,1)
    n=b.shape[1]; w=P['derivative_fit_weight']; m=len(t)
    H=(b.T@b+w*v.T@v)/m+P['ridge']*np.eye(n)
    C=np.vstack([d1,d2]); coeff=[]; records=[]
    for j in range(q.shape[1]):
        progress('constrained_fit',joint=j)
        if time.monotonic()>deadline:raise TimeoutError('REFERENCE_FIT_WALL_BUDGET')
        fixed=np.array([q[0,j],q[0,j]+dq[0,j]*(knots[4]-knots[1])/3])
        target=(b.T@q[:,j]+w*v.T@dq[:,j])/m
        h=H[2:,2:]; f=H[2:,:2]@fixed-target[2:]
        unconstrained=np.linalg.solve(h,-f)
        upper_deriv=np.r_[np.full(n-1,vmax[j]*P['derivative_fraction']),
                          np.full(n-2,amax[j]*P['acceleration_fraction'])]
        offset=C[:,:2]@fixed
        lc=LinearConstraint(C[:,2:],-upper_deriv-offset,upper_deriv-offset)
        bound=Bounds(np.full(n-2,lower[j]+P['coefficient_margin']),np.full(n-2,upper[j]-P['coefficient_margin']))
        def feasible(x):
            return bool(np.all(x>=bound.lb) and np.all(x<=bound.ub) and
                np.all(C[:,2:]@x>=lc.lb) and np.all(C[:,2:]@x<=lc.ub))
        def callback(x):
            if time.monotonic()>deadline:raise TimeoutError('REFERENCE_FIT_WALL_BUDGET')
        if feasible(unconstrained):
            x=unconstrained;record=dict(method='UNCONSTRAINED_OPTIMUM_FEASIBLE',success=True,iterations=0)
        else:
            # A constant initial reference is feasible when the frozen initial velocity is zero.
            # Nonzero initial velocities are retained in the first two coefficients and checked below.
            x0=np.full(n-2,q[0,j])
            result=minimize(lambda x:.5*x@h@x+f@x,x0,jac=lambda x:h@x+f,
                bounds=bound,constraints=[lc],method='SLSQP',callback=callback,
                options={'maxiter':P['optimizer_maxiter'],'ftol':P['optimizer_ftol']})
            x=result.x;record=dict(method='CONSTRAINED_QUADRATIC_SLSQP',success=bool(result.success),
                iterations=int(result.nit),message=str(result.message))
        c=np.r_[fixed,x];coeff.append(c)
        record.update(joint=j,objective=float(.5*c@H@c-target@c),
            maximum_hull_violation=float(np.max(np.abs(C@c)-upper_deriv)))
        records.append(record)
    spline=BSpline(knots,np.array(coeff).T,3,extrapolate=False)
    bounds=polynomial_bounds(spline)
    gates=dict(finite=bool(np.isfinite(spline.c).all()),
        position=bool(np.all(bounds[0][0]>=lower) and np.all(bounds[0][1]<=upper)),
        velocity=bool(np.all(np.maximum(abs(bounds[1][0]),abs(bounds[1][1]))<=vmax)),
        acceleration=bool(np.all(np.maximum(abs(bounds[2][0]),abs(bounds[2][1]))<=amax)),
        initial_q=bool(np.max(abs(spline(0)-q[0]))<=P['initial_tolerance']),
        initial_dq=bool(np.max(abs(spline(0,1)-dq[0]))<=P['initial_tolerance']),
        solver=all(r['success'] for r in records))
    report=dict(gates=gates,polynomial_bounds=bounds,solver=records,knots=len(knots),
        maximum_sample_fit_error_rad=float(np.max(abs(spline(t)-q))),
        rms_sample_fit_error_rad=float(np.sqrt(np.mean((spline(t)-q)**2))),
        maximum_acceleration_ratio=float(np.max(np.maximum(abs(bounds[2][0]),abs(bounds[2][1]))/amax)),
        duration_s=P['duration'],source_last_time=float(t[-1]),physics_validated=False)
    return spline,report
