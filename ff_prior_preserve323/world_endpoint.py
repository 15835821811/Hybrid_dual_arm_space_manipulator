"""Current full16 output -> free-floating world residual -> complete 323 Jacobian.

This is a training derivative, never a geometry or execution validator.
Independent scalar probes are evaluated by the physical predictor, not by J.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.contracts import assert_frozen
from ff_prior.spline import Reference
from ff_prior.storage import digest
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from .protocol import P


def residual(ee, goals):
    ee, goals=np.asarray(ee),np.asarray(goals)
    position=(ee[:,:3,3]-goals[:,:3,3])/.005
    rotation=Rotation.from_matrix(np.swapaxes(goals[:,:3,:3],-1,-2)@ee[:,:3,:3]).as_rotvec()/np.deg2rad(1.)
    return np.r_[position.reshape(-1),rotation.reshape(-1)]


def groups(r):
    return np.sum(np.asarray(r).reshape(4,3)**2,axis=1)


def evaluate(x,codec,center,base,task,robot,initial,deadline,n=256):
    assert_frozen(task)
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(323,) or not np.isfinite(x).all():raise ValueError('INVALID_WORLD_OUTPUT')
    z=codec.decode(codec.coefficients(x,center),base)
    c=task['condition'];reference=Reference(z,c['initial_features'],c['duration'])
    p=predict_coupled(initial,reference,robot,n,deadline)
    ee=p['ee'][-1];r=residual(ee,c['goals'])
    if not np.isfinite(r).all():raise ValueError('NONFINITE_WORLD_RESIDUAL')
    return r,dict(ee=ee,numerical=p['numerical'],reference_hash=reference.reference_hash,Z_hash=digest(z))


def jacobian(x,codec,center,base,task,robot,initial,deadline):
    eps=P['world_difference'];before=robot.calls.copy()
    r0,central=evaluate(x,codec,center,base,task,robot,initial,deadline)
    perturb=np.empty((323,2,12));poses=np.empty((323,2,2,4,4))
    for j in range(323):
        for k,sign in enumerate((-1.,1.)):
            v=x.copy();v[j]+=sign*eps
            perturb[j,k],record=evaluate(v,codec,center,base,task,robot,initial,deadline)
            poses[j,k]=record['ee']
    J=(perturb[:,1]-perturb[:,0]).T/(2*eps)
    halves=[]
    for j in (0,161):
        records=[];rr=[]
        for sign in (-1.,1.):
            v=x.copy();v[j]+=sign*eps/2
            r,rec=evaluate(v,codec,center,base,task,robot,initial,deadline)
            rr.append(r);records.append(rec)
        slope=(rr[1]-rr[0])/eps
        err=float(np.linalg.norm(slope-J[:,j])/max(np.linalg.norm(slope),np.linalg.norm(J[:,j]),1e-6))
        halves.append(dict(dimension=j,residuals=rr,records=records,relative_error=err,passed=err<=P['world_half_tolerance']))
    direction=np.sin(np.arange(323)+.37);direction/=np.linalg.norm(direction)
    probes=[];rp=[]
    for sign in (-1.,1.):
        r,rec=evaluate(x+sign*eps*direction,codec,center,base,task,robot,initial,deadline)
        rp.append(r);probes.append(rec)
    # Mean four squared group errors is an independent smooth scalar objective.
    slope=(groups(rp[1]).mean()-groups(rp[0]).mean())/(2*eps)
    expected=float((r0@J@direction)/2)
    error=float(abs(slope-expected)/max(abs(slope),abs(expected),1e-6))
    rf,fine=evaluate(x,codec,center,base,task,robot,initial,deadline,P['world_fine_intervals'])
    dp,dr=pose_errors(central['ee'],fine['ee'])
    convergence=bool(max(dp)<=.001 and max(dr)<=np.deg2rad(.2))
    calls={k:robot.calls[k]-before[k] for k in before}
    passed=bool(np.isfinite(J).all() and all(v['passed'] for v in halves) and
        error<=P['world_direction_tolerance'] and convergence and calls['coupled']==654)
    proof=dict(status='PASS' if passed else 'BLOCKED',dimensions=323,residual_dimensions=12,
        epsilon=eps,central=central,half_checks=halves,direction=direction,direction_records=probes,
        direction_residuals=rp,direction_slope=float(slope),direction_expected=expected,
        direction_relative_error=error,convergence=dict(passed=convergence,position_m=dp,rotation_rad=dr,fine=fine),
        calls=calls,task_hash=task['task_hash'],codec_hash=digest(codec.state),x_hash=digest(x),
        current_full16_output=True,physical_validation_level='NOT_A_VALIDATOR')
    return r0,J,proof,dict(x=x,residual=r0,J=J,perturbation_residuals=perturb,perturbation_ee=poses,
                           central_ee=central['ee'],fine_residual=rf)
