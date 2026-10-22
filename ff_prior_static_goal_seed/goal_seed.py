"""Public nonlinear endpoint recovery along 17 full-path smooth directions.

Every derivative and line search propagates the floating base. Collision is
deliberately deferred to the shared full323 solver; no seed is MODEL_VALID.
"""
import numpy as np
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from ff_prior.proposal_generator import quick_gates
from ff_prior_pilot_train.storage import atomic_npz,atomic_json
from ff_prior_static_fullchain_v2.solver import admissible_difference
from .protocol import P

def directions():
    s=np.r_[np.linspace(0,1,20)[1:-1],1.]
    h=s*s*(3-2*s)
    return np.eye(17)[:,None,:]*h[None,:,None]

def descent_step(j,e):
    if not np.isfinite(j).all() or not np.isfinite(e).all():
        raise ValueError('NONFINITE_GOAL_LINEARIZATION')
    step=-np.linalg.solve(j.T@j+P['goal_damping']*np.eye(17),j.T@e)
    step*=min(1.,P['goal_step_rad']/max(float(abs(step).max()),1e-12))
    return step

def accepts(old,new):
    a,b=group_norms(old),group_norms(new)
    return (all(x>1 or y<=1 for x,y in zip(a,b))
            and float(new@new)<float(old@old)-1e-10)

def recover(out,start,ev,progress):
    z=np.asarray(start).copy();basis=directions();history=[];stop='ITERATION_LIMIT'
    def endpoint(control):
        pred=ev.predict(control,P['goal_intervals'])
        return pose_vector(pred['ee'][-1],ev.task['condition']['goals']),pred['ee'][-1]
    e,ee=endpoint(z)
    atomic_npz(out/'initial.npz',Z=z,ee=ee,goal=e)
    for iteration in range(P['goal_iterations']):
        progress('nonlinear_goal_seed',candidate=out.parent.name,iteration=iteration+1)
        if max(group_norms(e))<=P['goal_target_Q']:
            stop='TARGET_MARGIN_AT_SEED_GRID';break
        if not ev.budget.can_fit(24,24*P['goal_intervals']):
            stop='RESERVED_GOAL_BUDGET';break
        columns=[];perturbed=[];eps=[]
        for d in basis:
            h=admissible_difference(z,d,ev.robot.lower,ev.robot.upper)
            de,pose=endpoint(z+h*d) if h else (e,ee)
            columns.append((de-e)/h if h else np.zeros(12));perturbed.append(pose);eps.append(h)
        j=np.asarray(columns).T;step=descent_step(j,e);delta=np.einsum('i,ijk->jk',step,basis)
        before=z.copy();old_e=e.copy();old_ee=ee.copy();trials=[];accepted=False
        for scale in P['line_scales'] if abs(delta).max()>1e-12 else []:
            trial=z+scale*delta
            gates=quick_gates(ev.robot,ev.reference(trial))
            row=dict(scale=scale,quick=gates,accepted=False)
            if all(gates.values()):
                new_e,new_ee=endpoint(trial);ok=accepts(e,new_e)
                row.update(goal=new_e,ee=new_ee,accepted=ok)
                if ok:z=trial;e=new_e;ee=new_ee;accepted=True
            trials.append(row)
            if accepted:break
        d=out/'iterations'/f'{iteration:02d}'
        atomic_npz(d/'step.npz',before_Z=before,after_Z=z,goal=old_e,ee=old_ee,
                   jacobian=j,epsilon=np.asarray(eps),perturb_ee=np.asarray(perturbed),step=step,directions=basis)
        record=dict(iteration=iteration,accepted=accepted,trials=trials,
                    before_Q=float(max(group_norms(old_e))),after_Q=float(max(group_norms(e))))
        atomic_json(d/'record.json',record);history.append(record)
        if not accepted:stop='NO_ACCEPTED_GOAL_DESCENT';break
    atomic_npz(out/'result.npz',Z=z,ee=ee,goal=e)
    result=dict(status='SEED_ONLY_NOT_MODEL_VALID',stop=stop,iterations=len(history),
                Q=float(max(group_norms(e))),goal_groups=group_norms(e),
                goal_margin_at_seed_grid=bool(max(group_norms(e))<=P['goal_target_Q']),
                collision_checked=False,goal_relabel=False)
    atomic_json(out/'result.json',result)
    return z,result
