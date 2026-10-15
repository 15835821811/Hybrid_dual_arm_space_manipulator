"""Bounded TRAIN-only goal-preserving correction in 17 smooth joint directions."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior.storage import digest
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior.model_gates import evaluate_reference

MAX_ITERATIONS=8


def choose_targets(rows,count=8):
    selected=[]; families=set()
    pool=sorted((r for r,s in rows if s=='TRAIN'),key=lambda r:digest({'selection':'fixed_train_v1','id':r['sample_id']}))
    for row in pool:
        if row['family_id'] in families: continue
        selected.append(row); families.add(row['family_id'])
        if len(selected)==count: break
    if len(selected)!=count: raise ValueError('insufficient independent TRAIN families')
    return selected


def pose_vector(actual,goal):
    goal=np.asarray(goal)
    return np.r_[((actual[:,:3,3]-goal[:,:3,3])/.005).ravel(),
        (Rotation.from_matrix(np.swapaxes(goal[:,:3,:3],-1,-2)@actual[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]


def is_eligible(validation,goal_valid):
    return validation.get('validation_level')=='MODEL_VALID' and bool(goal_valid)


def correct(task,source_id,split,variant,robot,initial,collision,profile,deadline,max_iterations=MAX_ITERATIONS):
    if split!='TRAIN': raise ValueError('correction targets must be TRAIN only')
    if not 0<=max_iterations<=MAX_ITERATIONS or variant not in (0,1): raise ValueError('correction budget invalid')
    assert_task_compatible(task,robot,initial,collision,profile)
    original_hash=task['task_hash']; c=task['condition']; q0=np.asarray(c['initial_features'])
    seed=20261019+int(digest(source_id)[:8],16)%100000+variant*997
    rng=np.random.default_rng(seed)
    z=public_goal_initializer(task,robot,initial)
    z+=.01*np.sin(np.linspace(0,np.pi,19))[:,None]*rng.normal(size=(1,17))
    z=np.clip(z,robot.lower,robot.upper)
    # Fixed smooth basis affects the complete path and its shared terminal.
    s=np.r_[np.linspace(0,1,20)[1:-1],1.]
    h=(s*s*(3-2*s))[:,None]
    directions=np.eye(17)[:,None,:]*h[None,:,:]
    evaluations=0
    def evaluate(proposal):
        nonlocal evaluations
        if time.time()>=deadline: raise RuntimeError('CORRECTION_WALL_BUDGET_EXHAUSTED')
        p=predict_coupled(initial,Reference(proposal,q0,c['duration']),robot,128,deadline)
        evaluations+=1
        return pose_vector(p['ee'][-1],c['goals']),p['ee'][-1]
    e,actual=evaluate(z); history=[]
    for iteration in range(max_iterations):
        pos,rot=pose_errors(actual,c['goals'])
        if np.max(pos)<=.001 and np.max(rot)<=np.deg2rad(.2): break
        epsilon=1e-4
        J=np.stack([(evaluate(z+epsilon*d)[0]-evaluate(z-epsilon*d)[0])/(2*epsilon) for d in directions],axis=1)
        # Damped minimum-norm task correction, no high-dimensional search.
        step=-J.T@np.linalg.solve(J@J.T+1e-3*np.eye(12),e)
        step*=min(1.,.1/max(np.linalg.norm(step),1e-12))
        before=float(e@e); accepted=False; trials=[]
        for scale in (1.,.5,.25,.125):
            trial=np.clip(z+scale*h*step,robot.lower,robot.upper)
            ref=Reference(trial,q0,c['duration'])
            if not all(quick_gates(robot,ref).values()):
                trials.append({'scale':scale,'status':'QUICK_REJECTED'}); continue
            ne,na=evaluate(trial); score=float(ne@ne)
            trials.append({'scale':scale,'normalized_pose_cost':score})
            if score<before:
                z,e,actual=trial,ne,na; accepted=True; break
        history.append({'iteration':iteration,'before_cost':before,'after_cost':float(e@e),
            'accepted':accepted,'step_norm':float(np.linalg.norm(step)),
            'jacobian_singular_values':np.linalg.svd(J,compute_uv=False),'trials':trials})
        if not accepted: break
    prediction,validation=evaluate_reference(robot,initial,Reference(z,q0,c['duration']),collision,profile,deadline)
    goal_valid=False; pos=rot=None
    if prediction is not None:
        pos,rot=pose_errors(prediction['ee'][-1],c['goals'])
        goal_valid=bool(np.all(pos<=profile.position_error_m) and np.all(rot<=profile.rotation_error_rad))
    assert_frozen(task)
    if task['task_hash']!=original_hash: raise ValueError('TRAIN goal relabeled')
    eligible=is_eligible(validation,goal_valid)
    return z,{'status':'ACCEPTED' if eligible else 'REJECTED','source_sample_id':source_id,'split':'TRAIN',
        'variant':variant,'seed':seed,'task_hash':original_hash,'condition_hash':digest(c),
        'goal_source':'FROZEN_TRAIN_WORLD_GOAL','goal_relabelled':False,'private_terminal_used':False,
        'validation_level':'MODEL_VALID' if eligible else 'REJECTED','model_validation':validation,
        'frozen_goal_pass':goal_valid,'position_error_m':pos,'rotation_error_rad':rot,
        'training_eligible':eligible,'data_role':'FF_MODEL_PRIOR','execution_status':'NOT_RUN',
        'optimizer':'17_smooth_direction_damped_goal_correction','iterations':len(history),
        'optimization_coupled_calls':evaluations,'history':history,
        'reference_hash':Reference(z,q0,c['duration']).reference_hash}
