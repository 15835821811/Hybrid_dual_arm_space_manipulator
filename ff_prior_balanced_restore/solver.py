"""Full-horizon 323 restoration. Optimization iterates are never executed."""
import time
import mujoco
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from ff_prior.storage import digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior_static_fullchain_v2.propagation import predict_phases
from ff_prior_fixed.correction import pose_vector
from .protocol import P


def merit(goal,distances):
    v=np.minimum((np.asarray(distances)-P['margin_m'])/P['distance_scale_m'],0.)
    return P['goal_weight']*float(goal@goal)+(float(np.mean(v*v)+.25*np.max(v*v)) if len(v) else 0.)


def metrics(value):
    r=value['goal'];groups=np.linalg.norm(r.reshape(4,3),axis=1)
    return dict(Q=float(max(groups)),goal_groups=groups,
        minimum_clearance_m=float(min(value['minima'])),geometry_step_m=value['extent'],
        merit=merit(r,value['minima']),intervals=value['intervals'])


def select_active(near,cap):
    """Distinct pair then phase-bin coverage, then distance, with all omissions saved."""
    if not len(near):return np.empty(0,dtype=int),[]
    ranked=sorted(range(len(near)),key=lambda i:(near[i,3],*near[i,:3]))
    chosen=[];reasons=[];seen_pairs=set();seen_bins=set()
    def take(i,why):
        if i not in chosen and len(chosen)<cap:chosen.append(i);reasons.append(why)
    for i in ranked:
        pair=tuple(near[i,1:3].astype(int))
        if pair not in seen_pairs:take(i,'worst_for_geometry_pair');seen_pairs.add(pair)
    for i in ranked:
        phase_bin=min(63,int(near[i,0]*64))
        if phase_bin not in seen_bins:take(i,'worst_for_phase_bin');seen_bins.add(phase_bin)
    for i in ranked:take(i,'remaining_by_distance')
    order=np.argsort(chosen)
    return np.asarray(chosen,dtype=int)[order],[reasons[i] for i in order]


class Evaluator:
    def __init__(self,task,robot,initial,collision,budget):
        self.task,self.robot,self.initial,self.cc,self.budget=task,robot,initial,collision,budget
        self.scratch=initial.reset(robot);self.seen={};self.serial=0

    def ref(self,z):return Reference(z,self.task['condition']['initial_features'],P['duration_s'])

    def predict(self,z,grid):
        count=int(grid) if np.isscalar(grid) else len(grid)-1
        self.budget.consume(count)
        result=(predict_coupled(self.initial,self.ref(z),self.robot,int(grid),self.budget.deadline)
                if np.isscalar(grid) else predict_phases(self.initial,self.ref(z),self.robot,grid,self.budget.deadline))
        n=result['numerical']
        if not (n['solve_residual']<1e-10 and n['robot_momentum_residual']<1e-10 and n['quaternion_norm_error']<1e-12):
            raise ValueError('MODEL_NUMERICAL_GATE_FAILED')
        return result

    def dense(self,z,path):
        key=digest(dict(z=z,initial=self.initial.state_id,task=self.task['task_hash'],
            model=self.initial.model_hash,policy=self.cc.policy_hash,protocol=P))
        # The only cache is exact full reference from the identical root and fixed implementation.
        if key in self.seen:
            value=self.seen[key]
            atomic_json(path.with_suffix('.json'),dict(reused_saved_evaluation=value['path'],key=key,new_queries=0))
            return value
        history=[]
        for n in P['dense_grids']:
            pred=self.predict(z,n)
            extent=float(np.max(np.linalg.norm(np.diff(pred['geom'],axis=0),axis=-1)+
                np.linalg.norm(np.diff(pred['geom_rotation'],axis=0).reshape(-1,self.robot.model.ngeom,3,3),axis=(-1,-2))*self.robot.model.geom_rbound))
            history.append(dict(intervals=n,geometry_step_m=extent,numerical=pred['numerical']))
            if extent<=P['geometry_step_m']:break
        minima=[];pairs=[];near=[]
        predictor_bytes=sum(a.nbytes for a in pred.values() if isinstance(a,np.ndarray))
        self.budget.parent.ensure_output(predictor_bytes+64*1024**2)
        try:
            for j,state in enumerate(pred['qpos']):
                self.budget.check();self.scratch.qpos[:]=state;self.robot.kinematics(self.scratch)
                scan=self.cc.scan(self.scratch,P['active_buffer_m'])
                minima.append(scan['minimum_clearance_m']);pairs.append(scan['minimum_pair'])
                for v in scan['violations']:
                    near.append((pred['phase'][j],self.robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_a']),
                                 self.robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_b']),v['distance_m']))
                if j%64==0:
                    estimated=len(near)*256+predictor_bytes+64*1024**2
                    if estimated>1024**3:raise RuntimeError('MODEL_NEAR_RECORD_MEMORY_BUDGET')
                    self.budget.parent.ensure_output(estimated)
        except Exception as error:
            # The already observed records remain evidence even when the scan stops.
            atomic_npz(path.with_suffix('.partial.npz'),Z=z,phase=pred['phase'][:len(minima)],
                minima=np.asarray(minima),near=np.asarray(near,dtype=float).reshape(-1,4))
            atomic_json(path.with_suffix('.partial.json'),dict(status='INCOMPLETE_SCAN',reason=repr(error),
                completed_states=len(minima),required_states=len(pred['qpos']),minimum_pairs=pairs,
                all_completed_scan_buffer_records_saved=True,selection_not_run=True,
                qualification=False,near_columns=['phase','geom_a_id','geom_b_id','distance_m']))
            raise
        near=np.asarray(near,dtype=float).reshape(-1,4);selected,reasons=select_active(near,P['active_cap'])
        active=near[selected,:3];dist=near[selected,3]
        result=dict(goal=pose_vector(pred['ee'][-1],self.task['condition']['goals']),ee=pred['ee'][-1],
            minima=np.asarray(minima),extent=extent,intervals=n,
            active=active,distances=dist,history=history,path=str(path),key=key)
        self.budget.parent.ensure_output(near.nbytes+sum(a.nbytes for a in pred.values() if isinstance(a,np.ndarray))+32*1024**2)
        atomic_npz(path.with_suffix('.npz'),Z=z,phase=pred['phase'],qpos=pred['qpos'],velocity=pred['velocity'],
            ee=pred['ee'],base=pred['base'],minima=np.asarray(minima),near=near,selected=selected,
            omitted=np.setdiff1d(np.arange(len(near)),selected),goal=result['goal'])
        atomic_json(path.with_suffix('.json'),dict(metrics(result),grid_history=history,
            minimum_pairs=pairs,selected_reasons=reasons,near_columns=['phase','geom_a_id','geom_b_id','distance_m'],
            all_buffer_records_saved=True,phase_count=len(pred['phase']),pair_policy_hash=self.cc.policy_hash,
            key=key,classification='OPTIMIZATION_MODEL_EVALUATION',continuous_time_certified=False))
        # Keep a bounded scalar/active cache; qpos is read from disk if needed, not repeatedly propagated.
        self.seen[key]=result
        return result

    def sparse(self,z,phase,active):
        pred=self.predict(z,phase);dist=[];states=[];last=-1
        for t,ga,gb in active:
            j=int(np.searchsorted(phase,t));assert phase[j]==t
            if j!=last:self.scratch.qpos[:]=pred['qpos'][j];self.robot.kinematics(self.scratch);last=j
            dist.append(float(mujoco.mj_geomDistance(self.robot.model,self.scratch,int(ga),int(gb),2.5,None)))
            states.append(pred['qpos'][j])
        self.robot.calls['distance']+=len(active)
        return dict(goal=pose_vector(pred['ee'][-1],self.task['condition']['goals']),distances=np.asarray(dist),
            ee=pred['ee'][-1],active_qpos=np.asarray(states).reshape(len(active),self.robot.model.nq),
            terminal_qpos=pred['qpos'][-1],numerical=pred['numerical'])

    def feedback(self,z,current):
        checks=[]
        for n in P['sparse_grids']:
            phase=np.unique(np.r_[np.linspace(0,1,n+1),current['active'][:,0]])
            value=self.sparse(z,phase,current['active']);pos,rot=pose_errors(value['ee'],current['ee'])
            difference=float(np.max(abs(value['distances']-current['distances']),initial=0))
            ok=bool(max(pos)<=P['sparse_agreement_m'] and max(rot)<=P['sparse_agreement_rad'] and difference<=P['sparse_agreement_m'])
            checks.append(dict(intervals=len(phase)-1,position_m=pos,rotation_rad=rot,distance_m=difference,passed=ok))
            if ok:return value,phase,checks
        raise ValueError('SPARSE_DENSE_NUMERICAL_DISAGREEMENT')


def solve_step(z,center,jg,jd,motion,radius,robot):
    x=z.ravel();lo=np.maximum(motion.lower-x,-radius)/radius;hi=np.minimum(motion.upper-x,radius)/radius
    A=motion.A*radius;lb=motion.lo-motion.A@x;ub=motion.hi-motion.A@x
    G=jg*radius;D=jd*radius/P['distance_scale_m'];b=(center['distances']-P['margin_m'])/P['distance_scale_m']
    r=center['goal'];normalizer=max(1.,merit(r,center['distances']))
    def fun(u):
        v=np.minimum(b+D@u,0.);g=r+G@u
        return (P['goal_weight']*float(g@g)+(float(np.mean(v*v)+.25*max(v*v)) if len(v) else 0.)+
                1e-4*float(u@u)/323)/normalizer
    def jac(u):
        v=np.minimum(b+D@u,0.);g=r+G@u
        out=2*P['goal_weight']*G.T@g+2e-4*u/323
        if len(v):
            k=int(np.argmax(v*v));out+=2*D.T@v/len(v)+.5*v[k]*D[k]
        return out/normalizer
    best=np.zeros(323);best_f=fun(best);callbacks=0
    def callback(u):
        nonlocal best,best_f,callbacks
        callbacks+=1;value=fun(u)
        if (np.isfinite(u).all() and np.all(u>=lo-1e-10) and np.all(u<=hi+1e-10)
                and motion.valid(x+radius*u) and value<best_f):best=u.copy();best_f=value
    robot.calls['sqp']+=1
    sol=minimize(fun,np.zeros(323),jac=jac,method='SLSQP',bounds=Bounds(lo,hi),
        constraints=LinearConstraint(A,lb,ub),callback=callback,
        options={'maxiter':P['solver_iterations'],'ftol':1e-10})
    callback(sol.x)
    return radius*best,dict(success=bool(sol.success),message=str(sol.message),iterations=int(sol.nit),
        callbacks=callbacks,solved_u=sol.x,returned_u=best,initial_objective=fun(np.zeros(323)),
        returned_objective=best_f,returned_gradient=jac(best),normalizer=normalizer,
        returned_motion_valid=motion.valid(x+radius*best),radius_rad=radius,
        initialization='ZERO_MOTION_FEASIBLE',world_components_protected=False)


def optimize(directory,z,motion,ev,progress):
    z=np.asarray(z).copy();current=ev.dense(z,directory/'initial');radius=P['initial_radius_rad']
    history=[];stop='ITERATION_LIMIT'
    for i in range(P['iterations']):
        met=metrics(current)
        if met['Q']<=.8 and met['minimum_clearance_m']>=P['margin_m'] and met['geometry_step_m']<=.002:
            stop='RESTORATION_TARGET_REACHED';break
        if not ev.budget.can_fit(P['iteration_query_reserve'],P['iteration_interval_reserve']):
            stop='WHOLE_ITERATION_BUDGET_RESERVE';break
        pd=directory/'iter'/f'{i:02d}';progress('full323_restoration',iteration=i+1,maximum=P['iterations'])
        center,phase,agreement=ev.feedback(z,current);active=current['active'].copy()
        goals=[];distances=[];poses=[];qposes=[];terminal=[];steps=[]
        for j in range(323):
            h=P['difference_rad'] if motion.upper[j]-z.ravel()[j]>=z.ravel()[j]-motion.lower[j] else -P['difference_rad']
            h=np.sign(h)*min(abs(h),max(motion.upper[j]-z.ravel()[j],z.ravel()[j]-motion.lower[j])*.5)
            assert abs(h)>1e-12
            trial=z.copy();trial.ravel()[j]+=h;v=ev.sparse(trial,phase,current['active'])
            steps.append(h);goals.append(v['goal']);distances.append(v['distances']);poses.append(v['ee'])
            qposes.append(v['active_qpos']);terminal.append(v['terminal_qpos'])
        steps=np.asarray(steps);goals=np.asarray(goals);distances=np.asarray(distances)
        jg=(goals-center['goal']).T/steps;jd=(distances-center['distances']).T/steps
        probes=[]
        # Observations are saved, never called a gradient safety certificate.
        for j in (int(np.argmax(np.linalg.norm(jg,axis=0))),int(np.argmax(np.linalg.norm(jd,axis=0))) if len(jd) else 161):
            trial=z.copy();trial.ravel()[j]+=steps[j]/2;v=ev.sparse(trial,phase,current['active'])
            probes.append(dict(kind='coordinate_half',coordinate=j,step=steps[j]/2,**v))
        direction=np.sin(np.arange(323)+.37);direction/=np.linalg.norm(direction)
        for sign in (-1.,1.):
            v=ev.sparse(z+sign*P['difference_rad']*direction.reshape(19,17),phase,current['active'])
            probes.append(dict(kind='direction',sign=sign,direction=direction,step=P['difference_rad'],**v))
        step,solver=solve_step(z,center,jg,jd,motion,radius,ev.robot);trials=[];accepted=False;before=z.copy()
        predicted_before=merit(center['goal'],center['distances'])
        for k,scale in enumerate(P['line_scales'] if np.max(abs(step))>1e-12 else []):
            trial=z+scale*step.reshape(19,17)
            if not motion.valid(trial):
                trials.append(dict(scale=scale,status='MOTION_REJECTED'));continue
            prediction=merit(center['goal']+jg@(scale*step),center['distances']+jd@(scale*step))
            value=ev.dense(trial,pd/f'trial{k}')
            reduction=met['merit']-metrics(value)['merit']
            predicted=predicted_before-prediction
            ratio=reduction/predicted if predicted>1e-12 else None
            ok=bool(reduction>1e-10 and ratio is not None and ratio>=P['ratio_accept'])
            trials.append(dict(scale=scale,status='EVALUATED',accepted=ok,metrics=metrics(value),
                actual_reduction=reduction,predicted_reduction=predicted,ratio=ratio))
            if ok:z=trial;current=value;accepted=True;break
        ev.budget.parent.ensure_output(np.asarray(qposes).nbytes+np.asarray(terminal).nbytes+64*1024**2)
        atomic_npz(pd/'jacobian.npz',Z=before,next_Z=z,phase=phase,active=active,
            goal=center['goal'],distances=center['distances'],ee=center['ee'],active_qpos=center['active_qpos'],
            terminal_qpos=center['terminal_qpos'],steps=steps,perturb_goal=goals,perturb_distances=distances,
            perturb_ee=np.asarray(poses),perturb_active_qpos=np.asarray(qposes),perturb_terminal_qpos=np.asarray(terminal),jg=jg,jd=jd,step=step)
        atomic_json(pd/'probes.json',dict(observations=probes,acceptance_threshold=None,
            role='derivative diagnostics only, not geometry or safety certification'))
        next_radius=max(P['min_radius_rad'],radius/2) if not accepted or trials[-1]['ratio']<.25 or trials[-1]['scale']<1 else (min(P['max_radius_rad'],2*radius) if trials[-1]['ratio']>.75 and max(abs(step))>=.9*radius else radius)
        row=dict(iteration=i,accepted=accepted,before=met,after=metrics(current),trials=trials,solver=solver,
            sparse_agreement=agreement,radius_rad=radius,next_radius_rad=next_radius)
        atomic_json(pd/'record.json',row);history.append(row)
        ev.budget.parent.save()
        if not accepted and radius<=P['min_radius_rad']:stop='MINIMUM_TRUST_NO_DESCENT';break
        radius=next_radius
    result=dict(metrics(current),Z=z,iterations=len(history),stop=stop,qualification='UNQUALIFIED_RESTORED_REFERENCE',
        reference_hash=ev.ref(z).reference_hash,last_evaluation=current['path'])
    atomic_json(directory/'result.json',result)
    return result
