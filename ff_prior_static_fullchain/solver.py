"""Feasible incumbent, physical-control trust region, dense-to-sparse feedback."""
import time
import mujoco
import numpy as np
from scipy.optimize import minimize
from ff_prior.storage import digest
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from .propagation import predict_phases
from .protocol import P

def basis(kind, codec=None):
    if kind != 'full323': raise ValueError('only full323 is frozen for this run')
    return np.eye(323).reshape(323,19,17)

def admissible_difference(z, direction, lower, upper):
    capacities=[]
    for sign in (1.,-1.):
        d=direction*sign; caps=np.full(d.shape,np.inf)
        np.divide(np.broadcast_to(upper,z.shape)-z,d,out=caps,where=d>0)
        np.divide(np.broadcast_to(lower,z.shape)-z,d,out=caps,where=d<0)
        capacities.append(float(np.min(caps)))
    sign=1. if capacities[0]>=capacities[1] else -1.
    capacity=max(capacities)
    return sign*min(P['difference_rad'],capacity*.5) if capacity>1e-9 else 0.

def motion(pred,robot):
    center=np.linalg.norm(np.diff(pred['geom'],axis=0),axis=-1)
    rotation=np.diff(pred['geom_rotation'],axis=0).reshape(-1,robot.model.ngeom,3,3)
    return float((center+np.linalg.norm(rotation,axis=(-1,-2))*robot.model.geom_rbound).max())

def metric(r):
    e,d=np.asarray(r['goal']),np.asarray(r['minima'])
    violation=np.minimum((d-P['repair_clearance_m'])/P['collision_scale_m'],0.)**2
    return {'goal_groups':group_norms(e),'Q':float(max(group_norms(e))),
            'minimum_clearance_m':float(min(d)),
            'collision_violation':float(violation.mean()+.25*violation.max()),
            'geometry_step_m':r.get('geometry_step_m')}

def accept(old,new):
    a,b=metric(old),metric(new)
    # Protect every physically-qualified component, even when another is not.
    if any(x<=1 and y>1 for x,y in zip(a['goal_groups'],b['goal_groups'])): return False
    if a['Q']<=1: return b['collision_violation'] < a['collision_violation']-1e-10
    return b['collision_violation']+.05*np.dot(new['goal'],new['goal']) < a['collision_violation']+.05*np.dot(old['goal'],old['goal'])-1e-10

class Evaluator:
    def __init__(self,task,robot,initial,collision,budget):
        self.task,self.robot,self.initial,self.cc,self.budget=task,robot,initial,collision,budget
        self.scratch=initial.reset(robot)
    def reference(self,z):
        c=self.task['condition']; return Reference(z,c['initial_features'],c['duration'])
    def predict(self,z,grid):
        n=int(grid) if np.isscalar(grid) else len(grid)-1
        self.budget.consume(n)
        if np.isscalar(grid): return predict_coupled(self.initial,self.reference(z),self.robot,n,self.budget.deadline)
        return predict_phases(self.initial,self.reference(z),self.robot,grid,self.budget.deadline)
    def dense(self,z,minimum_grid=512,matched_active=()):
        history=[]
        for n in P['dense_grids']:
            if n<minimum_grid:continue
            pred=self.predict(z,n); extent=motion(pred,self.robot)
            history.append({'intervals':n,'maximum_geometry_step_m':extent})
            if extent<=.002:break
        minima=[];pairs=[];values={}; matched=np.empty(len(matched_active)); by_phase={}
        for k,(t,ga,gb) in enumerate(matched_active):
            by_phase.setdefault(float(t),[]).append((k,ga,gb))
        assert all(t in pred['phase'] for t in by_phase)
        for j,state in enumerate(pred['qpos']):
            self.budget.check()
            self.scratch.qpos[:]=state; self.robot.kinematics(self.scratch)
            r=self.cc.scan(self.scratch,P['active_buffer_m'])
            minima.append(r['minimum_clearance_m']); pairs.append(r['minimum_pair'])
            for k,ga,gb in by_phase.get(float(pred['phase'][j]),[]):
                matched[k]=mujoco.mj_geomDistance(self.robot.model,self.scratch,ga,gb,2.5,None)
                self.robot.calls['distance']+=1
            for v in r['violations']:
                ga=self.robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_a'])
                gb=self.robot.id(mujoco.mjtObj.mjOBJ_GEOM,v['geom_b'])
                values[(float(pred['phase'][j]),ga,gb)]=v['distance_m']
        active=sorted(values,key=lambda k:(values[k],k))[:P['active_pair_cap']];active.sort()
        return {'goal':pose_vector(pred['ee'][-1],self.task['condition']['goals']),
          'ee':pred['ee'][-1],'qpos':pred['qpos'],'phase':pred['phase'],'minima':np.asarray(minima),
          'pairs':pairs,'active':active,'active_distances':np.array([values[k] for k in active]),'matched_distances':matched,
          'geometry_step_m':extent,'grid_history':history,'intervals':n,
          'reference_hash':self.reference(z).reference_hash}
    def sparse(self,z,phase,active):
        pred=self.predict(z,phase); ds=[]; last=None
        for t,ga,gb in active:
            j=int(np.searchsorted(phase,t)); assert phase[j]==t
            if j!=last:
                self.scratch.qpos[:]=pred['qpos'][j];self.robot.kinematics(self.scratch);last=j
            ds.append(float(mujoco.mj_geomDistance(self.robot.model,self.scratch,ga,gb,2.5,None)))
        self.robot.calls['distance']+=len(active)
        return {'goal':pose_vector(pred['ee'][-1],self.task['condition']['goals']),
                'active_distances':np.asarray(ds),'ee':pred['ee'][-1],'qpos':pred['qpos']}
    def feedback(self,z,current):
        checks=[]
        for n in P['sparse_base_grids']:
            phase=np.unique(np.r_[np.linspace(0,1,n+1),[v[0] for v in current['active']]])
            sparse=self.sparse(z,phase,current['active'])
            pos,rot=pose_errors(sparse['ee'],current['ee'])
            error=float(np.max(abs(sparse['active_distances']-current['active_distances']),initial=0))
            checks.append({'base_intervals':n,'actual_intervals':len(phase)-1,'position_m':pos,'rotation_rad':rot,'distance_difference_m':error})
            if max(pos)<=P['sparse_agreement_m'] and max(rot)<=P['sparse_agreement_rad'] and error<=P['sparse_agreement_m']:
                return sparse,phase,checks
        raise ValueError('SPARSE_DENSE_NUMERICAL_DISAGREEMENT')

def restore_feasible(candidate,feasible):
    """Convex linearized feasible set contains zero; preserve original constraints."""
    candidate=np.asarray(candidate)
    if not np.isfinite(candidate).all():return np.zeros_like(candidate),0.
    if feasible(candidate):return candidate.copy(),1.
    lo,hi=0.,1.
    assert feasible(np.zeros_like(candidate))
    for _ in range(P['restore_bisections']):
        mid=(lo+hi)/2
        if feasible(mid*candidate):lo=mid
        else:hi=mid
    alpha=lo*(1-1e-10)
    return alpha*candidate,alpha

def active_merit(e,distances,with_goal=False):
    v=np.minimum((distances-P['repair_clearance_m'])/P['collision_scale_m'],0.)**2
    return (float(v.mean()+.25*v.max()) if len(v) else 0.)+(.05*float(e@e) if with_goal else 0.)

def trust_decision(radius,accepted,ratio=None,scale=None,boundary=False):
    if not accepted or ratio<P['ratio_shrink'] or scale<1:
        return max(P['trust_min_rad'],radius/2)
    if ratio>P['ratio_expand'] and boundary:
        return min(P['trust_rad'],radius*2)
    return radius

def solve_step(z,directions,e,distances,jg,jd,lower,upper,radius=None):
    n=len(directions); radius=P['trust_rad'] if radius is None else radius
    matrix=directions.reshape(n,-1).T*radius
    G=jg*radius; D=jd*radius/P['collision_scale_m']
    b=(distances-P['repair_clearance_m'])/P['collision_scale_m']
    lo=np.maximum((np.broadcast_to(lower,z.shape)-z).ravel(),-radius)
    hi=np.minimum((np.broadcast_to(upper,z.shape)-z).ravel(),radius)
    limits=np.maximum(1.,group_norms(e))
    C=np.diff(directions,n=2,axis=1).reshape(n,-1).T*radius
    c0=np.diff(z,n=2,axis=0).ravel()
    def rawfun(u):
        g=e+G@u;v=np.minimum(b+D@u,0.)**2
        collision=float(v.mean()+.25*v.max()) if len(v) else 0.
        return collision+.01*float(g@g)+.01*float(np.mean((c0+C@u)**2))+.001*float(np.mean((matrix@u)**2))
    normalizer=max(1.,rawfun(np.zeros(n)))
    def fun(u):return rawfun(u)/normalizer
    def jac(u):
        g=e+G@u;v=np.minimum(b+D@u,0.)
        out=.02*G.T@g+.02*C.T@(c0+C@u)/len(c0)+.002*matrix.T@(matrix@u)/len(lo)
        if len(v):
            out+=2*D.T@v/len(v)
            k=int(np.argmax(v*v));out+=.5*v[k]*D[k]
        return out/normalizer
    def pose(u):return 1-np.sum((e+G@u).reshape(4,3)**2,axis=1)/(limits*limits)
    def pjac(u):return -2*np.einsum('gi,gij->gj',(e+G@u).reshape(4,3),G.reshape(4,3,n))/(limits[:,None]**2)
    def domain(u):return np.r_[matrix@u-lo,hi-matrix@u]
    def residual(u):
        return {'pose_min':float(min(pose(u))), 'control_min':float(min(domain(u))),
                'maximum_control_step_rad':float(abs(matrix@u).max())}
    def feasible(u):
        r=residual(u);return np.isfinite(u).all() and min(r['pose_min'],r['control_min'])>=-1e-8
    zero=np.zeros(n);assert feasible(zero),'current point must be feasible in linear model'
    best=zero.copy();bestvalue=fun(zero);callbacks=[];bestsource='ZERO'
    def callback(u):
        nonlocal best,bestvalue,bestsource
        ok=feasible(u);value=fun(u)
        restored,alpha=restore_feasible(u,feasible);rv=fun(restored)
        callbacks.append({'u':np.asarray(u).copy(),'feasible':ok,'objective':float(value),'residual':residual(u),
                          'restoration_alpha':alpha,'restored_objective':rv,'restored_residual':residual(restored)})
        if feasible(restored) and rv<bestvalue:
            best=restored.copy();bestvalue=rv;bestsource='RESTORED_RAY' if alpha<1 else 'FEASIBLE_ITERATE'
    constraints=[{'type':'ineq','fun':pose,'jac':pjac}]
    if n==323:
        bounds=list(zip(lo/radius,hi/radius))
    else:
        bounds=[(-np.sqrt(323),np.sqrt(323))]*n
        constraints.append({'type':'ineq','fun':domain,'jac':lambda u:np.vstack([matrix,-matrix])})
    solved=minimize(fun,zero,jac=jac,bounds=bounds,constraints=constraints,method='SLSQP',
                    callback=callback,options={'maxiter':P['solver_max_iterations'],'ftol':1e-9})
    callback(solved.x)
    step=radius*best
    return step,{'success':bool(solved.success),'message':str(solved.message),'iterations':int(solved.nit),
      'initialization':'ZERO_FEASIBLE','initial_residual':residual(zero),'objective_normalizer':normalizer,
      'initial_objective':fun(zero),'returned_objective':fun(best),'solved_u':solved.x,
      'solved_residual':residual(solved.x),'solved_feasible':bool(feasible(solved.x)),
      'returned_u':best,'returned_residual':residual(best),'returned_feasible':bool(feasible(best)),
      'returned_source':bestsource,'radius_rad':radius,
      'callbacks':callbacks,'pose_limits':limits,'trust_is_control_space':True,'variables':n,
      'returned_objective_gradient':jac(best),'solved_objective_gradient':jac(solved.x),
      'gradient_is_unconstrained_not_KKT':True}

def optimize(directory,start,kind,codec,ev,progress,max_iterations):
    z=np.asarray(start).copy();directions=basis(kind,codec);history=[];stop='ITERATION_LIMIT'
    radius=P['trust_initial_rad']
    current=ev.dense(z)
    atomic_json(directory/'initial.json',dict(metric(current),Z=z,grid_history=current['grid_history']))
    for iteration in range(max_iterations):
        progress('dense_feedback_search',path=directory.name,variant=kind,iteration=iteration+1)
        if metric(current)['Q']<=1 and min(current['minima'])>=P['repair_clearance_m'] and current['geometry_step_m']<=.002:
            stop='DENSE_SEARCH_CONDITION';break
        # Reserve a whole iteration and final validation; leave work completed if cap cannot fit.
        if not ev.budget.can_fit(len(directions)+50,(len(directions)+4)*2288+300000):
            stop='RESERVED_BUDGET_LIMIT';break
        center,phase,agreement=ev.feedback(z,current)
        ge=[];de=[];eps=[];ee=[];dd=[]
        for direction in directions:
            h=admissible_difference(z,direction,ev.robot.lower,ev.robot.upper)
            v=ev.sparse(z+h*direction,phase,current['active']) if h else center
            ge.append((v['goal']-center['goal'])/h if h else np.zeros(12))
            de.append((v['active_distances']-center['active_distances'])/h if h else np.zeros(len(current['active'])))
            eps.append(h);ee.append(v['ee']);dd.append(v['active_distances'])
        jg,jd=np.asarray(ge).T,np.asarray(de).T
        ev.robot.calls['sqp']+=1
        step,solver=solve_step(z,directions,center['goal'],center['active_distances'],jg,jd,ev.robot.lower,ev.robot.upper,radius)
        delta=np.einsum('i,ijk->jk',step,directions)
        assert max(abs(delta).ravel())<=P['trust_rad']+1e-8
        pd=directory/'iterations'/f'{iteration:02d}'
        before=z.copy();trials=[];accepted=False
        with_goal=metric(current)['Q']>1
        predicted_before=active_merit(center['goal'],center['active_distances'],with_goal)
        actual_before=active_merit(current['goal'],current['active_distances'],with_goal)
        for scale in P['line_scales'] if np.max(abs(delta))>1e-12 else []:
            trial=z+scale*delta
            gates=quick_gates(ev.robot,ev.reference(trial))
            if not all(gates.values()):
                trials.append({'scale':scale,'status':'QUICK_REJECTED','gates':gates});continue
            predicted_after=active_merit(center['goal']+jg@(step*scale),center['active_distances']+jd@(step*scale),with_goal)
            predicted_reduction=predicted_before-predicted_after
            value=ev.dense(trial,minimum_grid=current['intervals'],matched_active=current['active'])
            actual_after=active_merit(value['goal'],value['matched_distances'],with_goal)
            actual_reduction=actual_before-actual_after
            ratio=actual_reduction/predicted_reduction if predicted_reduction>1e-12 else None
            physical_descent=accept(current,value)
            ok=physical_descent and ratio is not None and ratio>=P['ratio_accept']
            trials.append(dict(metric(value),scale=scale,status='EVALUATED',accepted=ok,physical_descent=physical_descent,
                predicted_reduction=predicted_reduction,actual_reduction=actual_reduction,ratio=ratio,grid_history=value['grid_history']))
            atomic_npz(pd/f'trial_{len(trials):02d}.npz',Z=trial,phase=value['phase'],qpos=value['qpos'],
                       ee=value['ee'],minima=value['minima'],matched_distances=value['matched_distances'])
            atomic_json(pd/f'trial_{len(trials):02d}_pairs.json',value['pairs'])
            if ok:z=trial;after=value;accepted=True;break
        atomic_npz(pd/'linearization.npz',Z=before,next_Z=z,goal=center['goal'],current_ee=center['ee'],
          active_distances=center['active_distances'],steps=np.asarray(eps),perturb_ee=np.asarray(ee),
          perturb_distances=np.asarray(dd),jg=jg,jd=jd,step=step,phase=phase,
          sparse_qpos=center['qpos'],dense_active_distances=current['active_distances'],
          dense_phase=current['phase'],dense_qpos=current['qpos'],dense_minima=current['minima'],dense_ee=current['ee'])
        next_radius=trust_decision(radius,accepted,trials[-1].get('ratio') if accepted else None,
            trials[-1]['scale'] if accepted else None,np.max(abs(delta))>=.9*radius)
        row={'iteration':iteration,'accepted':accepted,'before':metric(current),'trials':trials,'solver':solver,
             'active_pairs':current['active'],'sparse_agreement':agreement,'before_Z_hash':digest(before),
             'after_Z_hash':digest(z),'direction_basis':kind,'grid_history':current['grid_history'],
             'radius_rad':radius,'next_radius_rad':next_radius,'no_nonzero_feasible_step':np.max(abs(delta))<=1e-12}
        atomic_json(pd/'record.json',row);history.append(row)
        if not accepted and radius<=P['trust_min_rad']:
            stop='MINIMUM_TRUST_NO_ACCEPTED_STEP';break
        radius=next_radius
        if accepted:current=after
    result=dict(metric(current),Z=z,iterations=len(history),kind=kind,stop=stop,
                status='SEARCH_COMPLETE',reference_hash=ev.reference(z).reference_hash,goal_relabel=False)
    atomic_npz(directory/'final_path.npz',Z=z,phase=current['phase'],qpos=current['qpos'],ee=current['ee'],minima=current['minima'])
    atomic_json(directory/'final_scan.json',{'pairs':current['pairs'],'grid_history':current['grid_history']})
    atomic_json(directory/'result.json',result)
    return result
