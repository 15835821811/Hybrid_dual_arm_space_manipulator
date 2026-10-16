"""Bounded coupled repair with hard goal constraints and verified incumbents."""
import time
import numpy as np
from scipy.optimize import lsq_linear, minimize
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_goal.projection import smooth
from ff_prior_fixed.correction import pose_vector
from ff_prior.storage import digest

CONTRACT = {'version':'goal_preserving_repair_1', 'repair_queries':110,
    'phase_intervals':128, 'geometry_stride':1, 'directions':24, 'iterations':4,
    'difference_rad':1e-4, 'maximum_coefficient_step_rad':.15,
    'line_scales':[1.,.5,.25], 'goal_inner_fraction':.65,
    'goal_accept_fraction':.8, 'collision_repair_buffer_m':.0002,
    'collision_scale_m':.003, 'schedule':[15], 'curvature_weight':10.,
    'goal_feasible_incumbent_cannot_be_sacrificed':True,
    'acceptance_clearance_m':0., 'geometry_step_budget_m':.002}


def group_norms(error):
    return np.linalg.norm(np.asarray(error).reshape(4,3),axis=1)


def accept_update(old_goal, old_dist, new_goal, new_dist, collision_enabled):
    """Nonlinear acceptance, independent of a solver's success flag."""
    old_max, new_max = max(group_norms(old_goal)), max(group_norms(new_goal))
    if old_max <= CONTRACT['goal_accept_fraction']:
        if new_max > CONTRACT['goal_accept_fraction']: return False
        if not collision_enabled: return new_goal @ new_goal < old_goal @ old_goal
        violation=lambda d: float(np.sum(np.maximum(CONTRACT['collision_repair_buffer_m']-d,0.)**2))
        return violation(new_dist) < violation(old_dist)-1e-16
    return new_max < old_max-1e-9


def constrained_step(e, distances, jg, jd, collision_enabled):
    """Hard four pose-norm constraints in the local model, not a weighted tradeoff."""
    start=lsq_linear(np.vstack([jg,np.sqrt(.001)*np.eye(24)]),
        np.r_[-e,np.zeros(24)],bounds=(-.15,.15),tol=1e-7,max_iter=100).x
    if not collision_enabled:
        return start, {'solver':'bounded_goal_lsq','success':True}
    scale=CONTRACT['collision_scale_m']; buffer=CONTRACT['collision_repair_buffer_m']
    def fun(x):
        g=e+jg@x; v=np.minimum((distances+jd@x-buffer)/scale,0.)
        return float(v@v+.01*(g@g)+.001*(x@x))
    def jac(x):
        g=e+jg@x; v=np.minimum((distances+jd@x-buffer)/scale,0.)
        return 2*jd.T@v/scale+.02*jg.T@g+.002*x
    def constraint(x):
        return CONTRACT['goal_inner_fraction']**2-np.sum((e+jg@x).reshape(4,3)**2,axis=1)
    def constraint_jac(x):
        return -2*np.einsum('gi,gij->gj',(e+jg@x).reshape(4,3),jg.reshape(4,3,24))
    solved=minimize(fun,start,jac=jac,bounds=[(-.15,.15)]*24,
        constraints={'type':'ineq','fun':constraint,'jac':constraint_jac},method='SLSQP',
        options={'maxiter':80,'ftol':1e-9})
    feasible=bool(np.isfinite(solved.x).all() and min(constraint(solved.x))>=-1e-7)
    return (solved.x if feasible else start), {'solver':'pose_constrained_collision_sqp',
        'success':bool(solved.success),'linear_pose_feasible':feasible,'message':str(solved.message),
        'fallback_goal_only':not feasible,'iterations':int(solved.nit)}


class FeasibleProjector:
    def __init__(self,task,robot,initial,collision,profile,deadline,collision_enabled=True):
        assert_task_compatible(task,robot,initial,collision,profile)
        self.task,self.robot,self.initial,self.collision=task,robot,initial,collision
        self.deadline,self.collision_enabled=deadline,bool(collision_enabled)
        self.used=0; self.geometry_states=0
        s=np.r_[np.linspace(0,1,20)[1:-1],1.]; h=s*s*(3-2*s); bow=np.sin(np.pi*s); bow[-1]=0.
        self.directions=np.concatenate([np.eye(17)[:,None,:]*h[None,:,None],np.eye(17)[10:,None,:]*bow[None,:,None]])
        self.scratch=initial.reset(robot)

    def evaluate(self,z):
        if self.used>=110 or time.time()>=self.deadline: raise RuntimeError('REPAIR_BUDGET_EXHAUSTED')
        self.used+=1; c=self.task['condition']
        pred=predict_coupled(self.initial,Reference(z,c['initial_features'],c['duration']),self.robot,128,self.deadline)
        e=pose_vector(pred['ee'][-1],c['goals']); distances=[]
        if self.collision_enabled:
            for state in pred['qpos']:
                if time.time()>=self.deadline: raise RuntimeError('REPAIR_WALL_BUDGET_EXHAUSTED')
                self.scratch.qpos[:]=state; self.robot.kinematics(self.scratch)
                distances.append(self.collision.scan(self.scratch,0.)['minimum_clearance_m']); self.geometry_states+=1
        return e,np.asarray(distances),pred['ee'][-1]

    def __call__(self,z,iteration):
        if iteration!=15: return z,{'iteration':iteration,'active':False,'coupled_calls':0}
        assert_frozen(self.task); c=self.task['condition']; raw=np.asarray(z,dtype=np.float64)
        if not np.isfinite(raw).all(): raise ValueError('nonfinite clean estimate')
        proposal=np.clip(smooth(raw,c['initial_features']),self.robot.lower,self.robot.upper)
        e,d,actual=self.evaluate(proposal); initial_error=e.copy(); history=[]
        for it in range(4):
            if max(group_norms(e))<=.65 and (not self.collision_enabled or min(d)>=.0002): break
            if 110-self.used<25: break
            values=[self.evaluate(proposal+1e-4*direction) for direction in self.directions]
            jg=np.stack([(v[0]-e)/1e-4 for v in values],axis=1)
            jd=np.stack([(v[1]-d)/1e-4 for v in values],axis=1)
            step,solver=constrained_step(e,d,jg,jd,self.collision_enabled)
            delta=np.einsum('i,ijk->jk',step,self.directions); accepted=False; trials=[]
            before=group_norms(e).copy()
            for scale in (1.,.5,.25):
                if self.used>=110: break
                trial=np.clip(proposal+scale*delta,self.robot.lower,self.robot.upper)
                if not all(quick_gates(self.robot,Reference(trial,c['initial_features'],c['duration'])).values()):
                    trials.append({'scale':scale,'status':'QUICK_REJECTED'}); continue
                ne,nd,na=self.evaluate(trial)
                accepted=bool(accept_update(e,d,ne,nd,self.collision_enabled))
                trials.append({'scale':scale,'pose_group_norms':group_norms(ne),
                    'minimum_clearance_m':float(min(nd)) if len(nd) else None,'accepted':accepted})
                if accepted: proposal,e,d,actual=trial,ne,nd,na; break
            history.append({'iteration':it,'before_pose_norms':before,'after_pose_norms':group_norms(e),
                'accepted':accepted,'solver':solver,'trials':trials,'goal_jacobian_rank':int(np.linalg.matrix_rank(jg))})
            if not accepted: break
        pos,rot=pose_errors(actual,c['goals'])
        return proposal,{'iteration':15,'active':True,'coupled_calls':self.used,'total_repair_queries':self.used,
            'reserved_integration_intervals':128*self.used,'geometry_states':self.geometry_states,
            'collision_enabled':self.collision_enabled,'position_error_m':pos,'rotation_error_rad':rot,
            'minimum_guidance_clearance_m':float(min(d)) if len(d) else None,
            'initial_goal_error':initial_error,'final_goal_error':e,'history':history,
            'input_hash':digest(raw),'output_hash':digest(proposal),'task_hash':self.task['task_hash'],
            'private_witness_used':False,'full_path_coupled':True,'guidance_grid_is_not_acceptance':True}
