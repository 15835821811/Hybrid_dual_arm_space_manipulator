"""Shared full-path goal/collision repair inside the final DDIM clean step."""
import time
import numpy as np
from scipy.optimize import lsq_linear
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_goal.projection import smooth
from ff_prior_fixed.correction import pose_vector
from ff_prior.storage import digest

CONTRACT={'version':'full_path_joint_repair_1','repair_queries':110,'validation_reserve':3,
    'phase_intervals':128,'geometry_stride':4,'directions':24,'iterations':4,
    'difference_rad':1e-4,'difference':'forward_full_coupled','damping':.001,
    'maximum_each_coefficient_step_rad':.15,'line_scales':[1.,.5,.25],
    'collision_activation_m':.003,'collision_weight_sqrt':3.,
    'geometry_extent_activation_at_128_m':.0072,'geometry_extent_weight_sqrt':3.,
    'goal_stop_m':.001,'goal_stop_rad':float(np.deg2rad(.2)),
    'curvature_weight':10.,'schedule':[15],'physical_acceptance_unchanged':True}


class JointProjector:
    def __init__(self,task,robot,initial,collision,profile,deadline,collision_enabled=True):
        assert_task_compatible(task,robot,initial,collision,profile)
        self.task,self.robot,self.initial,self.collision=task,robot,initial,collision
        self.deadline,self.collision_enabled=deadline,bool(collision_enabled)
        self.used=0; self.geometry_states=0
        s=np.r_[np.linspace(0,1,20)[1:-1],1.]
        h=s*s*(3-2*s); bow=np.sin(np.pi*s); bow[-1]=0.
        # Smooth temporal shapes keep a tractable local search space; endpoint and path vary independently.
        self.directions=np.concatenate([np.eye(17)[:,None,:]*h[None,:,None],np.eye(17)[10:,None,:]*bow[None,:,None]])
        self.scratch=initial.reset(robot)

    def evaluate(self,z):
        if self.used>=110 or time.time()>=self.deadline: raise RuntimeError('REPAIR_BUDGET_EXHAUSTED')
        self.used+=1; c=self.task['condition']
        pred=predict_coupled(self.initial,Reference(z,c['initial_features'],c['duration']),self.robot,128,self.deadline)
        e=pose_vector(pred['ee'][-1],c['goals'])
        distances=[]
        if self.collision_enabled:
            for state in pred['qpos'][::4]:
                if time.time()>=self.deadline: raise RuntimeError('REPAIR_WALL_BUDGET_EXHAUSTED')
                self.scratch.qpos[:]=state; self.robot.kinematics(self.scratch)
                report=self.collision.scan(self.scratch,0.)
                distances.append(report['minimum_clearance_m']); self.geometry_states+=1
            violation=3.*np.maximum(.003-np.array(distances),0.)/.003
            e=np.r_[e,violation]
        # Penalize excessive swept geometry motion on the predictor grid.
        # This is a proposal cost, NOT a substitute for the independent 2 mm gate.
        centers=np.linalg.norm(np.diff(pred['geom'],axis=0),axis=-1)
        rotations=np.diff(pred['geom_rotation'],axis=0).reshape(-1,self.robot.model.ngeom,3,3)
        extent=(centers+np.linalg.norm(rotations,axis=(-1,-2))*self.robot.model.geom_rbound).max(axis=1)
        e=np.r_[e,3.*np.maximum(extent-.0072,0.)/.008]
        return e,pred['ee'][-1],min(distances) if distances else None

    def __call__(self,z,iteration):
        if iteration!=15: return z,{'iteration':iteration,'active':False,'coupled_calls':0}
        assert_frozen(self.task); start=self.used; c=self.task['condition']; robot=self.robot
        raw=np.asarray(z,dtype=np.float64)
        if not np.isfinite(raw).all(): raise ValueError('nonfinite clean estimate')
        proposal=np.clip(smooth(raw,c['initial_features']),robot.lower,robot.upper)
        e,actual,clearance=self.evaluate(proposal); first=e.copy(); history=[]
        for it in range(4):
            pos,rot=pose_errors(actual,c['goals'])
            if max(pos)<=.001 and max(rot)<=np.deg2rad(.2) and not np.any(e[-128:]) and (not self.collision_enabled or clearance>=.003): break
            if 110-self.used<25: break
            eps=CONTRACT['difference_rad']
            j=np.stack([(self.evaluate(proposal+eps*d)[0]-e)/eps for d in self.directions],axis=1)
            # Bound each coordinate inside the solve. Scaling an ill-conditioned
            # unconstrained vector suppresses all useful goal/collision directions.
            solved=lsq_linear(np.vstack([j,np.sqrt(.001)*np.eye(24)]),
                np.r_[-e,np.zeros(24)],bounds=(-.15,.15),tol=1e-7,max_iter=100)
            step=solved.x
            delta=np.einsum('i,ijk->jk',step,self.directions)
            old=float(e@e); trials=[]; accepted=False
            for scale in (1.,.5,.25):
                if self.used>=110: break
                trial=np.clip(proposal+scale*delta,robot.lower,robot.upper)
                if not all(quick_gates(robot,Reference(trial,c['initial_features'],c['duration'])).values()):
                    trials.append({'scale':scale,'status':'QUICK_REJECTED'}); continue
                ne,na,nc=self.evaluate(trial); cost=float(ne@ne)
                trials.append({'scale':scale,'cost':cost,'minimum_clearance_m':nc})
                if cost<old:
                    proposal,e,actual,clearance=trial,ne,na,nc; accepted=True; break
            history.append({'iteration':it,'before_cost':old,'after_cost':float(e@e),
                'accepted':accepted,'goal_jacobian_rank':int(np.linalg.matrix_rank(j[:12])),
                'coefficient_step_norm':float(np.linalg.norm(step)),'trials':trials})
            if not accepted: break
        pos,rot=pose_errors(actual,c['goals'])
        return proposal,{'iteration':15,'active':True,'coupled_calls':self.used-start,'total_repair_queries':self.used,
            'collision_enabled':self.collision_enabled,'geometry_states':self.geometry_states,
            'minimum_guidance_clearance_m':clearance,'position_error_m':pos,'rotation_error_rad':rot,
            'initial_normalized_residual':first,'final_normalized_residual':e,'history':history,
            'input_hash':digest(raw),'output_hash':digest(proposal),'task_hash':self.task['task_hash'],
            'private_witness_used':False,'full_path_coupled':True,'guidance_grid_is_not_acceptance':True}
