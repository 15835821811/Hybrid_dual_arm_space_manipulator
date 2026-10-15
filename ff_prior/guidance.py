"""Auditable eight-direction full-path finite differences, DEV ONLY."""
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import assert_frozen,assert_task_compatible
from .spline import Reference
from .coupled_predictor import predict_coupled,pose_errors
from .model_gates import path_geometry
from .storage import digest


class FullPathCost:
    def __init__(self,task,robot,initial,collision,profile,deadline=None):
        assert_frozen(task)
        assert_task_compatible(task,robot,initial,collision,profile)
        self.task,self.robot,self.initial,self.collision,self.profile = task,robot,initial,collision,profile
        self.calls = 0
        self.deadline=deadline

    def __call__(self,z):
        if self.deadline is not None and time.time()>=self.deadline:
            raise RuntimeError("BUDGET_EXHAUSTED")
        assert_frozen(self.task)
        self.calls += 1
        r = self.robot
        ref = Reference(z,np.asarray(self.task["condition"]["initial_features"]),self.task["condition"]["duration"])
        prediction = predict_coupled(self.initial,ref,r,128,self.deadline)
        pos,rot = pose_errors(prediction["ee"][-1],self.task["condition"]["goals"])
        geometry = path_geometry(r,self.initial,prediction["qpos"],self.collision,self.profile.planning_clearance_m,self.deadline)
        lower,upper = ref.bounds()
        domain = np.maximum(r.lower-lower,0)**2+np.maximum(upper-r.upper,0)**2
        motion = 0.
        for order,limits in ((1,r.spec.planner_velocity_limits),(2,r.spec.planner_acceleration_limits)):
            lo,hi = ref.bounds(order)
            motion += float(np.sum(np.maximum(np.maximum(abs(lo),abs(hi))/limits-1,0)**2))
        # All terms use the complete reference; geometry is evaluated at every
        # propagation state. Final acceptance separately uses convergence grids.
        cost = float(np.sum((pos/.005)**2)+np.sum((rot/np.deg2rad(1))**2)
            +10000*np.sum(domain)+10000*motion+1e6*max(0,self.profile.planning_clearance_m-geometry["minimum_clearance_m"])**2)
        return cost,{"position_error_m":pos,"rotation_error_rad":rot,"minimum_clearance_m":geometry["minimum_clearance_m"],
                     "phase_intervals":128,"coupled_entire_path":True}


class SubspaceGuidance:
    def __init__(self,cost,seed=20261015,epsilon=1e-3):
        self.cost,self.epsilon = cost,epsilon
        rng = np.random.default_rng(seed)
        # Smooth directions span internal paths AND a shared terminal vector.
        s = np.linspace(0,1,19)
        raw = []
        for _ in range(8):
            spatial = rng.normal(size=17)
            temporal = s + rng.uniform(-.5,.5)*np.sin(np.pi*s)
            raw.append((temporal[:,None]*spatial).ravel())
        q,_ = np.linalg.qr(np.stack(raw,axis=1))
        self.directions = q.T.reshape(8,19,17)
        self.direction_hash = digest(self.directions)

    def __call__(self,z,iteration):
        wall = time.perf_counter()
        start = self.cost.calls
        projected = np.clip(z,self.cost.robot.lower,self.cost.robot.upper)
        before,details = self.cost(projected)
        derivatives = []
        for d in self.directions:
            plus,_ = self.cost(projected+self.epsilon*d)
            minus,_ = self.cost(projected-self.epsilon*d)
            derivatives.append((plus-minus)/(2*self.epsilon))
        gradient = np.einsum("i,ijk->jk",derivatives,self.directions)
        norm = np.linalg.norm(gradient)
        update = -gradient/max(norm,1e-12)*min(.15,norm*.001)
        best,best_z = before,projected
        trials = []
        for scale in (1.,.5,.25):
            candidate = np.clip(projected+scale*update,self.cost.robot.lower,self.cost.robot.upper)
            value,_ = self.cost(candidate)
            trials.append(value)
            if value < best:
                best,best_z = value,candidate
                break
        # Independent step-halving check of one direction is DEV diagnostics,
        # not a claim of exact full-dimensional gradients.
        half_plus,_ = self.cost(projected+self.epsilon/2*self.directions[0])
        half_minus,_ = self.cost(projected-self.epsilon/2*self.directions[0])
        half_derivative = (half_plus-half_minus)/self.epsilon
        return best_z,{"backend":"SUBSPACE_FD_DEV_ONLY","iteration":iteration,"directions":8,
            "direction_hash":self.direction_hash,"centered_evaluations":16,"total_cost_evaluations":self.cost.calls-start,
            "before_cost":before,"after_cost":best,"update_norm":float(np.linalg.norm(best_z-z)),
            "domain_projection_norm":float(np.linalg.norm(projected-z)),"epsilon":self.epsilon,
            "first_direction_derivative":derivatives[0],"half_step_derivative":half_derivative,
            "step_halving_absolute_difference":abs(derivatives[0]-half_derivative),
            "trial_costs":trials,"wall_s":time.perf_counter()-wall,"full_exact_gradient":False,
            "production_eligible":False,"initial_cost_terms":details}


def public_goal_initializer(task,robot,initial):
    """Local generalized-Jacobian seed using only public initial state/goals.

This is an initializer, not the guidance backend. Its seed remains mutable.
"""
    assert_frozen(task)
    d = initial.reset(robot)
    robot.kinematics(d,mass=True)
    M = robot.mass(d)
    a,b = robot.dof_ids,robot.base_dofs
    tangent = np.zeros((robot.model.nv,17))
    tangent[a] = robot.B
    tangent[b] = -np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@robot.B)
    actual,goals = robot.ee(d),np.asarray(task["condition"]["goals"])
    jacobians,errors = [],[]
    for k,body in enumerate((robot.tip_c,robot.tip_r)):
        jp,jr = np.zeros((3,robot.model.nv)),np.zeros((3,robot.model.nv))
        mujoco.mj_jac(robot.model,d,jp,jr,actual[k,:3,3],body)
        jacobians.extend([jp@tangent,jr@tangent])
        errors.extend([goals[k,:3,3]-actual[k,:3,3],Rotation.from_matrix(goals[k,:3,:3]@actual[k,:3,:3].T).as_rotvec()])
    J,e = np.vstack(jacobians),np.concatenate(errors)
    delta = np.linalg.solve(J.T@J+1e-4*np.eye(17),J.T@e)
    terminal = np.clip(robot.spec.planner_zero+np.clip(delta,-.3,.3),robot.lower,robot.upper)
    s = np.linspace(0,1,20)[1:-1]
    return np.vstack([robot.spec.planner_zero+(s*s*(3-2*s))[:,None]*(terminal-robot.spec.planner_zero),terminal])
