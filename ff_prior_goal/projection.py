"""Bounded low-noise clean-estimate repair through the full floating-base path.

This local nonlinear solver is not a global feasibility guarantee. Only the
unchanged independent model/execution gates can certify its returned proposal.
"""
import time
import numpy as np
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.spline import Reference, decode
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior.proposal_generator import quick_gates
from ff_prior_fixed.correction import pose_vector
from ff_prior.storage import digest

CONTRACT = {'version': 'coupled_goal_projection_1', 'directions': 17,
    'iterations_per_call': 4, 'calls_per_candidate': 2, 'phase_intervals': 128,
    'difference_rad': 1e-4, 'damping': 1e-3, 'maximum_parameter_step_rad': .1,
    'line_scales': [1., .5, .25, .125], 'stop_position_m': .001,
    'stop_rotation_rad': float(np.deg2rad(.2)), 'curvature_weight': 10.,
    'frozen_goal_relabel': False, 'validation_gates_changed': False}


def smooth(z, q0, weight=10.):
    """Quadratic control-polygon curvature regularizer with exact endpoints.

    The first/last three controls remain fixed, preserving zero endpoint rates.
    No private terminal witness is used: the proposed terminal is retained.
    """
    z = np.asarray(z, dtype=np.float64)
    c = decode(z, q0)
    d = np.diff(np.eye(24), n=2, axis=0)
    a = d[:, 3:21]
    fixed = c.copy(); fixed[3:21] = 0.
    out = z.copy()
    out[:18] = np.linalg.solve(np.eye(18) + weight*a.T@a,
                              z[:18] - weight*a.T@(d@fixed))
    return out


class GoalProjector:
    def __init__(self, task, robot, initial, collision, profile, deadline,
                 goal_enabled=True):
        assert_task_compatible(task, robot, initial, collision, profile)
        self.task, self.robot, self.initial = task, robot, initial
        self.profile, self.deadline = profile, deadline
        self.goal_enabled = goal_enabled
        self.calls = 0
        s = np.r_[np.linspace(0, 1, 20)[1:-1], 1.]
        self.h = (s*s*(3-2*s))[:, None]
        self.directions = np.eye(17)[:, None, :]*self.h[None, :, :]

    def __call__(self, z, iteration):
        assert_frozen(self.task)
        before_hash = digest(self.task)
        raw = np.asarray(z, dtype=np.float64)
        c = self.task['condition']; q0 = np.asarray(c['initial_features'])
        proposal = smooth(raw, q0, CONTRACT['curvature_weight'])
        proposal = np.clip(proposal, self.robot.lower, self.robot.upper)
        initial_proposal = proposal.copy()
        probes = 0
        def evaluate(value):
            nonlocal probes
            if time.time() >= self.deadline: raise RuntimeError('PROJECTION_WALL_BUDGET_EXHAUSTED')
            ref = Reference(value, q0, c['duration'])
            predicted = predict_coupled(self.initial, ref, self.robot, 128, self.deadline)
            probes += 1
            ee = predicted['ee'][-1]
            return pose_vector(ee, c['goals']), ee
        history = []
        if self.goal_enabled:
            error, actual = evaluate(proposal)
            initial_error = error.copy()
            for _ in range(CONTRACT['iterations_per_call']):
                pos, rot = pose_errors(actual, c['goals'])
                if max(pos) <= CONTRACT['stop_position_m'] and max(rot) <= CONTRACT['stop_rotation_rad']: break
                eps = CONTRACT['difference_rad']
                j = np.stack([(evaluate(proposal+eps*d)[0]-evaluate(proposal-eps*d)[0])/(2*eps)
                              for d in self.directions], axis=1)
                step = -j.T@np.linalg.solve(j@j.T+CONTRACT['damping']*np.eye(12), error)
                step *= min(1., CONTRACT['maximum_parameter_step_rad']/max(np.linalg.norm(step), 1e-12))
                before = float(error@error); trials = []; accepted = False
                for scale in CONTRACT['line_scales']:
                    trial = np.clip(proposal+scale*self.h*step, self.robot.lower, self.robot.upper)
                    if not all(quick_gates(self.robot, Reference(trial, q0, c['duration'])).values()):
                        trials.append({'scale': scale, 'status': 'QUICK_REJECTED'}); continue
                    ne, na = evaluate(trial); cost = float(ne@ne)
                    trials.append({'scale': scale, 'cost': cost})
                    if cost < before:
                        proposal, error, actual = trial, ne, na; accepted = True; break
                history.append({'before_cost': before, 'after_cost': float(error@error),
                    'jacobian_rank': int(np.linalg.matrix_rank(j)),
                    'singular_values': np.linalg.svd(j, compute_uv=False), 'accepted': accepted,
                    'parameter_step_norm': float(np.linalg.norm(step)), 'trials': trials})
                if not accepted: break
            pos, rot = pose_errors(actual, c['goals'])
            errors = {'initial_normalized_pose_error': initial_error, 'final_normalized_pose_error': error,
                      'position_error_m': pos, 'rotation_error_rad': rot}
        else:
            errors = {}
        if digest(self.task) != before_hash: raise ValueError('frozen task mutated')
        self.calls += 1
        return proposal, {'backend': 'COUPLED_GOAL_PROJECTION' if self.goal_enabled else 'SMOOTH_ONLY',
            'iteration': iteration, 'goal_enabled': self.goal_enabled, 'history': history,
            'smoothing_norm_rad': float(np.linalg.norm(initial_proposal-raw)),
            'goal_correction_norm_rad': float(np.linalg.norm(proposal-initial_proposal)),
            'proposal_hash': digest(raw), 'output_hash': digest(proposal),
            'task_hash': self.task['task_hash'], 'coupled_calls': probes,
            'private_teacher_fields_used': False, 'all_physical_gates_still_required': True, **errors}
