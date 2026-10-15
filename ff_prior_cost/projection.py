"""Shared query-budgeted goal repair; no changes to physical acceptance."""
import time
import numpy as np
from ff_prior_goal.projection import smooth, CONTRACT as OLD_CONTRACT
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior_fixed.correction import pose_vector
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.storage import digest

CONTRACT = dict(OLD_CONTRACT, version='shared_budget_projection_1',
    repair_queries_per_candidate=110, validation_query_reserve=3,
    total_coupled_queries_per_candidate=113, iterations_per_call=4,
    allocation='remaining budget; two-call method reserves 36 for final call')


class Projector:
    def __init__(self, task, robot, initial, collision, profile, deadline, schedule=(15,), observer=None):
        assert_task_compatible(task, robot, initial, collision, profile)
        self.task, self.robot, self.initial = task, robot, initial
        self.deadline, self.schedule, self.observer = deadline, tuple(schedule), observer
        self.used = 0
        s = np.r_[np.linspace(0, 1, 20)[1:-1], 1.]
        self.h = (s*s*(3-2*s))[:, None]
        self.directions = np.eye(17)[:, None, :]*self.h[None, :, :]

    def __call__(self, z, iteration):
        assert_frozen(self.task)
        if iteration not in self.schedule:
            return z, {'iteration': iteration, 'active': False, 'coupled_calls': 0}
        c = self.task['condition']; raw = np.asarray(z, dtype=np.float64)
        if not np.isfinite(raw).all(): raise ValueError('nonfinite repair input')
        start = self.used
        cap = CONTRACT['repair_queries_per_candidate']-(36 if iteration != self.schedule[-1] else 0)
        proposal = np.clip(smooth(raw, c['initial_features']), self.robot.lower, self.robot.upper)
        smoothed = proposal.copy()
        def evaluate(value):
            if self.used >= cap: raise RuntimeError('repair query budget exceeded')
            if time.time() >= self.deadline: raise RuntimeError('repair wall budget exhausted')
            self.used += 1
            p = predict_coupled(self.initial, Reference(value, c['initial_features'], c['duration']), self.robot, 128, self.deadline)
            return pose_vector(p['ee'][-1], c['goals']), p['ee'][-1]
        history = []
        error, actual = evaluate(proposal); first = error.copy()
        for _ in range(CONTRACT['iterations_per_call']):
            pos, rot = pose_errors(actual, c['goals'])
            if max(pos) <= .001 and max(rot) <= np.deg2rad(.2): break
            # Never compute a partial Jacobian or silently borrow the validation reserve.
            if cap-self.used < 35: break
            eps = CONTRACT['difference_rad']
            j = np.stack([(evaluate(proposal+eps*d)[0]-evaluate(proposal-eps*d)[0])/(2*eps)
                          for d in self.directions], axis=1)
            step = -j.T@np.linalg.solve(j@j.T+CONTRACT['damping']*np.eye(12), error)
            step *= min(1., .1/max(np.linalg.norm(step), 1e-12))
            cost = float(error@error); trials = []; accepted = False
            for scale in CONTRACT['line_scales']:
                if self.used >= cap: break
                trial = np.clip(proposal+scale*self.h*step, self.robot.lower, self.robot.upper)
                if not all(quick_gates(self.robot, Reference(trial, c['initial_features'], c['duration'])).values()):
                    trials.append({'scale': scale, 'status': 'QUICK_REJECTED'}); continue
                ne, na = evaluate(trial); trials.append({'scale': scale, 'cost': float(ne@ne)})
                if ne@ne < cost:
                    proposal, error, actual = trial, ne, na; accepted = True; break
            history.append({'before_cost': cost, 'after_cost': float(error@error),
                'jacobian_rank': int(np.linalg.matrix_rank(j)), 'accepted': accepted,
                'parameter_step_norm': float(np.linalg.norm(step)), 'trials': trials})
            if not accepted: break
        if self.observer:
            self.observer(raw, iteration, 'before_smoothing')
            self.observer(smoothed, iteration, 'after_smoothing')
            self.observer(proposal, iteration, 'after_repair')
        pos, rot = pose_errors(actual, c['goals'])
        assert self.used <= 110
        return proposal, {'iteration': iteration, 'active': True, 'coupled_calls': self.used-start,
            'total_repair_queries': self.used, 'call_query_ceiling': cap, 'history': history,
            'initial_normalized_pose_error': first, 'final_normalized_pose_error': error,
            'position_error_m': pos, 'rotation_error_rad': rot,
            'smoothing_norm_rad': float(np.linalg.norm(smoothed-raw)),
            'goal_correction_norm_rad': float(np.linalg.norm(proposal-smoothed)),
            'proposal_hash': digest(raw), 'output_hash': digest(proposal), 'task_hash': self.task['task_hash'],
            'private_teacher_fields_used': False, 'all_physical_gates_still_required': True}
