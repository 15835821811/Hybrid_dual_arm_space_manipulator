"""Private 17D QP endpoint rows for a frozen ten-step shape-domain ramp.

The box is a necessary kinematic condition. The separate pre-servo MuJoCo
branch remains responsible for checking every realized 2 ms microstate.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from v6_lite.b2_cholesky_unconstrained import CholeskyUnconstrainedScreenQP
from v6_lite.execution_ramp import ramp_mean_weights
from v6_lite.safety_contract import ExecutionMode, FailureReason


def ramp_endpoint_velocity_box(
    shape_q: np.ndarray, start_velocity: np.ndarray,
    work_lower: np.ndarray, work_upper: np.ndarray,
    task_period_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return admissible 10D endpoint rates for q+T(.45u0+.55u1) in domain."""
    shape_q = np.asarray(shape_q, dtype=np.float64)
    start_velocity = np.asarray(start_velocity, dtype=np.float64)
    work_lower = np.asarray(work_lower, dtype=np.float64)
    work_upper = np.asarray(work_upper, dtype=np.float64)
    if (any(value.shape != (10,) for value in (
            shape_q, start_velocity, work_lower, work_upper))
            or any(not np.all(np.isfinite(value)) for value in (
                shape_q, start_velocity, work_lower, work_upper))
            or not np.isfinite(task_period_s) or task_period_s <= 0
            or np.any(work_lower >= work_upper)):
        raise ValueError("invalid work-domain endpoint box input")
    old_weight, new_weight = ramp_mean_weights()
    denom = task_period_s * new_weight
    offset = shape_q + task_period_s * old_weight * start_velocity
    return (work_lower - offset) / denom, (work_upper - offset) / denom


class WorkDomainBoundedCholeskyQP(CholeskyUnconstrainedScreenQP):
    """Add terminal-domain rows to the existing QP, preserving ramp-rate checks."""

    def solve(self, data, *args, **kwargs):
        planner_q = self.spec.low_level_to_planner @ data.qpos[self.qpos_ids]
        start = kwargs.get("ramp_start_velocity")
        if start is None:
            start = self.previous_velocity
        domain = self.interval_evaluator.shape_spec
        self._domain_rate_lower, self._domain_rate_upper = ramp_endpoint_velocity_box(
            planner_q[:10], np.asarray(start)[:10],
            domain.work_domain_lower_rad, domain.work_domain_upper_rad,
            self.config.task_period_s,
        )
        before = self.previous_velocity.copy()
        result = super().solve(data, *args, **kwargs)
        candidate = result.solver_candidate[:10]
        minimum_rate_slack = min(
            float(np.min(candidate - self._domain_rate_lower)),
            float(np.min(self._domain_rate_upper - candidate)))
        self.last_domain_endpoint_rate_slack_rad_s = minimum_rate_slack
        if result.planner_velocity is not None and minimum_rate_slack < -1e-12:
            self.previous_velocity = before
            self._previous_constraint_dual.clear()
            validation = replace(
                result.action_validation, mode=ExecutionMode.UNCERTIFIED,
                failure_reason=FailureReason.CANDIDATE_VIOLATION,
                selected_command=None, candidate_valid=False,
                selected_clearance_min_slack_m_s=float("nan"),
                selected_velocity_min_slack_rad_s=float("nan"),
            )
            return replace(result, planner_velocity=None,
                           action_validation=validation)
        return result

    def _solve_qp_admm(self, hessian, linear, matrix, lower, upper,
                       initial, initial_dual=None):
        endpoint_rows = np.eye(17, dtype=np.float64)[:10]
        augmented_matrix = np.vstack((matrix, endpoint_rows))
        augmented_lower = np.concatenate((lower, self._domain_rate_lower))
        augmented_upper = np.concatenate((upper, self._domain_rate_upper))
        augmented_dual = (None if initial_dual is None else np.concatenate((
            initial_dual, np.zeros(10, dtype=np.float64))))
        value, feasible, status, iterations, dual = super()._solve_qp_admm(
            hessian, linear, augmented_matrix, augmented_lower,
            augmented_upper, initial, augmented_dual)
        return value, feasible, status, iterations, dual[:len(lower)]
