"""Local execution gate for the existing 17-coordinate velocity QP.

The checks concern the *current linearized rows* and the commanded velocity
ramp. They do not certify MuJoCo dynamics or intersample collision clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ExecutionMode(str, Enum):
    TRACK = "TRACK"
    CAUTION = "CAUTION"
    BACKUP = "BACKUP"
    UNCERTIFIED = "UNCERTIFIED"


class FailureReason(str, Enum):
    NONE = "none"
    STALE_STATE = "stale_state"
    STALE_TARGET = "stale_target"
    EXPIRED_COMMAND = "expired_command"
    NONFINITE_CANDIDATE = "nonfinite_candidate"
    QP_INFEASIBLE = "qp_infeasible"
    ITERATION_LIMIT = "iteration_limit"
    SOLVER_FAILURE = "solver_failure"
    CANDIDATE_VIOLATION = "candidate_violation"
    RAMP_VIOLATION = "ramp_violation"


@dataclass(frozen=True)
class ActionValidation:
    mode: ExecutionMode
    failure_reason: FailureReason
    selected_command: np.ndarray | None
    candidate_clearance_min_slack_m_s: float
    candidate_velocity_min_slack_rad_s: float
    selected_clearance_min_slack_m_s: float
    selected_velocity_min_slack_rad_s: float
    ramp_clearance_min_slack_m_s: float
    ramp_velocity_min_slack_rad_s: float
    state_age_s: float
    target_age_s: float
    expires_at_s: float
    candidate_valid: bool
    ramp_valid: bool
    scope: str = "current_linearized_qp_rows_and_commanded_velocity_ramp_only"


def _slacks(
    velocity: np.ndarray,
    clearance_matrix: np.ndarray,
    clearance_lower: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
) -> tuple[float, float]:
    if velocity.shape != (17,) or not np.all(np.isfinite(velocity)):
        return float("-inf"), float("-inf")
    clearance = (
        float(np.min(clearance_matrix @ velocity - clearance_lower))
        if clearance_lower.size else float("inf")
    )
    bounds = float(min(np.min(velocity - velocity_lower), np.min(velocity_upper - velocity)))
    return clearance, bounds


def validate_action(
    *,
    candidate: np.ndarray,
    solver_feasible: bool,
    solver_status: str,
    start_velocity: np.ndarray,
    clearance_matrix: np.ndarray,
    clearance_lower: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    now_s: float,
    state_timestamp_s: float,
    target_timestamp_s: float,
    max_input_age_s: float,
    command_period_s: float,
    clearance_rate_tolerance_m_s: float,
    velocity_tolerance_rad_s: float,
    caution_clearance_slack_m_s: float = 1e-3,
) -> ActionValidation:
    """Check a candidate and both ends of its affine 20 ms velocity ramp.

    Each frozen QP row is affine in velocity, hence its minimum on this ramp
    occurs at an endpoint. This says nothing about changes in the row as the
    physical state and target evolve; those require later execution bounds.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    start_velocity = np.asarray(start_velocity, dtype=np.float64)
    candidate_clearance, candidate_bounds = _slacks(
        candidate, clearance_matrix, clearance_lower, velocity_lower, velocity_upper
    )
    start_clearance, start_bounds = _slacks(
        start_velocity, clearance_matrix, clearance_lower, velocity_lower, velocity_upper
    )
    ramp_clearance = min(candidate_clearance, start_clearance)
    ramp_bounds = min(candidate_bounds, start_bounds)
    candidate_valid = (candidate_clearance >= -clearance_rate_tolerance_m_s
                       and candidate_bounds >= -velocity_tolerance_rad_s)
    ramp_valid = (ramp_clearance >= -clearance_rate_tolerance_m_s
                  and ramp_bounds >= -velocity_tolerance_rad_s)
    state_age = now_s - state_timestamp_s
    target_age = now_s - target_timestamp_s
    expires = now_s + command_period_s
    if not np.isfinite(state_age) or state_age < -1e-9 or state_age > max_input_age_s:
        reason = FailureReason.STALE_STATE
    elif not np.isfinite(target_age) or target_age < -1e-9 or target_age > max_input_age_s:
        reason = FailureReason.STALE_TARGET
    elif not np.all(np.isfinite(candidate)) or candidate.shape != (17,):
        reason = FailureReason.NONFINITE_CANDIDATE
    elif solver_status == "maximum_iterations":
        reason = FailureReason.ITERATION_LIMIT
    elif not solver_feasible:
        reason = FailureReason.QP_INFEASIBLE
    elif solver_status != "solved":
        reason = FailureReason.SOLVER_FAILURE
    elif not candidate_valid:
        reason = FailureReason.CANDIDATE_VIOLATION
    elif not ramp_valid:
        reason = FailureReason.RAMP_VIOLATION
    else:
        reason = FailureReason.NONE
    selected = candidate.copy() if reason is FailureReason.NONE else None
    mode = (
        ExecutionMode.UNCERTIFIED if selected is None else
        ExecutionMode.CAUTION if ramp_clearance <= caution_clearance_slack_m_s else
        ExecutionMode.TRACK
    )
    return ActionValidation(
        mode=mode, failure_reason=reason, selected_command=selected,
        candidate_clearance_min_slack_m_s=candidate_clearance,
        candidate_velocity_min_slack_rad_s=candidate_bounds,
        selected_clearance_min_slack_m_s=candidate_clearance if selected is not None else float("nan"),
        selected_velocity_min_slack_rad_s=candidate_bounds if selected is not None else float("nan"),
        ramp_clearance_min_slack_m_s=ramp_clearance,
        ramp_velocity_min_slack_rad_s=ramp_bounds,
        state_age_s=state_age, target_age_s=target_age, expires_at_s=expires,
        candidate_valid=candidate_valid, ramp_valid=ramp_valid,
    )


def command_is_current(validation: ActionValidation, execution_time_s: float) -> bool:
    return (
        validation.selected_command is not None
        and np.isfinite(execution_time_s)
        and execution_time_s < validation.expires_at_s + 1e-9
    )
