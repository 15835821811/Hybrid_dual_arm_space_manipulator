"""Read-only feasibility of B.2 interval rows on a frozen A.1 replay state.

This is a diagnostic linear program, never an online planner or a replacement
for the single weighted 17-D control QP. It asks whether any endpoint can pass
the *declared* instantaneous, ten-step ramp endpoint, and frozen next-start
linear rows. It does not prove nonlinear or continuous-time safety.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from v6_lite.execution_ramp import ramp_mean_weights
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import IntervalSafetyBatch
from v6_lite.recompute_execution_constraints import RecomputedRows


@dataclass(frozen=True)
class FrozenFeasibilityResult:
    status: str
    lp_status: str
    candidate_feasible: bool | None
    executable_candidate_exists: bool | None
    start_clearance_min_slack_m_s: float | None
    start_velocity_min_slack_rad_s: float
    historical_endpoint_clearance_min_slack_m_s: float | None
    historical_endpoint_lookahead_min_slack_m_s: float | None
    historical_endpoint_velocity_min_slack_rad_s: float | None
    historical_endpoint_accepted_by_frozen_rows: bool | None
    feasible_endpoint_clearance_min_slack_m_s: float | None
    feasible_endpoint_lookahead_min_slack_m_s: float | None
    feasible_endpoint_velocity_min_slack_rad_s: float | None
    worst_start_source: str | None
    row_count: int
    interval_row_count: int
    lp_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def velocity_box(spec: RobotModelSpecV5, config: HierarchicalQPConfig,
                 planner_q: np.ndarray, previous_velocity: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the existing planner box, including acceleration limits."""
    q = np.asarray(planner_q, dtype=np.float64)
    previous = np.asarray(previous_velocity, dtype=np.float64)
    if q.shape != (17,) or previous.shape != (17,):
        raise ValueError("planner configuration and prior velocity must be 17-D")
    speed = config.velocity_limit_scale * spec.planner_velocity_limits
    dt = config.task_period_s
    lower = np.maximum(-speed, previous - spec.planner_acceleration_limits * dt)
    upper = np.minimum(speed, previous + spec.planner_acceleration_limits * dt)
    lower = np.maximum(lower, -config.joint_barrier_gain * (
        q - (spec.planner_lower + config.joint_position_margin_rad)
    ))
    upper = np.minimum(upper, config.joint_barrier_gain * (
        (spec.planner_upper - config.joint_position_margin_rad) - q
    ))
    return lower, upper


def ramp_velocity_abs_bound(spec: RobotModelSpecV5,
                            config: HierarchicalQPConfig,
                            planner_q: np.ndarray,
                            previous_velocity: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Bound every rate on the declared affine ramp to a planner-box endpoint.

    An empty endpoint box cannot admit an action. In that case the shadow
    screen uses the global speed limit and the recorded ramp start rather
    than pretending that the empty box reduces the possible approach.
    """
    lower, upper = velocity_box(spec, config, planner_q, previous_velocity)
    start = np.asarray(previous_velocity, dtype=np.float64)
    global_speed = config.velocity_limit_scale * spec.planner_velocity_limits
    if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or not np.all(np.isfinite(start))):
        raise ValueError("non-finite planner velocity box or ramp start")
    box_valid = bool(np.all(lower <= upper))
    if box_valid:
        bound = np.maximum.reduce((np.abs(lower), np.abs(upper), np.abs(start)))
    else:
        bound = np.maximum(global_speed, np.abs(start))
    return lower, upper, bound, box_valid


def combined_rows(original: RecomputedRows, batch: IntervalSafetyBatch,
                  selected_ids: set[str], config: HierarchicalQPConfig,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             tuple[str, ...]]:
    if not (batch.interval_well_formed and batch.coverage_complete
            and batch.analytic_bound_assumptions_satisfied):
        raise ValueError("interval partition is not a supported complete cover")
    found = {row.interval_id for row in batch.rows}
    if not selected_ids <= found:
        raise ValueError("required interval is missing from the frozen partition")
    selected = [row for row in batch.rows if row.interval_id in selected_ids]
    if any(row.derivative_status != "SUPPORTED" for row in selected):
        raise ValueError("required interval derivative is unsupported")
    matrix = np.vstack([
        original.matrix,
        *(row.generalized_gradient.reshape(1, 17) for row in selected),
    ])
    lower = np.concatenate([
        original.lower,
        np.asarray([-config.pcc_clearance_barrier_gain * row.h_m
                    - row.target_drift_m_s for row in selected]),
    ])
    drifts = np.concatenate([
        original.target_drifts_m_s,
        np.asarray([row.target_drift_m_s for row in selected]),
    ])
    gains = np.concatenate([
        original.barrier_gains_s_inv,
        np.full(len(selected), config.pcc_clearance_barrier_gain),
    ])
    sources = (*original.sources, *(row.interval_id for row in selected))
    return matrix, lower, drifts, gains, sources


def solve_frozen_linear_feasibility(
    matrix: np.ndarray, lower: np.ndarray, drifts: np.ndarray,
    gains: np.ndarray, sources: tuple[str, ...],
    velocity_lower: np.ndarray, velocity_upper: np.ndarray,
    start_velocity: np.ndarray, config: HierarchicalQPConfig,
    *, interval_row_count: int,
    historical_endpoint: np.ndarray | None = None,
) -> FrozenFeasibilityResult:
    """Test existence using the *existing* execution tolerances, no relaxation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    drifts = np.asarray(drifts, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    v_lower = np.asarray(velocity_lower, dtype=np.float64)
    v_upper = np.asarray(velocity_upper, dtype=np.float64)
    start = np.asarray(start_velocity, dtype=np.float64)
    row_count = len(lower)
    if (matrix.shape != (row_count, 17) or drifts.shape != (row_count,)
            or gains.shape != (row_count,) or len(sources) != row_count
            or any(x.shape != (17,) for x in (v_lower, v_upper, start))
            or not 0 <= interval_row_count <= row_count):
        raise ValueError("inconsistent frozen feasibility dimensions")
    if (any(np.any(~np.isfinite(x)) for x in (
            matrix, lower, drifts, gains, v_lower, v_upper, start))
            or np.any(gains <= 0)):
        raise ValueError("nonfinite or invalid frozen feasibility rows")
    old_weight, new_weight = ramp_mean_weights()
    gain_dt = gains * config.task_period_s
    lookahead_matrix = matrix * (1.0 + gain_dt * new_weight)[:, None]
    lookahead_lower = lower - gain_dt * (
        old_weight * (matrix @ start) + drifts
    ) + config.lookahead_model_margin_m_s
    start_slacks = matrix @ start - lower
    start_clearance = float(np.min(start_slacks)) if row_count else None
    start_box = float(min(np.min(start - v_lower), np.min(v_upper - start)))
    worst_start = sources[int(np.argmin(start_slacks))] if row_count else None
    clearance_tol = config.clearance_rate_tolerance_m_s
    velocity_tol = config.velocity_tolerance_rad_s
    start_ok = ((start_clearance is None or start_clearance >= -clearance_tol)
                and start_box >= -velocity_tol)

    history_clearance = history_lookahead = history_box = None
    history_accepted = None
    if historical_endpoint is not None:
        history = np.asarray(historical_endpoint, dtype=np.float64)
        if history.shape != (17,) or np.any(~np.isfinite(history)):
            raise ValueError("historical endpoint must be finite 17-D")
        history_clearance = (float(np.min(matrix @ history - lower))
                             if row_count else None)
        history_lookahead = (float(np.min(lookahead_matrix @ history - lookahead_lower))
                             if row_count else None)
        history_box = float(min(np.min(history - v_lower),
                                np.min(v_upper - history)))
        history_accepted = bool(
            start_ok and (history_clearance is None or history_clearance >= -clearance_tol)
            and (history_lookahead is None or history_lookahead >= -clearance_tol)
            and history_box >= -velocity_tol
        )

    started = time.perf_counter()
    stacked = np.vstack((matrix, lookahead_matrix))
    thresholds = np.concatenate((lower, lookahead_lower))
    solved = linprog(
        np.zeros(17),
        A_ub=-stacked if row_count else None,
        b_ub=-thresholds + clearance_tol if row_count else None,
        bounds=list(zip(v_lower - velocity_tol, v_upper + velocity_tol)),
        method="highs",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidate_feasible = True if solved.status == 0 else (
        False if solved.status == 2 else None
    )
    if candidate_feasible:
        endpoint = np.asarray(solved.x, dtype=np.float64)
        endpoint_clearance = (float(np.min(matrix @ endpoint - lower))
                              if row_count else None)
        endpoint_lookahead = (float(np.min(lookahead_matrix @ endpoint - lookahead_lower))
                              if row_count else None)
        endpoint_box = float(min(np.min(endpoint - v_lower),
                                 np.min(v_upper - endpoint)))
        if (not np.all(np.isfinite(endpoint))
                or (endpoint_clearance is not None
                    and endpoint_clearance < -clearance_tol - 1e-7)
                or (endpoint_lookahead is not None
                    and endpoint_lookahead < -clearance_tol - 1e-7)
                or endpoint_box < -velocity_tol - 1e-7):
            candidate_feasible = None
            endpoint_clearance = endpoint_lookahead = endpoint_box = None
    else:
        endpoint_clearance = endpoint_lookahead = endpoint_box = None
    executable = (start_ok and candidate_feasible) if candidate_feasible is not None else None
    if start_clearance is not None and start_clearance < -clearance_tol:
        status = "START_CLEARANCE_VIOLATION"
    elif start_box < -velocity_tol:
        status = "START_VELOCITY_VIOLATION"
    elif candidate_feasible is False:
        status = "NO_FEASIBLE_ENDPOINT"
    elif candidate_feasible is None:
        status = "LP_UNKNOWN"
    else:
        status = "FROZEN_ROWS_FEASIBLE"
    return FrozenFeasibilityResult(
        status, str(solved.message), candidate_feasible, executable,
        start_clearance, start_box,
        history_clearance, history_lookahead, history_box, history_accepted,
        endpoint_clearance, endpoint_lookahead, endpoint_box,
        worst_start, row_count, interval_row_count, elapsed_ms,
    )
