"""Opt-in bounded interval PCC rows and per-tick admission for V6-lite.

This module changes only the PCC backend. The original MuJoCo and actual-chain
capsule rows, 17-dimensional weighted QP, and A.1 action validator remain in
the same solve. A partition is selected at a task boundary and frozen through
the ensuing ten torque steps.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import ramp_mean_weights
from v6_lite.hierarchical_qp import HierarchicalVelocityQP
from v6_lite.pcc_interval_cbf import (
    FixedIntervalCBFEvaluator, IntervalPartition,
    SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD,
)
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


POINT_BUDGET = 255


def ramp_endpoint_velocity_box(shape_q, start_velocity, work_lower,
                               work_upper, task_period_s):
    old_weight, new_weight = ramp_mean_weights()
    offset = shape_q + task_period_s * old_weight * start_velocity
    denominator = task_period_s * new_weight
    return ((work_lower - offset) / denominator,
            (work_upper - offset) / denominator)


class IntervalPreflightFailure(RuntimeError):
    def __init__(self, reason: str, diagnostic: dict):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic


class BoundedIntervalVelocityQP(HierarchicalVelocityQP):
    """Append selected interval rows to the existing clearance row block."""

    def __init__(self, *args, evaluator: FixedIntervalCBFEvaluator, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval_evaluator = evaluator
        self.partition: IntervalPartition | None = None
        self.selected_ids: set[str] = set()
        self.last_batch = None

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
        return super().solve(data, *args, **kwargs)

    def _solve_qp_admm(self, hessian, linear, matrix, lower, upper,
                       initial, initial_dual=None):
        # The 0.45/0.55 ramp endpoint box is solved with the original 17D QP.
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

    def _build_all_clearance_constraints(self, data, generalized_map):
        original = super()._build_all_clearance_constraints(data, generalized_map)
        if self.partition is None:
            raise RuntimeError("interval partition was not selected at task boundary")
        batch = self.interval_evaluator.evaluate_state(
            data, self.partition, generalized_map=generalized_map,
            derivative_interval_ids=self.selected_ids,
        )
        self.last_batch = batch
        if (not batch.interval_well_formed or not batch.coverage_complete
                or not batch.analytic_bound_assumptions_satisfied
                or batch.geometry_domain_status
                != "INSIDE_DECLARED_WORK_DOMAIN;ON_DECLARED_SHAPE_SUBSPACE"):
            raise RuntimeError("interval cover or declared geometry domain unsupported")
        rows = [row for row in batch.rows if row.interval_id in self.selected_ids]
        if (len(rows) != len(self.selected_ids) or any(
                row.derivative_status != "SUPPORTED" for row in rows)):
            raise RuntimeError("required interval derivative unsupported")
        gain = self.config.pcc_clearance_barrier_gain
        return replace(
            original,
            matrix=np.vstack((original.matrix,
                              *(row.generalized_gradient.reshape(1, 17) for row in rows))),
            lower=np.concatenate((original.lower, np.asarray(
                [-gain * row.h_m - row.target_drift_m_s for row in rows]))),
            sources=(*original.sources, *(row.interval_id for row in rows)),
            distances_m=np.concatenate((original.distances_m, np.asarray(
                [row.distance_lower_bound_m for row in rows]))),
            target_drifts_m_s=np.concatenate((original.target_drifts_m_s, np.asarray(
                [row.target_drift_m_s for row in rows]))),
            barrier_gains_s_inv=np.concatenate((original.barrier_gains_s_inv,
                                                 np.full(len(rows), gain))),
            minimum_clearance_m=(
                min(original.minimum_clearance_m,
                    *(row.distance_lower_bound_m for row in rows))
                if rows else original.minimum_clearance_m
            ),
        )


class BoundedIntervalAdmission:
    """Select a full frozen cover and reject unsupported current states."""

    def __init__(self, spec, model, qp: BoundedIntervalVelocityQP):
        self.spec = spec
        self.model = model
        self.qp = qp
        self.evaluator = qp.interval_evaluator
        self.envelope = StateLocalPCCEnvelopeAudit(
            model, self.evaluator.shape_spec)
        self.query = PersistentIntervalDecisionQuery(self.evaluator.shape_model)

    def prepare(self, data: mujoco.MjData, previous: np.ndarray) -> dict:
        evaluator = self.evaluator
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        shape_q = projection.planner_configuration
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        coverage = self.envelope.evaluate(data, shape_q, base)
        box = target_box_from_mujoco(self.model, data, evaluator.target_geom_id)
        decision = self.query.evaluate(
            shape_q, base, box, IntervalPartition.uniform(),
            max_point_evaluations=POINT_BUDGET,
        )
        diagnostic = {
            "time_s": float(data.time),
            "point_evaluations": decision.point_evaluation_count,
            "query_budget_exhausted": decision.budget_exhausted,
            "proxy_status": decision.proxy_clearance_status,
            "proxy_lower_m": decision.distance_lower_bound_m,
            "proxy_upper_m": decision.distance_upper_bound_m,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
            "current_envelope_status": coverage.status,
            "current_envelope_margin_m": coverage.min_margin_m,
        }
        def fail(reason: str):
            raise IntervalPreflightFailure(reason, diagnostic)
        if (not decision.bounds_valid or decision.proxy_clearance_status
                != "PROXY_CLEARANCE_AT_LEAST_GATE"):
            fail("PROXY_QUERY_UNSUPPORTED_OR_NOT_SAFE")
        if coverage.status != "COVERED_AT_THIS_STATE":
            fail("ACTUAL_CHAIN_NOT_ENVELOPED_AT_CURRENT_STATE")
        if projection.residual_linf_rad > SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD:
            fail("OUTSIDE_DECLARED_SHAPE_SUBSPACE")
        planner_q = self.spec.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
        _, _, speed, box_valid = ramp_velocity_abs_bound(
            self.spec, self.qp.config, planner_q, previous)
        diagnostic["velocity_box_valid"] = box_valid
        if not box_valid:
            fail("RAMP_VELOCITY_BOX_INVALID")
        generalized_map, _ = self.qp.reaction_velocity_map(data)
        ids, excluded = select_intervals_by_frozen_reach(
            decision.lower_by_interval_id, decision.partition, evaluator,
            data, self.qp.config, generalized_map, velocity_abs_bound=speed,
        )
        self.qp.partition = decision.partition
        self.qp.selected_ids = ids
        diagnostic.update({"selected_interval_count": len(ids),
                           "excluded_interval_count": len(excluded),
                           "interval_count": decision.interval_count})
        return diagnostic
