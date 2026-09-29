"""Equivalent 17D B.2 QP implementation with conservative pair screening.

The solver reuses matrix products within the original ADMM iteration. Pair
screening skips an exact MuJoCo query only if a bounding-sphere lower bound is
strictly above the unchanged original query range. A full exact fallback
preserves diagnostic minima when no retained pair is within that range.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from v6_lite.b2_batched_capsule_bounds import BatchedCapsuleMidpointBounds
from v6_lite.b2_interval_online import (
    BoundedIntervalAdmission, BoundedIntervalVelocityQP,
)
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.hierarchical_qp import _ShapeClearanceKinematics
from v6_lite.pcc_fixed_arc_state_envelope import FixedArcStateLocalPCCEnvelopeAudit
from v6_lite.pcc_vectorized_fixed_arc_positions import VectorizedFixedArcPCCPositions
from v6_lite.shape_clearance import target_box_from_mujoco


SPHERE_SCREEN_PAD_M = 1e-6


class OptimizedBoundedIntervalVelocityQP(BoundedIntervalVelocityQP):
    """Keep all safety rows and solve the same weighted 17D problem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pair_a = np.asarray(
            [pair.geom_a for pair in self.collision_pairs], dtype=np.int32)
        self._pair_b = np.asarray(
            [pair.geom_b for pair in self.collision_pairs], dtype=np.int32)
        self.last_exact_pair_call_count = 0
        self.last_sphere_fallback = False
        self._batched_capsules = (
            BatchedCapsuleMidpointBounds(self._capsule_envelopes)
            if self._capsule_envelopes is not None else None)

    def _unconstrained_solve(self, hessian, linear):
        try:
            factor = cho_factor(hessian, lower=True, check_finite=False)
            return cho_solve(factor, linear, check_finite=False)
        except np.linalg.LinAlgError:
            return np.linalg.solve(hessian, linear)

    def _capsule_clearance_kinematics(self, data, generalized_map):
        if self._batched_capsules is None or self._shape_target_geom_id < 0:
            raise RuntimeError("capsule CBF was requested without geometry")
        box = target_box_from_mujoco(
            self.model, data, self._shape_target_geom_id)
        result = self._batched_capsules.minimum(self.model, data, box)
        arm_body_id = int(self.model.geom_bodyid[result.source_index])
        arm_jacobian = self.point_jacobian(
            data, arm_body_id, result.point_on_arm)
        controlled_gradient = (
            result.normal_box_to_arm @ arm_jacobian @ generalized_map)
        target_jacobian = self._target_point_velocity_jacobian(
            data, result.point_on_box)
        target_rate = float(
            -result.normal_box_to_arm @ target_jacobian
            @ self.target_exogenous_qvel(data))
        return _ShapeClearanceKinematics(
            distance_m=float(result.signed_distance_m),
            gradient=np.asarray(controlled_gradient, dtype=np.float64),
            target_distance_rate_m_s=target_rate,
            source_name=f"capsule:{result.source_name}",
        )

    def _mujoco_clearance_constraint_set(self, data, generalized_map):
        model = self.model
        a, b = self._pair_a, self._pair_b
        radius_a = np.asarray(model.geom_rbound[a], dtype=np.float64)
        radius_b = np.asarray(model.geom_rbound[b], dtype=np.float64)
        center_a = np.asarray(data.geom_xpos[a], dtype=np.float64)
        center_b = np.asarray(data.geom_xpos[b], dtype=np.float64)
        finite = (np.isfinite(radius_a) & np.isfinite(radius_b)
                  & (radius_a > 0) & (radius_b > 0)
                  & np.all(np.isfinite(center_a), axis=1)
                  & np.all(np.isfinite(center_b), axis=1))
        lower = np.linalg.norm(center_a - center_b, axis=1) - radius_a - radius_b
        far = finite & (lower > self.config.clearance_query_max_m
                        + SPHERE_SCREEN_PAD_M)
        original_pairs = self.collision_pairs
        retained = tuple(pair for index, pair in enumerate(original_pairs)
                         if not far[index])
        self.collision_pairs = retained
        try:
            block = super()._mujoco_clearance_constraint_set(
                data, generalized_map)
        finally:
            self.collision_pairs = original_pairs
        fallback = (not retained or block.minimum_clearance_m
                    >= self.config.clearance_query_max_m)
        self.last_exact_pair_call_count = len(retained)
        self.last_sphere_fallback = bool(fallback)
        if fallback:
            block = super()._mujoco_clearance_constraint_set(
                data, generalized_map)
            self.last_exact_pair_call_count += len(original_pairs)
        elif not any(pair.pair_class == "continuum_target" for pair in retained):
            block = replace(
                block,
                mujoco_continuum_target_distance_m=self.config.clearance_query_max_m,
                mujoco_continuum_target_gradient=np.zeros(17),
                mujoco_continuum_target_gradient_valid=False,
            )
        return block

    def _solve_qp_admm(self, hessian, linear, matrix, lower, upper,
                       initial, initial_dual=None):
        endpoint_rows = np.eye(17, dtype=np.float64)[:10]
        matrix = np.vstack((matrix, endpoint_rows))
        lower = np.concatenate((lower, self._domain_rate_lower))
        upper = np.concatenate((upper, self._domain_rate_upper))
        original_row_count = len(lower) - 10
        initial_dual = (None if initial_dual is None else np.concatenate((
            initial_dual, np.zeros(10, dtype=np.float64))))
        cfg = self.config
        rho = cfg.admm_rho
        sigma = cfg.admm_sigma
        gram = matrix.T @ matrix
        identity = np.eye(17)
        system = hessian + sigma * identity + rho * gram
        factor = cho_factor(system, lower=True, check_finite=False)
        value = np.asarray(initial, dtype=np.float64).copy()
        product = matrix @ value
        dual = (np.zeros(matrix.shape[0], dtype=np.float64)
                if initial_dual is None else
                np.asarray(initial_dual, dtype=np.float64).copy())
        if dual.shape != (matrix.shape[0],) or np.any(~np.isfinite(dual)):
            raise ValueError("ADMM dual warm start has the wrong shape or is non-finite")
        auxiliary = np.minimum(np.maximum(product + dual / rho, lower), upper)
        linear_scale = float(np.max(np.abs(linear)))
        status = "maximum_iterations"
        for iteration in range(1, cfg.qp_max_iterations + 1):
            rhs = sigma * value - linear + matrix.T @ (rho * auxiliary - dual)
            value = cho_solve(factor, rhs, check_finite=False)
            product = matrix @ value
            previous_auxiliary = auxiliary
            relaxed = (cfg.admm_relaxation * product
                       + (1.0 - cfg.admm_relaxation) * previous_auxiliary)
            auxiliary = np.minimum(
                np.maximum(relaxed + dual / rho, lower), upper)
            dual += rho * (relaxed - auxiliary)
            primal_residual = float(np.max(np.abs(product - auxiliary)))
            hessian_product = hessian @ value
            transpose_dual = matrix.T @ dual
            dual_residual = float(np.max(np.abs(
                hessian_product + linear + transpose_dual)))
            primal_scale = max(
                1.0, float(np.max(np.abs(product))),
                float(np.max(np.abs(auxiliary))))
            dual_scale = max(
                1.0, float(np.max(np.abs(hessian_product))),
                float(np.max(np.abs(transpose_dual))), linear_scale)
            if (primal_residual <= cfg.qp_ftol * primal_scale
                    and dual_residual <= 5.0 * cfg.qp_ftol * dual_scale):
                status = "solved"
                break
            if iteration % 100 == 0 and iteration < cfg.qp_max_iterations:
                primal_ratio = primal_residual / (cfg.qp_ftol * primal_scale)
                dual_ratio = dual_residual / (5.0 * cfg.qp_ftol * dual_scale)
                next_rho = rho
                if dual_ratio > 3.0 * primal_ratio:
                    next_rho = max(rho / 5.0, 0.1)
                elif primal_ratio > 3.0 * dual_ratio:
                    next_rho = min(rho * 5.0, 500.0)
                if next_rho != rho:
                    rho = next_rho
                    system = hessian + sigma * identity + rho * gram
                    factor = cho_factor(system, lower=True, check_finite=False)
        feasibility = np.minimum(product - lower, upper - product)
        feasible = bool(float(np.min(feasibility)) >= -cfg.feasibility_tolerance)
        return value, feasible, status, iteration, dual[:original_row_count]


class OptimizedBoundedIntervalAdmission(BoundedIntervalAdmission):
    """Reuse same-state PCC prefixes without changing query decisions."""

    def __init__(self, spec, model, qp):
        super().__init__(spec, model, qp)
        self.envelope = FixedArcStateLocalPCCEnvelopeAudit(
            model, self.evaluator.shape_spec)
        self.envelope._positions = VectorizedFixedArcPCCPositions(
            self.evaluator.shape_spec, self.envelope.arclengths)
        self.query = BatchedDistanceDecisionQuery(self.evaluator.shape_model)
