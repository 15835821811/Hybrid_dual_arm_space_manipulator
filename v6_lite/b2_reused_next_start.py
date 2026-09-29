"""Same next-start reconstruction with model-constant row geometry reused.

The original private audit rebuilt 61 continuum capsule envelopes every
cycle. This version constructs the independent row builder once per model;
all state-dependent distances, Jacobians, reaction maps and rows are still
recomputed at every next state.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


POINT_BUDGET = 255


class ReusedNextStart:
    def __init__(self) -> None:
        self._model = None
        self._builder = None
        self.builder_constructions = 0

    def __call__(self, model, robot, verifier, cfg, evaluator, data,
                 endpoint_command: np.ndarray) -> dict:
        if self._model is not model:
            no_legacy = replace(cfg, enable_pcc_cbf=False,
                                enable_capsule_cbf=True)
            self._builder = ReplayConstraintBuilder(
                robot, model, verifier.pairs, no_legacy)
            self._model = model
            self.builder_constructions += 1
        independent = self._builder
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
        query = PersistentIntervalDecisionQuery(evaluator.shape_model)
        decision = query.evaluate(
            projection.planner_configuration, base, box,
            IntervalPartition.uniform(), max_point_evaluations=POINT_BUDGET)
        result = {
            "proxy_status": decision.proxy_clearance_status,
            "proxy_lower_m": decision.distance_lower_bound_m,
            "proxy_upper_m": decision.distance_upper_bound_m,
            "point_evaluations": decision.point_evaluation_count,
            "budget_exhausted": decision.budget_exhausted,
            "query_failure_reason": decision.failure_reason,
            "interval_count": decision.interval_count,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
            "frozen_rows_status": "NOT_EVALUATED",
        }
        if not decision.bounds_valid:
            return result
        generalized_map, reaction_residual = independent._reaction_map(data)
        planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
        _lower, _upper, speed, _box_valid = ramp_velocity_abs_bound(
            robot, cfg, planner_q, endpoint_command)
        ids, _excluded = select_intervals_by_frozen_reach(
            decision.lower_by_interval_id, decision.partition, evaluator,
            data, cfg, generalized_map, velocity_abs_bound=speed)
        batch = evaluator.evaluate_state(
            data, decision.partition, generalized_map=generalized_map,
            derivative_interval_ids=ids)
        result["selected_interval_count"] = len(ids)
        result["reaction_map_residual"] = reaction_residual
        result["interval_coverage_complete"] = batch.coverage_complete
        result["interval_well_formed"] = batch.interval_well_formed
        result["analytic_bound_assumptions_satisfied"] = (
            batch.analytic_bound_assumptions_satisfied)
        if not (batch.interval_well_formed and batch.coverage_complete
                and batch.analytic_bound_assumptions_satisfied
                and all(row.derivative_status == "SUPPORTED"
                        for row in batch.rows if row.interval_id in ids)):
            result["frozen_rows_status"] = "UNSUPPORTED"
            return result
        original = independent.build(data)
        matrix, lower, _drifts, _gains, sources = combined_rows(
            original, batch, ids, cfg)
        slacks = matrix @ endpoint_command - lower
        index = int(np.argmin(slacks)) if len(slacks) else None
        minimum = float(slacks[index]) if index is not None else None
        result.update({
            "frozen_rows_status": (
                "START_ROWS_SATISFIED" if minimum is None
                or minimum >= -cfg.clearance_rate_tolerance_m_s
                else "START_CLEARANCE_VIOLATION"),
            "row_count": len(slacks),
            "worst_source": sources[index] if index is not None else None,
            "minimum_start_slack_m_s": minimum,
        })
        return result
