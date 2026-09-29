"""Read-only interval-CBF shadow audit over native A.1 torque replay states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_shadow_feasibility import (
    combined_rows, ramp_velocity_abs_bound, solve_frozen_linear_feasibility,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import ramp_mean_weights
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_bounded_clearance import (
    BoundedClearanceResult, PCCBoundedClearanceEvaluator,
)
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_interval_cbf import (
    FixedIntervalCBFEvaluator, IntervalPartition, MaterialInterval,
)
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    ShapeClearanceShadow, minimum_mujoco_geom_clearance,
    target_box_from_mujoco,
)


DEFAULT_A1_ROOT = Path("v6_lite/output/v6_2_a1")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(arr.min()),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(arr.max())}


def partition_from_bounded_result(
    result: BoundedClearanceResult, lengths_m: np.ndarray,
) -> IntervalPartition:
    """Recover the complete dyadic leaf topology from B.1's candidate set."""
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if lengths.shape != (5,) or not result.bounds_valid:
        raise ValueError("invalid B.1 result or segment lengths")
    leaves = []
    for candidate in result.candidate_intervals:
        length = float(lengths[candidate.segment_id])
        width = candidate.local_end_m - candidate.local_start_m
        if not (0.0 < width <= length):
            raise ValueError("invalid B.1 candidate width")
        depth = int(round(math.log2(length / width)))
        if depth < 0 or depth > 20:
            raise ValueError("B.1 partition depth exceeds supported range")
        ordinal = int(round(candidate.local_start_m / length * (1 << depth)))
        if ordinal < 0 or ordinal >= (1 << depth):
            raise ValueError("B.1 candidate ordinal is outside its section")
        leaf = MaterialInterval(candidate.segment_id,
                                format(ordinal, f"0{depth}b") if depth else "")
        a, b = leaf.local_bounds(length)
        if (abs(a - candidate.local_start_m) > 1e-10
                or abs(b - candidate.local_end_m) > 1e-10):
            raise ValueError("B.1 candidate is not a dyadic material interval")
        leaves.append(leaf)
    partition = IntervalPartition(tuple(sorted(leaves)))
    if not partition.coverage(lengths).coverage_complete:
        raise ValueError("B.1 candidate intervals do not cover all five sections")
    return partition


def select_intervals_by_frozen_reach(
    result: BoundedClearanceResult | dict[str, float],
    partition: IntervalPartition,
    evaluator: FixedIntervalCBFEvaluator,
    data: mujoco.MjData,
    cfg: HierarchicalQPConfig,
    generalized_map: np.ndarray,
    *, velocity_abs_bound: np.ndarray | None = None,
) -> tuple[set[str], dict[str, float]]:
    """Conservative frozen-model screen, never a fixed top-k truncation.

    The shape term bounds the angular perturbation's integrated effect on a
    downstream material point. The base term bounds G's free-joint twist for
    every planner rate inside its supplied absolute bound (or the global
    speed box by default). Target twist is held at the measured value.
    This is a *one-cycle frozen-model screen*, not a
    continuous-time or uncertain-dynamics certificate.
    """
    spec = evaluator.shape_spec
    lengths = spec.segment_lengths_m
    speeds = (cfg.velocity_limit_scale * evaluator.spec.planner_velocity_limits
              if velocity_abs_bound is None
              else np.asarray(velocity_abs_bound, dtype=np.float64))
    if (speeds.shape != (17,) or not np.all(np.isfinite(speeds))
            or np.any(speeds < 0.0)):
        raise ValueError("frozen reach needs finite nonnegative 17-D rate bounds")
    base_map = generalized_map[evaluator.base_dof_slice, :]
    radius_from_base = (float(np.linalg.norm(spec.base_to_shape_start[:3, 3]))
                        + spec.total_length_m)
    base_rate = sum(
        (float(np.linalg.norm(base_map[:3, j]))
         + radius_from_base * float(np.linalg.norm(base_map[3:, j]))) * speeds[j]
        for j in range(17)
    )
    twist = np.asarray(data.qvel[evaluator.target_dof_slice], dtype=np.float64)
    target_half = np.asarray(evaluator.model.geom_size[evaluator.target_geom_id])
    target_rate = (float(np.linalg.norm(twist[:3]))
                   + float(np.linalg.norm(twist[3:])) * float(np.linalg.norm(target_half)))
    bend_norms = np.linalg.norm(spec.pcc_bending_map, axis=0)
    selected = set()
    reach_by_id = {}
    candidate_by_bounds = (
        {(item.segment_id, round(item.local_start_m, 12),
          round(item.local_end_m, 12)): item.lower_bound_m
         for item in result.candidate_intervals}
        if isinstance(result, BoundedClearanceResult) else None
    )
    for leaf in partition.leaves:
        length = float(lengths[leaf.segment_id])
        a, b = leaf.local_bounds(length)
        candidate_lower = (
            candidate_by_bounds[(leaf.segment_id, round(a, 12), round(b, 12))]
            if candidate_by_bounds is not None else result[leaf.interval_id]
        )
        local = 0.5 * (a + b)
        shape_rate = 0.0
        for segment in range(leaf.segment_id + 1):
            if segment == leaf.segment_id:
                coefficient = local * local / (2.0 * float(lengths[segment]))
            else:
                after = local + float(np.sum(lengths[segment + 1:leaf.segment_id]))
                coefficient = float(lengths[segment]) / 2.0 + after
            for axis in range(2):
                shape_rate += coefficient * bend_norms[axis] * speeds[2 * segment + axis]
        approach = (shape_rate + base_rate + target_rate
                    + cfg.lookahead_model_margin_m_s) * cfg.task_period_s
        reach_by_id[leaf.interval_id] = approach
        if candidate_lower <= cfg.pcc_clearance_activation_m + approach:
            selected.add(leaf.interval_id)
    return selected, reach_by_id


@dataclass(frozen=True)
class ShadowBudget:
    branch_point_evaluations: int = 31
    total_point_evaluations: int = 64
    jacobian_evaluations: int = 96
    local_refinement_evaluations: int = 0
    total_query_time_ms: float = 20.0


def shadow_mode(mode: str, root: Path, *, sample_stride: int,
                budget: ShadowBudget,
                query_mode: str = "bounded_cold") -> dict:
    if query_mode not in ("bounded_cold", "persistent_warm"):
        raise ValueError("unsupported interval query mode")
    metrics_path = root / "v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    robot = default_v6_lite_robot_spec()
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = metrics["run_config"]
    aggregates: dict[str, list[float]] = {key: [] for key in (
        "bound_gap_m", "query_time_ms", "branch_query_time_ms",
        "interval_assembly_time_ms", "point_evaluations",
        "jacobian_evaluations", "partition_size", "subspace_residual_linf_rad",
        "selected_interval_count", "excluded_interval_count", "frozen_reach_m",
        "global_speed_selected_interval_count", "ramp_box_selection_delta",
        "historical_full_control_ms", "historical_shape_query_ms",
        "historical_qp_ms", "paired_replacement_estimate_ms",
        "next_h_prediction_abs_error_m", "next_start_residual_abs_error_m_s",
        "warm_all_task_query_ms", "warm_all_task_partition_size",
        "frozen_original_assembly_ms", "frozen_lp_ms", "frozen_row_count",
    )}
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    warm_unknown_examples: list[dict] = []
    feasibility_records: list[dict] = []
    scenario_reports = []
    for scenario_result in metrics["scenarios"]:
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(scenario_result["scenario"]),
            WholeBodyVerificationConfig(
                minimum_clearance=run_cfg["whole_body_minimum_clearance_m"],
                query_distance_max=2.5,
                adaptive_subdivisions=run_cfg["verification_subdivisions"],
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data = mujoco.MjData(model)
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        feasibility_builder = ReplayConstraintBuilder(
            robot, model, verifier.pairs,
            replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True),
        )
        bounded = PCCBoundedClearanceEvaluator(evaluator.shape_model)
        persistent = PersistentIntervalDecisionQuery(evaluator.shape_model)
        persistent_partition = IntervalPartition.uniform()
        shadow = ShapeClearanceShadow(
            model, evaluator.target_geom_id, V61A_PCC_TUBE_RADII_M,
            spec=evaluator.shape_spec,
        )
        path = Path(scenario_result["trace"]["path"])
        trace_hash = _sha(path)
        hash_matches = trace_hash == scenario_result["trace"]["sha256"]
        if not hash_matches:
            raise ValueError(f"A.1 trace hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as trace:
            initial_qpos = trace["initial_qpos"].copy()
            initial_qvel = trace["initial_qvel"].copy()
            torque = trace["torque"].copy()
            selected = trace["task_selected_command"].copy()
            task_qpos = trace["task_qpos"].copy()
            task_time = trace["task_time"].copy()
            historical_full = trace["task_full_latency"].copy()
            historical_shape = trace["task_shape_clearance_latency"].copy()
            historical_qp = trace["task_solver_latency"].copy()
        if len(torque) != len(selected) * 10:
            raise ValueError("A.1 physics/task tick count mismatch")
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        max_state_error = 0.0
        samples = 0
        pending: tuple[int, IntervalPartition, set[str], dict[str, float],
                       dict[str, float], np.ndarray] | None = None
        for step, control in enumerate(torque):
            if step % 10 == 0:
                tick = step // 10
                # mj_step leaves position-dependent spatial fields at the
                # pre-integration state. The online QP calls mj_forward at
                # each task tick; shadow geometry must use that same state.
                mujoco.mj_forward(model, data)
                max_state_error = max(
                    max_state_error,
                    float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                    abs(float(data.time - task_time[tick])),
                )
                warm_result = None
                warm_started = None
                warm_projection = None
                warm_base = None
                warm_box = None
                if query_mode == "persistent_warm":
                    warm_started = time.perf_counter()
                    warm_low_level = data.qpos[evaluator.qpos_ids[:60]]
                    warm_projection = evaluator.shape_spec.project_actual_configuration(
                        warm_low_level
                    )
                    warm_base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                    warm_box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                    warm_result = persistent.evaluate(
                        warm_projection.planner_configuration, warm_base,
                        warm_box, persistent_partition,
                        max_point_evaluations=budget.branch_point_evaluations,
                    )
                    persistent_partition = warm_result.partition
                    aggregates["warm_all_task_query_ms"].append(
                        (time.perf_counter() - warm_started) * 1000.0
                    )
                    aggregates["warm_all_task_partition_size"].append(
                        warm_result.interval_count
                    )
                    counts["warm_all_task_unknown"] += int(
                        warm_result.proxy_clearance_status == "UNKNOWN_CROSSES_GATE"
                    )
                    counts["warm_all_task_budget_exhausted"] += int(
                        warm_result.budget_exhausted
                    )
                    counts["warm_all_task_invalid"] += int(not warm_result.bounds_valid)
                    if (warm_result.proxy_clearance_status == "UNKNOWN_CROSSES_GATE"
                            and len(warm_unknown_examples) < 20):
                        warm_unknown_examples.append({
                            "scenario_id": scenario_result["scenario"]["scenario_id"],
                            "tick": tick, "time_s": float(data.time),
                            "lower_m": warm_result.distance_lower_bound_m,
                            "upper_m": warm_result.distance_upper_bound_m,
                            "interval_count": warm_result.interval_count,
                            "point_evaluations": warm_result.point_evaluation_count,
                            "failure_reason": warm_result.failure_reason,
                        })
                if pending is not None and tick == pending[0] + 1:
                    _, fixed_partition, selected_ids, h_hat, residual_hat, endpoint = pending
                    next_batch = evaluator.evaluate_state(
                        data, fixed_partition, derivative_interval_ids=selected_ids,
                    )
                    next_rows = {x.interval_id: x for x in next_batch.rows}
                    counts["excluded_reached_activation_next_tick"] += sum(
                        x.distance_lower_bound_m <= cfg.pcc_clearance_activation_m
                        for x in next_batch.rows if x.interval_id not in selected_ids
                    )
                    for interval_id, prediction in h_hat.items():
                        actual = next_rows[interval_id]
                        aggregates["next_h_prediction_abs_error_m"].append(
                            abs(actual.h_m - prediction)
                        )
                        if actual.derivative_status == "SUPPORTED":
                            actual_residual = float(
                                actual.generalized_gradient @ endpoint
                                + actual.target_drift_m_s
                                + cfg.pcc_clearance_barrier_gain * actual.h_m
                            )
                            aggregates["next_start_residual_abs_error_m_s"].append(
                                abs(actual_residual - residual_hat[interval_id])
                            )
                        else:
                            counts["next_derivative_unsupported"] += 1
                    pending = None
                if tick % sample_stride == 0:
                    samples += 1
                    pipeline_started = (warm_started if warm_started is not None
                                        else time.perf_counter())
                    if query_mode == "persistent_warm":
                        if not warm_result.bounds_valid:
                            counts["sampled_invalid_query"] += 1
                            raise RuntimeError(
                                f"warm interval query cannot cover this state: "
                                f"{warm_result.failure_reason}; tick={tick}"
                            )
                        result = warm_result
                        partition = warm_result.partition
                        lower_source = warm_result.lower_by_interval_id
                        query_point_count = warm_result.point_evaluation_count
                        local_count = 0
                    else:
                        low_level = data.qpos[evaluator.qpos_ids[:60]]
                        projection = evaluator.shape_spec.project_actual_configuration(low_level)
                        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                        result = bounded.evaluate(
                            projection.planner_configuration, base, box,
                            actual_configuration=low_level,
                            max_evaluations=budget.branch_point_evaluations,
                        )
                        partition = partition_from_bounded_result(
                            result, evaluator.shape_spec.segment_lengths_m
                        )
                        lower_source = result
                        query_point_count = result.evaluation_count
                        local_count = result.local_refinement_evaluation_count
                    previous = np.zeros(17) if tick == 0 else selected[tick - 1]
                    planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
                    velocity_lower, velocity_upper, ramp_speed, box_valid = (
                        ramp_velocity_abs_bound(robot, cfg, planner_q, previous)
                    )
                    counts["empty_candidate_velocity_box"] += int(not box_valid)
                    generalized_map = evaluator.reaction_map(data)
                    global_selected_ids, _ = select_intervals_by_frozen_reach(
                        lower_source, partition, evaluator, data, cfg, generalized_map,
                    )
                    selected_ids, reach_by_id = select_intervals_by_frozen_reach(
                        lower_source, partition, evaluator, data, cfg, generalized_map,
                        velocity_abs_bound=ramp_speed,
                    )
                    batch = evaluator.evaluate_state(
                        data, partition, generalized_map=generalized_map,
                        derivative_interval_ids=selected_ids,
                    )
                    geometry = minimum_mujoco_geom_clearance(
                        model, data, shadow.enveloped_geom_ids,
                        evaluator.target_geom_id, query_distance_max_m=0.5,
                    )
                    point_count = query_point_count + local_count + batch.point_evaluation_count
                    total_time = (time.perf_counter() - pipeline_started) * 1000.0
                    within_budget = (
                        point_count <= budget.total_point_evaluations
                        and batch.jacobian_evaluation_count <= budget.jacobian_evaluations
                        and local_count <= budget.local_refinement_evaluations
                        and total_time <= budget.total_query_time_ms
                    )
                    aggregates["bound_gap_m"].append(result.bound_gap_m)
                    aggregates["query_time_ms"].append(total_time)
                    old_full_ms = float(historical_full[tick] * 1000.0)
                    old_shape_ms = float(historical_shape[tick] * 1000.0)
                    paired_estimate = old_full_ms - old_shape_ms + total_time
                    aggregates["historical_full_control_ms"].append(old_full_ms)
                    aggregates["historical_shape_query_ms"].append(old_shape_ms)
                    aggregates["historical_qp_ms"].append(
                        float(historical_qp[tick] * 1000.0)
                    )
                    aggregates["paired_replacement_estimate_ms"].append(paired_estimate)
                    counts["paired_replacement_estimate_over_20ms"] += int(
                        paired_estimate > 20.0
                    )
                    aggregates["branch_query_time_ms"].append(result.elapsed_ms)
                    aggregates["interval_assembly_time_ms"].append(batch.total_query_time_ms)
                    aggregates["point_evaluations"].append(point_count)
                    aggregates["jacobian_evaluations"].append(batch.jacobian_evaluation_count)
                    aggregates["partition_size"].append(len(partition.leaves))
                    aggregates["selected_interval_count"].append(len(selected_ids))
                    aggregates["global_speed_selected_interval_count"].append(
                        len(global_selected_ids)
                    )
                    aggregates["ramp_box_selection_delta"].append(
                        len(selected_ids) - len(global_selected_ids)
                    )
                    counts["ramp_box_newly_selected"] += len(
                        selected_ids - global_selected_ids
                    )
                    counts["ramp_box_newly_excluded"] += len(
                        global_selected_ids - selected_ids
                    )
                    aggregates["excluded_interval_count"].append(
                        len(partition.leaves) - len(selected_ids)
                    )
                    aggregates["frozen_reach_m"].extend(reach_by_id.values())
                    aggregates["subspace_residual_linf_rad"].append(
                        batch.subspace_residual_linf_rad
                    )
                    counts[result.proxy_clearance_status] += 1
                    counts["budget_exhausted"] += int(result.budget_exhausted)
                    counts["unified_budget_exceeded"] += int(not within_budget)
                    counts["coverage_incomplete"] += int(not batch.coverage_complete)
                    counts["derivative_unsupported"] += sum(
                        row.derivative_status != "SUPPORTED" for row in batch.rows
                        if row.interval_id in selected_ids
                    )
                    counts["off_shape_subspace"] += int(
                        "OUTSIDE_DECLARED_SHAPE_SUBSPACE" in batch.geometry_domain_status
                    )
                    actual_safe = geometry.signed_distance_m >= cfg.pcc_clearance_safe_m
                    counts["mujoco_geometry_safe"] += int(actual_safe)
                    counts["proxy_false_safe_vs_mujoco"] += int(
                        batch.all_intervals_safe and not actual_safe
                    )
                    counts["proxy_false_reject_vs_mujoco"] += int(
                        not batch.all_intervals_safe and actual_safe
                    )
                    selected_unsupported = any(
                        row.derivative_status != "SUPPORTED"
                        for row in batch.rows if row.interval_id in selected_ids
                    )
                    if (not within_budget or selected_unsupported
                            or result.proxy_clearance_status == "UNKNOWN_CROSSES_GATE") and len(examples) < 20:
                        examples.append({
                            "scenario_id": scenario_result["scenario"]["scenario_id"],
                            "tick": tick, "time_s": float(data.time),
                            "proxy_status": result.proxy_clearance_status,
                            "minimum_h_m": min(x.h_m for x in batch.rows),
                            "actual_mujoco_distance_m": geometry.signed_distance_m,
                            "subspace_residual_linf_rad": batch.subspace_residual_linf_rad,
                            "query_time_ms": total_time,
                            "point_evaluations": point_count,
                            "jacobian_evaluations": batch.jacobian_evaluation_count,
                            "selected_interval_count": len(selected_ids),
                            "unsupported_interval_ids": [
                                x.interval_id for x in batch.rows
                                if x.interval_id in selected_ids
                                and x.derivative_status != "SUPPORTED"
                            ][:12],
                        })
                    feasibility_started = time.perf_counter()
                    original_rows = feasibility_builder.build(data)
                    aggregates["frozen_original_assembly_ms"].append(
                        (time.perf_counter() - feasibility_started) * 1000.0
                    )
                    try:
                        matrix, row_lower, drifts, gains, sources = combined_rows(
                            original_rows, batch, selected_ids, cfg,
                        )
                        feasibility = solve_frozen_linear_feasibility(
                            matrix, row_lower, drifts, gains, sources,
                            velocity_lower, velocity_upper, previous, cfg,
                            interval_row_count=len(selected_ids),
                            historical_endpoint=selected[tick],
                        )
                        feasibility_record = feasibility.to_dict()
                        counts[f"frozen_{feasibility.status}"] += 1
                        counts["frozen_executable_false"] += int(
                            feasibility.executable_candidate_exists is False
                        )
                        counts["frozen_lp_unknown"] += int(
                            feasibility.candidate_feasible is None
                        )
                        counts["historical_endpoint_rejected_by_new_rows"] += int(
                            feasibility.historical_endpoint_accepted_by_frozen_rows is False
                        )
                        aggregates["frozen_lp_ms"].append(feasibility.lp_time_ms)
                        aggregates["frozen_row_count"].append(feasibility.row_count)
                    except ValueError as exc:
                        feasibility_record = {
                            "status": "UNKNOWN_REQUIRED_ROW",
                            "reason": str(exc),
                            "executable_candidate_exists": None,
                        }
                        counts["frozen_UNKNOWN_REQUIRED_ROW"] += 1
                    feasibility_records.append({
                        "scenario_id": scenario_result["scenario"]["scenario_id"],
                        "tick": tick,
                        "query_budget_acceptable": within_budget,
                        "proxy_status": result.proxy_clearance_status,
                        "subspace_residual_linf_rad": batch.subspace_residual_linf_rad,
                        "feasibility": feasibility_record,
                    })
                    if tick + 1 < len(selected):
                        old = np.zeros(17) if tick == 0 else selected[tick - 1]
                        endpoint = selected[tick]
                        old_weight, new_weight = ramp_mean_weights()
                        mean_velocity = old_weight * old + new_weight * endpoint
                        predicted = {}
                        residuals = {}
                        for row in batch.rows:
                            if row.interval_id not in selected_ids or row.derivative_status != "SUPPORTED":
                                continue
                            rate = float(row.generalized_gradient @ mean_velocity
                                         + row.target_drift_m_s)
                            h_hat = row.h_m + cfg.task_period_s * rate
                            predicted[row.interval_id] = h_hat
                            residuals[row.interval_id] = float(
                                row.generalized_gradient @ endpoint
                                + row.target_drift_m_s
                                + cfg.pcc_clearance_barrier_gain * h_hat
                            )
                        pending = (tick, partition, selected_ids,
                                   predicted, residuals, endpoint)
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
        scenario_reports.append({
            "scenario_id": scenario_result["scenario"]["scenario_id"],
            "trace_sha256": trace_hash, "trace_hash_matches": hash_matches,
            "sampled_task_ticks": samples,
            "max_native_replay_state_error": max_state_error,
        })
        if max_state_error > 1e-8:
            counts["native_replay_state_mismatch"] += 1
        print(f"[b2-shadow] {mode} {scenario_reports[-1]['scenario_id']}: {samples} samples", flush=True)
    return {
        "mode": mode,
        "metrics_sha256": _sha(metrics_path),
        "sample_stride_task_ticks": sample_stride,
        "query_mode": query_mode,
        "budget": budget.__dict__,
        "scenarios": scenario_reports,
        "sample_count": sum(x["sampled_task_ticks"] for x in scenario_reports),
        "counts": dict(counts),
        "summaries": {name: _summary(values) for name, values in aggregates.items()},
        "examples": examples,
        "warm_unknown_examples": warm_unknown_examples,
        "frozen_feasibility_records": feasibility_records,
    }


def run_shadow(output_dir: Path, *, a1_root: Path = DEFAULT_A1_ROOT,
               sample_stride: int = 50,
               budget: ShadowBudget = ShadowBudget(),
               query_mode: str = "bounded_cold") -> dict:
    if sample_stride < 2:
        raise ValueError("sample stride must leave a next-cycle comparison tick")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "v6_2_b2_native_replay_interval_shadow_v1",
        "source_sha256": {
            "v6_lite/pcc_interval_cbf.py": _sha(Path(__file__).with_name("pcc_interval_cbf.py")),
            "v6_lite/continuum_shape_model.py": _sha(Path(__file__).with_name("continuum_shape_model.py")),
            "v6_lite/pcc_persistent_interval_query.py": _sha(Path(__file__).with_name("pcc_persistent_interval_query.py")),
            "v6_lite/b2_shadow_feasibility.py": _sha(Path(__file__).with_name("b2_shadow_feasibility.py")),
            "v6_lite/recompute_execution_constraints.py": _sha(Path(__file__).with_name("recompute_execution_constraints.py")),
            "v6_lite/shadow_b2_interval_cbf.py": _sha(Path(__file__)),
        },
        "sample_plan": "all five scenarios in each A.1 mode; task ticks 0, stride, ...; evaluate next tick on frozen partition",
        "query_mode": query_mode,
        "modes": {},
        "online_control_changed": False,
        "old_trace_reused_as_new_control_acceptance": False,
        "task_spatial_refresh": "mj_forward_before_each_50Hz_geometry_query",
    }
    for mode in ("baseline", "enabled"):
        report["modes"][mode] = shadow_mode(
            mode, a1_root / f"{mode}_root" / "output",
            sample_stride=sample_stride, budget=budget, query_mode=query_mode,
        )
    report["checks"] = {
        "two_modes_five_scenarios_each": all(
            len(x["scenarios"]) == 5 for x in report["modes"].values()),
        "all_a1_trace_hashes_match": all(
            all(s["trace_hash_matches"] for s in x["scenarios"])
            for x in report["modes"].values()),
        "native_replay_states_match": all(
            x["counts"].get("native_replay_state_mismatch", 0) == 0
            for x in report["modes"].values()),
        "all_partitions_cover_five_sections": all(
            x["counts"].get("coverage_incomplete", 0) == 0
            for x in report["modes"].values()),
        "warm_query_valid_if_enabled": all(
            x["counts"].get("warm_all_task_invalid", 0) == 0
            for x in report["modes"].values()),
    }
    report["passed_as_read_only_audit"] = all(report["checks"].values())
    report["online_admission_gate"] = {
        "status": "NOT_MET",
        "reason": "new-mode full-loop timing and unknown/budget handling require further work",
        "shadow_conditions": {
            "no_unknown_proxy_decisions": all(
                x["counts"].get("warm_all_task_unknown", 0) == 0
                and x["counts"].get("UNKNOWN_CROSSES_GATE", 0) == 0
                for x in report["modes"].values()),
            "no_query_budget_exhaustion": all(
                x["counts"].get("warm_all_task_budget_exhausted", 0) == 0
                and x["counts"].get("unified_budget_exceeded", 0) == 0
                for x in report["modes"].values()),
            "no_excluded_interval_reached_activation_next_tick": all(
                x["counts"].get("excluded_reached_activation_next_tick", 0) == 0
                for x in report["modes"].values()),
            "no_required_derivative_unsupported": all(
                x["counts"].get("derivative_unsupported", 0) == 0
                for x in report["modes"].values()),
            "no_empty_candidate_velocity_box": all(
                x["counts"].get("empty_candidate_velocity_box", 0) == 0
                for x in report["modes"].values()),
            "no_frozen_action_infeasibility": all(
                x["counts"].get("frozen_executable_false", 0) == 0
                for x in report["modes"].values()),
            "no_frozen_lp_unknown": all(
                x["counts"].get("frozen_lp_unknown", 0) == 0
                and x["counts"].get("frozen_UNKNOWN_REQUIRED_ROW", 0) == 0
                for x in report["modes"].values()),
        },
        "paired_timing_is_estimate_not_new_execution": True,
    }
    shadow_conditions = report["online_admission_gate"]["shadow_conditions"]
    if all(shadow_conditions.values()):
        report["online_admission_gate"]["status"] = "REQUIRES_TRUE_NEW_MODE_TIMING"
        report["online_admission_gate"]["reason"] = (
            "shadow checks passed; new-mode full-loop timing is not yet measured"
        )
    report_path = output_dir / "shadow_report.json"
    _write(report_path, report)
    lines = ["# V6.2-B.2 A.1 原生力矩重放状态区间影子审计", "",
             f"只读证据检查：{'通过' if report['passed_as_read_only_audit'] else '未通过'}。这不是新模式闭环验收。", "",
             f"在线接入门禁：**{report['online_admission_gate']['status']}**。", "",
             "| 模式 | 采样状态 | 代理未知 | 完整预算超限 | 几何假安全 | 几何误拒绝 | 子空间外 | 查询 p95 (ms) | 配对容量估计 p95 (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, item in report["modes"].items():
        c = item["counts"]
        lines.append(
            f"| {mode} | {item['sample_count']} | {c.get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{c.get('unified_budget_exceeded', 0)} | {c.get('proxy_false_safe_vs_mujoco', 0)} | "
            f"{c.get('proxy_false_reject_vs_mujoco', 0)} | {c.get('off_shape_subspace', 0)} | "
            f"{item['summaries']['query_time_ms'].get('p95', float('nan')):.3f} | "
            f"{item['summaries']['paired_replacement_estimate_ms'].get('p95', float('nan')):.3f} |"
        )
    if query_mode == "persistent_warm":
        lines += ["", "持久分区在全部 50 Hz task tick 上更新，不合并区间；"
                  "下面的未知与耗尽次数包含未采样的 tick：", "",
                  "| 模式 | 全部 tick 未知 | 全部 tick 预算耗尽 | 最大叶区间数 | 排除区间下一 tick 进入激活区 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for mode, item in report["modes"].items():
            c = item["counts"]
            lines.append(
                f"| {mode} | {c.get('warm_all_task_unknown', 0)} | "
                f"{c.get('warm_all_task_budget_exhausted', 0)} | "
                f"{item['summaries']['warm_all_task_partition_size'].get('max', 0):.0f} | "
                f"{c.get('excluded_reached_activation_next_tick', 0)} |"
            )
    lines += [
        "", "## 冻结行的只读可行性诊断", "",
        "诊断从当前原生重放状态独立重算 MuJoCo 与实际链胶囊行，再加入所需区间行；"
        "两组都启用胶囊行，以模拟新模式保留该约束。离线线性规划只判断在既有容差、"
        "速度盒、十步斜坡起点和冻结前瞻行下是否存在终点，不是第二个在线 QP，"
        "也不是新动作的执行证据。", "",
        "| 模式 | 抽样状态 | 起点行违反 | 无可行终点 | LP 未知 | 历史终点不通过新区间行 | LP p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in report["modes"].items():
        c = item["counts"]
        lines.append(
            f"| {mode} | {item['sample_count']} | "
            f"{c.get('frozen_START_CLEARANCE_VIOLATION', 0) + c.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{c.get('frozen_NO_FEASIBLE_ENDPOINT', 0)} | "
            f"{c.get('frozen_lp_unknown', 0) + c.get('frozen_UNKNOWN_REQUIRED_ROW', 0)} | "
            f"{c.get('historical_endpoint_rejected_by_new_rows', 0)} | "
            f"{item['summaries']['frozen_lp_ms'].get('p95', float('nan')):.3f} |"
        )
    lines += [
        "", "查询预算包含 B.1 分支点、局部优化点、区间中点和 Jacobian；"
        "此影子运行未启用局部优化。记录的墙钟查询时间不包含新 QP 求解，"
        "不能与 A.1 全链 p95 直接相减或作为新模式 20 ms 验收。"
        "逐状态 `paired_replacement_estimate_ms` 仅用历史全链耗时扣除历史形状查询再加影子查询；"
        "它是配对容量预警，不是真实新模式时延。",
        "", "历史 trace 的力矩按 500 Hz 原生重放，每个 50 Hz 规划边界先执行"
        " `mj_forward` 更新空间几何量，再与保存状态逐点对比。"
        "区间拓扑在采样 tick 到下一 tick 的预测检查中冻结。实际形状子空间残差"
        "与几何包络状态分开报告。旧 trace 的起点违反新区间行，不能用一个新终点"
        "的线性可行性消除；本审计不能证明全域或连续时间安全。", "",
        "区间激活筛选使用当前速度、加速度与关节限位形成的候选盒，并把十步斜坡"
        "起点速度纳入逐轴绝对上界；若候选盒为空，则退回全局速度上界加起点速度，"
        "同时记录空盒并使在线门禁失败。此筛选仍冻结基座反作用映射、目标漂移"
        "和几何灵敏度，不构成跨周期或连续时间安全证明。", "",
    ]
    doc_path = output_dir / "SHADOW_AUDIT.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {"schema": "v6_2_b2_shadow_manifest_v1",
                "artifacts": [{"path": p.name, "sha256": _sha(p),
                               "bytes": p.stat().st_size}
                              for p in (report_path, doc_path)]}
    _write(output_dir / "shadow_manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--query-mode", choices=("bounded_cold", "persistent_warm"),
                        default="bounded_cold")
    args = parser.parse_args()
    budget = (ShadowBudget(branch_point_evaluations=64,
                           total_point_evaluations=128,
                           jacobian_evaluations=96,
                           total_query_time_ms=20.0)
              if args.query_mode == "persistent_warm" else ShadowBudget())
    result = run_shadow(args.output_dir, a1_root=args.a1_root,
                        sample_stride=args.sample_stride, budget=budget,
                        query_mode=args.query_mode)
    print(json.dumps({"passed_as_read_only_audit": result["passed_as_read_only_audit"],
                      "checks": result["checks"]}, indent=2))
    if not result["passed_as_read_only_audit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
