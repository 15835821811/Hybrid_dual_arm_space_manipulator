"""Replay new-mode torques and independently rebuild fixed interval rows.

This verifier reads saved states and commands, never calls the online QP or
its constraint assembly, and preserves every mismatch in a separate run dir.
It shares the declared MuJoCo/PCC models; it is not physical measurement or a
continuous-time safety proof.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_screened_next_start_rows import ScreenedReplayConstraintBuilder
from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import ramp_mean_weights, ramp_velocity
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import (
    _obstacles,
)
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


STEPS = 10
POINT_BUDGET = 255


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recomputed_ramp_and_lookahead(matrix, lower, drifts, gains,
                                  previous, endpoint, cfg):
    ramp_min = min(float(np.min(
        matrix @ ramp_velocity(previous, endpoint, step) - lower))
        for step in range(STEPS + 1))
    old_weight, new_weight = ramp_mean_weights()
    gain_dt = gains * cfg.task_period_s
    lookahead_matrix = matrix * (1.0 + gain_dt * new_weight)[:, None]
    lookahead_lower = (lower - gain_dt * (
        old_weight * (matrix @ previous) + drifts)
        + cfg.lookahead_model_margin_m_s)
    lookahead_min = float(np.min(
        lookahead_matrix @ endpoint - lookahead_lower))
    return ramp_min, lookahead_min


def run(run_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics["run_config"].get("pcc_mode") != "bounded_interval_pcc"
            or len(metrics["scenarios"]) != 5):
        raise ValueError("five bounded interval scenarios required")
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    robot = default_v6_lite_robot_spec()
    check_rows = []
    interval_rows = []
    failures = []
    inputs = []
    for scene in metrics["scenarios"]:
        scene_id = scene["scenario"]["scenario_id"]
        trace_path = Path(scene["trace"]["path"])
        trace_sha = _sha(trace_path)
        if trace_sha != scene["trace"]["sha256"]:
            raise ValueError(f"{scene_id}: trace SHA mismatch")
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(scene["scenario"]), WholeBodyVerificationConfig(
                minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
                query_distance_max=2.5,
                adaptive_subdivisions=run_cfg.verification_subdivisions,
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
        builder = ScreenedReplayConstraintBuilder(
            robot, model, verifier.pairs, no_legacy)
        query = PersistentIntervalDecisionQuery(evaluator.shape_model)
        with np.load(trace_path, allow_pickle=False) as trace:
            initial_qpos = trace["initial_qpos"].copy()
            initial_qvel = trace["initial_qvel"].copy()
            torque = trace["torque"].copy()
            task_qpos = trace["task_qpos"].copy()
            selected = trace["task_selected_command"].copy()
            logged_query = trace["task_interval_proxy_lower_m"].copy()
            logged_rows = trace["task_interval_selected_rows"].copy()
            logged_current_margin = trace[
                "task_interval_current_envelope_margin_m"].copy()
            logged_next_start = trace[
                "task_interval_realized_next_start_minimum_slack_m_s"].copy()
            logged_ramp = trace["task_ramp_clearance_min_slack_m_s"].copy()
            logged_lookahead = trace["task_lookahead_min_slack_m_s"].copy()
        ticks = len(selected)
        if (ticks != 1350 or torque.shape != (ticks * STEPS, 67)
                or task_qpos.shape[0] != ticks + 1):
            raise ValueError(f"{scene_id}: online trace dimensions changed")
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0
        max_qpos_error = 0.0
        max_query_error = 0.0
        max_start_error = 0.0
        max_ramp_error = 0.0
        max_lookahead_error = 0.0
        interval_row_count = 0
        for tick in range(ticks + 1):
            mujoco.mj_forward(model, data)
            errors = []
            qpos_error = float(np.max(np.abs(data.qpos - task_qpos[tick])))
            max_qpos_error = max(max_qpos_error, qpos_error)
            if qpos_error > 1e-9:
                errors.append("NATIVE_TASK_STATE_MISMATCH")
            projection = evaluator.shape_spec.project_actual_configuration(
                data.qpos[evaluator.qpos_ids[:60]])
            base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
            current_envelope = envelope.evaluate(
                data, projection.planner_configuration, base)
            box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
            decision = query.evaluate(
                projection.planner_configuration, base, box,
                IntervalPartition.uniform(), max_point_evaluations=POINT_BUDGET)
            previous = (np.zeros(17, dtype=np.float64) if tick == 0
                        else selected[tick-1])
            planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
            _, _, speed, box_valid = ramp_velocity_abs_bound(
                robot, cfg, planner_q, previous)
            generalized_map, _ = builder._reaction_map(data)
            ids, _ = select_intervals_by_frozen_reach(
                decision.lower_by_interval_id, decision.partition,
                evaluator, data, cfg, generalized_map,
                velocity_abs_bound=speed)
            batch = evaluator.evaluate_state(
                data, decision.partition, generalized_map=generalized_map,
                derivative_interval_ids=ids)
            original = builder.build(data)
            matrix, lower, drifts, gains, sources = combined_rows(
                original, batch, ids, cfg)
            starts = matrix @ previous - lower
            start_slack = float(np.min(starts))
            if (not box_valid or not decision.bounds_valid
                    or decision.proxy_clearance_status
                    != "PROXY_CLEARANCE_AT_LEAST_GATE"
                    or current_envelope.status != "COVERED_AT_THIS_STATE"
                    or not batch.interval_well_formed
                    or not batch.coverage_complete
                    or not batch.analytic_bound_assumptions_satisfied
                    or any(row.derivative_status != "SUPPORTED"
                           for row in batch.rows if row.interval_id in ids)):
                errors.append("RECOMPUTED_INTERVAL_UNSUPPORTED")
            if tick < ticks:
                query_error = max(
                    abs(decision.distance_lower_bound_m - logged_query[tick]),
                    abs(current_envelope.min_margin_m
                        - logged_current_margin[tick]))
                max_query_error = max(max_query_error, query_error)
                if query_error > 1e-8 or len(ids) != logged_rows[tick]:
                    errors.append("QUERY_OR_INTERVAL_SELECTION_MISMATCH")
                ramp_min, lookahead_min = _recomputed_ramp_and_lookahead(
                    matrix, lower, drifts, gains, previous, selected[tick], cfg)
                ramp_error = abs(ramp_min - logged_ramp[tick])
                lookahead_error = abs(
                    lookahead_min - logged_lookahead[tick])
                max_ramp_error = max(max_ramp_error, ramp_error)
                max_lookahead_error = max(max_lookahead_error, lookahead_error)
                if (ramp_error > 1e-7 or lookahead_error > 1e-7):
                    errors.append("RAMP_OR_LOOKAHEAD_LOG_MISMATCH")
                if (ramp_min < -cfg.clearance_rate_tolerance_m_s
                        or lookahead_min < -cfg.clearance_rate_tolerance_m_s):
                    errors.append("RECOMPUTED_ACTION_CONSTRAINT_VIOLATION")
            if tick:
                start_error = abs(start_slack - logged_next_start[tick-1])
                max_start_error = max(max_start_error, start_error)
                if start_error > 1e-7:
                    errors.append("NEXT_START_LOG_MISMATCH")
                if start_slack < -cfg.clearance_rate_tolerance_m_s:
                    errors.append("REALIZED_NEXT_START_VIOLATION")
            row = {
                "scenario_id": scene_id, "tick": tick,
                "query_status": decision.proxy_clearance_status,
                "selected_interval_count": len(ids),
                "recomputed_row_count": len(sources),
                "recomputed_start_slack_m_s": start_slack,
                "geometry_domain_status": batch.geometry_domain_status,
                "errors": errors,
            }
            check_rows.append(row)
            if errors:
                failures.append(row)
            for interval in batch.rows:
                if interval.interval_id in ids:
                    interval_row_count += 1
                    interval_rows.append({
                        "scenario_id": scene_id, "tick": tick,
                        "interval_id": interval.interval_id,
                        "h_m": interval.h_m,
                        "distance_lower_bound_m": interval.distance_lower_bound_m,
                        "gradient_17d": interval.generalized_gradient.tolist(),
                        "target_drift_m_s": interval.target_drift_m_s,
                        "derivative_status": interval.derivative_status,
                    })
            if tick < ticks:
                for step in range(STEPS):
                    data.ctrl[:] = torque[tick*STEPS + step]
                    mujoco.mj_step(model, data)
        inputs.append({
            "scenario_id": scene_id,
            "trace_sha256": trace_sha,
            "task_ticks": ticks,
            "torque_steps": len(torque),
            "interval_rows_recomputed": interval_row_count,
            "maximum_native_qpos_error": max_qpos_error,
            "maximum_query_error_m": max_query_error,
            "maximum_next_start_error_m_s": max_start_error,
            "maximum_ramp_error_m_s": max_ramp_error,
            "maximum_lookahead_error_m_s": max_lookahead_error,
            "failure_count": sum(bool(row["errors"]) for row in check_rows
                                 if row["scenario_id"] == scene_id),
        })
        print(f"[b2-online-recompute] {scene_id}: "
              f"{inputs[-1]['failure_count']} mismatches", flush=True)
    output_paths = {
        "checks": output_dir / "online_recompute_checks.jsonl",
        "interval_rows": output_dir / "online_recompute_interval_rows.jsonl",
        "failures": output_dir / "online_recompute_failures.jsonl",
    }
    for name, path in output_paths.items():
        values = (check_rows if name == "checks" else interval_rows
                  if name == "interval_rows" else failures)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":"),
                                        allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_online_interval_independent_recompute_v1",
        "scope": "new_online_trace_native_torque_replay_and_separate_interval_row_recompute",
        "passed": not failures,
        "scenario_count": len(inputs),
        "task_state_checks": len(check_rows),
        "interval_rows_recomputed": len(interval_rows),
        "failure_count": len(failures),
        "inputs": inputs,
        "input_metrics_sha256": _sha(metrics_path),
        "output_sha256": {name: _sha(path)
                          for name, path in output_paths.items()},
        "continuous_time_certified": False,
    }
    (output_dir / "online_recompute_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir, args.output_dir)
    print(json.dumps({"passed": result["passed"],
                      "task_state_checks": result["task_state_checks"],
                      "interval_rows_recomputed": result[
                          "interval_rows_recomputed"],
                      "failure_count": result["failure_count"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
