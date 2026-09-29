"""Independently replay private torques and rebuild frozen interval start rows.

This verifier never calls the QP solver or consumes the controller's interval
batch. It does share the declared MuJoCo and PCC geometry implementations;
it is a separate computation, not independent physical measurement or a
continuous-time proof. Failed comparisons remain in the output directory.
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
from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


SCENES = tuple(f"v6_lite_scenario_{index:02d}" for index in range(5))
TICKS = 400
STEPS = 10
POINT_BUDGET = 255


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(root: Path, a1_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    scene_data = {item["scenario"]["scenario_id"]: item
                  for item in metrics["scenarios"]}
    checks = []
    interval_rows = []
    failures = []
    inputs = []
    for index, scene in enumerate(SCENES):
        saved = scene_data[scene]
        folder = root / f"scene_{index:02d}"
        report_path = folder / "private_rollout_summary.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = folder / "private_rollout_trace.npz"
        records_path = folder / "private_rollout_records.jsonl"
        records = [json.loads(line) for line in records_path.read_text(
            encoding="utf-8").splitlines()]
        if (report["scenario_id"] != scene or report["executed_ticks"] != TICKS
                or report["stop_reason"] != "HORIZON_COMPLETE"
                or report["trace_sha256"] != _sha(trace_path)
                or report["records_sha256"] != _sha(records_path)
                or len(records) != TICKS):
            raise ValueError(f"{scene} private trace protocol changed")
        inputs.append({"scenario_id": scene,
                       "summary_path": report_path.as_posix(),
                       "summary_sha256": _sha(report_path),
                       "trace_sha256": _sha(trace_path),
                       "records_sha256": _sha(records_path),
                       "claimed_strict_online_domain_all_executed_ticks":
                           report["strict_online_domain_all_executed_ticks"]})
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
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
        data = mujoco.MjData(model)
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
        independent = ReplayConstraintBuilder(robot, model, verifier.pairs,
                                              no_legacy)
        query = PersistentIntervalDecisionQuery(evaluator.shape_model)
        with np.load(trace_path, allow_pickle=False) as trace:
            initial_qpos = trace["initial_qpos"].copy()
            initial_qvel = trace["initial_qvel"].copy()
            torques = trace["torque"].copy()
            expected_qpos = trace["qpos_states"].copy()
            expected_task_qvel = trace["task_qvel_states"].copy()
        if (torques.shape != (TICKS * STEPS, 67)
                or expected_qpos.shape[0] != TICKS * STEPS + 1
                or expected_task_qvel.shape[0] != TICKS + 1):
            raise ValueError(f"{scene} trace dimensions changed")
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0
        max_state_error = 0.0
        max_query_error = 0.0
        max_start_error = 0.0
        max_microstep_subspace_residual = 0.0
        first_microstep_outside_strict_subspace = None
        torque_limit_violation_count = int(np.count_nonzero(
            np.abs(torques) > robot.torque_limits[None, :] + 1e-10))
        torque_at_limit_step_count = int(np.count_nonzero(np.any(
            np.abs(torques) >= robot.torque_limits[None, :] - 1e-9,
            axis=1)))
        if torque_limit_violation_count:
            raise ValueError(f"{scene} saved torque exceeded original limits")
        for step in range(TICKS * STEPS + 1):
            max_state_error = max(max_state_error, float(np.max(np.abs(
                data.qpos - expected_qpos[step]))))
            actual_projection = evaluator.shape_spec.project_actual_configuration(
                data.qpos[evaluator.qpos_ids[:60]])
            microstep_residual = actual_projection.residual_linf_rad
            max_microstep_subspace_residual = max(
                max_microstep_subspace_residual, microstep_residual)
            if (first_microstep_outside_strict_subspace is None
                    and microstep_residual > 1e-10):
                first_microstep_outside_strict_subspace = step
            if step % STEPS == 0:
                tick = step // STEPS
                max_state_error = max(max_state_error, float(np.max(np.abs(
                    data.qvel - expected_task_qvel[tick]))))
                mujoco.mj_forward(model, data)
                projection = evaluator.shape_spec.project_actual_configuration(
                    data.qpos[evaluator.qpos_ids[:60]])
                base = transform_from_free_qpos(
                    data.qpos[evaluator.base_qpos_slice])
                current_envelope = envelope.evaluate(
                    data, projection.planner_configuration, base)
                box = target_box_from_mujoco(
                    model, data, evaluator.target_geom_id)
                decision = query.evaluate(
                    projection.planner_configuration, base, box,
                    IntervalPartition.uniform(),
                    max_point_evaluations=POINT_BUDGET,
                )
                previous = (np.zeros(17, dtype=np.float64) if tick == 0 else
                            np.asarray(records[tick - 1]["selected_command"],
                                       dtype=np.float64))
                planner_q = robot.low_level_to_planner @ data.qpos[
                    evaluator.qpos_ids]
                _lo, _hi, speed, box_valid = ramp_velocity_abs_bound(
                    robot, cfg, planner_q, previous)
                generalized_map, _reaction_residual = independent._reaction_map(data)
                ids, _excluded = select_intervals_by_frozen_reach(
                    decision.lower_by_interval_id, decision.partition,
                    evaluator, data, cfg, generalized_map,
                    velocity_abs_bound=speed,
                )
                batch = evaluator.evaluate_state(
                    data, decision.partition, generalized_map=generalized_map,
                    derivative_interval_ids=ids,
                )
                original = independent.build(data)
                matrix, lower, _drifts, _gains, sources = combined_rows(
                    original, batch, ids, cfg)
                slacks = matrix @ previous - lower
                worst_index = int(np.argmin(slacks)) if len(slacks) else None
                start_slack = (float(slacks[worst_index])
                               if worst_index is not None else None)
                start_source = (sources[worst_index]
                                if worst_index is not None else None)
                reference = records[tick] if tick < TICKS else None
                prior = (records[tick - 1]["realized_next_start"]
                         if tick else None)
                query_error = 0.0
                start_error = 0.0
                errors = []
                if reference is not None:
                    query_error = max(
                        abs(decision.distance_lower_bound_m
                            - reference["proxy_lower_m"]),
                        abs(decision.distance_upper_bound_m
                            - reference["proxy_upper_m"]),
                        abs(current_envelope.min_margin_m
                            - reference["current_envelope_margin_m"]),
                    )
                    if (decision.proxy_clearance_status
                            != reference["proxy_status"]
                            or decision.point_evaluation_count
                            != reference["point_evaluations"]
                            or len(ids) != reference["selected_interval_count"]
                            or current_envelope.status
                            != reference["current_envelope_status"]):
                        errors.append("CURRENT_QUERY_OR_SELECTION_CHANGED")
                if prior is not None:
                    if start_slack is None:
                        errors.append("RECOMPUTED_START_HAS_NO_ROWS")
                    else:
                        start_error = abs(start_slack
                                          - prior["minimum_start_slack_m_s"])
                    if (start_source != prior["worst_source"]
                            or len(slacks) != prior["row_count"]
                            or len(ids) != prior["selected_interval_count"]
                            or decision.proxy_clearance_status
                            != prior["proxy_status"]):
                        errors.append("NEXT_START_ROW_IDENTITY_CHANGED")
                if (not box_valid or not batch.coverage_complete
                        or not batch.interval_well_formed
                        or not batch.analytic_bound_assumptions_satisfied
                        or any(row.derivative_status != "SUPPORTED"
                               for row in batch.rows if row.interval_id in ids)):
                    errors.append("RECOMPUTED_INTERVAL_UNSUPPORTED")
                if query_error > 1e-8 or start_error > 1e-7:
                    errors.append("NUMERICAL_RECOMPUTE_MISMATCH")
                max_query_error = max(max_query_error, query_error)
                max_start_error = max(max_start_error, start_error)
                check = {"scenario_id": scene, "tick": tick,
                         "time_s": float(data.time),
                         "query_status": decision.proxy_clearance_status,
                         "point_evaluations": decision.point_evaluation_count,
                         "selected_interval_count": len(ids),
                         "recomputed_row_count": len(slacks),
                         "recomputed_start_slack_m_s": start_slack,
                         "recomputed_worst_source": start_source,
                         "query_error_m": query_error,
                         "start_slack_error_m_s": start_error,
                         "state_error": max_state_error,
                         "strict_subspace_status":
                             batch.geometry_domain_status,
                         "subspace_residual_linf_rad":
                             batch.subspace_residual_linf_rad,
                         "actual_envelope_status": current_envelope.status,
                         "errors": errors}
                checks.append(check)
                if errors:
                    failures.append(check)
                for row in batch.rows:
                    if row.interval_id in ids:
                        interval_rows.append({
                            "scenario_id": scene, "tick": tick,
                            "interval_id": row.interval_id,
                            "h_m": row.h_m,
                            "distance_lower_bound_m":
                                row.distance_lower_bound_m,
                            "gradient_17d": row.generalized_gradient.tolist(),
                            "target_drift_m_s": row.target_drift_m_s,
                            "derivative_status": row.derivative_status,
                        })
            if step < TICKS * STEPS:
                data.ctrl[:] = torques[step]
                mujoco.mj_step(model, data)
        print(f"[b2-private-recompute] {scene}: "
              f"{TICKS + 1} task states, "
              f"{sum(item['scenario_id'] == scene for item in failures)} mismatches",
              flush=True)
        inputs[-1]["max_state_error"] = max_state_error
        inputs[-1]["max_query_error_m"] = max_query_error
        inputs[-1]["max_start_slack_error_m_s"] = max_start_error
        inputs[-1]["max_500hz_subspace_residual_linf_rad"] = (
            max_microstep_subspace_residual)
        inputs[-1]["first_500hz_step_outside_strict_subspace"] = (
            first_microstep_outside_strict_subspace)
        inputs[-1]["torque_limit_violation_count"] = torque_limit_violation_count
        inputs[-1]["torque_at_limit_step_count"] = torque_at_limit_step_count
        independent_strict = all(
            item["strict_subspace_status"]
            == "INSIDE_DECLARED_WORK_DOMAIN;ON_DECLARED_SHAPE_SUBSPACE"
            for item in checks if item["scenario_id"] == scene
            and item["tick"] < TICKS)
        inputs[-1]["independent_strict_online_domain_all_executed_ticks"] = (
            independent_strict)
        if independent_strict != report["strict_online_domain_all_executed_ticks"]:
            failures.append({"scenario_id": scene,
                             "error": "STRICT_DOMAIN_CLAIM_DISAGREES_WITH_REPLAY"})
    paths = {"checks": output_dir / "private_recompute_checks.jsonl",
             "interval_rows": output_dir / "private_recompute_interval_rows.jsonl",
             "failures": output_dir / "private_recompute_failures.jsonl"}
    for name, path in paths.items():
        values = (checks if name == "checks" else
                  interval_rows if name == "interval_rows" else failures)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for item in values:
                stream.write(json.dumps(item, ensure_ascii=False,
                                        separators=(",", ":"),
                                        allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_private_torque_interval_recompute_v1",
        "scope": "private_trace_native_torque_replay_and_separate_interval_start_recompute",
        "scenes": list(SCENES), "task_ticks_per_scene": TICKS,
        "torque_steps_per_scene": TICKS * STEPS,
        "checked_task_states": len(checks),
        "checked_interval_rows": len(interval_rows),
        "failure_count": len(failures),
        "maximum_native_state_error": max(item["max_state_error"] for item in inputs),
        "maximum_query_error_m": max(item["max_query_error_m"] for item in inputs),
        "maximum_start_slack_error_m_s": max(
            item["max_start_slack_error_m_s"] for item in inputs),
        "pass_recompute": not failures and all(
            item["max_state_error"] <= 1e-8 for item in inputs),
        "independent_strict_online_domain_all_executed_ticks": all(
            item["independent_strict_online_domain_all_executed_ticks"]
            for item in inputs),
        "all_500hz_states_on_declared_shape_subspace": all(
            item["first_500hz_step_outside_strict_subspace"] is None
            for item in inputs),
        "maximum_500hz_subspace_residual_linf_rad": max(
            item["max_500hz_subspace_residual_linf_rad"] for item in inputs),
        "torque_limit_violation_count": sum(
            item["torque_limit_violation_count"] for item in inputs),
        "torque_at_limit_step_count": sum(
            item["torque_at_limit_step_count"] for item in inputs),
        "new_mode_online_admitted": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "shared_model_geometry_with_private_controller": True,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_private_recompute.py", "pcc_interval_cbf.py",
            "pcc_persistent_interval_query.py",
            "recompute_execution_constraints.py",
            "b2_shadow_feasibility.py",
        )},
        "input_metrics_sha256": _sha(metrics_path),
        "inputs": inputs,
        **{f"{name}_sha256": _sha(path) for name, path in paths.items()},
    }
    report_path = output_dir / "private_recompute_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 私有力矩轨迹的单独重放与区间重算", "",
        f"五场景各 {TICKS * STEPS} 个原生力矩步，从私有 trace 保存的初态"
        "重新执行 MuJoCo；每个规划边界重新做根区间查询、固定区间 17 维"
        "梯度及目标漂移、原 MuJoCo／胶囊约束和当前起点残差。"
        "本程序不调用 QP 求解，也不复用控制分支的区间批次。", "",
        f"检查规划状态 {len(checks)}，区间行 {len(interval_rows)}，"
        f"不一致 {len(failures)}。最大原生状态误差 "
        f"{report['maximum_native_state_error']:.3e}；最大代理查询差 "
        f"{report['maximum_query_error_m']:.3e} m；最大起点松弛差 "
        f"{report['maximum_start_slack_error_m_s']:.3e} m/s。", "",
        f"重放的 20,005 个 500 Hz 状态中，严格子空间判据全满足："
        f"{report['all_500hz_states_on_declared_shape_subspace']}；"
        f"最大残差 {report['maximum_500hz_subspace_residual_linf_rad']:.3e} rad。"
        f"原力矩上限违例 {report['torque_limit_violation_count']}，"
        f"触及上限的力矩步 {report['torque_at_limit_step_count']}。", "",
        "仍共享声明的 MuJoCo 与 PCC 几何实现，属于独立运行的模型内重算，"
        "不是独立物理测量。"
        + ("所测规划执行状态满足严格子空间判据；"
           if report["independent_strict_online_domain_all_executed_ticks"]
           else "所测规划执行状态越过了严格子空间判据；")
        + "本重算仍不构成 stage 3 在线验收或连续时间证明。", "",
    ]
    document_path = output_dir / "PRIVATE_RECOMPUTE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_private_recompute_manifest_v1",
                "summary_sha256": _sha(report_path),
                "document_sha256": _sha(document_path),
                **{f"{name}_sha256": _sha(path) for name, path in paths.items()}}
    with (output_dir / "private_recompute_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/private_rollout_400"))
    parser.add_argument("--a1-root", type=Path, default=Path(
        "v6_lite/output/v6_2_a1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.a1_root, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "checked_task_states", "checked_interval_rows", "failure_count",
        "maximum_native_state_error", "maximum_start_slack_error_m_s",
        "pass_recompute",
    )}, ensure_ascii=False, indent=2))
    if not report["pass_recompute"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
