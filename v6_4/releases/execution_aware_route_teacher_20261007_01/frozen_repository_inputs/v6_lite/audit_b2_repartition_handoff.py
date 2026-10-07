"""Read-only ramp and row checks after counterfactual B.2 repartition.

Only the frozen warm-unknown sample ticks are checked. A.1 torque is replayed
at 500 Hz to obtain current and next task states; the old executed commands
are never replaced, and the cold partition is never admitted online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
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
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


DEFAULT_REPARTITION = Path(
    "v6_lite/output/v6_2_b2/repartition_counterfactual/repartition_counterfactual.json"
)
DEFAULT_A1_ROOT = Path("v6_lite/output/v6_2_a1")
POINT_BUDGET = 64
TOTAL_POINT_BUDGET = 128
JACOBIAN_BUDGET = 96
INTERVAL_TIME_BUDGET_MS = 20.0


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
    array = np.asarray(values, dtype=np.float64)
    return ({"count": len(values), "p50": float(np.quantile(array, .5)),
             "p95": float(np.quantile(array, .95)),
             "p99": float(np.quantile(array, .99)), "max": float(np.max(array))}
            if values else {"count": 0})


def run(output_dir: Path, *, repartition_path: Path = DEFAULT_REPARTITION,
        a1_root: Path = DEFAULT_A1_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = json.loads(repartition_path.read_text(encoding="utf-8"))
        if (source["point_budget"] != POINT_BUDGET or source["gate_m"] != .005
                or source["repartition_applied_to_execution"]):
            raise ValueError("repartition counterfactual protocol changed")
        expected_metrics = {item["path"]: item["sha256"]
                            for item in source["input_metrics"]}
        expected_traces = {item["path"]: item["sha256"]
                           for item in source["input_traces"]}
        grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for entry in source["entries"]:
            tick = int(entry["tick"])
            if tick % 50 != 0:
                continue
            key = (entry["mode"], entry["scenario_id"])
            if tick in grouped[key]:
                raise ValueError(f"duplicate unknown sample: {key}/{tick}")
            grouped[key][tick] = entry
        robot = default_v6_lite_robot_spec()
        records = []
        replay_checks = []
        metrics_sources = []
        trace_sources = []
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics_sha = _sha(metrics_path)
            if metrics_sha != expected_metrics.get(metrics_path.as_posix()):
                raise ValueError(f"A.1 metrics hash mismatch: {mode}")
            metrics_sources.append({"path": metrics_path.as_posix(),
                                    "sha256": metrics_sha})
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg = HierarchicalQPConfig(**metrics["qp_config"])
            if cfg.pcc_clearance_safe_m != .005:
                raise ValueError("PCC safe distance changed")
            for scenario_result in metrics["scenarios"]:
                scenario = scenario_result["scenario"]
                scenario_id = scenario["scenario_id"]
                key = (mode, scenario_id)
                planned = grouped.pop(key, {})
                verifier = WholeBodyCollisionVerifier(
                    robot, _obstacles(scenario),
                    WholeBodyVerificationConfig(
                        minimum_clearance=metrics["run_config"]["whole_body_minimum_clearance_m"],
                        query_distance_max=2.5,
                        adaptive_subdivisions=metrics["run_config"]["verification_subdivisions"],
                        self_collision_ancestor_exclusion_depth=3,
                        include_target_satellite_pairs=True,
                    ),
                )
                model = verifier.model
                model.geom_contype[:] = 0
                model.geom_conaffinity[:] = 0
                data = mujoco.MjData(model)
                evaluator = FixedIntervalCBFEvaluator(robot, model)
                query = PersistentIntervalDecisionQuery(evaluator.shape_model)
                original_builder = ReplayConstraintBuilder(
                    robot, model, verifier.pairs,
                    replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True),
                )
                trace_path = Path(scenario_result["trace"]["path"])
                trace_sha = _sha(trace_path)
                if (trace_sha != scenario_result["trace"]["sha256"]
                        or trace_sha != expected_traces.get(trace_path.as_posix())):
                    raise ValueError(f"A.1 trace hash mismatch: {mode}/{scenario_id}")
                trace_sources.append({"path": trace_path.as_posix(),
                                      "sha256": trace_sha})
                with np.load(trace_path, allow_pickle=False) as trace:
                    initial_qpos = trace["initial_qpos"].copy()
                    initial_qvel = trace["initial_qvel"].copy()
                    torque = trace["torque"].copy()
                    task_qpos = trace["task_qpos"].copy()
                    task_time = trace["task_time"].copy()
                    selected = trace["task_selected_command"].copy()
                if len(torque) != len(selected) * 10:
                    raise ValueError(f"A.1 task/physics count mismatch: {mode}/{scenario_id}")
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                max_state_error = 0.0
                pending = None
                for step, control in enumerate(torque):
                    if step % 10 == 0:
                        tick = step // 10
                        mujoco.mj_forward(model, data)
                        max_state_error = max(
                            max_state_error,
                            float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                            abs(float(data.time - task_time[tick])),
                        )
                        if pending is not None and tick == pending["tick"] + 1:
                            next_batch = evaluator.evaluate_state(
                                data, pending["partition"],
                                derivative_interval_ids=pending["selected_ids"],
                            )
                            next_by_id = {row.interval_id: row for row in next_batch.rows}
                            pending["record"]["excluded_reached_activation_next_tick"] = sum(
                                row.distance_lower_bound_m <= cfg.pcc_clearance_activation_m
                                for row in next_batch.rows
                                if row.interval_id not in pending["selected_ids"]
                            )
                            h_errors = []
                            residual_errors = []
                            for interval_id, h_hat in pending["predicted_h"].items():
                                actual = next_by_id[interval_id]
                                h_errors.append(abs(actual.h_m - h_hat))
                                if actual.derivative_status == "SUPPORTED":
                                    actual_residual = float(
                                        actual.generalized_gradient @ pending["endpoint"]
                                        + actual.target_drift_m_s
                                        + cfg.pcc_clearance_barrier_gain * actual.h_m
                                    )
                                    residual_errors.append(abs(
                                        actual_residual
                                        - pending["predicted_residual"][interval_id]
                                    ))
                            pending["record"]["next_h_error_max_m"] = (
                                max(h_errors) if h_errors else None
                            )
                            pending["record"]["next_start_residual_error_max_m_s"] = (
                                max(residual_errors) if residual_errors else None
                            )
                            pending = None
                        if tick in planned:
                            prior = planned.pop(tick)
                            if hashlib.sha256(task_qpos[tick].tobytes()).hexdigest() != prior[
                                "input_qpos_sha256"
                            ]:
                                raise ValueError(f"frozen qpos hash mismatch: {mode}/{scenario_id}/{tick}")
                            started = time.perf_counter()
                            low_level = data.qpos[evaluator.qpos_ids[:60]]
                            projection = evaluator.shape_spec.project_actual_configuration(low_level)
                            base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                            box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                            decision = query.evaluate(
                                projection.planner_configuration, base, box,
                                IntervalPartition.uniform(), gate_m=.005,
                                max_point_evaluations=POINT_BUDGET,
                            )
                            if (decision.proxy_clearance_status != prior["cold_status"]
                                    or abs(decision.distance_lower_bound_m - prior["cold_lower_m"]) > 1e-10
                                    or abs(decision.distance_upper_bound_m - prior["cold_upper_m"]) > 1e-10):
                                raise ValueError(f"same-state repartition changed: {mode}/{scenario_id}/{tick}")
                            previous = np.zeros(17) if tick == 0 else selected[tick - 1]
                            planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
                            velocity_lower, velocity_upper, ramp_speed, box_valid = (
                                ramp_velocity_abs_bound(robot, cfg, planner_q, previous)
                            )
                            generalized_map = evaluator.reaction_map(data)
                            selected_ids, reach = select_intervals_by_frozen_reach(
                                decision.lower_by_interval_id, decision.partition,
                                evaluator, data, cfg, generalized_map,
                                velocity_abs_bound=ramp_speed,
                            )
                            batch = evaluator.evaluate_state(
                                data, decision.partition, generalized_map=generalized_map,
                                derivative_interval_ids=selected_ids,
                            )
                            interval_time_ms = (time.perf_counter() - started) * 1000.0
                            total_points = (decision.point_evaluation_count
                                            + batch.point_evaluation_count)
                            original = original_builder.build(data)
                            try:
                                matrix, lower, drifts, gains, sources = combined_rows(
                                    original, batch, selected_ids, cfg,
                                )
                                feasibility = solve_frozen_linear_feasibility(
                                    matrix, lower, drifts, gains, sources,
                                    velocity_lower, velocity_upper, previous, cfg,
                                    interval_row_count=len(selected_ids),
                                    historical_endpoint=selected[tick],
                                ).to_dict()
                            except ValueError as error:
                                feasibility = {"status": "UNKNOWN_REQUIRED_ROW",
                                               "reason": str(error),
                                               "executable_candidate_exists": None}
                            record = {
                                "mode": mode, "scenario_id": scenario_id, "tick": tick,
                                "warm_status": prior["warm_status"],
                                "cold_status": decision.proxy_clearance_status,
                                "input_qpos_sha256": prior["input_qpos_sha256"],
                                "partition_leaf_count": len(decision.partition.leaves),
                                "selected_interval_count": len(selected_ids),
                                "candidate_velocity_box_valid": box_valid,
                                "coverage_complete": batch.coverage_complete,
                                "static_all_intervals_safe": batch.all_intervals_safe,
                                "point_evaluations": total_points,
                                "jacobian_evaluations": batch.jacobian_evaluation_count,
                                "interval_query_and_assembly_ms": interval_time_ms,
                                "within_128_point_budget": total_points <= TOTAL_POINT_BUDGET,
                                "within_96_jacobian_budget": (
                                    batch.jacobian_evaluation_count <= JACOBIAN_BUDGET
                                ),
                                "within_20ms_interval_budget": (
                                    interval_time_ms <= INTERVAL_TIME_BUDGET_MS
                                ),
                                "frozen_reach_max_m": max(reach.values()),
                                "feasibility": feasibility,
                                "excluded_reached_activation_next_tick": None,
                                "next_h_error_max_m": None,
                                "next_start_residual_error_max_m_s": None,
                            }
                            if (record["cold_status"] == "PROXY_CLEARANCE_AT_LEAST_GATE"
                                    and not record["static_all_intervals_safe"]):
                                raise ValueError(f"query and fixed rows disagree on safe partition: {mode}/{scenario_id}/{tick}")
                            records.append(record)
                            if tick + 1 < len(selected):
                                old_weight, new_weight = ramp_mean_weights()
                                mean_velocity = old_weight * previous + new_weight * selected[tick]
                                predicted_h = {}
                                predicted_residual = {}
                                for row in batch.rows:
                                    if (row.interval_id not in selected_ids
                                            or row.derivative_status != "SUPPORTED"):
                                        continue
                                    h_hat = float(row.h_m + cfg.task_period_s * (
                                        row.generalized_gradient @ mean_velocity
                                        + row.target_drift_m_s
                                    ))
                                    predicted_h[row.interval_id] = h_hat
                                    predicted_residual[row.interval_id] = float(
                                        row.generalized_gradient @ selected[tick]
                                        + row.target_drift_m_s
                                        + cfg.pcc_clearance_barrier_gain * h_hat
                                    )
                                pending = {
                                    "tick": tick, "partition": decision.partition,
                                    "selected_ids": selected_ids,
                                    "predicted_h": predicted_h,
                                    "predicted_residual": predicted_residual,
                                    "endpoint": selected[tick], "record": record,
                                }
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                if planned:
                    raise ValueError(f"unknown sample ticks not replayed: {mode}/{scenario_id}")
                if max_state_error > 1e-8:
                    raise ValueError(f"native torque replay state mismatch: {mode}/{scenario_id}")
                replay_checks.append({"mode": mode, "scenario_id": scenario_id,
                                      "max_state_error": max_state_error,
                                      "trace_sha256": trace_sha})
                print(f"[b2-handoff] {mode} {scenario_id}: replay complete", flush=True)
        if grouped:
            raise ValueError(f"unknown scenario IDs in counterfactual: {sorted(grouped)}")
        mode_summary = {}
        for mode in ("baseline", "enabled"):
            own = [record for record in records if record["mode"] == mode]
            safe = [record for record in own
                    if record["cold_status"] == "PROXY_CLEARANCE_AT_LEAST_GATE"]
            mode_summary[mode] = {
                "sample_count": len(own),
                "cold_status": dict(Counter(x["cold_status"] for x in own)),
                "safe_repartition_start_status": dict(Counter(
                    x["feasibility"]["status"] for x in safe
                )),
                "safe_repartition_executable_candidate_count": sum(
                    x["feasibility"].get("executable_candidate_exists") is True
                    for x in safe
                ),
                "safe_repartition_not_executable_count": sum(
                    x["feasibility"].get("executable_candidate_exists") is False
                    for x in safe
                ),
                "safe_repartition_unknown_feasibility_count": sum(
                    x["feasibility"].get("executable_candidate_exists") is None
                    for x in safe
                ),
                "point_budget_exceeded": sum(not x["within_128_point_budget"] for x in own),
                "jacobian_budget_exceeded": sum(not x["within_96_jacobian_budget"] for x in own),
                "interval_20ms_exceeded": sum(not x["within_20ms_interval_budget"] for x in own),
                "excluded_reached_activation_next_tick": sum(
                    x["excluded_reached_activation_next_tick"] or 0 for x in own
                ),
                "query_and_assembly_ms": _summary([
                    x["interval_query_and_assembly_ms"] for x in own
                ]),
                "next_h_error_max_m": _summary([
                    x["next_h_error_max_m"] for x in own
                    if x["next_h_error_max_m"] is not None
                ]),
                "next_start_residual_error_max_m_s": _summary([
                    x["next_start_residual_error_max_m_s"] for x in own
                    if x["next_start_residual_error_max_m_s"] is not None
                ]),
            }
        report = {
            "schema": "v6_2_b2_native_replay_repartition_handoff_v1",
            "scope": "read-only native A.1 torque replay at warm-unknown sampled ticks",
            "online_control_changed": False,
            "repartition_admitted_online": False,
            "budgets": {"point_query": POINT_BUDGET,
                        "total_points": TOTAL_POINT_BUDGET,
                        "jacobians": JACOBIAN_BUDGET,
                        "interval_ms": INTERVAL_TIME_BUDGET_MS},
            "source_sha256": {
                "v6_lite/audit_b2_repartition_handoff.py": _sha(Path(__file__)),
                "v6_lite/pcc_interval_cbf.py": _sha(Path(__file__).with_name("pcc_interval_cbf.py")),
                "v6_lite/pcc_persistent_interval_query.py": _sha(Path(__file__).with_name("pcc_persistent_interval_query.py")),
                "v6_lite/b2_shadow_feasibility.py": _sha(Path(__file__).with_name("b2_shadow_feasibility.py")),
                "v6_lite/recompute_execution_constraints.py": _sha(Path(__file__).with_name("recompute_execution_constraints.py")),
            },
            "input_repartition": {"path": repartition_path.as_posix(),
                                  "sha256": _sha(repartition_path)},
            "input_metrics": metrics_sources,
            "input_traces": trace_sources,
            "native_replay_checks": replay_checks,
            "modes": mode_summary,
            "records": records,
        }
        report_path = output_dir / "repartition_handoff.json"
        _write(report_path, report)
        lines = [
            "# V6.2-B.2 重分区后的斜坡起点与约束只读审计", "",
            "本审计仅对正式持久影子未知的抽样 tick 进行 500 Hz A.1 力矩原生重放，"
            "从五段根区间以 64 点预算重建区间行，独立重算 MuJoCo 与实际链胶囊行，"
            "并检查旧斜坡起点、终点线性可行性及下一 tick 的固定分区预测。"
            "它不是新区间模式闭环，也没有执行任何新候选命令。", "",
            "| 模式 | 抽样未知 | 根分区确定安全 | 安全状态起点及终点均可行 | 安全状态不可执行 | 安全状态 LP/导数未知 | 点预算超限 | Jacobian 超限 | 区间计算超 20 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ("baseline", "enabled"):
            item = mode_summary[mode]
            lines.append(
                f"| {mode} | {item['sample_count']} | "
                f"{item['cold_status'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
                f"{item['safe_repartition_executable_candidate_count']} | "
                f"{item['safe_repartition_not_executable_count']} | "
                f"{item['safe_repartition_unknown_feasibility_count']} | "
                f"{item['point_budget_exceeded']} | "
                f"{item['jacobian_budget_exceeded']} | "
                f"{item['interval_20ms_exceeded']} |"
            )
        lines += [
            "", "这些 LP 是离线冻结行存在性诊断，未求解第二个在线 QP。"
            "重分区使安全函数集合发生变化，必须以新分区重新检查斜坡起点；"
            "代理净空判安全不等于动作获准。计算耗时不含独立 MuJoCo／胶囊装配及 LP，"
            "不能作为新模式全链时延。逐状态失败与预测误差见 JSON。", "",
        ]
        doc_path = output_dir / "REPARTITION_HANDOFF.md"
        with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        _write(output_dir / "repartition_handoff_manifest.json", {
            "schema": "v6_2_b2_repartition_handoff_manifest_v1",
            "artifacts": [{"path": path.name, "sha256": _sha(path),
                           "bytes": path.stat().st_size}
                          for path in (report_path, doc_path)],
        })
        return report
    except Exception as error:
        _write(output_dir / "FAILURE.json", {
            "schema": "v6_2_b2_repartition_handoff_failure_v1",
            "error_type": type(error).__name__, "error": str(error),
            "input_repartition": repartition_path.as_posix(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repartition", type=Path, default=DEFAULT_REPARTITION)
    parser.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    args = parser.parse_args()
    report = run(args.output_dir, repartition_path=args.repartition,
                 a1_root=args.a1_root)
    print(json.dumps(report["modes"], indent=2))


if __name__ == "__main__":
    main()
