"""Read-only high-budget interval start check on native A.1 torque replay.

Budget 255 was fixed by the separate frontier audit before this run. It is a
diagnostic upper effort, not an online admission configuration or a new
closed-loop controller.
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
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


DIAGNOSTIC_POINT_BUDGET = 255
DEFAULT_FRONTIER = Path("v6_lite/output/v6_2_b2/budget_frontier/budget_frontier.json")
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
    array = np.asarray(values, dtype=np.float64)
    return ({"count": len(values), "p50": float(np.quantile(array, .5)),
             "p95": float(np.quantile(array, .95)),
             "p99": float(np.quantile(array, .99)), "max": float(np.max(array))}
            if values else {"count": 0})


def run(output_dir: Path, *, frontier_path: Path = DEFAULT_FRONTIER,
        a1_root: Path = DEFAULT_A1_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        if (frontier["budgets"] != [31, 63, 127, 255, 511]
                or frontier["gate_m"] != .005):
            raise ValueError("the frozen frontier protocol changed")
        expected_metrics = {item["path"]: item["sha256"]
                            for item in frontier["input_metrics"]}
        expected_traces = {item["path"]: item["sha256"]
                           for item in frontier["input_traces"]}
        grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for entry in frontier["entries"]:
            key = (entry["mode"], entry["scenario_id"])
            tick = int(entry["tick"])
            if tick in grouped[key]:
                raise ValueError(f"duplicate frozen tick: {key}/{tick}")
            grouped[key][tick] = entry
        robot = default_v6_lite_robot_spec()
        records = []
        traces = []
        metrics_sources = []
        replay_checks = []
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            if _sha(metrics_path) != expected_metrics.get(metrics_path.as_posix()):
                raise ValueError(f"A.1 metrics hash mismatch: {mode}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_sources.append({"path": metrics_path.as_posix(), "sha256": _sha(metrics_path)})
            cfg = HierarchicalQPConfig(**metrics["qp_config"])
            if cfg.pcc_clearance_safe_m != .005:
                raise ValueError("PCC safety distance changed")
            for scenario_result in metrics["scenarios"]:
                scenario = scenario_result["scenario"]
                scenario_id = scenario["scenario_id"]
                key = (mode, scenario_id)
                planned = grouped.pop(key, {})
                if len(planned) != 27:
                    raise ValueError(f"frozen sample plan mismatch: {key}")
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
                if trace_sha != expected_traces.get(trace_path.as_posix()):
                    raise ValueError(f"A.1 trace hash mismatch: {key}")
                traces.append({"path": trace_path.as_posix(), "sha256": trace_sha})
                with np.load(trace_path, allow_pickle=False) as trace:
                    initial_qpos = trace["initial_qpos"].copy()
                    initial_qvel = trace["initial_qvel"].copy()
                    torque = trace["torque"].copy()
                    task_qpos = trace["task_qpos"].copy()
                    task_time = trace["task_time"].copy()
                    selected = trace["task_selected_command"].copy()
                if len(torque) != len(selected) * 10 or len(task_time) != len(selected):
                    raise ValueError(f"physics/task tick mismatch: {key}")
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                max_state_error = 0.0
                for step, control in enumerate(torque):
                    if step % 10 == 0:
                        tick = step // 10
                        mujoco.mj_forward(model, data)
                        max_state_error = max(
                            max_state_error,
                            float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                            abs(float(data.time - task_time[tick])),
                        )
                        if tick in planned:
                            prior = planned[tick]
                            if hashlib.sha256(task_qpos[tick].tobytes()).hexdigest() != prior["input_qpos_sha256"]:
                                raise ValueError(f"frozen state hash mismatch: {key}/{tick}")
                            started = time.perf_counter()
                            low_level = data.qpos[evaluator.qpos_ids[:60]]
                            projection = evaluator.shape_spec.project_actual_configuration(low_level)
                            if abs(projection.residual_linf_rad - prior["subspace_residual_linf_rad"]) > 1e-12:
                                raise ValueError(f"subspace residual changed: {key}/{tick}")
                            base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                            box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                            decision = query.evaluate(
                                projection.planner_configuration, base, box,
                                IntervalPartition.uniform(), gate_m=.005,
                                max_point_evaluations=DIAGNOSTIC_POINT_BUDGET,
                                max_leaves=512,
                            )
                            expected = next(x for x in prior["budget_ladder"]
                                            if x["point_budget"] == DIAGNOSTIC_POINT_BUDGET)
                            if (decision.proxy_clearance_status != expected["proxy_status"]
                                    or abs(decision.distance_lower_bound_m - expected["lower_m"]) > 1e-10
                                    or abs(decision.distance_upper_bound_m - expected["upper_m"]) > 1e-10):
                                raise ValueError(
                                    f"frontier classification changed: {key}/{tick}; "
                                    f"saved={expected['proxy_status']},"
                                    f"{expected['lower_m']},{expected['upper_m']}; "
                                    f"replay={decision.proxy_clearance_status},"
                                    f"{decision.distance_lower_bound_m},{decision.distance_upper_bound_m}; "
                                    f"qpos_error={float(np.max(np.abs(data.qpos - task_qpos[tick])))}"
                                )
                            generalized_map = evaluator.reaction_map(data)
                            old_velocity = np.zeros(17) if tick == 0 else selected[tick - 1]
                            planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
                            velocity_lower, velocity_upper, ramp_speed, box_valid = (
                                ramp_velocity_abs_bound(robot, cfg, planner_q, old_velocity)
                            )
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
                            original = original_builder.build(data)
                            matrix, row_lower, drifts, gains, sources = combined_rows(
                                original, batch, selected_ids, cfg,
                            )
                            feasibility = solve_frozen_linear_feasibility(
                                matrix, row_lower, drifts, gains, sources,
                                velocity_lower, velocity_upper, old_velocity, cfg,
                                interval_row_count=len(selected_ids),
                                historical_endpoint=selected[tick],
                            )
                            point_total = decision.point_evaluation_count + batch.point_evaluation_count
                            jac_total = batch.jacobian_evaluation_count
                            records.append({
                                "mode": mode, "scenario_id": scenario_id, "tick": tick,
                                "old_warm_start_status": prior["warm_start_status"],
                                "proxy_status": decision.proxy_clearance_status,
                                "point_evaluations": point_total,
                                "jacobian_evaluations": jac_total,
                                "selected_interval_count": len(selected_ids),
                                "candidate_velocity_box_valid": box_valid,
                                "partition_leaf_count": len(decision.partition.leaves),
                                "frozen_reach_max_m": max(reach.values()),
                                "interval_query_and_assembly_ms": interval_time_ms,
                                "within_existing_128_point_budget": point_total <= 128,
                                "within_existing_96_jacobian_budget": jac_total <= 96,
                                "within_20ms_interval_budget": interval_time_ms <= 20.0,
                                "feasibility": feasibility.to_dict(),
                            })
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                if max_state_error > 1e-8:
                    raise ValueError(f"native torque replay state mismatch: {key}")
                replay_checks.append({"mode": mode, "scenario_id": scenario_id,
                                      "max_state_error": max_state_error,
                                      "trace_sha256": trace_sha})
                print(f"[b2-refined] {mode} {scenario_id}: {len(planned)} states", flush=True)
        if grouped:
            raise ValueError(f"unmatched frozen frontier states: {sorted(grouped)}")
        summary = {}
        for mode in ("baseline", "enabled"):
            own = [x for x in records if x["mode"] == mode]
            old_bad = [x for x in own if x["old_warm_start_status"].startswith("START_")]
            summary[mode] = {
                "sample_count": len(own),
                "old_warm_start_violation_count": len(old_bad),
                "diagnostic_refined_start_violation_count": sum(
                    x["feasibility"]["status"].startswith("START_") for x in own
                ),
                "old_bad_cured_by_diagnostic_refinement": sum(
                    x["feasibility"]["status"] == "FROZEN_ROWS_FEASIBLE" for x in old_bad
                ),
                "old_bad_remains_unexecutable": sum(
                    x["feasibility"]["executable_candidate_exists"] is False for x in old_bad
                ),
                "new_start_violation_after_partition_change": sum(
                    x["feasibility"]["status"].startswith("START_")
                    and not x["old_warm_start_status"].startswith("START_")
                    for x in own
                ),
                "no_feasible_endpoint_count": sum(
                    x["feasibility"]["candidate_feasible"] is False for x in own
                ),
                "unknown_lp_count": sum(
                    x["feasibility"]["candidate_feasible"] is None for x in own
                ),
                "original_point_budget_exceeded": sum(
                    not x["within_existing_128_point_budget"] for x in own
                ),
                "original_jacobian_budget_exceeded": sum(
                    not x["within_existing_96_jacobian_budget"] for x in own
                ),
                "interval_20ms_exceeded": sum(
                    not x["within_20ms_interval_budget"] for x in own
                ),
                "interval_query_and_assembly_ms": _summary([
                    x["interval_query_and_assembly_ms"] for x in own
                ]),
                "partition_leaf_count": _summary([
                    x["partition_leaf_count"] for x in own
                ]),
                "proxy_status": dict(Counter(x["proxy_status"] for x in own)),
            }
        report = {
            "schema": "v6_2_b2_native_replay_refined_start_diagnostic_v1",
            "diagnostic_point_budget": DIAGNOSTIC_POINT_BUDGET,
            "scope": "read-only native torque replay; high-budget cold partition; no online action",
            "online_admission_claim": False,
            "source_sha256": {
                "v6_lite/audit_b2_refined_start.py": _sha(Path(__file__)),
                "v6_lite/b2_shadow_feasibility.py": _sha(Path(__file__).with_name("b2_shadow_feasibility.py")),
                "v6_lite/pcc_persistent_interval_query.py": _sha(Path(__file__).with_name("pcc_persistent_interval_query.py")),
                "v6_lite/shadow_b2_interval_cbf.py": _sha(Path(__file__).with_name("shadow_b2_interval_cbf.py")),
            },
            "input_frontier": {"path": frontier_path.as_posix(), "sha256": _sha(frontier_path)},
            "input_metrics": metrics_sources,
            "input_traces": traces,
            "native_replay_checks": replay_checks,
            "modes": summary,
            "records": records,
        }
        report_path = output_dir / "refined_start_report.json"
        _write(report_path, report)
        lines = [
            "# V6.2-B.2 高预算区间细分对斜坡起点的只读诊断", "",
            "从已发布 A.1 力矩按 500 Hz 原生重放，在正式影子的相同 270 个规划状态"
            "重新构造 255 点预算的完整区间分区，再独立重算 MuJoCo、胶囊及新区间行。"
            "255 点是诊断预算，不是在线准入配置。", "",
            "| 模式 | 旧持久起点违反 | 诊断细分后起点违反 | 旧违例消除 | 新分区出现起点违反 | 旧违例仍不可执行 | 原 128 点预算超限 | 原 96 Jacobian 预算超限 | 区间计算超过 20 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ("baseline", "enabled"):
            item = summary[mode]
            lines.append(
                f"| {mode} | {item['old_warm_start_violation_count']} | "
                f"{item['diagnostic_refined_start_violation_count']} | "
                f"{item['old_bad_cured_by_diagnostic_refinement']} | "
                f"{item['new_start_violation_after_partition_change']} | "
                f"{item['old_bad_remains_unexecutable']} | "
                f"{item['original_point_budget_exceeded']} | "
                f"{item['original_jacobian_budget_exceeded']} | "
                f"{item['interval_20ms_exceeded']} |"
            )
        lines += [
            "", "线性 LP 仅诊断冻结行是否有可行终点；历史斜坡起点不通过时，"
            "终点可行不能批准动作。区间计算耗时不含独立 MuJoCo/胶囊重算或 LP，"
            "更不等于新控制器全链耗时。代理低于门槛不等于实际链碰撞。"
            "逐状态行、预算、起点来源和失败结果见 JSON。", "",
        ]
        doc_path = output_dir / "REFINED_START.md"
        with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        manifest = {"schema": "v6_2_b2_refined_start_manifest_v1",
                    "artifacts": [{"path": p.name, "sha256": _sha(p),
                                   "bytes": p.stat().st_size}
                                  for p in (report_path, doc_path)]}
        _write(output_dir / "refined_start_manifest.json", manifest)
        return report
    except Exception as error:
        _write(output_dir / "FAILURE.json", {
            "schema": "v6_2_b2_refined_start_failure_v1",
            "error_type": type(error).__name__, "error": str(error),
            "input_frontier": frontier_path.as_posix(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    args = parser.parse_args()
    report = run(args.output_dir, frontier_path=args.frontier, a1_root=args.a1_root)
    print(json.dumps(report["modes"], indent=2))


if __name__ == "__main__":
    main()
