"""Read-only same-state repartition counterfactual for B.2 persistent unknowns.

This reads saved A.1 task positions. The formal shadow separately establishes
that native 500 Hz torque replay reaches those positions; this audit does not
replay torque and never changes an executed action or an online partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


DEFAULT_SHADOW = Path("v6_lite/output/v6_2_b2/shadow_warm/shadow_report.json")
DEFAULT_A1_ROOT = Path("v6_lite/output/v6_2_a1")
POINT_BUDGET = 64
GATE_M = 0.005


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


def _timing(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return ({"count": len(values), "p50": float(np.quantile(array, .5)),
             "p95": float(np.quantile(array, .95)),
             "p99": float(np.quantile(array, .99)), "max": float(np.max(array))}
            if values else {"count": 0})


def run(output_dir: Path, *, shadow_path: Path = DEFAULT_SHADOW,
        a1_root: Path = DEFAULT_A1_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        if (shadow["query_mode"] != "persistent_warm"
                or not shadow["passed_as_read_only_audit"]):
            raise ValueError("a valid persistent native-replay shadow is required")
        robot = default_v6_lite_robot_spec()
        entries: list[dict] = []
        metrics_sources: list[dict] = []
        trace_sources: list[dict] = []
        modes = {}
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics_sha = _sha(metrics_path)
            if metrics_sha != shadow["modes"][mode]["metrics_sha256"]:
                raise ValueError(f"A.1 metrics hash mismatch: {mode}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics["qp_config"]["pcc_clearance_safe_m"] != GATE_M:
                raise ValueError(f"PCC safety gate changed: {mode}")
            metrics_sources.append({"path": metrics_path.as_posix(),
                                    "sha256": metrics_sha})
            counts: Counter[str] = Counter()
            cold_times = []
            max_warm_leaves = 0
            sampled = {}
            for record in shadow["modes"][mode]["frozen_feasibility_records"]:
                key = (record["scenario_id"], int(record["tick"]))
                if key in sampled:
                    raise ValueError(f"duplicate frozen shadow sample: {mode}/{key}")
                sampled[key] = record["proxy_status"]
            for scenario_result in metrics["scenarios"]:
                scenario = scenario_result["scenario"]
                scenario_id = scenario["scenario_id"]
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
                partition = IntervalPartition.uniform()
                trace_path = Path(scenario_result["trace"]["path"])
                trace_sha = _sha(trace_path)
                formal_scene = next(
                    item for item in shadow["modes"][mode]["scenarios"]
                    if item["scenario_id"] == scenario_id
                )
                if (trace_sha != scenario_result["trace"]["sha256"]
                        or trace_sha != formal_scene["trace_sha256"]):
                    raise ValueError(f"A.1 trace hash mismatch: {mode}/{scenario_id}")
                trace_sources.append({"path": trace_path.as_posix(), "sha256": trace_sha})
                with np.load(trace_path, allow_pickle=False) as trace:
                    task_qpos = trace["task_qpos"].copy()
                    selected = trace["task_selected_command"].copy()
                    torque_ticks = len(trace["torque"])
                if (torque_ticks != len(selected) * 10
                        or len(task_qpos) not in (len(selected), len(selected) + 1)):
                    raise ValueError(f"A.1 executed task tick count mismatch: {mode}/{scenario_id}")
                for tick in range(len(selected)):
                    qpos = task_qpos[tick]
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    low_level = data.qpos[evaluator.qpos_ids[:60]]
                    projection = evaluator.shape_spec.project_actual_configuration(low_level)
                    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                    box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                    warm = query.evaluate(
                        projection.planner_configuration, base, box, partition,
                        gate_m=GATE_M, max_point_evaluations=POINT_BUDGET,
                    )
                    if (not warm.bounds_valid
                            or not warm.partition.coverage(
                                evaluator.shape_spec.segment_lengths_m
                            ).coverage_complete):
                        raise ValueError(f"invalid warm partition: {mode}/{scenario_id}/{tick}")
                    partition = warm.partition
                    max_warm_leaves = max(max_warm_leaves, warm.interval_count)
                    counts["task_ticks"] += 1
                    counts[f"warm_{warm.proxy_clearance_status}"] += 1
                    counts["warm_budget_exhausted"] += int(warm.budget_exhausted)
                    sample_key = (scenario_id, tick)
                    if sample_key in sampled:
                        if warm.proxy_clearance_status != sampled.pop(sample_key):
                            raise ValueError(f"saved-state sample disagrees with native replay: {mode}/{sample_key}")
                        counts["sampled_status_matches"] += 1
                    if warm.proxy_clearance_status != "UNKNOWN_CROSSES_GATE":
                        continue
                    cold = query.evaluate(
                        projection.planner_configuration, base, box,
                        IntervalPartition.uniform(), gate_m=GATE_M,
                        max_point_evaluations=POINT_BUDGET,
                    )
                    if (not cold.bounds_valid
                            or not cold.partition.coverage(
                                evaluator.shape_spec.segment_lengths_m
                            ).coverage_complete):
                        raise ValueError(f"invalid cold partition: {mode}/{scenario_id}/{tick}")
                    counts[f"cold_{cold.proxy_clearance_status}"] += 1
                    cold_times.append(cold.elapsed_ms)
                    entries.append({
                        "mode": mode, "scenario_id": scenario_id, "tick": tick,
                        "input_qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                        "warm_status": warm.proxy_clearance_status,
                        "warm_lower_m": warm.distance_lower_bound_m,
                        "warm_upper_m": warm.distance_upper_bound_m,
                        "warm_leaves": warm.interval_count,
                        "warm_point_evaluations": warm.point_evaluation_count,
                        "cold_status": cold.proxy_clearance_status,
                        "cold_lower_m": cold.distance_lower_bound_m,
                        "cold_upper_m": cold.distance_upper_bound_m,
                        "cold_leaves": cold.interval_count,
                        "cold_point_evaluations": cold.point_evaluation_count,
                        "cold_elapsed_ms": cold.elapsed_ms,
                    })
                print(f"[b2-repartition] {mode} {scenario_id}: {len(selected)} executed task ticks",
                      flush=True)
            if sampled:
                raise ValueError(f"unmatched frozen shadow samples: {mode}/{len(sampled)}")
            expected = shadow["modes"][mode]["counts"]
            if (counts["warm_UNKNOWN_CROSSES_GATE"] != expected.get("warm_all_task_unknown", 0)
                    or counts["warm_budget_exhausted"]
                    != expected.get("warm_all_task_budget_exhausted", 0)
                    or max_warm_leaves != shadow["modes"][mode]["summaries"][
                        "warm_all_task_partition_size"]["max"]):
                raise ValueError(f"saved-state warm query disagrees with native replay: {mode}")
            modes[mode] = {
                "counts": dict(counts),
                "max_warm_leaves": max_warm_leaves,
                "counterfactual_cold_query_time_ms": _timing(cold_times),
            }
        report = {
            "schema": "v6_2_b2_saved_state_repartition_counterfactual_v1",
            "point_budget": POINT_BUDGET,
            "gate_m": GATE_M,
            "scope": "saved task_qpos inspection; native torque replay belongs to the bound shadow",
            "online_control_changed": False,
            "repartition_applied_to_execution": False,
            "source_sha256": {
                "v6_lite/audit_b2_repartition_counterfactual.py": _sha(Path(__file__)),
                "v6_lite/pcc_persistent_interval_query.py": _sha(
                    Path(__file__).with_name("pcc_persistent_interval_query.py")),
                "v6_lite/pcc_interval_cbf.py": _sha(
                    Path(__file__).with_name("pcc_interval_cbf.py")),
                "v6_lite/continuum_shape_model.py": _sha(
                    Path(__file__).with_name("continuum_shape_model.py")),
                "v6_lite/shape_clearance.py": _sha(
                    Path(__file__).with_name("shape_clearance.py")),
                "v6_lite/recompute_execution_constraints.py": _sha(
                    Path(__file__).with_name("recompute_execution_constraints.py")),
            },
            "input_shadow": {"path": shadow_path.as_posix(), "sha256": _sha(shadow_path)},
            "input_metrics": metrics_sources,
            "input_traces": trace_sources,
            "modes": modes,
            "entries": entries,
        }
        report_path = output_dir / "repartition_counterfactual.json"
        _write(report_path, report)
        lines = [
            "# V6.2-B.2 持久分区未知状态的同状态重分区对照", "",
            "输入是已发布 A.1 trace 的保存规划状态；正式影子报告另以 500 Hz 力矩原生重放"
            "并核对了这些状态。本诊断只在持久查询未知时，用相同 64 点预算从五段根区间"
            "重新查询同一个状态；没有将重分区用于动作或在线分区。", "",
            "| 模式 | 全部 tick | 持久未知 | 从根分区后确定安全 | 确定低于门槛 | 仍未知 | 最大持久叶数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ("baseline", "enabled"):
            item = modes[mode]
            c = item["counts"]
            lines.append(
                f"| {mode} | {c.get('task_ticks', 0)} | "
                f"{c.get('warm_UNKNOWN_CROSSES_GATE', 0)} | "
                f"{c.get('cold_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
                f"{c.get('cold_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
                f"{c.get('cold_UNKNOWN_CROSSES_GATE', 0)} | "
                f"{item['max_warm_leaves']} |"
            )
        lines += [
            "", "从根分区重新查询可能减少历史叶区间造成的未知，但它会改变安全函数集合。"
            "正式在线接入仍需验证分区切换后的斜坡起点、全部必要约束和计算预算；"
            "此对照不构成可直接采用的合并策略，也不改变现有接入门禁。", "",
        ]
        doc_path = output_dir / "REPARTITION_COUNTERFACTUAL.md"
        with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        _write(output_dir / "repartition_manifest.json", {
            "schema": "v6_2_b2_repartition_counterfactual_manifest_v1",
            "artifacts": [{"path": path.name, "sha256": _sha(path),
                           "bytes": path.stat().st_size}
                          for path in (report_path, doc_path)],
        })
        return report
    except Exception as error:
        _write(output_dir / "FAILURE.json", {
            "schema": "v6_2_b2_repartition_counterfactual_failure_v1",
            "error_type": type(error).__name__, "error": str(error),
            "input_shadow": shadow_path.as_posix(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    args = parser.parse_args()
    report = run(args.output_dir, shadow_path=args.shadow, a1_root=args.a1_root)
    print(json.dumps({mode: report["modes"][mode] for mode in ("baseline", "enabled")},
                     indent=2))


if __name__ == "__main__":
    main()
