"""Cold-query parity and timing trial on frozen A.1 task states.

Each of ten scenes contributes tick 50, its first sampled historical CBF
start violation, and tick 1000. The two point backends receive identical
root partitions and 63/127 point budgets. This is read-only, excludes MuJoCo
forward, Jacobian assembly and QP, and never changes an executed command.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_batched_point_model import BatchedPointContinuumShapeModel
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "min": float(np.min(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array))}


def _first_violations(refined: dict) -> dict[tuple[str, str], int]:
    result = {}
    for item in refined["records"]:
        if item["feasibility"]["status"] == "START_CLEARANCE_VIOLATION":
            key = item["mode"], item["scenario_id"]
            result[key] = min(result.get(key, item["tick"]), item["tick"])
    if len(result) != 10:
        raise ValueError("frozen first-violation tick set changed")
    return result


def _parity(reference, alternative) -> dict:
    names = set(reference.lower_by_interval_id)
    if (reference.partition != alternative.partition
            or reference.proxy_clearance_status != alternative.proxy_clearance_status
            or reference.point_evaluation_count != alternative.point_evaluation_count
            or reference.split_count != alternative.split_count
            or names != set(alternative.lower_by_interval_id)):
        raise ValueError("query decision, partition or work count changed")
    lower_error = max((abs(reference.lower_by_interval_id[key]
                           - alternative.lower_by_interval_id[key])
                       for key in names), default=0.0)
    upper_error = max((abs(reference.midpoint_upper_by_interval_id[key]
                           - alternative.midpoint_upper_by_interval_id[key])
                       for key in names), default=0.0)
    endpoint_error = max(
        abs(reference.distance_lower_bound_m - alternative.distance_lower_bound_m),
        abs(reference.distance_upper_bound_m - alternative.distance_upper_bound_m),
    )
    if max(lower_error, upper_error, endpoint_error) > 1e-12:
        raise ValueError("query distance parity exceeds 1e-12 m")
    return {"lower_max_abs_error_m": lower_error,
            "midpoint_upper_max_abs_error_m": upper_error,
            "global_endpoint_max_abs_error_m": endpoint_error}


def run(output_dir: Path, a1_root: Path, refined_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    first = _first_violations(refined)
    robot = default_v6_lite_robot_spec()
    report = {
        "schema": "v6_2_b2_batched_point_trial_v1",
        "scope": "30_frozen_A1_task_states_cold_root_queries_only",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {
            name: _source_sha(Path("v6_lite") / name) for name in (
                "audit_b2_batched_query.py", "pcc_batched_point_model.py",
                "pcc_persistent_interval_query.py", "continuum_shape_model.py",
                "pcc_interval_cbf.py", "shape_clearance.py",
            )
        },
        "input_refined_start_sha256": _sha(refined_path),
        "selected_ticks": ["early_probe_50", "first_sampled_old_start_violation",
                           "late_1000"],
        "point_budgets": [63, 127],
        "timing_order": ["reference", "batched", "batched", "reference"],
        "timing_scope": "query_evaluate_only_no_mj_forward_no_jacobian_no_qp",
        "online_control_changed": False,
        "new_interval_mode_executed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "inputs": {}, "records": [], "modes": {},
    }
    for mode in ("baseline", "enabled"):
        metrics_path = a1_root / f"{mode}_root/output/v6_lite_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report["inputs"][mode] = {
            "metrics_path": metrics_path.as_posix(),
            "metrics_sha256": _sha(metrics_path), "traces": [],
        }
        if len(metrics["scenarios"]) != 5:
            raise ValueError("A.1 must contain five scenes per mode")
        for scene in metrics["scenarios"]:
            scenario_id = scene["scenario"]["scenario_id"]
            ticks = (("early_probe_50", 50),
                     ("first_sampled_old_start_violation", first[mode, scenario_id]),
                     ("late_1000", 1000))
            verifier = WholeBodyCollisionVerifier(
                robot, _obstacles(scene["scenario"]), WholeBodyVerificationConfig(
                    minimum_clearance=metrics["run_config"]["whole_body_minimum_clearance_m"],
                    query_distance_max=2.5,
                    adaptive_subdivisions=metrics["run_config"]["verification_subdivisions"],
                    self_collision_ancestor_exclusion_depth=3,
                    include_target_satellite_pairs=True,
                ),
            )
            model = verifier.model
            data = mujoco.MjData(model)
            evaluator = FixedIntervalCBFEvaluator(robot, model)
            reference = PersistentIntervalDecisionQuery(evaluator.shape_model)
            batched = PersistentIntervalDecisionQuery(
                BatchedPointContinuumShapeModel(evaluator.shape_spec))
            trace_path = Path(scene["trace"]["path"])
            trace_sha = _sha(trace_path)
            if trace_sha != scene["trace"]["sha256"]:
                raise ValueError("A.1 trace hash changed")
            report["inputs"][mode]["traces"].append({
                "scenario_id": scenario_id, "path": trace_path.as_posix(),
                "sha256": trace_sha,
            })
            with np.load(trace_path, allow_pickle=False) as trace:
                task_qpos = trace["task_qpos"].copy()
            for label, tick in ticks:
                qpos = task_qpos[tick]
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                projection = evaluator.shape_spec.project_actual_configuration(
                    data.qpos[evaluator.qpos_ids[:60]])
                base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                for budget in (63, 127):
                    timings = defaultdict(list)
                    results = {}
                    for implementation in report["timing_order"]:
                        query = reference if implementation == "reference" else batched
                        result = query.evaluate(
                            projection.planner_configuration, base, box,
                            IntervalPartition.uniform(), max_point_evaluations=budget,
                        )
                        timings[implementation].append(result.elapsed_ms)
                        results[implementation] = result
                    parity = _parity(results["reference"], results["batched"])
                    record = {
                        "mode": mode, "scenario_id": scenario_id,
                        "tick_label": label, "tick": tick,
                        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                        "point_budget": budget,
                        "status": results["reference"].proxy_clearance_status,
                        "point_evaluations": results["reference"].point_evaluation_count,
                        "leaf_count": results["reference"].interval_count,
                        "distance_lower_bound_m": results["reference"].distance_lower_bound_m,
                        "distance_upper_bound_m": results["reference"].distance_upper_bound_m,
                        "reference_query_ms": timings["reference"],
                        "batched_query_ms": timings["batched"],
                        "reference_median_ms": float(np.median(timings["reference"])),
                        "batched_median_ms": float(np.median(timings["batched"])),
                        "parity": parity,
                    }
                    report["records"].append(record)
            print(f"[b2-batch-trial] {mode} {scenario_id}: 3 states x 2 budgets",
                  flush=True)
    for mode in ("baseline", "enabled"):
        own = [item for item in report["records"] if item["mode"] == mode]
        report["modes"][mode] = {}
        for budget in (63, 127):
            rows = [item for item in own if item["point_budget"] == budget]
            report["modes"][mode][str(budget)] = {
                "state_count": len(rows),
                "status_counts": {name: sum(item["status"] == name for item in rows)
                                  for name in sorted({item["status"] for item in rows})},
                "reference_median_query_ms": _summary(
                    [item["reference_median_ms"] for item in rows]),
                "batched_median_query_ms": _summary(
                    [item["batched_median_ms"] for item in rows]),
                "paired_speedup": _summary([
                    item["reference_median_ms"] / item["batched_median_ms"]
                    for item in rows]),
                "max_parity_error_m": max(
                    max(item["parity"].values()) for item in rows),
            }
    summary_path = output_dir / "batched_query_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 只读批量 PCC 点查询试验", "",
        "每组五场景各预定 tick 50、首次抽样旧 CBF 起点违例 tick 与 tick 1000。"
        "同一 MuJoCo 保存状态、同一根区间和 63/127 点预算下比较原位置计算与批量位置计算。"
        "查询顺序 ABBA；每状态取两次的中位数。只计查询，不计 MuJoCo 正运动学、"
        "完整梯度、约束装配、QP 或力矩执行。", "",
        "| 模式 | 点预算 | 状态 | 原查询 p95 ms | 批量查询 p95 ms | 配对加速比 p50 | 最大下界/上界偏差 m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for budget in (63, 127):
            row = report["modes"][mode][str(budget)]
            lines.append(
                f"| {mode} | {budget} | {row['state_count']} | "
                f"{row['reference_median_query_ms']['p95']:.3f} | "
                f"{row['batched_median_query_ms']['p95']:.3f} | "
                f"{row['paired_speedup']['p50']:.3f} | "
                f"{row['max_parity_error_m']:.2e} |"
            )
    lines += [
        "", "所有状态的分区、决策、点数和区间 ID 必须精确一致；"
        "距离偏差须不大于 1e-12 m。"
        "这只是冻结状态的实现一致性与耗时试验，不能证明浮点全域认证。"
        "现有影子证据和在线控制仍使用原实现；加速结果不得拼接成 20 ms 全链验收。", "",
    ]
    doc_path = output_dir / "BATCHED_QUERY_TRIAL.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_batched_point_trial_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "document_sha256": _sha(doc_path),
    }
    with (output_dir / "batched_query_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--refined-start", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/refined_start/refined_start_report.json"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.refined_start)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
