"""Read-only prepared-prefix query trial on warm-unknown A.1 states.

The protocol takes every sampled warm UNKNOWN_CROSSES_GATE state (58) and
tick 50 of each scene (10). Each is cold-queried from the five roots under
63/127 point budgets with ABBA timings. No old or new command is executed.
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
from v6_lite.audit_b2_batched_query import _parity, _sha, _source_sha, _summary
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


def run(output_dir: Path, a1_root: Path, warm_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    robot = default_v6_lite_robot_spec()
    source_names = (
        "audit_b2_prepared_query.py", "audit_b2_batched_query.py",
        "pcc_batched_distance_query.py", "pcc_persistent_interval_query.py",
        "continuum_shape_model.py", "pcc_interval_cbf.py", "shape_clearance.py",
    )
    report = {
        "schema": "v6_2_b2_prepared_prefix_trial_v1",
        "scope": "all_58_sampled_warm_unknown_plus_10_early_A1_states",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in source_names},
        "input_warm_shadow_sha256": _sha(warm_path),
        "point_budgets": [63, 127],
        "timing_order": ["reference", "prepared", "prepared", "reference"],
        "timing_scope": "cold_query_evaluate_only_no_mj_forward_no_jacobian_no_qp",
        "online_control_changed": False,
        "new_interval_mode_executed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "inputs": {}, "records": [], "modes": {},
    }
    for mode in ("baseline", "enabled"):
        metrics_path = a1_root / f"{mode}_root/output/v6_lite_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(metrics["scenarios"]) != 5:
            raise ValueError("A.1 must have five scenes per mode")
        selected: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for item in warm["modes"][mode]["frozen_feasibility_records"]:
            if item["proxy_status"] == "UNKNOWN_CROSSES_GATE":
                selected[item["scenario_id"]].append(("warm_unknown_sample", item["tick"]))
        expected_unknown = warm["modes"][mode]["counts"]["UNKNOWN_CROSSES_GATE"]
        if sum(map(len, selected.values())) != expected_unknown:
            raise ValueError("warm sampled unknown count changed")
        report["inputs"][mode] = {
            "metrics_path": metrics_path.as_posix(),
            "metrics_sha256": _sha(metrics_path), "traces": [],
        }
        for scene in metrics["scenarios"]:
            scenario_id = scene["scenario"]["scenario_id"]
            ticks = [("early_probe", 50), *selected[scenario_id]]
            if len({tick for _, tick in ticks}) != len(ticks):
                raise ValueError("duplicate selected state")
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
            prepared = BatchedDistanceDecisionQuery(evaluator.shape_model)
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
                        query = reference if implementation == "reference" else prepared
                        result = query.evaluate(
                            projection.planner_configuration, base, box,
                            IntervalPartition.uniform(), max_point_evaluations=budget,
                        )
                        timings[implementation].append(result.elapsed_ms)
                        results[implementation] = result
                    parity = _parity(results["reference"], results["prepared"])
                    report["records"].append({
                        "mode": mode, "scenario_id": scenario_id,
                        "state_group": label, "tick": tick,
                        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                        "point_budget": budget,
                        "status": results["reference"].proxy_clearance_status,
                        "point_evaluations": results["reference"].point_evaluation_count,
                        "leaf_count": results["reference"].interval_count,
                        "reference_query_ms": timings["reference"],
                        "prepared_query_ms": timings["prepared"],
                        "reference_median_ms": float(np.median(timings["reference"])),
                        "prepared_median_ms": float(np.median(timings["prepared"])),
                        "parity": parity,
                    })
            print(f"[b2-prepared-trial] {mode} {scenario_id}: {len(ticks)} states",
                  flush=True)
    for mode in ("baseline", "enabled"):
        report["modes"][mode] = {}
        for group in ("early_probe", "warm_unknown_sample"):
            report["modes"][mode][group] = {}
            for budget in (63, 127):
                own = [row for row in report["records"] if row["mode"] == mode
                       and row["state_group"] == group
                       and row["point_budget"] == budget]
                report["modes"][mode][group][str(budget)] = {
                    "state_count": len(own),
                    "status_counts": {name: sum(row["status"] == name for row in own)
                                      for name in sorted({row["status"] for row in own})},
                    "reference_median_query_ms": _summary(
                        [row["reference_median_ms"] for row in own]),
                    "prepared_median_query_ms": _summary(
                        [row["prepared_median_ms"] for row in own]),
                    "paired_speedup": _summary([
                        row["reference_median_ms"] / row["prepared_median_ms"]
                        for row in own]),
                    "max_parity_error_m": max(
                        max(row["parity"].values()) for row in own),
                }
    summary_path = output_dir / "prepared_query_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 同状态前缀复用的只读查询试验", "",
        "抽取全部 58 个已保存的持久热查询抽样未知状态，另加每场景 tick 50。"
        "两种实现从同一根分区重新查询，固定 63/127 点预算。"
        "按 ABBA 顺序计时，每状态取两次中位数。", "",
        "| 模式 | 状态组 | 点预算 | 状态数 | 原查询 p95 ms | 前缀复用 p95 ms | 配对加速比 p50 | 最大距离偏差 m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for group in ("early_probe", "warm_unknown_sample"):
            for budget in (63, 127):
                row = report["modes"][mode][group][str(budget)]
                lines.append(
                    f"| {mode} | {group} | {budget} | {row['state_count']} | "
                    f"{row['reference_median_query_ms']['p95']:.3f} | "
                    f"{row['prepared_median_query_ms']['p95']:.3f} | "
                    f"{row['paired_speedup']['p50']:.3f} | "
                    f"{row['max_parity_error_m']:.2e} |"
                )
    lines += [
        "", "逐状态要求分区、决策、点数和区间 ID 精确一致，距离偏差不大于 1e-12 m。"
        "旧力矩 trace 仅提供冻结状态；这里没有新命令执行。"
        "耗时不含 MuJoCo 正运动学、广义 Jacobian、原约束、QP 或力矩伺服；"
        "不能用本查询 p95 代替 20 ms 全链。数值实现未获形式化认证。", "",
    ]
    doc_path = output_dir / "PREPARED_QUERY_TRIAL.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_prepared_prefix_trial_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "document_sha256": _sha(doc_path),
    }
    with (output_dir / "prepared_query_manifest.json").open(
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
    parser.add_argument("--warm-shadow", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/shadow_warm/shadow_report.json"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.warm_shadow)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
