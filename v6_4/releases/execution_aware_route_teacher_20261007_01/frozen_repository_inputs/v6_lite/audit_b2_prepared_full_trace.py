"""Compare persistent PCC query implementations on every saved A.1 task state.

Both implementations see the same previous reference partition at each tick.
The partition is then advanced only from the reference result, so a mismatch
does not hide subsequent mismatches. This is direct MuJoCo forward kinematics
from saved qpos, not a second native torque replay or a new controller run.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    report = {
        "schema": "v6_2_b2_prepared_full_saved_trace_query_parity_v1",
        "scope": "all_13500_saved_A1_task_qpos_direct_mj_forward_not_torque_replay",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {
            name: _source_sha(Path("v6_lite") / name) for name in (
                "audit_b2_prepared_full_trace.py", "audit_b2_batched_query.py",
                "pcc_batched_distance_query.py", "pcc_persistent_interval_query.py",
                "continuum_shape_model.py", "pcc_interval_cbf.py", "shape_clearance.py",
            )
        },
        "input_warm_shadow_sha256": _sha(warm_path),
        "point_budget": 64, "max_leaves": 128,
        "timing_scope": "query_evaluate_only_no_mj_forward_no_jacobian_no_qp",
        "online_control_changed": False,
        "new_interval_mode_executed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "inputs": {}, "modes": {},
    }
    records_path = output_dir / "prepared_full_trace_states.jsonl"
    failures_path = output_dir / "prepared_full_trace_failures.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as states, \
            failures_path.open("x", encoding="utf-8", newline="\n") as failures:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root/output/v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if len(metrics["scenarios"]) != 5:
                raise ValueError("A.1 must contain five scenes per mode")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path), "traces": [],
            }
            counts: Counter[str] = Counter()
            reference_times = []
            prepared_times = []
            speedups = []
            parity_errors = []
            for scene in metrics["scenarios"]:
                scenario_id = scene["scenario"]["scenario_id"]
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
                model.geom_contype[:] = 0
                model.geom_conaffinity[:] = 0
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
                    selected = trace["task_selected_command"].copy()
                if len(selected) != 1350 or len(task_qpos) != len(selected)+1:
                    raise ValueError("A.1 task state cadence changed")
                partition = IntervalPartition.uniform()
                for tick in range(len(selected)):
                    qpos = task_qpos[tick]
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    projection = evaluator.shape_spec.project_actual_configuration(
                        data.qpos[evaluator.qpos_ids[:60]])
                    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                    box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                    if tick % 2:
                        candidate = prepared.evaluate(
                            projection.planner_configuration, base, box,
                            partition, max_point_evaluations=64)
                        original = reference.evaluate(
                            projection.planner_configuration, base, box,
                            partition, max_point_evaluations=64)
                    else:
                        original = reference.evaluate(
                            projection.planner_configuration, base, box,
                            partition, max_point_evaluations=64)
                        candidate = prepared.evaluate(
                            projection.planner_configuration, base, box,
                            partition, max_point_evaluations=64)
                    reason = None
                    parity = None
                    try:
                        parity = _parity(original, candidate)
                    except ValueError as exc:
                        reason = str(exc)
                    record = {
                        "mode": mode, "scenario_id": scenario_id, "tick": tick,
                        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                        "status": original.proxy_clearance_status,
                        "point_evaluations": original.point_evaluation_count,
                        "leaf_count": original.interval_count,
                        "reference_query_ms": original.elapsed_ms,
                        "prepared_query_ms": candidate.elapsed_ms,
                        "parity": parity, "parity_failure": reason,
                    }
                    states.write(json.dumps(record, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
                    if reason:
                        failures.write(json.dumps({
                            **record,
                            "prepared_status": candidate.proxy_clearance_status,
                            "reference_partition_ids": [
                                item.interval_id for item in original.partition.leaves],
                            "prepared_partition_ids": [
                                item.interval_id for item in candidate.partition.leaves],
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts["parity_failures"] += 1
                    else:
                        parity_errors.append(max(parity.values()))
                    partition = original.partition
                    counts["task_states"] += 1
                    counts[original.proxy_clearance_status] += 1
                    counts["budget_exhausted"] += int(original.budget_exhausted)
                    reference_times.append(original.elapsed_ms)
                    prepared_times.append(candidate.elapsed_ms)
                    speedups.append(original.elapsed_ms / candidate.elapsed_ms)
                print(f"[b2-prepared-full] {mode} {scenario_id}: 1350 states",
                      flush=True)
            expected = warm["modes"][mode]["counts"]
            if (counts["task_states"] != 6750 or
                    counts["UNKNOWN_CROSSES_GATE"] != expected["warm_all_task_unknown"] or
                    counts["budget_exhausted"] != expected["warm_all_task_budget_exhausted"]):
                raise ValueError("reference direct saved-state sweep disagrees with warm shadow")
            report["modes"][mode] = {
                "counts": dict(counts),
                "reference_query_ms": _summary(reference_times),
                "prepared_query_ms": _summary(prepared_times),
                "per_state_speedup": _summary(speedups),
                "maximum_parity_error_m": max(parity_errors, default=float("nan")),
            }
    report["state_records_sha256"] = _sha(records_path)
    report["failure_records_sha256"] = _sha(failures_path)
    summary_path = output_dir / "prepared_full_trace_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 保存状态全量持久热查询的前缀复用试验", "",
        "两组五场景、每场景 1,350 个旧 A.1 保存规划 qpos。每状态直接 MuJoCo 正运动学后，"
        "参考实现和试验实现从相同的前一时刻参考分区查询；分区只按参考结果向前推进。"
        "交替先运行哪一种实现。此项不重放力矩，不执行新区间命令。", "",
        "| 模式 | 状态 | 持久未知 | 决策/分区不一致 | 最大距离偏差 m | 原查询 p95 ms | 前缀复用 p95 ms | 单状态加速比 p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        counts = item["counts"]
        lines.append(
            f"| {mode} | {counts['task_states']} | {counts['UNKNOWN_CROSSES_GATE']} | "
            f"{counts.get('parity_failures', 0)} | "
            f"{item['maximum_parity_error_m']:.2e} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} | "
            f"{item['per_state_speedup']['p50']:.3f} |"
        )
    lines += [
        "", "逐状态记录与失败快照保存在独立 JSONL。"
        "这只比较几何查询实现，不含 MuJoCo 正运动学、广义 Jacobian、约束装配、QP、"
        "十步力矩伺服或 20 ms 全链。未改变在线控制及持久未知状态本身。"
        "距离偏差是双精度实现间的观察值，不是形式化浮点证明。", "",
    ]
    doc_path = output_dir / "PREPARED_FULL_TRACE.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_prepared_full_saved_trace_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("states", records_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "prepared_full_trace_manifest.json").open(
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
