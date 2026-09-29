"""Read-only deeper root-query frontier for frozen persistent unknown states.

The input is the complete same-state 64-point root counterfactual. Only its
998 still-unknown entries are revisited at 127 and 255 points. Both the
reference and state-prepared query are evaluated at each frozen saved qpos.
This is diagnostic: no online repartition, QP command, or torque is executed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


BUDGETS = (127, 255)
MAX_LEAVES = 256


def run(output_dir: Path, a1_root: Path, counterfactual_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    counterfactual = json.loads(counterfactual_path.read_text(encoding="utf-8"))
    if (counterfactual["point_budget"] != 64
            or counterfactual["repartition_applied_to_execution"]):
        raise ValueError("frozen 64-point root-counterfactual protocol changed")
    robot = default_v6_lite_robot_spec()
    report = {
        "schema": "v6_2_b2_unknown_root_budget_frontier_v1",
        "scope": "998_frozen_A1_states_unknown_after_64_point_root_requery",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {
            name: _source_sha(Path("v6_lite") / name) for name in (
                "audit_b2_root_rescue_frontier.py", "audit_b2_batched_query.py",
                "pcc_batched_distance_query.py", "pcc_persistent_interval_query.py",
                "continuum_shape_model.py", "pcc_interval_cbf.py", "shape_clearance.py",
            )
        },
        "input_repartition_counterfactual_sha256": _sha(counterfactual_path),
        "budgets": list(BUDGETS), "max_leaves": MAX_LEAVES,
        "query_start": "five_segment_roots_independently_at_each_budget",
        "timing_scope": "cold_query_only_no_mj_forward_no_jacobian_no_qp",
        "online_control_changed": False,
        "new_interval_mode_executed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "inputs": {}, "modes": {},
    }
    selected: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in counterfactual["entries"]:
        if item["cold_status"] == "UNKNOWN_CROSSES_GATE":
            selected[item["mode"], item["scenario_id"]].append(item)
    if sum(map(len, selected.values())) != 998:
        raise ValueError("frozen root-64 unknown population changed")
    records_path = output_dir / "root_rescue_states.jsonl"
    failures_path = output_dir / "root_rescue_failures.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as records, \
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
            per_budget = {budget: {"reference_ms": [], "prepared_ms": [],
                                   "speedup": [], "point_count": []}
                          for budget in BUDGETS}
            maximum_error = 0.0
            for scene in metrics["scenarios"]:
                scenario_id = scene["scenario"]["scenario_id"]
                source_rows = sorted(selected[mode, scenario_id],
                                     key=lambda item: item["tick"])
                if len({item["tick"] for item in source_rows}) != len(source_rows):
                    raise ValueError("duplicate root-64 unknown input")
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
                    selected_commands = trace["task_selected_command"]
                    if len(selected_commands) != 1350:
                        raise ValueError("A.1 task cadence changed")
                for source in source_rows:
                    tick = source["tick"]
                    qpos = task_qpos[tick]
                    qpos_sha = hashlib.sha256(qpos.tobytes()).hexdigest()
                    if qpos_sha != source["input_qpos_sha256"]:
                        raise ValueError("root-64 input qpos hash changed")
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    projection = evaluator.shape_spec.project_actual_configuration(
                        data.qpos[evaluator.qpos_ids[:60]])
                    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                    box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                    for budget in BUDGETS:
                        partition = IntervalPartition.uniform()
                        if tick % 2:
                            candidate = prepared.evaluate(
                                projection.planner_configuration, base, box,
                                partition, max_point_evaluations=budget,
                                max_leaves=MAX_LEAVES,
                            )
                            original = reference.evaluate(
                                projection.planner_configuration, base, box,
                                partition, max_point_evaluations=budget,
                                max_leaves=MAX_LEAVES,
                            )
                        else:
                            original = reference.evaluate(
                                projection.planner_configuration, base, box,
                                partition, max_point_evaluations=budget,
                                max_leaves=MAX_LEAVES,
                            )
                            candidate = prepared.evaluate(
                                projection.planner_configuration, base, box,
                                partition, max_point_evaluations=budget,
                                max_leaves=MAX_LEAVES,
                            )
                        parity = None
                        reason = None
                        try:
                            parity = _parity(original, candidate)
                            maximum_error = max(maximum_error, *parity.values())
                        except ValueError as exc:
                            reason = str(exc)
                        record = {
                            "mode": mode, "scenario_id": scenario_id,
                            "tick": tick, "qpos_sha256": qpos_sha,
                            "prior_root64_status": source["cold_status"],
                            "budget": budget,
                            "status": original.proxy_clearance_status,
                            "lower_m": original.distance_lower_bound_m,
                            "upper_m": original.distance_upper_bound_m,
                            "point_evaluations": original.point_evaluation_count,
                            "leaf_count": original.interval_count,
                            "reference_query_ms": original.elapsed_ms,
                            "prepared_query_ms": candidate.elapsed_ms,
                            "parity": parity, "parity_failure": reason,
                        }
                        records.write(json.dumps(record, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n")
                        if reason:
                            failures.write(json.dumps({
                                **record,
                                "prepared_status": candidate.proxy_clearance_status,
                                "reference_partition_ids": [
                                    leaf.interval_id for leaf in original.partition.leaves],
                                "prepared_partition_ids": [
                                    leaf.interval_id for leaf in candidate.partition.leaves],
                            }, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts["parity_failure"] += 1
                        counts[f"{budget}_{original.proxy_clearance_status}"] += 1
                        counts[f"{budget}_budget_exhausted"] += int(
                            original.budget_exhausted)
                        timings = per_budget[budget]
                        timings["reference_ms"].append(original.elapsed_ms)
                        timings["prepared_ms"].append(candidate.elapsed_ms)
                        timings["speedup"].append(original.elapsed_ms /
                                                  candidate.elapsed_ms)
                        timings["point_count"].append(original.point_evaluation_count)
                    counts["frozen_states"] += 1
                print(f"[b2-root-rescue] {mode} {scenario_id}: "
                      f"{len(source_rows)} root-64 unknown states", flush=True)
            expected_unknown = counterfactual["modes"][mode]["counts"][
                "cold_UNKNOWN_CROSSES_GATE"]
            if counts["frozen_states"] != expected_unknown:
                raise ValueError("root-64 unresolved source count disagrees")
            report["modes"][mode] = {
                "counts": dict(counts),
                "maximum_parity_error_m": maximum_error,
                "budgets": {
                    str(budget): {
                        "reference_query_ms": _summary(per_budget[budget]["reference_ms"]),
                        "prepared_query_ms": _summary(per_budget[budget]["prepared_ms"]),
                        "per_state_speedup": _summary(per_budget[budget]["speedup"]),
                        "point_evaluations": _summary(per_budget[budget]["point_count"]),
                    } for budget in BUDGETS
                },
            }
    report["state_records_sha256"] = _sha(records_path)
    report["failure_records_sha256"] = _sha(failures_path)
    summary_path = output_dir / "root_rescue_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 64 点根分区后仍未知状态的诊断预算阶梯", "",
        "对旧 A.1 保存状态中、64 点根分区重查后仍未知的 998 个状态，"
        "分别从五段根区间以 127 和 255 点独立查询。每个状态逐预算比较参考"
        "与前缀复用实现的判定、分区、点数、下界和上界。没有执行新控制动作。", "",
        "| 模式 | 冻结状态 | 点预算 | 确定安全 | 确定低于门槛 | 仍未知 | 参考查询 p95 ms | 前缀复用 p95 ms | 最大距离偏差 m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        for budget in BUDGETS:
            counts = item["counts"]
            row = item["budgets"][str(budget)]
            lines.append(
                f"| {mode} | {counts['frozen_states']} | {budget} | "
                f"{counts.get(f'{budget}_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
                f"{counts.get(f'{budget}_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
                f"{counts.get(f'{budget}_UNKNOWN_CROSSES_GATE', 0)} | "
                f"{row['reference_query_ms']['p95']:.3f} | "
                f"{row['prepared_query_ms']['p95']:.3f} | "
                f"{item['maximum_parity_error_m']:.2e} |"
            )
    lines += [
        "", "逐状态结果和失败快照位于独立 JSONL。这里的高预算只用于诊断，"
        "不等于原 64 点在线配置，更没有证明 20 ms 全链。"
        "代理低于门槛是 PCC 函数的状态，不等于 MuJoCo 实际碰撞；"
        "旧斜坡的起点违例和真实链包络证据也不因静态查询预算增加而自动解决。", "",
    ]
    doc_path = output_dir / "ROOT_RESCUE_FRONTIER.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_unknown_root_budget_frontier_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("states", records_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "root_rescue_manifest.json").open(
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
    parser.add_argument("--counterfactual", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/repartition_counterfactual/"
                                     "repartition_counterfactual.json"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.counterfactual)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
