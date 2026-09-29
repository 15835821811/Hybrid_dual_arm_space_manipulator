"""Read-only 255-point root-query census of all saved A.1 planning states.

Each state starts from five material-section roots. The reference and prepared
queries see the same frozen MuJoCo pose; neither result changes the old action
or advances a persistent partition. This is a diagnostic budget, not online
admission or a new-mode torque replay.
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


POINT_BUDGET = 255
MAX_LEAVES = 256
STATE_COUNT_PER_SCENE = 1350


def run(output_dir: Path, a1_root: Path, warm_path: Path,
        rescue_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    rescue_summary_path = rescue_dir / "root_rescue_summary.json"
    rescue_states_path = rescue_dir / "root_rescue_states.jsonl"
    rescue_manifest_path = rescue_dir / "root_rescue_manifest.json"
    rescue = json.loads(rescue_summary_path.read_text(encoding="utf-8"))
    rescue_manifest = json.loads(rescue_manifest_path.read_text(encoding="utf-8"))
    if (rescue["budgets"] != [127, POINT_BUDGET]
            or rescue_manifest["states_sha256"] != _sha(rescue_states_path)):
        raise ValueError("frozen rescue input changed")
    expected = {}
    for line in rescue_states_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["budget"] == POINT_BUDGET:
            key = (row["mode"], row["scenario_id"], row["tick"])
            if key in expected:
                raise ValueError("duplicate rescue state")
            expected[key] = row
    if len(expected) != 998:
        raise ValueError("rescue population changed")
    report = {
        "schema": "v6_2_b2_all_saved_state_root255_census_v1",
        "scope": "all_13500_saved_A1_task_qpos_direct_mj_forward_not_torque_replay",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {
            name: _source_sha(Path("v6_lite") / name) for name in (
                "audit_b2_full_root255.py", "audit_b2_batched_query.py",
                "pcc_batched_distance_query.py", "pcc_persistent_interval_query.py",
                "continuum_shape_model.py", "pcc_interval_cbf.py", "shape_clearance.py",
            )
        },
        "input_warm_shadow_sha256": _sha(warm_path),
        "input_root_rescue_summary_sha256": _sha(rescue_summary_path),
        "input_root_rescue_states_sha256": _sha(rescue_states_path),
        "point_budget": POINT_BUDGET,
        "max_leaves": MAX_LEAVES,
        "query_start": "five_segment_roots_independently_at_every_saved_state",
        "timing_scope": "cold_query_only_no_mj_forward_no_jacobian_no_qp",
        "online_control_changed": False,
        "new_interval_mode_executed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "inputs": {}, "modes": {},
    }
    robot = default_v6_lite_robot_spec()
    states_path = output_dir / "full_root255_states.jsonl"
    failures_path = output_dir / "full_root255_failures.jsonl"
    with states_path.open("x", encoding="utf-8", newline="\n") as states, \
            failures_path.open("x", encoding="utf-8", newline="\n") as failures:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root/output/v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (len(metrics["scenarios"]) != 5
                    or _sha(metrics_path) != warm["modes"][mode]["metrics_sha256"]):
                raise ValueError("A.1 five-scene metrics changed")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path), "traces": [],
            }
            counts: Counter[str] = Counter()
            per_scene = {}
            reference_ms = []
            prepared_ms = []
            speedups = []
            maximum_error = 0.0
            for scene in metrics["scenarios"]:
                scenario_id = scene["scenario"]["scenario_id"]
                verifier = WholeBodyCollisionVerifier(
                    robot, _obstacles(scene["scenario"]),
                    WholeBodyVerificationConfig(
                        minimum_clearance=metrics["run_config"][
                            "whole_body_minimum_clearance_m"],
                        query_distance_max=2.5,
                        adaptive_subdivisions=metrics["run_config"][
                            "verification_subdivisions"],
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
                    selected = trace["task_selected_command"]
                    if (len(selected) != STATE_COUNT_PER_SCENE
                            or len(task_qpos) != STATE_COUNT_PER_SCENE + 1):
                        raise ValueError("A.1 task cadence changed")
                scene_counts: Counter[str] = Counter()
                first_below = None
                first_unknown = None
                for tick in range(STATE_COUNT_PER_SCENE):
                    qpos = task_qpos[tick]
                    qpos_sha = hashlib.sha256(qpos.tobytes()).hexdigest()
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    projection = evaluator.shape_spec.project_actual_configuration(
                        data.qpos[evaluator.qpos_ids[:60]])
                    base = transform_from_free_qpos(
                        data.qpos[evaluator.base_qpos_slice])
                    box = target_box_from_mujoco(model, data,
                                                 evaluator.target_geom_id)
                    partition = IntervalPartition.uniform()
                    kwargs = dict(max_point_evaluations=POINT_BUDGET,
                                  max_leaves=MAX_LEAVES)
                    if tick % 2:
                        candidate = prepared.evaluate(
                            projection.planner_configuration, base, box,
                            partition, **kwargs)
                        original = reference.evaluate(
                            projection.planner_configuration, base, box,
                            partition, **kwargs)
                    else:
                        original = reference.evaluate(
                            projection.planner_configuration, base, box,
                            partition, **kwargs)
                        candidate = prepared.evaluate(
                            projection.planner_configuration, base, box,
                            partition, **kwargs)
                    parity = None
                    reason = None
                    try:
                        parity = _parity(original, candidate)
                        maximum_error = max(maximum_error, *parity.values())
                    except ValueError as exc:
                        reason = str(exc)
                    key = (mode, scenario_id, tick)
                    if key in expected:
                        prior = expected.pop(key)
                        if (prior["qpos_sha256"] != qpos_sha
                                or prior["status"] != original.proxy_clearance_status
                                or prior["point_evaluations"]
                                != original.point_evaluation_count
                                or prior["leaf_count"] != original.interval_count
                                or abs(prior["lower_m"]
                                       - original.distance_lower_bound_m) > 1e-12
                                or abs(prior["upper_m"]
                                       - original.distance_upper_bound_m) > 1e-12):
                            raise ValueError(f"root rescue result changed: {key}")
                        counts["root_rescue_matches"] += 1
                    status = original.proxy_clearance_status
                    row = {
                        "mode": mode, "scenario_id": scenario_id,
                        "tick": tick, "qpos_sha256": qpos_sha,
                        "status": status,
                        "lower_m": original.distance_lower_bound_m,
                        "upper_m": original.distance_upper_bound_m,
                        "point_evaluations": original.point_evaluation_count,
                        "leaf_count": original.interval_count,
                        "budget_exhausted": original.budget_exhausted,
                        "reference_query_ms": original.elapsed_ms,
                        "prepared_query_ms": candidate.elapsed_ms,
                        "parity": parity, "parity_failure": reason,
                    }
                    states.write(json.dumps(row, ensure_ascii=False,
                                            separators=(",", ":"),
                                            allow_nan=False) + "\n")
                    if reason:
                        failures.write(json.dumps({
                            **row,
                            "prepared_status": candidate.proxy_clearance_status,
                            "reference_partition_ids": [
                                leaf.interval_id for leaf in original.partition.leaves],
                            "prepared_partition_ids": [
                                leaf.interval_id for leaf in candidate.partition.leaves],
                        }, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")
                        counts["parity_failures"] += 1
                    counts["task_states"] += 1
                    counts[status] += 1
                    counts["budget_exhausted"] += int(original.budget_exhausted)
                    scene_counts[status] += 1
                    if status == "PROXY_CLEARANCE_BELOW_GATE" and first_below is None:
                        first_below = tick
                    if status == "UNKNOWN_CROSSES_GATE" and first_unknown is None:
                        first_unknown = tick
                    reference_ms.append(original.elapsed_ms)
                    prepared_ms.append(candidate.elapsed_ms)
                    speedups.append(original.elapsed_ms / candidate.elapsed_ms)
                per_scene[scenario_id] = {
                    "status_counts": dict(scene_counts),
                    "first_below_tick": first_below,
                    "first_unknown_tick": first_unknown,
                }
                print(f"[b2-root255] {mode} {scenario_id}: "
                      f"{dict(scene_counts)}", flush=True)
            if (counts["task_states"] != 5 * STATE_COUNT_PER_SCENE
                    or counts["root_rescue_matches"]
                    != rescue["modes"][mode]["counts"]["frozen_states"]):
                raise ValueError("full census population changed")
            report["modes"][mode] = {
                "counts": dict(counts), "scenarios": per_scene,
                "maximum_parity_error_m": maximum_error,
                "reference_query_ms": _summary(reference_ms),
                "prepared_query_ms": _summary(prepared_ms),
                "per_state_speedup": _summary(speedups),
            }
    if expected:
        raise ValueError(f"unmatched rescue states: {len(expected)}")
    report["state_records_sha256"] = _sha(states_path)
    report["failure_records_sha256"] = _sha(failures_path)
    summary_path = output_dir / "full_root255_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 全部保存状态的 255 点根区间查询", "",
        "两组旧 A.1 五场景、共 13,500 个保存规划状态；每状态独立从五段根区间查询。"
        "参考与前缀复用实现逐状态比较；998 个先前根64未知状态的 255 点结果"
        "亦与独立诊断记录核对。没有重放新区间模式力矩，也没有在线动作。", "",
        "| 模式 | 状态 | 安全 | 低于代理门槛 | 未知 | 预算耗尽 | 实现不一致 | 参考查询 p95 ms | 前缀复用 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        counts = item["counts"]
        lines.append(
            f"| {mode} | {counts['task_states']} | "
            f"{counts.get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{counts.get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{counts.get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{counts['budget_exhausted']} | {counts.get('parity_failures', 0)} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} |"
        )
    lines += [
        "", "这些计数属于旧轨迹的 PCC 代理分类；低于门槛不等于 MuJoCo 实际碰撞。"
        "查询计时不含 MuJoCo 正运动学、Jacobian、约束装配、QP 或力矩伺服。"
        "高预算不等于可接受的 20 ms 全链性能，也不消除旧速度起点违反。"
        "逐状态结果和失败快照单独保存并由哈希清单绑定。", "",
    ]
    doc_path = output_dir / "FULL_ROOT255_CENSUS.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_all_saved_state_root255_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("states", states_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "full_root255_manifest.json").open(
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
    parser.add_argument("--rescue-dir", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/root_rescue_frontier"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.warm_shadow,
                 args.rescue_dir)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
