"""Pair PCC root queries on every compensated private planning state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_batched_query import _parity
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data))}


def run(output_dir: Path, private_root: Path, a1_root: Path,
        *, scene_limit: int = 5, tick_limit: int = 401) -> dict:
    if not 1 <= scene_limit <= 5 or not 1 <= tick_limit <= 401:
        raise ValueError("trial extent outside five 400-cycle private scenes")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = {item["scenario"]["scenario_id"]: item
             for item in metrics["scenarios"]}
    scenes = []
    records_path = output_dir / "prepared_private_query_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index in range(scene_limit):
            scene_id = f"v6_lite_scenario_{index:02d}"
            folder = private_root / f"scene_{index:02d}"
            trace_path = folder / "private_rollout_trace.npz"
            summary_path = folder / "private_rollout_summary.json"
            private = json.loads(summary_path.read_text(encoding="utf-8"))
            if (private["scenario_id"] != scene_id
                    or private["executed_ticks"] != 400
                    or private["trace_sha256"] != _sha(trace_path)):
                raise ValueError(f"private trace changed: {scene_id}")
            verifier = WholeBodyCollisionVerifier(
                robot, _obstacles(saved[scene_id]["scenario"]),
                WholeBodyVerificationConfig(
                    minimum_clearance=cfg.whole_body_minimum_clearance_m,
                    query_distance_max=2.5,
                    adaptive_subdivisions=cfg.verification_subdivisions,
                    self_collision_ancestor_exclusion_depth=3,
                    include_target_satellite_pairs=True,
                ),
            )
            model = verifier.model
            data = mujoco.MjData(model)
            evaluator = FixedIntervalCBFEvaluator(robot, model)
            reference = PersistentIntervalDecisionQuery(evaluator.shape_model)
            prepared = BatchedDistanceDecisionQuery(evaluator.shape_model)
            with np.load(trace_path, allow_pickle=False) as trace:
                qpos_states = trace["qpos_states"][::10][:tick_limit].copy()
            if len(qpos_states) != tick_limit:
                raise ValueError("private trace is shorter than requested ticks")
            reference_ms = []
            prepared_ms = []
            max_error = 0.0
            safe_count = 0
            for tick, qpos in enumerate(qpos_states):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                projection = evaluator.shape_spec.project_actual_configuration(
                    data.qpos[evaluator.qpos_ids[:60]])
                base = transform_from_free_qpos(
                    data.qpos[evaluator.base_qpos_slice])
                box = target_box_from_mujoco(
                    model, data, evaluator.target_geom_id)
                order = ((reference, prepared, prepared, reference)
                         if (tick + index) % 2 == 0 else
                         (prepared, reference, reference, prepared))
                own = {"reference": [], "prepared": []}
                outputs = {}
                for query in order:
                    name = "reference" if query is reference else "prepared"
                    value = query.evaluate(
                        projection.planner_configuration, base, box,
                        IntervalPartition.uniform(),
                        max_point_evaluations=255)
                    own[name].append(value.elapsed_ms)
                    outputs[name] = value
                parity = _parity(outputs["reference"], outputs["prepared"])
                error = max(parity.values())
                max_error = max(max_error, error)
                safe_count += (outputs["reference"].proxy_clearance_status
                               == "PROXY_CLEARANCE_AT_LEAST_GATE")
                reference_ms.append(float(np.median(own["reference"])))
                prepared_ms.append(float(np.median(own["prepared"])))
                stream.write(json.dumps({
                    "scenario_id": scene_id, "planning_tick": tick,
                    "point_evaluations": outputs["reference"].point_evaluation_count,
                    "interval_count": outputs["reference"].interval_count,
                    "status": outputs["reference"].proxy_clearance_status,
                    "maximum_distance_error_m": error,
                    "reference_median_ms": reference_ms[-1],
                    "prepared_median_ms": prepared_ms[-1],
                }, separators=(",", ":"), allow_nan=False) + "\n")
            scenes.append({
                "scenario_id": scene_id,
                "summary_sha256": _sha(summary_path),
                "trace_sha256": _sha(trace_path),
                "task_states": tick_limit,
                "proxy_safe_states": safe_count,
                "maximum_distance_error_m": max_error,
                "reference_median_query_ms": _stats(reference_ms),
                "prepared_median_query_ms": _stats(prepared_ms),
            })
            print(f"[prepared-private-query] {scene_id}: {tick_limit} states",
                  flush=True)
    count = scene_limit * tick_limit
    max_error = max(x["maximum_distance_error_m"] for x in scenes)
    if max_error > 1e-12:
        raise ValueError("private query parity failed")
    complete = scene_limit == 5 and tick_limit == 401
    report = {
        "schema": "v6_2_b2_prepared_private_query_parity_v1",
        "status": ("FULL_PRIVATE_QUERY_PARITY_PASS_NOT_ONLINE"
                   if complete else "PARTIAL_PRIVATE_QUERY_PROBE"),
        "checked_task_states": count,
        "scene_limit": scene_limit,
        "tick_limit": tick_limit,
        "point_budget": 255,
        "scenes": scenes,
        "maximum_distance_error_m": max_error,
        "production_online_controller_changed": False,
        "online_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "input_metrics_sha256": _sha(metrics_path),
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_prepared_query_private.py",
                              "audit_b2_batched_query.py",
                              "pcc_batched_distance_query.py",
                              "pcc_persistent_interval_query.py",
                              "continuum_shape_model.py")},
        "records_sha256": _sha(records_path),
    }
    summary_path = output_dir / "prepared_private_query_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 补偿私有轨迹的根区间批量查询对照", "",
        f"冻结 {count} 个私有规划状态；每状态对原与共享段前缀的批量查询"
        "按交替 ABBA 顺序执行，分别取两次中位数。"
        "每次从五段根区间开始，保持 255 点预算、5 mm 门槛和原半径。", "",
        "| 场景 | 状态 | 代理安全 | 最大距离差 m | 原查询 p95 ms | 批量 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenes:
        lines.append(
            f"| {row['scenario_id'][-2:]} | {row['task_states']} | "
            f"{row['proxy_safe_states']} | "
            f"{row['maximum_distance_error_m']:.3e} | "
            f"{row['reference_median_query_ms']['p95']:.3f} | "
            f"{row['prepared_median_query_ms']['p95']:.3f} |"
        )
    lines += [
        "", "逐状态核对分区、决策、求值次数、区间 ID 及距离；"
        "这仍是只读同状态测量，未计入 QP、包络或力矩伺服，"
        "不能代替 20 ms 全链验收。", "",
    ]
    doc_path = output_dir / "PREPARED_PRIVATE_QUERY.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_prepared_private_query_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "records_sha256": _sha(records_path),
                "document_sha256": _sha(doc_path)}
    with (output_dir / "prepared_private_query_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_rollout_400"))
    parser.add_argument("--a1-root", type=Path, default=Path(
        "v6_lite/output/v6_2_a1"))
    parser.add_argument("--scene-limit", type=int, default=5)
    parser.add_argument("--tick-limit", type=int, default=401)
    args = parser.parse_args()
    report = run(args.output_dir, args.private_root, args.a1_root,
                 scene_limit=args.scene_limit, tick_limit=args.tick_limit)
    print(json.dumps({"status": report["status"],
                      "states": report["checked_task_states"],
                      "max_error_m": report["maximum_distance_error_m"]},
                     indent=2))


if __name__ == "__main__":
    main()
