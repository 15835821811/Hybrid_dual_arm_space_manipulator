"""Compare prepared and reference PCC containment on private 500 Hz states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_prepared_state_envelope import PreparedStateLocalPCCEnvelopeAudit
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


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
        *, scene_limit: int = 5, step_limit: int = 4001) -> dict:
    if not 1 <= scene_limit <= 5 or not 1 <= step_limit <= 4001:
        raise ValueError("trial extent outside five 400-cycle private scenes")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = {item["scenario"]["scenario_id"]: item
             for item in metrics["scenarios"]}
    scene_reports = []
    records_path = output_dir / "prepared_envelope_records.jsonl"
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
            model.geom_contype[:] = 0
            model.geom_conaffinity[:] = 0
            data = mujoco.MjData(model)
            evaluator = FixedIntervalCBFEvaluator(robot, model)
            reference = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
            prepared = PreparedStateLocalPCCEnvelopeAudit(
                model, evaluator.shape_spec)
            with np.load(trace_path, allow_pickle=False) as trace:
                qpos_states = trace["qpos_states"][:step_limit].copy()
            if len(qpos_states) != step_limit:
                raise ValueError("private trace is shorter than declared trial")
            elapsed = {"reference": [], "prepared": []}
            max_field_error = 0.0
            status_mismatch = 0
            capsule_name_mismatch = 0
            min_reference_margin = float("inf")
            for step, qpos in enumerate(qpos_states):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                projection = evaluator.shape_spec.project_actual_configuration(
                    data.qpos[evaluator.qpos_ids[:60]])
                base = transform_from_free_qpos(
                    data.qpos[evaluator.base_qpos_slice])
                order = ("reference", "prepared") if (step + index) % 2 == 0 else (
                    "prepared", "reference")
                result = {}
                for method in order:
                    checker = reference if method == "reference" else prepared
                    started = time.perf_counter()
                    result[method] = checker.evaluate(
                        data, projection.planner_configuration, base)
                    elapsed[method].append((time.perf_counter() - started) * 1000)
                old, new = result["reference"], result["prepared"]
                status_mismatch += old.status != new.status
                min_reference_margin = min(min_reference_margin,
                                           old.min_margin_m)
                if len(old.capsules) != len(new.capsules):
                    raise ValueError("prepared capsule count changed")
                row_error = abs(old.min_margin_m - new.min_margin_m)
                for left, right in zip(old.capsules, new.capsules):
                    capsule_name_mismatch += (
                        left.geom_name != right.geom_name
                        or left.segment_id != right.segment_id)
                    row_error = max(
                        row_error,
                        abs(left.sampled_axis_to_pcc_max_m
                            - right.sampled_axis_to_pcc_max_m),
                        abs(left.required_tube_radius_upper_m
                            - right.required_tube_radius_upper_m),
                        abs(left.margin_m - right.margin_m),
                    )
                max_field_error = max(max_field_error, row_error)
                stream.write(json.dumps({
                    "scenario_id": scene_id, "servo_step": step,
                    "reference_status": old.status,
                    "prepared_status": new.status,
                    "reference_minimum_margin_m": old.min_margin_m,
                    "prepared_minimum_margin_m": new.min_margin_m,
                    "maximum_capsule_field_difference_m": row_error,
                    "reference_ms": elapsed["reference"][-1],
                    "prepared_ms": elapsed["prepared"][-1],
                }, separators=(",", ":"), allow_nan=False) + "\n")
            scene_reports.append({
                "scenario_id": scene_id,
                "summary_sha256": _sha(summary_path),
                "trace_sha256": _sha(trace_path),
                "state_count": step_limit,
                "status_mismatch_count": status_mismatch,
                "capsule_name_mismatch_count": capsule_name_mismatch,
                "maximum_capsule_field_difference_m": max_field_error,
                "minimum_reference_margin_m": min_reference_margin,
                "reference_evaluate_ms": _stats(elapsed["reference"]),
                "prepared_evaluate_ms": _stats(elapsed["prepared"]),
            })
            print(f"[prepared-envelope] {scene_id}: {step_limit} states",
                  flush=True)
    max_error = max(x["maximum_capsule_field_difference_m"]
                    for x in scene_reports)
    failures = sum(x["status_mismatch_count"]
                   + x["capsule_name_mismatch_count"] for x in scene_reports)
    if max_error > 1e-9 or failures:
        raise ValueError("prepared envelope differs from reference")
    complete = scene_limit == 5 and step_limit == 4001
    report = {
        "schema": "v6_2_b2_prepared_envelope_parity_v1",
        "status": ("FULL_500HZ_PRIVATE_PARITY_PASS_NOT_ONLINE"
                   if complete else "PARTIAL_PRIVATE_PARITY_PROBE"),
        "scene_limit": scene_limit,
        "step_limit": step_limit,
        "checked_500hz_states": scene_limit * step_limit,
        "scenes": scene_reports,
        "maximum_capsule_field_difference_m": max_error,
        "mismatch_count": failures,
        "production_online_controller_changed": False,
        "online_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "input_metrics_sha256": _sha(metrics_path),
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_prepared_envelope.py",
                              "pcc_prepared_state_envelope.py",
                              "pcc_batched_point_model.py",
                              "pcc_state_local_envelope.py",
                              "continuum_shape_model.py",
                              "shape_clearance.py")},
        "records_sha256": _sha(records_path),
    }
    summary_path = output_dir / "prepared_envelope_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 私有 500 Hz 状态的 PCC 包络批量点计算对照", "",
        f"检查 {scene_limit} 场景、每场景 {step_limit} 个状态；"
        "每状态逐胶囊比较参考实现与共享段前缀的无 Jacobian 批量点计算。"
        "沿用同一原半径、采样点、轴间允差和数值余量。", "",
        "| 场景 | 状态 | 覆盖状态不一致 | 最大字段差 m | "
        "参考 p95 ms | 批量 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scene_reports:
        lines.append(
            f"| {row['scenario_id'][-2:]} | {row['state_count']} | "
            f"{row['status_mismatch_count']} | "
            f"{row['maximum_capsule_field_difference_m']:.3e} | "
            f"{row['reference_evaluate_ms']['p95']:.3f} | "
            f"{row['prepared_evaluate_ms']['p95']:.3f} |"
        )
    lines += [
        "", "本试验仅核对保存的离散状态与局部函数耗时；"
        "不构成新闭环全链时延、2 ms 步间包含或连续时间安全证明。", "",
    ]
    doc_path = output_dir / "PREPARED_ENVELOPE.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_prepared_envelope_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "records_sha256": _sha(records_path),
        "document_sha256": _sha(doc_path),
    }
    with (output_dir / "prepared_envelope_manifest.json").open(
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
    parser.add_argument("--step-limit", type=int, default=4001)
    args = parser.parse_args()
    report = run(args.output_dir, args.private_root, args.a1_root,
                 scene_limit=args.scene_limit, step_limit=args.step_limit)
    print(json.dumps({"status": report["status"],
                      "states": report["checked_500hz_states"],
                      "max_error_m": report[
                          "maximum_capsule_field_difference_m"]}, indent=2))


if __name__ == "__main__":
    main()
