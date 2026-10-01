"""Independently check saved private endpoint rows and every 2 ms shape state.

The upstream private recompute separately replays native torques and rebuilds
interval CBF rows. This audit uses saved replay-checked states but reconstructs
the work-domain endpoint inequality without calling the controller's helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(private_dir: Path, recompute_dir: Path, a1_root: Path,
        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = private_dir / "private_rollout_summary.json"
    records_path = private_dir / "private_rollout_records.jsonl"
    trace_path = private_dir / "private_rollout_trace.npz"
    manifest_path = private_dir / "private_rollout_manifest.json"
    recompute_path = recompute_dir / "private_recompute_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    scene = summary["scenario_id"]
    ticks = summary["executed_ticks"]
    if (summary["stop_reason"] != "HORIZON_COMPLETE"
            or not summary["work_domain_endpoint_rows_in_same_qp"]
            or summary["attempted_ticks"] != ticks or len(records) != ticks
            or summary["records_sha256"] != _sha(records_path)
            or summary["trace_sha256"] != _sha(trace_path)
            or manifest["summary_sha256"] != _sha(summary_path)
            or manifest["records_sha256"] != _sha(records_path)
            or manifest["trace_sha256"] != _sha(trace_path)
            or not recompute["pass_recompute"]
            or recompute["failure_count"] != 0
            or recompute["task_ticks_per_scene"] != ticks
            or recompute["inputs"][0]["summary_sha256"] != _sha(summary_path)
            or recompute["inputs"][0]["trace_sha256"] != _sha(trace_path)):
        raise ValueError("private trace or independent torque/row replay changed")
    for name, digest in summary["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"private controller source changed: {name}")

    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = next(item for item in metrics["scenarios"] if
                 item["scenario"]["scenario_id"] == scene)
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    shape = default_continuum_model_spec(robot)
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=run_cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    qpos_ids, _ = joint_addresses(verifier.model, robot)
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = trace["qpos_states"].copy()
        torques = trace["torque"].copy()
    if (qpos.shape != (ticks * 10 + 1, verifier.model.nq)
            or torques.shape != (ticks * 10, 67)):
        raise ValueError("private trace dimensions changed")
    projected = np.asarray([
        shape.project_actual_configuration(state[qpos_ids[:60]])
        .planner_configuration for state in qpos
    ])
    lower = shape.work_domain_lower_rad
    upper = shape.work_domain_upper_rad
    actual_margin = np.minimum(projected - lower, upper - projected)
    minimum_actual_margin = float(np.min(actual_margin))
    first_outside = np.argwhere(actual_margin < -1e-12)
    endpoint_margins = []
    endpoint_prediction_errors = []
    unsupported_records = []
    for tick, row in enumerate(records):
        selected = row.get("selected_command")
        if (selected is None or row.get("action_mode") == "UNCERTIFIED"
                or row.get("failure_reason") != "none"):
            unsupported_records.append(tick)
            continue
        candidate = np.asarray(selected, dtype=np.float64)
        previous = (np.zeros(17, dtype=np.float64) if tick == 0 else
                    np.asarray(records[tick - 1]["selected_command"],
                               dtype=np.float64))
        if candidate.shape != (17,) or not np.all(np.isfinite(candidate)):
            unsupported_records.append(tick)
            continue
        # Independent reconstruction of the declared .45/.55 ten-step ramp.
        frozen_next = projected[tick * 10] + .02 * (
            .45 * previous[:10] + .55 * candidate[:10])
        endpoint_margins.append(float(np.min(np.minimum(
            frozen_next - lower, upper - frozen_next))))
        endpoint_prediction_errors.append(float(np.max(np.abs(
            projected[(tick + 1) * 10] - frozen_next))))
    minimum_endpoint_margin = min(endpoint_margins, default=float("nan"))
    failures = {
        "outside_executed_microstates": int(len(np.unique(first_outside[:, 0]))),
        "first_outside_microstep": (int(first_outside[0, 0])
                                     if len(first_outside) else None),
        "unsupported_executed_records": unsupported_records,
        "negative_frozen_endpoint_margins": int(sum(
            margin < -1e-10 for margin in endpoint_margins)),
    }
    report = {
        "schema": "v6_2_b2_domain_endpoint_independent_recompute_v1",
        "scenario_id": scene,
        "executed_ticks": ticks,
        "checked_2ms_states": len(qpos),
        "checked_frozen_endpoint_rows": len(endpoint_margins) * 20,
        "minimum_executed_domain_margin_rad": minimum_actual_margin,
        "minimum_frozen_endpoint_margin_rad": minimum_endpoint_margin,
        "maximum_frozen_endpoint_prediction_error_rad": max(
            endpoint_prediction_errors, default=float("nan")),
        "upstream_native_torque_and_interval_recompute_passed": True,
        "upstream_interval_rows": recompute["checked_interval_rows"],
        "failures": failures,
        "pass_domain_recompute": (minimum_actual_margin >= -1e-12
                                  and minimum_endpoint_margin >= -1e-10
                                  and not unsupported_records),
        "production_online_admitted": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "source_sha256": _source_sha(Path(__file__)),
        "inputs_sha256": {
            "private_summary": _sha(summary_path),
            "private_records": _sha(records_path),
            "private_trace": _sha(trace_path),
            "private_manifest": _sha(manifest_path),
            "independent_recompute": _sha(recompute_path),
            "a1_metrics": _sha(metrics_path),
        },
    }
    output_path = output_dir / "domain_endpoint_recompute.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8",
                           newline="\n")
    document = output_dir / "DOMAIN_ENDPOINT_RECOMPUTE.md"
    document.write_text(
        "# B.2 工作域端点与微状态独立核对\n\n"
        f"场景 `{scene}` 的新私有轨迹执行 {ticks} 个任务周期。"
        f"上游单独进程原生力矩重放和 {recompute['checked_interval_rows']} 条"
        "区间行重算均通过。此程序另从保存状态独立计算 20 条／周期的"
        "工作域端点不等式，并检查每个 2 ms 实际形状状态。\n\n"
        f"共检查 {len(qpos)} 个微状态，最小实际工作域余量 "
        f"{minimum_actual_margin:.6g} rad；最小冻结端点余量 "
        f"{minimum_endpoint_margin:.6g} rad；冻结预测与实际下一状态"
        f"最大差 {report['maximum_frozen_endpoint_prediction_error_rad']:.6g} rad。"
        f"域外微状态 {failures['outside_executed_microstates']}，"
        f"不合格已执行记录 {len(unsupported_records)}。\n\n"
        "仍共享声明的机器人和形状模型，不是独立物理测量；"
        "这个私有闭环结果不构成生产在线接入、20 ms 全链验收或连续时间证明。\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--recompute-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.private_dir, args.recompute_dir, args.a1_root,
                 args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "scenario_id", "executed_ticks", "checked_2ms_states",
        "minimum_executed_domain_margin_rad",
        "minimum_frozen_endpoint_margin_rad",
        "maximum_frozen_endpoint_prediction_error_rad",
        "pass_domain_recompute",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
