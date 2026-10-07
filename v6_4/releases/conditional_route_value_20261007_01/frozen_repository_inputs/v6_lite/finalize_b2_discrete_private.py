"""Hash-bind the five compensated private branches and independent replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(root: Path, recompute_dir: Path, origin_dir: Path,
        first_tick_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    origin_path = origin_dir / "servo_subspace_origin_summary.json"
    first_tick_path = first_tick_dir / "discrete_servo_summary.json"
    recompute_path = recompute_dir / "private_recompute_summary.json"
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    first_tick = json.loads(first_tick_path.read_text(encoding="utf-8"))
    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    if (origin["record_count"] != 50
            or [scene["first_step_outside_1e_10_rad"]
                for scene in origin["scenes"]] != [1] * 5
            or origin["maximum_saved_torque_error_nm"] != 0.0
            or origin["maximum_saved_qpos_error"] != 0.0
            or first_tick["record_count"] != 50
            or first_tick["online_controller_changed"]
            or first_tick["stage3_admission"]
            or first_tick["continuous_time_certified"]
            or any(scene["maximum_position_residual_rad"] > 1e-10
                   or scene["torque_saturation_step_count"]
                   for scene in first_tick["scenes"])
            or not recompute["pass_recompute"]
            or not recompute[
                "independent_strict_online_domain_all_executed_ticks"]
            or not recompute["all_500hz_states_on_declared_shape_subspace"]
            or recompute["maximum_500hz_subspace_residual_linf_rad"] > 1e-10
            or recompute["torque_limit_violation_count"]
            or recompute["failure_count"]
            or recompute["checked_task_states"] != 2005
            or recompute["checked_interval_rows"] != 5257
            or recompute["maximum_native_state_error"] > 1e-8):
        raise ValueError("compensated private evidence changed or failed")
    for source in (origin, first_tick, recompute):
        for name, digest in source["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"compensated evidence source changed: {name}")
    for directory, manifest_name, outputs in (
        (origin_dir, "servo_subspace_origin_manifest.json",
         {"summary": origin_path,
          "records": origin_dir / "servo_subspace_origin_records.jsonl"}),
        (first_tick_dir, "discrete_servo_manifest.json",
         {"summary": first_tick_path,
          "records": first_tick_dir / "discrete_servo_records.jsonl"}),
        (recompute_dir, "private_recompute_manifest.json",
         {"summary": recompute_path,
          "checks": recompute_dir / "private_recompute_checks.jsonl",
          "interval_rows": recompute_dir /
          "private_recompute_interval_rows.jsonl",
          "failures": recompute_dir / "private_recompute_failures.jsonl",
          "document": recompute_dir / "PRIVATE_RECOMPUTE.md"}),
    ):
        manifest = json.loads((directory / manifest_name).read_text(
            encoding="utf-8"))
        for label, path in outputs.items():
            if manifest[f"{label}_sha256"] != _sha(path):
                raise ValueError(f"compensated {directory.name} {label} changed")
    scenes = []
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        folder = root / f"scene_{index:02d}"
        summary_path = folder / "private_rollout_summary.json"
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = folder / "private_rollout_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = {
            "summary": summary_path,
            "records": folder / "private_rollout_records.jsonl",
            "trace": folder / "private_rollout_trace.npz",
            "document": folder / "PRIVATE_ROLLOUT.md",
        }
        for label, path in paths.items():
            if manifest[f"{label}_sha256"] != _sha(path):
                raise ValueError(f"compensated {scene_id} {label} changed")
        for name, digest in report["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"compensated source changed: {name}")
        if (report["schema"] != "v6_2_b2_discrete_private_multicycle_v1"
                or report["scope"] !=
                "private_interval_qp_closed_loop_with_implicitfast_damping_compensated_servo"
                or report["scenario_id"] != scene_id
                or report["predeclared_horizon_ticks"] != 400
                or report["executed_ticks"] != 400
                or report["stop_reason"] != "HORIZON_COMPLETE"
                or not report["strict_online_domain_all_executed_ticks"]
                or report["first_tick_outside_strict_online_domain"] is not None
                or report["maximum_subspace_residual_linf_rad"] > 1e-10
                or report["minimum_ramp_envelope_margin_m"] <= 0.0
                or report["minimum_realized_next_start_slack_m_s"] < 0.0
                or report["native_replay_max_qpos_error"] > 1e-8
                or report["production_online_controller_changed"]
                or report["stage3_admission"]
                or report["full_cycle_20ms_acceptance"]
                or report["continuous_time_certified"]
                or report["robot_mapping_changed"]
                or report["actuator_torque_limits_changed"]
                or report["physics_integrator_changed"]):
            raise ValueError(f"compensated private scene failed: {scene_id}")
        replay_input = recompute["inputs"][index]
        if (replay_input["scenario_id"] != scene_id
                or replay_input["summary_sha256"] != _sha(summary_path)
                or replay_input["trace_sha256"] != _sha(paths["trace"])
                or replay_input["records_sha256"] != _sha(paths["records"])
                or not replay_input[
                    "independent_strict_online_domain_all_executed_ticks"]
                or replay_input["first_500hz_step_outside_strict_subspace"]
                is not None):
            raise ValueError(f"compensated independent input changed: {scene_id}")
        scenes.append({
            "scenario_id": scene_id,
            "folder": folder.as_posix(),
            "summary_sha256": _sha(summary_path),
            "records_sha256": _sha(paths["records"]),
            "trace_sha256": _sha(paths["trace"]),
            "manifest_sha256": _sha(manifest_path),
            "executed_ticks": report["executed_ticks"],
            "maximum_subspace_residual_linf_rad":
                report["maximum_subspace_residual_linf_rad"],
            "minimum_ramp_envelope_margin_m":
                report["minimum_ramp_envelope_margin_m"],
            "minimum_next_start_slack_m_s":
                report["minimum_realized_next_start_slack_m_s"],
            "preflight_plus_qp_timing":
                report["private_preflight_plus_qp_timing"],
        })
    aggregate = {
        "schema": "v6_2_b2_discrete_private_five_scene_v1",
        "status": "PRIVATE_DIAGNOSTIC_COMPLETE_NOT_ONLINE_ADMISSION",
        "scenes": scenes,
        "complete_400_tick_scene_count": len(scenes),
        "checked_500hz_states": 5 * (4000 + 1),
        "maximum_500hz_subspace_residual_linf_rad": recompute[
            "maximum_500hz_subspace_residual_linf_rad"],
        "independent_interval_row_count": recompute["checked_interval_rows"],
        "independent_recompute_failure_count": recompute["failure_count"],
        "torque_limit_violation_count": recompute[
            "torque_limit_violation_count"],
        "origin_summary_sha256": _sha(origin_path),
        "first_tick_summary_sha256": _sha(first_tick_path),
        "recompute_summary_sha256": _sha(recompute_path),
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
    }
    report_path = output_dir / "discrete_private_five_scene_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 隐式积分阻尼补偿的私有五场景分支", "",
        "同一 17 维区间 QP、原 MuJoCo／胶囊行、十步斜坡、67 路力矩上限及"
        "MuJoCo implicitfast 模型保持；私有伺服逆动力学改用 "
        "M＋dt·diag(关节阻尼) 抵消离散步的已声明阻尼项。"
        "这不是生产在线切换。", "",
        "| 场景 | 周期 | 最大 50 Hz 子空间残差 rad | 最小包络余量 mm | "
        "最小下一起点松弛 mm/s | 预检＋QP p95 / 最大 ms | 超 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in scenes:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['executed_ticks']} | "
            f"{item['maximum_subspace_residual_linf_rad']:.2e} | "
            f"{1000*item['minimum_ramp_envelope_margin_m']:.3f} | "
            f"{1000*item['minimum_next_start_slack_m_s']:.3f} | "
            f"{timing['p95_ms']:.3f} / {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |"
        )
    lines += [
        "", f"五场景全部完成；独立力矩重放的 500 Hz 状态 "
        f"{aggregate['checked_500hz_states']} 个，最大子空间残差 "
        f"{aggregate['maximum_500hz_subspace_residual_linf_rad']:.2e} rad；"
        f"区间行 {aggregate['independent_interval_row_count']}，不一致 "
        f"{aggregate['independent_recompute_failure_count']}，原力矩上限违例 "
        f"{aggregate['torque_limit_violation_count']}。", "",
        "原未补偿伺服在第一 2 ms 步偏离严格子空间；十步补偿反事实没有。"
        "这些有限仿真和共享模型重算没有形成完整 20 ms 全链、故障注入、"
        "连续时间或生产在线安全验收；阶段三门禁仍关闭。", "",
    ]
    document_path = output_dir / "DISCRETE_PRIVATE_FIVE_SCENE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_discrete_private_five_scene_manifest_v1",
        "summary_sha256": _sha(report_path),
        "document_sha256": _sha(document_path),
    }
    with (output_dir / "discrete_private_five_scene_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_rollout_400"))
    parser.add_argument("--recompute-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_recompute_400"))
    parser.add_argument("--origin-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/servo_subspace_origin"))
    parser.add_argument("--first-tick-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_servo_first_tick"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.recompute_dir, args.origin_dir,
                 args.first_tick_dir, args.output_dir)
    print(json.dumps({"status": result["status"],
                      "complete_400_tick_scene_count":
                      result["complete_400_tick_scene_count"],
                      "maximum_500hz_subspace_residual_linf_rad":
                      result["maximum_500hz_subspace_residual_linf_rad"]},
                     indent=2))


if __name__ == "__main__":
    main()
