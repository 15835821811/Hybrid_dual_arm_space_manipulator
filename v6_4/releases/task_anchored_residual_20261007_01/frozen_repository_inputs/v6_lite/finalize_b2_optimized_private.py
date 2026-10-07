"""Bind private performance diagnostics to exact replay and independent checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_bound(root: Path, folder: str, stem: str, files: dict[str, str]) -> tuple[dict, str]:
    directory = root / folder
    summary_path = directory / f"{stem}_summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((directory / f"{stem}_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {"summary": summary_path.name, **files}.items():
        if manifest[f"{label}_sha256"] != _sha(directory / filename):
            raise ValueError(f"{stem} {label} output hash changed")
    for name, digest in report["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"{stem} source changed: {name}")
    return report, _sha(summary_path)


def run(root: Path, output_dir: Path) -> dict:
    prepared_envelope, envelope_sha = _read_bound(root, "prepared_envelope_500hz",
        "prepared_envelope", {"records": "prepared_envelope_records.jsonl",
                              "document": "PREPARED_ENVELOPE.md"})
    prepared_query, query_sha = _read_bound(root, "prepared_private_query_2005",
        "prepared_private_query", {"records": "prepared_private_query_records.jsonl",
                                   "document": "PREPARED_PRIVATE_QUERY.md"})
    admm, admm_sha = _read_bound(root, "admm_solver_trial_2000",
        "admm_solver_trial", {"records": "admm_solver_trial_records.jsonl",
                              "document": "ADMM_SOLVER_TRIAL.md"})
    recompute, recompute_sha = _read_bound(root, "optimized_private_recompute_400",
        "private_recompute", {"checks": "private_recompute_checks.jsonl",
                              "interval_rows": "private_recompute_interval_rows.jsonl",
                              "failures": "private_recompute_failures.jsonl",
                              "document": "PRIVATE_RECOMPUTE.md"})
    if (prepared_envelope["checked_500hz_states"] != 20005
            or prepared_envelope["mismatch_count"]
            or prepared_envelope["maximum_capsule_field_difference_m"]
            or prepared_query["checked_task_states"] != 2005
            or prepared_query["maximum_distance_error_m"] > 1e-12
            or admm["screened_qp_inputs"] != 2000
            or admm["maximum_candidate_error"]
            or admm["maximum_dual_error"]
            or not admm["all_status_and_iterations_identical"]
            or admm["instrumented_full_qp_timing_valid"]
            or not recompute["pass_recompute"]
            or recompute["failure_count"]
            or recompute["checked_task_states"] != 2005
            or recompute["checked_interval_rows"] != 5257
            or not recompute["all_500hz_states_on_declared_shape_subspace"]
            or recompute["torque_limit_violation_count"]):
        raise ValueError("private diagnostic parity or independent replay failed")
    scenes = []
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        folder = root / "optimized_private_rollout_400" / f"scene_{index:02d}"
        baseline_folder = root / "discrete_private_rollout_400" / f"scene_{index:02d}"
        summary_path = folder / "private_rollout_summary.json"
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = folder / "private_rollout_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {"summary": summary_path,
                 "records": folder / "private_rollout_records.jsonl",
                 "trace": folder / "private_rollout_trace.npz",
                 "document": folder / "PRIVATE_ROLLOUT.md"}
        for label, path in files.items():
            if manifest[f"{label}_sha256"] != _sha(path):
                raise ValueError(f"{scene_id} {label} hash changed")
        for name, digest in report["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"{scene_id} source changed: {name}")
        replay_input = recompute["inputs"][index]
        if (report["schema"] != "v6_2_b2_optimized_private_multicycle_v1"
                or report["scenario_id"] != scene_id
                or report["executed_ticks"] != 400
                or report["stop_reason"] != "HORIZON_COMPLETE"
                or not report["strict_online_domain_all_executed_ticks"]
                or report["minimum_ramp_envelope_margin_m"] <= 0.0
                or report["minimum_realized_next_start_slack_m_s"] < 0.0
                or report["native_replay_max_qpos_error"] > 1e-8
                or report["production_online_controller_changed"]
                or report["stage3_admission"]
                or report["full_cycle_20ms_acceptance"]
                or report["continuous_time_certified"]
                or report["robot_mapping_changed"]
                or report["actuator_torque_limits_changed"]
                or report["physics_integrator_changed"]
                or replay_input["scenario_id"] != scene_id
                or replay_input["summary_sha256"] != _sha(summary_path)
                or replay_input["trace_sha256"] != _sha(files["trace"])
                or replay_input["records_sha256"] != _sha(files["records"])):
            raise ValueError(f"{scene_id} private protocol or recompute input changed")
        baseline_trace_path = baseline_folder / "private_rollout_trace.npz"
        baseline_summary_path = baseline_folder / "private_rollout_summary.json"
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        if baseline_summary["trace_sha256"] != _sha(baseline_trace_path):
            raise ValueError(f"{scene_id} baseline trace hash changed")
        with np.load(files["trace"]) as trace, np.load(baseline_trace_path) as baseline:
            parity = {key: float(np.max(np.abs(trace[key] - baseline[key])))
                      for key in ("torque", "qpos_states", "task_qvel_states")}
        if any(parity.values()):
            raise ValueError(f"{scene_id} changed the saved torque or state trajectory")
        scenes.append({"scenario_id": scene_id,
                       "summary_sha256": _sha(summary_path),
                       "manifest_sha256": _sha(manifest_path),
                       "trace_sha256": _sha(files["trace"]),
                       "baseline_trace_sha256": _sha(baseline_trace_path),
                       "exact_trace_parity": parity,
                       "preflight_plus_qp_timing": report["private_preflight_plus_qp_timing"],
                       "minimum_ramp_envelope_margin_m": report["minimum_ramp_envelope_margin_m"],
                       "minimum_next_start_slack_m_s": report["minimum_realized_next_start_slack_m_s"]})
    failed = [item["scenario_id"] for item in scenes
              if item["preflight_plus_qp_timing"]["p95_ms"] > 20.0]
    output = {"schema": "v6_2_b2_optimized_private_five_scene_v1",
              "status": "PRIVATE_DIAGNOSTIC_COMPLETE_TIMING_GATE_NOT_MET",
              "generator_source_sha256": _source_sha(Path("v6_lite/finalize_b2_optimized_private.py")),
              "scene_count": len(scenes),
              "ticks_per_scene": 400,
              "prepared_envelope_summary_sha256": envelope_sha,
              "prepared_query_summary_sha256": query_sha,
              "admm_solver_trial_summary_sha256": admm_sha,
              "independent_recompute_summary_sha256": recompute_sha,
              "checked_500hz_states": prepared_envelope["checked_500hz_states"],
              "checked_task_states": recompute["checked_task_states"],
              "independent_interval_rows": recompute["checked_interval_rows"],
              "independent_recompute_failure_count": recompute["failure_count"],
              "admm_solver_parity_inputs": admm["screened_qp_inputs"],
              "p95_over_20ms_scenes": failed,
              "scenes": scenes,
              "production_online_controller_changed": False,
              "stage3_admission": False,
              "full_cycle_20ms_acceptance": False,
              "continuous_time_certified": False}
    if not failed:
        raise ValueError("timing status requires at least one observed p95 failure")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "optimized_private_five_scene_summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# B.2 私有优化路径五场景诊断", "",
             "保持原 17 维加权速度 QP、MuJoCo／胶囊安全行、67 路力矩伺服、"
             "机器人映射和安全裕度；在私有分支中试验球界筛选、批量几何查询、"
             "状态包络点复用及等价 ADMM 乘积复用。没有接入正式在线控制。", "",
             "| 场景 | 周期 | 预检＋QP p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 | 力矩／状态轨迹差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in scenes:
        timing = item["preflight_plus_qp_timing"]
        lines.append(f"| {item['scenario_id'][-2:]} | 400 | {timing['p95_ms']:.3f} | "
                     f"{timing['p99_ms']:.3f} | {timing['max_ms']:.3f} | "
                     f"{timing['over_20ms_count']} | 0 / 0 |")
    lines += ["", f"p95 超预算场景：{', '.join(failed)}。单次 Windows 计时受环境扰动，"
              "该表只记实测值，不把不同运行间的全部差异归因于优化；尚需预先声明的重复计时。",
              "", f"完整 500 Hz 状态 {output['checked_500hz_states']} 个，"
              f"独立重放任务状态 {output['checked_task_states']} 个、区间行 "
              f"{output['independent_interval_rows']} 条，失败 "
              f"{output['independent_recompute_failure_count']}。原轨迹逐项完全一致。",
              "", "仪器化全 QP 试验每输入调用多个求解器，其总时延无效；"
              "只使用独立 ABBA 求解器计时。该阶段没有完整五场景正式运行、"
              "真实闭环切换验收或连续时间安全证明。", ""]
    document_path = output_dir / "OPTIMIZED_PRIVATE_FIVE_SCENE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"schema": "v6_2_b2_optimized_private_five_scene_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "document_sha256": _sha(document_path)}
    (output_dir / "optimized_private_five_scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.output_dir)
    print(json.dumps({key: report[key] for key in
                      ("status", "scene_count", "p95_over_20ms_scenes",
                       "independent_recompute_failure_count")}, indent=2))


if __name__ == "__main__":
    main()
