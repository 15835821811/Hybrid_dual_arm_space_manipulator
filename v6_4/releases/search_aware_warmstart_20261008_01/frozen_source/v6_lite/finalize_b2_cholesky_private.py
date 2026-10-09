"""Hash-bind private Cholesky timing, exact trajectories and independent replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ORDERS = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (0, 1, 2, 3, 4))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_rollout(folder: Path) -> tuple[dict, dict]:
    summary_path = folder / "private_rollout_summary.json"
    manifest_path = folder / "private_rollout_manifest.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_rollout_summary.json",
        "records": "private_rollout_records.jsonl",
        "trace": "private_rollout_trace.npz",
        "document": "PRIVATE_ROLLOUT.md",
    }.items():
        if manifest[f"{label}_sha256"] != _sha(folder / filename):
            raise ValueError(f"rollout {label} hash changed: {folder}")
    for name, digest in report["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"rollout source changed: {name}")
    if (report["schema"] != "v6_2_b2_cholesky_private_multicycle_v1"
            or report["executed_ticks"] != 400
            or report["stop_reason"] != "HORIZON_COMPLETE"
            or not report["strict_online_domain_all_executed_ticks"]
            or report["cholesky_unconstrained_call_count"] != 400
            or report["cholesky_unconstrained_fallback_count"]
            or report["maximum_unconstrained_linear_residual"] > 1e-9
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
        raise ValueError(f"rollout protocol changed or failed: {folder}")
    return report, manifest


def run(root: Path, output_dir: Path) -> dict:
    recompute_dir = root / "cholesky_private_recompute_400"
    recompute_path = recompute_dir / "private_recompute_summary.json"
    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    recompute_manifest = json.loads((recompute_dir /
        "private_recompute_manifest.json").read_text(encoding="utf-8"))
    for label, filename in {
        "summary": "private_recompute_summary.json",
        "checks": "private_recompute_checks.jsonl",
        "interval_rows": "private_recompute_interval_rows.jsonl",
        "failures": "private_recompute_failures.jsonl",
        "document": "PRIVATE_RECOMPUTE.md",
    }.items():
        if recompute_manifest[f"{label}_sha256"] != _sha(recompute_dir / filename):
            raise ValueError(f"cholesky independent recompute {label} changed")
    for name, digest in recompute["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"cholesky recompute source changed: {name}")
    if (not recompute["pass_recompute"] or recompute["failure_count"]
            or recompute["checked_task_states"] != 2005
            or recompute["checked_interval_rows"] != 5257
            or not recompute["all_500hz_states_on_declared_shape_subspace"]
            or recompute["torque_limit_violation_count"]
            or recompute["maximum_native_state_error"] > 1e-8):
        raise ValueError("cholesky independent recompute failed")
    scenes = []
    for scene_index in range(5):
        scene_id = f"v6_lite_scenario_{scene_index:02d}"
        folder = root / "cholesky_private_rollout_400" / f"scene_{scene_index:02d}"
        report, _manifest = _load_rollout(folder)
        trace_path = folder / "private_rollout_trace.npz"
        baseline_trace_path = (root / "discrete_private_rollout_400" /
                               f"scene_{scene_index:02d}" / "private_rollout_trace.npz")
        replay_input = recompute["inputs"][scene_index]
        if (report["scenario_id"] != scene_id
                or replay_input["scenario_id"] != scene_id
                or replay_input["summary_sha256"]
                != _sha(folder / "private_rollout_summary.json")
                or replay_input["trace_sha256"] != _sha(trace_path)
                or replay_input["records_sha256"]
                != _sha(folder / "private_rollout_records.jsonl")):
            raise ValueError(f"cholesky recompute input changed: {scene_id}")
        with np.load(trace_path) as trace, np.load(baseline_trace_path) as baseline:
            parity = {key: float(np.max(np.abs(trace[key] - baseline[key])))
                      for key in ("torque", "qpos_states", "task_qvel_states")}
        if any(parity.values()) or _sha(trace_path) != _sha(baseline_trace_path):
            raise ValueError(f"cholesky executed trace changed: {scene_id}")
        timing = report["private_preflight_plus_qp_timing"]
        if timing["p95_ms"] > 20.0:
            raise ValueError(f"cholesky first-round p95 exceeds budget: {scene_id}")
        scenes.append({
            "scenario_id": scene_id,
            "summary_sha256": _sha(folder / "private_rollout_summary.json"),
            "manifest_sha256": _sha(folder / "private_rollout_manifest.json"),
            "trace_sha256": _sha(trace_path),
            "baseline_trace_sha256": _sha(baseline_trace_path),
            "exact_trace_parity": parity,
            "preflight_plus_qp_timing": timing,
            "minimum_ramp_envelope_margin_m": report["minimum_ramp_envelope_margin_m"],
            "minimum_next_start_slack_m_s":
                report["minimum_realized_next_start_slack_m_s"],
        })
    repeat_dir = root / "cholesky_private_repeats_3x5"
    protocol_path = repeat_dir / "repeat_protocol.json"
    repeat_path = repeat_dir / "repeat_summary.json"
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    repeat_manifest = json.loads((repeat_dir / "repeat_manifest.json").read_text(
        encoding="utf-8"))
    for label, path in {"protocol": protocol_path, "summary": repeat_path,
                        "document": repeat_dir / "REPEAT_TIMING.md"}.items():
        if repeat_manifest[f"{label}_sha256"] != _sha(path):
            raise ValueError(f"cholesky repeat {label} hash changed")
    for name, digest in protocol["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"cholesky repeat source changed: {name}")
    expected_order = [(round_number, f"v6_lite_scenario_{scene_index:02d}")
                      for round_number, order in enumerate(ORDERS, start=1)
                      for scene_index in order]
    if (protocol["round_scene_order"] != [[scene for round_id, scene in
            expected_order if round_id == round_number]
            for round_number in (1, 2, 3)]
            or protocol["ticks_per_scene"] != 400
            or protocol["diagnostic_budget_ms"] != 20.0
            or repeat["protocol_sha256"] != _sha(protocol_path)
            or repeat["run_count"] != 15
            or repeat["p95_over_20ms_run_count"]
            or repeat["p95_over_20ms_runs"]
            or repeat["cholesky_unconstrained_fallback_count"]
            or repeat["status"]
            != "PRIVATE_REPEAT_TIMING_DIAGNOSTIC_PASS_NOT_ONLINE"
            or [(item["round"], item["scenario_id"]) for item in repeat["runs"]]
            != expected_order
            or repeat["production_online_controller_changed"]
            or repeat["stage3_admission"]
            or repeat["full_cycle_20ms_acceptance"]
            or repeat["continuous_time_certified"]):
        raise ValueError("cholesky repeat protocol or result changed")
    repeat_runs = []
    for item in repeat["runs"]:
        scene_index = int(item["scenario_id"][-2:])
        folder = (repeat_dir / f"round_{item['round']:02d}" /
                  f"scene_{scene_index:02d}")
        report, _manifest = _load_rollout(folder)
        trace_path = folder / "private_rollout_trace.npz"
        if (item["summary_sha256"] != _sha(folder / "private_rollout_summary.json")
                or item["manifest_sha256"]
                != _sha(folder / "private_rollout_manifest.json")
                or item["trace_sha256"] != _sha(trace_path)
                or item["trace_sha256"] != scenes[scene_index]["trace_sha256"]
                or item["preflight_plus_qp_timing"]
                != report["private_preflight_plus_qp_timing"]
                or item["preflight_plus_qp_timing"]["p95_ms"] > 20.0):
            raise ValueError(f"cholesky repeat raw run changed: {folder}")
        repeat_runs.append(item)
    phase_dir = root / "qp_phase_probe_scene02_400"
    phase_path = phase_dir / "qp_phase_summary.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    phase_manifest = json.loads((phase_dir / "qp_phase_manifest.json").read_text(
        encoding="utf-8"))
    for label, path in {"summary": phase_path,
                        "records": phase_dir / "qp_phase_records.jsonl",
                        "document": phase_dir / "QP_PHASE_PROBE.md"}.items():
        if phase_manifest[f"{label}_sha256"] != _sha(path):
            raise ValueError(f"QP phase {label} changed")
    for name, digest in phase["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"QP phase source changed: {name}")
    if (phase["ticks"] != 400 or phase["dense_solve_call_counts"] != [2]
            or phase["baseline_trace_sha256"] != scenes[2]["trace_sha256"]
            or phase["production_online_controller_changed"]):
        raise ValueError("QP phase provenance changed")
    total_over = sum(item["preflight_plus_qp_timing"]["over_20ms_count"]
                     for item in repeat_runs)
    maximum_elapsed = max(item["preflight_plus_qp_timing"]["max_ms"]
                          for item in repeat_runs)
    output = {
        "schema": "v6_2_b2_cholesky_private_five_scene_v1",
        "status": "PRIVATE_400_TICK_TIMING_DIAGNOSTIC_PASS_NOT_ONLINE",
        "generator_source_sha256": _source_sha(Path(
            "v6_lite/finalize_b2_cholesky_private.py")),
        "phase_probe_summary_sha256": _sha(phase_path),
        "independent_recompute_summary_sha256": _sha(recompute_path),
        "repeat_protocol_sha256": _sha(protocol_path),
        "repeat_summary_sha256": _sha(repeat_path),
        "scene_count": 5,
        "ticks_per_scene": 400,
        "repeat_round_count": 3,
        "repeat_run_count": 15,
        "repeat_p95_over_20ms_count": 0,
        "repeat_over_20ms_cycle_count": total_over,
        "repeat_maximum_preflight_plus_qp_ms": maximum_elapsed,
        "independent_task_states": recompute["checked_task_states"],
        "independent_interval_rows": recompute["checked_interval_rows"],
        "independent_failure_count": recompute["failure_count"],
        "scenes": scenes,
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "cholesky_private_five_scene_summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# B.2 Cholesky 私有诊断五场景与三轮计时", "",
             "仅将加权 17 维 Hessian 的无约束诊断线性方程改用 SPD Cholesky。"
             "原约束、17 维 ADMM、67 路力矩、机器人映射、安全裕度和数值容差"
             "均保持。此私有路径未接入生产在线控制。", "",
             "| 场景 | 首轮 p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 | 力矩／状态轨迹差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in scenes:
        timing = item["preflight_plus_qp_timing"]
        lines.append(f"| {item['scenario_id'][-2:]} | {timing['p95_ms']:.3f} | "
                     f"{timing['p99_ms']:.3f} | {timing['max_ms']:.3f} | "
                     f"{timing['over_20ms_count']} | 0 / 0 |")
    lines += ["", f"固定三轮 {len(repeat_runs)} 个场景轮次的 p95 全部低于 20 ms；"
              f"共 {total_over} 个单周期超 20 ms，最大 {maximum_elapsed:.3f} ms。"
              "p95 通过仍不构成硬实时保证。", "",
              f"独立重放 {output['independent_task_states']} 个任务状态、"
              f"{output['independent_interval_rows']} 条区间行，"
              f"失败 {output['independent_failure_count']}；全部新 trace "
              "与已独立重放的补偿基线哈希相同。", "",
              "此次仅验证各场景前 400 个规划周期的私有预检＋QP 时延。"
              "完整五场景在线目标、旧 26/11 项新 trace 验收、故障注入与"
              "连续时间安全证明均未由本报告建立。", ""]
    document_path = output_dir / "CHOLESKY_PRIVATE_FIVE_SCENE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "cholesky_private_five_scene_manifest.json").write_text(
        json.dumps({"schema": "v6_2_b2_cholesky_private_five_scene_manifest_v1",
                    "summary_sha256": _sha(summary_path),
                    "document_sha256": _sha(document_path)},
                   indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output_dir)
    print(json.dumps({key: result[key] for key in
                      ("status", "scene_count", "repeat_run_count",
                       "repeat_p95_over_20ms_count",
                       "repeat_over_20ms_cycle_count")}, indent=2))


if __name__ == "__main__":
    main()
