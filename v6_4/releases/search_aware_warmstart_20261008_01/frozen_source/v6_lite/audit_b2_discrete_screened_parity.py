"""Hash-bind the compensated scene-00 screen trial and paired QP census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_scene(folder: Path) -> dict:
    report_path = folder / "private_rollout_summary.json"
    report = _read_json(report_path)
    manifest = _read_json(folder / "private_rollout_manifest.json")
    for label, filename in (
        ("summary", "private_rollout_summary.json"),
        ("records", "private_rollout_records.jsonl"),
        ("trace", "private_rollout_trace.npz"),
        ("document", "PRIVATE_ROLLOUT.md"),
    ):
        if manifest[f"{label}_sha256"] != _sha(folder / filename):
            raise ValueError(f"scene {folder.name} {label} hash changed")
    for name, digest in report["source_sha256"].items():
        if digest != _source_sha(Path("v6_lite") / name):
            raise ValueError(f"scene source changed: {name}")
    if (report["executed_ticks"] != 400
            or report["stop_reason"] != "HORIZON_COMPLETE"
            or not report["strict_online_domain_all_executed_ticks"]
            or report["native_replay_max_qpos_error"] > 1e-8
            or report["stage3_admission"]
            or report["full_cycle_20ms_acceptance"]):
        raise ValueError(f"scene {folder.name} no longer meets private checks")
    return report


def run(reference_dir: Path, screened_dir: Path, paired_dir: Path,
        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    reference = _check_scene(reference_dir)
    screened = _check_scene(screened_dir)
    if (reference["schema"] != "v6_2_b2_discrete_private_multicycle_v1"
            or screened["schema"]
            != "v6_2_b2_discrete_screened_private_multicycle_v1"
            or reference["scenario_id"] != screened["scenario_id"]
            or len(screened["sphere_screen_exact_pair_calls"]) != 400):
        raise ValueError("screened scene provenance changed")
    ref_trace = reference_dir / "private_rollout_trace.npz"
    screen_trace = screened_dir / "private_rollout_trace.npz"
    with np.load(ref_trace, allow_pickle=False) as original, np.load(
            screen_trace, allow_pickle=False) as modified:
        state_error = float(np.max(np.abs(
            original["qpos_states"] - modified["qpos_states"])))
        torque_error = float(np.max(np.abs(
            original["torque"] - modified["torque"])))
    if state_error > 1e-8 or torque_error > 1e-8:
        raise ValueError("screen changed executed private trajectory")
    paired_path = paired_dir / "private_full_qp_sphere_summary.json"
    paired = _read_json(paired_path)
    paired_manifest = _read_json(
        paired_dir / "private_full_qp_sphere_manifest.json")
    for label, filename in (
        ("summary", "private_full_qp_sphere_summary.json"),
        ("records", "private_full_qp_sphere_records.jsonl"),
        ("failures", "private_full_qp_sphere_failures.jsonl"),
    ):
        if paired_manifest[f"{label}_sha256"] != _sha(paired_dir / filename):
            raise ValueError(f"paired QP {label} hash changed")
    for name, digest in paired["source_sha256"].items():
        if digest != _source_sha(Path("v6_lite") / name):
            raise ValueError(f"paired QP source changed: {name}")
    for index in range(5):
        scene = f"v6_lite_scenario_{index:02d}"
        folder = reference_dir.parent / f"scene_{index:02d}"
        expected = paired["input_private_hashes"][scene]
        if (expected["summary_sha256"] != _sha(
                folder / "private_rollout_summary.json")
                or expected["records_sha256"] != _sha(
                    folder / "private_rollout_records.jsonl")
                or expected["trace_sha256"] != _sha(
                    folder / "private_rollout_trace.npz")):
            raise ValueError(f"paired QP input changed: {scene}")
    if (paired["record_count"] != 4000 or paired["failure_count"]
            or paired["private_servo_commanded_by_trial"]
            or paired["full_cycle_timing_measured"]):
        raise ValueError("paired QP census changed")
    timing = screened["private_preflight_plus_qp_timing"]
    pair_timing = {
        mode: paired["summary"][mode]["preflight_plus_qp_ms"]
        for mode in ("reference", "sphere_screen")
    }
    if (timing["p95_ms"] <= 20.0
            or any(item["p95"] <= 20.0 for item in pair_timing.values())):
        raise ValueError("timing conclusion changed; review gate status")
    report = {
        "schema": "v6_2_b2_discrete_screened_parity_v1",
        "status": "SCREEN_PARITY_PASS_TIMING_GATE_NOT_MET",
        "scope": "one_executed_private_scene_and_five_scene_read_only_paired_qp",
        "scenario_id": reference["scenario_id"],
        "reference_trace_sha256": _sha(ref_trace),
        "screened_trace_sha256": _sha(screen_trace),
        "reference_summary_sha256": _sha(
            reference_dir / "private_rollout_summary.json"),
        "screened_summary_sha256": _sha(
            screened_dir / "private_rollout_summary.json"),
        "paired_qp_summary_sha256": _sha(paired_path),
        "maximum_qpos_difference_m_or_rad": state_error,
        "maximum_torque_difference_nm": torque_error,
        "screened_scene_00_preflight_plus_qp": timing,
        "screened_scene_00_pair_exact_calls_p95": float(np.percentile(
            screened["sphere_screen_exact_pair_calls"], 95)),
        "paired_five_scene_preflight_plus_qp": pair_timing,
        "paired_record_count": paired["record_count"],
        "paired_failure_count": paired["failure_count"],
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "source_sha256": {"audit_b2_discrete_screened_parity.py":
                          _source_sha(Path(__file__))},
    }
    report_path = output_dir / "discrete_screened_parity_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 补偿伺服与球界筛选的时延反例", "",
        "场景 00 私有闭环均执行 400 个规划周期；筛选与原精确碰撞对版本"
        f"的完整状态差为 {state_error:.3e}，力矩差为 {torque_error:.3e} Nm。"
        "原 MuJoCo 约束由独立重构逐周期核对。", "",
        "| 实验 | 原对查询 p95 ms | 球界筛选 p95 ms | 筛选后超 20 ms |",
        "| --- | ---: | ---: | ---: |",
        f"| 五场景冻结状态配对、各 2000 次 | "
        f"{pair_timing['reference']['p95']:.3f} | "
        f"{pair_timing['sphere_screen']['p95']:.3f} | "
        f"{pair_timing['sphere_screen']['over_20ms_count']} |",
        f"| 场景 00 筛选私有闭环、400 次 | — | {timing['p95_ms']:.3f} | "
        f"{timing['over_20ms_count']} |", "",
        "冻结状态的配对测量改善了局部耗时，但 p95 仍超过 20 ms；"
        "私有闭环时延也超出门槛。这些均非完整调度周期的在线验收。"
        "阶段三门禁保持关闭。", "",
    ]
    doc_path = output_dir / "DISCRETE_SCREENED_PARITY.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_discrete_screened_parity_manifest_v1",
                "summary_sha256": _sha(report_path),
                "document_sha256": _sha(doc_path)}
    manifest_path = output_dir / "discrete_screened_parity_manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_rollout_400/scene_00"))
    parser.add_argument("--screened-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_screened_private_rollout_400/scene_00"))
    parser.add_argument("--paired-dir", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_full_qp_sphere"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.reference_dir, args.screened_dir, args.paired_dir,
                 args.output_dir)
    print(json.dumps({"status": report["status"],
                      "scene_00_p95_ms": report[
                          "screened_scene_00_preflight_plus_qp"]["p95_ms"],
                      "paired_screen_p95_ms": report[
                          "paired_five_scene_preflight_plus_qp"][
                              "sphere_screen"]["p95"]}, indent=2))


if __name__ == "__main__":
    main()
