"""Trial batched fixed-arc positions in the exact private B.2 decision chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import pcc_fixed_arc_state_envelope as fixed_module
from v6_lite.audit_b2_batched_capsule_private_timing import run as batched_run
from v6_lite.pcc_vectorized_fixed_arc_positions import VectorizedFixedArcPCCPositions


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, batched_reference_dir: Path,
        compact_reference_dir: Path, screened_dir: Path,
        fixed_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    original = fixed_module.FixedArcPCCPositions
    try:
        fixed_module.FixedArcPCCPositions = VectorizedFixedArcPCCPositions
        trial = batched_run(output_dir, compact_reference_dir,
                            screened_dir, fixed_dir, baseline_dir,
                            a1_root, scenario_id, ticks)
    finally:
        fixed_module.FixedArcPCCPositions = original
    reference = json.loads((batched_reference_dir
                            / "batched_capsule_private_timing_summary.json")
                           .read_text(encoding="utf-8"))
    if (not trial["all_non_timing_records_exact"]
            or not trial["same_native_67_channel_torque_trace"]
            or trial["inputs_sha256"]["trial_trace"]
            != reference["inputs_sha256"]["trial_trace"]):
        raise ValueError("vectorized positions changed private control output")
    report = {
        "schema": "v6_2_b2_vectorized_positions_private_timing_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "all_non_timing_records_exact": True,
        "same_native_67_channel_torque_trace": True,
        "reference_necessary_component_sum":
            reference["trial_necessary_component_sum"],
        "trial_necessary_component_sum": trial["trial_necessary_component_sum"],
        "reference_ten_step_branch": reference["trial_ten_step_branch"],
        "trial_ten_step_branch": trial["trial_ten_step_branch"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_vectorized_positions_private_timing.py",
                              "audit_b2_batched_capsule_private_timing.py",
                              "pcc_vectorized_fixed_arc_positions.py",
                              "pcc_fixed_arc_positions.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                batched_reference_dir / "private_rollout_records.jsonl"),
            "reference_trace": _sha(
                batched_reference_dir / "private_rollout_trace.npz"),
            "trial_records": _sha(output_dir / "private_rollout_records.jsonl"),
            "trial_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
    }
    (output_dir / "vectorized_positions_private_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    old = report["reference_necessary_component_sum"]
    new = report["trial_necessary_component_sum"]
    lines = ["# B.2 私有五段固定弧长批量计算试验", "",
             "五段 Rodrigues 系数按相同公式批量计算；原 61 条胶囊、"
             "5 mm 门槛、十步斜坡和 67 路力矩不变。"
             "场景 01 全部 1350 周期的非计时记录与力矩 trace 和参考逐值相同。", "",
             "| 指标 | 参考 p95 ms | 试验 p95 ms | 试验 p99 ms | 最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, old_key, new_key in (
            ("十步候选分支", "reference_ten_step_branch", "trial_ten_step_branch"),
            ("必要计算和", "reference_necessary_component_sum",
             "trial_necessary_component_sum")):
        before, after = report[old_key], report[new_key]
        lines.append(f"| {label} | {before['p95_ms']:.3f} | "
                     f"{after['p95_ms']:.3f} | {after['p99_ms']:.3f} | "
                     f"{after['max_ms']:.3f} |")
    lines += ["", f"必要计算和超过 20 ms 的周期为 {new['over_20ms_count']} / "
              f"{ticks}。私有计时不含完整在线调度；阶段二总门禁仍须单独审核。", ""]
    (output_dir / "VECTORIZED_POSITIONS_PRIVATE_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batched-reference-dir", type=Path, required=True)
    parser.add_argument("--compact-reference-dir", type=Path, required=True)
    parser.add_argument("--screened-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.batched_reference_dir,
                 args.compact_reference_dir, args.screened_dir,
                 args.fixed_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.ticks)
    print(json.dumps({key: report[key] for key in (
        "all_non_timing_records_exact", "same_native_67_channel_torque_trace",
        "reference_necessary_component_sum", "trial_necessary_component_sum")},
        indent=2))


if __name__ == "__main__":
    main()
