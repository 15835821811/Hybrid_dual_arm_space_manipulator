"""Compare private full-result and summary-only envelope timing exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_screened_next_start_timing as screened
from v6_lite.audit_b2_prepared_torque_timing import run as prepared_run
from v6_lite.b2_compact_fixed_arc_envelope import (
    CompactFixedArcStateLocalPCCEnvelopeAudit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, screened_dir: Path,
        fixed_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    original = screened.FixedArcStateLocalPCCEnvelopeAudit
    try:
        screened.FixedArcStateLocalPCCEnvelopeAudit = (
            CompactFixedArcStateLocalPCCEnvelopeAudit)
        trial = prepared_run(output_dir, screened_dir, fixed_dir,
                             baseline_dir, a1_root, scenario_id, ticks)
    finally:
        screened.FixedArcStateLocalPCCEnvelopeAudit = original
    records_path = output_dir / "private_rollout_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    reference = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
            encoding="utf-8").splitlines()]
    if len(records) != ticks or len(reference) != ticks:
        raise ValueError("compact envelope trial lacks full tick coverage")
    ignored_timings = {"qp_full_ms", "preflight_plus_qp_ms",
                       "independent_row_recompute_ms"}
    mismatch = next((tick for tick, (row, old) in enumerate(zip(
        records, reference)) if {key: value for key, value in row.items()
                                if key not in ignored_timings}
        != {key: value for key, value in old.items()
            if key not in ignored_timings}), None)
    if (mismatch is not None or not trial["all_non_timing_records_exact"]
            or not trial["trajectory_exact_to_original_baseline"]):
        raise ValueError(f"compact envelope changed private tick {mismatch}")
    old = json.loads((reference_dir / "prepared_torque_timing_summary.json")
                     .read_text(encoding="utf-8"))
    report = {
        "schema": "v6_2_b2_compact_envelope_private_timing_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "all_non_timing_records_exact": True,
        "all_500hz_envelope_states_checked": True,
        "capsule_sample_arithmetic_unchanged": True,
        "per_capsule_result_objects_omitted_in_private_trial": True,
        "trajectory_exact_to_original_baseline": True,
        "reference_necessary_component_sum": old["necessary_component_sum"],
        "trial_necessary_component_sum": trial["necessary_component_sum"],
        "reference_ten_step_branch": old["ten_step_branch"],
        "trial_ten_step_branch": trial["ten_step_branch"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_compact_envelope_timing.py",
                              "b2_compact_fixed_arc_envelope.py",
                              "pcc_fixed_arc_state_envelope.py",
                              "audit_b2_prepared_torque_timing.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                reference_dir / "private_rollout_records.jsonl"),
            "reference_trace": _sha(
                reference_dir / "private_rollout_trace.npz"),
            "trial_records": _sha(records_path),
            "trial_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
    }
    (output_dir / "compact_envelope_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 私有包络结果对象复用试验", "",
             "保留 11 个 500 Hz 状态、全部胶囊和弧长采样及原数值裕度；"
             "只省略私有分支未读取的逐胶囊结果对象。"
             "场景 01 的 1350 周期全部非计时记录和原生力矩 trace 逐值相同。", "",
             "| 指标 | 原实现 p95 ms | 试验 p95 ms | 试验 p99 ms | 试验最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, old_key, new_key in (
            ("十步候选分支", "ten_step_branch", "ten_step_branch"),
            ("必要计算和", "necessary_component_sum", "necessary_component_sum")):
        old_item, new_item = old[old_key], trial[new_key]
        lines.append(f"| {label} | {old_item['p95_ms']:.3f} | "
                     f"{new_item['p95_ms']:.3f} | {new_item['p99_ms']:.3f} | "
                     f"{new_item['max_ms']:.3f} |")
    lines += ["", "本试验未改变生产在线控制器；私有必要计算和不是完整在线周期。", ""]
    (output_dir / "COMPACT_ENVELOPE_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--screened-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.reference_dir, args.screened_dir,
                 args.fixed_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.ticks)
    print(json.dumps({key: report[key] for key in (
        "all_non_timing_records_exact", "reference_ten_step_branch",
        "trial_ten_step_branch", "trial_necessary_component_sum")},
        indent=2))


if __name__ == "__main__":
    main()
