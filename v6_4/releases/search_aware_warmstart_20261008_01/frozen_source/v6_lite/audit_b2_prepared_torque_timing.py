"""Private full-decision timing without redundant same-state servo forward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_discrete_private_rollout as discrete
from v6_lite.audit_b2_screened_next_start_timing import run as screened_run
from v6_lite.b2_prepared_compensated_torque import prepared_compensated_torque


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, fixed_dir: Path,
        baseline_dir: Path, a1_root: Path, scenario_id: str,
        ticks: int) -> dict:
    original = discrete._compensated_torque
    try:
        discrete._compensated_torque = prepared_compensated_torque
        timing = screened_run(output_dir, fixed_dir, baseline_dir, a1_root,
                              scenario_id, ticks)
    finally:
        discrete._compensated_torque = original
    records = [json.loads(line) for line in (
        output_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    reference = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(records) != ticks or len(reference) != ticks:
        raise ValueError("prepared torque records incomplete")
    ignored_timings = {"qp_full_ms", "preflight_plus_qp_ms",
                       "independent_row_recompute_ms"}
    first_mismatch = next((index for index, (row, old) in enumerate(zip(
        records, reference)) if {key: value for key, value in row.items()
                                if key not in ignored_timings}
        != {key: value for key, value in old.items()
            if key not in ignored_timings}), None)
    if first_mismatch is not None:
        raise ValueError(f"prepared torque changed tick {first_mismatch}")
    report = {
        "schema": "v6_2_b2_prepared_torque_private_timing_v1",
        "scenario_id": scenario_id, "task_ticks": ticks,
        "all_non_timing_records_exact": True,
        "trajectory_exact_to_original_baseline": timing[
            "trajectory_exact_to_original_baseline"],
        "same_67_channel_torque_and_limits": True,
        "physics_integrator_changed": False,
        "preflight_plus_qp": timing["preflight_plus_qp"],
        "ten_step_branch": timing["ten_step_branch"],
        "screened_next_start": timing["screened_next_start"],
        "necessary_component_sum": timing["necessary_component_sum"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "b2_prepared_compensated_torque.py",
                              "audit_b2_prepared_torque_timing.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                reference_dir / "private_rollout_records.jsonl"),
            "reference_timing": _sha(
                reference_dir / "reused_next_start_timing_summary.json"),
            "new_records": _sha(
                output_dir / "private_rollout_records.jsonl"),
            "new_timing": _sha(
                output_dir / "reused_next_start_timing_summary.json"),
            "new_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
    }
    (output_dir / "prepared_torque_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    old = json.loads((reference_dir / "reused_next_start_timing_summary.json")
                     .read_text(encoding="utf-8"))
    lines = ["# B.2 已准备状态的力矩计算私有计时", "",
             "十步候选分支已在每个 2 ms 状态执行 `mj_forward` 以检查实际链。"
             "力矩计算复用同一状态的质量矩阵、偏置力和被动力，"
             "保留原 67 路补偿力矩公式、限制、隐式积分器和诊断。"
             "场景 01 全部非计时记录与力矩 trace 逐值相同。", "",
             "| 部件 | 参考 p95 ms | 复用后 p95 ms | p99 ms | 最大 ms | 超 20 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, old_key, new_key in (
            ("预检＋QP", "preflight_plus_qp", "preflight_plus_qp"),
            ("十步候选分支", "ten_step_branch", "ten_step_branch"),
            ("下一起点", "reused_next_start", "screened_next_start"),
            ("三项之和", "necessary_component_sum", "necessary_component_sum")):
        old_t, new_t = old[old_key], report[new_key]
        lines.append(f"| {label} | {old_t['p95_ms']:.3f} | "
                     f"{new_t['p95_ms']:.3f} | {new_t['p99_ms']:.3f} | "
                     f"{new_t['max_ms']:.3f} | {new_t['over_20ms_count']} |")
    lines += ["", "这仍是私有必要计算的计时，不是完整在线时延。"
              "阶段二性能门禁未通过前，生产模式不变。", ""]
    (output_dir / "PREPARED_TORQUE_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.reference_dir, args.fixed_dir,
                 args.baseline_dir, args.a1_root, args.scenario_id,
                 args.ticks)
    print(json.dumps({key: report[key] for key in (
        "all_non_timing_records_exact", "ten_step_branch",
        "necessary_component_sum")}, indent=2))


if __name__ == "__main__":
    main()
