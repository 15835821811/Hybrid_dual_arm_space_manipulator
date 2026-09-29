"""Private timing with fixed arc indexing and unchanged 11-state checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_prepared_screened_private_rollout as prepared
from v6_lite.audit_b2_reused_next_start_timing import run as reused_run
from v6_lite.pcc_fixed_arc_state_envelope import (
    FixedArcStateLocalPCCEnvelopeAudit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, baseline_dir: Path,
        a1_root: Path, scenario_id: str, ticks: int) -> dict:
    original = prepared.PreparedStateLocalPCCEnvelopeAudit
    try:
        prepared.PreparedStateLocalPCCEnvelopeAudit = (
            FixedArcStateLocalPCCEnvelopeAudit)
        timing = reused_run(output_dir, baseline_dir, a1_root,
                            scenario_id, ticks)
    finally:
        prepared.PreparedStateLocalPCCEnvelopeAudit = original
    records = [json.loads(line) for line in (
        output_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    reference = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(records) != ticks or len(reference) != ticks:
        raise ValueError("fixed-arc evidence has incomplete records")
    exact_keys = (
        "current_envelope_status", "current_envelope_margin_m",
        "proxy_status", "selected_interval_count", "excluded_interval_count",
        "interval_rows", "geometry_domain_status",
        "analytic_bound_assumptions_satisfied", "envelope_evidence_status",
        "strict_online_domain_met", "qp_solver_status", "qp_iterations",
        "action_mode", "failure_reason", "candidate_valid", "ramp_valid",
        "selected_command", "ramp_domain_failures",
        "minimum_ramp_envelope_margin_m", "ramp_covered_microstates",
        "realized_next_start", "row_parity",
    )
    first_mismatch = next((index for index, (row, old) in enumerate(
        zip(records, reference)) if any(row.get(key) != old.get(key)
                                          for key in exact_keys)), None)
    if first_mismatch is not None:
        raise ValueError(f"fixed-arc envelope changed tick {first_mismatch}")
    report = {
        "schema": "v6_2_b2_fixed_arc_private_full_decision_timing_v1",
        "scenario_id": scenario_id, "task_ticks": ticks,
        "reference_record_decisions_and_envelope_margins_exact": True,
        "trajectory_exact_to_original_baseline": timing[
            "trajectory_exact_to_baseline"],
        "fixed_arc_preflight_plus_qp": timing["preflight_plus_qp"],
        "fixed_arc_ten_step_branch": timing["ten_step_branch"],
        "reused_next_start": timing["reused_next_start"],
        "necessary_component_sum": timing["necessary_component_sum"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "pcc_fixed_arc_positions.py",
                              "pcc_fixed_arc_state_envelope.py",
                              "audit_b2_fixed_arc_full_decision_timing.py")},
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
    (output_dir / "fixed_arc_full_decision_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    old = json.loads((reference_dir / "reused_next_start_timing_summary.json")
                     .read_text(encoding="utf-8"))
    lines = ["# B.2 固定弧长索引的私有决策计时", "",
             "只缓存 85 个固定弧长样本的段索引；每周期仍重算 10 维形状、"
             "11 个 MuJoCo 状态、全部实际链胶囊距离和原 17 维 QP。"
             "未生成包络检查不用的旋转/Jacobian 结果对象。", "",
             "场景 01 完整 1350 个周期的决策、包络最小裕度、"
             "独立下一起点及力矩轨迹与前一私有变体逐值相同。", "",
             "| 部件 | 参考 p95 ms | 固定索引 p95 ms | p99 ms | 最大 ms | 超 20 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, old_key, new_key in (
            ("预检＋QP", "preflight_plus_qp", "fixed_arc_preflight_plus_qp"),
            ("十步候选分支", "ten_step_branch", "fixed_arc_ten_step_branch"),
            ("下一起点", "reused_next_start", "reused_next_start"),
            ("三项之和", "necessary_component_sum", "necessary_component_sum")):
        old_t, new_t = old[old_key], report[new_key]
        lines.append(f"| {label} | {old_t['p95_ms']:.3f} | "
                     f"{new_t['p95_ms']:.3f} | {new_t['p99_ms']:.3f} | "
                     f"{new_t['max_ms']:.3f} | {new_t['over_20ms_count']} |")
    lines += ["", "三项之和仍不是完整在线时延；阶段二未达到 20 ms 门槛，"
              "生产模式保持不变。", ""]
    (output_dir / "FIXED_ARC_FULL_DECISION.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    result = run(args.output_dir, args.reference_dir, args.baseline_dir,
                 args.a1_root, args.scenario_id, args.ticks)
    print(json.dumps({key: result[key] for key in (
        "reference_record_decisions_and_envelope_margins_exact",
        "trajectory_exact_to_original_baseline", "fixed_arc_ten_step_branch",
        "necessary_component_sum")}, indent=2))


if __name__ == "__main__":
    main()
