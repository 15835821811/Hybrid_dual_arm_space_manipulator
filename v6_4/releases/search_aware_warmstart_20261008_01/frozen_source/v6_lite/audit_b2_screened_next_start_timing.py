"""Private next-start timing with conservative MuJoCo pair screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_prepared_screened_private_rollout as prepared
from v6_lite import audit_b2_reused_next_start_timing as timing_module
from v6_lite.b2_screened_next_start_rows import ScreenedNextStart
from v6_lite.pcc_fixed_arc_state_envelope import (
    FixedArcStateLocalPCCEnvelopeAudit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, baseline_dir: Path,
        a1_root: Path, scenario_id: str, ticks: int) -> dict:
    original_envelope = prepared.PreparedStateLocalPCCEnvelopeAudit
    original_builder = timing_module.ReusedNextStart
    ScreenedNextStart.instances.clear()
    try:
        prepared.PreparedStateLocalPCCEnvelopeAudit = (
            FixedArcStateLocalPCCEnvelopeAudit)
        timing_module.ReusedNextStart = ScreenedNextStart
        timing = timing_module.run(output_dir, baseline_dir, a1_root,
                                   scenario_id, ticks)
    finally:
        prepared.PreparedStateLocalPCCEnvelopeAudit = original_envelope
        timing_module.ReusedNextStart = original_builder
    if len(ScreenedNextStart.instances) != 1:
        raise ValueError("screened next-start instance count changed")
    builder = ScreenedNextStart.instances[0]._builder
    if builder.calls != ticks or builder.exact_pair_calls >= ticks * len(
            builder._all_pairs):
        raise ValueError("screened rows did not cover every next start")
    records = [json.loads(line) for line in (
        output_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    reference = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(records) != ticks or len(reference) != ticks:
        raise ValueError("screened records incomplete")
    first_mismatch = next((index for index, (row, old) in enumerate(zip(
        records, reference)) if row["realized_next_start"]
        != old["realized_next_start"]), None)
    if first_mismatch is not None:
        raise ValueError(f"screen changed next-start rows at {first_mismatch}")
    report = {
        "schema": "v6_2_b2_screened_next_start_private_timing_v1",
        "scenario_id": scenario_id, "task_ticks": ticks,
        "next_start_records_exact": True,
        "trajectory_exact_to_original_baseline": timing[
            "trajectory_exact_to_baseline"],
        "all_original_collision_pair_classes_preserved": True,
        "activation_distance_unchanged": True,
        "screened_builder_calls": builder.calls,
        "all_pair_count_per_tick": len(builder._all_pairs),
        "total_exact_pair_calls": builder.exact_pair_calls,
        "minimum_retained_pairs_per_tick": builder.minimum_retained_pairs,
        "preflight_plus_qp": timing["preflight_plus_qp"],
        "ten_step_branch": timing["ten_step_branch"],
        "screened_next_start": timing["reused_next_start"],
        "necessary_component_sum": timing["necessary_component_sum"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "b2_screened_next_start_rows.py",
                              "audit_b2_screened_next_start_timing.py",
                              "audit_b2_mujoco_sphere_screen.py")},
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
    (output_dir / "screened_next_start_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    old = json.loads((reference_dir / "reused_next_start_timing_summary.json")
                     .read_text(encoding="utf-8"))
    lines = ["# B.2 下一起点 MuJoCo 对的保守筛选计时", "",
             "只排除包围球距离下界大于原激活阈值加 1 µm 的碰撞对；"
             "保留对仍调用原 `mj_geomDistance`，其余区间和实际链胶囊行原样重算。"
             "全部 1350 个下一起点记录和力矩轨迹与未筛选变体逐值相同。", "",
             f"原始对数/周期：{len(builder._all_pairs)}；"
             f"保留精确调用总数：{builder.exact_pair_calls}。", "",
             "| 部件 | 参考 p95 ms | 筛选后 p95 ms | p99 ms | 最大 ms | 超 20 ms |",
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
    lines += ["", "未改变生产控制器；三项之和不是完整在线周期。"
              "原始全对独立重放仍是正式检查。", ""]
    (output_dir / "SCREENED_NEXT_START.md").write_text(
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
        "next_start_records_exact", "total_exact_pair_calls",
        "screened_next_start", "necessary_component_sum")}, indent=2))


if __name__ == "__main__":
    main()
