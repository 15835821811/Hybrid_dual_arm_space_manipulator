"""Trial exact-input next-start decision reuse in the private B.2 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite.audit_b2_vectorized_positions_private_timing import run as vectorized_run
from v6_lite.b2_same_state_decision_cache import SameStateDecisionCache
from v6_lite.pcc_interval_cbf import NUMERICAL_PAD_M
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, vectorized_reference_dir: Path,
        batched_reference_dir: Path, compact_reference_dir: Path,
        screened_dir: Path, fixed_dir: Path, baseline_dir: Path,
        a1_root: Path, scenario_id: str, ticks: int) -> dict:
    original = PersistentIntervalDecisionQuery.evaluate
    cache = SameStateDecisionCache(original)

    def cached_evaluate(self, q, base_transform, target_box, partition,
                        *, gate_m=0.005, max_point_evaluations=64,
                        max_leaves=128,
                        numerical_pad_m=NUMERICAL_PAD_M):
        return cache.evaluate(
            self, q, base_transform, target_box, partition,
            gate_m=gate_m, max_point_evaluations=max_point_evaluations,
            max_leaves=max_leaves, numerical_pad_m=numerical_pad_m)

    try:
        PersistentIntervalDecisionQuery.evaluate = cached_evaluate
        trial = vectorized_run(
            output_dir, batched_reference_dir, compact_reference_dir,
            screened_dir, fixed_dir, baseline_dir, a1_root,
            scenario_id, ticks)
    finally:
        PersistentIntervalDecisionQuery.evaluate = original
    reference = json.loads((vectorized_reference_dir
                            / "vectorized_positions_private_timing_summary.json")
                           .read_text(encoding="utf-8"))
    diagnostic = {"cache_hits": cache.hit_count,
                  "cache_misses": cache.miss_count,
                  "mismatch_counts": cache.mismatch_counts,
                  "maximum_input_difference": cache.maximum_input_difference,
                  "all_non_timing_records_exact":
                      trial["all_non_timing_records_exact"],
                  "trace_sha_equal":
                      trial["inputs_sha256"]["trial_trace"]
                      == reference["inputs_sha256"]["trial_trace"]}
    (output_dir / "same_state_cache_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    if (not trial["all_non_timing_records_exact"]
            or not trial["same_native_67_channel_torque_trace"]
            or cache.hit_count < ticks - 1
            or trial["inputs_sha256"]["trial_trace"]
            != reference["inputs_sha256"]["trial_trace"]):
        raise ValueError(f"same-state query reuse not validated: {diagnostic}")
    report = {
        "schema": "v6_2_b2_same_state_decision_private_timing_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "all_non_timing_records_exact": True,
        "same_native_67_channel_torque_trace": True,
        "cache_hits": cache.hit_count,
        "cache_misses": cache.miss_count,
        "cache_hit_lookup_total_ms": cache.hit_lookup_ms,
        "cache_exact_input_key_includes": [
            "shape_model_identity", "10d_q", "base_transform",
            "target_obb_center_rotation_extents", "partition_leaves",
            "gate", "point_budget", "leaf_budget", "numerical_pad"],
        "decision_point_counts_on_cache_hits_are_original_provenance": True,
        "reference_necessary_component_sum":
            reference["trial_necessary_component_sum"],
        "trial_necessary_component_sum": trial["trial_necessary_component_sum"],
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_same_state_cache_private_timing.py",
                              "audit_b2_vectorized_positions_private_timing.py",
                              "b2_same_state_decision_cache.py",
                              "pcc_persistent_interval_query.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                vectorized_reference_dir / "private_rollout_records.jsonl"),
            "reference_trace": _sha(
                vectorized_reference_dir / "private_rollout_trace.npz"),
            "trial_records": _sha(output_dir / "private_rollout_records.jsonl"),
            "trial_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
    }
    (output_dir / "same_state_cache_private_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    before = report["reference_necessary_component_sum"]
    after = report["trial_necessary_component_sum"]
    lines = ["# B.2 相邻周期同状态区间决策复用", "",
             "仅当形变状态、基座变换、目标 OBB、区间集合、门槛和全部预算逐值相同，"
             "才复用上周期已计算的下一起点决策。失配时调用原区间查询。"
             "缓存命中记录中的点查询次数保留原决策的来源计数，"
             "不代表本周期重新执行了这些查询。", "",
             f"缓存命中 {cache.hit_count} 次，未命中 {cache.miss_count} 次。"
             "1350 周期的非计时记录与 67 路力矩 trace 逐值相同。", "",
             "| 指标 | 参考 | 同状态复用 |",
             "| --- | ---: | ---: |",
             f"| 必要计算和 p95 ms | {before['p95_ms']:.3f} | "
             f"{after['p95_ms']:.3f} |",
             f"| 必要计算和 p99 ms | {before['p99_ms']:.3f} | "
             f"{after['p99_ms']:.3f} |",
             f"| 必要计算和最大 ms | {before['max_ms']:.3f} | "
             f"{after['max_ms']:.3f} |",
             f"| 超 20 ms 周期 | {before['over_20ms_count']} | "
             f"{after['over_20ms_count']} |", "",
             "该试验没有测完整生产在线调度；阶段二总门禁与第三阶段接入仍须"
             "按完整要求审核。", ""]
    (output_dir / "SAME_STATE_CACHE_PRIVATE_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vectorized-reference-dir", type=Path, required=True)
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
    report = run(args.output_dir, args.vectorized_reference_dir,
                 args.batched_reference_dir, args.compact_reference_dir,
                 args.screened_dir, args.fixed_dir, args.baseline_dir,
                 args.a1_root, args.scenario_id, args.ticks)
    print(json.dumps({key: report[key] for key in (
        "cache_hits", "cache_misses", "all_non_timing_records_exact",
        "reference_necessary_component_sum", "trial_necessary_component_sum")},
        indent=2))


if __name__ == "__main__":
    main()
