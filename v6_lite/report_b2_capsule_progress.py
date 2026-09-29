"""Generate a hash-bound summary of the private capsule timing experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, frozen_dir: Path, adversarial_dir: Path,
        private_dir: Path, vectorized_dir: Path, positions_dir: Path,
        cache_failed_dir: Path, squared_dir: Path, cross_impl_dir: Path,
        visual_dir: Path) -> Path:
    inputs = {
        "frozen": frozen_dir / "batched_capsule_summary.json",
        "adversarial": adversarial_dir / "adversarial_capsule_summary.json",
        "private": private_dir / "batched_capsule_private_timing_summary.json",
        "vectorized": vectorized_dir
        / "vectorized_positions_private_timing_summary.json",
        "positions": positions_dir / "fixed_arc_position_summary.json",
        "cache_failed": cache_failed_dir / "same_state_cache_diagnostic.json",
        "squared": squared_dir / "squared_envelope_summary.json",
        "cross_impl": cross_impl_dir / "private_query_cross_impl_summary.json",
        "visual": visual_dir / "capsule_progress_visual_summary.json",
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8"))
               for key, path in inputs.items()}
    frozen, adversarial, private, vectorized, positions, cache_failed, squared, cross_impl, visual = (
        reports[key] for key in inputs)
    if (frozen["state_count"] != 6755 or adversarial["case_count"] != 315
            or adversarial["near_five_mm_count"] != 126
            or frozen["source_mismatch_count"]
            or adversarial["source_mismatch_count"]
            or frozen["maximum_distance_error_m"] > 1e-12
            or adversarial["maximum_distance_error_m"] > 1e-12
            or not private["all_non_timing_records_exact"]
            or not private["same_native_67_channel_torque_trace"]
            or not vectorized["all_non_timing_records_exact"]
            or not vectorized["same_native_67_channel_torque_trace"]
            or positions["maximum_absolute_position_error_m"] > 1e-12
            or cache_failed["cache_hits"] != 0
            or not cache_failed["all_non_timing_records_exact"]
            or not cache_failed["trace_sha_equal"]
            or squared["status_or_capsule_count_mismatches"]
            or squared["maximum_margin_error_m"] > 1e-12
            or cross_impl["state_count"] != 1351
            or cross_impl["decision_partition_or_status_mismatch_count"]
            or cross_impl["maximum_bound_error_m"] > 1e-12
            or visual["stage2_20ms_gate_met"]):
        raise ValueError("capsule evidence contract failed")
    reference = private["reference_necessary_component_sum"]
    trial = private["trial_necessary_component_sum"]
    vectorized_trial = vectorized["trial_necessary_component_sum"]
    lines = [
        "# B.2 几何批量计算与周期性能进展", "",
        "该私有试验沿用 A.1 的 50 Hz 任务周期，即每个周期 20 ms；"
        "这来自 `task_period_s=0.02` 的执行配置，不是 CBF 定理给出的时间。"
        "现有指标是必要计算项的计时合计，不包含完整生产在线调度。", "",
        "| 检查 | 结果 |", "| --- | ---: |",
        f"| 五场景冻结状态 | {frozen['state_count']} |",
        f"| 冻结状态来源不一致 | {frozen['source_mismatch_count']} |",
        f"| 胶囊查询原实现 p95 | {frozen['reference_timing']['p95_ms']:.3f} ms |",
        f"| 胶囊查询批量筛选 p95 | {frozen['batched_timing']['p95_ms']:.3f} ms |",
        f"| PCC 位置原实现 p95 | {positions['reference_timing']['p95_ms']:.3f} ms |",
        f"| PCC 位置批量计算 p95 | {positions['trial_timing']['p95_ms']:.3f} ms |",
        f"| 对抗目标位置 | {adversarial['case_count']} |",
        f"| 5 mm 附近位置 | {adversarial['near_five_mm_count']} |",
        f"| 对抗位置来源不一致 | {adversarial['source_mismatch_count']} |",
        f"| 场景 01 私有周期 | {private['task_ticks']} |",
        f"| 参考必要计算和 p95 | {reference['p95_ms']:.3f} ms |",
        f"| 试验必要计算和 p95 / p99 / 最大 | {trial['p95_ms']:.3f} / "
        f"{trial['p99_ms']:.3f} / {trial['max_ms']:.3f} ms |",
        f"| 超 20 ms 的试验周期 | {trial['over_20ms_count']} |",
        f"| 叠加五段批量计算后的必要计算和 p95 / p99 / 最大 | "
        f"{vectorized_trial['p95_ms']:.3f} / {vectorized_trial['p99_ms']:.3f} / "
        f"{vectorized_trial['max_ms']:.3f} ms |",
        f"| 叠加后超 20 ms 周期 | {vectorized_trial['over_20ms_count']} |", "",
        f"两种私有区间查询实现于 {cross_impl['state_count']} 个场景 01 状态上"
        f"得到相同决策分区和状态；距离界最大差为"
        f" {cross_impl['maximum_bound_error_m']:.3e} m。"
        "这仅是两种实现的有限状态对照，不能视为生产缓存认证。", "",
        "原 61 条胶囊、精确线段—OBB 胜出查询、安全半径与 67 路力矩"
        "保持一致。当前必要计算和 p95 仍超过 20 ms，B.2 第二阶段总门禁"
        "为 `GATE_NOT_MET`；不能据此切换生产在线 PCC 模式，也不构成"
        "连续时间安全或硬实时证明。", "",
        "## 未采用的性能试验", "",
        "| 试验 | 观察 | 决定 |", "| --- | --- | --- |",
        f"| 同状态区间决策缓存 | {cache_failed['cache_hits']} 次命中 / "
        f"{cache_failed['cache_misses']} 次查询 | 当前实现跨两种查询类，"
        "单类缓存无效；不计入提速 |",
        f"| 胶囊包络先规约平方距离 | p95 "
        f"{squared['reference_timing']['p95_ms']:.3f} → "
        f"{squared['trial_timing']['p95_ms']:.3f} ms | 提速不足；不接入 |", "",
        "![批量胶囊筛选与周期计时](b2-capsule-timing-progress.png)", "",
        "## 输入 SHA-256", "",
    ]
    lines += [f"- `{name}`: `{_sha(path)}`"
              for name, path in inputs.items()]
    lines += [f"- `image`: `{_sha(visual_dir / 'b2-capsule-timing-progress.png')}`",
              ""]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "CAPSULE_PROGRESS.md"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--adversarial-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--vectorized-dir", type=Path, required=True)
    parser.add_argument("--positions-dir", type=Path, required=True)
    parser.add_argument("--cache-failed-dir", type=Path, required=True)
    parser.add_argument("--squared-dir", type=Path, required=True)
    parser.add_argument("--cross-impl-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output_dir, args.frozen_dir, args.adversarial_dir,
              args.private_dir, args.vectorized_dir, args.positions_dir,
              args.cache_failed_dir, args.squared_dir, args.cross_impl_dir,
              args.visual_dir))


if __name__ == "__main__":
    main()
