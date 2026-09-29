"""Generate a hash-bound summary of the private capsule timing experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, frozen_dir: Path, adversarial_dir: Path,
        private_dir: Path, visual_dir: Path) -> Path:
    inputs = {
        "frozen": frozen_dir / "batched_capsule_summary.json",
        "adversarial": adversarial_dir / "adversarial_capsule_summary.json",
        "private": private_dir / "batched_capsule_private_timing_summary.json",
        "visual": visual_dir / "capsule_progress_visual_summary.json",
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8"))
               for key, path in inputs.items()}
    frozen, adversarial, private, visual = (reports[key] for key in inputs)
    if (frozen["state_count"] != 6755 or adversarial["case_count"] != 315
            or adversarial["near_five_mm_count"] != 126
            or frozen["source_mismatch_count"]
            or adversarial["source_mismatch_count"]
            or frozen["maximum_distance_error_m"] > 1e-12
            or adversarial["maximum_distance_error_m"] > 1e-12
            or not private["all_non_timing_records_exact"]
            or not private["same_native_67_channel_torque_trace"]
            or visual["stage2_20ms_gate_met"]):
        raise ValueError("capsule evidence contract failed")
    reference = private["reference_necessary_component_sum"]
    trial = private["trial_necessary_component_sum"]
    lines = [
        "# B.2 胶囊筛选与周期性能进展", "",
        "该私有试验沿用 A.1 的 50 Hz 任务周期，即每个周期 20 ms；"
        "这来自 `task_period_s=0.02` 的执行配置，不是 CBF 定理给出的时间。"
        "现有指标是必要计算项的计时合计，不包含完整生产在线调度。", "",
        "| 检查 | 结果 |", "| --- | ---: |",
        f"| 五场景冻结状态 | {frozen['state_count']} |",
        f"| 冻结状态来源不一致 | {frozen['source_mismatch_count']} |",
        f"| 胶囊查询原实现 p95 | {frozen['reference_timing']['p95_ms']:.3f} ms |",
        f"| 胶囊查询批量筛选 p95 | {frozen['batched_timing']['p95_ms']:.3f} ms |",
        f"| 对抗目标位置 | {adversarial['case_count']} |",
        f"| 5 mm 附近位置 | {adversarial['near_five_mm_count']} |",
        f"| 对抗位置来源不一致 | {adversarial['source_mismatch_count']} |",
        f"| 场景 01 私有周期 | {private['task_ticks']} |",
        f"| 参考必要计算和 p95 | {reference['p95_ms']:.3f} ms |",
        f"| 试验必要计算和 p95 / p99 / 最大 | {trial['p95_ms']:.3f} / "
        f"{trial['p99_ms']:.3f} / {trial['max_ms']:.3f} ms |",
        f"| 超 20 ms 的试验周期 | {trial['over_20ms_count']} |", "",
        "原 61 条胶囊、精确线段—OBB 胜出查询、安全半径与 67 路力矩"
        "保持一致。当前必要计算和 p95 仍超过 20 ms，B.2 第二阶段总门禁"
        "为 `GATE_NOT_MET`；不能据此切换生产在线 PCC 模式，也不构成"
        "连续时间安全或硬实时证明。", "",
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
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output_dir, args.frozen_dir, args.adversarial_dir,
              args.private_dir, args.visual_dir))


if __name__ == "__main__":
    main()
