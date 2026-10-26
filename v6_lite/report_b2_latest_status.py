"""Generate the current B.2 stage-2 status from immutable evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, geometry_dir: Path, repeat_dir: Path,
        aabb_dir: Path, qp_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "geometry": geometry_dir / "capsule_progress_visual_summary.json",
        "repeats": repeat_dir / "forward_reuse_repeats_summary.json",
        "aabb": aabb_dir / "aabb_screen_visual_summary.json",
        "qp": qp_dir / "latest_qp_breakdown_summary.json",
    }
    evidence = {key: json.loads(path.read_text(encoding="utf-8"))
                for key, path in inputs.items()}
    geometry, repeats, aabb, qp = (evidence[key] for key in inputs)
    if (geometry["stage2_20ms_gate_met"]
            or repeats["stage2_full_online_20ms_gate_met"]
            or not repeats["all_native_67_channel_torque_traces_equal"]
            or not repeats["all_non_timing_records_exact"]
            or aabb["missed_original_active_pair_count"]
            or aabb["adopted_for_private_controller"]
            or not qp["all_non_timing_records_exact"]
            or not qp["trajectory_exact_to_reference"]):
        raise ValueError("latest B.2 evidence contract failed")
    latest_p95 = [item["necessary_component_sum"]["p95_ms"]
                  for item in repeats["runs"][1:4]]
    qp_metrics = qp["metrics"]
    report = {
        "schema": "v6_2_b2_latest_status_v1",
        "stage": "B.2-2_SHADOW_AND_PERFORMANCE",
        "stage2_20ms_full_online_gate": "GATE_NOT_MET",
        "stage3_online_mode_enabled": False,
        "private_necessary_component_p95_repeat_range_ms": [
            min(latest_p95), max(latest_p95)],
        "latest_qp_clearance_assembly_p95_ms": qp_metrics[
            "clearance_assembly_ms"]["p95_ms"],
        "latest_qp_solver_latency_p95_ms": qp_metrics[
            "solver_latency_ms"]["p95_ms"],
        "aabb_trial_adopted": False,
        "inputs_sha256": {key: _sha(path) for key, path in inputs.items()},
    }
    (output_dir / "latest_status_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    lines = ["# V6.2-B.2 最新状态", "",
             "当前处于 **B.2-2：离线影子与性能评估**；B.2-1 的固定区间安全函数、"
             "完整梯度和覆盖测试已建立。B.2-3 开关式生产在线接入尚未开始。", "",
             "| 证据 | 最新结果 |", "| --- | ---: |",
             f"| 私有场景 01 三轮必要计算和 p95 范围 | "
             f"{min(latest_p95):.3f}～{max(latest_p95):.3f} ms |",
             f"| QP 约束行装配 p95 | "
             f"{qp_metrics['clearance_assembly_ms']['p95_ms']:.3f} ms |",
             f"| QP 求解循环 p95 | "
             f"{qp_metrics['solver_latency_ms']['p95_ms']:.3f} ms |",
             f"| MuJoCo 行 p95 | "
             f"{qp_metrics['mujoco_rows_ms']['p95_ms']:.3f} ms |",
             f"| PCC 区间求值 p95 | "
             f"{qp_metrics['interval_evaluation_ms']['p95_ms']:.3f} ms |",
             f"| 胶囊行 p95 | "
             f"{qp_metrics['capsule_rows_ms']['p95_ms']:.3f} ms |",
             f"| AABB 筛选相对包围球的查询 p95 | "
             f"{aabb['screen_plus_exact_p95_ms'][0]:.3f} → "
             f"{aabb['screen_plus_exact_p95_ms'][1]:.3f} ms |", "",
             "20 ms 来自原 50 Hz 任务周期配置；正式性能门槛要求完整在线链路"
             " p95 ≤20 ms。上述重复值仅为私有必要计算项合计，"
             "尚缺完整在线调度。QP 子项彼此嵌套且分位数独立计算，不可相加。"
             "第二阶段总门禁为 `GATE_NOT_MET`；不切换生产在线模式。", "",
             "## 最新图表", "",
             "![下一起点复用三轮计时]"
             "(../forward_reuse_repeats_report/forward-reuse-repeats.png)", "",
             "![AABB 筛选反例]"
             "(../aabb_screen_visual/aabb-screen-comparison.png)", "",
             "早先胶囊与 PCC 批量计算进展见 "
             "[几何进展图](../current_geometry_progress_visual/b2-capsule-timing-progress.png)。", "",
             "## 输入 SHA-256", ""]
    lines += [f"- `{key}`: `{_sha(path)}`" for key, path in inputs.items()]
    lines += [""]
    (output_dir / "LATEST_STATUS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--aabb-dir", type=Path, required=True)
    parser.add_argument("--qp-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.geometry_dir,
                         args.repeat_dir, args.aabb_dir, args.qp_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
