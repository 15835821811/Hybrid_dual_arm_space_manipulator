"""Generate hash-bound B.2 private full-decision timing metrics and plot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PERIOD_MS = 20.0
TRIALS = (
    ("original", "原始分支", "full_decision_timing_summary.json",
     "full_decision_timing_rows.jsonl", "necessary_decision_component_sum",
     "necessary_decision_components_ms"),
    ("reused_builder", "复用模型常量", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
    ("vectorized_envelope", "批量包络", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _longest_overrun(values: np.ndarray) -> int:
    longest = current = 0
    for over in values > PERIOD_MS:
        current = current + 1 if over else 0
        longest = max(longest, current)
    return longest


def _load_trial(folder: Path, spec: tuple[str, ...]) -> dict:
    key, label, summary_name, rows_name, sum_key, row_key = spec
    summary_path = folder / summary_name
    rows_path = folder / rows_name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(
        encoding="utf-8").splitlines()]
    values = np.asarray([row[row_key] for row in rows], dtype=np.float64)
    if (len(values) != 1350 or not np.all(np.isfinite(values))
            or summary["rows_sha256"] != _sha(rows_path)
            or abs(float(np.percentile(values, 95))
                   - summary[sum_key]["p95_ms"]) > 1e-9):
        raise ValueError(f"invalid full-decision timing: {folder}")
    return {
        "key": key, "label": label, "task_ticks": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "over_20ms_count": int(np.count_nonzero(values > PERIOD_MS)),
        "longest_over_20ms_run": _longest_overrun(values),
        "trace_sha256": _sha(folder / "private_rollout_trace.npz"),
        "input_sha256": {
            summary_name: _sha(summary_path),
            rows_name: _sha(rows_path),
            "private_rollout_summary.json": _sha(
                folder / "private_rollout_summary.json"),
            "private_rollout_trace.npz": _sha(
                folder / "private_rollout_trace.npz"),
        },
    }


def run(output_dir: Path, original: Path, reused: Path,
        vectorized: Path, five_gate: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    folders = (original, reused, vectorized)
    trials = [_load_trial(folder, spec) for folder, spec in zip(folders, TRIALS)]
    if len({item["trace_sha256"] for item in trials}) != 1:
        raise ValueError("timing variants changed the native torque trace")
    vectorized_report_path = vectorized / "vectorized_full_decision_summary.json"
    vectorized_report = json.loads(vectorized_report_path.read_text(
        encoding="utf-8"))
    if not (vectorized_report["reference_record_decisions_exact"]
            and vectorized_report["trajectory_exact_to_original_baseline"]
            and vectorized_report["maximum_envelope_margin_difference_m"]
            <= 1e-12):
        raise ValueError("vectorized run lacks full decision parity")
    five_path = five_gate / "domain_private_five_summary.json"
    five = json.loads(five_path.read_text(encoding="utf-8"))
    if (five["private_diagnostic_status"] != "PASS_PRIVATE_FIVE_WITH_LIMITS"
            or five["stage2_online_gate_status"] != "GATE_NOT_MET"):
        raise ValueError("five-scene gate status changed")
    report = {
        "schema": "v6_2_b2_private_full_decision_timing_comparison_v1",
        "scenario_id": "v6_lite_scenario_01",
        "five_scene_private_diagnostic_status": five[
            "private_diagnostic_status"],
        "stage2_online_gate_status": "GATE_NOT_MET",
        "stage3_admission": False,
        "production_online_controller_changed": False,
        "full_online_cycle_evaluated": False,
        "component_sum_is_full_online_latency": False,
        "p95_20ms_gate_met": False,
        "continuous_time_certified": False,
        "performance_repeat_protocol_completed": False,
        "task_period_ms": PERIOD_MS,
        "original_26_11_online_acceptance_replayed_here": False,
        "original_26_11_historical_artifacts_referenced": True,
        "trials": trials,
        "source_sha256": {
            "finalize_b2_full_decision_timing.py": _source_sha(
                Path("v6_lite/finalize_b2_full_decision_timing.py")),
            "run_v6_lite.py": _source_sha(Path("v6_lite/run_v6_lite.py")),
        },
        "additional_inputs_sha256": {
            "vectorized_full_decision_summary.json": _sha(
                vectorized_report_path),
            "domain_private_five_summary.json": _sha(five_path),
        },
    }
    plot_path = output_dir / "b2-full-decision-timing.png"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=160)
    fig.patch.set_facecolor("white")
    markers = ["o", "s", "D"]
    colors = ["#5276a1", "#cf8439", "#268a80"]
    for index, item in enumerate(trials):
        y = 2 - index
        ax.plot([item["p50_ms"], item["p99_ms"]], [y, y],
                color=colors[index], linewidth=5, solid_capstyle="round",
                label=item["label"])
        ax.scatter([item["p50_ms"], item["p95_ms"], item["p99_ms"]],
                   [y, y, y], color=colors[index], marker=markers[index],
                   s=[32, 90, 32], zorder=3)
        ax.text(item["p95_ms"] + 1.8, y + 0.14,
                f"p95 {item['p95_ms']:.1f} ms", color=colors[index],
                fontsize=10, weight="medium")
        ax.annotate(f"max {item['max_ms']:.1f}",
                    (item["max_ms"], y), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#424a55")
        ax.scatter([item["max_ms"]], [y], facecolors="none",
                   edgecolors=colors[index], s=38, zorder=3)
    ax.axvline(PERIOD_MS, color="#b54040", linestyle="--", linewidth=1.5)
    ax.text(PERIOD_MS + 1.2, 2.47, "20 ms 周期", color="#b54040", fontsize=9)
    ax.set_yticks([2, 1, 0], [item["label"] for item in trials])
    ax.set_xlim(0, max(item["max_ms"] for item in trials) + 8)
    ax.set_ylim(-0.55, 2.7)
    ax.set_xlabel("三项必要决策计算之和 (ms)，场景 01 / 1350 周期")
    ax.set_title("B.2 私有决策计时：相同力矩轨迹，仍未满足 20 ms")
    ax.grid(axis="x", color="#e2e6ea", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.5, .005, "线段 = p50–p99；实心大点 = p95；空心点 = 最大值。非完整在线时延。",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(plot_path)
    plt.close(fig)
    report["plot_sha256"] = _sha(plot_path)
    report_path = output_dir / "full_decision_timing_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    lines = ["# B.2 完整决策必要计算的私有计时", "",
             "场景 01 的 1350 个周期，三次变体的力矩 trace SHA-256 相同。"
             "计时包括预检＋QP、十步 MuJoCo 候选分支及下一起点重算；"
             "不包括诊断用独立行核对，也不等于生产在线全链时延。", "",
             "| 变体 | p50 ms | p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 | 最长连续超期 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in trials:
        lines.append(f"| {item['label']} | {item['p50_ms']:.3f} | "
                     f"{item['p95_ms']:.3f} | {item['p99_ms']:.3f} | "
                     f"{item['max_ms']:.3f} | {item['over_20ms_count']} | "
                     f"{item['longest_over_20ms_run']} |")
    lines += ["", "![三项必要决策计时](b2-full-decision-timing.png)", "",
              "五场景私有力矩诊断已通过，但阶段二在线门禁仍为 `GATE_NOT_MET`。"
              "本次只是场景 01 的一次配对计时，未完成预声明重复轮次、"
              "完整在线全链计时或原 26/11 正式重放。"
              "不接入生产控制器，不声称连续时间安全证明。", "",
              "全部数字由保存的逐周期计时行生成；摘要记录了源码、输入和图像的 SHA-256。", ""]
    (output_dir / "FULL_DECISION_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reused", type=Path, required=True)
    parser.add_argument("--vectorized", type=Path, required=True)
    parser.add_argument("--five-gate", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir, args.original, args.reused,
                 args.vectorized, args.five_gate)
    print(json.dumps({"stage2_online_gate_status": result[
        "stage2_online_gate_status"],
        "trials": [{key: row[key] for key in (
            "key", "p95_ms", "over_20ms_count")}
            for row in result["trials"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
