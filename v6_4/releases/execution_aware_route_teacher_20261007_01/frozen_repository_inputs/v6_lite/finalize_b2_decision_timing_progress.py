"""Regenerate the six-stage private timing plot from saved tick rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v6_lite.finalize_b2_full_decision_timing import _load_trial


SPECS = (
    ("original", "原始", "full_decision_timing_summary.json",
     "full_decision_timing_rows.jsonl", "necessary_decision_component_sum",
     "necessary_decision_components_ms"),
    ("reused_builder", "常量复用", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
    ("vectorized_envelope", "包络批量", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
    ("fixed_arc_positions", "固定弧长", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
    ("screened_next_start", "保守筛选", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
    ("prepared_torque", "前向复用", "reused_next_start_timing_summary.json",
     "reused_next_start_timing_rows.jsonl", "necessary_component_sum",
     "necessary_component_sum_ms"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, folders: tuple[Path, ...],
        parity_dir: Path, five_gate_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(folders) != len(SPECS):
        raise ValueError("six fixed private timing stages required")
    trials = [_load_trial(folder, spec) for folder, spec in zip(folders, SPECS)]
    if len({item["trace_sha256"] for item in trials}) != 1:
        raise ValueError("paired private torque traces changed")
    fixed_path = folders[3] / "fixed_arc_full_decision_summary.json"
    screened_path = folders[4] / "screened_next_start_summary.json"
    prepared_path = folders[5] / "prepared_torque_timing_summary.json"
    parity_path = parity_dir / "fixed_arc_all_state_summary.json"
    five_path = five_gate_dir / "domain_private_five_summary.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    screened = json.loads(screened_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    five = json.loads(five_path.read_text(encoding="utf-8"))
    if not (fixed["reference_record_decisions_and_envelope_margins_exact"]
            and screened["next_start_records_exact"]
            and prepared["all_non_timing_records_exact"]
            and parity["all_results_exact"]
            and parity["checked_2ms_states"] == 13501
            and parity["capsules_per_state"] == 61
            and five["private_diagnostic_status"]
            == "PASS_PRIVATE_FIVE_WITH_LIMITS"
            and five["stage2_online_gate_status"] == "GATE_NOT_MET"):
        raise ValueError("private parity or five-scene gate changed")
    report = {
        "schema": "v6_2_b2_six_stage_private_decision_timing_v1",
        "scenario_id": "v6_lite_scenario_01",
        "task_ticks_per_stage": 1350,
        "all_six_traces_sha256_equal": True,
        "fixed_arc_all_2ms_capsule_results_exact": True,
        "fixed_arc_checked_2ms_states": 13501,
        "fixed_arc_capsules_per_state": 61,
        "five_scene_private_diagnostic_status": five[
            "private_diagnostic_status"],
        "stage2_online_gate_status": "GATE_NOT_MET",
        "stage3_admission": False,
        "full_online_cycle_evaluated": False,
        "p95_20ms_gate_met": False,
        "performance_repeat_protocol_completed": False,
        "production_online_controller_changed": False,
        "continuous_time_certified": False,
        "trials": trials,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "finalize_b2_decision_timing_progress.py",
                              "finalize_b2_full_decision_timing.py")},
        "additional_inputs_sha256": {
            "fixed_arc_full_decision_summary.json": _sha(fixed_path),
            "screened_next_start_summary.json": _sha(screened_path),
            "prepared_torque_timing_summary.json": _sha(prepared_path),
            "fixed_arc_all_state_summary.json": _sha(parity_path),
            "domain_private_five_summary.json": _sha(five_path),
        },
    }
    plot_path = output_dir / "b2-decision-timing-progress.png"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor("white")
    colors = ["#6787aa", "#d48a42", "#37918c", "#6a8c65", "#396bb3",
              "#8f6b9b"]
    for index, item in enumerate(trials):
        y = 5 - index
        ax.plot([item["p50_ms"], item["p99_ms"]], [y, y],
                color=colors[index], linewidth=5, solid_capstyle="round")
        ax.scatter([item["p50_ms"], item["p99_ms"]], [y, y],
                   color=colors[index], s=29, zorder=3)
        ax.scatter([item["p95_ms"]], [y], color=colors[index],
                   s=88, zorder=4)
        ax.scatter([item["max_ms"]], [y], facecolors="none",
                   edgecolors=colors[index], s=37, zorder=3)
        ax.text(item["p95_ms"] + 1.7, y + .12,
                f"p95 {item['p95_ms']:.1f} ms", color=colors[index],
                fontsize=9, weight="medium")
    ax.axvline(20, color="#b54040", linestyle="--", linewidth=1.5)
    ax.text(21.1, 5.48, "20 ms 周期", color="#b54040", fontsize=9)
    ax.set_yticks([5, 4, 3, 2, 1, 0],
                  [item["label"] for item in trials])
    ax.set_xlim(0, max(item["max_ms"] for item in trials) + 5)
    ax.set_ylim(-.55, 5.7)
    ax.set_xlabel("三项必要决策计算之和 (ms)，场景 01 / 每阶段 1350 周期")
    ax.set_title("B.2 私有决策计时进展：轨迹相同，20 ms 门禁仍未通过")
    ax.grid(axis="x", color="#e2e6ea", linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.5, .005,
             "线段 = p50–p99；实心大点 = p95；空心点 = 最大值。非完整在线时延。",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(plot_path)
    plt.close(fig)
    report["plot_sha256"] = _sha(plot_path)
    (output_dir / "decision_timing_progress_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 六阶段私有决策计时进展", "",
             "六阶段在场景 01 各运行 1350 个周期，原生力矩 trace SHA-256 相同。"
             "计时只包括预检＋QP、十步候选分支和下一起点重算；"
             "它们的逐周期和仍不是完整在线全链时延。", "",
             "| 私有变体 | p50 ms | p95 ms | p99 ms | 最大 ms | 超 20 ms | 最长连续超期 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in trials:
        lines.append(f"| {item['label']} | {item['p50_ms']:.3f} | "
                     f"{item['p95_ms']:.3f} | {item['p99_ms']:.3f} | "
                     f"{item['max_ms']:.3f} | {item['over_20ms_count']} | "
                     f"{item['longest_over_20ms_run']} |")
    lines += ["", "![六阶段必要决策计时](b2-decision-timing-progress.png)", "",
              "固定弧长变体在 13,501 个保存的 2 ms 状态上逐个核对了"
              "每状态 61 个实际链胶囊的完整结果，零不一致。"
              "保守筛选变体的全部下一起点记录也与全对重算逐值相同。", "",
              "五场景私有闭环诊断通过，但阶段二在线门禁仍为 `GATE_NOT_MET`。"
              "没有执行正式性能重复协议、完整在线全链时延或新区间模式的原 26/11 验收；"
              "生产在线控制器未切换。", "",
              "本报告由保存的逐周期行自动生成，摘要绑定源码、输入和图像 SHA-256。", ""]
    (output_dir / "DECISION_TIMING_PROGRESS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reused", type=Path, required=True)
    parser.add_argument("--vectorized", type=Path, required=True)
    parser.add_argument("--fixed-arc", type=Path, required=True)
    parser.add_argument("--screened", type=Path, required=True)
    parser.add_argument("--prepared-torque", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--five-gate", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir, (args.original, args.reused, args.vectorized,
                                   args.fixed_arc, args.screened,
                                   args.prepared_torque),
                 args.parity_dir, args.five_gate)
    print(json.dumps({"gate": report["stage2_online_gate_status"],
                      "p95_ms": [item["p95_ms"] for item in report["trials"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
