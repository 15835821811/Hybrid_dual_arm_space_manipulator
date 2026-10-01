"""Summarize immutable private next-start forward-reuse trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIMING_KEYS = {"qp_full_ms", "preflight_plus_qp_ms",
               "independent_row_recompute_ms"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [{key: value for key, value in json.loads(line).items()
                 if key not in TIMING_KEYS} for line in stream]


def run(output_dir: Path, baseline_dir: Path,
        retained_dirs: tuple[Path, ...], unadopted_dir: Path) -> dict:
    if len(retained_dirs) != 3:
        raise ValueError("exactly three predeclared retained trials required")
    output_dir.mkdir(parents=True, exist_ok=False)
    ordered = (("prior private", baseline_dir),
               *((f"one forward {i}", path)
                 for i, path in enumerate(retained_dirs, 1)),
               ("zero forward trial", unadopted_dir))
    baseline_rows = _read_rows(baseline_dir / "private_rollout_records.jsonl")
    baseline_trace_sha = _sha(baseline_dir / "private_rollout_trace.npz")
    if len(baseline_rows) != 1350:
        raise ValueError("baseline is not a complete 1350 tick trace")
    summaries = []
    for label, directory in ordered:
        summary_path = directory / "vectorized_positions_private_timing_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records_path = directory / "private_rollout_records.jsonl"
        trace_path = directory / "private_rollout_trace.npz"
        rows = _read_rows(records_path)
        if (summary["task_ticks"] != 1350
                or not summary["all_non_timing_records_exact"]
                or not summary["same_native_67_channel_torque_trace"]
                or rows != baseline_rows
                or _sha(trace_path) != baseline_trace_sha):
            raise ValueError(f"non-timing or native trace parity failed: {label}")
        screened_path = directory / "screened_next_start_summary.json"
        screened = json.loads(screened_path.read_text(encoding="utf-8"))
        summaries.append({
            "label": label,
            "directory": directory.as_posix(),
            "necessary_component_sum": summary["trial_necessary_component_sum"],
            "screen_source_sha256": screened["source_sha256"][
                "b2_screened_next_start_rows.py"],
            "summary_sha256": _sha(summary_path),
            "records_sha256": _sha(records_path),
            "trace_sha256": _sha(trace_path),
        })
    current_screen_sha = _source_sha(
        Path("v6_lite/b2_screened_next_start_rows.py"))
    if (any(item["screen_source_sha256"] != current_screen_sha
            for item in summaries[1:4])
            or summaries[-1]["screen_source_sha256"] == current_screen_sha):
        raise ValueError("retained and unadopted source variants confused")
    p95 = [item["necessary_component_sum"]["p95_ms"] for item in summaries]
    private_p95_all_over_20ms = all(value > 20.0 for value in p95)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars = ax.bar(np.arange(len(ordered)), p95,
                  color=["#64748b", "#0f766e", "#0f766e", "#0f766e",
                         "#c2410c"], width=.67)
    ax.axhline(20.0, color="#dc2626", linestyle="--", linewidth=1.5)
    for bar, item in zip(bars, summaries):
        value = item["necessary_component_sum"]["p95_ms"]
        misses = item["necessary_component_sum"]["over_20ms_count"]
        ax.text(bar.get_x() + bar.get_width() / 2, value + .025,
                f"{value:.2f} ms\n{misses}/1350 >20 ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(ordered)), [item[0] for item in ordered])
    ax.set_ylim(19.7, max(p95) + .35)
    ax.set_ylabel("Necessary component sum p95 (ms)")
    ax.set_title("Scene 01 private repeats: next-start forward reuse",
                 loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.10, .005,
             "All non-timing records and 67-channel torque traces match. "
             "Private necessary sums exclude full online scheduling.",
             fontsize=8.5, color="#334155")
    fig.tight_layout(rect=(0, .045, 1, 1))
    image_path = output_dir / "forward-reuse-repeats.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = {
        "schema": "v6_2_b2_forward_reuse_repeat_report_v1",
        "scenario_id": "v6_lite_scenario_01",
        "ticks_per_run": 1350,
        "all_non_timing_records_exact": True,
        "all_native_67_channel_torque_traces_equal": True,
        "retained_variant": "one_mj_forward_in_screened_next_start_builder",
        "unadopted_variant": "zero_mj_forward_uses_branch_prepared_state",
        "private_necessary_p95_all_over_20ms": private_p95_all_over_20ms,
        "stage2_full_online_20ms_gate_met": False,
        "runs": summaries,
        "source_sha256": {
            "report_b2_forward_reuse_repeats.py": _source_sha(Path(__file__)),
            "b2_screened_next_start_rows.py": current_screen_sha,
            "recompute_execution_constraints.py": _source_sha(
                Path("v6_lite/recompute_execution_constraints.py")),
            "image": _sha(image_path),
        },
    }
    (output_dir / "forward_reuse_repeats_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 下一起点状态更新复用的重复计时", "",
             "私有场景 01；各轮均为完整 1350 个 50 Hz 任务周期。"
             "去掉下一起点构造器内部第二次 `mj_forward`，独立重放构造器"
             "对任意输入状态仍自行调用一次。", "",
             "| 试验 | 必要计算和 p50 / p95 / p99 / 最大 ms | 超 20 ms 周期 |",
             "| --- | ---: | ---: |"]
    for item in summaries:
        timing = item["necessary_component_sum"]
        lines.append(
            f"| {item['label']} | {timing['p50_ms']:.3f} / "
            f"{timing['p95_ms']:.3f} / {timing['p99_ms']:.3f} / "
            f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} / 1350 |")
    lines += ["", "三轮保留变体的 p95 范围为 "
              f"{min(p95[1:4]):.3f}～{max(p95[1:4]):.3f} ms。"
              "零次状态更新变体依赖候选分支已更新 MuJoCo 状态，"
              "本轮计时没有显示稳定额外收益，故未采用。", "",
              "全部非计时记录与原生 67 路力矩轨迹逐值一致。"
              "这只是必要计算项合计，不包含完整在线调度；"
              + ("所有轮次 p95 仍超过 20 ms；"
                 if private_p95_all_over_20ms else
                 "部分必要计算 p95 不超过 20 ms；")
              + "B.2 第二阶段完整在线性能门禁仍未通过。", "",
              "![重复计时](forward-reuse-repeats.png)", "",
              "## SHA-256", ""]
    for item in summaries:
        lines.append(f"- `{item['label']} summary`: `{item['summary_sha256']}`")
        lines.append(f"- `{item['label']} records`: `{item['records_sha256']}`")
        lines.append(f"- `{item['label']} trace`: `{item['trace_sha256']}`")
    lines += [f"- `image`: `{_sha(image_path)}`", ""]
    (output_dir / "FORWARD_REUSE_REPEATS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--retained-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--unadopted-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir, args.baseline_dir,
                 tuple(args.retained_dirs), args.unadopted_dir)
    print(json.dumps({"ticks_per_run": report["ticks_per_run"],
                      "p95_ms": [item["necessary_component_sum"]["p95_ms"]
                                 for item in report["runs"]],
                      "stage2_full_online_20ms_gate_met": report[
                          "stage2_full_online_20ms_gate_met"]}, indent=2))


if __name__ == "__main__":
    main()
