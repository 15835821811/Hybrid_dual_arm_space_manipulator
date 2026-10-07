"""Visualize an immutable snapshot of the opt-in B.2 online run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(run_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = sorted((run_dir / "traces").glob("v6_lite_scenario_*.npz"))
    if not paths:
        raise ValueError("no completed scenario trace is available")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            keys = set(trace.files)
            required = {"torque", "task_full_latency",
                        "task_interval_preflight_latency_s",
                        "task_interval_current_envelope_margin_m",
                        "task_interval_ramp_minimum_envelope_margin_m",
                        "task_interval_realized_next_start_minimum_slack_m_s",
                        "task_interval_selected_rows", "task_failure_reason"}
            if not required <= keys:
                raise ValueError(f"missing interval trace fields: {path}")
            full_available = "task_interval_full_control_latency_s" in keys
            qp_ms = np.asarray(trace[
                "task_interval_qp_only_latency_s" if full_available
                else "task_full_latency"]) * 1000
            preflight_ms = np.asarray(
                trace["task_interval_preflight_latency_s"]) * 1000
            full_ms = (np.asarray(trace["task_interval_full_control_latency_s"])
                       * 1000 if full_available else None)
            records.append({
                "scenario_id": path.stem,
                "task_ticks": int(len(qp_ms)),
                "torque_steps": int(len(trace["torque"])),
                "qp_only_p95_ms": float(np.percentile(qp_ms, 95)),
                "qp_only_p99_ms": float(np.percentile(qp_ms, 99)),
                "qp_only_max_ms": float(np.max(qp_ms)),
                "preflight_p95_ms": float(np.percentile(preflight_ms, 95)),
                "necessary_preflight_plus_qp_p95_ms": float(np.percentile(
                    preflight_ms + qp_ms, 95)),
                "full_control_p95_ms": (float(np.percentile(full_ms, 95))
                                        if full_ms is not None else None),
                "full_control_p99_ms": (float(np.percentile(full_ms, 99))
                                        if full_ms is not None else None),
                "full_control_max_ms": (float(np.max(full_ms))
                                        if full_ms is not None else None),
                "full_control_over_20ms_count": (
                    int(np.count_nonzero(full_ms > 20))
                    if full_ms is not None else None),
                "qp_only_over_20ms_count": int(np.count_nonzero(qp_ms > 20)),
                "minimum_current_envelope_margin_m": float(np.min(
                    trace["task_interval_current_envelope_margin_m"])),
                "minimum_ramp_envelope_margin_m": float(np.min(
                    trace["task_interval_ramp_minimum_envelope_margin_m"])),
                "minimum_realized_next_start_slack_m_s": float(np.min(
                    trace["task_interval_realized_next_start_minimum_slack_m_s"])),
                "max_interval_rows": int(np.max(
                    trace["task_interval_selected_rows"])),
                "failure_reason_values": sorted(set(
                    str(value) for value in trace["task_failure_reason"])),
                "trace_sha256": _sha(path),
            })
    x = np.arange(len(records))
    names = [row["scenario_id"].rsplit("_", 1)[-1] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             constrained_layout=True)
    ax = axes[0]
    ax.bar(x - .17, [r["qp_only_p95_ms"] for r in records], .34,
           label="Saved QP call p95", color="#365eb4")
    all_full = all(r["full_control_p95_ms"] is not None for r in records)
    ax.bar(x + .17, [r["full_control_p95_ms"] if all_full else
                       r["necessary_preflight_plus_qp_p95_ms"] for r in records],
           .34, label=("Full control p95" if all_full else
                       "Preflight + QP p95 (necessary part)"), color="#ed9b3d")
    ax.axhline(20, color="#ad3340", linestyle="--", linewidth=1.5,
               label="Original 20 ms threshold")
    ax.set_ylabel("Milliseconds")
    ax.set_title("B.2 bounded interval online run: completed scenario traces")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    ax.plot(x, [1000*r["minimum_current_envelope_margin_m"]
                for r in records], "o-", label="Current envelope margin")
    ax.plot(x, [1000*r["minimum_ramp_envelope_margin_m"]
                for r in records], "s-", label="Previewed 11-state ramp margin")
    ax.axhline(0, color="#ad3340", linestyle="--", linewidth=1)
    ax.set_ylabel("Minimum margin (mm)")
    ax.set_xlabel("Scenario")
    ax.set_xticks(x, names)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    figure = output_dir / "online-progress.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "v6_2_b2_online_progress_visual_v1",
        "scope": "completed_saved_scenario_traces_only",
        "full_online_cycle_timed": all_full,
        "original_20ms_gate_met": (all_full and all(
            row["full_control_p95_ms"] <= 20.0 for row in records)),
        "records": records,
        "figure_sha256": _sha(figure),
    }
    summary_path = output_dir / "online_progress_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8",
                            newline="\n")
    lines = ["# B.2 新模式在线轨迹进展", "",
             ("本快照只读取已经完整写出的场景 trace。橙色柱包含预检、QP、"
              "十步力矩预演和下一起点重算；不构成硬实时保证。" if all_full else
              "本快照只读取已经完整写出的场景 trace。预检与 QP 是完整在线周期的"
              "必要部分；橙色柱仍未包含力矩斜坡预演、独立下一起点检查和调度开销。"),
             "", "![在线进度](online-progress.png)", "",
             "| 场景 | 规划周期 | 力矩步 | QP p95 ms | "
             + ("完整控制 p95 ms" if all_full else "预检+QP p95 ms")
             + " | 最小斜坡包络 mm |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in records:
        lines.append(f"| {row['scenario_id']} | {row['task_ticks']} | "
                     f"{row['torque_steps']} | {row['qp_only_p95_ms']:.3f} | "
                     f"{(row['full_control_p95_ms'] if all_full else row['necessary_preflight_plus_qp_p95_ms']):.3f} | "
                     f"{1000*row['minimum_ramp_envelope_margin_m']:.3f} |")
    lines += ["", "原 20 ms 门槛与完整 26 项重放验收尚需另行判定。", ""]
    (output_dir / "ONLINE_PROGRESS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.run_dir, args.output_dir)
    print(json.dumps({"completed_scenarios": len(report["records"]),
                      "qp_only_p95_ms": [r["qp_only_p95_ms"]
                                         for r in report["records"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
