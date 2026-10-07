"""Plot the 50 Hz timing budget and the latest private B.2 measurements."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, stage_dir: Path, decision_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    stage_path = stage_dir / "qp_breakdown_summary.json"
    decision_path = decision_dir / "prepared_torque_timing_summary.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (stage["task_ticks"] != 1350 or decision["task_ticks"] != 1350
            or not stage["all_non_timing_records_exact"]
            or not decision["all_non_timing_records_exact"]
            or stage["full_online_cycle_evaluated"]
            or decision["full_online_cycle_evaluated"]):
        raise ValueError("timing sources do not match the declared private protocol")

    period_ms = 20.0
    servo_ms = 2.0
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.9),
                             gridspec_kw={"height_ratios": [1.0, 2.5]})
    fig.patch.set_facecolor("white")
    first, second = axes
    colors = {"period": "#2563eb", "measured": "#0f766e",
              "failed": "#b45309", "neutral": "#334155"}

    for index in range(10):
        first.barh(0, servo_ms, left=index * servo_ms, height=0.42,
                   color=colors["period"] if index % 2 == 0 else "#60a5fa",
                   edgecolor="white", linewidth=1.5)
        first.text((index + .5) * servo_ms, 0, str(index + 1),
                   ha="center", va="center", color="white", fontsize=9)
    first.set_xlim(0, period_ms)
    first.set_yticks([])
    first.set_xticks(np.arange(0, 21, 2))
    first.set_xlabel("Time in one task period (ms)")
    first.set_title("50 Hz task period = ten 2 ms torque steps", loc="left",
                    fontsize=13, fontweight="medium")
    first.spines[["top", "right", "left"]].set_visible(False)
    first.grid(axis="x", alpha=.15)

    metrics = [
        ("Preflight + QP", stage["metrics"]["preflight_plus_qp_ms"]["p95_ms"],
         colors["measured"]),
        ("  QP constraint assembly", stage["metrics"]["clearance_assembly_ms"]["p95_ms"],
         colors["neutral"]),
        ("  ADMM solver", stage["metrics"]["solver_latency_ms"]["p95_ms"],
         colors["neutral"]),
        ("Ten-step candidate branch", decision["ten_step_branch"]["p95_ms"],
         colors["measured"]),
        ("Next-start check", decision["screened_next_start"]["p95_ms"],
         colors["measured"]),
        ("Necessary component sum", decision["necessary_component_sum"]["p95_ms"],
         colors["failed"]),
    ]
    ypos = np.arange(len(metrics))[::-1]
    second.barh(ypos, [item[1] for item in metrics], height=.61,
                color=[item[2] for item in metrics])
    for y, (_, value, _) in zip(ypos, metrics):
        second.text(value + .35, y, f"{value:.2f}", va="center",
                    fontsize=10, color=colors["neutral"])
    second.axvline(period_ms, color="#dc2626", linestyle="--", linewidth=1.6)
    second.text(period_ms + .2, ypos[0] + .68, "20 ms period",
                color="#dc2626", fontsize=10)
    second.set_yticks(ypos, [item[0] for item in metrics])
    second.set_xlim(0, 29)
    second.set_xlabel("Measured p95 (ms)")
    second.set_title("Latest private scene 01 timing, 1350 task ticks", loc="left",
                     fontsize=13, fontweight="medium")
    second.spines[["top", "right"]].set_visible(False)
    second.grid(axis="x", alpha=.15)
    second.set_axisbelow(True)
    fig.text(.13, .025,
             "Each p95 is computed separately. Subrows are included in QP; do not add the bars. "
             "Private timing is not a full online-cycle or hard real-time result.",
             fontsize=9, color=colors["neutral"])
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=1.8)
    image_path = output_dir / "b2-20ms-timing-basis.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema": "v6_2_b2_timing_basis_visual_v1",
        "task_period_ms": period_ms,
        "servo_period_ms": servo_ms,
        "task_ticks": 1350,
        "historical_threshold_metric": "HierarchicalQPResult.full_latency_s from qp.solve",
        "new_stage2_scope": "private_necessary_components_not_full_online_cycle",
        "necessary_component_sum_p95_ms": decision["necessary_component_sum"]["p95_ms"],
        "necessary_component_sum_over_20ms_count": decision[
            "necessary_component_sum"]["over_20ms_count"],
        "stage2_20ms_gate_met": False,
        "source_sha256": {
            "qp_breakdown_summary": _sha(stage_path),
            "prepared_torque_timing_summary": _sha(decision_path),
            "visualize_b2_timing_basis.py": _source_sha(Path(__file__)),
            "image": _sha(image_path),
        },
    }
    (output_dir / "timing_basis_visual_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--decision-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.stage_dir, args.decision_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
