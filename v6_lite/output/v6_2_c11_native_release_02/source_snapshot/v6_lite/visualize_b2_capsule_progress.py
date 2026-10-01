"""Plot private B.2 decision timing and exact geometry/PCC parity."""

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


def run(output_dir: Path, baseline_dir: Path, compact_dir: Path,
        batched_dir: Path, frozen_dir: Path, adversarial_dir: Path,
        vectorized_dir: Path, positions_dir: Path,
        cross_impl_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "baseline": baseline_dir / "prepared_torque_timing_summary.json",
        "compact": compact_dir / "compact_envelope_timing_summary.json",
        "batched": batched_dir / "batched_capsule_private_timing_summary.json",
        "frozen": frozen_dir / "batched_capsule_summary.json",
        "adversarial": adversarial_dir / "adversarial_capsule_summary.json",
        "vectorized": vectorized_dir
        / "vectorized_positions_private_timing_summary.json",
        "positions": positions_dir / "fixed_arc_position_summary.json",
        "cross_impl": cross_impl_dir / "private_query_cross_impl_summary.json",
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8"))
               for key, path in sources.items()}
    if (reports["baseline"]["task_ticks"] != 1350
            or reports["compact"]["task_ticks"] != 1350
            or reports["batched"]["task_ticks"] != 1350
            or reports["frozen"]["state_count"] != 6755
            or reports["adversarial"]["case_count"] != 315
            or reports["adversarial"]["near_five_mm_count"] != 126
            or reports["vectorized"]["task_ticks"] != 1350
            or reports["positions"]["state_count"] != 6755
            or reports["cross_impl"]["state_count"] != 1351
            or not reports["compact"]["all_non_timing_records_exact"]
            or not reports["batched"]["all_non_timing_records_exact"]
            or not reports["vectorized"]["all_non_timing_records_exact"]
            or not reports["vectorized"]["same_native_67_channel_torque_trace"]
            or reports["positions"]["maximum_absolute_position_error_m"] > 1e-12
            or reports["cross_impl"]["maximum_bound_error_m"] > 1e-12
            or reports["frozen"]["source_mismatch_count"]
            or reports["adversarial"]["source_mismatch_count"]
            or reports["adversarial"]["maximum_distance_error_m"] > 1e-12
            or reports["adversarial"]["maximum_witness_error"] > 1e-12):
        raise ValueError("capsule progression source contract failed")
    decision_p95 = [
        reports["baseline"]["necessary_component_sum"]["p95_ms"],
        reports["compact"]["trial_necessary_component_sum"]["p95_ms"],
        reports["batched"]["trial_necessary_component_sum"]["p95_ms"],
        reports["vectorized"]["trial_necessary_component_sum"]["p95_ms"],
    ]
    capsule_p95 = [reports["frozen"][key]["p95_ms"]
                   for key in ("reference_timing", "batched_timing")]
    positions_p95 = [reports["positions"][key]["p95_ms"]
                     for key in ("reference_timing", "trial_timing")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.3))
    fig.patch.set_facecolor("white")
    left, middle = axes[0]
    bottom_left, right = axes[1]
    x = np.arange(4)
    left.plot(x, decision_p95, color="#0f766e", marker="o", linewidth=2,
              markersize=7)
    for xpos, value in zip(x, decision_p95):
        left.text(xpos, value + .24, f"{value:.2f}", ha="center", fontsize=10)
    left.axhline(20, color="#dc2626", linestyle="--", linewidth=1.5)
    left.set_xticks(x, ["Full results", "Compact envelope",
                        "Batched capsules", "Vectorized PCC"], rotation=12)
    left.set_ylim(18.8, 26)
    left.set_ylabel("Necessary component sum, p95 (ms)")
    left.set_title("Scene 01 private decision timing", loc="left", fontsize=12)
    left.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", alpha=.15)
    left.set_axisbelow(True)

    bars = middle.bar([0, 1], capsule_p95, color=["#64748b", "#0f766e"],
                     width=.55)
    for bar, value in zip(bars, capsule_p95):
        middle.text(bar.get_x() + bar.get_width()/2, value + .035,
                   f"{value:.2f}", ha="center", fontsize=10)
    middle.set_xticks([0, 1], ["Original", "Batched bounds"])
    middle.set_ylim(0, 1.75)
    middle.set_ylabel("Capsule query p95 (ms)")
    middle.set_title("Five-scene frozen states", loc="left", fontsize=12)
    middle.spines[["top", "right"]].set_visible(False)
    middle.grid(axis="y", alpha=.15)
    middle.set_axisbelow(True)
    bars = bottom_left.bar([0, 1], positions_p95,
                           color=["#64748b", "#0f766e"], width=.55)
    for bar, value in zip(bars, positions_p95):
        bottom_left.text(bar.get_x() + bar.get_width()/2, value + .006,
                         f"{value:.3f}", ha="center", fontsize=10)
    bottom_left.set_xticks([0, 1], ["Fixed arc", "Five-section batch"])
    bottom_left.set_ylim(0, .34)
    bottom_left.set_ylabel("PCC positions p95 (ms)")
    bottom_left.set_title("Five-scene frozen states", loc="left", fontsize=12)
    bottom_left.spines[["top", "right"]].set_visible(False)
    bottom_left.grid(axis="y", alpha=.15)
    bottom_left.set_axisbelow(True)
    right.axis("off")
    right.set_title("Geometry and query parity", loc="left", fontsize=12)
    right.text(.02, .83, "6,755 frozen states", fontsize=12, weight="bold",
               transform=right.transAxes)
    right.text(.02, .70, "0 capsule or PCC position mismatches", fontsize=10,
               transform=right.transAxes)
    right.text(.02, .54, "1,351 cross-query states", fontsize=12,
               weight="bold", transform=right.transAxes)
    right.text(.02, .43,
               f"Max bound error: {reports['cross_impl']['maximum_bound_error_m']:.1e} m",
               fontsize=10,
               transform=right.transAxes)
    right.text(.02, .28, "315 adversarial positions", fontsize=12,
               weight="bold", transform=right.transAxes)
    right.text(.02, .17, "126 near the 5 mm gate", fontsize=10,
               transform=right.transAxes)
    right.text(.02, .07, "0 source / witness mismatches", fontsize=10,
               transform=right.transAxes)
    fig.text(.06, .01,
             "Private timings exclude online scheduling. Exact capsule winners and the 67-channel torque trace match reference; Stage 2 gate is not met.",
             fontsize=9, color="#334155")
    fig.tight_layout(rect=(0, .045, 1, 1), w_pad=3.0, h_pad=2.5)
    image_path = output_dir / "b2-capsule-timing-progress.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = {
        "schema": "v6_2_b2_capsule_timing_visual_v1",
        "private_scene01_necessary_p95_ms": decision_p95,
        "frozen_five_scene_capsule_p95_ms": capsule_p95,
        "frozen_five_scene_positions_p95_ms": positions_p95,
        "adversarial_case_count": reports["adversarial"]["case_count"],
        "adversarial_near_five_mm_count": reports["adversarial"]["near_five_mm_count"],
        "cross_impl_state_count": reports["cross_impl"]["state_count"],
        "cross_impl_max_bound_error_m": reports["cross_impl"]["maximum_bound_error_m"],
        "stage2_20ms_gate_met": False,
        "source_sha256": {key: _sha(path) for key, path in sources.items()}
        | {"visualize_b2_capsule_progress.py": _source_sha(Path(__file__)),
           "image": _sha(image_path)},
    }
    (output_dir / "capsule_progress_visual_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--batched-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--adversarial-dir", type=Path, required=True)
    parser.add_argument("--vectorized-dir", type=Path, required=True)
    parser.add_argument("--positions-dir", type=Path, required=True)
    parser.add_argument("--cross-impl-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.baseline_dir, args.compact_dir,
                         args.batched_dir, args.frozen_dir,
                         args.adversarial_dir, args.vectorized_dir,
                         args.positions_dir, args.cross_impl_dir), indent=2))


if __name__ == "__main__":
    main()
