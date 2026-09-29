"""Plot B.2 private branch bottlenecks and the compact-result timing trial."""

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


def run(output_dir: Path, branch_dir: Path, trial_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    branch_path = branch_dir / "branch_breakdown_summary.json"
    trial_path = trial_dir / "compact_envelope_timing_summary.json"
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if (branch["task_ticks"] != 1350 or trial["task_ticks"] != 1350
            or not branch["all_non_timing_records_exact"]
            or not trial["all_non_timing_records_exact"]
            or branch["full_online_cycle_evaluated"]
            or trial["full_online_cycle_evaluated"]):
        raise ValueError("branch visualization inputs do not match the protocol")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 7.0), gridspec_kw={"height_ratios": [2.1, 1.1]})
    fig.patch.set_facecolor("white")
    names = [
        ("Full candidate branch", "branch_wall_ms"),
        ("Envelope checks (11 states)", "envelope_ms"),
        ("  PCC positions inside envelope", "positions_ms"),
        ("MuJoCo forward", "forward_ms"),
        ("MuJoCo step", "step_ms"),
        ("67-channel torque", "torque_ms"),
        ("Other branch work", "unattributed_ms"),
    ]
    y = np.arange(len(names))[::-1]
    values = [branch["metrics"][key]["p95_ms"] for _, key in names]
    colors = ["#b45309", "#0f766e", "#334155", "#334155", "#334155",
              "#334155", "#334155"]
    ax1.barh(y, values, color=colors, height=.62)
    for pos, value in zip(y, values):
        ax1.text(value + .18, pos, f"{value:.2f}", va="center", fontsize=10)
    ax1.set_yticks(y, [label for label, _ in names])
    ax1.set_xlim(0, 15)
    ax1.set_xlabel("Per-component p95 (ms)")
    ax1.set_title("Ten-step private candidate branch: where time is spent",
                  loc="left", fontsize=13, fontweight="medium")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=.15)
    ax1.set_axisbelow(True)

    labels = ["Candidate branch", "Necessary component sum"]
    old = [trial["reference_ten_step_branch"]["p95_ms"],
           trial["reference_necessary_component_sum"]["p95_ms"]]
    new = [trial["trial_ten_step_branch"]["p95_ms"],
           trial["trial_necessary_component_sum"]["p95_ms"]]
    x = np.arange(2)
    width = .32
    ax2.bar(x - width / 2, old, width, color="#64748b", label="Full result")
    ax2.bar(x + width / 2, new, width, color="#0f766e",
            label="Summary only")
    for xpos, value in zip(np.r_[x - width / 2, x + width / 2], old + new):
        ax2.text(xpos, value + .25, f"{value:.2f}", ha="center", fontsize=9)
    ax2.axhline(20.0, color="#dc2626", linestyle="--", linewidth=1.4)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 27)
    ax2.set_ylabel("p95 (ms)")
    ax2.legend(frameon=False, ncol=2, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=.15)
    ax2.set_axisbelow(True)
    fig.text(.14, .027,
             "PCC positions are included in envelope time. Separate p95 values cannot be added. "
             "This is a private diagnostic, not an online-cycle acceptance result.",
             fontsize=9, color="#334155")
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=2.1)
    image_path = output_dir / "b2-branch-bottleneck.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema": "v6_2_b2_branch_visual_v1",
        "task_ticks": 1350,
        "branch_full_result_p95_ms": old[0],
        "branch_summary_only_p95_ms": new[0],
        "necessary_sum_full_result_p95_ms": old[1],
        "necessary_sum_summary_only_p95_ms": new[1],
        "stage2_20ms_gate_met": False,
        "source_sha256": {
            "branch_breakdown_summary": _sha(branch_path),
            "compact_envelope_timing_summary": _sha(trial_path),
            "visualize_b2_branch_breakdown.py": _source_sha(Path(__file__)),
            "image": _sha(image_path),
        },
    }
    (output_dir / "branch_visual_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.branch_dir, args.trial_dir),
                     indent=2))


if __name__ == "__main__":
    main()
