"""Plot the hash-bound five-scene private domain-QP evidence."""

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


def generate(summary_path: Path, manifest_path: Path, output: Path) -> dict:
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["summary_sha256"] != _sha(summary_path)
            or report["scene_count"] != 5
            or report["failure_count"] != 0
            or report["private_diagnostic_status"]
            != "PASS_PRIVATE_FIVE_WITH_LIMITS"
            or report["stage2_online_gate_status"] != "GATE_NOT_MET"):
        raise ValueError("five-scene visualization evidence changed")
    scenes = report["scenes"]
    names = [item["scenario_id"][-2:] for item in scenes]
    x = np.arange(5)
    domain = np.asarray([item["minimum_work_domain_margin_rad"] * 1000
                         for item in scenes])
    path_rmse = np.asarray([item["continuum_path_rmse_m"] * 1e6
                            for item in scenes])
    p95 = np.asarray([item["preflight_plus_qp_timing"]["p95_ms"]
                      for item in scenes])
    maximum = np.asarray([item["preflight_plus_qp_timing"]["max_ms"]
                          for item in scenes])
    over = [item["preflight_plus_qp_timing"]["over_20ms_count"]
            for item in scenes]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained")
    fig.suptitle(
        "B.2 five complete private traces: domain QP + pre-servo gate\n"
        "Model-based diagnostic; full online cycle and 26/11 acceptance pending",
        fontsize=14)
    axes[0].bar(x, domain, color="#247b85", width=.62)
    axes[0].set_ylabel("Minimum work-domain margin (mrad)")
    axes[0].set_ylim(0, max(domain) * 1.16)
    for idx, value in enumerate(domain):
        axes[0].text(idx, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    axes[0].set_title("All 67,505 executed 2 ms states remained inside the declared domain")

    axes[1].bar(x, path_rmse, color="#5c8f5f", width=.62)
    axes[1].axhline(180, color="#b34835", linestyle="--", linewidth=1.5,
                    label="Original A.1 threshold: 180 µm")
    axes[1].set_ylabel("Continuum active-path RMSE (µm)")
    axes[1].set_ylim(0, 205)
    axes[1].legend(loc="upper right", fontsize=8)
    for idx, value in enumerate(path_rmse):
        axes[1].text(idx, value + 3, f"{value:.1f}", ha="center", fontsize=9)

    axes[2].bar(x - .16, p95, width=.32, color="#596e9e", label="Preflight + QP p95")
    axes[2].bar(x + .16, maximum, width=.32, color="#c27d43",
                label="Preflight + QP maximum")
    axes[2].axhline(20, color="#b34835", linestyle="--", linewidth=1.5,
                    label="20 ms task period")
    axes[2].set_ylim(0, max(maximum) * 1.25)
    axes[2].set_ylabel("Measured partial-chain time (ms)")
    axes[2].set_xlabel("Scenario")
    axes[2].legend(loc="upper left", fontsize=8)
    for idx, count in enumerate(over):
        axes[2].text(idx + .16, maximum[idx] + .7,
                     f"{count} >20", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, names)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    evidence = {
        "schema": "v6_2_b2_domain_private_five_visual_v1",
        "image_sha256": _sha(output),
        "summary_sha256": _sha(summary_path),
        "manifest_sha256": _sha(manifest_path),
        "scene_count": len(scenes),
        "stage2_online_gate_status": report["stage2_online_gate_status"],
        "full_cycle_20ms_acceptance": False,
    }
    output.with_suffix(".json").write_text(json.dumps(evidence, indent=2)
                                           + "\n", encoding="utf-8",
                                           newline="\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.summary, args.manifest, args.output),
                     indent=2))


if __name__ == "__main__":
    main()
