"""Postprocessing charts from frozen C4-A metrics and native tracking CSVs."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..route_optimizer_protocol import read, sha

METHODS = ("P0", "P1", "P2")
COLORS = ("#52708d", "#d4823c", "#168d84")


def build_charts(run, output, slots):
    run, output = Path(run), Path(output)
    results = run.parent / "results_01"
    summary = read(results / "summary.json")
    metrics = summary["strategies"]
    folder = output / "charts"
    folder.mkdir(exist_ok=True)
    paths = {}

    def save(fig, name):
        fig.savefig(folder / name, dpi=160)
        plt.close(fig)
        paths[name] = "charts/" + name

    def bars(ax, values, title, ylabel, denominator=None):
        bars = ax.bar(METHODS, values, color=COLORS, width=.58)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.25 or 1))
        ax.grid(axis="y", alpha=.2)
        ax.bar_label(bars, labels=[f"{v:g}" + (f"/{denominator}" if denominator else "") for v in values], padding=4)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, key, title, denom in zip(axes.flat,
            ("full_nominal_qualified", "B30_candidates", "full_actual_passed", "near_P0_A", "near_P0_B", "prediction_calls"),
            ("Nominal full-task qualified", "Nominal B clearance >= 30 mm", "Actual task + original five gates",
             "Preference A: near P0", "Preference B: near P0", "Real predictions (16 consumed slots)"),
            (16, 16, 4, 2, 2, 16)):
        bars(ax, [metrics[m][key] for m in METHODS], title, "Count", denom)
    fig.suptitle("C4-A DEV | two independent tasks | fixed budget\nP0 Rule8 · P1 Diffusion replacement · P2 rule-preserving portfolio", fontsize=14)
    fig.supxlabel("Raw D legality: P1 = P2 = 3/4 (paired raws); P0 N/A. R12 not run. Aliases are not independent repeats.", fontsize=10)
    save(fig, "coverage_and_near_quality.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    tasks = list(dict.fromkeys(s["task_id"] for s in slots))
    labels = [t.replace("c4a_", "") + "/" + p for t in tasks for p in ("A", "B")]
    x = np.arange(4)
    definitions = (("I_support", 1., "Support intervention [rad/s]"),
                   ("L_full", 1., "Full continuum path length [m]"),
                   ("d_support", 1000., "Support minimum clearance [mm]"),
                   ("base_translation_peak_m", 1000., "Peak base translation [mm]"),
                   ("base_rotation_peak_rad", 180 / np.pi, "Peak base rotation [deg]"))
    for ax, (key, factor, title) in zip(axes.flat, definitions):
        for i, m in enumerate(METHODS):
            values = [next(s for s in slots if (s["task_id"], s["endpoint"], s["preference"]) == (t, m, p))["quality"][key] * factor
                      for t in tasks for p in ("A", "B")]
            ax.bar(x + (i - 1) * .23, values, .23, label=m, color=COLORS[i])
        ax.set(title=title, xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
        if key == "d_support":
            ax.axhline(30., color="#b44343", ls="--", lw=1, label="B target 30 mm")
        ax.legend(fontsize=8)
    bars(axes[1, 2], [metrics[m]["cold_service_s"] for m in METHODS], "Cold service: two A/B pools", "Seconds")
    fig.suptitle("C4-A DEV | selected saved actual quality and planning cost", fontsize=14)
    fig.supxlabel("Cost counts each shared A/B search once. P1/P2 reject one raw before prediction; lower totals do not establish speedup.", fontsize=10)
    save(fig, "actual_quality_and_cost.png")

    for task in tasks:
        for pref in ("A", "B"):
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
            for i, m in enumerate(METHODS):
                slot = next(s for s in slots if (s["task_id"], s["endpoint"], s["preference"]) == (task, m, pref))
                source = Path(slot["alias_of_slot"]).parent.name if slot.get("alias_of_slot") else slot["method"]
                data = np.loadtxt(output / "figures" / f"{task}_{source}_tracking_native.csv", delimiter=",", skiprows=1)
                label = m + (" (alias " + source + ")" if slot.get("alias_of_slot") else "")
                for ax, col, ylabel in zip(axes, (1, 3, 4),
                        ("Continuum vs generated [mm]", "Rigid vs Task grasp [mm]", "Continuum orientation [deg]")):
                    factor = 180 / np.pi if col == 4 else 1000.
                    ax.plot(data[:, 0], data[:, col] * factor, label=label, color=COLORS[i],
                            ls=("-", "--", ":")[i], lw=1.4)
                    ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
            axes[-1].set(xlabel="Saved physical time [s]", xlim=(0, 27))
            fig.suptitle(f"C4-A DEV | {task} | Preference {pref}\nNative 2 ms samples; coincident curves retain exact values")
            save(fig, f"{task}_{pref}_tracking_comparison.png")

    from .build_preference_warmstart_media import write
    write(folder / "sources.json", {"sources": {str(p.relative_to(run.parent)): sha(p) for p in
          (results / "summary.json", results / "dev_endpoints.csv", results / "dev_streams.csv")},
          "logical_slots": 12, "unique_actuals": 7, "new_physics_steps": 0,
          "interpretation": "DEV only; MODIFY; ARCHITECTURE_NOT_JUSTIFIED; no speedup claim"})
    return paths
