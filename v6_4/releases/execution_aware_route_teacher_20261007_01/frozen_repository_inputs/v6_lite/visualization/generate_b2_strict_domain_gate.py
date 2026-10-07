"""Plot the original domain crossing and the private pre-servo rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(strict_dir: Path, baseline_dir: Path, replay_dir: Path,
             a1_root: Path, output: Path) -> dict:
    strict_summary_path = strict_dir / "private_rollout_summary.json"
    strict_trace_path = strict_dir / "private_rollout_trace.npz"
    baseline_trace_path = baseline_dir / "private_rollout_trace.npz"
    replay_path = replay_dir / "strict_domain_replay.json"
    strict = json.loads(strict_summary_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = next(item for item in metrics["scenarios"] if
                 item["scenario"]["scenario_id"] == "v6_lite_scenario_01")
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=run_cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    evaluator = FixedIntervalCBFEvaluator(robot, model)

    def excess(trace_path: Path) -> np.ndarray:
        with np.load(trace_path, allow_pickle=False) as trace:
            states = trace["qpos_states"][::10].copy()
        upper = evaluator.shape_spec.work_domain_upper_rad
        return np.asarray([
            np.max(evaluator.shape_spec.project_actual_configuration(
                state[evaluator.qpos_ids[:60]]).planner_configuration - upper)
            for state in states
        ]) * 1000.0

    old = excess(baseline_trace_path)
    stopped = excess(strict_trace_path)
    if (strict["executed_ticks"] != 887 or len(stopped) != 888
            or replay["checked_native_states"] != 8871
            or not np.array_equal(old[:len(stopped)], stopped)):
        raise ValueError("plotted traces changed")
    first_failure = strict["failure_record"]["ramp_domain_failures"][0]
    if first_failure["servo_substep"] != 4:
        raise ValueError("first predicted crossing changed")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [4, 1]},
                             sharex=True, layout="constrained")
    ax = axes[0]
    lo, hi = 860, 970
    ax.plot(np.arange(lo, hi + 1), old[lo:hi + 1], color="#bc4b35",
            linewidth=2, label="Earlier complete private diagnostic")
    ax.plot(np.arange(lo, len(stopped)), stopped[lo:], color="#1f7285",
            linewidth=3, label="Strict gate: executed states")
    ax.axhline(0, color="#27333a", linewidth=1.2)
    ax.axvspan(888, 950, color="#bc4b35", alpha=.12,
               label="Old out-of-domain window (63 task ticks)")
    ax.axvline(887, color="#1f7285", linestyle="--", linewidth=1.5)
    ax.scatter([887.4], [first_failure["maximum_upper_excess_rad"] * 1000],
               color="#bc4b35", s=55, zorder=5,
               label="Rejected prediction: servo substep 4")
    ax.annotate("Reject before torque step 8871", xy=(887.4,
                first_failure["maximum_upper_excess_rad"] * 1000),
                xytext=(899, 2.0), arrowprops={"arrowstyle": "->"},
                fontsize=10)
    ax.set_ylabel("Largest upper work-domain excess (mrad)")
    ax.set_title("Scene 01: strict gate stops before the first out-of-domain 2 ms step")
    ax.set_xlim(lo, hi)
    ax.grid(alpha=.25)
    ax.legend(loc="upper right", fontsize=8)
    ax2 = axes[1]
    ax2.barh([0], [887], left=0, height=.45, color="#1f7285")
    ax2.barh([0], [1350 - 887], left=887, height=.45,
             color="#d9e4e6", hatch="//")
    ax2.set_xlim(lo, hi)
    ax2.set_yticks([0], ["Strict trace"])
    ax2.set_xlabel("Task tick (20 ms each)")
    ax2.grid(axis="x", alpha=.2)
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    evidence = {
        "schema": "v6_2_b2_strict_gate_visual_v1",
        "image_sha256": _sha(output),
        "strict_summary_sha256": _sha(strict_summary_path),
        "strict_trace_sha256": _sha(strict_trace_path),
        "baseline_trace_sha256": _sha(baseline_trace_path),
        "independent_replay_sha256": _sha(replay_path),
        "earlier_out_of_domain_tick_window": [888, 950],
        "strict_last_executed_tick": 886,
        "strict_rejected_tick": 887,
        "first_predicted_crossing_servo_substep": 4,
        "visualization_is_private_diagnostic": True,
    }
    output.with_suffix(".json").write_text(json.dumps(
        evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.strict_dir, args.baseline_dir, args.replay_dir,
                      args.a1_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
