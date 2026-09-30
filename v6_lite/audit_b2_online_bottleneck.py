"""Profile the actual ten-step preview and next-start gate on a saved trace."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_interval_online_optimized import (
    OptimizedBoundedIntervalAdmission, OptimizedBoundedIntervalVelocityQP,
)
from v6_lite.b2_interval_runtime import preview_ramp
from v6_lite.b2_screened_next_start_rows import ScreenedNextStart
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


def run(run_dir: Path, output_dir: Path, *, tick: int, repeats: int,
        prepared_step: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = json.loads((run_dir / "v6_lite_metrics.json").read_text(
        encoding="utf-8"))
    saved = metrics["scenarios"][1]
    trace_path = Path(saved["trace"]["path"])
    spec = default_v6_lite_robot_spec()
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = metrics["run_config"]
    verifier = WholeBodyCollisionVerifier(
        spec, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_cfg["whole_body_minimum_clearance_m"],
            query_distance_max=2.5,
            adaptive_subdivisions=run_cfg["verification_subdivisions"],
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = trace["task_qpos"][tick].copy()
        initial_qpos = trace["task_qpos"][0].copy()
        initial_qvel = trace["initial_qvel"].copy()
        prior_torques = trace["torque"][:tick * 10].copy()
        expected_torques = trace["torque"][tick * 10:(tick + 1) * 10].copy()
        reference = trace["reference_q"][tick * 10 - 1].copy()
        previous = trace["task_selected_command"][tick - 1].copy()
        endpoint = trace["task_selected_command"][tick].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    for torque in prior_torques:
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
    qpos_error = float(np.max(np.abs(qpos - data.qpos)))
    if qpos_error > 1e-9:
        raise ValueError(f"saved trace state replay mismatch: {qpos_error}")
    evaluator = FixedIntervalCBFEvaluator(spec, model)
    qp = OptimizedBoundedIntervalVelocityQP(
        spec, model, verifier.pairs, cfg, evaluator=evaluator)
    admission = OptimizedBoundedIntervalAdmission(spec, model, qp)
    checker = ScreenedNextStart()

    def one_cycle(use_prepared_step: bool):
        branch, next_data = preview_ramp(
            model, spec, evaluator, admission.envelope,
            data.qpos.copy(), data.qvel.copy(), float(data.time),
            reference, previous, endpoint,
            physics_period_s=run_cfg["physics_period_s"],
            task_period_s=run_cfg["task_period_s"],
            prepared_step=use_prepared_step,
        )
        next_start = checker(model, spec, verifier, cfg, evaluator,
                             next_data, endpoint)
        return branch, next_start, next_data.qvel.copy()

    original_branch, original_next, original_qvel = one_cycle(False)
    split_branch, split_next, split_qvel = one_cycle(True)
    qpos_split_error = float(np.max(np.abs(
        original_branch["qpos_states"] - split_branch["qpos_states"])))
    qvel_split_error = float(np.max(np.abs(original_qvel - split_qvel)))
    torque_split_error = float(np.max(np.abs(
        original_branch["torques"] - split_branch["torques"])))
    saved_torque_error = float(np.max(np.abs(
        expected_torques - split_branch["torques"])))
    if (qpos_split_error > 1e-9 or qvel_split_error > 1e-9
            or torque_split_error > 1e-9 or saved_torque_error > 1e-9
            or original_branch["coverage"] != split_branch["coverage"]
            or original_next != split_next):
        raise ValueError("prepared MuJoCo step changed preview or next-start rows")
    branch, next_start = (split_branch, split_next) if prepared_step else (
        original_branch, original_next)
    if next_start["frozen_rows_status"] != "START_ROWS_SATISFIED":
        raise ValueError("saved tick next-start gate is not satisfied")
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        one_cycle(prepared_step)
        durations.append(1000 * (time.perf_counter() - start))
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeats):
        one_cycle(prepared_step)
    profiler.disable()
    profile_path = output_dir / "profile.txt"
    with profile_path.open("w", encoding="utf-8") as stream:
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
    report = {
        "schema": "v6_2_b2_online_saved_state_bottleneck_v1",
        "tick": tick,
        "repeats": repeats,
        "qpos_replay_linf_error": qpos_error,
        "prepared_step": prepared_step,
        "prepared_step_qpos_linf_error": qpos_split_error,
        "prepared_step_qvel_linf_error": qvel_split_error,
        "prepared_step_torque_linf_error": torque_split_error,
        "saved_torque_linf_error": saved_torque_error,
        "cycle_p50_ms": float(np.percentile(durations, 50)),
        "cycle_p95_ms": float(np.percentile(durations, 95)),
        "minimum_envelope_margin_m": min(
            item["minimum_margin_m"] for item in branch["coverage"]),
        "next_start_status": next_start["frozen_rows_status"],
        "profile_path": str(profile_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tick", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--prepared-step", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir, args.output_dir,
                         tick=args.tick, repeats=args.repeats,
                         prepared_step=args.prepared_step), indent=2))


if __name__ == "__main__":
    main()
