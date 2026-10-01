"""Same-state parity of the opt-in exact and optimized B.2 interval QPs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_interval_online import (
    BoundedIntervalAdmission, BoundedIntervalVelocityQP,
)
from v6_lite.b2_interval_online_optimized import (
    OptimizedBoundedIntervalAdmission, OptimizedBoundedIntervalVelocityQP,
)
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _body_pose_and_twist, build_scenarios,
    default_v6_lite_robot_spec,
)


TICKS = (0, 50, 100)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, a1_root: Path, trace_override: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    spec = default_v6_lite_robot_spec()
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    scenario = build_scenarios(spec, run_cfg)[0]
    saved = metrics["scenarios"][0]
    cfg = replace(HierarchicalQPConfig(**metrics["qp_config"]),
                  enable_pcc_cbf=False, enable_capsule_cbf=True)
    verifier = WholeBodyCollisionVerifier(
        spec, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=run_cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    trace_path = (Path(saved["trace"]["path"])
                  if trace_override is None else trace_override)
    if (trace_override is None
            and _sha(trace_path) != saved["trace"]["sha256"]):
        raise ValueError("frozen A.1 trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as trace:
        states = trace["task_qpos"][list(TICKS)].copy()
        previous = np.zeros((len(TICKS), 17))
        for i, tick in enumerate(TICKS):
            if tick:
                previous[i] = trace["task_selected_command"][tick-1]
        initial_qvel = trace["initial_qvel"].copy()
        torque = trace["torque"].copy()
    target_body = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite"))
    data = mujoco.MjData(model)
    data.qpos[:] = states[0]
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0
    qvel_states = {0: initial_qvel.copy()}
    for step in range(max(TICKS) * 10):
        data.ctrl[:] = torque[step]
        mujoco.mj_step(model, data)
        if (step + 1) % 10 == 0:
            tick = (step + 1) // 10
            if tick in TICKS:
                qvel_states[tick] = data.qvel.copy()
    records = []
    for i, tick in enumerate(TICKS):
        current = []
        for qp_type, admission_type in (
            (BoundedIntervalVelocityQP, BoundedIntervalAdmission),
            (OptimizedBoundedIntervalVelocityQP,
             OptimizedBoundedIntervalAdmission),
        ):
            data = mujoco.MjData(model)
            data.qpos[:] = states[i]
            data.qvel[:] = qvel_states[tick]
            data.time = tick * run_cfg.task_period_s
            mujoco.mj_forward(model, data)
            evaluator = FixedIntervalCBFEvaluator(spec, model)
            qp = qp_type(spec, model, verifier.pairs, cfg, evaluator=evaluator)
            qp.previous_velocity = previous[i].copy()
            admission = admission_type(spec, model, qp)
            preflight = admission.prepare(data, previous[i])
            rigid_target, rigid_velocity, rotation, angular = _body_pose_and_twist(
                model, data, target_body, scenario.grasp_point_target_frame_m)
            continuum_target, continuum_velocity = scenario.continuum_target.sample(
                float(data.time))
            result = qp.solve(
                data,
                rigid_target_position=rigid_target,
                rigid_target_velocity=rigid_velocity,
                rigid_target_rotation=(rotation @
                                       scenario.grasp_rotation_target_frame),
                rigid_target_angular_velocity=angular,
                continuum_target_position=continuum_target,
                continuum_target_velocity=continuum_velocity,
                continuum_target_rotation=scenario.continuum_target_rotation_world,
                continuum_target_angular_velocity=np.zeros(3),
                state_timestamp_s=float(data.time),
                target_timestamp_s=float(data.time),
                ramp_start_velocity=previous[i],
                prepared_state=(qp_type is OptimizedBoundedIntervalVelocityQP),
            )
            current.append((preflight, qp, result))
        a, b = current
        ra, rb = a[2], b[2]
        same_sources = ra.clearance_sources == rb.clearance_sources
        same_ids = a[1].selected_ids == b[1].selected_ids
        row_error = float(np.max(np.abs(
            ra.clearance_matrix - rb.clearance_matrix)))
        lower_error = float(np.max(np.abs(
            ra.clearance_lower - rb.clearance_lower)))
        candidate_error = float(np.max(np.abs(
            ra.solver_candidate - rb.solver_candidate)))
        selected_error = (None if ra.planner_velocity is None
                          or rb.planner_velocity is None else float(np.max(
                              np.abs(ra.planner_velocity - rb.planner_velocity))))
        record = {
            "tick": tick, "same_interval_ids": same_ids,
            "same_constraint_sources": same_sources,
            "matrix_max_abs_error": row_error,
            "lower_max_abs_error": lower_error,
            "candidate_max_abs_error": candidate_error,
            "selected_max_abs_error": selected_error,
            "same_solver_status": ra.solver_status == rb.solver_status,
            "same_action_mode": (ra.action_validation.mode
                                 == rb.action_validation.mode),
            "same_failure_reason": (ra.action_validation.failure_reason
                                    == rb.action_validation.failure_reason),
            "reference_exact_pair_count": len(verifier.pairs),
            "optimized_exact_pair_calls": b[1].last_exact_pair_call_count,
            "sphere_fallback": b[1].last_sphere_fallback,
            "reference_qp_ms": ra.full_latency_s * 1000,
            "optimized_qp_ms": rb.full_latency_s * 1000,
        }
        record["passed"] = (
            same_ids and same_sources and row_error <= 1e-8
            and lower_error <= 1e-8 and candidate_error <= 1e-6
            and selected_error is not None and selected_error <= 1e-6
            and record["same_solver_status"] and record["same_action_mode"]
            and record["same_failure_reason"])
        records.append(record)
    report = {
        "schema": "v6_2_b2_online_optimization_same_state_parity_v1",
        "predeclared_ticks": list(TICKS),
        "input_metrics_sha256": _sha(metrics_path),
        "input_trace_sha256": _sha(trace_path),
        "all_passed": all(row["passed"] for row in records),
        "records": records,
    }
    (output_dir / "parity.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--trace-path", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.trace_path)
    print(json.dumps({"all_passed": result["all_passed"],
                      "records": result["records"]}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
