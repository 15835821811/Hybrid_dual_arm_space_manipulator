"""Read-only first-tick counterfactual for implicitfast servo compensation.

Keep the compiled model, 67 actuators, planner map, ramp, torque limits and
physics integrator unchanged. In a private clone, replace the continuous-time
mass matrix in the inverse-dynamics torque calculation by M + h diag(damping).
The counterfactual is diagnostic only and is never an admitted online action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_servo_subspace_origin import _residual, _sha, _source_sha
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _model_based_servo_torque,
    default_v6_lite_robot_spec, free_joint_slices,
)


def _compensated_torque(model, data, robot, qpos_ids, dof_ids, base_dof,
                        reference_position, reference_velocity,
                        feedforward_acceleration, full_mass):
    original, desired_acceleration, original_diagnostics = _model_based_servo_torque(
        model, data, robot, qpos_ids, dof_ids, base_dof,
        reference_position, reference_velocity, feedforward_acceleration,
        full_mass,
    )
    modified = full_mass + model.opt.timestep * np.diag(model.dof_damping)
    base_ids = np.arange(base_dof.start, base_dof.stop, dtype=np.int32)
    base_acceleration = -np.linalg.solve(
        modified[np.ix_(base_ids, base_ids)],
        modified[np.ix_(base_ids, dof_ids)] @ desired_acceleration
        + np.asarray(data.qfrc_bias)[base_ids]
        - np.asarray(data.qfrc_passive)[base_ids],
    )
    required = (
        modified[np.ix_(dof_ids, base_ids)] @ base_acceleration
        + modified[np.ix_(dof_ids, dof_ids)] @ desired_acceleration
        + np.asarray(data.qfrc_bias)[dof_ids]
        - np.asarray(data.qfrc_passive)[dof_ids]
    )
    command = np.clip(required, -robot.torque_limits, robot.torque_limits)
    return command, desired_acceleration, {
        "original_torque_saturation_count":
            original_diagnostics.torque_saturation_count,
        "compensated_torque_saturation_count": int(np.count_nonzero(
            command != required)),
        "acceleration_clip_count": original_diagnostics.acceleration_clip_count,
        "torque_change_linf_nm": float(np.max(np.abs(command - original))),
        "required_torque_abs_max_nm": float(np.max(np.abs(required))),
    }


def run(output_dir: Path, private_root: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    shape_spec = default_continuum_model_spec(robot)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = {item["scenario"]["scenario_id"]: item
             for item in metrics["scenarios"]}
    report = {
        "schema": "v6_2_b2_discrete_servo_counterfactual_v1",
        "scope": "first_ten_500hz_steps_from_private_initial_states",
        "input_metrics_sha256": _sha(metrics_path),
        "inputs": {},
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_discrete_servo_counterfactual.py",
                                       "audit_b2_servo_subspace_origin.py",
                                       "run_v6_lite.py", "execution_ramp.py",
                                       "continuum_model_spec.py")},
        "robot_mapping_changed": False,
        "actuator_torque_limits_changed": False,
        "physics_integrator_changed": False,
        "online_controller_changed": False,
        "stage3_admission": False,
        "continuous_time_certified": False,
        "scenes": [],
    }
    rows = []
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        folder = private_root / f"scene_{index:02d}"
        trace_path = folder / "private_rollout_trace.npz"
        records_path = folder / "private_rollout_records.jsonl"
        summary_path = folder / "private_rollout_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary["executed_ticks"] != 400
                or summary["trace_sha256"] != _sha(trace_path)
                or summary["records_sha256"] != _sha(records_path)):
            raise ValueError(f"private trace changed: {scene_id}")
        report["inputs"][scene_id] = {
            "trace_sha256": _sha(trace_path),
            "records_sha256": _sha(records_path),
            "summary_sha256": _sha(summary_path),
        }
        with np.load(trace_path, allow_pickle=False) as trace:
            initial_qpos = trace["initial_qpos"].copy()
            initial_qvel = trace["initial_qvel"].copy()
            reference_torque = trace["torque"][:10].copy()
            reference_qpos = trace["qpos_states"][:11].copy()
        endpoint = np.asarray(json.loads(records_path.read_text(
            encoding="utf-8").splitlines()[0])["selected_command"],
                              dtype=np.float64)
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(saved[scene_id]["scenario"]),
            WholeBodyVerificationConfig(
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
        if mujoco.mjtIntegrator(model.opt.integrator) != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError("the published integrator is no longer implicitfast")
        qpos_ids, dof_ids = joint_addresses(model, robot)
        _base_qpos, base_dof = free_joint_slices(model, robot.base_joint_name)
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0
        full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
        reference = robot.planner_zero.copy()
        scene_rows = []
        for step in range(10):
            mujoco.mj_forward(model, data)
            measured = robot.decode_position(data.qpos[qpos_ids])
            ramp = advance_reference(
                reference, np.zeros(17), endpoint, step + 1, measured,
                robot.planner_lower, robot.planner_upper,
                physics_period_s=run_cfg.physics_period_s,
                task_period_s=run_cfg.task_period_s,
            )
            reference = ramp.position
            torque, desired_acceleration, diagnostics = _compensated_torque(
                model, data, robot, qpos_ids, dof_ids, base_dof,
                ramp.position, ramp.velocity, ramp.feedforward_acceleration,
                full_mass,
            )
            velocity_before = data.qvel[dof_ids].copy()
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            finite_acceleration = (
                data.qvel[dof_ids] - velocity_before
            ) / run_cfg.physics_period_s
            row = {
                "scene_id": scene_id, "servo_step": step,
                "position_residual_rad": _residual(data.qpos[qpos_ids], shape_spec),
                "velocity_residual_rad_s": _residual(data.qvel[dof_ids], shape_spec),
                "desired_acceleration_residual_rad_s2":
                    _residual(desired_acceleration, shape_spec),
                "finite_acceleration_residual_rad_s2":
                    _residual(finite_acceleration, shape_spec),
                "desired_vs_finite_acceleration_error_rad_s2":
                    float(np.max(np.abs(finite_acceleration
                                        - desired_acceleration))),
                "torque_change_linf_nm": diagnostics["torque_change_linf_nm"],
                "required_torque_abs_max_nm": diagnostics[
                    "required_torque_abs_max_nm"],
                "torque_saturation_count": diagnostics[
                    "compensated_torque_saturation_count"],
                "acceleration_clip_count": diagnostics["acceleration_clip_count"],
                "qpos_linf_difference_from_original": float(np.max(np.abs(
                    data.qpos - reference_qpos[step + 1]))),
                "first_step_original_torque_error_nm": (
                    float(np.max(np.abs(torque - reference_torque[step])))
                    if step == 0 else None),
            }
            rows.append(row)
            scene_rows.append(row)
        report["scenes"].append({
            "scene_id": scene_id,
            "record_count": 10,
            "first_step_position_residual_rad": scene_rows[0][
                "position_residual_rad"],
            "maximum_position_residual_rad": max(
                row["position_residual_rad"] for row in scene_rows),
            "maximum_velocity_residual_rad_s": max(
                row["velocity_residual_rad_s"] for row in scene_rows),
            "maximum_desired_vs_finite_acceleration_error_rad_s2": max(
                row["desired_vs_finite_acceleration_error_rad_s2"]
                for row in scene_rows),
            "maximum_torque_change_nm": max(
                row["torque_change_linf_nm"] for row in scene_rows),
            "torque_saturation_step_count": sum(
                row["torque_saturation_count"] > 0 for row in scene_rows),
            "acceleration_clip_step_count": sum(
                row["acceleration_clip_count"] > 0 for row in scene_rows),
            "maximum_qpos_linf_difference_from_original": max(
                row["qpos_linf_difference_from_original"] for row in scene_rows),
        })
        print(f"[b2-discrete-servo] {scene_id}: first-step residual "
              f"{scene_rows[0]['position_residual_rad']:.3e} rad", flush=True)
    report["record_count"] = len(rows)
    records_path = output_dir / "discrete_servo_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report["records_sha256"] = _sha(records_path)
    summary_path = output_dir / "discrete_servo_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": "v6_2_b2_discrete_servo_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "records_sha256": _sha(records_path)}
    with (output_dir / "discrete_servo_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/private_rollout_400"))
    parser.add_argument("--a1-root", type=Path, default=Path(
        "v6_lite/output/v6_2_a1"))
    args = parser.parse_args()
    result = run(args.output_dir, args.private_root, args.a1_root)
    print(json.dumps(result["scenes"], indent=2))


if __name__ == "__main__":
    main()
