"""Locate the first off-subspace state in the unchanged 67-torque servo.

Replay the first ten 500 Hz steps of each saved private branch. Record
subspace residuals before and after each step alongside acceleration clipping,
torque saturation, and the realized acceleration. This does not alter control.
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
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _model_based_servo_torque, build_scenarios,
    default_v6_lite_robot_spec, free_joint_slices,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _residual(vector: np.ndarray, shape_spec) -> float:
    actual = np.asarray(vector[:60], dtype=np.float64)
    planner = shape_spec.actuated_to_planner @ actual
    return float(np.max(np.abs(actual - shape_spec.planner_to_actuated @ planner)))


def run(output_dir: Path, private_root: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    shape_spec = default_continuum_model_spec(robot)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    frozen = {item["scenario"]["scenario_id"]: item
              for item in metrics["scenarios"]}
    scenarios = {item.scenario_id: item for item in build_scenarios(robot, run_cfg)}
    report = {
        "schema": "v6_2_b2_servo_subspace_origin_v1",
        "scope": "unchanged_first_ten_private_500hz_servo_steps",
        "input_metrics_sha256": _sha(metrics_path),
        "inputs": {},
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_servo_subspace_origin.py",
                                       "run_v6_lite.py", "execution_ramp.py",
                                       "continuum_model_spec.py")},
        "robot_mapping_changed": False,
        "torque_limit_changed": False,
        "production_controller_changed": False,
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
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
            expected_torque = trace["torque"][:10].copy()
            expected_qpos = trace["qpos_states"][:11].copy()
        endpoint = np.asarray(json.loads(records_path.read_text(
            encoding="utf-8").splitlines()[0])["selected_command"],
                              dtype=np.float64)
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(frozen[scene_id]["scenario"]),
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
        integrator = mujoco.mjtIntegrator(model.opt.integrator).name
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0
        qpos_ids, dof_ids = joint_addresses(model, robot)
        _base_qpos, base_dof = free_joint_slices(model, robot.base_joint_name)
        full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
        reference = robot.planner_zero.copy()
        maximum_torque_error = 0.0
        maximum_state_error = 0.0
        scene_rows = []
        for step in range(10):
            mujoco.mj_forward(model, data)
            maximum_state_error = max(maximum_state_error, float(np.max(
                np.abs(data.qpos - expected_qpos[step]))))
            measured_q = robot.decode_position(data.qpos[qpos_ids])
            ramp = advance_reference(
                reference, np.zeros(17), endpoint, step + 1, measured_q,
                robot.planner_lower, robot.planner_upper,
                physics_period_s=run_cfg.physics_period_s,
                task_period_s=run_cfg.task_period_s,
            )
            reference = ramp.position
            torque, desired_acceleration, diagnostics = _model_based_servo_torque(
                model, data, robot, qpos_ids, dof_ids, base_dof,
                ramp.position, ramp.velocity, ramp.feedforward_acceleration,
                full_mass,
            )
            maximum_torque_error = max(maximum_torque_error, float(np.max(
                np.abs(torque - expected_torque[step]))))
            before_position = _residual(data.qpos[qpos_ids], shape_spec)
            before_velocity = _residual(data.qvel[dof_ids], shape_spec)
            desired_acceleration_residual = _residual(desired_acceleration,
                                                      shape_spec)
            velocity_before_step = data.qvel[dof_ids].copy()
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            realized_acceleration_residual = _residual(data.qacc[dof_ids],
                                                       shape_spec)
            finite_step_acceleration = (
                data.qvel[dof_ids] - velocity_before_step
            ) / run_cfg.physics_period_s
            finite_step_acceleration_residual = _residual(
                finite_step_acceleration, shape_spec)
            integration_acceleration_defect = float(np.max(np.abs(
                finite_step_acceleration - data.qacc[dof_ids])))
            after_position = _residual(data.qpos[qpos_ids], shape_spec)
            after_velocity = _residual(data.qvel[dof_ids], shape_spec)
            row = {
                "scene_id": scene_id, "servo_step": step,
                "before_position_residual_rad": before_position,
                "before_velocity_residual_rad_s": before_velocity,
                "desired_acceleration_residual_rad_s2":
                    desired_acceleration_residual,
                "realized_acceleration_residual_rad_s2":
                    realized_acceleration_residual,
                "finite_step_acceleration_residual_rad_s2":
                    finite_step_acceleration_residual,
                "integration_acceleration_defect_rad_s2":
                    integration_acceleration_defect,
                "after_position_residual_rad": after_position,
                "after_velocity_residual_rad_s": after_velocity,
                "acceleration_clip_count": diagnostics.acceleration_clip_count,
                "torque_saturation_count": diagnostics.torque_saturation_count,
                "torque_unclipped_max_nm": diagnostics.torque_unclipped_max_nm,
                "torque_abs_max_nm": float(np.max(np.abs(torque))),
            }
            rows.append(row)
            scene_rows.append(row)
        maximum_state_error = max(maximum_state_error, float(np.max(
            np.abs(data.qpos - expected_qpos[10]))))
        report["scenes"].append({
            "scene_id": scene_id,
            "record_count": len(scene_rows),
            "mujoco_integrator": integrator,
            "continuum_damping_min": float(np.min(model.dof_damping[
                dof_ids[:60]])),
            "continuum_damping_max": float(np.max(model.dof_damping[
                dof_ids[:60]])),
            "first_step_outside_1e_10_rad": next(
                (row["servo_step"] + 1 for row in scene_rows
                 if row["after_position_residual_rad"] > 1e-10), None),
            "maximum_first_tick_position_residual_rad": max(
                row["after_position_residual_rad"] for row in scene_rows),
            "torque_saturation_step_count": sum(
                row["torque_saturation_count"] > 0 for row in scene_rows),
            "acceleration_clip_step_count": sum(
                row["acceleration_clip_count"] > 0 for row in scene_rows),
            "maximum_saved_torque_error_nm": maximum_torque_error,
            "maximum_saved_qpos_error": maximum_state_error,
        })
        print(f"[b2-subspace-origin] {scene_id}: "
              f"first off-subspace servo step "
              f"{report['scenes'][-1]['first_step_outside_1e_10_rad']}",
              flush=True)
    report["record_count"] = len(rows)
    report["maximum_saved_torque_error_nm"] = max(
        item["maximum_saved_torque_error_nm"] for item in report["scenes"])
    report["maximum_saved_qpos_error"] = max(
        item["maximum_saved_qpos_error"] for item in report["scenes"])
    records_path = output_dir / "servo_subspace_origin_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report["records_sha256"] = _sha(records_path)
    summary_path = output_dir / "servo_subspace_origin_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": "v6_2_b2_servo_subspace_origin_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "records_sha256": _sha(records_path)}
    with (output_dir / "servo_subspace_origin_manifest.json").open(
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
    print(json.dumps({"scenes": result["scenes"],
                      "maximum_saved_torque_error_nm":
                      result["maximum_saved_torque_error_nm"]}, indent=2))


if __name__ == "__main__":
    main()
