"""Private interval-QP rollout with model-preserving implicitfast compensation.

This diagnostic reuses the frozen stage-2 private-loop logic, replacing only
its 67-channel servo torque calculation in a private MuJoCo branch. The same
robot map, torque limits, physics model, ten-step ramp and original collision
constraints remain. It never switches the production online controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from v6_lite import audit_b2_private_rollout as private_loop
from v6_lite.audit_b2_discrete_servo_counterfactual import _compensated_torque
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.run_v6_lite import free_joint_slices


STEPS = 10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _branch(model, robot, evaluator, envelope, qpos, qvel, start_time,
            reference_start, previous_command, endpoint_command,
            *, physics_period_s, task_period_s):
    """Use the unchanged private-loop signature and a compensated torque law."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.time = start_time
    data.ctrl[:] = 0.0
    qpos_ids, dof_ids = joint_addresses(model, robot)
    _base_qpos, base_dof = free_joint_slices(model, robot.base_joint_name)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    reference = reference_start.copy()
    torques = []
    states = []
    coverage_records = []
    saturation_counts = []
    torque_changes = []
    for step in range(STEPS + 1):
        mujoco.mj_forward(model, data)
        states.append(data.qpos.copy())
        actual = data.qpos[evaluator.qpos_ids[:60]]
        projection = evaluator.shape_spec.project_actual_configuration(actual)
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        coverage = envelope.evaluate(data, projection.planner_configuration, base)
        coverage_records.append({
            "servo_substep": step, "time_s": float(data.time),
            "status": coverage.status,
            "minimum_margin_m": coverage.min_margin_m,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
        })
        if step == STEPS:
            break
        measured_q = robot.decode_position(data.qpos[qpos_ids])
        ref = advance_reference(
            reference, previous_command, endpoint_command, step + 1,
            measured_q, robot.planner_lower, robot.planner_upper,
            physics_period_s=physics_period_s,
            task_period_s=task_period_s,
        )
        reference = ref.position
        torque, _acc, diagnostics = _compensated_torque(
            model, data, robot, qpos_ids, dof_ids, base_dof,
            ref.position, ref.velocity, ref.feedforward_acceleration, mass,
        )
        torques.append(torque.copy())
        saturation_counts.append(diagnostics["compensated_torque_saturation_count"])
        torque_changes.append(diagnostics["torque_change_linf_nm"])
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
    return {
        "torques": np.asarray(torques),
        "qpos_states": np.asarray(states),
        "coverage": coverage_records,
        "torque_saturation_count": sum(saturation_counts),
        "maximum_torque_change_from_continuous_servo_nm": max(torque_changes),
    }, data


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    # The frozen loop uses this one imported global for its private branch.
    # Restore it even on failure; the production controller is never patched.
    prior_branch = private_loop._branch
    try:
        private_loop._branch = _branch
        report = private_loop.run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        private_loop._branch = prior_branch
    report["schema"] = "v6_2_b2_discrete_private_multicycle_v1"
    report["scope"] = (
        "private_interval_qp_closed_loop_with_implicitfast_damping_compensated_servo"
    )
    report["servo_torque_variant"] = "M_plus_dt_joint_damping"
    report["robot_mapping_changed"] = False
    report["actuator_torque_limits_changed"] = False
    report["physics_integrator_changed"] = False
    report["production_online_controller_changed"] = False
    report["stage3_admission"] = False
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name)
        for name in ("audit_b2_discrete_private_rollout.py",
                     "audit_b2_discrete_servo_counterfactual.py")
    })
    summary_path = output_dir / "private_rollout_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    document_path = output_dir / "PRIVATE_ROLLOUT.md"
    original_document = document_path.read_text(encoding="utf-8")
    original_document = original_document.replace(
        "及原 67 路力矩伺服推进。",
        "及保留原力矩限制的 67 路隐式积分阻尼补偿伺服推进。",
    ).replace(
        "# B.2 私有多周期区间 QP 力矩诊断",
        "# B.2 隐式积分阻尼补偿的私有多周期区间 QP 力矩诊断",
    )
    document_path.write_text(original_document, encoding="utf-8", newline="\n")
    manifest_path = output_dir / "private_rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"] = _sha(summary_path)
    manifest["document_sha256"] = _sha(document_path)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_00")
    parser.add_argument("--max-ticks", type=int, default=400)
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.scenario_id,
                 args.max_ticks)
    print(json.dumps({key: report[key] for key in (
        "scenario_id", "attempted_ticks", "executed_ticks", "stop_reason",
        "native_replay_max_qpos_error", "strict_online_domain_all_executed_ticks",
        "first_tick_outside_strict_online_domain",
        "maximum_subspace_residual_linf_rad",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
