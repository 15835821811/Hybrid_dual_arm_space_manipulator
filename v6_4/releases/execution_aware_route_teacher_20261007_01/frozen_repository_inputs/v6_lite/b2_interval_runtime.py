"""B.2 bounded interval ten-step torque preview and realized next-start gate."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.b2_prepared_compensated_torque import prepared_compensated_torque
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder
from v6_lite.run_v6_lite import _model_based_servo_torque
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


STEPS = 10
POINT_BUDGET = 255


class RuntimeWorkspace:
    """Private per-model buffers; every preview reloads the integration state.

    getState/setState includes time, act, warm start, applied forces, ctrl,
    equality activity, mocap, userdata and plugin state, beyond qpos/qvel.
    Derived MuJoCo quantities are refreshed at each microstate. Returned
    arrays are borrowed until the next preview, after the current ramp ends.
    """

    def __init__(self, model, robot):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_ids, self.dof_ids = joint_addresses(model, robot)
        _, self.base_dof = free_joint_slices(model, robot.base_joint_name)
        self.mass = np.zeros((model.nv, model.nv), dtype=np.float64)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration_state = np.empty(mujoco.mj_stateSize(model, self.state_spec))
        self.qpos_states = np.empty((STEPS + 1, model.nq))
        self.qvel_states = np.empty((STEPS + 1, model.nv))
        self.torques = np.empty((STEPS, model.nu))
        self.natural_frequency = np.concatenate([np.full(60, 42.), np.full(7, 34.)])
        self.acceleration_limit = np.concatenate([np.full(60, 45.), np.full(7, 70.)])
        self.base_ids = np.arange(self.base_dof.start, self.base_dof.stop, dtype=np.int32)

    def load(self, source):
        mujoco.mj_getState(self.model, source, self.integration_state, self.state_spec)
        mujoco.mj_setState(self.model, self.data, self.integration_state, self.state_spec)


def compensated_torque(model, data, robot, qpos_ids, dof_ids, base_dof,
                       reference_position, reference_velocity,
                       feedforward_acceleration, full_mass):
    """Use the unchanged 67 actuators with implicitfast damping compensation."""
    original, desired, original_diag = _model_based_servo_torque(
        model, data, robot, qpos_ids, dof_ids, base_dof,
        reference_position, reference_velocity, feedforward_acceleration,
        full_mass)
    modified = full_mass + model.opt.timestep * np.diag(model.dof_damping)
    base_ids = np.arange(base_dof.start, base_dof.stop, dtype=np.int32)
    base_acc = -np.linalg.solve(
        modified[np.ix_(base_ids, base_ids)],
        modified[np.ix_(base_ids, dof_ids)] @ desired
        + np.asarray(data.qfrc_bias)[base_ids]
        - np.asarray(data.qfrc_passive)[base_ids])
    required = (
        modified[np.ix_(dof_ids, base_ids)] @ base_acc
        + modified[np.ix_(dof_ids, dof_ids)] @ desired
        + np.asarray(data.qfrc_bias)[dof_ids]
        - np.asarray(data.qfrc_passive)[dof_ids])
    command = np.clip(required, -robot.torque_limits, robot.torque_limits)
    return command, desired, {
        "acceleration_clip_count": original_diag.acceleration_clip_count,
        "acceleration_unclipped_max_rad_s2": (
            original_diag.acceleration_unclipped_max_rad_s2),
        "compensated_torque_saturation_count": int(np.count_nonzero(
            command != required)),
        "torque_change_linf_nm": float(np.max(np.abs(command - original))),
        "required_torque_abs_max_nm": float(np.max(np.abs(required))),
    }


def preview_ramp(model, robot, evaluator, envelope, qpos, qvel, start_time,
                 reference_start, previous_command, endpoint_command,
                 *, physics_period_s, task_period_s,
                 prepared_step: bool = False, workspace=None, source_data=None,
                 capture_execution: bool = False):
    """Predict 11 MuJoCo microstates and all ten 67-channel torque commands."""
    if workspace is None:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.time = start_time
        data.ctrl[:] = 0
        qpos_ids, dof_ids = joint_addresses(model, robot)
        _base_qpos, base_dof = free_joint_slices(model, robot.base_joint_name)
        mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    else:
        if workspace.model is not model or source_data is None:
            raise ValueError("workspace must match model and have a full source state")
        workspace.load(source_data)
        data = workspace.data
        qpos_ids, dof_ids = workspace.qpos_ids, workspace.dof_ids
        base_dof, mass = workspace.base_dof, workspace.mass
    reference = reference_start.copy()
    torques = []
    states = []
    coverage_records = []
    integration_states = []
    execution_records = []
    for step in range(STEPS + 1):
        if capture_execution:
            integration = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
            integration_states.append(integration)
        if prepared_step:
            mujoco.mj_step1(model, data)
        else:
            mujoco.mj_forward(model, data)
        if workspace is None:
            states.append(data.qpos.copy())
        else:
            workspace.qpos_states[step] = data.qpos
            workspace.qvel_states[step] = data.qvel
        actual = data.qpos[evaluator.qpos_ids[:60]]
        projection = evaluator.shape_spec.project_actual_configuration(actual)
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        coverage = envelope.evaluate(data, projection.planner_configuration, base)
        coverage_records.append({
            "servo_substep": step,
            "time_s": float(data.time),
            "status": coverage.status,
            "minimum_margin_m": coverage.min_margin_m,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
            "shape_q": projection.planner_configuration.tolist(),
        })
        if step == STEPS:
            break
        measured = robot.decode_position(data.qpos[qpos_ids])
        ramp = advance_reference(
            reference, previous_command, endpoint_command, step + 1,
            measured, robot.planner_lower, robot.planner_upper,
            physics_period_s=physics_period_s, task_period_s=task_period_s)
        reference = ramp.position
        torque, _desired, diagnostic = prepared_compensated_torque(
            model, data, robot, qpos_ids, dof_ids, base_dof,
            ramp.position, ramp.velocity, ramp.feedforward_acceleration, mass,
            legacy_diagnostic=False, workspace=workspace)
        if capture_execution:
            natural = workspace.natural_frequency
            unclipped = (robot.encode_velocity(ramp.feedforward_acceleration)
                + natural**2 * (robot.encode_position(ramp.position) - data.qpos[qpos_ids])
                + 2 * natural * (robot.encode_velocity(ramp.velocity) - data.qvel[dof_ids]))
            execution_records.append({
                "reference_step": ramp,
                "desired_arm_acceleration": _desired.copy(),
                "diagnostic": {**diagnostic, "acceleration_unclipped_max_rad_s2": float(np.max(np.abs(unclipped)))},
            })
        if workspace is None:
            torques.append(torque.copy())
        else:
            workspace.torques[step] = torque
        data.ctrl[:] = torque
        if prepared_step:
            # step1 prepared position and velocity dependent terms at this
            # state; step2 computes control dependent stages and retains the
            # model's implicitfast integrator.
            mujoco.mj_step2(model, data)
        else:
            mujoco.mj_step(model, data)
    return {
        "torques": np.asarray(torques) if workspace is None else workspace.torques,
        "qpos_states": np.asarray(states) if workspace is None else workspace.qpos_states,
        "qvel_states": None if workspace is None else workspace.qvel_states,
        "coverage": coverage_records,
        "maximum_torque_change_from_legacy_nm": None,
        "integration_states": np.asarray(integration_states) if capture_execution else None,
        "execution_records": execution_records if capture_execution else None,
        "reference_end": reference.copy() if capture_execution else None,
    }, data


def check_next_start(model, robot, verifier, cfg, evaluator, data,
                     endpoint_command):
    """Independently reconstruct all original and interval start rows."""
    projection = evaluator.shape_spec.project_actual_configuration(
        data.qpos[evaluator.qpos_ids[:60]])
    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
    box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
    query = PersistentIntervalDecisionQuery(evaluator.shape_model)
    decision = query.evaluate(
        projection.planner_configuration, base, box, IntervalPartition.uniform(),
        max_point_evaluations=POINT_BUDGET)
    result = {
        "proxy_status": decision.proxy_clearance_status,
        "proxy_lower_m": decision.distance_lower_bound_m,
        "proxy_upper_m": decision.distance_upper_bound_m,
        "point_evaluations": decision.point_evaluation_count,
        "budget_exhausted": decision.budget_exhausted,
        "query_failure_reason": decision.failure_reason,
        "interval_count": decision.interval_count,
        "subspace_residual_linf_rad": projection.residual_linf_rad,
        "frozen_rows_status": "NOT_EVALUATED",
    }
    if not decision.bounds_valid:
        return result
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    independent = ReplayConstraintBuilder(robot, model, verifier.pairs, no_legacy)
    generalized_map, reaction_residual = independent._reaction_map(data)
    planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
    _, _, speed, _ = ramp_velocity_abs_bound(
        robot, cfg, planner_q, endpoint_command)
    ids, _ = select_intervals_by_frozen_reach(
        decision.lower_by_interval_id, decision.partition, evaluator,
        data, cfg, generalized_map, velocity_abs_bound=speed)
    batch = evaluator.evaluate_state(
        data, decision.partition, generalized_map=generalized_map,
        derivative_interval_ids=ids)
    result.update({
        "selected_interval_count": len(ids),
        "reaction_map_residual": reaction_residual,
        "interval_coverage_complete": batch.coverage_complete,
        "interval_well_formed": batch.interval_well_formed,
        "analytic_bound_assumptions_satisfied": (
            batch.analytic_bound_assumptions_satisfied),
    })
    if not (batch.interval_well_formed and batch.coverage_complete
            and batch.analytic_bound_assumptions_satisfied
            and all(row.derivative_status == "SUPPORTED"
                    for row in batch.rows if row.interval_id in ids)):
        result["frozen_rows_status"] = "UNSUPPORTED"
        return result
    original = independent.build(data)
    matrix, lower, _drifts, _gains, sources = combined_rows(
        original, batch, ids, cfg)
    slacks = matrix @ endpoint_command - lower
    index = int(np.argmin(slacks)) if len(slacks) else None
    minimum = float(slacks[index]) if index is not None else None
    result.update({
        "frozen_rows_status": (
            "START_ROWS_SATISFIED" if minimum is None
            or minimum >= -cfg.clearance_rate_tolerance_m_s
            else "START_CLEARANCE_VIOLATION"),
        "row_count": len(slacks),
        "worst_source": sources[index] if index is not None else None,
        "minimum_start_slack_m_s": minimum,
    })
    return result
