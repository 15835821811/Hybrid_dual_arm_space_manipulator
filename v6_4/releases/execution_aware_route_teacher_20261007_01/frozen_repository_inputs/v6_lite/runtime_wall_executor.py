"""Fixed 20ms next-segment planning and independent 2ms wall execution.

The worker predicts from the committed predecessor's cached native Phi
trajectory, whose observed start is checked before releasing the request.
It never writes a predicted state into the live executor. Published torque
microcommands use the original compensated servo and exact native-state gates.
The simulation is unarmed during workspace and startup validation; the task
clock starts at a predeclared future release, and no state is retimestamped.
"""
from __future__ import annotations
import hashlib
import gc
import json
import multiprocessing as mp
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import mujoco
import numpy as np
from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import (joint_addresses, free_joint_slices,
    HierarchicalVelocityQP, rotation_error_angle_rad, CONTINUUM_EE_OFFSET_M)
from v6_lite.pcc_monitor import PCCMonitor
from v6_lite.safety_contract import ActionValidation
from v6_lite.runtime_command import model_id, partition_id, state_id
from v6_lite.runtime_timing import CycleTimeline, latency_summary, write_timelines, runtime_identity
from v6_lite.runtime_handoff import (CommandPacket, HandoffCertificate, HandoffBuffer, array_id, payload_id)
from v6_lite.runtime_scheduler_environment import ThreadScheduling
from v6_lite.runtime_shared_slot import SharedResultSlot
from v6_lite.run_v6_lite import (_body_id, _body_pose_and_twist, _robot_momentum,
    _config_sha256, _write_json, _sha256, finalize_scenario, ServoDiagnostics,
    UncertifiedExecutionError)


def _context(spec, run_config, qp_config, scenario):
    from threadpoolctl import threadpool_limits, threadpool_info
    numerical_thread_limit = threadpool_limits(limits=1)
    numerical_thread_pools = threadpool_info()
    scenario_initialization_started = time.perf_counter()
    verification_config = WholeBodyVerificationConfig(
        minimum_clearance=run_config.whole_body_minimum_clearance_m,
        query_distance_max=2.5,
        adaptive_subdivisions=run_config.verification_subdivisions,
        self_collision_ancestor_exclusion_depth=3,
        include_target_satellite_pairs=True,
    )
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, verification_config)
    model = verifier.model
    if abs(float(model.opt.timestep) - run_config.physics_period_s) > 1e-12:
        raise RuntimeError("compiled MuJoCo timestep does not match V6-lite")
    # Collision response is disabled so successful clearance cannot be caused
    # by contact impulses.  Signed distances remain directly queryable.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    qpos_ids, dof_ids = joint_addresses(model, spec)
    base_qpos_slice, base_dof_slice = free_joint_slices(model, spec.base_joint_name)
    target_qpos_slice, target_dof_slice = free_joint_slices(
        model, spec.target_free_joint_name
    )
    data.qpos[qpos_ids] = spec.encode_position(spec.planner_zero)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[target_qpos_slice.start : target_qpos_slice.start + 3] += (
        scenario.target_satellite_position_shift_m
    )
    data.qvel[target_dof_slice.start : target_dof_slice.start + 3] = (
        scenario.target_satellite_linear_velocity_m_s
    )
    data.qvel[target_dof_slice.start + 3 : target_dof_slice.stop] = (
        scenario.target_satellite_angular_velocity_rad_s
    )
    mujoco.mj_forward(model, data)
    initial_qpos = np.asarray(data.qpos).copy()
    initial_qvel = np.asarray(data.qvel).copy()
    qpos_write_count_after_initialization = 0
    qvel_write_count_after_initialization = 0

    interval_admission = None
    interval_next_start_checker = None
    runtime_workspace = None
    if run_config.pcc_mode == "bounded_interval_pcc":
        if qp_config.enable_pcc_cbf or not qp_config.enable_capsule_cbf:
            raise ValueError("bounded interval PCC requires legacy PCC off and capsule CBF on")
        from v6_lite.b2_interval_online_optimized import (
            OptimizedBoundedIntervalAdmission,
            OptimizedBoundedIntervalVelocityQP,
        )
        from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
        evaluator = FixedIntervalCBFEvaluator(spec, model)
        from v6_lite.runtime_compiled_qp import CompiledBoundedIntervalVelocityQP
        qp = CompiledBoundedIntervalVelocityQP(
            spec, model, verifier.pairs, qp_config, evaluator=evaluator)
        interval_admission = OptimizedBoundedIntervalAdmission(spec, model, qp)
        from v6_lite.b2_screened_next_start_rows import ScreenedNextStart
        interval_next_start_checker = ScreenedNextStart(track_instances=False)
        interval_next_start_checker.initialize(
            model, spec, verifier, qp_config, evaluator)
        from v6_lite.b2_interval_runtime import RuntimeWorkspace
        runtime_workspace = RuntimeWorkspace(model, spec)
    else:
        qp = HierarchicalVelocityQP(spec, model, verifier.pairs, qp_config)
    pcc_monitor = (
        PCCMonitor(scenario_id=scenario.scenario_id)
        if qp_config.enable_pcc_cbf or qp_config.enable_capsule_cbf
        else None
    )
    task_stride = int(round(run_config.task_period_s / run_config.physics_period_s))
    physics_steps = int(round(run_config.duration_s / run_config.physics_period_s))
    rigid_body_id = _body_id(model, spec.rigid_tip_body_name)
    continuum_body_id = _body_id(model, spec.continuum_tip_body_name)
    target_body_id = _body_id(model, "target_satellite")
    base_body_id = _body_id(model, "base_of_satelltte")
    reference_q_state = spec.planner_zero.copy()
    segment_start_velocity = np.zeros(17, dtype=np.float64)
    command_velocity = np.zeros(17, dtype=np.float64)
    active_validation: ActionValidation | None = None
    segment_step = 0
    recent_snapshots: deque[dict[str, Any]] = deque(maxlen=3)
    scenario_wall_start = time.perf_counter()
    full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    source_contract_hash = spec.runtime_contract_sha256()
    source_model_hash = model_id(model, source_contract_hash)
    controller_source_hashes = {}
    for name in ("runtime_wall_executor.py", "runtime_handoff.py", "runtime_shared_slot.py",
                 "runtime_compiled_qp.py", "runtime_scheduler_environment.py",
                 "b2_interval_runtime.py", "b2_interval_online_optimized.py",
                 "runtime_native_loop.py", "runtime_native_executor.py"):
        raw = (Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")
        controller_source_hashes[name] = hashlib.sha256(raw).hexdigest()
    controller_config_hash = hashlib.sha256(json.dumps({
        "run_and_qp_config": _config_sha256(run_config, qp_config),
        "controller_sources": controller_source_hashes,
        "version": "v6_2_c11_wall_handoff",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    return SimpleNamespace(**locals())


def _new_logs(interval_admission):
    log: dict[str, list[Any]] = {
        "time": [],
        "rigid_error": [],
        "continuum_error": [],
        "rigid_orientation_error_deg": [],
        "continuum_orientation_error_deg": [],
        "rigid_tip": [],
        "rigid_target": [],
        "rigid_rotation": [],
        "rigid_target_rotation": [],
        "continuum_tip": [],
        "continuum_tip_body_origin": [],
        "continuum_target": [],
        "continuum_target_velocity": [],
        "continuum_rotation": [],
        "continuum_target_rotation": [],
        "planner_q": [],
        "planner_dq": [],
        "command_velocity": [],
        "reference_q": [],
        "reference_velocity": [],
        "measured_planner_q_before_servo": [],
        "measured_velocity_before_servo": [],
        "reference_velocity_error_norm_rad_s": [],
        "reference_joint_limit_clip_count": [],
        "reference_measured_window_clip_count": [],
        "acceleration_clip_count": [],
        "acceleration_unclipped_max_rad_s2": [],
        "torque_saturation_count": [],
        "torque_unclipped_max_nm": [],
        "wall_time_since_start_s": [],
        "feedforward_acceleration": [],
        "desired_arm_acceleration": [],
        "torque": [],
        "base_qpos": [],
        "base_twist": [],
        "robot_momentum": [],
        "torque_latency": [],
    }
    task_log: dict[str, list[Any]] = {
        "time": [],
        "wall_time_since_start_s": [],
        "full_latency": [],
        "solver_latency": [],
        "success": [],
        "iterations": [],
        "active_clearance": [],
        "binding_clearance": [],
        "instantaneous_binding_clearance": [],
        "lookahead_binding_clearance": [],
        "minimum_queried_clearance": [],
        "minimum_constraint_slack": [],
        "avoidance_intervention": [],
        "momentum_map_residual": [],
        "degenerate_clearance_gradients": [],
        "rigid_velocity_residual": [],
        "continuum_velocity_residual": [],
        "rigid_angular_velocity_residual": [],
        "continuum_angular_velocity_residual": [],
        "solver_status": [],
        "solver_candidate": [],
        "selected_command": [],
        "execution_mode": [],
        "failure_reason": [],
        "candidate_clearance_min_slack_m_s": [],
        "candidate_velocity_min_slack_rad_s": [],
        "selected_clearance_min_slack_m_s": [],
        "selected_velocity_min_slack_rad_s": [],
        "ramp_clearance_min_slack_m_s": [],
        "ramp_velocity_min_slack_rad_s": [],
        "lookahead_min_slack_m_s": [],
        "state_age_s": [],
        "target_age_s": [],
        "certificate_expiry_s": [],
        "pcc_clearance": [],
        "capsule_clearance": [],
        "mujoco_continuum_target_clearance": [],
        "pcc_active_clearance": [],
        "capsule_active_clearance": [],
        "pcc_binding_clearance": [],
        "capsule_binding_clearance": [],
        "pcc_avoidance_intervention": [],
        "pcc_mujoco_distance_error": [],
        "pcc_mujoco_gradient_error": [],
        "capsule_mujoco_gradient_error": [],
        "pcc_closest_segment_id": [],
        "pcc_closest_arclength": [],
        "shape_clearance_latency": [],
    }
    task_qpos_trace: list[np.ndarray] = []
    if interval_admission is not None:
        task_log.update({
            "interval_preflight_latency_s": [],
            "interval_qp_only_latency_s": [],
            "interval_preview_latency_s": [],
            "interval_branch_latency_s": [],
            "interval_next_start_latency_s": [],
            "interval_full_control_latency_s": [],
            "interval_point_evaluations": [],
            "interval_selected_rows": [],
            "interval_proxy_lower_m": [],
            "interval_current_envelope_margin_m": [],
            "interval_ramp_minimum_envelope_margin_m": [],
            "interval_realized_next_start_minimum_slack_m_s": [],
        })
    return log, task_log


def _integration(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def _plan_one(ctx, request):
    spec, model, data = ctx.spec, ctx.model, ctx.data
    run_config, qp_config, scenario = ctx.run_config, ctx.qp_config, ctx.scenario
    qp = ctx.qp
    interval_admission = ctx.interval_admission
    interval_next_start_checker = ctx.interval_next_start_checker
    runtime_workspace = ctx.runtime_workspace
    verifier, target_body_id = ctx.verifier, ctx.target_body_id
    pcc_monitor = None
    log, task_log = _new_logs(interval_admission)
    reference_q_state = request["reference_start"]
    command_velocity = request["previous_command"]
    current_time = request["simulation_start"]
    scenario_wall_start = request["epoch"]
    mujoco.mj_setState(model, data, request["predicted_start"], mujoco.mjtState.mjSTATE_INTEGRATION)
    original_warmstart = data.qacc_warmstart.copy()
    task_tick_started = time.perf_counter()
    timeline = CycleTimeline(request["command_id"], current_time)
    def reject_interval(reason, diagnostic):
        raise UncertifiedExecutionError(reason + ": " + json.dumps(diagnostic, default=str))
    mujoco.mj_forward(model, data)
    acquisition_state_id = state_id(model, data) if interval_admission is not None else None
    timeline.mark("state_refresh")
    interval_diagnostic = None
    if interval_admission is not None:
        from v6_lite.b2_interval_online import IntervalPreflightFailure
        interval_started = time.perf_counter()
        try:
            interval_diagnostic = interval_admission.prepare(
                data, command_velocity)
        except IntervalPreflightFailure as error:
            reject_interval(error.reason, error.diagnostic)
        except Exception as error:
            reject_interval("INTERVAL_PREFLIGHT_EXCEPTION", {
                "exception_type": type(error).__name__,
                "exception": str(error),
            })
        interval_diagnostic["preflight_latency_s"] = (
            time.perf_counter() - interval_started)
    timeline.mark("interval_query_and_envelope_preflight")
    (
        rigid_target,
        rigid_target_velocity,
        target_rotation,
        target_angular_velocity,
    ) = _body_pose_and_twist(
        model,
        data,
        target_body_id,
        scenario.grasp_point_target_frame_m,
    )
    rigid_target_rotation = (
        target_rotation @ scenario.grasp_rotation_target_frame
    )
    continuum_target, continuum_target_velocity = scenario.continuum_target.sample(
        current_time
    )
    try:
        result = qp.solve(
            data,
        rigid_target_position=rigid_target,
        rigid_target_velocity=rigid_target_velocity,
        rigid_target_rotation=rigid_target_rotation,
        rigid_target_angular_velocity=target_angular_velocity,
        continuum_target_position=continuum_target,
        continuum_target_velocity=continuum_target_velocity,
        continuum_target_rotation=scenario.continuum_target_rotation_world,
        continuum_target_angular_velocity=np.zeros(3, dtype=np.float64),
        state_timestamp_s=current_time,
        target_timestamp_s=current_time,
            ramp_start_velocity=command_velocity,
            prepared_state=interval_admission is not None,
        )
    except Exception as error:
        if interval_admission is None:
            raise
        reject_interval("INTERVAL_QP_EXCEPTION", {
            "exception_type": type(error).__name__,
            "exception": str(error),
            "preflight": interval_diagnostic,
        })
    timeline.mark("qp_assembly_and_solve")
    solve_finished_time = timeline.last * 1e-9
    if interval_admission is not None:
        # The preflight forwarded this same execution state. The QP
        # reads its geometry and does not mutate qpos or qvel; the
        # preview constructs and prepares its own MjData.
        interval_diagnostic.update({
            "solver_status": result.solver_status,
            "solver_candidate": result.solver_candidate.tolist(),
            "validated_command": (
                result.planner_velocity.tolist()
                if result.planner_velocity is not None else None),
            "execution_mode": result.action_validation.mode.value,
            "action_failure_reason": (
                result.action_validation.failure_reason.value),
            "candidate_valid": result.action_validation.candidate_valid,
            "ramp_valid": result.action_validation.ramp_valid,
        })
    if interval_admission is not None and result.planner_velocity is not None:
        # Simulate the exact requested ten-step torque ramp before its
        # first servo step. A failed realized envelope or next-start
        # check blocks execution, including a nominally valid QP.
        from v6_lite.b2_interval_runtime import preview_ramp
        preview_started = time.perf_counter()
        try:
            data.qacc_warmstart[:] = original_warmstart
            approved_branch, preview_next = preview_ramp(
                model, spec, interval_admission.evaluator,
                interval_admission.envelope,
                data.qpos.copy(), data.qvel.copy(), float(data.time),
                reference_q_state.copy(), command_velocity.copy(),
                result.planner_velocity.copy(),
                physics_period_s=run_config.physics_period_s,
                task_period_s=run_config.task_period_s,
                prepared_step=True,
                workspace=runtime_workspace, source_data=data, capture_execution=True,
            )
            branch_finished = time.perf_counter()
            timeline.mark("ten_step_preview")
            next_start = interval_next_start_checker(
                model, spec, verifier, qp_config,
                interval_admission.evaluator, preview_next,
                result.planner_velocity)
            timeline.mark("next_start_check")
        except Exception as error:
            reject_interval("INTERVAL_RAMP_PREVIEW_EXCEPTION", {
                "exception_type": type(error).__name__,
                "exception": str(error),
                "preflight": interval_diagnostic,
            })
        covered = all(item["status"] == "COVERED_AT_THIS_STATE"
                      for item in approved_branch["coverage"])
        interval_diagnostic["ramp_minimum_envelope_margin_m"] = min(
            item["minimum_margin_m"]
            for item in approved_branch["coverage"])
        interval_diagnostic["preview_latency_s"] = (
            time.perf_counter() - preview_started)
        interval_diagnostic["branch_latency_s"] = (
            branch_finished - preview_started)
        interval_diagnostic["next_start_latency_s"] = (
            time.perf_counter() - branch_finished)
        interval_diagnostic["realized_next_start"] = next_start
        if not covered:
            reject_interval("RAMP_MICROSTATE_NOT_ENVELOPED",
                            interval_diagnostic)
        from v6_lite.pcc_interval_cbf import SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD
        domain = interval_admission.evaluator.shape_spec
        for microstate in approved_branch["coverage"]:
            shape_q = np.asarray(microstate["shape_q"])
            if (microstate["subspace_residual_linf_rad"]
                    > SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD
                    or np.any(shape_q < domain.work_domain_lower_rad)
                    or np.any(shape_q > domain.work_domain_upper_rad)):
                reject_interval("RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN", {
                    **interval_diagnostic,
                    "failed_microstate": microstate,
                })
        if (next_start["proxy_status"]
                != "PROXY_CLEARANCE_AT_LEAST_GATE"
                or next_start["frozen_rows_status"]
                != "START_ROWS_SATISFIED"):
            reject_interval("REALIZED_NEXT_START_UNSUPPORTED",
                            interval_diagnostic)
    if interval_admission is None or result.planner_velocity is None:
        timeline.mark("ten_step_preview")
        timeline.mark("next_start_check")
    timeline.mark("execution_validation")
    if result.planner_velocity is None:
        reject_interval(result.action_validation.failure_reason.value, interval_diagnostic)
    validated = time.perf_counter()
    active_validation = result.action_validation
    interval_full_control_latency_s = validated - task_tick_started
    task_log["time"].append(current_time)
    task_log["wall_time_since_start_s"].append(
        time.perf_counter() - scenario_wall_start
    )
    task_log["full_latency"].append(
        interval_full_control_latency_s if interval_admission is not None
        else result.full_latency_s)
    task_log["solver_latency"].append(result.solver_latency_s)
    if interval_diagnostic is not None:
        task_log["interval_qp_only_latency_s"].append(
            result.full_latency_s)
        task_log["interval_preview_latency_s"].append(
            interval_diagnostic.get("preview_latency_s", np.nan))
        task_log["interval_branch_latency_s"].append(
            interval_diagnostic.get("branch_latency_s", np.nan))
        task_log["interval_next_start_latency_s"].append(
            interval_diagnostic.get("next_start_latency_s", np.nan))
        task_log["interval_full_control_latency_s"].append(
            interval_full_control_latency_s)
        task_log["interval_preflight_latency_s"].append(
            interval_diagnostic["preflight_latency_s"])
        task_log["interval_point_evaluations"].append(
            interval_diagnostic["point_evaluations"])
        task_log["interval_selected_rows"].append(
            interval_diagnostic["selected_interval_count"])
        task_log["interval_proxy_lower_m"].append(
            interval_diagnostic["proxy_lower_m"])
        task_log["interval_current_envelope_margin_m"].append(
            interval_diagnostic["current_envelope_margin_m"])
        task_log["interval_ramp_minimum_envelope_margin_m"].append(
            interval_diagnostic.get("ramp_minimum_envelope_margin_m", np.nan))
        task_log["interval_realized_next_start_minimum_slack_m_s"].append(
            interval_diagnostic.get("realized_next_start", {}).get(
                "minimum_start_slack_m_s", np.nan))
    task_log["success"].append(result.success)
    task_log["iterations"].append(result.solver_iterations)
    task_log["active_clearance"].append(result.active_clearance_constraint_count)
    task_log["binding_clearance"].append(result.binding_clearance_constraint_count)
    task_log["instantaneous_binding_clearance"].append(
        result.instantaneous_binding_constraint_count
    )
    task_log["lookahead_binding_clearance"].append(
        result.lookahead_binding_constraint_count
    )
    task_log["minimum_queried_clearance"].append(
        result.minimum_queried_clearance_m
    )
    task_log["minimum_constraint_slack"].append(result.minimum_constraint_slack)
    task_log["avoidance_intervention"].append(
        result.unconstrained_to_command_norm
    )
    task_log["momentum_map_residual"].append(
        result.reaction_momentum_residual_norm
    )
    task_log["degenerate_clearance_gradients"].append(
        result.degenerate_clearance_gradient_count
    )
    task_log["rigid_velocity_residual"].append(
        result.rigid_velocity_residual_m_s
    )
    task_log["continuum_velocity_residual"].append(
        result.continuum_velocity_residual_m_s
    )
    task_log["rigid_angular_velocity_residual"].append(
        result.rigid_angular_velocity_residual_rad_s
    )
    task_log["continuum_angular_velocity_residual"].append(
        result.continuum_angular_velocity_residual_rad_s
    )
    task_log["solver_status"].append(result.solver_status)
    task_log["solver_candidate"].append(result.solver_candidate.copy())
    task_log["selected_command"].append(
        result.planner_velocity.copy() if result.planner_velocity is not None
        else np.full(17, np.nan)
    )
    task_log["execution_mode"].append(active_validation.mode.value)
    task_log["failure_reason"].append(active_validation.failure_reason.value)
    task_log["candidate_clearance_min_slack_m_s"].append(
        active_validation.candidate_clearance_min_slack_m_s
    )
    task_log["candidate_velocity_min_slack_rad_s"].append(
        active_validation.candidate_velocity_min_slack_rad_s
    )
    task_log["selected_clearance_min_slack_m_s"].append(
        active_validation.selected_clearance_min_slack_m_s
    )
    task_log["selected_velocity_min_slack_rad_s"].append(
        active_validation.selected_velocity_min_slack_rad_s
    )
    task_log["ramp_clearance_min_slack_m_s"].append(
        active_validation.ramp_clearance_min_slack_m_s
    )
    task_log["ramp_velocity_min_slack_rad_s"].append(
        active_validation.ramp_velocity_min_slack_rad_s
    )
    task_log["lookahead_min_slack_m_s"].append(
        active_validation.lookahead_min_slack_m_s
    )
    task_log["state_age_s"].append(active_validation.state_age_s)
    task_log["target_age_s"].append(active_validation.target_age_s)
    task_log["certificate_expiry_s"].append(active_validation.expires_at_s)
    task_log["pcc_clearance"].append(result.pcc_clearance_m)
    task_log["capsule_clearance"].append(result.capsule_clearance_m)
    task_log["mujoco_continuum_target_clearance"].append(
        result.mujoco_continuum_target_clearance_m
    )
    task_log["pcc_active_clearance"].append(
        result.pcc_constraint_active_count
    )
    task_log["capsule_active_clearance"].append(
        result.capsule_constraint_active_count
    )
    task_log["pcc_binding_clearance"].append(
        result.pcc_binding_constraint_count
    )
    task_log["capsule_binding_clearance"].append(
        result.capsule_binding_constraint_count
    )
    task_log["pcc_avoidance_intervention"].append(
        result.pcc_avoidance_intervention
    )
    task_log["pcc_mujoco_distance_error"].append(
        result.pcc_mujoco_distance_error_m
    )
    task_log["pcc_mujoco_gradient_error"].append(
        result.pcc_mujoco_gradient_error_norm
    )
    task_log["capsule_mujoco_gradient_error"].append(
        result.capsule_mujoco_gradient_error_norm
    )
    task_log["pcc_closest_segment_id"].append(
        result.pcc_closest_segment_id
    )
    task_log["pcc_closest_arclength"].append(
        result.pcc_closest_arclength_m
    )
    task_log["shape_clearance_latency"].append(
        result.shape_clearance_latency_s
    )
    if pcc_monitor is not None:
        pcc_monitor.record(current_time, result)

    states = approved_branch["integration_states"]
    c = HandoffCertificate(
        request["command_id"], request["predecessor_id"], request["source_state_id"],
        request["acquired"], request["release"],
        next(p["start_ns"] * 1e-9 for p in timeline.phases if p["name"] == "qp_assembly_and_solve"),
        solve_finished_time, validated, request["execution_start"],
        request["execution_start"], request["execution_start"] + run_config.task_period_s,
        current_time, model_id(model, ctx.source_contract_hash),
        ctx.controller_config_hash, partition_id(qp.partition),
        array_id(states[0]), request["prediction_id"],
        payload_id(states, approved_branch["torques"], result.planner_velocity, approved_branch["reference_end"]))
    packet = CommandPacket.create(c, states, approved_branch["torques"],
                                  result.planner_velocity, approved_branch["reference_end"])
    return {"packet": packet, "result": result,
        "task_row": {k: v[0] for k, v in task_log.items()},
        "execution_records": approved_branch["execution_records"],
        "partition_leaves": [leaf.interval_id for leaf in qp.partition.leaves],
        "timeline": timeline.record(), "solve_count": qp.solve_count}


def _planner_worker(connection, spec, run_config, qp_config, scenario, result_slot):
    scheduling = ThreadScheduling("planner", run_config.wall_scheduler_policy)
    gc_was_enabled = gc.isenabled()
    try:
        ctx = _context(spec, run_config, qp_config, scenario)
        # Load lazy numerical modules and exercise all scratch paths before
        # acquiring the startup state. This private initialization is logged
        # separately and is not a command applied to the executor.
        warm_state = _integration(ctx.model, ctx.data)
        warm_now = time.perf_counter()
        _plan_one(ctx, {"command_id": 0, "predecessor_id": -1,
            "predicted_start": warm_state, "prediction_id": "startup",
            "reference_start": spec.planner_zero.copy(), "previous_command": np.zeros(17),
            "simulation_start": 0., "epoch": warm_now + 60.,
            "execution_start": warm_now + 60., "source_state_id": array_id(warm_state),
            "acquired": warm_now, "release": warm_now})
        # Rebuild solver state so initialization cannot carry an uncommitted
        # candidate or dual warm start into the task.
        ctx = _context(spec, run_config, qp_config, scenario)
        gc.disable()
        connection.send({"ready": True, "private_initialization_solves": 1,
                         "thread_scheduling": scheduling.record})
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                payload = _plan_one(ctx, request)
                result_slot.publish(payload)
            except BaseException as error:
                result_slot.publish({"error": str(error), "type": type(error).__name__})
    except BaseException as error:
        connection.send({"error": str(error), "type": type(error).__name__})
    finally:
        if gc_was_enabled:
            gc.enable()
        scheduling.restore()
        connection.close()


def _wait_until(deadline):
    # Absolute monotonic deadlines; work duration is never added to a period.
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        # The declared 2ms executor experiment uses an active wait. A normal
        # Windows sleep occasionally returned several ms after the boundary.
        # Initialization can sleep; task ticks retain their absolute deadline.
        if remaining > .002:
            time.sleep(max(0, remaining - .001))


def run_wall_scenario(spec, run_config, qp_config, scenario, trace_dir):
    ctx = _context(spec, run_config, qp_config, scenario)
    model, data, verifier = ctx.model, ctx.data, ctx.verifier
    qpos_ids, dof_ids = ctx.qpos_ids, ctx.dof_ids
    base_qpos_slice, base_dof_slice = ctx.base_qpos_slice, ctx.base_dof_slice
    target_body_id, rigid_body_id, continuum_body_id = ctx.target_body_id, ctx.rigid_body_id, ctx.continuum_body_id
    base_body_id = ctx.base_body_id
    source_model_hash = ctx.source_model_hash
    config_hash = ctx.controller_config_hash
    gate = HandoffBuffer(source_model_hash, config_hash)
    log, task_log = _new_logs(ctx.interval_admission)
    task_qpos_trace, timing_rows, servo_timing = [], [], []
    initial_momentum = _robot_momentum(model, data, base_body_id)
    timing_path = trace_dir.parent / "timing" / f"{scenario.scenario_id}.jsonl"
    pcc_monitor = ctx.pcc_monitor
    context = mp.get_context("spawn")
    result_slot = SharedResultSlot(context)
    parent, child = context.Pipe()
    worker = context.Process(target=_planner_worker,
        args=(child, spec, run_config, qp_config, scenario, result_slot), name="c11-next-segment-planner")
    worker.start()
    child.close()
    ready = parent.recv()
    if not ready.get("ready"):
        worker.join()
        raise UncertifiedExecutionError(str(ready))
    scheduling = ThreadScheduling("executor")
    import numba
    import llvmlite
    environment_record = {
        "schema": "c11_runtime_environment_v1", "numba": numba.__version__,
        "llvmlite": llvmlite.__version__, "numerical_thread_pools": ctx.numerical_thread_pools,
        "thread_scheduling": [scheduling.record, ready["thread_scheduling"]],
        "private_initialization_solves": ready["private_initialization_solves"],
        "controller_config_hash": config_hash, "controller_source_hashes": ctx.controller_source_hashes,
        "wait_policy": "active_wait_for_final_2ms", "result_slot_bytes": result_slot.CAPACITY,
        "task_duration_s": run_config.duration_s,
    }
    _write_json(trace_dir.parent / "timing" / f"{scenario.scenario_id}_environment.json", environment_record)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    initialized = time.perf_counter()
    # This model is unarmed before epoch. A new complete state is acquired
    # after both workspaces are ready; its original time is kept in the packet.
    acquired = time.perf_counter()
    snapshot = _integration(model, data)
    scenario_wall_start = epoch = acquired + .100
    source_id = array_id(snapshot)
    def request_for(i, predicted, predecessor, reference, previous, snapshot_id, acquired_at):
        return {"command_id": i, "predecessor_id": predecessor,
            "predicted_start": predicted, "prediction_id": array_id(predicted) if i else "startup",
            "reference_start": reference, "previous_command": previous,
            "simulation_start": i * run_config.task_period_s,
            "epoch": epoch, "execution_start": epoch + i * run_config.task_period_s,
            "source_state_id": snapshot_id, "acquired": acquired_at,
            "source_simulation_s": float(data.time),
            "prediction_remaining_microsteps": 0 if i == 0 else 9,
            "release": time.perf_counter()}
    request = request_for(0, snapshot, -1, spec.planner_zero.copy(), np.zeros(17), source_id, acquired)
    parent.send(request)
    active_payload = pending_payload = None
    def accept_result(payload, published_at):
        nonlocal pending_payload
        now = published_at
        if "error" in payload:
            raise UncertifiedExecutionError(payload["error"])
        leaves_hash = hashlib.sha256(json.dumps(payload["partition_leaves"], separators=(",", ":")).encode()).hexdigest()
        publication = gate.publish(payload["packet"], now,
            source_state_id=request["source_state_id"], partition_id=leaves_hash)
        row = payload["timeline"]
        first_stage_start = row["phases"][0]["start_ns"]
        source_ns = int(request["acquired"] * 1e9)
        row["phases"].insert(0, {"name": "snapshot_ipc_and_planning_release",
            "start_ns": source_ns, "end_ns": first_stage_start,
            "wall_s": (first_stage_start - source_ns) * 1e-9, "thread_cpu_s": None})
        last_stage_end = row["phases"][-1]["end_ns"]
        row["phases"].append({"name": "packet_logging_and_bounded_publication",
            "start_ns": last_stage_end, "end_ns": int(now * 1e9),
            "wall_s": now - last_stage_end * 1e-9, "thread_cpu_s": None})
        row.update({"schema": "v6_2_c11_wall_handoff_v1",
            "certificate": asdict(payload["packet"].certificate),
            "state_acquisition_monotonic_ns": int(request["acquired"] * 1e9),
            "source_simulation_time_s": request["source_simulation_s"],
            "actual_dispatch_monotonic_ns": int(now * 1e9),
            "planning_release_monotonic_s": request["release"],
            "prediction_remaining_microsteps": request["prediction_remaining_microsteps"],
            "actual_publish_monotonic_s": now,
            "executor_consumed_publication_monotonic_s": time.perf_counter(),
            "dispatch_latency_s": now - request["acquired"],
            "publication_check": publication, "servo_dispatch_checks": []})
        row["nested_measurements"]["algorithm_full_latency"] = "planner private-state refresh through execution validation"
        timing_rows.append(row)
        if not publication["accepted"]:
            raise UncertifiedExecutionError(publication["reason"])
        pending_payload = payload
    def persist_failure(error, physics_step):
        rejected_at = time.perf_counter()
        failure_dir = trace_dir.parent / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        partial = failure_dir / f"{scenario.scenario_id}_interval_partial_trace.npz"
        np.savez_compressed(partial, **{k: np.asarray(v) for k,v in log.items()},
            **{f"task_{k}": np.asarray(v) for k,v in task_log.items()},
            task_qpos=np.asarray(task_qpos_trace), initial_qpos=ctx.initial_qpos,
            initial_qvel=ctx.initial_qvel, observed_integration_state=_integration(model, data))
        _write_json(failure_dir / f"{scenario.scenario_id}_interval_failure.json", {
            "schema": "c11_wall_rejection_v1", "scenario_id": scenario.scenario_id,
            "time_s": float(data.time), "wall_time": rejected_at,
            "failure_reason": str(error), "physics_step": physics_step,
            "next_servo_step_executed": False, "continuation_guaranteed": False,
            "physics_steps_executed": int(round(float(data.time) / run_config.physics_period_s)),
            "trace_recorded_steps": len(log["time"]),
            "rejected_servo_attempt": {
                "physics_step": physics_step,
                "scheduled": epoch + max(physics_step, 0) * run_config.physics_period_s,
                "actual_start": servo_started if physics_step >= 0 else None,
                "rejection_observed": rejected_at,
                "torque_consumed": False},
            "last_request": {k: v for k, v in request.items() if k not in
                ("predicted_start", "reference_start", "previous_command")},
            "simulation_stop_is_safe_backup": False,
            "partial_trace": {"path": partial.as_posix(), "sha256": _sha256(partial)}})
        write_timelines(timing_path, timing_rows)
        _write_json(trace_dir.parent / "timing" / f"{scenario.scenario_id}_servo.json", servo_timing)
    physics_step = -1
    rejection_happened = False
    try:
        while time.perf_counter() < epoch:
            if pending_payload is None:
                incoming = result_slot.take()
                if incoming is not None:
                    accept_result(*incoming)
            _wait_until(min(epoch, time.perf_counter() + .001))
        for physics_step in range(ctx.physics_steps):
            scheduled = epoch + physics_step * run_config.physics_period_s
            _wait_until(scheduled)
            servo_started = time.perf_counter()
            segment_step = physics_step % ctx.task_stride
            incoming = result_slot.take()
            if incoming is not None:
                accept_result(*incoming)
            observed = _integration(model, data)
            if model_id(model, ctx.source_contract_hash) != source_model_hash:
                raise UncertifiedExecutionError("LIVE_MODEL_MISMATCH")
            if segment_step == 0:
                handoff = gate.handoff(time.perf_counter(), observed, float(data.time))
                if not handoff["accepted"]:
                    raise UncertifiedExecutionError(str(handoff))
                active_payload, pending_payload = pending_payload, None
                timing_rows[-1]["handoff_check"] = handoff
                timing_rows[-1]["handoff_offset_s"] = handoff["time"] - gate.active.certificate.execution_start
                for key,value in active_payload["task_row"].items():
                    task_log[key].append(value)
                task_qpos_trace.append(data.qpos.copy())
                if pcc_monitor is not None:
                    pcc_monitor.record(float(data.time), active_payload["result"])
            command_velocity = gate.active.endpoint_velocity
            execution = active_payload["execution_records"][segment_step]
            reference_step = execution["reference_step"]
            reference_q, reference_dq = reference_step.position, reference_step.velocity
            feedforward_ddq = reference_step.feedforward_acceleration
            desired_arm_acceleration = execution["desired_arm_acceleration"]
            d = execution["diagnostic"]
            servo_diagnostics = ServoDiagnostics(d["acceleration_unclipped_max_rad_s2"],
                d["acceleration_clip_count"], d["required_torque_abs_max_nm"], d["compensated_torque_saturation_count"])
            measured_planner_q = spec.decode_position(data.qpos[qpos_ids])
            measured_planner_dq = spec.decode_velocity(data.qvel[dof_ids])
            check = gate.servo(time.perf_counter(), observed, float(data.time), segment_step)
            if not check["accepted"]:
                raise UncertifiedExecutionError(str(check))
            active_row = timing_rows[gate.active.certificate.command_id]
            active_row["servo_dispatch_checks"].append(check)
            torque = gate.active.torques[segment_step]
            data.ctrl[:] = torque
            consumed = time.perf_counter()
            if segment_step == 0:
                active_row["first_servo_application_monotonic_s"] = consumed
                active_row["application_start_offset_s"] = consumed - gate.active.certificate.execution_start
                active_row["remaining_execution_coverage_at_first_application_s"] = gate.active.certificate.execution_end - consumed
            if consumed >= scheduled + run_config.physics_period_s:
                raise UncertifiedExecutionError("MISSED_PHYSICS_CONSUMPTION_WINDOW")
            mujoco.mj_step(model, data)
            if np.max(np.abs(_integration(model, data) - gate.active.integration_states[segment_step + 1])) > 1e-9:
                raise UncertifiedExecutionError("NATIVE_REALIZATION_MISMATCH")
            torque_latency = time.perf_counter() - servo_started
            (
                rigid_target,
                _rigid_target_velocity,
                target_rotation,
                _target_angular_velocity,
            ) = _body_pose_and_twist(
                model,
                data,
                target_body_id,
                scenario.grasp_point_target_frame_m,
            )
            rigid_target_rotation = target_rotation @ scenario.grasp_rotation_target_frame
            continuum_target, continuum_target_velocity = scenario.continuum_target.sample(
                float(data.time)
            )
            rigid_tip = np.asarray(data.xpos[rigid_body_id]).copy()
            rigid_rotation = np.asarray(data.xmat[rigid_body_id]).reshape(3, 3).copy()
            continuum_rotation = (
                np.asarray(data.xmat[continuum_body_id]).reshape(3, 3).copy()
            )
            continuum_tip_body_origin = np.asarray(
                data.xpos[continuum_body_id]
            ).copy()
            continuum_tip = (
                continuum_tip_body_origin
                + continuum_rotation @ CONTINUUM_EE_OFFSET_M
            )
            low_q = np.asarray(data.qpos[qpos_ids], dtype=np.float64)
            low_dq = np.asarray(data.qvel[dof_ids], dtype=np.float64)
            log["time"].append(float(data.time))
            log["rigid_error"].append(float(np.linalg.norm(rigid_target - rigid_tip)))
            log["continuum_error"].append(
                float(np.linalg.norm(continuum_target - continuum_tip))
            )
            log["rigid_orientation_error_deg"].append(
                float(
                    np.rad2deg(
                        rotation_error_angle_rad(rigid_target_rotation, rigid_rotation)
                    )
                )
            )
            log["continuum_orientation_error_deg"].append(
                float(
                    np.rad2deg(
                        rotation_error_angle_rad(
                            scenario.continuum_target_rotation_world,
                            continuum_rotation,
                        )
                    )
                )
            )
            log["rigid_tip"].append(rigid_tip)
            log["rigid_target"].append(rigid_target)
            log["rigid_rotation"].append(rigid_rotation)
            log["rigid_target_rotation"].append(rigid_target_rotation)
            log["continuum_tip"].append(continuum_tip)
            log["continuum_tip_body_origin"].append(continuum_tip_body_origin)
            log["continuum_target"].append(continuum_target)
            log["continuum_target_velocity"].append(continuum_target_velocity)
            log["continuum_rotation"].append(continuum_rotation)
            log["continuum_target_rotation"].append(
                scenario.continuum_target_rotation_world.copy()
            )
            log["planner_q"].append(spec.decode_position(low_q))
            log["planner_dq"].append(spec.decode_velocity(low_dq))
            log["command_velocity"].append(command_velocity.copy())
            log["reference_q"].append(reference_q.copy())
            log["reference_velocity"].append(reference_dq.copy())
            log["measured_planner_q_before_servo"].append(measured_planner_q.copy())
            log["measured_velocity_before_servo"].append(measured_planner_dq.copy())
            log["reference_velocity_error_norm_rad_s"].append(
                float(np.linalg.norm(reference_dq - measured_planner_dq))
            )
            log["reference_joint_limit_clip_count"].append(
                reference_step.joint_limit_clip_count
            )
            log["reference_measured_window_clip_count"].append(
                reference_step.measured_window_clip_count
            )
            log["acceleration_clip_count"].append(servo_diagnostics.acceleration_clip_count)
            log["acceleration_unclipped_max_rad_s2"].append(
                servo_diagnostics.acceleration_unclipped_max_rad_s2
            )
            log["torque_saturation_count"].append(servo_diagnostics.torque_saturation_count)
            log["torque_unclipped_max_nm"].append(servo_diagnostics.torque_unclipped_max_nm)
            log["wall_time_since_start_s"].append(time.perf_counter() - scenario_wall_start)
            log["feedforward_acceleration"].append(feedforward_ddq.copy())
            log["desired_arm_acceleration"].append(
                desired_arm_acceleration.copy()
            )
            log["torque"].append(torque.copy())
            log["base_qpos"].append(np.asarray(data.qpos[base_qpos_slice]).copy())
            log["base_twist"].append(np.asarray(data.qvel[base_dof_slice]).copy())
            log["robot_momentum"].append(_robot_momentum(model, data, base_body_id))
            log["torque_latency"].append(torque_latency)

            if segment_step == 0:
                next_i = gate.active.certificate.command_id + 1
                if next_i * ctx.task_stride < ctx.physics_steps:
                    # Acquire a new observed full state after the first real
                    # update. The earlier native realization check binds it
                    # to cached state 1, so Phi covers the nine committed
                    # updates remaining before the fixed next boundary.
                    source_acquired = time.perf_counter()
                    source_snapshot = _integration(model, data)
                    request = request_for(next_i, gate.active.integration_states[-1],
                        gate.active.certificate.command_id, gate.active.reference_end,
                        gate.active.endpoint_velocity, array_id(source_snapshot), source_acquired)
                    parent.send(request)
            servo_timing.append({"physics_step": physics_step, "scheduled": scheduled,
                "actual_start": servo_started, "consumed": consumed,
                "finished": time.perf_counter(), "jitter_s": servo_started - scheduled,
                "servo_latency_s": time.perf_counter() - servo_started})
        _wait_until(epoch + run_config.duration_s)
        wall_task_finished = time.perf_counter()
    except BaseException as error:
        rejection_happened = True
        persist_failure(error, physics_step)
        raise
    finally:
        # Shutdown occurs outside execution, including on failed coverage.
        if worker.is_alive():
            parent.send(None)
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        if rejection_happened:
            # A candidate completing after rejection is diagnostic evidence.
            # It is never admitted and no further physical step is executed.
            late = result_slot.take()
            evidence = {"available_after_worker_stop": late is not None,
                "candidate_consumed_by_physics": False}
            if late is not None:
                candidate, publication_time = late
                evidence.update({"actual_publish_monotonic_s": publication_time,
                    "error": candidate.get("error"), "timeline": candidate.get("timeline"),
                    "certificate": asdict(candidate["packet"].certificate) if "packet" in candidate else None})
            _write_json(trace_dir.parent / "failures" / f"{scenario.scenario_id}_post_rejection_candidate.json", evidence)
        parent.close()
        scheduling.restore()
        if gc_was_enabled:
            gc.enable()
    task_qpos_trace.append(data.qpos.copy())
    write_timelines(timing_path, timing_rows)
    _write_json(trace_dir.parent / "timing" / f"{scenario.scenario_id}_servo.json", servo_timing)
    payload = finalize_scenario(spec, run_config, qp_config, scenario, trace_dir,
        log=log, task_log=task_log, task_qpos_trace=task_qpos_trace, model=model,
        verifier=verifier, qp=SimpleNamespace(solve_count=active_payload["solve_count"]),
        pcc_monitor=pcc_monitor, initial_qpos=ctx.initial_qpos, initial_qvel=ctx.initial_qvel,
        initial_momentum=initial_momentum, base_qpos_slice=base_qpos_slice,
        physics_steps=ctx.physics_steps, task_stride=ctx.task_stride,
        qpos_write_count_after_initialization=0, qvel_write_count_after_initialization=0,
        scenario_initialization_latency_s=initialized - ctx.scenario_initialization_started,
        timing_rows=timing_rows, source_model_hash=source_model_hash,
        interval_admission=ctx.interval_admission)
    payload["execution_contract"].update({"dispatch_clock_scope": "independent_wall_executor_and_next_segment_planner",
        "runtime_identity": runtime_identity(run_config.pcc_mode, spec, model, dispatch_clock_policy="wall_deadline"),
        "controller_config_hash": config_hash,
        "controller_version": "v6_2_c11_wall_handoff", "published_packets_immutable": True,
        "predecessor_prediction": "cached_native_Phi_of_nine_committed_remaining_steps_bound_to_observed_state_after_first_step",
        "planning_source_phase": "after_first_native_microstep",
        "startup_scope": "new_full_state_after_workspace_init; simulation_unarmed_before_predeclared_task_release",
        "state_acquisition_retimestamps": 0,
        "numerical_thread_pools": ctx.numerical_thread_pools,
        "thread_scheduling": [scheduling.record, ready["thread_scheduling"]],
        "executor_wait_policy": "active_wait_for_final_2ms_of_absolute_deadline",
        "qp_numerical_implementation": "c11_numba_same_admm_recurrence_no_fastmath; roundoff may differ",
        "cyclic_gc_during_task": False,
        "pending_transport": "bounded_2MiB_shared_slot; executor_lock_acquire_nonblocking",
        "private_initialization_solves": ready["private_initialization_solves"]})
    payload["metrics"]["rates_and_latency"].update({
        "servo_jitter": latency_summary([r["jitter_s"] for r in servo_timing], .002),
        "full_servo_cycle": latency_summary([r["servo_latency_s"] for r in servo_timing], .002),
        "acquisition_to_first_application": latency_summary([
            r["first_servo_application_monotonic_s"] - r["certificate"]["source_acquisition_time"]
            for r in timing_rows], .020),
        "publication_to_first_application": latency_summary([
            r["first_servo_application_monotonic_s"] - r["actual_publish_monotonic_s"]
            for r in timing_rows], .020),
        "application_start_offset": latency_summary([
            r["application_start_offset_s"] for r in timing_rows], .002),
        "wall_task_duration_s": wall_task_finished - epoch,
        "startup_validation_ms": (timing_rows[0]["actual_publish_monotonic_s"] - acquired) * 1e3})
    return payload
