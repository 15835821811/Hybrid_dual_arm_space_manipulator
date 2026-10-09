"""Native 500 Hz actor; Python planning coordinator and verified postrun logs."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
from types import SimpleNamespace
import threading
import time

import mujoco
import numpy as np

from v6_lite.runtime_handoff import HandoffBuffer, array_id
from v6_lite.runtime_native_loop import (
    NativeBuffers, NativeHashContext, ERRORS, ATTEMPT_PHASES,
    atomic_read, atomic_write, now, release_now,
)
from v6_lite.runtime_scheduler_environment import ThreadScheduling
from v6_lite.runtime_shared_slot import SharedResultSlot
from v6_lite.runtime_timing import latency_summary, runtime_identity, write_timelines
from v6_lite.runtime_wall_executor import _context, _integration, _new_logs
from v6_lite.run_v6_lite import (
    UncertifiedExecutionError, _write_json, _sha256, _robot_momentum,
    _body_pose_and_twist, ServoDiagnostics, finalize_scenario,
)
from v6_lite.hierarchical_qp import rotation_error_angle_rad, CONTINUUM_EE_OFFSET_M


def _receive_planner_records(connection, records, maximum_records, timeout_s=5.):
    """Bound complete receives, including a sender stalled midway through pickle."""
    events = queue.Queue()
    def receive():
        try:
            header = connection.recv()
            if not isinstance(header, dict):
                raise ValueError("invalid planner transfer header")
            record_count = header.get("record_count")
            if (not isinstance(record_count, int) or isinstance(record_count, bool)
                    or not 0 <= record_count <= maximum_records):
                raise ValueError("invalid planner transfer record count")
            events.put(("header", header))
            for _ in range(record_count):
                events.put(("record", connection.recv()))
            events.put(("complete", None))
        except BaseException as error:
            events.put(("error", error))
    receiver = threading.Thread(target=receive, name="c11-postrun-record-receiver", daemon=True)
    receiver.start()
    header, problem = {}, None
    try:
        while True:
            try:
                kind, value = events.get(timeout=timeout_s)
            except queue.Empty:
                problem = "PLANNER_RECORD_TRANSFER_TIMEOUT"
                break
            if kind == "header":
                header = value
            elif kind == "record":
                records.append(value)
            elif kind == "error":
                problem = f"PLANNER_RECORD_TRANSFER_FAILED: {type(value).__name__}: {value}"
                break
            else:
                break
    finally:
        # Closing the pipe plus terminating its sender in outer cleanup unblocks
        # an incomplete receive; this reader never accesses live model or data.
        if receiver.is_alive():
            try:
                connection.close()
            except BaseException as error:
                problem = problem or f"PLANNER_CONNECTION_CLOSE_FAILED: {error}"
        receiver.join(timeout=.1)
    return header, problem


def _cleanup_native_runtime(worker, connection, supervisor, crypto, gc_was_enabled,
                            *, actor_stopped=True):
    """Every resource gets its cleanup attempt, even when another cleanup fails."""
    problems = []
    def attempt(name, action):
        try:
            action()
        except BaseException as error:
            problems.append(f"{name}: {type(error).__name__}: {error}")
    attempt("planner_join", lambda: worker.join(timeout=2))
    try:
        worker_alive = worker.is_alive()
    except BaseException as error:
        problems.append(f"planner_liveness: {type(error).__name__}: {error}")
        worker_alive = True
    if worker_alive:
        attempt("planner_termination", worker.terminate)
        attempt("planner_termination_join", lambda: worker.join(timeout=2))
        try:
            if worker.is_alive():
                problems.append("planner_termination: planner remains alive")
        except BaseException as error:
            problems.append(f"planner_liveness: {type(error).__name__}: {error}")
    attempt("planner_connection_close", connection.close)
    attempt("supervisor_scheduling_restore", supervisor.restore)
    if gc_was_enabled:
        attempt("cyclic_gc_restore", gc.enable)
    # A live native loop still owns the hash handle; never free it underneath C.
    if actor_stopped:
        attempt("native_hash_close", crypto.close)
    return problems


def _rejected_native_attempt(buffers, epoch, total_steps, rejected_at):
    signals = buffers.view("signals")
    step = int(atomic_read(signals, 4))
    phase = int(atomic_read(signals, 5))
    consumed = bool(atomic_read(signals, 6))
    valid_step = 0 <= step < total_steps
    timing = buffers.view("timings")[step] if valid_step else None
    def stamp(column):
        value = float(timing[column]) if timing is not None else 0.
        return value if np.isfinite(value) and value != 0. else None
    return {"physics_step": step if valid_step else None, "attempted": valid_step,
        "scheduled": epoch+step*.002 if valid_step else None,
        "actual_start": stamp(1), "consumption_check_time": stamp(2),
        "actual_finish": stamp(3), "rejection_observed": rejected_at,
        "operation_phase": ATTEMPT_PHASES.get(phase, f"unknown_{phase}"),
        "torque_consumed": consumed,
        "physical_step_executed": consumed}


def _native_planner_worker(connection, spec, run_config, qp_config, scenario, buffers, error_slot):
    """One planner is the sole publisher; actual source arrives from the C actor."""
    from v6_lite.runtime_wall_executor import _plan_one
    scheduling = ThreadScheduling("planner", run_config.wall_scheduler_policy)
    gc_was_enabled = gc.isenabled()
    crypto = None
    accepted, rejected, problem = [], None, None
    try:
        ctx = _context(spec, run_config, qp_config, scenario)
        warm = _integration(ctx.model, ctx.data)
        stamp = time.perf_counter()
        _plan_one(ctx, {"command_id": 0, "predecessor_id": -1,
            "predicted_start": warm, "prediction_id": "startup",
            "reference_start": spec.planner_zero.copy(), "previous_command": np.zeros(17),
            "simulation_start": 0., "epoch": stamp+60., "execution_start": stamp+60.,
            "source_state_id": array_id(warm), "acquired": stamp, "release": stamp})
        ctx = _context(spec, run_config, qp_config, scenario)
        crypto = NativeHashContext(ctx.model, ctx.source_contract_hash, ctx.source_model_hash)
        blank = NativeBuffers(mp.get_context("spawn"), warm.size, 0)
        release_now(blank.view("ready"), blank.view("slot_meta"), 0, 0, 0.,
                    crypto.publication_counter, crypto.frequency)
        atomic_read(buffers.view("signals"), 1)
        gc.disable()
        connection.send({"ready": True, "private_initialization_solves": 1,
            "thread_scheduling": scheduling.record})
        request = connection.recv()
        if request is None:
            return
        gate = HandoffBuffer(request["executor_model_hash"], request["executor_config_hash"])
        ticks = (ctx.physics_steps+9)//10
        for command in range(ticks):
            if command:
                previous = accepted[-1]["payload"]["packet"]
                source_step = (command-1)*10+1
                while atomic_read(buffers.view("signals"), 1) < source_step:
                    if atomic_read(buffers.view("signals"), 2) < 0 or atomic_read(buffers.view("signals"), 0):
                        raise UncertifiedExecutionError("ACTOR_STOPPED_BEFORE_NEXT_SOURCE")
                # Acquire flag observes completed immutable state/counter stores.
                source = buffers.view("states")[source_step].copy()
                if np.max(np.abs(source-previous.integration_states[1])) > 1e-9:
                    raise UncertifiedExecutionError("OBSERVED_SOURCE_PREDICTION_MISMATCH")
                acquired = float(buffers.view("source_times")[command-1])
                gate.active, gate.pending = previous, None
                gate.last_id = command-1
                request = {"command_id": command, "predecessor_id": command-1,
                    "predicted_start": previous.integration_states[-1].copy(),
                    "prediction_id": array_id(previous.integration_states[-1]),
                    "reference_start": previous.reference_end, "previous_command": previous.endpoint_velocity,
                    "simulation_start": command*.020, "epoch": request["epoch"],
                    "execution_start": request["epoch"]+command*.020,
                    "source_state_id": array_id(source), "acquired": acquired,
                    "source_simulation_s": float(source[0]),
                    "prediction_remaining_microsteps": 9, "release": time.perf_counter()}
            payload = _plan_one(ctx, request)
            rejected = {"payload": payload, "request": request, "observed": time.perf_counter()}
            packet = payload["packet"]
            leaves_hash = hashlib.sha256(json.dumps(payload["partition_leaves"],
                separators=(",", ":")).encode()).hexdigest()
            buffers.stage(packet)
            check = gate.publish(packet, time.perf_counter(),
                source_state_id=request["source_state_id"], partition_id=leaves_hash)
            if not check["accepted"]:
                raise UncertifiedExecutionError(check["reason"])
            published = buffers.release_on_clock(packet, crypto)
            check.update({"time": published, "publish_slack_s": packet.certificate.publish_deadline-published})
            accepted.append({"payload": payload, "request": request,
                "published": published, "publication_check": check})
            rejected = None
    except BaseException as error:
        problem = str(error)
        try:
            error_slot.publish({"error": problem})
        except BaseException:
            pass
        if not accepted and "request" not in locals():
            connection.send({"error": problem})
            return
    finally:
        if crypto is not None:
            crypto.close()
        scheduling.restore()
        if gc_was_enabled:
            gc.enable()
    # Transfer retained records only once wall physics has ended.
    while atomic_read(buffers.view("signals"), 2) in (0, 1):
        if atomic_read(buffers.view("signals"), 0):
            break
        time.sleep(.001)
    connection.send({"record_count": len(accepted), "error": problem,
        "rejected_candidate": rejected})
    for record in accepted:
        connection.send(record)
    connection.close()


def _timelines(records, buffers, count):
    rows, servo = [], []
    if not records:
        return rows, servo
    first_packet = records[0]["payload"]["packet"]
    gate = HandoffBuffer(first_packet.certificate.model_hash, first_packet.certificate.controller_config_hash)
    for command, record in enumerate(records):
        payload, request, published = record["payload"], record["request"], record["published"]
        packet = payload["packet"]
        row = payload["timeline"]
        start, end = row["phases"][0]["start_ns"], row["phases"][-1]["end_ns"]
        source = int(request["acquired"]*1e9)
        row["phases"].insert(0, {"name": "native_source_and_planning_release",
            "start_ns": source, "end_ns": start, "wall_s": (start-source)*1e-9, "thread_cpu_s": None})
        row["phases"].append({"name": "packet_validation_and_direct_native_publication",
            "start_ns": end, "end_ns": int(published*1e9),
            "wall_s": published-end*1e-9, "thread_cpu_s": None})
        row.update({"schema": "v6_2_c11_wall_handoff_v1", "certificate": asdict(packet.certificate),
            "state_acquisition_monotonic_ns": source, "source_simulation_time_s": request["source_simulation_s"],
            "actual_dispatch_monotonic_ns": int(published*1e9),
            "planning_release_monotonic_s": request["release"],
            "prediction_remaining_microsteps": request["prediction_remaining_microsteps"],
            "actual_publish_monotonic_s": published,
            "executor_consumed_publication_monotonic_s": published,
            "dispatch_latency_s": published-request["acquired"],
            "publication_check": record["publication_check"], "servo_dispatch_checks": []})
        row["nested_measurements"]["algorithm_full_latency"] = "planner refresh through original execution validation"
        gate.pending = packet
        for substep in range(min(10, max(0, count-command*10))):
            step = command*10+substep
            scheduled, started, consumed, finished = buffers.view("timings")[step]
            observed = buffers.view("states")[step]
            if substep == 0:
                handoff = gate.handoff(consumed, observed, float(observed[0]))
                row.update({"handoff_check": handoff,
                    "handoff_offset_s": consumed-packet.certificate.execution_start,
                    "first_servo_application_monotonic_s": consumed,
                    "application_start_offset_s": consumed-packet.certificate.execution_start,
                    "remaining_execution_coverage_at_first_application_s": packet.certificate.execution_end-consumed})
            check = gate.servo(consumed, observed, float(observed[0]), substep)
            row["servo_dispatch_checks"].append(check)
            servo.append({"physics_step": step, "scheduled": scheduled, "actual_start": started,
                "consumed": consumed, "finished": finished, "jitter_s": started-scheduled,
                "servo_latency_s": finished-started, "native_guards_included": True})
        rows.append(row)
    return rows, servo


def run_native_scenario(spec, run_config, qp_config, scenario, trace_dir):
    ctx = _context(spec, run_config, qp_config, scenario)
    model, data = ctx.model, ctx.data
    context = mp.get_context("spawn")
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    buffers = NativeBuffers(context, size, ctx.physics_steps)
    crypto = NativeHashContext(model, ctx.source_contract_hash, ctx.source_model_hash)
    warm = NativeBuffers(context, size, 0)
    warm.execute(model, data, crypto, 0., 0, np.empty((0, 2)))
    atomic_read(buffers.view("signals"), 1)
    atomic_write(buffers.view("signals"), 0, 0)
    lo = time.perf_counter()
    native_time = now(crypto.counter, crypto.frequency)
    hi = time.perf_counter()
    if not lo-1e-8 <= native_time <= hi+1e-8:
        crypto.close()
        raise RuntimeError("native QPC and Python monotonic clocks do not share an epoch")
    error_slot = SharedResultSlot(context)
    parent, child = context.Pipe()
    worker = context.Process(target=_native_planner_worker,
        args=(child, spec, run_config, qp_config, scenario, buffers, error_slot),
        name="c11-direct-native-planner")
    worker.start()
    child.close()
    worker_ready = parent.recv()
    if not worker_ready.get("ready"):
        worker.join()
        crypto.close()
        raise UncertifiedExecutionError(str(worker_ready))
    supervisor = ThreadScheduling("supervisor", run_config.wall_scheduler_policy)
    native_ready, arm = threading.Event(), threading.Event()
    native_record, thread_error, epoch_box = {}, [], []
    def actor():
        scheduling = ThreadScheduling("executor", run_config.wall_scheduler_policy)
        native_record.update(scheduling.record)
        native_ready.set()
        try:
            arm.wait()
            if not atomic_read(buffers.view("signals"), 0):
                buffers.execute(model, data, crypto, epoch_box[0], ctx.physics_steps)
        except BaseException as error:
            thread_error.append(error)
            atomic_write(buffers.view("signals"), -11, 2)
        finally:
            scheduling.restore()
    native_thread = threading.Thread(target=actor, name="c11-native-physics-actor", daemon=True)
    native_thread.start()
    native_ready.wait()
    conditions = [supervisor.record, native_record, worker_ready["thread_scheduling"]]
    expected_class = 0x100 if run_config.wall_scheduler_policy == "realtime" else 0x80
    conditions_available = all(r.get("applied") and r.get("process_priority_set")
        and r.get("actual_process_priority_class") == expected_class for r in conditions)
    error, records, header = None, [], {}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    initialized = time.perf_counter()
    acquired = time.perf_counter()
    snapshot = _integration(model, data)
    buffers.view("states")[0] = snapshot
    epoch = acquired+.100
    epoch_box.append(epoch)
    initial_momentum = _robot_momentum(model, data, ctx.base_body_id)
    request = {"command_id": 0, "predecessor_id": -1, "predicted_start": snapshot,
        "prediction_id": "startup", "reference_start": spec.planner_zero.copy(),
        "previous_command": np.zeros(17), "simulation_start": 0., "epoch": epoch,
        "execution_start": epoch, "source_state_id": array_id(snapshot),
        "acquired": acquired, "source_simulation_s": float(snapshot[0]),
        "executor_model_hash": ctx.source_model_hash,
        "executor_config_hash": ctx.controller_config_hash,
        "prediction_remaining_microsteps": 0, "release": time.perf_counter()}
    if not conditions_available:
        error = UncertifiedExecutionError("DECLARED_SCHEDULER_CONDITIONS_UNAVAILABLE")
        atomic_write(buffers.view("signals"), 1, 0)
        parent.send(None)
    else:
        parent.send(request)
    arm.set()
    rejected_at, wall_task_finished = None, None
    transfer_error, cleanup_errors = None, []
    try:
        if error:
            raise error
        while True:
            status = atomic_read(buffers.view("signals"), 2)
            if status < 0:
                raise UncertifiedExecutionError(ERRORS.get(-status, f"NATIVE_ERROR_{status}"))
            if status == 2:
                wall_task_finished = time.perf_counter()
                break
            incoming = error_slot.take()
            if incoming is not None:
                raise UncertifiedExecutionError(incoming[0]["error"])
            if time.perf_counter() > epoch+run_config.duration_s+.1:
                raise UncertifiedExecutionError("NATIVE_ACTOR_WATCHDOG")
            time.sleep(.001)
    except BaseException as caught:
        error, rejected_at = caught, time.perf_counter()
        atomic_write(buffers.view("signals"), 1, 0)
    finally:
        actor_stopped = False
        try:
            native_thread.join(timeout=2)
            actor_stopped = not native_thread.is_alive()
            if actor_stopped and conditions_available:
                header, transfer_error = _receive_planner_records(
                    parent, records, (ctx.physics_steps+9)//10)
        finally:
            cleanup_errors = _cleanup_native_runtime(worker, parent, supervisor,
                crypto, gc_was_enabled, actor_stopped=actor_stopped)
        if not actor_stopped:
            raise RuntimeError("native cancellation failed; postrun model access forbidden")
    if transfer_error or cleanup_errors:
        error = error or UncertifiedExecutionError(transfer_error or "NATIVE_RUNTIME_CLEANUP_FAILED")
        rejected_at = rejected_at or time.perf_counter()
    if thread_error:
        error = thread_error[0]
    count = atomic_read(buffers.view("signals"), 1)
    timing_rows, servo_timing = _timelines(records, buffers, count)
    raw_path = trace_dir.parent/"native"/f"{scenario.scenario_id}.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, integration_states=buffers.view("states")[:count+1],
        applied_torques=buffers.view("applied_torques")[:count],
        timings=buffers.view("timings")[:count], source_times=buffers.view("source_times"),
        final_actual_integration_state=_integration(model, data),
        epoch=epoch, acquired=acquired, initial_qpos=ctx.initial_qpos, initial_qvel=ctx.initial_qvel)
    timing_path = trace_dir.parent/"timing"/f"{scenario.scenario_id}.jsonl"
    write_timelines(timing_path, timing_rows)
    _write_json(timing_path.with_name(f"{scenario.scenario_id}_servo.json"), servo_timing)
    environment = {"schema": "c11_native_runtime_environment_v2",
        "native_abi": "Windows x64; MuJoCo 3.3.2; compiled no-GIL loop; BCrypt SHA256",
        "numerical_thread_pools": ctx.numerical_thread_pools,
        "thread_scheduling": conditions, "controller_config_hash": ctx.controller_config_hash,
        "controller_source_hashes": ctx.controller_source_hashes,
        "private_initialization_solves": worker_ready["private_initialization_solves"],
        "wait_policy": "native absolute QPC grid; busy wait; planner directly observes source and publishes native slot",
        "task_duration_s": run_config.duration_s, "cyclic_gc_during_task": False,
        "native_physics_steps": count, "native_status": atomic_read(buffers.view("signals"), 2),
        "record_transfer_error": transfer_error, "cleanup_errors": cleanup_errors,
        "supervisor_on_per_segment_critical_path": False,
        "native_raw_trace": {"path": raw_path.as_posix(), "sha256": _sha256(raw_path)}}
    _write_json(timing_path.with_name(f"{scenario.scenario_id}_environment.json"), environment)
    if not error and (len(records) != (ctx.physics_steps+9)//10
            or header.get("error") or not all(row.get("handoff_check", {}).get("accepted")
                and all(c["accepted"] for c in row["servo_dispatch_checks"]) for row in timing_rows)):
        error = UncertifiedExecutionError("INCOMPLETE_OR_INVALID_NATIVE_RECORDS")
    if error:
        directory = trace_dir.parent/"failures"
        directory.mkdir(parents=True, exist_ok=True)
        rejected_attempt = _rejected_native_attempt(buffers, epoch, ctx.physics_steps, rejected_at)
        _write_json(directory/f"{scenario.scenario_id}_interval_failure.json", {
            "schema": "c11_native_wall_rejection_v2", "scenario_id": scenario.scenario_id,
            "time_s": float(data.time), "wall_time": rejected_at, "failure_reason": str(error),
            "physics_step": rejected_attempt["physics_step"],
            "physics_steps_executed": count, "trace_recorded_steps": count,
            "next_unexecuted_physics_step": count,
            "rejected_step_executed": rejected_attempt["physical_step_executed"],
            "native_status": environment["native_status"], "next_servo_step_executed": False,
            "continuation_guaranteed": False, "simulation_stop_is_safe_backup": False,
            "rejected_servo_attempt": rejected_attempt,
            "planner_error": header.get("error"),
            "record_transfer_error": transfer_error, "cleanup_errors": cleanup_errors,
            "partial_trace": {"path": raw_path.as_posix(), "sha256": _sha256(raw_path)}})
        rejected = header.get("rejected_candidate")
        _write_json(directory/f"{scenario.scenario_id}_post_rejection_candidate.json", {
            "available_after_worker_stop": rejected is not None, "candidate_consumed_by_physics": False,
            "timeline": rejected["payload"].get("timeline") if rejected else None,
            "certificate": asdict(rejected["payload"]["packet"].certificate) if rejected else None,
            "request": {k:v for k,v in rejected["request"].items() if k not in
                ("predicted_start", "reference_start", "previous_command")} if rejected else None})
        raise UncertifiedExecutionError(str(error))
    packets = {r["payload"]["packet"].certificate.command_id:r["payload"] for r in records}
    return _finalize_native(ctx, spec, run_config, qp_config, scenario, trace_dir,
        buffers, packets, timing_rows, servo_timing, initial_momentum, initialized,
        acquired, epoch, wall_task_finished, environment, worker_ready)


def _record_step(ctx, data, log, payload, segment_step, native_timing, epoch):
    # Original observer formulas, evaluated only in a fresh torque replay.
    model, spec, scenario = ctx.model, ctx.spec, ctx.scenario
    qpos_ids, dof_ids = ctx.qpos_ids, ctx.dof_ids
    target_body_id, rigid_body_id = ctx.target_body_id, ctx.rigid_body_id
    continuum_body_id, base_body_id = ctx.continuum_body_id, ctx.base_body_id
    base_qpos_slice, base_dof_slice = ctx.base_qpos_slice, ctx.base_dof_slice
    execution = payload["execution_records"][segment_step]
    reference_step = execution["reference_step"]
    reference_q, reference_dq = reference_step.position, reference_step.velocity
    feedforward_ddq = reference_step.feedforward_acceleration
    desired_arm_acceleration = execution["desired_arm_acceleration"]
    d = execution["diagnostic"]
    servo_diagnostics = ServoDiagnostics(d["acceleration_unclipped_max_rad_s2"],
        d["acceleration_clip_count"], d["required_torque_abs_max_nm"],
        d["compensated_torque_saturation_count"])
    command_velocity = payload["packet"].endpoint_velocity
    measured_planner_q = spec.decode_position(data.qpos[qpos_ids])
    measured_planner_dq = spec.decode_velocity(data.qvel[dof_ids])
    torque = payload["packet"].torques[segment_step]
    data.ctrl[:] = torque
    mujoco.mj_step(model, data)
    torque_latency = native_timing[3] - native_timing[1]
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
    log["wall_time_since_start_s"].append(native_timing[3] - epoch)
    log["feedforward_acceleration"].append(feedforward_ddq.copy())
    log["desired_arm_acceleration"].append(
        desired_arm_acceleration.copy()
    )
    log["torque"].append(torque.copy())
    log["base_qpos"].append(np.asarray(data.qpos[base_qpos_slice]).copy())
    log["base_twist"].append(np.asarray(data.qvel[base_dof_slice]).copy())
    log["robot_momentum"].append(_robot_momentum(model, data, base_body_id))
    log["torque_latency"].append(torque_latency)


def _finalize_native(ctx, spec, run_config, qp_config, scenario, trace_dir,
        buffers, packets, timing_rows, servo_timing, initial_momentum, initialized,
        acquired, epoch, wall_task_finished, environment, worker_ready):
    log, task_log = _new_logs(ctx.interval_admission)
    task_qpos_trace = []
    replay = mujoco.MjData(ctx.model)
    replay.qpos[:] = ctx.initial_qpos
    replay.qvel[:] = ctx.initial_qvel
    mujoco.mj_forward(ctx.model, replay)
    maximum_error = 0.
    bitwise_equal = True
    for step in range(ctx.physics_steps):
        observed = _integration(ctx.model, replay)
        original = buffers.view("states")[step]
        maximum_error = max(maximum_error, float(np.max(np.abs(observed-original))))
        bitwise_equal &= np.array_equal(observed, original)
        if maximum_error > 1e-9:
            raise UncertifiedExecutionError("POSTRUN_REPLAY_STATE_MISMATCH")
        command, substep = divmod(step, 10)
        payload = packets[command]
        if not np.array_equal(payload["packet"].torques[substep],
                              buffers.view("applied_torques")[step]):
            raise UncertifiedExecutionError("POSTRUN_REPLAY_ACTUAL_TORQUE_MISMATCH")
        if substep == 0:
            task_qpos_trace.append(replay.qpos.copy())
            for key, value in payload["task_row"].items():
                task_log[key].append(value)
            if ctx.pcc_monitor is not None:
                ctx.pcc_monitor.record(float(replay.time), payload["result"])
        _record_step(ctx, replay, log, payload, substep, buffers.view("timings")[step], epoch)
    final = _integration(ctx.model, replay)
    maximum_error = max(maximum_error, float(np.max(np.abs(final-buffers.view("states")[-1]))))
    bitwise_equal &= np.array_equal(final, buffers.view("states")[-1])
    if maximum_error > 1e-9:
        raise UncertifiedExecutionError("POSTRUN_REPLAY_FINAL_STATE_MISMATCH")
    task_qpos_trace.append(replay.qpos.copy())
    payload = finalize_scenario(spec, run_config, qp_config, scenario, trace_dir,
        log=log, task_log=task_log, task_qpos_trace=task_qpos_trace, model=ctx.model,
        verifier=ctx.verifier, qp=SimpleNamespace(solve_count=packets[max(packets)]["solve_count"]),
        pcc_monitor=ctx.pcc_monitor, initial_qpos=ctx.initial_qpos, initial_qvel=ctx.initial_qvel,
        initial_momentum=initial_momentum, base_qpos_slice=ctx.base_qpos_slice,
        physics_steps=ctx.physics_steps, task_stride=ctx.task_stride,
        qpos_write_count_after_initialization=0, qvel_write_count_after_initialization=0,
        scenario_initialization_latency_s=initialized-ctx.scenario_initialization_started,
        timing_rows=timing_rows, source_model_hash=ctx.source_model_hash,
        interval_admission=ctx.interval_admission)
    payload["execution_contract"].update({
        "dispatch_clock_scope": "native_no_GIL_wall_actor_and_independent_next_segment_planner",
        "runtime_identity": runtime_identity(run_config.pcc_mode, spec, ctx.model,
            dispatch_clock_policy="wall_deadline", wall_executor_backend="native"),
        "controller_config_hash": ctx.controller_config_hash,
        "controller_version": "v6_2_c11_native_wall_handoff",
        "published_packets_immutable": True,
        "predecessor_prediction": "cached_native_Phi_of_nine_committed_remaining_steps_bound_to_observed_state_after_first_step",
        "planning_source_phase": "after_first_native_microstep",
        "startup_scope": "new_full_state_after_workspace_init; simulation_unarmed_before_predeclared_task_release",
        "state_acquisition_retimestamps": 0, "thread_scheduling": environment["thread_scheduling"],
        "executor_wait_policy": environment["wait_policy"], "cyclic_gc_during_task": False,
        "qp_numerical_implementation": "c11_numba_same_admm_recurrence_no_fastmath; roundoff may differ",
        "pending_transport": "two bounded RawArray native slots; acquire/release atomics",
        "private_initialization_solves": worker_ready["private_initialization_solves"],
        "native_hot_loop": {"Python_calls": 0, "GIL_held": False, "disk_writes": 0,
            "model_and_payload_SHA256_each_microstep": True,
            "live_data_updates": "67 ctrl values and native mj_step only"},
        "native_raw_trace": environment["native_raw_trace"],
        "observer_log_scope": "postrun fresh MuJoCo torque replay checked against every captured live integration state",
        "postrun_observer_replay": {"integration_states": ctx.physics_steps+1,
            "maximum_state_error": maximum_error, "bitwise_equal": bool(bitwise_equal),
            "predicted_state_transplants": 0}})
    payload["metrics"]["rates_and_latency"].update({
        "servo_jitter": latency_summary([r["jitter_s"] for r in servo_timing], .002),
        "full_servo_cycle": latency_summary([r["servo_latency_s"] for r in servo_timing], .002),
        "acquisition_to_first_application": latency_summary([
            r["first_servo_application_monotonic_s"]-r["certificate"]["source_acquisition_time"]
            for r in timing_rows], .020),
        "publication_to_first_application": latency_summary([
            r["first_servo_application_monotonic_s"]-r["actual_publish_monotonic_s"]
            for r in timing_rows], .020),
        "application_start_offset": latency_summary([r["application_start_offset_s"] for r in timing_rows], .002),
        "wall_task_duration_s": wall_task_finished-epoch,
        "startup_validation_ms": (timing_rows[0]["actual_publish_monotonic_s"]-acquired)*1e3})
    return payload

