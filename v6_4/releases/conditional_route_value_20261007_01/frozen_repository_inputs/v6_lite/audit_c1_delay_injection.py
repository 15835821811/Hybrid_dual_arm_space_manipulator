"""Deterministic scheduler test: physics evolves on a separate live state.

Planning reads an acquisition snapshot. The live MuJoCo state keeps the last
torque during injected service delay. This held torque is explicitly NOT a
validated backup. Delays below one physical tick are retained in scheduler
time; fixed 500 Hz physics advances only at complete ticks. No preview state
is ever written into the live state. This tests rejection, not delay safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.b2_interval_online_optimized import OptimizedBoundedIntervalAdmission, OptimizedBoundedIntervalVelocityQP
from v6_lite.b2_interval_runtime import RuntimeWorkspace, preview_ramp
from v6_lite.b2_screened_next_start_rows import ScreenedNextStart
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.runtime_command import CommandCertificate, DispatchGate, partition_id, state_id
from v6_lite.run_v6_lite import (V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec,
                                  _body_id, _body_pose_and_twist)


PHASES = ("qp", "preview", "next_start", "before_publish")
DELAYS_MS = (0, 5, 10, 20, 40, 80)


def run(run_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "c1_predeclared_delay_protocol_v1", "phases": PHASES,
            "delay_ms": DELAYS_MS, "nominal_service_ms": 12,
            "clock": "deterministic virtual monotonic scheduler",
            "physics_period_s": .002, "sample_task_tick": 20,
            "during_delay": "unverified last torque held; not a safe backup",
            "extra_cases": ["consecutive_40ms_x3", "stale_target", "out_of_order"],
            "selection_policy": "all cases retained"}
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    metrics = json.loads((run_dir / "v6_lite_metrics.json").read_text())
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    scenario = build_scenarios(robot, run_cfg)[0]
    verifier = WholeBodyCollisionVerifier(robot, _obstacles(metrics["scenarios"][0]["scenario"]),
        WholeBodyVerificationConfig(minimum_clearance=.005, query_distance_max=2.5,
            adaptive_subdivisions=4, self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True))
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    source = mujoco.MjData(model)
    trace_path = run_dir / "traces" / "v6_lite_scenario_00.npz"
    with np.load(trace_path) as trace:
        source.qpos[:] = trace["initial_qpos"]
        source.qvel[:] = trace["initial_qvel"]
        for torque in trace["torque"][:200]:
            source.ctrl[:] = torque
            mujoco.mj_step(model, source)
        reference_start = trace["reference_q"][199].copy()
        previous = trace["task_selected_command"][19].copy()
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    workspace = RuntimeWorkspace(model, robot)
    acquisition = np.empty_like(workspace.integration_state)
    mujoco.mj_forward(model, source)
    mujoco.mj_getState(model, source, acquisition, workspace.state_spec)
    model_hash = robot.runtime_contract_sha256()
    target_body = _body_id(model, "target_satellite")
    from v6_lite.hierarchical_qp import free_joint_slices
    target_slice, _ = free_joint_slices(model, robot.target_free_joint_name)
    robot_ids = evaluator.qpos_ids
    rows = []

    cases = [(phase, delay, 1, None) for phase in PHASES for delay in DELAYS_MS]
    cases += [("qp", 40, 3, "consecutive_40ms_x3"),
              ("before_publish", 0, 1, "stale_target"),
              ("before_publish", 0, 1, "out_of_order")]
    for phase, delay, repeats, special in cases:
        planning, live = mujoco.MjData(model), mujoco.MjData(model)
        for data in (planning, live):
            mujoco.mj_setState(model, data, acquisition, workspace.state_spec)
            mujoco.mj_forward(model, data)
        qp = OptimizedBoundedIntervalVelocityQP(robot, model, verifier.pairs, cfg, evaluator=evaluator)
        qp.previous_velocity[:] = previous
        admission = OptimizedBoundedIntervalAdmission(robot, model, qp)
        checker = ScreenedNextStart(track_instances=False)
        checker.initialize(model, robot, verifier, cfg, evaluator)
        elapsed, evolved_steps = 0., 0
        events = []
        def stage(name):
            nonlocal elapsed, evolved_steps
            # Only the injected portion evolves physics. The nominal 12 ms
            # models snapshot service overhead; zero-delay success is a
            # scheduler control, not evidence of real-time physical motion.
            elapsed += .003
            if phase == name:
                for _ in range(repeats):
                    elapsed += delay * .001
                    steps = int(np.floor(delay * .001 / .002 + 1e-9))
                    for _ in range(steps):
                        mujoco.mj_step(model, live)
                    evolved_steps += steps
            events.append({"phase": name, "scheduler_elapsed_s": elapsed,
                           "live_simulation_time_s": float(live.time)})
        admission.prepare(planning, previous)
        pose, velocity, rotation, omega = _body_pose_and_twist(
            model, planning, target_body, scenario.grasp_point_target_frame_m)
        target, target_velocity = scenario.continuum_target.sample(float(planning.time))
        result = qp.solve(planning, rigid_target_position=pose, rigid_target_velocity=velocity,
            rigid_target_rotation=rotation @ scenario.grasp_rotation_target_frame,
            rigid_target_angular_velocity=omega, continuum_target_position=target,
            continuum_target_velocity=target_velocity,
            continuum_target_rotation=scenario.continuum_target_rotation_world,
            continuum_target_angular_velocity=np.zeros(3), state_timestamp_s=float(planning.time),
            target_timestamp_s=float(planning.time), ramp_start_velocity=previous, prepared_state=True)
        if result.planner_velocity is None:
            failure = {"reason": "fixture_qp_not_admissible", "phase": phase,
                       "delay_ms": delay, "solver_status": result.solver_status,
                       "validation_reason": result.action_validation.failure_reason.value,
                       "rows_completed": rows}
            (output_dir / "run_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError(f"delay fixture QP: {failure['solver_status']}, {failure['validation_reason']}")
        stage("qp")
        solve_finished = 100. + elapsed
        branch, next_data = preview_ramp(model, robot, evaluator, admission.envelope,
            planning.qpos.copy(), planning.qvel.copy(), float(planning.time), reference_start,
            previous, result.planner_velocity, physics_period_s=.002, task_period_s=.02,
            prepared_step=True, workspace=workspace, source_data=planning)
        stage("preview")
        next_start = checker(model, robot, verifier, cfg, evaluator, next_data, result.planner_velocity)
        stage("next_start")
        if (next_start["frozen_rows_status"] != "START_ROWS_SATISFIED" or
                not all(c["status"] == "COVERED_AT_THIS_STATE" for c in branch["coverage"])):
            raise RuntimeError("delay fixture ramp is not independently admissible")
        certificate = CommandCertificate.issue(command_id=20, source_state_id=state_id(model, planning),
            source_model_hash=model_hash, source_partition_id=partition_id(qp.partition),
            command=result.planner_velocity, acquired=100., source_simulation_s=float(planning.time),
            solve_finished=solve_finished, validated=100. + elapsed)
        stage("before_publish")
        gate = DispatchGate()
        if special == "stale_target":
            certificate = replace(certificate, target_acquisition_time=99.97)
        if special == "out_of_order":
            gate.last_command_id = 21
        verdict = gate.check(certificate, now=100. + elapsed, model_hash=model_hash,
            partition_hash=partition_id(qp.partition), command=result.planner_velocity,
            observed_state_id=state_id(model, live), observed_simulation_s=float(live.time))
        robot_change = float(np.max(np.abs(live.qpos[robot_ids] - planning.qpos[robot_ids])))
        target_change = float(np.max(np.abs(live.qpos[target_slice] - planning.qpos[target_slice])))
        expected_accept = delay == 0 and special is None
        check = verdict["accepted"] == expected_accept
        if delay:
            check = check and robot_change > 0 and target_change > 0 and evolved_steps > 0
        if verdict["accepted"]:
            live.ctrl[:] = branch["torques"][0]
            mujoco.mj_step(model, live)
        consecutive_attempts = []
        if special == "consecutive_40ms_x3":
            # Three separately delivered old packets, with no lifetime renewal.
            old_certificate = CommandCertificate(**rows[0]["certificate"])
            mujoco.mj_setState(model, live, acquisition, workspace.state_spec)
            consecutive_gate = DispatchGate()
            for attempt in range(1, 4):
                for _ in range(20):
                    mujoco.mj_step(model, live)
                rejected = consecutive_gate.check(old_certificate, now=100.012 + .04 * attempt,
                    model_hash=model_hash, partition_hash=partition_id(qp.partition),
                    command=result.planner_velocity, observed_state_id=state_id(model, live),
                    observed_simulation_s=float(live.time))
                consecutive_attempts.append(rejected)
            check = check and all(r["reason"] == "EXPIRED_COMMAND_WALL_CLOCK" for r in consecutive_attempts)
        rows.append({"phase": phase, "delay_ms": delay, "repeats": repeats, "special": special,
            "events": events, "physics_evolved_steps_before_dispatch": evolved_steps,
            "consecutive_attempts": consecutive_attempts,
            "robot_qpos_change": robot_change, "target_qpos_change": target_change,
            "certificate": certificate.to_dict(), "verdict": verdict,
            "candidate_torque_consumed": verdict["accepted"],
            "outcome": "on_time_exact_start" if expected_accept else "cannot_guarantee_continuation",
            "safe_recovery_claimed": False, "passed_rejection_test": bool(check)})
    report = {"schema": "c1_delay_injection_report_v1", "all_passed": all(r["passed_rejection_test"] for r in rows),
              "case_count": len(rows), "rows": rows,
              "reference_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              "claim_scope": "delay rejection with evolving physics, no validated backup or latency safety",
              "nominal_service_evolves_physics": False,
              "injected_service_evolves_physics": True}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = run(args.run_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ("all_passed", "case_count", "claim_scope")}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)
