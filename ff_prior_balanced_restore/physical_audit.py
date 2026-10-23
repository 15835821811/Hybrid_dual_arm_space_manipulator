"""Independent ctrl replay and native-state inspection; no goal replanning."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.geometry import CollisionBackend
from ff_prior.contracts import EXECUTION_GATES, all_gates
from .tracker import integration_state
from .native_budget import NativeBudgetError

def replay_agrees(maxima,atol):
    return ('integration' in maxima and bool(maxima)
            and all(np.isfinite(v) and 0<=v<=atol for v in maxima.values()))


def pose_errors(actual, goals):
    actual, goals = np.asarray(actual), np.asarray(goals)
    position = np.linalg.norm(actual[..., :3, 3] - goals[..., :3, 3], axis=-1)
    relative = np.swapaxes(goals[..., :3, :3], -1, -2) @ actual[..., :3, :3]
    rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(position.shape)
    return position, rotation


def replay(robot, initial, arrays, atol, wall_deadline=None, output=None, *, native_budget):
    import time
    data = initial.reset(robot)
    recorded={k:[] for k in ("qpos","qvel","act","time","qacc_warmstart","integration")}
    maxima = {"qpos": 0.0, "qvel": 0.0, "act": 0.0, "time": 0.0, "qacc_warmstart": 0.0, "integration": 0.0}
    if len(arrays["qpos"]) != len(arrays["ctrl"]) + 1:
        return {"passed": False, "reason": "state/control length mismatch", "physics_steps": 0}
    stop_reason = None
    native_budget.begin_phase('replay', len(arrays['ctrl']))
    try:
        for k in range(len(arrays["ctrl"]) + 1):
            # Mirror forward evaluation in the recorded tracker; this only updates
            # derived quantities and solver cache, never overwrites physical state.
            robot.forward(data)
            for name in maxima:
                val = integration_state(robot,data) if name=="integration" else np.asarray(getattr(data, name))
                recorded[name].append(val.copy())
                delta = np.max(np.abs(val - arrays[name][k]), initial=0.0)
                maxima[name] = max(maxima[name], float(delta)) if np.isfinite(delta) else float("inf")
            if k < len(arrays["ctrl"]):
                if wall_deadline is not None and time.time()>=wall_deadline:
                    stop_reason = 'BUDGET_EXHAUSTED'
                    break
                try:
                    native_budget.before_step('replay')
                except NativeBudgetError as error:
                    stop_reason = str(error)
                    break
                data.ctrl[:] = arrays["ctrl"][k]
                mujoco.mj_step(robot.model, data)
                robot.calls["physics_steps"] += 1
                native_budget.after_step('replay')
    finally:
        native_budget.finish_phase('replay')
        if output is not None:
            from ff_prior.storage import atomic_npz
            atomic_npz(output / "replay_trace.npz", **{k:np.asarray(v) for k,v in recorded.items()},
                       ctrl=arrays["ctrl"][:native_budget.completed['replay']])
    complete = (stop_reason is None and native_budget.completed['replay'] == len(arrays['ctrl'])
                and len(recorded['integration']) == len(arrays['ctrl'])+1)
    return {"passed": complete and replay_agrees(maxima,atol), "maximum_errors": maxima,
            "atol": atol, "physics_steps": native_budget.completed['replay'],
            "planned_prefix_steps": len(arrays['ctrl']), "stop_reason": stop_reason,
            "complete_recorded_prefix": complete, "copies_actual_state_each_step": False}


def validate(robot, initial, reference, arrays, metadata, goal_poses, config, wall_deadline=None, output=None, *, native_budget):
    import time
    m = robot.model
    collision = CollisionBackend(robot)
    data = initial.reset(robot)
    gates = {name: False for name in EXECUTION_GATES}
    gates["finite"] = all(np.isfinite(value).all() for value in arrays.values())
    if not gates["finite"]:
        return {"level": "EXECUTION_FAILED", "gates": gates, "reason": "NONFINITE"}
    gates["identity"] = (metadata.get("initial_state_id") == initial.state_id and
                         metadata.get("reference_hash") == reference.reference_hash and
                         metadata.get("scene_model_hash") == initial.model_hash)
    replay_report = replay(robot, initial, arrays, config.replay_atol,wall_deadline,output,native_budget=native_budget)
    gates["independent_replay"] = bool(replay_report["passed"])
    times = arrays["time"] - arrays["time"][0]
    expected_steps = int(round((reference.duration + config.hold_s) / m.opt.timestep))
    gates["complete_native_time"] = (len(times) == expected_steps + 1 and
        metadata.get("status") == "FINISHED" and np.allclose(np.diff(times), m.opt.timestep, atol=1e-12, rtol=0) and
        abs(times[-1] - reference.duration - config.hold_s) < 1e-9 and times[-1] <= config.deadline_s + 1e-9)
    minimum, minimum_pair, queries, checked = float("inf"), None, 0, 0
    clearance_path = []; scan_states=[]; scan_reports=[]
    ee, momentum, momentum_exchange = [], [], []
    previous_geom=None;max_geometry_step=0.
    def geometry_snapshot():
        return data.geom_xpos.copy(),data.geom_xmat.copy().reshape(-1,3,3)
    def extent(a,b):
        return float(np.max(np.linalg.norm(b[0]-a[0],axis=-1)+np.linalg.norm(b[1]-a[1],axis=(-1,-2))*m.geom_rbound))
    for k, qpos in enumerate(arrays["qpos"]):
        native_budget.check()
        if wall_deadline is not None and time.time()>=wall_deadline:
            raise RuntimeError("BUDGET_EXHAUSTED")
        data.qpos[:] = qpos
        data.qvel[:] = arrays["qvel"][k]
        robot.forward(data)
        ee.append(robot.ee(data))
        current_geom=geometry_snapshot()
        mass = robot.mass(data)
        momentum.append((mass @ data.qvel)[robot.base_dofs])
        momentum_exchange.append(mass[np.ix_(robot.base_dofs, robot.dof_ids)]@data.qvel[robot.dof_ids])
        report = collision.scan(data, config.actual_clearance_m)
        scan_states.append(data.qpos.copy()); scan_reports.append({"state":k,"fraction":0.,"minimum_clearance_m":report["minimum_clearance_m"],"pair":report["minimum_pair"]})
        clearance_path.append(report["minimum_clearance_m"])
        queries += report["query_count"]
        checked += 1
        if report["minimum_clearance_m"] < minimum:
            minimum, minimum_pair = report["minimum_clearance_m"], report["minimum_pair"]
        if k:
            tangent = np.empty(m.nv)
            mujoco.mj_differentiatePos(m, tangent, 1.0, arrays["qpos"][k-1], qpos)
            for j in range(1, config.subdivisions):
                data.qpos[:] = arrays["qpos"][k-1]
                mujoco.mj_integratePos(m, data.qpos, tangent, j / config.subdivisions)
                robot.forward(data)
                mid_geom=geometry_snapshot()
                max_geometry_step=max(max_geometry_step,extent(previous_geom,mid_geom))
                previous_geom=mid_geom
                report = collision.scan(data, config.actual_clearance_m)
                scan_states.append(data.qpos.copy()); scan_reports.append({"state":k-1,"fraction":j/config.subdivisions,"minimum_clearance_m":report["minimum_clearance_m"],"pair":report["minimum_pair"]})
                queries += report["query_count"]
                checked += 1
                if report["minimum_clearance_m"] < minimum:
                    minimum, minimum_pair = report["minimum_clearance_m"], report["minimum_pair"]
        if previous_geom is not None:
            max_geometry_step=max(max_geometry_step,extent(previous_geom,current_geom))
        previous_geom=current_geom
    gates["geometry_step"]=max_geometry_step<=.002+1e-12
    if output is not None:
        from ff_prior.storage import atomic_json,atomic_npz
        atomic_npz(output/"geometry_scans.npz",qpos=np.asarray(scan_states),minima=np.asarray([v["minimum_clearance_m"] for v in scan_reports]))
        atomic_json(output/"geometry_scans.json",scan_reports)
    gates["whole_body_clearance"] = minimum >= config.actual_clearance_m
    position, rotation = pose_errors(np.asarray(ee), goal_poses)
    hold = times >= reference.duration - 1e-9
    gates["dual_goal_hold"] = bool(gates["complete_native_time"] and hold.any() and
        np.all(position[hold] <= config.position_error_m) and np.all(rotation[hold] <= config.rotation_error_rad))
    low_q = arrays["qpos"][:, robot.qpos_ids]
    low_v = arrays["qvel"][:, robot.dof_ids]
    actual_q = low_q @ robot.Bplus.T
    actual_v = low_v @ robot.Bplus.T
    actual_a = np.diff(actual_v, axis=0) / m.opt.timestep
    low_a = np.diff(low_v, axis=0) / m.opt.timestep
    ref_q = reference.evaluate(times)
    joint_range = m.jnt_range[robot.joints]
    limited = m.jnt_limited[robot.joints].astype(bool)
    gates["joint_limits"] = bool(np.all(actual_q >= robot.lower) and np.all(actual_q <= robot.upper) and
        np.all(ref_q >= robot.lower) and np.all(ref_q <= robot.upper) and
        np.all(low_q[:, limited] >= joint_range[limited, 0]) and np.all(low_q[:, limited] <= joint_range[limited, 1]))
    spec = robot.spec
    gates["derivative_limits"] = bool(np.all(np.abs(actual_v) <= spec.planner_velocity_limits) and
        np.all(np.abs(actual_a) <= spec.planner_acceleration_limits) and
        np.all(np.abs(low_v) <= np.abs(robot.B) @ spec.planner_velocity_limits) and
        np.all(np.abs(low_a) <= np.abs(robot.B) @ spec.planner_acceleration_limits) and
        np.all(np.abs(reference.evaluate(times, 1)) <= spec.planner_velocity_limits) and
        np.all(np.abs(reference.evaluate(times, 2)) <= spec.planner_acceleration_limits))
    gates["raw_torque_limits"] = bool(np.all(np.abs(arrays["raw_torque"]) <= spec.torque_limits + 1e-12) and
        np.allclose(arrays["ctrl"][:, robot.actuator_ids],
                    np.clip(arrays["raw_torque"], -spec.torque_limits, spec.torque_limits), rtol=0, atol=1e-12))
    tracking = float(np.max(np.abs(actual_q - ref_q)))
    manifold = float(np.max(np.abs(low_q - actual_q @ robot.B.T)))
    gates["tracking"] = tracking <= config.tracking_error_rad
    gates["manifold"] = manifold <= config.manifold_residual_rad
    momentum_error = float(np.max(np.abs(momentum)))
    momentum, momentum_exchange = np.asarray(momentum), np.asarray(momentum_exchange)
    peak_groups = np.array([np.max(np.linalg.norm(momentum[:, s], axis=1)) for s in (slice(0, 3), slice(3, 6))])
    exchange_groups = np.array([np.max(np.linalg.norm(momentum_exchange[:, s], axis=1)) for s in (slice(0, 3), slice(3, 6))])
    momentum_thresholds = config.momentum_atol + config.momentum_relative_tolerance*exchange_groups
    gates["momentum"] = bool(np.all(peak_groups <= momentum_thresholds))
    gates["no_external_wrench"] = metadata.get("external_wrench_applied") is False
    gates["no_hidden_reset"] = metadata.get("robot_state_writes_during_step_loop") == 0 and gates["complete_native_time"]
    gates = {k: bool(v) for k, v in gates.items()}
    return {"level": "EXECUTION_VERIFIED" if all_gates(gates, EXECUTION_GATES) and all(gates.values()) else "EXECUTION_FAILED",
            "gates": gates, "maximum_geometry_step_m":max_geometry_step, "replay": replay_report, "minimum_clearance_m": minimum, "minimum_pair": minimum_pair,
            "native_states": len(times), "checked_states": checked, "collision_queries": queries,
            "final_position_error_m": position[-1], "final_rotation_error_rad": rotation[-1],
            "hold_max_position_error_m": np.max(position[hold], axis=0) if hold.any() else None,
            "hold_max_rotation_error_rad": np.max(rotation[hold], axis=0) if hold.any() else None,
            "tracking_max_rad": tracking, "manifold_max_rad": manifold, "zero_momentum_max_residual": momentum_error,
            "momentum_peak_linear_angular": peak_groups, "momentum_exchange_scale_linear_angular": exchange_groups,
            "momentum_threshold_linear_angular": momentum_thresholds, "momentum_units": ["kg*m/s", "kg*m^2/s"],
            "saturation_fraction": float(np.mean(np.abs(arrays["raw_torque"]) > spec.torque_limits)),
            "clearance_path_m": clearance_path,
            "pair_policy_hash": collision.policy_hash, "continuous_time_certified": False, "hardware_validated": False}


def audit_execution(frozen_sample, profile, robot, initial, reference, output, wall_deadline=None, *, native_budget):
    """Only this individual frozen sample receives its own execution evidence."""
    from pathlib import Path
    import time
    from ff_prior.contracts import assert_frozen,assert_task_compatible
    from ff_prior.storage import atomic_json, atomic_npz, digest
    from .tracker import track
    assert_frozen(frozen_sample)
    assert_task_compatible(frozen_sample,robot,initial,CollisionBackend(robot),profile)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    from ff_prior.initial_state import STATE
    atomic_npz(output/"initial_state.npz",values=initial.values)
    atomic_json(output/"initial_state.json",{"state_mask":int(STATE),"state_id":initial.state_id,"model_hash":initial.model_hash})
    before = frozen_sample["task_hash"]
    wall, cpu = time.perf_counter(), time.process_time()
    expected_steps = int(round((reference.duration+profile.hold_s)/robot.model.opt.timestep))
    if expected_steps != 13750 or native_budget.steps_each != 13750 or native_budget.total_cap != 27500:
        raise ValueError('FROZEN_STATIC_NATIVE_BUDGET_MISMATCH')
    arrays, metadata = track(robot, initial, reference, profile.hold_s, 13750,wall_deadline,native_budget=native_budget)
    if wall_deadline is not None and time.time()>=wall_deadline:
        report={"level":"EXECUTION_FAILED","gates":dict.fromkeys(EXECUTION_GATES,False),
            "reason":"BUDGET_EXHAUSTED","replay":{"passed":False,"reason":"NOT_RUN_BUDGET_EXHAUSTED","physics_steps":0}}
    else:
        try:
            report = validate(robot, initial, reference, arrays, metadata,
                              frozen_sample["condition"]["goals"], profile,wall_deadline,output,native_budget=native_budget)
        except Exception as exc:
            report={"level":"EXECUTION_FAILED","gates":dict.fromkeys(EXECUTION_GATES,False),
                "reason":repr(exc),"replay":{"passed":False,"reason":"VALIDATION_INCOMPLETE"}}
    assert_frozen(frozen_sample)
    if before != frozen_sample["task_hash"]:
        raise ValueError("task changed during execution")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clearance = np.asarray(report.pop("clearance_path_m", []))
    atomic_npz(output / "native_trace.npz", **arrays, clearance_m=clearance)
    report.update(task_hash=before, ctrl_hash=digest(arrays["ctrl"]),
        native_budget=native_budget.save(),
        scene_model_hash=initial.model_hash, initial_state_id=initial.state_id,
        wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu,
        control_alignment="ctrl[k] applied to state[k] -> state[k+1]",
        metadata=metadata, data_role="EXECUTION_AUDIT", training_eligible=False,
        validation_level="EXECUTION_VERIFIED" if report["level"]=="EXECUTION_VERIFIED" else "MODEL_VALID",
        execution_status="PASS" if report["level"]=="EXECUTION_VERIFIED" else "FAIL",
        validation_profile_id=profile.profile_id,
        goal_source="frozen_user_static_application_world_goal",
        maximum_base_translation_m=float(np.max(np.linalg.norm(arrays["base"][:,:3]-arrays["base"][0,:3],axis=1))))
    atomic_json(output / "report.json", report)
    atomic_json(output / "frozen_task.json", frozen_sample)
    return report
