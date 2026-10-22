"""Reference-only floating-base feedforward plus 67-channel torque PD.

Actual qpos/qvel are written only by reset and mj_step. Predictor and validator
own separate MjData. Raw and clipped torques are both archived.
"""
from __future__ import annotations

import time
import mujoco
import numpy as np
from ff_prior.initial_state import STATE

def integration_state(robot,data):
    value=np.empty(mujoco.mj_stateSize(robot.model,STATE))
    mujoco.mj_getState(robot.model,data,value,STATE)
    return value



def command(robot, data, q, qd, qdd):
    m, spec = robot.model, robot.spec
    robot.forward(data)
    mass = robot.mass(data)
    b, a = robot.base_dofs, robot.dof_ids
    low_q, low_qd, low_qdd = robot.B @ q, robot.B @ qd, robot.B @ qdd
    bias = data.qfrc_bias - data.qfrc_passive
    base_acc = -np.linalg.solve(mass[np.ix_(b, b)], mass[np.ix_(b, a)] @ low_qdd + bias[b])
    ff = mass[np.ix_(a, b)] @ base_acc + mass[np.ix_(a, a)] @ low_qdd + bias[a]
    raw = ff + spec.position_kp * (low_q - data.qpos[robot.qpos_ids]) + spec.velocity_kd * (low_qd - data.qvel[a])
    return np.clip(raw, -spec.torque_limits, spec.torque_limits), raw, base_acc


def track(robot, initial, reference, hold_s, max_steps, wall_deadline=None):
    robot.calls["torque_rollout"] += 1
    m = robot.model
    dt = float(m.opt.timestep)
    total = reference.duration + hold_s
    steps = int(round(total / dt))
    if not np.isclose(steps * dt, total, atol=1e-10) or steps < 1:
        raise ValueError("motion+hold must be an integral positive number of physics steps")
    if steps > max_steps:
        raise ValueError("frozen physical step budget cannot cover complete motion+hold")
    data = initial.reset(robot)
    start_time = float(data.time)
    arrays = {"integration": [], "time": [], "qpos": [], "qvel": [], "act": [], "qacc_warmstart": [], "qacc": [], "ee": [], "base": [],
              "q_actual_17": [], "manifold_residual": [], "base_generalized_momentum": [],
              "ctrl": [], "raw_torque": [], "reference_q": [], "reference_qd": [], "reference_qdd": []}

    def state():
        robot.forward(data)
        arrays["time"].append(float(data.time))
        for name in ("qpos", "qvel", "act", "qacc_warmstart", "qacc"):
            arrays[name].append(getattr(data, name).copy())
        arrays["ee"].append(robot.ee(data))
        arrays["base"].append(data.qpos[robot.base_qpos].copy())
        q = robot.Bplus @ data.qpos[robot.qpos_ids]
        arrays["q_actual_17"].append(q)
        arrays["manifold_residual"].append(data.qpos[robot.qpos_ids] - robot.B @ q)
        arrays["base_generalized_momentum"].append((robot.mass(data) @ data.qvel)[robot.base_dofs])
        arrays["integration"].append(integration_state(robot,data))

    wall = time.perf_counter()
    state()
    status = "FINISHED"
    for k in range(steps):
        if wall_deadline is not None and time.time()>=wall_deadline:
            status="BUDGET_EXHAUSTED"
            break
        q, qd, qdd = (reference.evaluate(k * dt, j) for j in range(3))
        torque, raw, _ = command(robot, data, q, qd, qdd)
        if not np.isfinite(raw).all():
            status = "NONFINITE_COMMAND"
            break
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise RuntimeError("undeclared external generalized force or body wrench")
        data.ctrl[robot.actuator_ids] = torque
        arrays["ctrl"].append(data.ctrl.copy())
        arrays["raw_torque"].append(raw)
        for name, val in (("reference_q", q), ("reference_qd", qd), ("reference_qdd", qdd)):
            arrays[name].append(val)
        mujoco.mj_step(m, data)
        robot.calls["physics_steps"] += 1
        state()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            status = "NUMERICAL_FAILURE"
            break
        if abs(data.time - (start_time + (k + 1) * dt)) > 1e-9 or np.any(data.warning.number):
            status = "RESET_OR_SIMULATOR_WARNING"
            break
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    return arrays, {"status": status, "physics_steps": len(arrays["ctrl"]),
                    "planned_steps": steps, "wall_seconds": time.perf_counter() - wall,
                    "initial_state_id": initial.state_id, "reference_hash": reference.reference_hash,
                    "scene_model_hash": initial.model_hash, "controller": "floating_ff_torque_pd_v1",
                    "robot_state_writes_during_step_loop": 0, "external_wrench_applied": False}
