"""Same zero-momentum midpoint equations on explicitly saved phase nodes."""
import time
import mujoco
import numpy as np

def predict_phases(initial, reference, robot, phase, wall_deadline=None):
    phase = np.asarray(phase, dtype=float)
    assert phase[0] == 0 and phase[-1] == 1 and np.all(np.diff(phase)>0)
    intervals = len(phase)-1
    robot.calls["coupled"] += 1
    d, m = initial.reset(robot), robot.model
    robot.kinematics(d, mass=True)
    p0 = robot.mass(d) @ d.qvel
    if np.linalg.norm(p0[robot.base_dofs]) > 1e-12 or np.linalg.norm(d.qvel) > 1e-12:
        raise ValueError("UNSUPPORTED_NONZERO_INITIAL_MOMENTUM_OR_VELOCITY")
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
        raise ValueError("UNSUPPORTED_EXTERNAL_WRENCH")
    b, a = robot.base_dofs, robot.dof_ids
    bb, ba = np.ix_(b,b), np.ix_(b,a)
    # Use T=1 phase derivatives. Phase path is retiming invariant only under
    # the fixed static zero-momentum assumptions enforced above.
    qs = reference.evaluate(phase * reference.duration)
    vs = reference.evaluate(phase * reference.duration, 1) * reference.duration
    qm = reference.evaluate(((phase[:-1]+phase[1:])*.5) * reference.duration)
    vm = reference.evaluate(((phase[:-1]+phase[1:])*.5) * reference.duration,1) * reference.duration
    result = {k: [] for k in ("qpos", "base", "ee", "body", "geom", "geom_rotation", "velocity")}
    residual, momentum, quaternion_error = 0., 0., 0.

    def velocity(q, qprime):
        nonlocal residual, momentum, quaternion_error
        d.qpos[robot.qpos_ids] = robot.B @ q
        robot.kinematics(d, mass=True)
        M = robot.mass(d)
        v = np.zeros(m.nv)
        v[a] = robot.B @ qprime
        rhs = M[ba] @ v[a]
        v[b] = -np.linalg.solve(M[bb], rhs)
        residual = max(residual, float(np.max(abs(M[bb] @ v[b] + rhs))))
        momentum = max(momentum, float(np.max(abs((M @ v)[b]))))
        quaternion_error = max(quaternion_error, abs(float(np.linalg.norm(d.qpos[robot.base_qpos[3:]])) - 1))
        return v

    for i in range(intervals+1):
        if wall_deadline is not None and time.time() >= wall_deadline:
            raise RuntimeError("BUDGET_EXHAUSTED")
        v = velocity(qs[i], vs[i])
        for key, value in (("qpos",d.qpos), ("base",d.qpos[robot.base_qpos]), ("ee",robot.ee(d)),
                           ("body",d.xpos), ("geom",d.geom_xpos), ("geom_rotation",d.geom_xmat), ("velocity",v/reference.duration)):
            result[key].append(value.copy())
        if i < intervals:
            origin = d.qpos.copy()
            mujoco.mj_integratePos(m, d.qpos, v, .5*(phase[i+1]-phase[i]))
            midpoint_velocity = velocity(qm[i], vm[i])
            d.qpos[:] = origin
            mujoco.mj_integratePos(m, d.qpos, midpoint_velocity, phase[i+1]-phase[i])
    result = {k: np.asarray(v) for k,v in result.items()}
    result.update(phase=phase, numerical={"solve_residual":residual, "robot_momentum_residual":momentum,
        "quaternion_norm_error":quaternion_error, "intervals":intervals, "method":"mj_integratePos_midpoint_nonuniform"})
    return result
