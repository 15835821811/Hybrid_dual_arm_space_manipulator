"""CPU midpoint zero-momentum reference propagation; never physical execution.

MuJoCo free joint velocities are world translation then body-frame angular
velocity. mj_integratePos uses that same tangent convention. Only robot base
and named arm blocks enter the solve; the other free body is excluded.
"""
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from . import VERSION
from .storage import digest


def pose_errors(actual, target):
    a, b = np.asarray(actual), np.asarray(target)
    position = np.linalg.norm(a[..., :3,3] - b[..., :3,3], axis=-1)
    rot = np.swapaxes(b[..., :3,:3], -1,-2) @ a[..., :3,:3]
    return position, Rotation.from_matrix(rot.reshape(-1,3,3)).magnitude().reshape(position.shape)


def predict_coupled(initial, reference, robot, intervals=128, wall_deadline=None):
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
    phase = np.linspace(0,1,intervals+1)
    # Use T=1 phase derivatives. Phase path is retiming invariant only under
    # the fixed static zero-momentum assumptions enforced above.
    qs = reference.evaluate(phase * reference.duration)
    vs = reference.evaluate(phase * reference.duration, 1) * reference.duration
    qm = reference.evaluate((phase[:-1] + .5/intervals) * reference.duration)
    vm = reference.evaluate((phase[:-1] + .5/intervals) * reference.duration,1) * reference.duration
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
            mujoco.mj_integratePos(m, d.qpos, v, .5/intervals)
            midpoint_velocity = velocity(qm[i], vm[i])
            d.qpos[:] = origin
            mujoco.mj_integratePos(m, d.qpos, midpoint_velocity, 1/intervals)
    result = {k: np.asarray(v) for k,v in result.items()}
    result.update(phase=phase, numerical={"solve_residual":residual, "robot_momentum_residual":momentum,
        "quaternion_norm_error":quaternion_error, "intervals":intervals, "method":"mj_integratePos_midpoint"})
    return result


def converge(initial, ref, robot, wall_deadline=None):
    coarse = predict_coupled(initial, ref, robot, 128, wall_deadline)
    comparisons = []
    for n in (256,512):
        fine = predict_coupled(initial, ref, robot, n, wall_deadline)
        position, rotation = pose_errors(fine["ee"][-1], coarse["ee"][-1])
        body_delta = np.linalg.norm(fine["body"][::2] - coarse["body"], axis=-1)
        # Corresponding finite-volume points: center plus bounding-radius frame
        # axes. Include rotation of geometry, not only link origins.
        geom_delta = np.linalg.norm(fine["geom"][::2]-coarse["geom"],axis=-1)
        rotation_delta = (fine["geom_rotation"][::2]-coarse["geom_rotation"]).reshape(len(coarse["phase"]),robot.model.ngeom,3,3)
        extent = np.linalg.norm(rotation_delta,axis=(-1,-2))*robot.model.geom_rbound
        point_delta = float(max(np.max(body_delta), np.max(geom_delta + extent)))
        passed = bool(np.max(position) <= .001 and np.max(rotation) <= np.deg2rad(.2) and point_delta <= .001)
        comparisons.append({"coarse":n//2,"fine":n,"terminal_position_m":position,
            "terminal_rotation_rad":rotation,"aligned_body_extent_m":point_delta,"passed":passed})
        if passed:
            return fine, {"passed":True, "comparisons":comparisons}
        coarse = fine
    return fine, {"passed":False, "status":"NUMERICAL_UNRESOLVED", "comparisons":comparisons}


def cache_key(initial, reference, robot, pair_policy_hash, static=True):
    d = initial.reset(robot)
    if not static or np.any(d.qvel) or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
        raise ValueError("phase cache assumptions not satisfied")
    return digest({"initial":initial.state_id,"model":initial.model_hash,"B":robot.B,
                   "path":reference.control,"version":VERSION,"pair_policy":pair_policy_hash})
