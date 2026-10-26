"""Read-only snapshots for the live UI; all points come from native MuJoCo FK."""
import mujoco
import numpy as np

from .trajectory import JointChunk


def rounded(value):
    return np.asarray(value).round(7).tolist()


def geometry(env):
    def body(name):
        i = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"Missing visualization body {name}")
        return env.data.xpos[i].copy()
    continuum = [body("base_link")] + [body(f"link_{i}") for i in range(1, 31)]
    continuum.append(env.pose(0)[0])
    rigid = [body(n) for n in ("base_link_r", "base_2_shoulder_link", "shoulder_link",
              "back_arm_link", "fore_arm_link", "arm_2_wrist_link", "wrist_link",
              "end_link", "end_effector_r")]
    return {"continuum": rounded(continuum), "rigid": rounded(rigid),
            "base": rounded(body("base_of_satelltte")),
            "base_rotation": rounded(env.data.xmat[mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, "base_of_satelltte")].reshape(3, 3)),
            "target": rounded(env.pose(2)[0]), "target_rotation": rounded(env.pose(2)[1])}


def scene(env):
    samples = [env.reference(t) for t in np.linspace(0., env.task.duration, 271)]
    return {"reference_c": rounded([x[0] for x in samples]),
            "reference_r": rounded([x[4] for x in samples]),
            "waypoints": rounded(env.path.waypoint_points_w),
            "obstacles": [{"center": rounded(x.center), "radius": x.radius} for x in env.obstacles],
            "duration": env.task.duration, "seed": env.task.seed,
            "pair_count": len(env.pairs), "geometry": geometry(env)}


def frame(env):
    ref = env.reference(float(env.data.time))
    error = env.task_error()
    q, dq = env.state()
    return {"time": round(float(env.data.time), 5), "geometry": geometry(env),
            "desired_c": rounded(ref[0]), "desired_r": rounded(ref[4]),
            "actual_c": rounded(env.pose(0)[0]), "actual_r": rounded(env.pose(1)[0]),
            "error_c_mm": float(1000*np.linalg.norm(error[:3])),
            "error_r_mm": float(1000*np.linalg.norm(error[6:9])),
            "orientation_c_deg": float(np.rad2deg(np.linalg.norm(error[3:6]))),
            "orientation_r_deg": float(np.rad2deg(np.linalg.norm(error[9:12]))),
            "margin_mm": None if env.last_margin is None else 1000*env.last_margin,
            "torque_ratio": env.maximum_torque_ratio,
            "torque_saturations": env.torque_saturations, "physics_steps": env.physics_steps,
            "q": rounded(q), "dq": rounded(dq), "q_reference": rounded(env.reference_q),
            "filter_delta": env.last_filter_delta}


def joint_candidates(env, free, points=31):
    q, dq = env.state()
    chunk_type = getattr(env,"active_chunk_type",JointChunk)
    horizon = getattr(env,"active_horizon",1.)
    chunks = [chunk_type.decode(x, q, dq, horizon) for x in free]
    return {"joint_curves": [rounded(x.sample(np.linspace(0., horizon, points))[0]) for x in chunks],
            "horizon": horizon,
            "bounds_ok": [x.bounds_ok(env.spec) for x in chunks]}
