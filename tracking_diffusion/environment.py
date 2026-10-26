"""Native MuJoCo model, a small tracking teacher, and torque execution.

Only the physical model, collision-pair policy and original Cartesian path are
reused. No V7 planner, dataset, network, safety hierarchy or experiment driver is
imported. Simulation time is authoritative; this is not a real-time certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere,
)
from v6_lite.irregular_waypoints import build_original_irregular_target

ROOT = Path(__file__).resolve().parents[1]
DT = 0.002
CONTROL_DT = 0.02
TIP_OFFSET = np.array([0.0475, 0., 0.])


def rotation_error(desired, actual):
    return Rotation.from_matrix(desired @ actual.T).as_rotvec()


def joint_addresses(model, names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if min(ids) < 0:
        raise ValueError("Missing model joint")
    return model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


def free_slices(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Expected a free joint: {name}")
    p, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    return slice(p, p + 7), slice(v, v + 6)


@dataclass(frozen=True)
class Task:
    seed: int = 20261012
    duration: float = 27.0
    obstacle_count: int = 2
    domain: str = "legacy-wide-v1"
    nominal: bool = False

    def __post_init__(self):
        if self.duration <= 0 or not np.isfinite(self.duration):
            raise ValueError("duration must be finite and positive")
        if self.obstacle_count not in (0, 2):
            raise ValueError("This initial task domain supports 0 or 2 workspace spheres")
        if self.domain not in ("legacy-wide-v1", "local10mm-v1"):
            raise ValueError("Unknown task domain")


class TrackingEnv:
    """Two arms, a free base and a freely drifting target; 67 torque motors."""

    def __init__(self, task: Task):
        self.task = task
        self.spec = RobotModelSpecV5.from_urdf(
            # MuJoCo 3.3.2 on Windows cannot open this non-ASCII absolute path;
            # the launcher sets cwd to ROOT, giving an ASCII relative asset path.
            Path(os.path.relpath(ROOT / "dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf")),
            continuum_kp=100., continuum_kd=5., rigid_kp=200., rigid_kd=20.,
        )
        home_model = self.spec.compile_dynamic_model()
        home = mujoco.MjData(home_model)
        ids, _ = joint_addresses(home_model, self.spec.low_level_joint_names)
        home.qpos[ids] = self.spec.encode_position(self.spec.planner_zero)
        mujoco.mj_forward(home_model, home)
        def home_pose(name, offset=None):
            b = mujoco.mj_name2id(home_model, mujoco.mjtObj.mjOBJ_BODY, name)
            r = home.xmat[b].reshape(3, 3).copy()
            p = home.xpos[b].copy()
            return p + (r @ offset if offset is not None else 0), r
        pc, rc = home_pose(self.spec.continuum_tip_body_name, TIP_OFFSET)
        pr, rr = home_pose(self.spec.rigid_tip_body_name)
        pt, rt = home_pose("target_satellite")
        self.path = build_original_irregular_target(pc, rc, task.seed)
        self.rigid_home = pr
        self.rigid_home_rotation = rr
        self.grasp_local = np.array([0., -0.205, 0.])
        self.grasp_rotation_local = rt.T @ rr
        rng = np.random.default_rng(task.seed)
        self.target_shift = rng.uniform([-.012, -.008, -.010], [.012, .008, .010])
        self.target_velocity = rng.uniform([-.0025, .001, -.0015], [.0025, .004, .0015])
        self.target_omega_local = rng.uniform(-.0015, .0015, 3)
        grasp = pt + self.target_shift + rt @ self.grasp_local
        direction = grasp - pr
        direction /= np.linalg.norm(direction)
        lateral = np.cross(direction, [0., 0., 1.])
        lateral /= max(np.linalg.norm(lateral), 1e-12)
        side = 1. if task.seed % 2 == 0 else -1.
        # The initial approach is part of reference construction. A straight
        # line crosses the fixed rigid-arm obstacle and cannot be tracked while
        # preserving clearance. Use a C2 lateral arc on its opposite side.
        self.approach_offset = -.14 * side * lateral if task.obstacle_count else np.zeros(3)
        self.obstacles = (
            WorkspaceSphere("rigid_path", pr + .52 * (grasp - pr) + .050 * side * lateral, .035),
            WorkspaceSphere("continuum_side", pc + [.120, side * .110, .060], .025),
        )[:task.obstacle_count]
        self.task_parameters = {}
        self.task_profile_hash = None
        if task.domain == "local10mm-v1":
            from .task_domain import profile, profile_hash, sample_task, build_path
            p = profile()
            if self.spec.runtime_contract_sha256() != p["model_runtime_contract_sha256"]:
                raise ValueError("Physical model changed since task domain was frozen")
            self.task_parameters = sample_task(task.seed, task.nominal)
            self.task_profile_hash = profile_hash()
            self.path = build_path(pc, rc, self.task_parameters)
            self.target_shift = np.asarray(self.task_parameters["target_shift"])
            self.target_velocity = np.asarray(self.task_parameters["target_velocity"])
            self.target_omega_local = np.asarray(self.task_parameters["target_omega_local"])
            self.obstacles = tuple(WorkspaceSphere(o["name"], np.asarray(o["center"]), o["radius"])
                                   for o in p["obstacles"])[:task.obstacle_count]
            self.approach_offset = np.asarray(p["approach_offset"]) if task.obstacle_count else np.zeros(3)
        self.verifier = WholeBodyCollisionVerifier(self.spec, self.obstacles,
            WholeBodyVerificationConfig(include_target_satellite_pairs=True,
                                        adaptive_subdivisions=1))
        self.model = self.verifier.model
        # Match the existing research model: clearance cannot be rescued by
        # contact impulses. Every named pair is queried explicitly instead.
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.data = mujoco.MjData(self.model)
        self.qids, self.vids = joint_addresses(self.model, self.spec.low_level_joint_names)
        self.bp, self.bv = free_slices(self.model, self.spec.base_joint_name)
        self.tp, self.tv = free_slices(self.model, self.spec.target_free_joint_name)
        self.base_ids = np.arange(self.bv.start, self.bv.stop)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in (self.spec.continuum_tip_body_name,
                                   self.spec.rigid_tip_body_name, "target_satellite")]
        self.data.qpos[self.qids] = self.spec.encode_position(self.spec.planner_zero)
        self.data.qvel[:] = 0.
        self.data.qpos[self.tp.start:self.tp.start+3] += self.target_shift
        self.data.qvel[self.tv.start:self.tv.start+3] = self.target_velocity
        self.data.qvel[self.tv.start+3:self.tv.stop] = self.target_omega_local
        mujoco.mj_forward(self.model, self.data)
        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.reference_q = self.spec.planner_zero.copy()
        self.command = np.zeros(17)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.torque_saturations = 0
        self.acceleration_clips = 0
        self.maximum_torque_ratio = 0.
        self.maximum_joint_reference_error = 0.
        self.qp_failures = 0
        self.minimum_clearance = float("inf")
        self.minimum_margin = float("inf")
        self.minimum_pair = None
        self.last_clearance = None
        self.last_margin = None
        self.last_filter_delta = 0.
        self.physics_steps = 0
        self.latencies = []
        self.physics_latencies = []
        self.maximum_actual_velocity_ratio = 0.
        self.maximum_actual_acceleration_ratio = 0.
        self.peak_acceleration = None
        self.minimum_joint_position_margin = float("inf")
        self.joint_limit_violations = 0
        self.pairs = self.verifier.pairs
        self.pair_a = np.array([p.geom_a for p in self.pairs])
        self.pair_b = np.array([p.geom_b for p in self.pairs])
        self.safe_distances = np.array([.005 if p.pair_class == "rigid_target" else .025
                                        for p in self.pairs])

    def state(self):
        return (self.spec.decode_position(self.data.qpos[self.qids]),
                self.spec.decode_velocity(self.data.qvel[self.vids]))

    def pose(self, index):
        b = self.body_ids[index]
        r = self.data.xmat[b].reshape(3, 3).copy()
        p = self.data.xpos[b].copy()
        if index == 0:
            p += r @ TIP_OFFSET
        return p, r

    def reference(self, t):
        """Current/future *desired* path. Target prediction uses present state only."""
        pc, vc = self.path.sample(t)
        pt, rt = self.pose(2)
        lead = float(t - self.data.time)
        # Free target angular velocity is expressed in its local frame.
        v = self.data.qvel[self.tv.start:self.tv.start+3]
        w = self.data.qvel[self.tv.start+3:self.tv.stop]
        rt_future = rt @ Rotation.from_rotvec(w * lead).as_matrix()
        point = pt + v * lead + rt_future @ self.grasp_local
        point_v = v + np.cross(rt_future @ w, rt_future @ self.grasp_local)
        rr = rt_future @ self.grasp_rotation_local
        # Smooth approach avoids the legacy instantaneous initial capture error.
        s = float(np.clip(t / 4.5, 0., 1.))
        a = s**3 * (10 - 15*s + 6*s*s)
        da = 30*s*s*(1-s)**2 / 4.5 if t < 4.5 else 0.
        pr = self.rigid_home + a * (point - self.rigid_home)
        vr = a * point_v + da * (point - self.rigid_home)
        bend = 64*s**3*(1-s)**3
        bend_rate = 192*s*s*(1-s)**2*(1-2*s)/4.5 if t < 4.5 else 0.
        pr += bend*self.approach_offset
        vr += bend_rate*self.approach_offset
        # Angular target remains slowly moving; starts at the home orientation.
        rv = rotation_error(rr, self.rigid_home_rotation)
        rref = Rotation.from_rotvec(a * rv).as_matrix() @ self.rigid_home_rotation
        wr = a * (rt_future @ w) + da * rv
        return (pc, self.path.target_rotation_world, vc, np.zeros(3),
                pr, rref, vr, wr)

    def generalized_map(self):
        mujoco.mj_fullM(self.model, self.mass, self.data.qM)
        arm = self.spec.planner_to_low_level
        bb = self.mass[np.ix_(self.base_ids, self.base_ids)]
        ba = self.mass[np.ix_(self.base_ids, self.vids)]
        g = np.zeros((self.model.nv, 17))
        g[self.vids] = arm
        g[self.base_ids] = -np.linalg.solve(bb, ba @ arm)
        # This affine term also accounts for measured nonzero base momentum and
        # small deviations of the 67 actual joints from the 17-dimensional shape.
        _, dq = self.state()
        drift = self.data.qvel - g @ dq
        return g, drift

    def task_jacobian(self, g, drift):
        matrices, offsets = [], []
        for i in range(2):
            p, _ = self.pose(i)
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, p, self.body_ids[i])
            j = np.vstack([jp, jr])
            matrices.append(j @ g)
            offsets.append(j @ drift)
        return np.vstack(matrices), np.concatenate(offsets)

    def task_error(self, t=None):
        ref = self.reference(float(self.data.time) if t is None else t)
        out = []
        for i, offset in ((0, 0), (1, 4)):
            p, r = self.pose(i)
            out.extend((ref[offset] - p, rotation_error(ref[offset+1], r)))
        return np.concatenate(out)

    def desired_velocity(self, feedforward=True):
        ref = self.reference(float(self.data.time))
        error = self.task_error()
        gains = np.repeat([12., 10., 5., 8.], 3)
        velocity = gains * error
        if feedforward:
            velocity += np.concatenate([ref[2], ref[3], ref[6], ref[7]])
        for start, limit in ((0, .24), (3, .45), (6, .24), (9, .45)):
            norm = np.linalg.norm(velocity[start:start+3])
            velocity[start:start+3] *= min(1., limit / max(norm, 1e-12))
        return velocity

    def clearance(self, g=None, drift=None, activation=.08):
        """Exact named-pair distances, with a conservative bounding-sphere screen."""
        model, data = self.model, self.data
        a, b = self.pair_a, self.pair_b
        bounds = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1)
        bounds -= model.geom_rbound[a] + model.geom_rbound[b]
        # Distance outside this range is a *lower bound*, not an exact value.
        selected = np.flatnonzero(bounds <= activation + .01)
        distances = bounds.copy()
        rows, lower, drift_rows, margins = [], [], [], []
        witness = np.zeros(6)
        for k in selected:
            pair = self.pairs[k]
            d = float(mujoco.mj_geomDistance(model, data, pair.geom_a, pair.geom_b,
                                           activation + .01, witness))
            distances[k] = d
            if g is None or d > activation:
                continue
            pa, pb = witness[:3].copy(), witness[3:].copy()
            if model.geom_type[pair.geom_a] > model.geom_type[pair.geom_b]:
                pa, pb = pb, pa
            normal = pa - pb
            norm = np.linalg.norm(normal)
            if norm < 1e-10 or d <= 0:
                raise RuntimeError(f"Invalid/penetrating clearance witness: {pair}")
            normal /= norm
            jac = []
            for geom, point in ((pair.geom_a, pa), (pair.geom_b, pb)):
                jp = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, None, point, int(model.geom_bodyid[geom]))
                jac.append(jp)
            j = normal @ (jac[0] - jac[1])
            rows.append(j @ g)
            lower.append(-8. * (d - self.safe_distances[k]) - j @ drift)
            drift_rows.append(float(j @ drift))
            margins.append(float(d-self.safe_distances[k]))
        k = int(np.argmin(distances - self.safe_distances))
        margin = float(distances[k] - self.safe_distances[k])
        self.last_clearance = float(np.min(distances))
        self.last_margin = margin
        self.minimum_clearance = min(self.minimum_clearance, float(np.min(distances)))
        if margin < self.minimum_margin:
            self.minimum_margin = margin
            self.minimum_pair = {"class": self.pairs[k].pair_class,
                                 "a": self.pairs[k].geom_a_name,
                                 "b": self.pairs[k].geom_b_name,
                                 "time": float(data.time)}
        if g is not None:
            self.collision_linearization = {
                "jacobian": np.asarray(rows).reshape(-1,17),
                "drift": np.asarray(drift_rows), "margin": np.asarray(margins)}
        return np.asarray(rows).reshape(-1, 17), np.asarray(lower), margin

    def velocity_filter(self, desired, g, drift, hessian=None):
        """One small box/clearance QP. Failure is explicit, never a hidden teacher."""
        q, _ = self.state()
        vmax = .70 * self.spec.planner_velocity_limits
        # Reserve servo tracking headroom; hard actual-state limits are unchanged.
        acc = .60 * self.spec.planner_acceleration_limits
        lo = np.maximum.reduce([-vmax, 4 * (self.spec.planner_lower + .04 - q),
                                self.command - acc * CONTROL_DT])
        hi = np.minimum.reduce([vmax, 4 * (self.spec.planner_upper - .04 - q),
                                self.command + acc * CONTROL_DT])
        a, b, margin = self.clearance(g, drift)
        # Leave a millimetre for the 20 ms velocity ramp and model integration.
        # This is an engineering buffer, not a continuous-time certificate.
        b = b + .008
        if np.any(lo > hi) or margin < -0.001:
            raise RuntimeError("Measured state outside the execution domain")
        x = np.clip(desired, lo, hi)
        active = len(b) and np.min(a @ x - b) < -1e-8
        if active or (hessian is not None and np.max(np.abs(x-desired)) > 1e-10):
            # Preserve the task objective when enforcing constraints. Euclidean
            # projection of an unconstrained IK velocity destroyed tracking in
            # the first pilot even though the torque servo followed faithfully.
            h = np.eye(17) if hessian is None else hessian / max(np.max(np.diag(hessian)), 1e-8)
            result = minimize(lambda z: .5 * (z-desired) @ h @ (z-desired), x,
                jac=lambda z: h @ (z-desired), bounds=list(zip(lo, hi)), method="SLSQP",
                constraints=({"type": "ineq", "fun": lambda z: a @ z - b,
                              "jac": lambda z: a},) if len(b) else (),
                options={"ftol": 1e-12, "maxiter": 200})
            x = result.x
            if not result.success or (len(b) and np.min(a @ x - b) < -1e-6):
                self.qp_failures += 1
                raise RuntimeError(f"No feasible velocity: {result.message}")
        self.last_filter_delta = float(np.linalg.norm(x-desired))
        return x

    def command_velocity(self, planned=None, cartesian_feedback=1.):
        g, drift = self.generalized_map()
        j, offset = self.task_jacobian(g, drift)
        # Weights express different units and arm priorities, not extra solvers.
        weights = np.repeat([1., 2., 1.4, 2.4], 3)
        jw = weights[:, None] * j
        reg = 1e-5
        if planned is None:
            residual = self.desired_velocity(True) - offset
            rhs = jw.T @ (weights * residual)
            q, _ = self.state()
            rhs += 1e-5 * .08 * (self.spec.planner_zero - q)
            hessian = jw.T @ jw + reg*np.eye(17)
            desired = np.linalg.solve(hessian, rhs)
        else:
            desired = np.asarray(planned).copy()
            if cartesian_feedback:
                # Feedback corrects task error only. Desired task feedforward
                # must come from the learned joint trajectory in this mode.
                residual = cartesian_feedback * self.desired_velocity(False) - offset
                correction = np.linalg.solve(jw.T @ jw + reg*np.eye(17),
                                             jw.T @ (weights * residual))
                desired += correction
            else:
                correction = np.zeros(17)
            self.last_feedback_norm = float(np.linalg.norm(correction))
            hessian = None
        return self.velocity_filter(desired, g, drift, hessian)

    def advance(self, velocity):
        """Ten torque steps. Actual qpos/qvel are never assigned after init."""
        started = time.perf_counter()
        previous = self.command.copy()
        acceleration = (velocity-previous) / CONTROL_DT
        for substep in range(10):
            physics_started = time.perf_counter()
            _, previous_actual_velocity = self.state()
            dqref = previous + (substep+1)/10. * (velocity-previous)
            self.reference_q += DT * dqref
            qref = self.spec.encode_position(self.reference_q)
            vref = self.spec.encode_velocity(dqref)
            aref = self.spec.encode_velocity(acceleration)
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_fullM(self.model, self.mass, self.data.qM)
            w = np.r_[np.full(60, 42.), np.full(7, 34.)]
            requested_a = (aref + w*w * (qref-self.data.qpos[self.qids])
                           + 2*w * (vref-self.data.qvel[self.vids]))
            limits = np.r_[np.full(60, 45.), np.full(7, 70.)]
            actual_a = np.clip(requested_a, -limits, limits)
            self.acceleration_clips += int(np.count_nonzero(actual_a != requested_a))
            bb = self.mass[np.ix_(self.base_ids, self.base_ids)]
            ba = self.mass[np.ix_(self.base_ids, self.vids)]
            force = self.data.qfrc_bias - self.data.qfrc_passive
            base_a = -np.linalg.solve(bb, ba @ actual_a + force[self.base_ids])
            torque = (self.mass[np.ix_(self.vids, self.base_ids)] @ base_a
                      + self.mass[np.ix_(self.vids, self.vids)] @ actual_a
                      + force[self.vids])
            self.maximum_torque_ratio = max(self.maximum_torque_ratio,
                float(np.max(np.abs(torque) / self.spec.torque_limits)))
            clipped = np.clip(torque, -self.spec.torque_limits, self.spec.torque_limits)
            self.torque_saturations += int(np.count_nonzero(clipped != torque))
            self.data.ctrl[:] = clipped
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            mujoco.mj_forward(self.model, self.data)
            # This is a sampled simulation check, not a proof between steps.
            _, _, margin = self.clearance()
            if margin < -0.001:
                raise RuntimeError(f"Clearance margin violated at {self.data.time:.3f}s")
            q, actual_velocity = self.state()
            actual_acceleration = (actual_velocity-previous_actual_velocity)/DT
            vr = float(np.max(np.abs(actual_velocity)/self.spec.planner_velocity_limits))
            ar = float(np.max(np.abs(actual_acceleration)/self.spec.planner_acceleration_limits))
            pm = float(min(np.min(q-self.spec.planner_lower), np.min(self.spec.planner_upper-q)))
            self.maximum_actual_velocity_ratio = max(self.maximum_actual_velocity_ratio, vr)
            if ar > self.maximum_actual_acceleration_ratio:
                self.maximum_actual_acceleration_ratio = ar
                self.peak_acceleration = {"time": float(self.data.time),
                    "coordinate": int(np.argmax(np.abs(actual_acceleration)/self.spec.planner_acceleration_limits)),
                    "ratio": ar}
            self.minimum_joint_position_margin = min(self.minimum_joint_position_margin, pm)
            self.joint_limit_violations += int(vr > 1.0001 or ar > 1.0001 or pm < -1e-7)
            self.maximum_joint_reference_error = max(self.maximum_joint_reference_error,
                float(np.max(np.abs(q-self.reference_q))))
            if not np.all(np.isfinite(self.data.qpos)):
                raise RuntimeError("Non-finite simulated state")
            self.physics_latencies.append(time.perf_counter()-physics_started)
        self.command = velocity.copy()
        self.latencies.append(time.perf_counter()-started)

    def condition(self, horizon=1., points=6):
        q, dq = self.state()
        features = [q, dq, self.data.qpos[self.bp], self.data.qvel[self.bv]]
        current = [self.pose(0), self.pose(1)]
        for delta in np.linspace(0., horizon, points):
            ref = self.reference(float(self.data.time) + delta)
            for arm, offset in ((0, 0), (1, 4)):
                p, r = current[arm]
                features.extend([ref[offset]-p, rotation_error(ref[offset+1], r)])
        # Current target twist and two fixed, masked sphere slots. No seed/time
        # identifier, teacher outputs or future measured robot states are inputs.
        features.append(self.data.qvel[self.tv])
        for i in range(2):
            if i < len(self.obstacles):
                ob = self.obstacles[i]
                features.append(np.r_[ob.center-self.data.qpos[self.bp][:3], ob.radius, 1.])
            else:
                features.append(np.zeros(5))
        return np.concatenate(features).astype(np.float32)

    def diagnostics(self):
        return {
            "physics_steps": self.physics_steps,
            "qpos_writes_after_initialization": 0,
            "qvel_writes_after_initialization": 0,
            "minimum_pair_distance_or_lower_bound_m": self.minimum_clearance if np.isfinite(self.minimum_clearance) else None,
            "minimum_pair_margin_m": self.minimum_margin if np.isfinite(self.minimum_margin) else None,
            "minimum_margin_pair": self.minimum_pair,
            "pair_count": len(self.pairs),
            "pair_policy_sha256": self.verifier._pair_policy_sha256,
            "torque_saturation_count": self.torque_saturations,
            "acceleration_clip_count": self.acceleration_clips,
            "maximum_requested_torque_ratio": self.maximum_torque_ratio,
            "maximum_servo_joint_error_rad": self.maximum_joint_reference_error,
            "qp_failures": self.qp_failures,
            "maximum_actual_velocity_ratio": self.maximum_actual_velocity_ratio,
            "maximum_actual_acceleration_ratio": self.maximum_actual_acceleration_ratio,
            "peak_acceleration": self.peak_acceleration,
            "minimum_joint_position_margin_rad": self.minimum_joint_position_margin if np.isfinite(self.minimum_joint_position_margin) else None,
            "joint_limit_violation_steps": self.joint_limit_violations,
            "execution_tick_p95_ms": 1e3*np.percentile(self.latencies, 95) if self.latencies else None,
            "physics_computation_p95_ms": 1e3*np.percentile(self.physics_latencies,95) if self.physics_latencies else None,
            "physics_computation_p99_ms": 1e3*np.percentile(self.physics_latencies,99) if self.physics_latencies else None,
            "physics_computation_max_ms": 1e3*max(self.physics_latencies) if self.physics_latencies else None,
            "physics_2ms_deadline_misses": sum(t>DT for t in self.physics_latencies),
            "continuous_time_safety_certified": False,
            "wall_realtime_certified": False,
        }
