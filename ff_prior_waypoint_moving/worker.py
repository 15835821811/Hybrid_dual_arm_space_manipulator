import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import time
import traceback
import numpy as np
import mujoco
from scipy.interpolate import CubicHermiteSpline
from tracking_diffusion.environment import TrackingEnv, Task, free_slices
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec, Obstacle
from ff_prior.geometry import CollisionBackend
from ff_prior.initial_state import InitialState
from ff_prior.storage import atomic_json, atomic_npz, file_digest, RunLock
from ff_prior_waypoint_domain.adapter import Environment
from ff_prior_waypoint_domain.experiment_v2 import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/v7_1/waypoint_moving_gate_001"
START = time.time()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def progress(stage, **info):
    if time.time()-START > 3600:
        raise TimeoutError("B0 1h cap")
    atomic_json(OUT/"status.json", dict(status="RUNNING", stage=stage, pid=os.getpid(),
                heartbeat=time.time(), elapsed_s=time.time()-START, progress=info))


def select():
    inventory = read(ROOT/"artifacts/v7_1/waypoint_direct_audit_001/episodes.json")
    conflict = {r["trace_sha256"] for r in inventory if r.get("available") and r["seed"] == 20261012}
    chosen, hashes, seeds = {"train": [], "validation": []}, set(), set()
    for r in sorted(inventory, key=lambda x: x["path"]):
        if not r.get("available") or not r.get("old_admitted") or r["seed"] == 20261012:
            continue
        if r["trace_sha256"] in conflict or r["trace_sha256"] in hashes or r["seed"] in seeds:
            continue
        if r.get("target_motion") != "moving":
            continue
        result = read(ROOT/r["path"]/"result.json")
        e = result["execution"]
        if e.get("joint_limit_violation_steps", 1) != 0 or e.get("maximum_actual_acceleration_ratio", 2) > 1:
            continue
        split = r["split"]
        if len(chosen[split]) >= (32 if split == "train" else 8):
            continue
        if file_digest(ROOT/r["path"]/"trace.npz") != r["trace_sha256"]:
            raise ValueError("Source identity changed")
        chosen[split].append(r)
        hashes.add(r["trace_sha256"]); seeds.add(r["seed"])
    if len(chosen["train"]) < 8 or len(chosen["validation"]) < 4:
        raise ValueError("Insufficient eligible moving source candidates")
    return chosen


def polynomial_bounds(spline):
    """Exact scalar extrema per cubic interval; no endpoint-only bound claim."""
    result = []
    for order in range(3):
        p = spline.derivative(order)
        lo, hi = np.full(17, np.inf), np.full(17, -np.inf)
        for i, width in enumerate(np.diff(p.x)):
            for j in range(17):
                coefficients = p.c[:, i, j]
                roots = np.roots(np.polyder(coefficients))
                points = [0., width] + [float(z.real) for z in roots if abs(z.imag) < 1e-12 and 0 < z.real < width]
                values = np.polyval(coefficients, points)
                lo[j] = min(lo[j], min(values)); hi[j] = max(hi[j], max(values))
        result.append((lo, hi))
    return result


class DirectEnvironment(Environment):
    def __init__(self, old, robot):
        # Copy task construction only; replace all physical/model state and geometry.
        self.__dict__.update(old.__dict__)
        self.robot = robot
        self.spec, self.model = robot.spec, robot.model
        self.data = robot.new_data()
        self.qids, self.vids = robot.qpos_ids, robot.dof_ids
        self.bp, self.bv = free_slices(self.model, self.spec.base_joint_name)
        self.tp, self.tv = free_slices(self.model, self.spec.target_free_joint_name)
        self.base_ids = robot.base_dofs
        self.body_ids = [robot.tip_c, robot.tip_r, robot.id(mujoco.mjtObj.mjOBJ_BODY, "target_satellite")]
        # Addresses are explicitly name-resolved in both source and destination.
        for dst, src in ((self.bp, old.bp), (self.tp, old.tp)):
            self.data.qpos[dst] = old.data.qpos[src]
        for dst, src in ((self.bv, old.bv), (self.tv, old.tv)):
            self.data.qvel[dst] = old.data.qvel[src]
        self.data.qpos[self.qids] = old.data.qpos[old.qids]
        self.data.qvel[self.vids] = old.data.qvel[old.vids]
        robot.forward(self.data)
        self.reference_q = robot.Bplus@self.data.qpos[self.qids]
        self.command = robot.Bplus@self.data.qvel[self.vids]
        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.collision = CollisionBackend(robot)
        self.initial = InitialState.capture(robot, self.data)
        self.scratch = robot.new_data()
        self.history = {k: [] for k in ("time", "qpos", "qvel", "ee", "desired", "base", "q17", "qref",
                                      "clearance", "momentum", "exchange", "skeleton_c", "skeleton_r")}
        self.controls, self.raw, self.command_qd, self.command_qdd = [], [], [], []
        self.max_geom_step, self.interpolation_states = 0., 0
        self.previous_qpos = self.previous_geom = self.previous_geom_r = None
        # Caller records and handles even a rejected initial frame.

    def step_reference(self, spline, t):
        r, d = self.robot, self.data
        self.reference_q = spline(t)
        v, a = spline(t, 1), spline(t, 2)
        if np.any(self.reference_q < r.lower) or np.any(self.reference_q > r.upper):
            raise RuntimeError("REFERENCE_DOMAIN")
        r.forward(d)
        mass = r.mass(d)
        w = np.r_[np.full(60, 42.), np.full(7, 34.)]
        requested = r.B@a + w*w*(r.B@self.reference_q-d.qpos[r.qpos_ids])+2*w*(r.B@v-d.qvel[r.dof_ids])
        aref = np.clip(requested, -np.r_[np.full(60, 45.), np.full(7, 70.)],
                       np.r_[np.full(60, 45.), np.full(7, 70.)])
        b, j = r.base_dofs, r.dof_ids
        force = d.qfrc_bias-d.qfrc_passive
        ba = -np.linalg.solve(mass[np.ix_(b, b)], mass[np.ix_(b, j)]@aref+force[b])
        raw = mass[np.ix_(j, b)]@ba+mass[np.ix_(j, j)]@aref+force[j]
        if not np.isfinite(raw).all():
            raise RuntimeError("NONFINITE_TORQUE")
        d.ctrl[r.actuator_ids] = np.clip(raw, -r.spec.torque_limits, r.spec.torque_limits)
        self.controls.append(d.ctrl.copy()); self.raw.append(raw.copy())
        self.command_qd.append(v.copy()); self.command_qdd.append(a.copy())
        mujoco.mj_step(r.model, d)
        self.record()
        if np.any(d.warning.number):
            raise RuntimeError("MUJOCO_WARNING")


def case(row, index):
    directory = OUT/f"train_{index:02d}"
    directory.mkdir()
    source = ROOT/row["path"]
    result = read(source/"result.json")
    old = TrackingEnv(Task(**result["task"]))
    scene = SceneSpec(tuple(Obstacle(o.name, "sphere", tuple(o.center), (o.radius, 0., 0.))
                            for o in old.obstacles), "legacy_moving_preserved")
    robot = RobotAdapter(scene=scene)
    with np.load(source/"trace.npz", allow_pickle=False) as z:
        t, q, dq = z["time"], z["q"], z["dq"]
        np.testing.assert_allclose(np.diff(t), .02, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(old.data.qpos, z["qpos"][0])
        np.testing.assert_array_equal(old.data.qvel, z["qvel"][0])
        ids = np.r_[np.arange(0, len(t)-1, 5), len(t)-1]
        spline = CubicHermiteSpline(t[ids], q[ids], dq[ids])
        atomic_npz(directory/"reference.npz", time=t[ids], q=q[ids], dq=dq[ids])
        bound = polynomial_bounds(spline)
        gates = dict(position=bool(np.all(bound[0][0] >= robot.lower) and np.all(bound[0][1] <= robot.upper)),
                     velocity=bool(np.all(np.maximum(abs(bound[1][0]), abs(bound[1][1])) <= robot.spec.planner_velocity_limits)),
                     acceleration=bool(np.all(np.maximum(abs(bound[2][0]), abs(bound[2][1])) <= robot.spec.planner_acceleration_limits)))
        atomic_json(directory/"representation.json", dict(gates=gates,
            maximum_sample_fit_error_rad=float(abs(spline(t)-q).max()),
            bounds=bound, source=row, knots=len(ids), physics_validated=False))
    if not all(gates.values()):
        report = dict(status="REPRESENTATION_REJECTED", gates=gates, steps=0)
        atomic_json(directory/"result.json", report)
        return report
    env = DirectEnvironment(old, robot)
    atomic_npz(directory/"initial.npz", values=env.initial.values)
    atomic_json(directory/"identity.json", dict(robot=robot.identity(), scene=scene.to_dict(),
                initial_state_id=env.initial.state_id, source=row, task=result["task"]))
    reason = None
    try:
        env.record()
        for k in range(13500):
            env.step_reference(spline, (k+1)*.002)
            if k % 500 == 0:
                progress("TRAIN_NATIVE_EXECUTION", case=index, steps=k+1)
    except RuntimeError as e:
        reason = str(e)
    if not env.controls:
        atomic_npz(directory/"native_trace.npz", **{k:np.asarray(v) for k,v in env.history.items()})
        report = dict(status="INITIAL_STATE_REJECTED", steps=0, stop_reason=reason)
        atomic_json(directory/"result.json", report)
        return report
    return evaluate(env, directory, reason, progress)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"status.json").exists():
        raise RuntimeError("Existing run immutable")
    with RunLock(OUT):
        progress("FREEZE")
        selected = select()
        sources = {}
        for folder in ("ff_prior", "ff_prior_waypoint_scene", "ff_prior_waypoint_domain",
                       "ff_prior_waypoint_moving", "tracking_diffusion", "model_test", "v6_lite"):
            for p in (ROOT/folder).glob("*.py"):
                sources[str(p.relative_to(ROOT))] = file_digest(p)
        for rows in selected.values():
            for row in rows:
                for name in ("result.json", "trace.npz"):
                    p = ROOT/row["path"]/name
                    sources[str(p.relative_to(ROOT))] = file_digest(p)
        for p in (ROOT/"tracking_diffusion/config/local10mm_v1.json",
                  ROOT/"dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf",
                  ROOT/"ff_prior_waypoint_moving/docs/EXPERIMENT_PLAN.md"):
            sources[str(p.relative_to(ROOT))] = file_digest(p)
        atomic_json(OUT/"freeze.json", dict(selected=selected, certification=selected["train"][:4],
                    bindings=sources, moving_only=True, stationary_DEV="OOD"))
        results = []
        for i, row in enumerate(selected["train"][:4]):
            progress("TRAIN_REPRESENTATION", case=i, source=row["path"])
            results.append(case(row, i))
            atomic_json(OUT/"progress_results.json", results)
        passed = all(r["status"] == "EXECUTION_VERIFIED" for r in results)
        changed = [p for p,h in sources.items() if file_digest(ROOT/p) != h]
        atomic_json(OUT/"preservation.json", dict(files_checked=len(sources), changed=changed))
        if changed:
            raise RuntimeError("Frozen source changed")
        terminal = "READY_FOR_TRAINING" if passed else "BLOCKED"
        atomic_json(OUT/"summary.json", dict(status=terminal, results=results,
                    certified=sum(r["status"] == "EXECUTION_VERIFIED" for r in results), denominator=4,
                    training_updates=0, DEV_candidates=0, elapsed_s=time.time()-START))
        atomic_json(OUT/"status.json", dict(status=terminal, stage="B0_COMPLETE", pid=os.getpid(),
                    heartbeat=time.time(), elapsed_s=time.time()-START))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT/"failure.json", dict(traceback=traceback.format_exc()))
        atomic_json(OUT/"status.json", dict(status="FAILED", pid=os.getpid(), heartbeat=time.time()))
        raise
