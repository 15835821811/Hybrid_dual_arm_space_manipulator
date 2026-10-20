"""Bounded dynamic-task transfer; all outcome claims come from saved native states."""
from pathlib import Path
from types import SimpleNamespace
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import traceback
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from tracking_diffusion.environment import TrackingEnv, Task, free_slices
from v6_lite.irregular_waypoints import build_original_irregular_target
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.geometry import CollisionBackend
from ff_prior.initial_state import InitialState
from ff_prior.storage import atomic_json, atomic_npz, file_digest, RunLock
from ff_prior.contracts import frozen_task
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior_endpoint_jacobian.learning import load
from ff_prior_endpoint_precision.comparison import unguided_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v7_1/waypoint_scene_001"
MODEL = ROOT / "artifacts/v7_1/endpoint_jacobian_bounded_001/learning"


class Environment(TrackingEnv):
    """Reuse tracking algebra, replace physical model and every collision pair."""
    def __init__(self, robot, moving, seed):
        self.robot = robot
        self.model, self.spec = robot.model, robot.spec
        self.data = robot.new_data()
        self.task = Task(seed=seed, obstacle_count=0)
        self.qids, self.vids = robot.qpos_ids, robot.dof_ids
        self.bp, self.bv = free_slices(self.model, self.spec.base_joint_name)
        self.tp, self.tv = free_slices(self.model, self.spec.target_free_joint_name)
        self.base_ids = robot.base_dofs
        self.body_ids = [robot.tip_c, robot.tip_r, robot.id(mujoco.mjtObj.mjOBJ_BODY, "target_satellite")]
        pc, rc = self.pose(0)
        self.rigid_home, self.rigid_home_rotation = self.pose(1)
        _, rt = self.pose(2)
        self.path = build_original_irregular_target(pc, rc, seed)
        self.grasp_local = np.array([0., -.235, 0.])
        self.grasp_rotation_local = rt.T @ self.rigid_home_rotation
        self.approach_offset = np.zeros(3)
        rng = np.random.default_rng(seed)
        self.data.qpos[self.tp.start:self.tp.start+3] += rng.uniform([-.012,-.008,-.010],[.012,.008,.010])
        if moving:
            self.data.qvel[self.tv] = np.r_[rng.uniform([-.0025,.001,-.0015],[.0025,.004,.0015]), rng.uniform(-.0015,.0015,3)]
        robot.forward(self.data)
        self.mass = np.zeros((self.model.nv,self.model.nv))
        self.reference_q = self.spec.planner_zero.copy()
        self.command = np.zeros(17)
        self.collision = CollisionBackend(robot)
        self.pairs = [SimpleNamespace(geom_a=int(a),geom_b=int(b),pair_class="V7_no_exempt",
            geom_a_name=self.collision.describe(a,b)["geom_a"],geom_b_name=self.collision.describe(a,b)["geom_b"])
            for a,b in self.collision.pairs]
        self.pair_a, self.pair_b = self.collision.pairs.T
        # Planner buffer only. Acceptance is still exactly nonpenetration.
        self.safe_distances = np.full(len(self.pairs), .001)
        self.minimum_clearance = self.minimum_margin = float("inf")
        self.minimum_pair = None
        self.qp_failures = 0
        self.initial = InitialState.capture(robot, self.data)
        self.scratch = robot.new_data()
        self.history = {k:[] for k in ("time","qpos","qvel","ee","desired","base","q17","qref",
            "clearance","momentum","exchange","skeleton_c","skeleton_r")}
        self.controls, self.raw, self.command_qd, self.command_qdd = [],[],[],[]
        self.max_geom_step = 0.
        self.interpolation_states = 0
        self.previous_qpos = None
        self.previous_geom = None
        self.previous_geom_r = None
        self.record()

    def desired_poses(self, t):
        ref = self.reference(t)
        poses = np.tile(np.eye(4),(2,1,1))
        for i,k in enumerate((0,4)):
            poses[i,:3,3],poses[i,:3,:3] = ref[k],ref[k+1]
        return poses

    def record(self):
        r,d = self.robot,self.data
        r.forward(d)
        clearance = self.collision.scan(d)["minimum_clearance_m"]
        # Subdivide measured geometry motion; quaternion interpolation uses MuJoCo.
        if self.previous_qpos is not None:
            angle = Rotation.from_matrix(d.geom_xmat.reshape(-1,3,3) @ self.previous_geom_r.transpose(0,2,1)).magnitude()
            motion = np.max(np.linalg.norm(d.geom_xpos-self.previous_geom,axis=1)+angle*self.model.geom_rbound)
            n = max(2,int(np.ceil(motion/.002)))
            if n > 100:
                raise RuntimeError("geometry interpolation budget exceeded")
            self.max_geom_step = max(self.max_geom_step,float(motion/n))
            tangent = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model,tangent,1.,self.previous_qpos,d.qpos)
            for j in range(1,n):
                self.scratch.qpos[:] = self.previous_qpos
                mujoco.mj_integratePos(self.model,self.scratch.qpos,tangent,j/n)
                r.kinematics(self.scratch)
                clearance = min(clearance,self.collision.scan(self.scratch)["minimum_clearance_m"])
                self.interpolation_states += 1
        self.previous_qpos = d.qpos.copy()
        self.previous_geom = d.geom_xpos.copy()
        self.previous_geom_r = d.geom_xmat.reshape(-1,3,3).copy()
        mass = r.mass(d)
        values = dict(time=d.time,qpos=d.qpos,qvel=d.qvel,ee=r.ee(d),
            desired=self.desired_poses(float(d.time)),base=d.qpos[r.base_qpos],
            q17=r.Bplus@d.qpos[self.qids],qref=self.reference_q,clearance=clearance,
            momentum=(mass@d.qvel)[r.base_dofs],
            exchange=mass[np.ix_(r.base_dofs,r.dof_ids)]@d.qvel[r.dof_ids])
        values["skeleton_c"] = d.xpos[[r.id(mujoco.mjtObj.mjOBJ_BODY,f"link_{j}") for j in range(1,31)]]
        values["skeleton_r"] = d.xpos[[r.id(mujoco.mjtObj.mjOBJ_BODY,n) for n in
            ("base_link_r","shoulder_link","back_arm_link","fore_arm_link","wrist_link","end_effector_r")]]
        for k,v in values.items(): self.history[k].append(np.array(v).copy())
        if clearance < 0:
            raise RuntimeError(f"penetration: {clearance} m at {d.time}s")

    def velocity(self, prior):
        g,drift = self.generalized_map()
        j,offset = self.task_jacobian(g,drift)
        weights = np.repeat([1.,2.,1.4,2.4],3)
        jw = weights[:,None]*j
        h = jw.T@jw+1e-5*np.eye(17)
        q,_ = self.state()
        rhs = jw.T@(weights*(self.desired_velocity(True)-offset))
        rhs += 1e-5*.08*(self.spec.planner_zero-q)
        velocity = np.linalg.solve(h,rhs)
        if prior is not None:
            # Explicit transfer adapter, NOT a claim that network encodes path.
            preferred = prior.evaluate(self.data.time,1)+.5*(prior.evaluate(self.data.time)-q)
            preferred = np.clip(preferred,-.05,.05)
            null = np.eye(17)-np.linalg.pinv(j,rcond=1e-7)@j
            velocity += null@preferred
        return self.velocity_filter(velocity,g,drift,h)

    def advance_native(self, velocity):
        r,d = self.robot,self.data
        old = self.command.copy()
        acceleration = (velocity-old)/.02
        for substep in range(10):
            v = old+(substep+1)/10*(velocity-old)
            self.reference_q += .002*v
            r.forward(d)
            mass = r.mass(d)
            w = np.r_[np.full(60,42.),np.full(7,34.)]
            aref = r.B@acceleration+w*w*(r.B@self.reference_q-d.qpos[r.qpos_ids])+2*w*(r.B@v-d.qvel[r.dof_ids])
            aref = np.clip(aref,-np.r_[np.full(60,45.),np.full(7,70.)],np.r_[np.full(60,45.),np.full(7,70.)])
            b,a = r.base_dofs,r.dof_ids
            force = d.qfrc_bias-d.qfrc_passive
            ba = -np.linalg.solve(mass[np.ix_(b,b)],mass[np.ix_(b,a)]@aref+force[b])
            raw = mass[np.ix_(a,b)]@ba+mass[np.ix_(a,a)]@aref+force[a]
            if not np.isfinite(raw).all(): raise RuntimeError("nonfinite torque")
            d.ctrl[r.actuator_ids] = np.clip(raw,-r.spec.torque_limits,r.spec.torque_limits)
            self.controls.append(d.ctrl.copy())
            self.raw.append(raw.copy())
            self.command_qd.append(v.copy())
            self.command_qdd.append(acceleration.copy())
            mujoco.mj_step(r.model,d)
            self.record()
            if np.any(d.warning.number): raise RuntimeError("MuJoCo warning")
        self.command = velocity.copy()


def evaluate(env, directory, reason, progress):
    r = env.robot
    h = {k:np.asarray(v) for k,v in env.history.items()}
    ctrl,raw = np.asarray(env.controls),np.asarray(env.raw)
    atomic_npz(directory/"native_trace.npz",**h,ctrl=ctrl,raw_torque=raw,
        reference_qd=np.asarray(env.command_qd),reference_qdd=np.asarray(env.command_qdd))
    # Independently replay exact saved torques, not a second controller run.
    replay = env.initial.reset(r)
    max_error = 0.
    for i,u in enumerate(ctrl):
        replay.ctrl[:] = u
        mujoco.mj_step(r.model,replay)
        max_error = max(max_error,float(np.max(abs(replay.qpos-h["qpos"][i+1]))),
                        float(np.max(abs(replay.qvel-h["qvel"][i+1]))))
        if i%1000==0: progress("independent_replay",steps=i,total=len(ctrl))
    pos,rot = pose_errors(h["ee"],h["desired"])
    lowq,lowv = h["qpos"][:,r.qpos_ids],h["qvel"][:,r.dof_ids]
    v = lowv@r.Bplus.T
    a = np.diff(v,axis=0)/.002
    la = np.diff(lowv,axis=0)/.002
    mp = np.array([np.max(np.linalg.norm(h["momentum"][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    me = np.array([np.max(np.linalg.norm(h["exchange"][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    hold = h["time"] >= 25.5-1e-9
    complete = len(ctrl)==13500 and reason is None
    limits = r.model.jnt_range[r.joints]
    gates = dict(complete_native_time=complete,independent_replay=max_error<=1e-10,
        whole_body_clearance=bool(np.min(h["clearance"])>=0),
        geometry_step=env.max_geom_step<=.002+1e-12,
        dynamic_position=bool(np.all(pos<=.005)),
        dynamic_rotation=bool(np.all(rot<=np.pi/180)),
        relative_hold=bool(complete and hold.any() and np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180)),
        joint_limits=bool(np.all(h["q17"]>=r.lower) and np.all(h["q17"]<=r.upper)
            and np.all(h["qref"]>=r.lower) and np.all(h["qref"]<=r.upper)
            and np.all(lowq>=limits[:,0]) and np.all(lowq<=limits[:,1])),
        derivative_limits=bool(len(a)>0 and np.all(abs(v)<=r.spec.planner_velocity_limits)
            and np.all(abs(a)<=r.spec.planner_acceleration_limits)
            and np.all(abs(lowv)<=abs(r.B)@r.spec.planner_velocity_limits)
            and np.all(abs(la)<=abs(r.B)@r.spec.planner_acceleration_limits)
            and np.all(abs(np.asarray(env.command_qd))<=r.spec.planner_velocity_limits)
            and np.all(abs(np.asarray(env.command_qdd))<=r.spec.planner_acceleration_limits)),
        raw_torque=bool(len(raw)>0 and np.all(abs(raw)<=r.spec.torque_limits+1e-12)),
        tracking=bool(np.max(abs(h["q17"]-h["qref"]))<=.02),
        manifold=bool(np.max(abs(lowq-h["q17"]@r.B.T))<=.02),
        momentum=bool(np.all(mp<=1e-5+.002*me)),
        no_hidden_reset=True,no_external_wrench=True,
        finite=bool(np.isfinite(h["qpos"]).all() and np.isfinite(h["qvel"]).all()))
    def dist(x): return dict(mean=np.mean(x,axis=0),p95=np.percentile(x,95,axis=0),maximum=np.max(x,axis=0),final=x[-1])
    report = dict(status="EXECUTION_VERIFIED" if all(gates.values()) else "EXECUTION_FAILED",
        gates=gates,stop_reason=reason,steps=len(ctrl),planned_steps=13500,
        position_mm=dist(pos*1000),rotation_deg=dist(np.rad2deg(rot)),
        hold_position_max_mm=np.max(pos[hold],axis=0)*1000 if hold.any() else None,
        minimum_clearance_mm=float(np.min(h["clearance"])*1000),replay_max_error=max_error,
        momentum_peak=mp,momentum_threshold=1e-5+.002*me,
        interpolation_states=env.interpolation_states,maximum_geometry_step_m=env.max_geom_step,
        native_trace_sha256=file_digest(directory/"native_trace.npz"),
        continuous_time_certificate=False,hardware_experiment=False)
    atomic_json(directory/"result.json",report)
    return report


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"summary.json").exists(): return
    start = time.time()
    def progress(stage,**info):
        atomic_json(OUT/"status.json",dict(status="RUNNING",pid=os.getpid(),started_at=start,
            heartbeat=time.time(),stage=stage,progress=info))
    with RunLock(OUT):
        progress("compile")
        r = RobotAdapter()
        bundle = load(MODEL)
        files = [Path(__file__),ROOT/"tracking_diffusion/environment.py",ROOT/"v6_lite/irregular_waypoints.py",
            ROOT/"ff_prior/robot_adapter.py",ROOT/"ff_prior/geometry.py",MODEL/"model.pt",MODEL/"codec.json"]
        atomic_json(OUT/"binding.json",{str(p.relative_to(ROOT)):file_digest(p) for p in files})
        atomic_json(OUT/"robot_identity.json",r.identity())
        results = {}
        for scene,moving,seed in (("stationary",False,20261020),("drifting",True,20261021)):
            frozen = None
            for method in ("task_controller","diffusion_null_prior"):
                directory = OUT/scene/method
                directory.mkdir(parents=True,exist_ok=False)
                progress("initialize",scene=scene,method=method)
                env = Environment(r,moving,seed)
                identity = dict(initial=env.initial.values,initial_id=env.initial.state_id,
                    waypoints=env.path.to_dict(),grasp_local=env.grasp_local,
                    target_velocity=env.data.qvel[env.tv],seed=seed,scene=scene)
                from ff_prior.storage import digest
                if frozen is not None and digest(identity)!=frozen: raise RuntimeError("paired initial/reference mismatch")
                frozen = digest(identity)
                atomic_json(directory/"task.json",identity)
                prior = None
                if method=="diffusion_null_prior":
                    task = frozen_task(dict(initial_features=r.spec.planner_zero,
                        goals=env.desired_poses(27.),duration=27.,pose_mask=np.ones((2,6))),
                        {"scope":"dynamic_transfer_terminal_only_NOT_native_V7_eligible"})
                    base = public_goal_initializer(task,r,env.initial)
                    z,steps = unguided_sample(bundle,base,task,seed)
                    prior = Reference(z,r.spec.planner_zero,27.)
                    atomic_json(directory/"prior.json",dict(task=task,base=base,Z=z,steps=steps,
                        condition_scope="terminal only; no waypoints or target twist inputs",
                        adapter="task_J_nullspace_velocity_0.05rad/s"))
                reason = None
                try:
                    for tick in range(1350):
                        if time.time()-start>7200: raise RuntimeError("wall budget")
                        env.advance_native(env.velocity(prior))
                        if tick%25==0: progress("native_torque",scene=scene,method=method,sim_time=float(env.data.time))
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    (directory/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
                results[f"{scene}/{method}"] = evaluate(env,directory,reason,progress)
                progress("case_saved",scene=scene,method=method)
        atomic_json(OUT/"summary.json",dict(results=results,elapsed_s=time.time()-start,
            training=False,expansion=False,final_TEST=False,scope="dynamic noncontact development, two cases"))
        atomic_json(OUT/"status.json",dict(status="FINISHED",pid=os.getpid(),heartbeat=time.time(),
            stage="simulation_complete_visualization_pending",elapsed_s=time.time()-start))


if __name__=="__main__":
    try: main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        atomic_json(OUT/"status.json",dict(status="FAILED",pid=os.getpid(),heartbeat=time.time()))
        raise
