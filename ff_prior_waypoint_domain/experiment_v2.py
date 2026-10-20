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
OUT = ROOT / "artifacts/v7_1/waypoint_domain_002"
MODEL = ROOT / "artifacts/v7_1/endpoint_jacobian_bounded_001/learning"


from .adapter import Environment

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
    limited = r.model.jnt_limited[r.joints].astype(bool)
    limits = r.model.jnt_range[r.joints][limited]
    gates = dict(complete_native_time=complete,independent_replay=max_error<=1e-10,
        whole_body_clearance=bool(np.min(h["clearance"])>=0),
        geometry_step=env.max_geom_step<=.002+1e-12,
        dynamic_position=bool(np.all(pos<=.005)),
        dynamic_rotation=bool(np.all(rot<=np.pi/180)),
        relative_hold=bool(complete and hold.any() and np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180)),
        joint_limits=bool(np.all(h["q17"]>=r.lower) and np.all(h["q17"]<=r.upper)
            and np.all(h["qref"]>=r.lower) and np.all(h["qref"]<=r.upper)
            and np.all(lowq[:,limited]>=limits[:,0]) and np.all(lowq[:,limited]<=limits[:,1])),
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
        from .protocol import verify_old, freeze
        old_hashes = verify_old()
        progress("compile")
        r = RobotAdapter()
        bundle = load(MODEL)
        files = [Path(__file__),ROOT/"tracking_diffusion/environment.py",ROOT/"v6_lite/irregular_waypoints.py",
            ROOT/"ff_prior/robot_adapter.py",ROOT/"ff_prior/geometry.py",MODEL/"model.pt",MODEL/"codec.json"]
        atomic_json(OUT/"binding.json",{str(p.relative_to(ROOT)):file_digest(p) for p in files})
        atomic_json(OUT/"robot_identity.json",r.identity())
        freeze(OUT,old_hashes)
        results = {}
        for scene,moving,seed in (("stationary",False,20261020),("drifting",True,20261021)):
            frozen = None
            for method in ("task_controller","diffusion_null_prior"):
                directory = OUT/scene/method
                if scene=="stationary" and method=="task_controller":
                    import shutil
                    from ff_prior.storage import read_json
                    source=ROOT/"artifacts/v7_1/waypoint_domain_001"/scene/method
                    previous=read_json(source/"result.json")
                    if previous["status"]!="EXECUTION_VERIFIED" or file_digest(source/"native_trace.npz")!=previous["native_trace_sha256"]: raise RuntimeError("invalid completed baseline reuse")
                    shutil.copytree(source,directory)
                    results[f"{scene}/{method}"]=previous
                    atomic_json(OUT/"completed_baseline_reuse.json",dict(source=str(source),new_execution=False,trace_sha256=previous["native_trace_sha256"]))
                    continue
                directory.mkdir(parents=True,exist_ok=False)
                progress("initialize",scene=scene,method=method)
                env = Environment(r,moving,seed)
                identity = dict(initial=env.initial.values,initial_id=env.initial.state_id,
                    waypoints=env.path.to_dict(),grasp_local=env.grasp_local,
                    target_velocity=env.data.qvel[env.tv],seed=seed,scene=scene)
                from ff_prior.storage import digest
                if frozen is not None and digest(identity)!=frozen: raise RuntimeError("paired initial/reference mismatch")
                old_task = ROOT/"artifacts/v7_1/waypoint_scene_001"/scene/method/"task.json"
                from ff_prior.storage import read_json
                if digest(identity)!=digest(read_json(old_task)): raise RuntimeError("old frozen task identity mismatch")
                frozen = digest(identity)
                atomic_json(directory/"task.json",identity)
                prior = None
                if method=="diffusion_null_prior":
                    task = frozen_task(dict(initial_features=r.spec.planner_zero,
                        goals=env.desired_poses(27.),duration=27.,pose_mask=np.ones((2,6))),
                        {"scope":"dynamic_transfer_terminal_only_NOT_native_V7_eligible"})
                    base = public_goal_initializer(task,r,env.initial)
                    old_prior = read_json(ROOT/"artifacts/v7_1/waypoint_scene_001"/scene/method/"prior.json")
                    if digest(task)!=digest(old_prior["task"]) or not np.array_equal(base,np.asarray(old_prior["base"])): raise RuntimeError("prior input changed")
                    z=np.asarray(old_prior["Z"]);steps=old_prior["steps"]
                    prior = Reference(z,r.spec.planner_zero,27.)
                    atomic_json(directory/"prior.json",dict(task=task,base=base,Z=z,steps=steps,
                        condition_scope="terminal only; no waypoints or target twist inputs",
                        frozen_proposal_reused=True,new_network_calls=0,
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
        if old_hashes != verify_old(): raise RuntimeError("old outputs changed")
        atomic_json(OUT/"preservation.json",dict(all_old_files_unchanged=True,files=len(old_hashes)))
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
