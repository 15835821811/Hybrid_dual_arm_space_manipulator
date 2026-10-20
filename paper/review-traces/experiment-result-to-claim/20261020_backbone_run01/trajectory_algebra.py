"""Read-only algebra over every saved NEW native trajectory; never creates an MjModel."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="2"
import sys,json
from pathlib import Path
DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[3]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True;os.chdir(ROOT)
import numpy as np
from ff_prior.storage import read_json,digest
from ff_prior.robot_adapter import physical_spec
from ff_prior.spline import Reference
from ff_prior.physical_audit import pose_errors
OUT=ROOT/"artifacts/v7_1/backbone_pair_smoke_002"
physical=read_json(OUT/"physical_contract.json")
spec=physical_spec()
B=spec.planner_to_low_level;Bp=spec.low_level_to_planner
assert B.shape==(67,17) and np.allclose(Bp@B,np.eye(17),atol=1e-12)
idx=physical["identity"]["joint_index_map"]
qi=[idx[n]["qpos"] for n in spec.low_level_joint_names]
vi=[idx[n]["dof"] for n in spec.low_level_joint_names]
ai=[idx[n]["actuator"] for n in spec.low_level_joint_names]
lower=np.asarray(physical["identity"]["planner_lower"]);upper=np.asarray(physical["identity"]["planner_upper"])
result={"scope":"72 new saved trajectories only; no simulator or old baseline reruns","rows":[]}
for path in sorted(OUT.glob("executions/*/*/*/native_trace.npz")):
    d=path.parent;r=read_json(d/"report.json");task=read_json(d/"frozen_task.json")
    seed,sid,kind=d.relative_to(OUT/"executions").parts
    select=read_json(d/"selection.json")
    candidate=read_json(OUT/"candidates"/seed/sid/kind/f'{select["selected_id"]:02d}.json')
    reference=Reference(candidate["Z"],task["condition"]["initial_features"],task["condition"]["duration"])
    assert r["metadata"]["reference_hash"]==reference.reference_hash
    assert r["pair_policy_hash"]==task["contract"]["pair_policy_hash"]
    with np.load(path) as z: a={k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in a.values())
    n=len(a["ctrl"]);t=a["time"]-a["time"][0];dt=spec.simulation_timestep
    assert n==5250 and a["qpos"].shape==(5251,81) and a["qvel"].shape==(5251,79)
    assert a["ctrl"].shape==(5250,67) and a["raw_torque"].shape==(5250,67)
    assert np.allclose(np.diff(t),dt,rtol=0,atol=1e-12)
    assert abs(t[-1]-task["condition"]["duration"]-.5)<1e-9
    q=a["qpos"][:,qi]@Bp.T; v=a["qvel"][:,vi]@Bp.T
    acc=np.diff(v,axis=0)/dt
    assert np.allclose(q,a["q_actual_17"],rtol=0,atol=1e-12)
    assert np.allclose(q[0],task["condition"]["initial_features"],rtol=0,atol=1e-12)
    assert np.max(abs(a["qvel"][0]))==0
    assert np.all(q>=lower) and np.all(q<=upper)
    assert np.all(abs(v)<=spec.planner_velocity_limits)
    assert np.all(abs(acc)<=spec.planner_acceleration_limits)
    assert np.all(abs(a["qvel"][:,vi])<=abs(B)@spec.planner_velocity_limits)
    assert np.all(abs(np.diff(a["qvel"][:,vi],axis=0)/dt)<=abs(B)@spec.planner_acceleration_limits)
    for order,key in ((0,"reference_q"),(1,"reference_qd"),(2,"reference_qdd")):
        expected=reference.evaluate(np.arange(n)*dt,order)
        assert np.allclose(expected,a[key],rtol=0,atol=1e-12)
    ref=reference.evaluate(t)
    assert np.all(ref>=lower) and np.all(ref<=upper)
    assert np.all(abs(reference.evaluate(t,1))<=spec.planner_velocity_limits)
    assert np.all(abs(reference.evaluate(t,2))<=spec.planner_acceleration_limits)
    torque_ratio=float(np.max(abs(a["raw_torque"])/spec.torque_limits))
    assert torque_ratio<=1 and np.allclose(a["ctrl"][:,ai],a["raw_torque"],rtol=0,atol=1e-12)
    tracking=float(np.max(abs(q-ref))); manifold=float(np.max(abs(a["qpos"][:,qi]-q@B.T)))
    assert abs(tracking-r["tracking_max_rad"])<1e-12 and tracking<=.02
    assert abs(manifold-r["manifold_max_rad"])<1e-12 and manifold<=.02
    position,rotation=pose_errors(a["ee"],task["condition"]["goals"])
    assert np.max(abs(position[-1]-r["final_position_error_m"]))<1e-12
    assert np.max(abs(rotation[-1]-r["final_rotation_error_rad"]))<1e-12
    assert np.all(position[t>=10-1e-9]<=.005) and np.all(rotation[t>=10-1e-9]<=np.pi/180)
    assert len(a["clearance_m"])==n+1 and a["clearance_m"].min()>=0
    assert r["minimum_clearance_m"]<=a["clearance_m"].min()+1e-12
    assert r["checked_states"]==2*n+1 and not r["continuous_time_certified"]
    momentum=np.array([np.max(np.linalg.norm(a["base_generalized_momentum"][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    assert np.allclose(momentum,r["momentum_peak_linear_angular"],rtol=0,atol=1e-12)
    assert np.all(momentum<=r["momentum_threshold_linear_angular"])
    base_motion=float(np.max(np.linalg.norm(a["base"][:,:3]-a["base"][0,:3],axis=1)))
    assert abs(base_motion-r["maximum_base_translation_m"])<1e-12 and base_motion>0
    assert digest(a["ctrl"])==r["ctrl_hash"]
    assert r["replay"]["physics_steps"]==n and not r["replay"]["copies_actual_state_each_step"]
    # The replay residual record is original evidence. Its physics is not rerun here.
    assert r["replay"]["passed"] and max(r["replay"]["maximum_errors"].values())<=1e-10
    assert all(r["gates"].values()) and not r["hardware_validated"]
    result["rows"].append({"key":f"{seed}/{sid}/{kind}","native_steps":n,"replay_steps":r["replay"]["physics_steps"],
       "torque_fraction_max":torque_ratio,"position_mm":1000*float(position[-1].max()),
       "rotation_deg":float(np.rad2deg(rotation[-1].max())),"clearance_m":float(a["clearance_m"].min()),
       "base_translation_m":base_motion,"tracking_rad":tracking,"manifold_rad":manifold})
assert len(result["rows"])==72
result["status"]="PASS"
result["physical_replay_rerun"]=False
result["limitations"]=["No separate raw replay trajectory retained; original per-field residuals and frozen replay implementation inspected.",
 "No geometry distance/midpoint or mass-matrix recomputation in this read-only review; stored per-native-state clearance/momentum arrays checked against original reports and frozen full-body evaluator.",
 "Full integration-state identity is bound by task/run/code; saved native arrays alone omit initial applied-force/mocap/userdata fields."]
(DEST/"reviewer_trajectory_algebra.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"status":result["status"],"trajectories":len(result["rows"]),"max_torque_fraction":max(r["torque_fraction_max"] for r in result["rows"]),"min_clearance":min(r["clearance_m"] for r in result["rows"]),"max_base_translation":max(r["base_translation_m"] for r in result["rows"])}))
