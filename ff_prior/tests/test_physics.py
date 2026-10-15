from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
import mujoco
import pytest
from ff_prior.initial_state import InitialState
from ff_prior.proposal_generator import propose,quick_gates
from ff_prior.coupled_predictor import predict_coupled,converge,cache_key,pose_errors
from ff_prior.contracts import AcceptanceConfig,frozen_task
from ff_prior.spline import Reference
from ff_prior.model_gates import evaluate_reference,path_geometry
from ff_prior.physical_audit import audit_execution,replay
from ff_prior.storage import atomic_json


def test_mapping_names_full_state_tip_and_units(physical):
    r,initial,_=physical
    np.testing.assert_allclose(r.Bplus@r.B,np.eye(17),atol=1e-12)
    assert len(r.qpos_ids)==67 and len(r.base_dofs)==6 and not set(r.base_dofs)&set(r.dof_ids)
    d=initial.reset(r)
    r.kinematics(d)
    ee=r.ee(d)
    np.testing.assert_allclose(ee[0,:3,3],d.xpos[r.tip_c]+d.xmat[r.tip_c].reshape(3,3)@r.shape_spec.tip_offset_m)
    d.time=2
    d.qacc_warmstart[:]=.01
    target=mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_JOINT,r.spec.target_free_joint_name)
    target_index=r.model.jnt_qposadr[target]
    d.qpos[target_index]+=.02
    snap=InitialState.capture(r,d)
    reset=snap.reset(r)
    assert reset.time==2
    np.testing.assert_array_equal(reset.qpos,d.qpos)
    assert initial.state_id!=snap.state_id
    assert r.lower[0]==-1 and r.upper[0]==1


def test_scratch_does_not_mutate_initial_and_rejects_momentum(physical):
    r,initial,_=physical
    d=initial.reset(r)
    d.qvel[r.base_dofs[0]]=.01
    moving=InitialState.capture(r,d)
    ref,_=propose(r,1,0,20261015)
    with pytest.raises(ValueError,match="NONZERO"): predict_coupled(moving,ref,r)
    before=initial.values.copy()
    predict_coupled(initial,ref,r)
    np.testing.assert_array_equal(before,initial.values)


@pytest.mark.parametrize("joint",[None,0,12,"both"])
def test_stationary_single_and_dual_reaction_convergence(physical,joint):
    r,initial,_=physical
    z=np.tile(r.spec.planner_zero,(19,1))
    phase=np.linspace(0,1,19)
    if joint=="both":
        z[:,0]+=.08*phase; z[:,12]+=.08*phase
    elif joint is not None:
        z[:,joint]+=.08*phase
    ref=Reference(z,r.spec.planner_zero,10)
    pred,convergence=converge(initial,ref,r)
    assert convergence["passed"]
    assert pred["numerical"]["robot_momentum_residual"]<1e-10
    assert pred["numerical"]["quaternion_norm_error"]<1e-12
    if joint is None:
        np.testing.assert_allclose(pred["base"],np.repeat(pred["base"][:1],len(pred["base"]),axis=0),atol=1e-12)
    else:
        assert np.linalg.norm(pred["base"][-1]-pred["base"][0])>1e-7


def test_path_cache_retiming_changes_and_static_assumption(physical):
    r,initial,c=physical
    a,_=propose(r,2,0,20261015)
    b=Reference(a.z,a.q0,20)
    assert cache_key(initial,a,r,c.policy_hash)==cache_key(initial,b,r,c.policy_hash)
    z=a.z.copy(); z[4,0]+=.02
    assert cache_key(initial,a,r,c.policy_hash)!=cache_key(initial,Reference(z,a.q0,10),r,c.policy_hash)
    with pytest.raises(ValueError): cache_key(initial,a,r,c.policy_hash,static=False)
    p=predict_coupled(initial,a,r); q=predict_coupled(initial,b,r)
    np.testing.assert_allclose(p["base"],q["base"],atol=1e-12)


def test_same_terminal_different_path_can_change_world_goal(physical):
    r,initial,_=physical
    a,_=propose(r,3,0,20261015)
    z=a.z.copy()
    z[3:12,0]+=.3*np.sin(np.linspace(0,np.pi,9))
    z[7:16,12]+=.3*np.sin(np.linspace(0,np.pi,9))
    b=Reference(z,a.q0,20)
    p,q=predict_coupled(initial,a,r),predict_coupled(initial,b,r)
    np.testing.assert_array_equal(a.z[-1],b.z[-1])
    assert np.linalg.norm(p["ee"][-1]-q["ee"][-1])>1e-5


def test_generator_has_zero_search_sqp_and_actual_rollout(physical):
    r,initial,c=physical
    from ff_prior import torque_tracker
    with patch.object(mujoco,"mj_step",side_effect=AssertionError("forbidden physical step")), \
         patch.object(torque_tracker,"track",side_effect=AssertionError("forbidden tracker")), \
         patch("scipy.optimize.minimize",side_effect=AssertionError("forbidden SQP")):
        before=r.calls.copy()
        for variant in range(5):
            ref,_=propose(r,4,variant,20261015)
            pred,report=evaluate_reference(r,initial,ref,c,AcceptanceConfig())
            assert report["validation_level"]=="MODEL_VALID"
        for k in ("physics_steps","torque_rollout","sqp","rrt"):
            assert r.calls[k]==before[k]


def test_pair_policy_retains_targets_mounts_and_interarm(physical):
    r,initial,c=physical
    pairs=[c.describe(*p) for p in c.pairs]
    assert c.policy["intentional_contact_geom_allowlist"]==[]
    assert any("target_satellite" in (p["body_a"],p["body_b"]) and "end_effector_r" in (p["body_a"],p["body_b"]) for p in pairs)
    assert any("link_30" in (p["body_a"],p["body_b"]) and "base_link_r" in (p["body_a"],p["body_b"]) for p in pairs)
    d=initial.reset(r)
    a=c.scan(d,0)
    b=c.scan(d,0,exhaustive=True)
    assert abs(a["minimum_clearance_m"]-b["minimum_clearance_m"])<1e-10
    assert a["passed"] and not c.scan(d,.02)["passed"]


def test_real_obstacle_and_middle_collision_are_rejected(physical):
    r,initial,c=physical
    d=initial.reset(r)
    safe=d.qpos.copy()
    target=mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_JOINT,r.spec.target_free_joint_name)
    adr=r.model.jnt_qposadr[target]
    d.qpos[adr:adr+3]=r.ee(d)[0,:3,3]
    r.kinematics(d)
    assert not c.scan(d,0)["passed"]
    report=path_geometry(r,initial,np.stack([safe,d.qpos,safe]),c,0)
    assert not report["passed"] and report["minimum_clearance_m"]<0


def test_native_off_manifold_range_and_geometry_not_projected(physical):
    r,initial,c=physical
    d=initial.reset(r)
    rng=np.random.default_rng(45)
    null_projector=np.eye(67)-r.B@r.Bplus
    for _ in range(100):
        low=null_projector@rng.normal(size=67)
        assert np.linalg.norm(r.Bplus@low)<1e-12
        d.qpos[r.qpos_ids]=r.B@r.spec.planner_zero+low
        r.kinematics(d)
        if c.scan(d,0)["minimum_clearance_m"]<0: break
    native=d.xpos.copy()
    native_clearance=c.scan(d,0)["minimum_clearance_m"]
    d.qpos[r.qpos_ids]=r.B@(r.Bplus@d.qpos[r.qpos_ids])
    r.kinematics(d)
    assert np.max(np.linalg.norm(native-d.xpos,axis=1))>.01
    assert native_clearance<0 and c.scan(d,0)["minimum_clearance_m"]>=0
    ref,_=propose(r,0,0,20261015)
    z=ref.z.copy(); z[5,0]=1.01
    assert not quick_gates(r,Reference(z,r.spec.planner_zero,10))["domain"]


def test_native_self_interarm_and_mount_counterexamples(physical):
    r,initial,c=physical
    rng=np.random.default_rng(573)
    found={}
    continuum={f"link_{i}" for i in range(1,31)}|{"end_effector"}
    mounts={"base_of_satelltte","driving_box","base_link","base_link_r"}
    d=initial.reset(r)
    for _ in range(3000):
        q=rng.uniform(r.lower,r.upper)
        d.qpos[r.qpos_ids]=r.B@q
        r.kinematics(d)
        result=c.scan(d,0)
        for pair in result["violations"]:
            a,b=pair["body_a"],pair["body_b"]
            if "target_satellite" in (a,b): continue
            category="arm_base" if a in mounts or b in mounts else "inter_arm" if (a in continuum)!=(b in continuum) else "self"
            found.setdefault(category,{"q17":q.copy(),"pair":pair})
        if len(found)==3: break
    atomic_json(Path("artifacts/v7_1/geometry_counterexamples.json"),found)
    assert set(found)=={"arm_base","inter_arm","self"}


def test_real_ctrl_execution_independent_replay_and_reaction(physical):
    r,initial,c=physical
    ref,_=propose(r,0,0,20261015)
    pred,model=evaluate_reference(r,initial,ref,c,AcceptanceConfig())
    assert model["validation_level"]=="MODEL_VALID"
    task=frozen_task({"initial_features":r.spec.planner_zero,"goals":pred["ee"][-1],"duration":ref.duration,
        "pose_mask":np.ones((2,6))},{"robot_hash":initial.model_hash,"initial_state_id":initial.state_id,
        "pair_policy_hash":c.policy_hash,"acceptance":asdict(AcceptanceConfig())})
    out=Path("artifacts/v7_1/test_physics")
    report=audit_execution(task,AcceptanceConfig(),r,initial,ref,out)
    assert report["replay"]["passed"],report["replay"]
    assert report["metadata"]["robot_state_writes_during_step_loop"]==0
    assert report["metadata"]["physics_steps"]==5250
    assert report["maximum_base_translation_m"]>1e-6
    assert report["gates"]["complete_native_time"]
    # This assertion is a real physical requirement, not a mocked pass.
    assert report["level"]=="EXECUTION_VERIFIED",report["gates"]
