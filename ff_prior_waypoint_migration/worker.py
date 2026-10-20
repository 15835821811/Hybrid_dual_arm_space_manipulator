import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import mujoco
from tracking_diffusion.environment import TrackingEnv, Task
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.geometry import CollisionBackend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v7_1/waypoint_migration_001"
START = time.time()
FILES = {}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def bind(p):
    FILES[str(p.relative_to(ROOT))] = sha(p)
    return p


def read(p):
    return json.loads(bind(p).read_text(encoding="utf-8-sig"))


def write(name, value):
    p = OUT / name
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(p)


def tick(stage, **more):
    if time.time()-START > 3600:
        raise TimeoutError("migration hour cap")
    write("status.json", dict(status="RUNNING", pid=os.getpid(), stage=stage,
                             heartbeat=time.time(), **more))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"status.json").exists():
        raise RuntimeError("Existing immutable run")
    tick("source_identity")
    bind(Path(__file__))
    bind(ROOT/"ff_prior_waypoint_migration/docs/EXPERIMENT_PLAN.md")
    for p in (ROOT/"tracking_diffusion").glob("*.py"):
        bind(p)
    for name in ("robot_model_spec_v5.py", "whole_body_verifier_v5.py"):
        bind(ROOT/"model_test"/name)
    bind(ROOT/"tracking_diffusion/config/local10mm_v1.json")
    bind(ROOT/"dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf")
    old = ROOT/"tracking_output/local10mm_v1/teacher_a01/pilot_00"
    copy = ROOT/"tracking_output/ui_pilot/data/episodes/train_20261012"
    a, b = read(old/"result.json"), read(copy/"result.json")
    bind(old/"trace.npz"); bind(copy/"trace.npz")
    write("label_conflict.json", dict(
        same_trace=sha(old/"trace.npz") == sha(copy/"trace.npz"),
        later_admission=a["accepted_for_teacher_dataset"],
        earlier_admission=b["accepted_for_teacher_dataset"],
        later_execution=a["execution"], earlier_execution=b["execution"],
        resolution="QUARANTINE_BOTH_COPIES_PENDING_CURRENT_REQUALIFICATION",
        reason="Recorded acceleration ratio 1.060242 and 18 motion-limit steps; earlier positive label omits these checks. No historical 500Hz torque trace available to independently recompute peak."))
    historical = read(old.parent/"code_version.json")["sha256"]
    snapshots = []
    for p in (ROOT/"tracking_output").rglob("source_snapshot"):
        if "test" in str(p).lower():
            continue
        names = ("environment.py", "trajectory.py", "pipeline.py", "task_domain.py")
        if all((p/n).exists() for n in names):
            h = hashlib.sha256()
            for n in names:
                h.update(bind(p/n).read_bytes())
            snapshots.append(dict(path=str(p.relative_to(ROOT)), hash=h.hexdigest(),
                                  matches_selected_run=h.hexdigest() == historical))
    write("source_recovery.json", dict(historical_four_file_hash=historical,
          candidates=snapshots, exact_snapshot_found=any(x["matches_selected_run"] for x in snapshots)))
    tick("initial_state_reconstruction")
    env = TrackingEnv(Task(**a["task"]))
    robot = RobotAdapter()
    geometry = CollisionBackend(robot)
    with np.load(old/"trace.npz", allow_pickle=False) as z:
        observed = dict(qpos_max_abs=float(abs(env.data.qpos-z["qpos"][0]).max()),
                        qvel_max_abs=float(abs(env.data.qvel-z["qvel"][0]).max()),
                        q17_max_abs=float(abs(env.state()[0]-z["q"][0]).max()),
                        dq17_max_abs=float(abs(env.state()[1]-z["dq"][0]).max()))
    mask = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, mask))
    mujoco.mj_getState(env.model, env.data, state, mask)
    restored = mujoco.MjData(env.model)
    mujoco.mj_setState(env.model, restored, state, mask)
    check = np.empty_like(state)
    mujoco.mj_getState(env.model, restored, check, mask)
    np.savez_compressed(OUT/"reconstructed_initial_candidate.npz", integration=state,
                        qpos=env.data.qpos, qvel=env.data.qvel, state_mask=int(mask))
    write("initial_state.json", dict(observed_initial_errors=observed,
          reconstruction_roundtrip_max_abs=float(abs(state-check).max()), state_size=len(state),
          state_mask=int(mask), source_kind="DETERMINISTIC_RECONSTRUCTION_NOT_ARCHIVED_FULL_STATE",
          historical_hidden_state_proven=False,
          initial_ctrl_max=float(abs(env.data.ctrl).max()),
          qacc_warmstart_max=float(abs(env.data.qacc_warmstart).max()),
          xfrc_applied_max=float(abs(env.data.xfrc_applied).max()),
          target_qpos_indices=[env.tp.start, env.tp.stop],
          target_qvel_indices=[env.tv.start, env.tv.stop],
          target_initial_velocity=env.data.qvel[env.tv].tolist(),
          physics_steps=0, provenance="Current initialization source bound in freeze.json; observed fields compared; hidden historical fields not recoverable from trace alone."))
    basis = np.column_stack([env.spec.encode_position(np.eye(17)[i]) for i in range(17)])
    write("contract_comparison.json", dict(
        old_runtime_hash=env.spec.runtime_contract_sha256(),
        recorded_runtime_hash=a["model_runtime_contract_sha256"],
        current_spec_hash=robot.spec.runtime_contract_sha256(),
        mapping_shape=list(basis.shape), mapping_max_abs_difference=float(abs(basis-robot.B).max()),
        old_continuum_lower=env.spec.planner_lower[:10].tolist(),
        old_continuum_upper=env.spec.planner_upper[:10].tolist(),
        current_continuum_lower=robot.lower[:10].tolist(), current_continuum_upper=robot.upper[:10].tolist(),
        torque_limits_max_abs_difference=float(abs(env.spec.torque_limits-robot.spec.torque_limits).max()),
        old_contact_enabled_geoms=int(np.count_nonzero(env.model.geom_contype | env.model.geom_conaffinity)),
        current_contact_enabled_geoms=int(np.count_nonzero(robot.model.geom_contype | robot.model.geom_conaffinity)),
        old_pair_count=len(env.pairs), current_pair_count=len(geometry.pairs),
        pair_count_comparison_caveat="Old model includes two workspace obstacles; current bare robot comparison is not a matched-scene pair-set equivalence test.",
        old_target_pregrasp_local=env.grasp_local.tolist(),
        current_DEV_target_pregrasp_local=[0, -.235, 0],
        overall_contract_equivalence=False))
    tick("stationary_inventory")
    rows = read(ROOT/"artifacts/v7_1/waypoint_direct_audit_001/episodes.json")
    motion = []
    for i, row in enumerate(rows):
        if not row["available"]:
            continue
        p = ROOT/row["path"]/"trace.npz"
        with np.load(bind(p), allow_pickle=False) as z:
            vel = z["qvel"][0, env.tv]
            motion.append(dict(path=row["path"], split=row["split"], seed=row["seed"],
                               linear_max=float(abs(vel[:3]).max()),
                               angular_max=float(abs(vel[3:]).max()),
                               stationary=bool(np.max(abs(vel)) <= 1e-12)))
        if i % 50 == 0:
            tick("stationary_inventory", records=len(motion))
    # Metadata-only discovery: do not open TEST episodes or treat unassigned runs as TRAIN.
    manifests = []
    for base in (ROOT/"tracking_output", ROOT/"artifacts/v7_1"):
        for directory, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not any(w in d.lower() for w in ("test", "diagnostic", "__pycache__"))]
            for name in files:
                if name in ("manifest.json", "split_manifest.json", "dataset_report.json", "pilot_manifest.json"):
                    p = Path(directory)/name
                    if p.stat().st_size > 5*1024*1024:
                        manifests.append(dict(path=str(p.relative_to(ROOT)), excluded="metadata_over_5MiB"))
                        continue
                    value = read(p)
                    manifests.append(dict(path=str(p.relative_to(ROOT)),
                        keys=list(value) if isinstance(value, dict) else ["list"],
                        domain=value.get("domain") if isinstance(value, dict) else None))
    write("stationary_search.json", dict(records=motion, examined=len(motion),
          stationary_count=sum(x["stationary"] for x in motion),
          target_velocity_address=[env.tv.start, env.tv.stop],
          manifests=manifests,
          excluded=["TEST episode trees", "diagnostic trees", "unassigned baselines as TRAIN",
                    "static endpoint-only samples", "DEV and derived outputs"],
          limitation="Metadata roots tracking_output and artifacts/v7_1; not a proof about every external drive. Existing split provenance required."))
    write("freeze.json", dict(files=FILES, new_physics_steps=0, training_updates=0))
    changed = [p for p, h in FILES.items() if sha(ROOT/p) != h]
    write("preservation.json", dict(files_checked=len(FILES), changed=changed))
    if changed:
        raise RuntimeError("Source changed during migration")
    summary = dict(status="BLOCKED", blockers=[
        "Selected TRAIN has recorded motion-limit failure; conflicting earlier positive label cannot override it",
        "No stationary TRAIN/VAL found in the examined 581 existing dynamic records",
        "Full historical integration state not archived; reconstructed observed fields do not prove hidden historical state",
        "Old and current contracts differ; no current physical recertification performed"],
        initial_observed_errors=observed, stationary_records=sum(x["stationary"] for x in motion),
        dynamic_records_examined=len(motion), new_training_updates=0, new_physics_steps=0,
        representation_execution="BLOCKED", training="BLOCKED", DEV="BLOCKED",
        elapsed_s=time.time()-START)
    write("summary.json", summary)
    write("status.json", dict(status="BLOCKED", stage="MIGRATION_DIAGNOSIS_COMPLETE",
                             pid=os.getpid(), heartbeat=time.time(), elapsed_s=time.time()-START))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write("failure.json", dict(traceback=traceback.format_exc()))
        write("status.json", dict(status="FAILED", pid=os.getpid(), heartbeat=time.time()))
        raise
