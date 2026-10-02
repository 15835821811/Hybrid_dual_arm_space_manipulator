"""Uncertified controller-free open-loop inertia shadow.

This program never calls run_scenario, a QP, a feedback torque controller, or
the runtime executor. The historical torque array is the complete input.
Geometry is observed in separate MjData, without changing the replay state.
Neither completion nor a positive measured clearance certifies closed-loop
robustness, wall deployment, hardware, or continuous-time collision safety.

Nothing executes on import. Explicit stages, each with a new exclusive output:
  python -m v6_lite.run_research_inertia_shadow --stage plan --output-dir v6_lite/output/runs/NEW
  python -m v6_lite.run_research_inertia_shadow --stage replay --output-dir v6_lite/output/runs/NEW
  python -m v6_lite.run_research_inertia_shadow --stage observe --output-dir v6_lite/output/runs/NEW --replay-manifest-sha SHA
`all` does these sequentially. `smoke` creates a separate output and runs only
scene 00 / alpha 1 for 1..10 physics steps; it is never a complete experiment.
Source and input identity are frozen before every explicit execution stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import mujoco
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / ".git").exists() and (p / "model_test").is_dir())
sys.path.insert(0, str(ROOT))

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere,
)
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id

BASELINE = ROOT / "v6_lite/output/runs/research_acceptance_01"
BASELINE_SOURCE = "9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c"
BASELINE_MANIFEST_SHA = "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"
MODEL_CONTRACT_SHA = "ec36bd1e6c1f47e5dcfb94be8da4c147729f4ccd2ed62aaf1ce8cf505584ac3f"
MODEL_BUNDLE_SHA = "638cee34a2dca5c79d62f1898c99caa810510791faa0e8caca18dab5c7693536"
ALPHAS = (1.0, 0.95, 1.05)  # nominal replay must qualify before perturbations
DT = .002
TASK_PERIOD = .020  # input segmentation only; shadow does NOT solve new QPs
STEPS = 13500
SUBDIVISIONS = 4
GATE = .005
EE_OFFSET = np.array([.0475, 0., 0.])
FIXED_TORQUE_SHA = (
    "0eaa9e47f2398ec91e7d57ba21ecb3f52ff3f6f0d24c2d6cf4e27304a12ff092",
    "9ae518f8a3e1d128be7a5f1372c567e4e3392b3ed2d8d08c29c704786bcd62f4",
    "487f93f956e7c4c55e1a7355b3f5a482a737701b0c1edb1c9b582424993d3b86",
    "0a32c80ee382b8973500182e211d87d62804e2d6ba2c7ff51f359a17c038eadc",
    "bad41848faa2de5c3d37f1d763a9a95d441a3a2590fd7f2d667739237148507d",
)
FIXED_SCENARIO_SHA = (
    "b60f8885aa60143f5f70536a16d196d68cb92f9d2ff55e9446e663a2f300e1d6",
    "1c13c4f92d133d927d607a6d24278a6ac020ccb877d56e416a2d767d89ac589a",
    "534021bf34e4c9b0d390e7513650087f897240400ff52cedce9df0b72cf42def",
    "69b849dcd22247da7a0d6758acb9760652561035a83bc5ab789da4e50be9925f",
    "99e5193a65abd36974844f617998fe387e8f50b3c025111e34c5c463901ee5fc",
)
LIMITATIONS = {
    "experiment": "UNCERTIFIED_CONTROLLER_FREE_OPEN_LOOP_NUMERICAL_SHADOW",
    "feedback_recomputed": False, "new_qp_solved": False,
    "runtime_executor_used": False, "runtime_certificate_reused": False,
    "closed_loop_robustness_established": False,
    "wall_deployment_certified": False, "hardware_established": False,
    "continuous_time_collision_certified": False,
    "contact_response_enabled": False,
    "shadow_interval_cbf_recomputed": False,
    "shadow_calibrated_envelope_support_established": False,
    "uncertainty_scope": "whole robot body principal inertia only; mass/COM/armature fixed",
    "small_response_caveat": "fixed reflected armature may dominate small physical link inertias",
    "old_acceptance_modified": False,
}
# Raw/direct model inputs. Derived constants (e.g. dof_M0/body_invweight0) are
# intentionally allowed to change after mj_setConst, and are not mislabelled
# as an additional independent parameter perturbation.
DIRECT_ARRAYS = (
    "qpos0", "body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_mass", "body_inertia", "body_gravcomp", "body_mocapid",
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis",
    "jnt_range", "jnt_limited", "jnt_stiffness", "jnt_margin",
    "dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp",
    "geom_type", "geom_bodyid", "geom_dataid", "geom_pos", "geom_quat", "geom_size",
    "geom_contype", "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp",
    "geom_margin", "geom_gap", "geom_rgba", "mesh_vert", "mesh_face",
    "site_pos", "site_quat", "site_size", "site_type", "site_bodyid",
    "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gaintype",
    "actuator_biastype", "actuator_dyntype", "actuator_gainprm", "actuator_biasprm",
    "actuator_dynprm", "actuator_ctrllimited", "actuator_forcelimited",
    "actuator_ctrlrange", "actuator_forcerange", "actuator_actrange",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {"sha256_raw_c_order": hashlib.sha256(value.tobytes()).hexdigest(),
            "dtype": value.dtype.str, "shape": list(value.shape)}


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_trace(index):
    with np.load(BASELINE / "simulation/traces" / f"v6_lite_scenario_{index:02d}.npz",
                 allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def frozen_inputs():
    require(sha(BASELINE / "manifest.json") == BASELINE_MANIFEST_SHA,
            "historical complete manifest SHA mismatch")
    manifest = read(BASELINE / "manifest.json")
    require(len(manifest) == 46, "historical manifest must retain all 46 files")
    for name, expected in manifest.items():
        path = (BASELINE / name.replace("\\", "/")).resolve()
        require(path.is_relative_to(BASELINE.resolve()), "manifest escaped baseline")
        require(path.is_file() and sha(path) == expected, "historical file changed: " + name)
    require(read(BASELINE / "plan.json")["source"]["git_commit"] == BASELINE_SOURCE,
            "baseline producer source identity mismatch")
    metrics = read(BASELINE / "simulation/v6_lite_metrics.json")
    config = metrics["run_config"]
    require(config["duration_s"] == 27.0 and config["physics_period_s"] == DT
            and config["task_period_s"] == TASK_PERIOD
            and config["verification_subdivisions"] == SUBDIVISIONS
            and config["whole_body_minimum_clearance_m"] == GATE,
            "historical run scope changed")
    require(len(metrics["scenarios"]) == 5, "five fixed scenarios required")
    inputs = []
    for index, item in enumerate(metrics["scenarios"]):
        trace = load_trace(index)
        scenario = item["scenario"]
        require(scenario["scenario_id"] == f"v6_lite_scenario_{index:02d}", "scene order changed")
        require(hashlib.sha256(canonical(scenario)).hexdigest() == FIXED_SCENARIO_SHA[index],
                "fixed scenario definition changed")
        require(array_identity(trace["torque"])["sha256_raw_c_order"] == FIXED_TORQUE_SHA[index],
                "fixed raw torque bytes changed")
        require(trace["torque"].dtype == np.dtype("float64")
                and trace["torque"].shape == (STEPS, 67)
                and trace["initial_qpos"].shape == (81,)
                and trace["initial_qvel"].shape == (79,)
                and trace["task_qpos"].shape == (1351, 81), "input array layout changed")
        require(np.allclose(trace["time"], np.arange(1, STEPS + 1) * DT, rtol=0., atol=1e-9),
                "original torque clock is not the declared 2ms grid")
        require(all(np.all(np.isfinite(trace[k])) for k in
                    ("torque", "initial_qpos", "initial_qvel", "task_qpos")), "nonfinite input")
        inputs.append({"scenario": scenario, "scenario_sha256": FIXED_SCENARIO_SHA[index],
                       "trace_sha256": item["trace"]["sha256"],
                       "torque": array_identity(trace["torque"]),
                       "initial_qpos": array_identity(trace["initial_qpos"]),
                       "initial_qvel": array_identity(trace["initial_qvel"]),
                       "nominal_compiled_model_id": item["execution_contract"][
                           "source_compiled_model_sha256"],
                       "pair_policy_sha256": item["metrics"]["whole_body_clearance"][
                           "pair_policy_sha256"]})
    return metrics, inputs


def verifier_for(scenario):
    spec = default_v6_lite_robot_spec()
    require(spec.runtime_contract_sha256() == MODEL_CONTRACT_SHA
            and spec.source_bundle_sha256() == MODEL_BUNDLE_SHA,
            "current model/servo/asset contract does not equal the frozen nominal producer")
    obstacles = tuple(WorkspaceSphere(str(o["name"]), np.array(o["center_w"], dtype=float),
                                      float(o["radius_m"])) for o in scenario["workspace_obstacles"])
    verifier = WholeBodyCollisionVerifier(spec, obstacles, WholeBodyVerificationConfig(
        minimum_clearance=GATE, query_distance_max=2.5,
        adaptive_subdivisions=SUBDIVISIONS, self_collision_ancestor_exclusion_depth=3,
        include_target_satellite_pairs=True))
    verifier.model.geom_contype[:] = 0
    verifier.model.geom_conaffinity[:] = 0
    return spec, verifier


def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index))


def descendants(model, ancestor):
    result = []
    for body in range(1, model.nbody):
        current = body
        while current > 0 and current != ancestor:
            current = int(model.body_parentid[current])
        if current == ancestor:
            result.append(body)
    return np.array(result, dtype=int)


def selected_bodies(model, spec):
    base_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.base_joint_name)
    target_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.target_free_joint_name)
    require(base_joint >= 0 and target_joint >= 0, "declared free joints missing")
    root = int(model.jnt_bodyid[base_joint])
    target_root = int(model.jnt_bodyid[target_joint])
    ids = descendants(model, root)
    target_ids = descendants(model, target_root)
    require(name(model, mujoco.mjtObj.mjOBJ_BODY, root) == "base_of_satelltte",
            "unexpected robot free-body root")
    require(root > 0 and target_root not in ids and not set(ids).intersection(target_ids),
            "robot ancestry selection included world/target")
    selected = set(ids.tolist())
    # All 67 motor-driven joints must belong to this physical robot subtree.
    for joint in spec.low_level_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        require(jid >= 0 and int(model.jnt_bodyid[jid]) in selected,
                "robot subtree does not include every actuated joint")
    obstacle_geoms = [i for i in range(model.ngeom)
                      if (name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                          "v5_workspace_sphere_")]
    require(all(int(model.geom_bodyid[i]) == 0 for i in obstacle_geoms),
            "workspace obstacles unexpectedly entered a dynamic body subtree")
    records = [{"body_id": int(i), "body_name": name(model, mujoco.mjtObj.mjOBJ_BODY, i),
                "parent_id": int(model.body_parentid[i]),
                "principal_inertia_kg_m2": model.body_inertia[i].tolist(),
                "mass_kg": float(model.body_mass[i]), "COM_body_m": model.body_ipos[i].tolist()}
               for i in ids]
    return ids, {"root_joint": spec.base_joint_name, "root_joint_id": int(base_joint),
                 "root_body_id": root, "root_body_name": "base_of_satelltte",
                 "selection_rule": "jnt_bodyid root plus descendants by body_parentid ancestry",
                 "target_root_body_id_excluded": target_root, "world_body_id_excluded": 0,
                 "obstacle_geom_ids_on_world_excluded": obstacle_geoms, "bodies": records}


def direct_parameters(model):
    require(all(hasattr(model, key) for key in DIRECT_ARRAYS), "direct-parameter inventory unsupported")
    return {key: np.asarray(getattr(model, key)).copy() for key in DIRECT_ARRAYS}


def options(model):
    result = {}
    for key in dir(model.opt):
        if key.startswith("_"):
            continue
        value = getattr(model.opt, key)
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (int, float, np.number)):
            result[key] = value.item() if isinstance(value, np.number) else value
        elif hasattr(value, "__int__"):
            result[key] = int(value)  # MuJoCo's enum-valued solver/integrator fields
    return result


def setup_model(index, alpha, inputs, run_dir=None):
    trace = load_trace(index)
    spec, verifier = verifier_for(inputs[index]["scenario"])
    model = verifier.model
    nominal_id = model_id(model, MODEL_CONTRACT_SHA)
    require(nominal_id == inputs[index]["nominal_compiled_model_id"],
            "nominal compiled model does not equal producer model")
    require(verifier._pair_policy_sha256 == inputs[index]["pair_policy_sha256"],
            "whole-body pair policy changed")
    require(np.all(np.abs(trace["torque"]) <= spec.torque_limits[None, :] + 1e-12),
            "frozen torque exceeds original unchanged actuator limits")
    ids, selection = selected_bodies(model, spec)
    before = direct_parameters(model)
    old_options = options(model)
    expected = before["body_inertia"].copy()
    expected[ids] *= alpha
    model.body_inertia[:] = expected
    data = mujoco.MjData(model)
    # Recompute derived dynamics constants for the modified inertia. Restoring
    # the initial state afterwards avoids treating mj_setConst workspace as plant state.
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = trace["initial_qpos"]
    data.qvel[:] = trace["initial_qvel"]
    data.ctrl[:] = 0.
    data.time = 0.
    mujoco.mj_forward(model, data)
    after = direct_parameters(model)
    require(options(model) == old_options, "an integration/dynamics option changed")
    unchanged = {key: bool(np.array_equal(before[key], after[key]))
                 for key in DIRECT_ARRAYS if key != "body_inertia"}
    require(all(unchanged.values()), "direct parameters changed outside principal inertia")
    require(np.array_equal(after["body_inertia"], expected), "requested principal inertia not realized")
    effective_id = model_id(model, MODEL_CONTRACT_SHA)
    require((effective_id == nominal_id) == (alpha == 1.), "effective model identity did not reflect alpha")
    metadata = {"alpha": alpha, "selection": selection,
                "nominal_model_id": nominal_id, "effective_model_id": effective_id,
                "same_geometry_and_actuator_contract": all(unchanged.values()),
                "unchanged_direct_parameters": unchanged, "options_before_after": old_options,
                "before": {k: array_identity(v) for k, v in before.items()},
                "after": {k: array_identity(v) for k, v in after.items()},
                "expected_principal_inertia": expected.tolist(),
                "derived_constants_refreshed_by": "mj_setConst, mj_resetData, initial qpos/qvel restore, mj_forward",
                "runtime_certificates_applicable": False}
    if run_dir is not None:
        run_dir.mkdir(exist_ok=False)
        np.savez_compressed(run_dir / "model_parameters.npz",
                            **{"before_" + k: v for k, v in before.items()},
                            **{"after_" + k: v for k, v in after.items()})
        write(run_dir / "model.json", metadata)  # before any physics or distance query
    return spec, verifier, data, metadata


def run_key(index, alpha):
    return f"scene_{index:02d}_alpha_{alpha:.2f}"


def identity_check(output, plan):
    metrics, inputs = frozen_inputs()
    require(inputs == plan["inputs"], "frozen plan input identity changed")
    require(sha(Path(__file__)) == plan["tool_sha256"], "producer changed after planning")
    require(sha(output / "producer.py") == plan["tool_sha256"], "saved producer changed")
    spec = default_v6_lite_robot_spec()  # no compile/physics: rehash assets at every terminal boundary
    require(spec.runtime_contract_sha256() == MODEL_CONTRACT_SHA
            and spec.source_bundle_sha256() == MODEL_BUNDLE_SHA,
            "model/servo/source asset identity changed after planning")
    current = current_source_snapshot(ROOT)
    provenance = source_provenance(plan["current_source"], current)
    require(provenance["source_unchanged"] and current["tracked_worktree_dirty"] is False
            and plan["current_source"]["tracked_worktree_dirty"] is False,
            "current source/HEAD changed or tracked worktree is dirty")
    return metrics, inputs, provenance


def create_plan(output, smoke_steps=None):
    output.mkdir(parents=True, exist_ok=False)
    metrics, inputs = frozen_inputs()
    source = current_source_snapshot(ROOT)
    require(not source["capture_errors"] and source["tracked_worktree_dirty"] is False,
            "source inventory incomplete or tracked worktree dirty; freeze committed core first")
    shutil.copyfile(__file__, output / "producer.py")
    plan = {"schema": "inertia_shadow_plan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": LIMITATIONS, "baseline_source_commit": BASELINE_SOURCE,
            "baseline_complete_manifest_sha256": BASELINE_MANIFEST_SHA,
            "baseline_path": str(BASELINE), "current_source": source,
            "tool_sha256": sha(__file__), "python": sys.version, "platform": platform.platform(),
            "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
            "inputs": inputs, "alphas": list(ALPHAS), "duration_s_per_run": 27.,
            "physics_period_s": DT, "original_command_period_s": TASK_PERIOD,
            "full_experiment_physics_steps": 202500, "full_experiment_run_count": 15,
            "input_feedback_note": "original nominal feedback-produced torques are now frozen open-loop inputs",
            "nominal_replay_tolerance": {"original_delivery_position_and_rotation": 1e-11,
                                         "physics_time_grid_s": 1e-9},
            "geometry_scopes": {"robot_target": "all 13501 actual 2ms states, all original target pair classes",
                                "whole_body": "all original pairs at 1351 20ms states, four native configuration subdivisions; 5401 states",
                                "minimum_clearance_m": GATE, "continuous_time_certified": False,
                                "target_distance_max_m": .25, "whole_body_distance_max_m": 2.5,
                                "censoring": "distmax/DBL_MAX returns are lower bounds, not exact distances"},
            "smoke_steps": smoke_steps, "execution_status": "NOT_RUN",
            "full_scope_complete": False, "runs": [
                {"key": run_key(i, a), "scenario_index": i, "alpha": a, "status": "NOT_RUN"}
                for a in ALPHAS for i in range(5)]}
    # Plan and input/source identities exist before model initialization,
    # physics, or signed distance queries. Per-model selection precedes steps.
    write(output / "plan.json", plan)
    write(output / "input_identities.json", inputs)
    return plan


def nominal_parity(spec, model, trajectory, trace, legacy, steps):
    qids, dids = joint_addresses(model, spec)
    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.base_joint_name)
    base_qadr = int(model.jnt_qposadr[base_jid])
    residuals = {
        "planner_q": float(np.max(np.abs(
            np.array([spec.decode_position(q[qids]) for q in trajectory["qpos"][1:]])
            - trace["planner_q"][:steps]))),
        "planner_dq": float(np.max(np.abs(
            np.array([spec.decode_velocity(v[dids]) for v in trajectory["qvel"][1:]])
            - trace["planner_dq"][:steps]))),
        "base_qpos": float(np.max(np.abs(trajectory["qpos"][1:, base_qadr:base_qadr + 7]
                                               - trace["base_qpos"][:steps]))),
        "task_qpos": float(np.max(np.abs(trajectory["qpos"][::10]
                                             - trace["task_qpos"][:steps // 10 + 1]))),
    }
    residuals.update({k: float(np.max(np.abs(np.array(v) - trace[k][:steps])))
                      for k, v in legacy.items()})
    return {"passed": all(value <= 1e-11 for value in residuals.values()),
            "maximum_absolute_residuals": residuals, "tolerance": 1e-11,
            "legacy_kinematics_note": "matches original post-mj_step cached logging; fresh geometry audited separately",
            "full_27s_nominal_reproduction": steps == STEPS}


def replay_one(output, plan, index, alpha, steps):
    _, inputs, _ = identity_check(output, plan)
    trace = load_trace(index)
    run_dir = output / "replays" / run_key(index, alpha)
    spec, verifier, data, model_meta = setup_model(index, alpha, inputs, run_dir)
    model = verifier.model
    trajectory = {"qpos": np.empty((steps + 1, model.nq)),
                  "qvel": np.empty((steps + 1, model.nv)), "time": np.empty(steps + 1)}
    trajectory["qpos"][0], trajectory["qvel"][0], trajectory["time"][0] = data.qpos, data.qvel, data.time
    initial_state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    body_ids = {key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                for key, body in (("rigid", spec.rigid_tip_body_name), ("continuum", spec.continuum_tip_body_name))}
    legacy = {k: [] for k in ("rigid_tip", "rigid_rotation", "continuum_tip", "continuum_tip_body_origin", "continuum_rotation")}
    finished = 0
    error = None
    started = time.perf_counter()
    try:
        for step in range(steps):
            data.ctrl[:] = trace["torque"][step]  # NO feedback, QP, or runtime dispatch
            mujoco.mj_step(model, data)
            finished = step + 1
            trajectory["qpos"][finished], trajectory["qvel"][finished] = data.qpos, data.qvel
            trajectory["time"][finished] = data.time
            require(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)), "nonfinite plant state")
            require(abs(float(data.time) - finished * DT) <= 1e-9, "plant left declared 2ms grid")
            if alpha == 1.:
                rigid, continuum = body_ids["rigid"], body_ids["continuum"]
                cr = data.xmat[continuum].reshape(3, 3).copy()
                cp = data.xpos[continuum].copy()
                legacy["rigid_tip"].append(data.xpos[rigid].copy())
                legacy["rigid_rotation"].append(data.xmat[rigid].reshape(3, 3).copy())
                legacy["continuum_tip_body_origin"].append(cp)
                legacy["continuum_rotation"].append(cr)
                legacy["continuum_tip"].append(cp + cr @ EE_OFFSET)
        require(model_id(model, MODEL_CONTRACT_SHA) == model_meta["effective_model_id"], "model changed during replay")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        saved = {k: v[:finished + 1].copy() for k, v in trajectory.items()}
        saved.update(torque_input=trace["torque"][:finished].copy(), initial_integration_state=initial_state,
                     scenario_index=np.array(index), alpha=np.array(alpha), physics_period_s=np.array(DT))
        np.savez_compressed(run_dir / "trajectory.npz", **saved)
    parity = nominal_parity(spec, model, saved, trace, legacy, finished) if alpha == 1. and error is None else None
    if parity is not None and not parity["passed"]:
        error = {"type": "NOMINAL_REPLAY_MISMATCH", "message": "alpha=1 failed original-delivery reproduction tolerance"}
    provenance = source_provenance(plan["current_source"], current_source_snapshot(ROOT))
    stable = provenance["source_unchanged"]
    try:
        identity_check(output, plan)
    except Exception as exc:
        error = error or {"type": "IDENTITY_CHANGED", "message": str(exc)}
    result = {"key": run_key(index, alpha), "alpha": alpha, "scenario_index": index,
              "status": "REPLAY_COMPLETED" if error is None and stable else "REPLAY_FAILED",
              "steps_completed": finished, "planned_steps": steps, "error": error,
              "full_27s_complete": finished == STEPS and error is None,
              "elapsed_seconds": time.perf_counter() - started,
              "trajectory_sha256": sha(run_dir / "trajectory.npz"),
              "torque_input_identity": array_identity(saved["torque_input"]),
              "original_full_torque_identity": inputs[index]["torque"], "nominal_parity": parity,
              "model_parameters_sha256": sha(run_dir / "model_parameters.npz"),
              "model_metadata_sha256": sha(run_dir / "model.json"),
              "source_provenance": provenance, "limitations": LIMITATIONS}
    write(run_dir / "replay.json", result)
    return result


def replay_stage(output, plan):
    identity_check(output, plan)
    (output / "replays").mkdir(exist_ok=False)
    rows = []
    smoke = plan["smoke_steps"]
    active_key = None
    try:
        for alpha in ALPHAS:
            for index in range(5):
                if smoke is not None and (index != 0 or alpha != 1.):
                    rows.append({"key": run_key(index, alpha), "status": "NOT_RUN_SMOKE_SCOPE"})
                    continue
                active_key = run_key(index, alpha)
                result = replay_one(output, plan, index, alpha, smoke or STEPS)
                rows.append(result)
                require(result["status"] == "REPLAY_COMPLETED", "replay failed; no further runs started")
                active_key = None
    except BaseException:
        existing = {r["key"] for r in rows}
        if active_key is not None and active_key not in existing:
            rows.append({"key": active_key, "status": "FAILED_DURING_STARTED_REPLAY_PHASE",
                         "partial_artifacts_retained": str(output / "replays" / active_key)})
            existing.add(active_key)
        rows.extend({"key": r["key"], "status": "NOT_STARTED_PREVIOUS_REPLAY_OR_IDENTITY_FAILURE"}
                    for r in plan["runs"] if r["key"] not in existing)
        write(output / "replay_report.json", {"complete": False, "evidence_valid": False,
              "status": "FAILED", "runs": rows, "limitations": LIMITATIONS})
        raise
    complete = smoke is None and len(rows) == 15 and all(r.get("full_27s_complete") for r in rows)
    report = {"schema": "inertia_shadow_replay_v1", "complete": complete,
              "evidence_valid": False, "status": "REPLAY_ONLY_GEOMETRY_NOT_RUN" if complete else "SMOKE_ONLY",
              "geometry_status": "NOT_RUN", "runs": rows, "limitations": LIMITATIONS,
              "physics_steps_completed": sum(r.get("steps_completed", 0) for r in rows)}
    write(output / "replay_report.json", report)
    replay_manifest = {p.relative_to(output).as_posix(): sha(p)
                       for p in sorted((output / "replays").rglob("*")) if p.is_file()}
    replay_manifest.update({name: sha(output / name) for name in
                            ("plan.json", "producer.py", "input_identities.json", "replay_report.json")})
    write(output / "replay_manifest.json", replay_manifest)
    report["replay_manifest_sha256"] = sha(output / "replay_manifest.json")
    return report


def audit_pairs(model, pairs, states, count, distmax, output_path, scope):
    """Native signed queries of current qpos. No capsule surrogate depths."""
    require(bool(pairs), "empty geometric pair policy")
    require(isinstance(count, (int, np.integer)) and count > 0,
            "geometry requires a positive planned state count")
    observer = mujoco.MjData(model)
    fromto = np.empty(6)
    minima = np.empty(count)
    censored_minimum = np.empty(count, dtype=bool)
    worst_pair = np.empty(count, dtype=int)
    phases = np.empty(count)
    class_minimum = {}
    class_minimum_censored = {}
    truncated = violations = penetrating = below_states = 0
    penetrating_states = 0
    first_below = None
    best = float("inf")
    best_pair = None
    best_censored = None
    visited = 0
    query_attempts = completed_queries = 0
    partial_distances, partial_censored = [], []
    active_state = active_pair = None
    active_qpos, active_qvel = np.empty(0), np.empty(0)
    active_phase = None
    failure = None
    try:
        for i, (qpos, qvel, phase) in enumerate(states):
            active_state, active_pair, active_phase = i, None, float(phase)
            active_qpos, active_qvel = np.asarray(qpos).copy(), np.asarray(qvel).copy()
            partial_distances, partial_censored = [], []
            require(i < count, "geometry supplied more states than planned")
            require(active_qpos.shape == (model.nq,) and active_qvel.shape == (model.nv,)
                    and np.all(np.isfinite(active_qpos)) and np.all(np.isfinite(active_qvel))
                    and np.isfinite(active_phase), "invalid geometry state or clock")
            observer.qpos[:] = qpos
            observer.qvel[:] = qvel
            observer.time = phase * DT
            observer.ctrl[:] = 0.
            mujoco.mj_forward(model, observer)  # fresh observer; NEVER replay MjData
            minimum = float("inf")
            minimum_censored = False
            minimum_pair = -1
            for j, pair in enumerate(pairs):
                active_pair = j
                query_attempts += 1
                raw = float(mujoco.mj_geomDistance(model, observer, int(pair.geom_a),
                                                  int(pair.geom_b), distmax, fromto))
                require(np.isfinite(raw), "nonfinite native signed distance")
                completed_queries += 1
                censored = raw >= distmax - 1e-12
                distance = distmax if censored else raw
                partial_distances.append(distance)
                partial_censored.append(censored)
                truncated += int(censored)
                violations += int(distance < GATE)
                penetrating += int(distance < 0.)
                old_class_minimum = class_minimum.get(pair.pair_class, float("inf"))
                if distance < old_class_minimum or (distance == old_class_minimum
                        and class_minimum_censored.get(pair.pair_class, True) and not censored):
                    class_minimum[pair.pair_class] = distance
                    class_minimum_censored[pair.pair_class] = bool(censored)
                if distance < minimum or (distance == minimum and minimum_censored and not censored):
                    minimum, minimum_censored, minimum_pair = distance, censored, j
                if distance < best or (distance == best and best_censored and not censored):
                    best, best_censored = distance, censored
                    best_pair = {"pair_class": pair.pair_class, "geom_a": pair.geom_a_name,
                                 "geom_b": pair.geom_b_name, "state_index": i,
                                 "physics_grid_phase": float(phase), "time_s": float(phase * DT),
                                 "native_signed_distance_or_lower_bound_m": distance,
                                 "censored_lower_bound": bool(censored),
                                 "witness_fromto_m": None if censored else fromto.tolist()}
            minima[i], censored_minimum[i], worst_pair[i], phases[i] = minimum, minimum_censored, minimum_pair, phase
            below_states += int(minimum < GATE)
            penetrating_states += int(minimum < 0.)
            if minimum < GATE and first_below is None:
                first_below = {"state_index": i, "physics_grid_phase": float(phase),
                               "time_s": float(phase * DT), "minimum_m": minimum,
                               "pair_index": minimum_pair}
            visited += 1
            partial_distances, partial_censored = [], []
            active_state = active_pair = active_phase = None
            active_qpos, active_qvel = np.empty(0), np.empty(0)
        require(visited == count and completed_queries == count * len(pairs),
                "geometry did not cover every planned state and pair")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Save every completed state's minimum prefix and all successful pair
        # queries of the interrupted current state. Never label it complete.
        np.savez_compressed(output_path, minimum_m=minima[:visited],
                            minimum_censored_lower_bound=censored_minimum[:visited],
                            minimum_pair_index=worst_pair[:visited], physics_grid_phase=phases[:visited],
                            planned_state_count=np.array(count), completed_state_count=np.array(visited),
                            completed_query_count=np.array(completed_queries),
                            complete=np.array(failure is None and visited == count),
                            partial_state_qpos=active_qpos, partial_state_qvel=active_qvel,
                            partial_state_pair_distance_m=np.asarray(partial_distances),
                            partial_state_pair_censored_lower_bound=np.asarray(partial_censored, dtype=bool))
        if failure is not None:
            failed_pair = pairs[active_pair] if active_pair is not None else None
            write(output_path.with_suffix(".partial.json"), {
                "status": "PARTIAL_OBSERVATION_FAILED", "complete": False, "evidence_valid": False,
                "failure": failure, "completed_state_count": visited,
                "completed_query_count": completed_queries, "attempted_query_count": query_attempts,
                "active_state_index": active_state,
                "active_physics_grid_phase": active_phase if active_phase is not None and np.isfinite(active_phase) else None,
                "active_physics_grid_phase_repr": repr(active_phase),
                "active_pair_index": active_pair,
                "active_pair_names": None if failed_pair is None else [failed_pair.geom_a_name, failed_pair.geom_b_name],
                "partial_state_pair_order": "first N pairs in observer_plan policy order",
                "minimum_over_successful_queries_m": best if np.isfinite(best) else None,
                "minimum_pair_over_successful_queries": best_pair,
                "minimum_by_class_m": class_minimum,
                "minimum_by_class_is_censored_lower_bound": class_minimum_censored,
                "truncated_query_count": truncated, "arrays_sha256": sha(output_path)})
    return {"time_scope": scope, "checked_state_count": visited, "pair_count": len(pairs),
            "query_count": visited * len(pairs), "distance_max_m": distmax,
            "truncated_query_count": truncated, "minimum_clearance_m": best,
            "minimum_is_censored_lower_bound": bool(best_censored), "minimum_pair": best_pair,
            "minimum_by_class_m": class_minimum, "below_required_clearance_state_count": below_states,
            "minimum_by_class_is_censored_lower_bound": class_minimum_censored,
            "first_below_required_clearance": first_below,
            "negative_native_signed_state_count": penetrating_states,
            "violating_pair_sample_count": violations, "negative_native_signed_query_count": penetrating,
            "measured_discrete_clearance_at_least_gate": best >= GATE,
            "negative_distance_scope": "native mj_geomDistance signed query; not capsule-box surrogate depth",
            "continuous_time_certified": False, "nativeccd_claimed": False,
            "arrays_sha256": sha(output_path)}


def dense_states(model, qpos):
    delta = np.empty(model.nv)
    for i in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, delta, 1., qpos[i], qpos[i + 1])
        for k in range(SUBDIVISIONS):
            fraction = k / SUBDIVISIONS
            q = qpos[i].copy()
            mujoco.mj_integratePos(model, q, delta, fraction)
            yield q, np.zeros(model.nv), (i + fraction) * 10.
    yield qpos[-1].copy(), np.zeros(model.nv), (len(qpos) - 1) * 10.


def rotation_angles(a, b):
    return np.arccos(np.clip((np.einsum("...ij,...ij->...", a, b) - 1.) / 2., -1., 1.))


def statistics(value):
    value = np.asarray(value)
    require(value.size > 0 and np.all(np.isfinite(value)), "nonfinite/empty summary")
    return {"count": int(value.size), "mean": float(np.mean(value)),
            "rmse": float(np.sqrt(np.mean(value ** 2))),
            "p95": float(np.percentile(value, 95.)), "max": float(np.max(value)),
            "min": float(np.min(value))}


def fresh_kinematics(spec, model, qpos, qvel, times, scenario, trace, path):
    observer = mujoco.MjData(model)
    rigid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.rigid_tip_body_name)
    continuum = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.continuum_tip_body_name)
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite")
    require(min(rigid, continuum, target) >= 0, "kinematic bodies missing")
    output = {key: [] for key in ("rigid_tip", "continuum_tip", "rigid_rotation", "continuum_rotation",
                                 "rigid_target", "rigid_target_rotation")}
    grasp_p = np.array(scenario["grasp_point_target_frame_m"])
    grasp_R = np.array(scenario["grasp_rotation_target_frame"])
    for q, v, t in zip(qpos, qvel, times):
        observer.qpos[:], observer.qvel[:], observer.time = q, v, float(t)
        mujoco.mj_forward(model, observer)
        cr = observer.xmat[continuum].reshape(3, 3).copy()
        tr = observer.xmat[target].reshape(3, 3).copy()
        output["rigid_tip"].append(observer.xpos[rigid].copy())
        output["continuum_tip"].append(observer.xpos[continuum].copy() + cr @ EE_OFFSET)
        output["rigid_rotation"].append(observer.xmat[rigid].reshape(3, 3).copy())
        output["continuum_rotation"].append(cr)
        output["rigid_target"].append(observer.xpos[target].copy() + tr @ grasp_p)
        output["rigid_target_rotation"].append(tr @ grasp_R)
    output = {k: np.asarray(v) for k, v in output.items()}
    np.savez_compressed(path, time=times, **output)
    errors = {"rigid_position_error_m": statistics(np.linalg.norm(
                  output["rigid_tip"][1:] - output["rigid_target"][1:], axis=1)),
              "continuum_position_error_m": statistics(np.linalg.norm(
                  output["continuum_tip"][1:] - trace["continuum_target"], axis=1)),
              "rigid_orientation_error_deg": statistics(np.rad2deg(rotation_angles(
                  output["rigid_rotation"][1:], output["rigid_target_rotation"][1:]))),
              "continuum_orientation_error_deg": statistics(np.rad2deg(rotation_angles(
                  output["continuum_rotation"][1:], trace["continuum_target_rotation"])))}
    measured_time = times[1:]
    path_definition = scenario["continuum_target"]
    path_mask = ((measured_time >= float(path_definition["path_start_s"]))
                 & (measured_time <= float(path_definition["path_end_s"])))
    steady_mask = measured_time >= 27. - 1.5
    phase_errors = {
        "continuum_original_path_window_error_m": statistics(np.linalg.norm(
            output["continuum_tip"][1:][path_mask] - trace["continuum_target"][path_mask], axis=1)),
        "rigid_final_position_error_m": float(np.linalg.norm(
            output["rigid_tip"][-1] - output["rigid_target"][-1])),
        "rigid_original_last_1_5s_position_error_m": statistics(np.linalg.norm(
            output["rigid_tip"][1:][steady_mask] - output["rigid_target"][1:][steady_mask], axis=1))}
    return {"arrays_sha256": sha(path), "tracking_statistics": errors,
            "original_phase_window_diagnostic_statistics": phase_errors,
            "clock_scope": "fresh post-integration current qpos; initial excluded from target-error metrics",
            "reference_scope": "rigid grasp from unchanged current target pose; continuum from frozen time-aligned references",
            "original_acceptance_relabelled": False}


def observe_one(output, plan, replay_row):
    _, inputs, _ = identity_check(output, plan)
    index, alpha = replay_row["scenario_index"], replay_row["alpha"]
    key = run_key(index, alpha)
    run_dir = output / "replays" / key
    require(sha(run_dir / "trajectory.npz") == replay_row["trajectory_sha256"], "replay artifact changed")
    require(sha(run_dir / "model.json") == replay_row["model_metadata_sha256"]
            and sha(run_dir / "model_parameters.npz") == replay_row["model_parameters_sha256"],
            "replay model record changed")
    with np.load(run_dir / "trajectory.npz", allow_pickle=False) as archive:
        trajectory = {k: archive[k].copy() for k in archive.files}
    require(trajectory["qpos"].shape == (STEPS + 1, 81)
            and trajectory["qvel"].shape == (STEPS + 1, 79), "complete trajectory missing")
    require(array_identity(trajectory["torque_input"]) == inputs[index]["torque"], "shadow torque differs")
    require(np.allclose(trajectory["time"], np.arange(STEPS + 1) * DT, rtol=0., atol=1e-9), "trajectory grid mismatch")
    spec, verifier = verifier_for(inputs[index]["scenario"])
    model = verifier.model  # geometry identical; this observer does not integrate perturbed dynamics
    require(model_id(model, MODEL_CONTRACT_SHA) == inputs[index]["nominal_compiled_model_id"]
            and verifier._pair_policy_sha256 == inputs[index]["pair_policy_sha256"], "geometry observer contract mismatch")
    obs_dir = output / "observations" / key
    obs_dir.mkdir(exist_ok=False)
    pairs = [{"pair_class": p.pair_class, "geom_a": p.geom_a_name, "geom_b": p.geom_b_name,
              "geom_a_id": int(p.geom_a), "geom_b_id": int(p.geom_b)} for p in verifier.pairs]
    target_pairs = tuple(p for p in verifier.pairs if p.pair_class.endswith("_target"))
    write(obs_dir / "observer_plan.json", {"input_trajectory_sha256": replay_row["trajectory_sha256"],
          "observer_model_id": model_id(model, MODEL_CONTRACT_SHA), "pair_policy_sha256": verifier._pair_policy_sha256,
          "pairs": pairs, "target_pair_count": len(target_pairs),
          "target_physics_state_count": STEPS + 1, "target_query_count": (STEPS + 1) * len(target_pairs),
          "whole_body_pair_count": len(verifier.pairs), "whole_body_dense_state_count": 5401,
          "whole_body_query_count": 5401 * len(verifier.pairs),
          "source": current_source_snapshot(ROOT), "limitations": LIMITATIONS})
    # All original robot-target policy classes are checked at every 2ms state,
    # including base_target and both arm target classes. Original terminal
    # grasp exemptions are retained; they are never silently reinstated/removed.
    target_states = ((q, v, float(i)) for i, (q, v) in enumerate(zip(trajectory["qpos"], trajectory["qvel"])))
    target_report = audit_pairs(model, target_pairs, target_states, STEPS + 1, .25,
                                obs_dir / "robot_target_500hz.npz", "all_actual_2ms_states_including_initial")
    task_qpos = trajectory["qpos"][::10]
    whole_body = audit_pairs(model, verifier.pairs, dense_states(model, task_qpos), 5401, 2.5,
                             obs_dir / "whole_body_dense.npz", "original_20ms_samples_with_four_configuration_subdivisions")
    kinematics = fresh_kinematics(spec, model, trajectory["qpos"], trajectory["qvel"], trajectory["time"],
                                  inputs[index]["scenario"], load_trace(index), obs_dir / "fresh_kinematics.npz")
    _, _, provenance = identity_check(output, plan)
    report = {"key": key, "scenario_index": index, "alpha": alpha,
              "status": "OBSERVATION_COMPLETED", "robot_target_500hz": target_report,
              "whole_body_dense_discrete": whole_body, "fresh_kinematics": kinematics,
              "pair_policy_sha256": verifier._pair_policy_sha256, "source_provenance": provenance,
              "limitations": LIMITATIONS}
    write(obs_dir / "observation.json", report)
    return report


def paired_deltas(output, inputs):
    result = []
    for index in range(5):
        nominal_dir = output / "observations" / run_key(index, 1.)
        with np.load(nominal_dir / "fresh_kinematics.npz") as a:
            nominal_k = {k: a[k].copy() for k in a.files}
        with np.load(output / "replays" / run_key(index, 1.) / "trajectory.npz") as a:
            nominal_q = a["qpos"].copy()
        with np.load(nominal_dir / "robot_target_500hz.npz") as a:
            nominal_d, nominal_c = a["minimum_m"].copy(), a["minimum_censored_lower_bound"].copy()
        spec, verifier = verifier_for(inputs[index]["scenario"])
        qids, _ = joint_addresses(verifier.model, spec)
        base_joint = mujoco.mj_name2id(verifier.model, mujoco.mjtObj.mjOBJ_JOINT, spec.base_joint_name)
        adr = int(verifier.model.jnt_qposadr[base_joint])
        for alpha in (.95, 1.05):
            obs = output / "observations" / run_key(index, alpha)
            with np.load(obs / "fresh_kinematics.npz") as a:
                k = {key: a[key].copy() for key in a.files}
            with np.load(output / "replays" / run_key(index, alpha) / "trajectory.npz") as a:
                q = a["qpos"].copy()
            with np.load(obs / "robot_target_500hz.npz") as a:
                d, c = a["minimum_m"].copy(), a["minimum_censored_lower_bound"].copy()
            exact = ~(c | nominal_c)
            quat, ref_quat = q[:, adr + 3:adr + 7], nominal_q[:, adr + 3:adr + 7]
            dots = np.abs(np.sum(quat * ref_quat, axis=1) /
                          (np.linalg.norm(quat, axis=1) * np.linalg.norm(ref_quat, axis=1)))
            result.append({"scenario_index": index, "alpha": alpha,
                "low_level_joint_delta_rad": statistics(np.abs(q[:, qids] - nominal_q[:, qids])),
                "base_translation_delta_m": statistics(np.linalg.norm(q[:, adr:adr + 3] - nominal_q[:, adr:adr + 3], axis=1)),
                "base_orientation_delta_deg": statistics(np.rad2deg(2 * np.arccos(np.clip(dots, 0., 1.)))),
                "rigid_tip_delta_m": statistics(np.linalg.norm(k["rigid_tip"] - nominal_k["rigid_tip"], axis=1)),
                "continuum_tip_delta_m": statistics(np.linalg.norm(k["continuum_tip"] - nominal_k["continuum_tip"], axis=1)),
                "rigid_orientation_delta_deg": statistics(np.rad2deg(rotation_angles(k["rigid_rotation"], nominal_k["rigid_rotation"]))),
                "continuum_orientation_delta_deg": statistics(np.rad2deg(rotation_angles(k["continuum_rotation"], nominal_k["continuum_rotation"]))),
                "paired_exact_target_clearance_delta_m": statistics(d[exact] - nominal_d[exact]) if np.any(exact) else None,
                "paired_exact_clearance_count": int(np.sum(exact)),
                "paired_censored_clearance_count_excluded_from_exact_delta": int(np.sum(~exact))})
    return result


def observe_stage(output, plan, replay_manifest_sha):
    _, inputs, _ = identity_check(output, plan)
    require(replay_manifest_sha and sha(output / "replay_manifest.json") == replay_manifest_sha,
            "replay qualification manifest does not equal caller-pinned SHA")
    replay_manifest = read(output / "replay_manifest.json")
    for filename, expected in replay_manifest.items():
        path = (output / filename).resolve()
        require(path.is_relative_to(output), "replay manifest path escaped experiment output")
        require(path.is_file() and sha(path) == expected, "frozen replay qualification file changed: " + filename)
    required_manifest_files = {"plan.json", "producer.py", "input_identities.json", "replay_report.json"}
    required_manifest_files.update("replays/" + row["key"] + "/" + filename
                                   for row in plan["runs"] for filename in
                                   ("model_parameters.npz", "model.json", "trajectory.npz", "replay.json"))
    require(set(replay_manifest) == required_manifest_files, "replay qualification manifest coverage changed")
    replay = read(output / "replay_report.json")
    require(replay["complete"] and plan["smoke_steps"] is None, "partial/smoke physics cannot qualify full geometry")
    require(replay["physics_steps_completed"] == 202500, "full physics count mismatch")
    expected_runs = [(i, a, run_key(i, a)) for a in ALPHAS for i in range(5)]
    actual_runs = [(r.get("scenario_index"), r.get("alpha"), r.get("key")) for r in replay["runs"]]
    require(actual_runs == expected_runs, "replay qualification must contain each of the 15 fixed runs exactly once")
    require(all(r.get("nominal_parity", {}).get("passed")
                for r in replay["runs"] if r["alpha"] == 1.), "nominal five-scene qualification missing")
    (output / "observations").mkdir(exist_ok=False)
    rows = []
    active_key = None
    try:
        for row in replay["runs"]:
            require(row["status"] == "REPLAY_COMPLETED" and row["full_27s_complete"], "incomplete replay")
            active_key = row["key"]
            rows.append(observe_one(output, plan, row))
            active_key = None
        deltas = paired_deltas(output, inputs)
        _, _, provenance = identity_check(output, plan)
        report = {"schema": "inertia_shadow_observation_v1", "complete": len(rows) == 15,
                  "evidence_valid": len(rows) == 15 and provenance["source_unchanged"],
                  "status": "DIAGNOSTIC_COMPLETE", "runs": rows, "paired_vs_nominal": deltas,
                  "physics_steps_completed": 202500, "source_provenance": provenance,
                  "caller_pinned_replay_manifest_sha256": replay_manifest_sha,
                  "limits_and_guard_status": "original runtime guards unchanged; shadow never submitted to them",
                  "limitations": LIMITATIONS}
        manifest = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"}
        report_raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        manifest["report.json"] = hashlib.sha256(report_raw).hexdigest()
        write(output / "manifest.json", manifest)
        write(output / "report.json", report)  # positive result is the final terminal artifact
        return report
    except BaseException:
        write(output / "report.json", {"complete": False, "evidence_valid": False,
              "status": "OBSERVATION_FAILED", "observations_completed": rows,
              "failed_started_observation": active_key,
              "partial_observation_artifacts_retained": None if active_key is None else str(output / "observations" / active_key),
              "unstarted_runs": [r["key"] for r in plan["runs"]
                                 if r["key"] != active_key and r["key"] not in {x["key"] for x in rows}],
              "limitations": LIMITATIONS})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("plan", "replay", "observe", "all", "smoke"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument("--replay-manifest-sha", help="required external pin for a separate observe stage")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT / "v6_lite/output/runs").resolve()
    require(output.is_relative_to(allowed) and output != allowed
            and not output.is_relative_to(BASELINE.resolve())
            and not Path(__file__).resolve().is_relative_to(output),
            "output must be a separate new directory inside ignored output/runs")
    require(1 <= args.smoke_steps <= 10, "smoke has only 1..10 explicitly incomplete steps")
    can_write_failure = False
    try:
        if args.stage in ("plan", "all", "smoke"):
            # Never write a failure marker into an existing unrelated output.
            require(not output.exists(), "new plan/smoke/all output already exists")
            can_write_failure = True
            plan = create_plan(output, args.smoke_steps if args.stage == "smoke" else None)
        else:
            plan = read(output / "plan.json")
            require(plan.get("schema") == "inertia_shadow_plan_v1"
                    and plan.get("baseline_complete_manifest_sha256") == BASELINE_MANIFEST_SHA
                    and plan.get("tool_sha256") == sha(__file__),
                    "existing output is not this frozen inertia shadow experiment")
            closed_markers = ("report.json", "manifest.json", "failure.json")
            require(not any((output / filename).exists() for filename in closed_markers),
                    "terminal experiment output is read-only; select a new experiment")
            phase_markers = (("replays", "replay_report.json", "replay_manifest.json")
                             if args.stage == "replay" else ("observations",))
            require(not any((output / filename).exists() for filename in phase_markers),
                    "this stage already started/completed; existing artifacts remain read-only")
            can_write_failure = True
        if args.stage in ("replay", "all", "smoke"):
            result = replay_stage(output, plan)
        elif args.stage == "plan":
            result = {"status": "NOT_RUN", "complete": False, "evidence_valid": False}
        if args.stage in ("observe", "all"):
            expected_replay_sha = (result["replay_manifest_sha256"] if args.stage == "all"
                                   else args.replay_manifest_sha)
            result = observe_stage(output, plan, expected_replay_sha)
        print(json.dumps({"status": result["status"], "complete": result["complete"],
                          "evidence_valid": result.get("evidence_valid", False), "output_dir": str(output),
                          "replay_manifest_sha256": result.get("replay_manifest_sha256")}))
        return 0
    except BaseException as exc:
        if can_write_failure and output.is_dir() and not (output / "failure.json").exists():
            write(output / "failure.json", {"status": "FAILED", "complete": False, "evidence_valid": False,
                  "stage": args.stage, "type": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc(), "limitations": LIMITATIONS})
        print(json.dumps({"status": "FAILED", "error": str(exc), "output_dir": str(output)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
