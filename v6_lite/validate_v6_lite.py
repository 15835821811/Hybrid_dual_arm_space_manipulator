"""Independent delivery audit for the V6-lite folder."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
    WorkspaceSphere,
)
from v6_lite.hierarchical_qp import joint_addresses, rotation_error_angle_rad
from v6_lite.irregular_waypoints import build_original_irregular_target
from v6_lite.run_v6_lite import (
    CONTRACT_VERSION,
    TARGET_SATELLITE_COLLISION_POLICY,
    default_v6_lite_robot_spec,
)


VALIDATED_CONTINUUM_EE_OFFSET_M = np.asarray(
    [0.0475, 0.0, 0.0], dtype=np.float64
)
EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M = np.asarray(
    [1.930, 0.626, 0.0], dtype=np.float64
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _obstacles(scenario: dict[str, Any]) -> tuple[WorkspaceSphere, ...]:
    return tuple(
        WorkspaceSphere(
            name=str(item["name"]),
            center=np.asarray(item["center_w"], dtype=np.float64),
            radius=float(item["radius_m"]),
        )
        for item in scenario["workspace_obstacles"]
    )


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _quaternion_geodesic_angle_rad(
    reference_quaternion: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    """Independently compute shortest SO(3) angles for wxyz quaternions."""

    reference = np.asarray(reference_quaternion, dtype=np.float64).reshape(4)
    values = np.asarray(quaternions, dtype=np.float64)
    reference /= np.linalg.norm(reference)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    dots = np.abs(np.sum(values * reference, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _replay_trace(
    scenario_result: dict[str, Any],
    trace_path: Path,
    minimum_clearance_m: float,
    subdivisions: int,
) -> dict[str, Any]:
    spec = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        spec,
        _obstacles(scenario_result["scenario"]),
        WholeBodyVerificationConfig(
            minimum_clearance=minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    trace = np.load(trace_path, allow_pickle=False)
    data.qpos[:] = trace["initial_qpos"]
    data.qvel[:] = trace["initial_qvel"]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    qpos_ids, _dof_ids = joint_addresses(model, spec)
    continuum_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.continuum_tip_body_name
    )
    rigid_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.rigid_tip_body_name
    )
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_of_satelltte"
    )
    base_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, spec.base_joint_name
    )
    qstart = int(model.jnt_qposadr[base_joint_id])
    replay_planner: list[np.ndarray] = []
    replay_rigid: list[np.ndarray] = []
    replay_continuum: list[np.ndarray] = []
    replay_continuum_body_origin: list[np.ndarray] = []
    replay_rigid_rotation: list[np.ndarray] = []
    replay_continuum_rotation: list[np.ndarray] = []
    replay_base: list[np.ndarray] = []
    replay_task_qpos = [np.asarray(data.qpos).copy()]
    continuum_target_pairs = tuple(
        pair for pair in verifier.pairs if pair.pair_class == "continuum_target"
    )
    if not continuum_target_pairs:
        raise RuntimeError("continuum-target pair policy is empty")
    target_query_max_m = max(0.25, minimum_clearance_m + 0.10)
    target_fromto = np.zeros(6, dtype=np.float64)
    target_minimum = float("inf")
    target_minimum_pair: dict[str, Any] = {}
    target_below_clearance_states = 0
    target_penetrating_states = 0

    def audit_continuum_target_state(state_index: int) -> None:
        nonlocal target_minimum
        nonlocal target_minimum_pair
        nonlocal target_below_clearance_states
        nonlocal target_penetrating_states
        state_minimum = float("inf")
        for pair in continuum_target_pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    int(pair.geom_a),
                    int(pair.geom_b),
                    target_query_max_m,
                    target_fromto,
                )
            )
            if distance >= target_query_max_m:
                distance = target_query_max_m
            state_minimum = min(state_minimum, distance)
            if distance < target_minimum:
                target_minimum = distance
                target_minimum_pair = {
                    "pair_class": pair.pair_class,
                    "geom_a": pair.geom_a_name,
                    "geom_b": pair.geom_b_name,
                    "body_a": pair.body_a_name,
                    "body_b": pair.body_b_name,
                    "state_index": int(state_index),
                    "time_s": float(state_index * model.opt.timestep),
                    "signed_distance_m": distance,
                    "fromto": target_fromto.tolist(),
                }
        if state_minimum < minimum_clearance_m:
            target_below_clearance_states += 1
        if state_minimum < 0.0:
            target_penetrating_states += 1

    audit_continuum_target_state(0)
    initial_continuum_rotation = np.asarray(data.xmat[continuum_id]).reshape(3, 3)
    initial_continuum_position = (
        np.asarray(data.xpos[continuum_id]).copy()
        + initial_continuum_rotation @ VALIDATED_CONTINUUM_EE_OFFSET_M
    )
    torque = trace["torque"]
    for step, control in enumerate(torque):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        audit_continuum_target_state(step + 1)
        replay_planner.append(spec.decode_position(np.asarray(data.qpos[qpos_ids])))
        replay_rigid.append(np.asarray(data.xpos[rigid_id]).copy())
        continuum_body_origin = np.asarray(data.xpos[continuum_id]).copy()
        continuum_rotation = np.asarray(data.xmat[continuum_id]).reshape(3, 3).copy()
        replay_continuum_body_origin.append(continuum_body_origin)
        replay_continuum.append(
            continuum_body_origin
            + continuum_rotation @ VALIDATED_CONTINUUM_EE_OFFSET_M
        )
        replay_rigid_rotation.append(
            np.asarray(data.xmat[rigid_id]).reshape(3, 3).copy()
        )
        replay_continuum_rotation.append(continuum_rotation)
        replay_base.append(np.asarray(data.qpos[qstart : qstart + 7]).copy())
        if (step + 1) % 10 == 0:
            replay_task_qpos.append(np.asarray(data.qpos).copy())
    replay_task_qpos_array = np.asarray(replay_task_qpos)
    replay_base_array = np.asarray(replay_base)
    initial_base_pose = np.asarray(trace["initial_qpos"][qstart : qstart + 7])
    base_translation_drift = np.linalg.norm(
        replay_base_array[:, :3] - initial_base_pose[None, :3], axis=1
    )
    base_orientation_drift_deg = np.rad2deg(
        _quaternion_geodesic_angle_rad(
            initial_base_pose[3:7], replay_base_array[:, 3:7]
        )
    )
    report = verifier.verify_qpos_sequence(replay_task_qpos_array)
    return {
        "planner_q_residual_max": float(
            np.max(np.abs(np.asarray(replay_planner) - trace["planner_q"]))
        ),
        "rigid_tip_residual_max_m": float(
            np.max(np.abs(np.asarray(replay_rigid) - trace["rigid_tip"]))
        ),
        "continuum_tip_residual_max_m": float(
            np.max(np.abs(np.asarray(replay_continuum) - trace["continuum_tip"]))
        ),
        "continuum_tip_body_origin_residual_max_m": float(
            np.max(
                np.abs(
                    np.asarray(replay_continuum_body_origin)
                    - trace["continuum_tip_body_origin"]
                )
            )
        ),
        "continuum_initial_position_m": initial_continuum_position.tolist(),
        "continuum_initial_position_contract_error_m": float(
            np.linalg.norm(
                initial_continuum_position
                - EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M
            )
        ),
        "rigid_rotation_residual_max": float(
            np.max(
                np.abs(
                    np.asarray(replay_rigid_rotation) - trace["rigid_rotation"]
                )
            )
        ),
        "continuum_rotation_residual_max": float(
            np.max(
                np.abs(
                    np.asarray(replay_continuum_rotation)
                    - trace["continuum_rotation"]
                )
            )
        ),
        "base_qpos_residual_max": float(
            np.max(np.abs(replay_base_array - trace["base_qpos"]))
        ),
        "base_translation_drift_trace_residual_max_m": float(
            np.max(
                np.abs(
                    base_translation_drift - trace["base_translation_drift_m"]
                )
            )
        ),
        "base_orientation_drift_trace_residual_max_deg": float(
            np.max(
                np.abs(
                    base_orientation_drift_deg
                    - trace["base_orientation_drift_deg"]
                )
            )
        ),
        "base_translation_drift_final_m": float(base_translation_drift[-1]),
        "base_translation_drift_max_m": float(np.max(base_translation_drift)),
        "base_orientation_drift_final_deg": float(
            base_orientation_drift_deg[-1]
        ),
        "base_orientation_drift_max_deg": float(
            np.max(base_orientation_drift_deg)
        ),
        "task_qpos_residual_max": float(
            np.max(np.abs(replay_task_qpos_array - trace["task_qpos"]))
        ),
        "whole_body_verification": report.to_dict(),
        "continuum_target_500hz_verification": {
            "passed": bool(
                target_minimum >= minimum_clearance_m
                and target_below_clearance_states == 0
                and target_penetrating_states == 0
            ),
            "minimum_clearance_m": float(target_minimum),
            "required_clearance_m": float(minimum_clearance_m),
            "minimum_pair": target_minimum_pair,
            "below_required_clearance_state_count": int(
                target_below_clearance_states
            ),
            "penetrating_state_count": int(target_penetrating_states),
            "checked_state_count": int(torque.shape[0] + 1),
            "pair_count": len(continuum_target_pairs),
            "query_count": int((torque.shape[0] + 1) * len(continuum_target_pairs)),
            "query_distance_max_m": float(target_query_max_m),
            "time_scope": "all_native_500hz_replay_states_including_initial",
            "uses_mj_geom_distance": True,
        },
    }


def validate_delivery(root: Path = Path("v6_lite")) -> dict[str, Any]:
    output_dir = root / "output"
    metrics_path = output_dir / "v6_lite_metrics.json"
    manifest_path = output_dir / "artifact_manifest.json"
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    checks: dict[str, bool] = {}
    checks["contract_version"] = (
        metrics.get("contract_version") == CONTRACT_VERSION
        and manifest.get("contract_version") == CONTRACT_VERSION
    )
    checks["authoritative_metrics_hash"] = (
        manifest["metrics"]["sha256"] == _sha256(metrics_path)
    )
    scenarios = metrics.get("scenarios", [])
    run_config = metrics.get("run_config", {})
    checks["five_seeded_scenarios"] = (
        len(scenarios) == 5
        and len({item["scenario"]["seed"] for item in scenarios}) == 5
        and len({item["scenario"]["scenario_id"] for item in scenarios}) == 5
    )
    checks["exact_control_rates"] = (
        metrics["architecture"]["task_rate_hz"] == 50.0
        and metrics["architecture"]["torque_rate_hz"] == 500.0
        and abs(float(run_config["task_period_s"]) - 0.02) <= 1e-12
        and abs(float(run_config["physics_period_s"]) - 0.002) <= 1e-12
    )
    checks["requested_acceptance_thresholds_locked"] = (
        float(run_config["rigid_final_error_threshold_m"]) <= 0.00010
        and float(run_config["rigid_steady_rmse_threshold_m"]) <= 0.00015
        and float(run_config["continuum_irregular_path_rmse_threshold_m"])
        <= 0.00018
        and float(run_config["whole_body_minimum_clearance_m"]) >= 0.005
        and float(run_config["orientation_error_threshold_deg"]) <= 0.25
    )
    checks["learning_free_architecture"] = (
        metrics["architecture"]["learning_module"] is None
        and all(
            not item["execution_contract"]["learned_model_used"]
            and not item["execution_contract"]["diffusion_used"]
            and not item["execution_contract"]["candidate_sampling_used"]
            and not item["execution_contract"]["trajectory_projection_used"]
            and not item["execution_contract"]["oracle_joint_target_used"]
            for item in scenarios
        )
    )
    checks["original_irregular_waypoint_contract"] = all(
        item["scenario"]["continuum_target"]["mode"] == "irregular_waypoints"
        and item["scenario"]["continuum_target"]["preset"]
        == "irregular_generalization_v3_randomized"
        and int(item["scenario"]["continuum_target"]["waypoint_count"]) == 7
        and abs(
            float(item["scenario"]["continuum_target"]["path_duration_s"])
            - 21.0
        )
        <= 1e-12
        for item in scenarios
    )
    kinematic_contract = metrics.get("kinematic_contract", {})
    checks["continuum_end_effector_offset_contract"] = (
        kinematic_contract.get("continuum_end_effector_body_name") == "link_30"
        and np.allclose(
            np.asarray(
                kinematic_contract.get(
                    "continuum_end_effector_local_offset_body_m", []
                ),
                dtype=np.float64,
            ),
            VALIDATED_CONTINUUM_EE_OFFSET_M,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            np.asarray(
                kinematic_contract.get(
                    "continuum_end_effector_initial_world_position_m", []
                ),
                dtype=np.float64,
            ),
            EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
            rtol=0.0,
            atol=1e-12,
        )
    )
    forbidden_imports = {"torch", "diffusion_net", "diffusion_transformer"}
    source_imports: set[str] = set()
    for source in (root / "hierarchical_qp.py", root / "run_v6_lite.py"):
        source_imports.update(_import_roots(source))
    checks["no_learning_runtime_imports"] = not bool(
        source_imports.intersection(forbidden_imports)
    )
    checks["reported_summary_passed"] = bool(metrics.get("passed")) and all(
        metrics.get("summary_checks", {}).values()
    )

    manifest_trace_by_path = {
        str(item["path"]): str(item["sha256"]) for item in manifest.get("traces", [])
    }
    replay_reports: list[dict[str, Any]] = []
    recomputed_rigid_final: list[float] = []
    recomputed_rigid_rmse: list[float] = []
    recomputed_continuum_rmse: list[float] = []
    recomputed_rigid_orientation_max: list[float] = []
    recomputed_continuum_orientation_max: list[float] = []
    recomputed_base_translation_drift_max: list[float] = []
    recomputed_base_orientation_drift_max: list[float] = []
    recomputed_clearance: list[float] = []
    recomputed_continuum_target_clearance: list[float] = []
    recomputed_continuum_target_500hz_clearance: list[float] = []
    trace_hashes_ok = True
    trace_metrics_ok = True
    replay_ok = True
    single_qp_ok = True
    avoidance_ok = True
    dynamic_only_ok = True
    collision_scope_ok = True
    latency_ok = True
    continuum_initial_position_ok = True
    continuum_initial_position_errors: list[float] = []
    for item in scenarios:
        trace_path = Path(item["trace"]["path"])
        trace_hash = _sha256(trace_path)
        trace_hashes_ok &= (
            trace_hash == item["trace"]["sha256"]
            and manifest_trace_by_path.get(trace_path.as_posix()) == trace_hash
        )
        trace = np.load(trace_path, allow_pickle=False)
        times = trace["time"]
        steady_mask = times >= float(run_config["duration_s"]) - float(
            run_config["steady_window_s"]
        )
        rigid_final = float(trace["rigid_error"][-1])
        rigid_rmse = float(np.sqrt(np.mean(trace["rigid_error"][steady_mask] ** 2)))
        target_contract = item["scenario"]["continuum_target"]
        path_mask = (
            (times >= float(target_contract["path_start_s"]))
            & (times <= float(target_contract["path_end_s"]))
        )
        continuum_rmse = float(
            np.sqrt(np.mean(trace["continuum_error"][path_mask] ** 2))
        )
        rebuilt_target = build_original_irregular_target(
            np.asarray(target_contract["initial_position_w"], dtype=np.float64),
            np.asarray(target_contract["target_rotation_world"], dtype=np.float64),
            int(target_contract["random_seed"]),
        )
        rebuilt_positions = np.asarray(
            [rebuilt_target.sample(time_s)[0] for time_s in times]
        )
        rigid_orientation_recomputed = np.rad2deg(
            np.asarray(
                [
                    rotation_error_angle_rad(target, actual)
                    for target, actual in zip(
                        trace["rigid_target_rotation"], trace["rigid_rotation"]
                    )
                ]
            )
        )
        continuum_orientation_recomputed = np.rad2deg(
            np.asarray(
                [
                    rotation_error_angle_rad(target, actual)
                    for target, actual in zip(
                        trace["continuum_target_rotation"],
                        trace["continuum_rotation"],
                    )
                ]
            )
        )
        rigid_orientation_max = float(np.max(rigid_orientation_recomputed))
        continuum_orientation_max = float(
            np.max(continuum_orientation_recomputed)
        )
        recomputed_rigid_final.append(rigid_final)
        recomputed_rigid_rmse.append(rigid_rmse)
        recomputed_continuum_rmse.append(continuum_rmse)
        recomputed_rigid_orientation_max.append(rigid_orientation_max)
        recomputed_continuum_orientation_max.append(continuum_orientation_max)
        reported = item["metrics"]
        trace_metrics_ok &= (
            abs(rigid_final - reported["rigid_grasp_point"]["final_error_m"]) <= 1e-12
            and abs(
                rigid_rmse - reported["rigid_grasp_point"]["steady_rmse_m"]
            )
            <= 1e-12
            and abs(
                continuum_rmse
                - reported["continuum_irregular_waypoint_tracking"][
                    "active_path_rmse_m"
                ]
            )
            <= 1e-12
            and float(
                np.max(np.abs(rebuilt_positions - trace["continuum_target"]))
            )
            <= 1e-12
            and float(
                np.max(
                    np.abs(
                        rebuilt_target.waypoint_points_w
                        - np.asarray(target_contract["waypoint_points_m"])
                    )
                )
            )
            <= 1e-12
            and abs(
                rigid_orientation_max
                - reported["orientation_tracking"]["rigid_full_max_error_deg"]
            )
            <= 1e-12
            and abs(
                continuum_orientation_max
                - reported["orientation_tracking"][
                    "continuum_full_max_error_deg"
                ]
            )
            <= 1e-12
            and float(
                np.max(
                    np.abs(
                        rigid_orientation_recomputed
                        - trace["rigid_orientation_error_deg"]
                    )
                )
            )
            <= 1e-10
            and float(
                np.max(
                    np.abs(
                        continuum_orientation_recomputed
                        - trace["continuum_orientation_error_deg"]
                    )
                )
            )
            <= 1e-10
        )
        expected_qp = int(trace["torque"].shape[0] // 10)
        single_qp_ok &= (
            trace["task_success"].shape[0] == expected_qp
            and bool(np.all(trace["task_success"]))
            and reported["rates_and_latency"]["qp_call_count"] == expected_qp
            and reported["rates_and_latency"]["torque_update_count"]
            == trace["torque"].shape[0]
        )
        avoidance_ok &= (
            int(np.sum(trace["task_binding_clearance"])) > 0
            and float(np.max(trace["task_avoidance_intervention"])) > 1e-5
        )
        dynamic_only_ok &= (
            item["execution_contract"]["qpos_write_count_after_initialization"] == 0
            and item["execution_contract"]["qvel_write_count_after_initialization"] == 0
            and not item["execution_contract"]["safe_stop_used"]
        )
        collision_scope_ok &= (
            not item["execution_contract"]["continuous_time_collision_certified"]
            and item["scenario"]["target_satellite_collision_policy"]
            == TARGET_SATELLITE_COLLISION_POLICY
            and item["execution_contract"]["moving_target_clearance_drift_in_qp"]
            and item["execution_contract"]["target_contact_exemption"]
            == "collision_0072_and_collision_0073_only"
            and "moving_target_except_rigid_terminal_contact_geoms"
            in item["execution_contract"]["collision_scope"]
        )
        latency_ok &= (
            float(np.percentile(trace["task_full_latency"], 95.0))
            <= float(run_config["task_latency_p95_threshold_s"])
            and float(np.percentile(trace["torque_latency"], 95.0))
            <= float(run_config["torque_latency_p95_threshold_s"])
        )
        replay = _replay_trace(
            item,
            trace_path,
            float(run_config["whole_body_minimum_clearance_m"]),
            int(run_config["verification_subdivisions"]),
        )
        replay_reports.append(
            {"scenario_id": item["scenario"]["scenario_id"], **replay}
        )
        initial_position_error = float(
            replay["continuum_initial_position_contract_error_m"]
        )
        continuum_initial_position_errors.append(initial_position_error)
        continuum_initial_position_ok &= (
            initial_position_error <= 1e-8
            and np.allclose(
                np.asarray(target_contract["initial_position_w"], dtype=np.float64),
                EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
                rtol=0.0,
                atol=1e-8,
            )
            and np.allclose(
                np.asarray(
                    reported["continuum_irregular_waypoint_tracking"][
                        "end_effector_local_offset_body_m"
                    ],
                    dtype=np.float64,
                ),
                VALIDATED_CONTINUUM_EE_OFFSET_M,
                rtol=0.0,
                atol=1e-12,
            )
        )
        recomputed_base_translation_drift_max.append(
            float(replay["base_translation_drift_max_m"])
        )
        recomputed_base_orientation_drift_max.append(
            float(replay["base_orientation_drift_max_deg"])
        )
        trace_metrics_ok &= (
            replay["base_translation_drift_trace_residual_max_m"] <= 1e-12
            and replay["base_orientation_drift_trace_residual_max_deg"] <= 1e-10
            and abs(
                replay["base_translation_drift_final_m"]
                - reported["free_floating_base"]["translation_drift_final_m"]
            )
            <= 1e-12
            and abs(
                replay["base_translation_drift_max_m"]
                - reported["free_floating_base"]["translation_drift_max_m"]
            )
            <= 1e-12
            and abs(
                replay["base_orientation_drift_final_deg"]
                - reported["free_floating_base"]["orientation_drift_final_deg"]
            )
            <= 1e-10
            and abs(
                replay["base_orientation_drift_max_deg"]
                - reported["free_floating_base"]["orientation_drift_max_deg"]
            )
            <= 1e-10
        )
        verifier_report = replay["whole_body_verification"]
        recomputed_clearance.append(float(verifier_report["minimum_clearance"]))
        minimum_by_class = verifier_report.get("minimum_by_class", {})
        continuum_target_clearance = float(
            minimum_by_class.get("continuum_target", float("-inf"))
        )
        recomputed_continuum_target_clearance.append(continuum_target_clearance)
        target_500hz_report = replay["continuum_target_500hz_verification"]
        recomputed_continuum_target_500hz_clearance.append(
            float(target_500hz_report["minimum_clearance_m"])
        )
        collision_scope_ok &= (
            {"continuum_target", "rigid_target", "base_target"}
            <= set(minimum_by_class)
            and continuum_target_clearance
            >= float(run_config["whole_body_minimum_clearance_m"])
            and verifier_report["pair_policy_sha256"]
            == reported["whole_body_clearance"]["pair_policy_sha256"]
            and int(verifier_report["pair_count"])
            == int(reported["whole_body_clearance"]["pair_count"])
        )
        replay_ok &= (
            replay["planner_q_residual_max"] <= 1e-11
            and replay["rigid_tip_residual_max_m"] <= 1e-11
            and replay["continuum_tip_residual_max_m"] <= 1e-11
            and replay["continuum_tip_body_origin_residual_max_m"] <= 1e-11
            and replay["rigid_rotation_residual_max"] <= 1e-11
            and replay["continuum_rotation_residual_max"] <= 1e-11
            and replay["base_qpos_residual_max"] <= 1e-11
            and replay["base_translation_drift_trace_residual_max_m"] <= 1e-12
            and replay["base_orientation_drift_trace_residual_max_deg"] <= 1e-10
            and replay["task_qpos_residual_max"] <= 1e-11
            and bool(verifier_report["feasible"])
            and not bool(verifier_report["continuous_time_certified"])
            and bool(target_500hz_report["passed"])
            and int(target_500hz_report["below_required_clearance_state_count"])
            == 0
            and int(target_500hz_report["penetrating_state_count"]) == 0
            and int(target_500hz_report["checked_state_count"])
            == int(trace["torque"].shape[0] + 1)
            and int(target_500hz_report["query_count"])
            == int(
                target_500hz_report["checked_state_count"]
                * target_500hz_report["pair_count"]
            )
        )

    checks["trace_hashes"] = trace_hashes_ok and len(manifest_trace_by_path) == 5
    checks["continuum_initial_position_is_1930_626_0_mm"] = (
        continuum_initial_position_ok
        and max(continuum_initial_position_errors) <= 1e-8
    )
    checks["metrics_recomputed_from_traces"] = trace_metrics_ok
    checks["native_torque_replay_exact"] = replay_ok
    checks["one_qp_per_50hz_tick"] = single_qp_ok
    checks["obstacle_constraints_materially_engaged"] = avoidance_ok
    checks["dynamic_ctrl_mj_step_only"] = dynamic_only_ok
    checks["honest_collision_claim_scope"] = collision_scope_ok
    checks["moving_target_continuum_clearance"] = (
        len(recomputed_continuum_target_clearance) == len(scenarios)
        and min(recomputed_continuum_target_clearance)
        >= float(run_config["whole_body_minimum_clearance_m"])
    )
    checks["continuum_target_all_500hz_replay_states_clear"] = (
        len(recomputed_continuum_target_500hz_clearance) == len(scenarios)
        and min(recomputed_continuum_target_500hz_clearance)
        >= float(run_config["whole_body_minimum_clearance_m"])
    )
    checks["measured_rate_deadlines"] = latency_ok
    checks["rigid_grasp_point_error"] = (
        max(recomputed_rigid_final)
        <= float(run_config["rigid_final_error_threshold_m"])
        and max(recomputed_rigid_rmse)
        <= float(run_config["rigid_steady_rmse_threshold_m"])
    )
    checks["continuum_irregular_waypoint_tracking_error"] = (
        max(recomputed_continuum_rmse)
        <= float(run_config["continuum_irregular_path_rmse_threshold_m"])
    )
    checks["both_arm_orientation_error"] = (
        max(recomputed_rigid_orientation_max)
        < float(run_config["orientation_error_threshold_deg"])
        and max(recomputed_continuum_orientation_max)
        < float(run_config["orientation_error_threshold_deg"])
    )
    checks["whole_body_minimum_clearance"] = (
        min(recomputed_clearance)
        >= float(run_config["whole_body_minimum_clearance_m"])
    )
    aggregate_metrics = metrics["aggregate_metrics"]
    checks["base_pose_drift_recomputed"] = (
        len(recomputed_base_translation_drift_max) == len(scenarios)
        and all(np.isfinite(recomputed_base_translation_drift_max))
        and all(np.isfinite(recomputed_base_orientation_drift_max))
        and min(recomputed_base_translation_drift_max) >= 0.0
        and min(recomputed_base_orientation_drift_max) >= 0.0
        and abs(
            max(recomputed_base_translation_drift_max)
            - float(aggregate_metrics["base_translation_drift_m_max"])
        )
        <= 1e-12
        and abs(
            max(recomputed_base_orientation_drift_max)
            - float(aggregate_metrics["base_orientation_drift_deg_max"])
        )
        <= 1e-10
    )
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "contract_version": CONTRACT_VERSION,
        "passed": not failures,
        "passed_count": int(sum(checks.values())),
        "total_count": len(checks),
        "checks": checks,
        "failures": failures,
        "recomputed_aggregate": {
            "scenario_count": len(scenarios),
            "rigid_final_error_m_max": max(recomputed_rigid_final),
            "rigid_steady_rmse_m_max": max(recomputed_rigid_rmse),
            "continuum_irregular_path_rmse_m_max": max(
                recomputed_continuum_rmse
            ),
            "rigid_orientation_error_deg_max": max(
                recomputed_rigid_orientation_max
            ),
            "continuum_orientation_error_deg_max": max(
                recomputed_continuum_orientation_max
            ),
            "base_translation_drift_m_max": max(
                recomputed_base_translation_drift_max
            ),
            "base_orientation_drift_deg_max": max(
                recomputed_base_orientation_drift_max
            ),
            "whole_body_minimum_clearance_m": min(recomputed_clearance),
            "continuum_target_minimum_clearance_m": min(
                recomputed_continuum_target_clearance
            ),
            "continuum_target_500hz_minimum_clearance_m": min(
                recomputed_continuum_target_500hz_clearance
            ),
            "continuum_initial_position_contract_error_m_max": max(
                continuum_initial_position_errors
            ),
        },
        "native_replays": replay_reports,
    }
    output_path = output_dir / "validation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("v6_lite"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = validate_delivery(args.root)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": f"{result['passed_count']}/{result['total_count']}",
                "failures": result["failures"],
                "recomputed_aggregate": result["recomputed_aggregate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
