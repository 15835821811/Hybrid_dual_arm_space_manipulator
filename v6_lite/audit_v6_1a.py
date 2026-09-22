"""Generate the independent V6.1-A shape and finite-volume audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF,
    CONTINUUM_MODEL_SPEC_VERSION,
    ContinuumModelSpec,
    default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import (
    ContinuumShapeModel,
    DiscreteContinuumKinematics,
    axis_angle_rotation,
    rotation_angle,
    transform_from_free_qpos,
    vee,
)
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    CapsuleEnvelope,
    ShapeClearanceShadow,
    build_continuum_capsule_envelopes,
    minimum_capsule_clearance,
    pcc_tube_clearance,
    target_box_from_mujoco,
)


AUDIT_VERSION = "v6.1-a.audit.1"
DEFAULT_SEED = 20260922
DEFAULT_CONFIGURATION_COUNT = 10_000
DEFAULT_CALIBRATION_COUNT = 8_000
DEFAULT_TARGET_CASE_COUNT = 10_000
DEFAULT_OUTPUT = Path("v6_lite/output/v6_1a")
SHAPE_SAMPLE_COUNT = 31
PCC_ENVELOPE_SAMPLES_PER_SEGMENT = 17
CAPSULE_AXIS_SAMPLES = 5
SHADOW_SAMPLES_PER_SEGMENT = 25
SAFETY_GATE_M = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "v6_lite" / "hierarchical_qp.py",
        root / "v6_lite" / "run_v6_lite.py",
        root / "v6_lite" / "irregular_waypoints.py",
    )
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def build_frozen_configurations(
    spec: ContinuumModelSpec,
    count: int,
    seed: int,
) -> np.ndarray:
    if count < 100:
        raise ValueError("configuration audit requires at least 100 configurations")
    structured: list[np.ndarray] = [np.zeros(10, dtype=np.float64)]
    for coordinate in range(10):
        for value in (-1.0, -0.5, 0.5, 1.0):
            item = np.zeros(10, dtype=np.float64)
            item[coordinate] = value
            structured.append(item)
    for segment in range(5):
        for first, second in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
            item = np.zeros(10, dtype=np.float64)
            item[2 * segment : 2 * segment + 2] = [first, second]
            structured.append(item)
    for sign in (-1.0, 1.0):
        structured.extend(
            [
                sign * np.asarray([1, 0, -1, 0, 1, 0, -1, 0, 1, 0], dtype=float),
                sign * np.asarray([0, 1, 0, -1, 0, 1, 0, -1, 0, 1], dtype=float),
                sign * np.asarray([1, 1, -1, -1, 1, 1, -1, -1, 1, 1], dtype=float),
            ]
        )
    values = np.asarray(structured, dtype=np.float64)
    rng = np.random.default_rng(seed)
    remaining = count - values.shape[0]
    if remaining < 0:
        return values[:count]
    random_values = rng.uniform(
        spec.work_domain_lower_rad,
        spec.work_domain_upper_rad,
        size=(remaining, 10),
    )
    return np.vstack([values, random_values])


def _random_base_transform(rng: np.random.Generator) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = rng.uniform(-0.2, 0.2, size=3)
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    transform[:3, :3] = axis_angle_rotation(axis, float(rng.uniform(-0.5, 0.5)))
    return transform


def _transform_to_free_qpos(transform: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, transform[:3, :3].reshape(-1))
    return np.concatenate([transform[:3, 3], quaternion])


def _progress(label: str, index: int, total: int) -> None:
    interval = max(total // 10, 1)
    if index == 0 or (index + 1) % interval == 0 or index + 1 == total:
        print(f"[{label}] {index + 1}/{total}", flush=True)


def audit_shape_model(
    configurations: np.ndarray,
    spec: ContinuumModelSpec,
    model: mujoco.MjModel,
    seed: int,
) -> dict[str, Any]:
    shape = ContinuumShapeModel(spec)
    discrete = DiscreteContinuumKinematics(spec)
    data = mujoco.MjData(model)
    robot = default_v6_lite_robot_spec()
    qpos_ids, _ = joint_addresses(model, robot)
    base_qpos, _ = free_joint_slices(model, robot.base_joint_name)
    rng = np.random.default_rng(seed + 1)
    sample_arclengths = np.linspace(0.0, spec.total_length_m, SHAPE_SAMPLE_COUNT)
    position_error = np.zeros((configurations.shape[0], SHAPE_SAMPLE_COUNT))
    orientation_error = np.zeros_like(position_error)
    discrete_position_max = 0.0
    discrete_orientation_max = 0.0
    for configuration_index, configuration in enumerate(configurations):
        base_transform = _random_base_transform(rng)
        actual = spec.planner_to_actuated @ configuration
        transforms = discrete.body_transforms(actual, base_transform)
        data.qpos[:] = model.qpos0
        data.qpos[qpos_ids[:CONTINUUM_ACTUATED_DOF]] = actual
        data.qpos[base_qpos] = _transform_to_free_qpos(base_transform)
        mujoco.mj_forward(model, data)
        for body_name, expected in transforms.items():
            body_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            )
            if body_id < 0:
                raise ValueError(f"MuJoCo model is missing body {body_name!r}")
            actual_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
            discrete_position_max = max(
                discrete_position_max,
                float(np.linalg.norm(expected[:3, 3] - data.xpos[body_id])),
            )
            discrete_orientation_max = max(
                discrete_orientation_max,
                rotation_angle(expected[:3, :3].T @ actual_rotation),
            )
        for sample_index, arclength in enumerate(sample_arclengths):
            pcc = shape.evaluate(
                configuration,
                base_transform,
                float(arclength),
                with_jacobians=False,
            )
            chain = discrete.evaluate(
                actual,
                base_transform,
                float(arclength),
                transforms=transforms,
            )
            position_error[configuration_index, sample_index] = np.linalg.norm(
                pcc.position - chain.position
            )
            orientation_error[configuration_index, sample_index] = rotation_angle(
                pcc.rotation.T @ chain.rotation
            )
        _progress("shape-model", configuration_index, configurations.shape[0])

    jacobian_case_count = min(256, configurations.shape[0])
    jacobian_rng = np.random.default_rng(seed + 2)
    position_jacobian_errors: list[float] = []
    rotation_jacobian_errors: list[float] = []
    step = 1e-6
    for index in range(jacobian_case_count):
        configuration = configurations[
            int(jacobian_rng.integers(0, configurations.shape[0]))
        ]
        arclength = float(jacobian_rng.uniform(0.0, spec.total_length_m))
        base_transform = _random_base_transform(jacobian_rng)
        center = shape.evaluate(configuration, base_transform, arclength)
        for coordinate in range(10):
            plus = configuration.copy()
            minus = configuration.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            value_plus = shape.evaluate(
                plus, base_transform, arclength, with_jacobians=False
            )
            value_minus = shape.evaluate(
                minus, base_transform, arclength, with_jacobians=False
            )
            numeric_position = (value_plus.position - value_minus.position) / (2.0 * step)
            rotation_rate = (value_plus.rotation - value_minus.rotation) / (2.0 * step)
            numeric_rotation = vee(rotation_rate @ center.rotation.T)
            position_jacobian_errors.append(
                float(
                    np.max(
                        np.abs(
                            numeric_position
                            - center.position_jacobian[:, coordinate]
                        )
                    )
                )
            )
            rotation_jacobian_errors.append(
                float(
                    np.max(
                        np.abs(
                            numeric_rotation
                            - center.rotation_jacobian[:, coordinate]
                        )
                    )
                )
            )

    zero_finite = True
    zero_values = []
    for scale in (0.0, 1e-12, -1e-12, 1e-9, -1e-9):
        configuration = np.full(10, scale, dtype=np.float64)
        for arclength in np.linspace(0.0, spec.total_length_m, 51):
            value = shape.evaluate(configuration, np.eye(4), float(arclength))
            arrays = (
                value.position,
                value.rotation,
                value.position_jacobian,
                value.rotation_jacobian,
            )
            zero_finite &= all(np.all(np.isfinite(item)) for item in arrays)
            zero_values.append(value.position_jacobian)
    zero_derivative_jump = float(
        np.max(np.abs(np.asarray(zero_values[51]) - np.asarray(zero_values[102])))
    )

    residual_values_l2: list[float] = []
    residual_values_linf: list[float] = []
    trace_state_count = 0
    trace_root = Path("v6_lite/output/traces")
    for trace_path in sorted(trace_root.glob("*.npz")):
        with np.load(trace_path, allow_pickle=False) as trace:
            task_qpos = np.asarray(trace["task_qpos"], dtype=np.float64)
        low = task_qpos[:, qpos_ids[:CONTINUUM_ACTUATED_DOF]]
        planner = low @ spec.actuated_to_planner.T
        residual = low - planner @ spec.planner_to_actuated.T
        residual_values_l2.extend(np.linalg.norm(residual, axis=1).tolist())
        residual_values_linf.extend(np.max(np.abs(residual), axis=1).tolist())
        trace_state_count += int(low.shape[0])

    per_arclength = []
    for index, arclength in enumerate(sample_arclengths):
        per_arclength.append(
            {
                "arclength_m": float(arclength),
                "normalized_arclength": float(arclength / spec.total_length_m),
                "position_error_m": _summary(position_error[:, index]),
                "orientation_error_rad": _summary(orientation_error[:, index]),
            }
        )
    checks = {
        "configuration_count_at_least_10000": configurations.shape[0] >= 10_000,
        "discrete_fk_position_matches_mujoco": discrete_position_max <= 1e-6,
        "discrete_fk_orientation_matches_mujoco": discrete_orientation_max <= 1e-6,
        "pcc_position_jacobian_matches_finite_difference": max(position_jacobian_errors) <= 1e-6,
        "pcc_rotation_jacobian_matches_finite_difference": max(rotation_jacobian_errors) <= 1e-6,
        "zero_curvature_is_finite": bool(zero_finite),
        "zero_curvature_derivative_is_continuous": zero_derivative_jump <= 1e-7,
        "full_arm_error_distribution_reported": len(per_arclength) == SHAPE_SAMPLE_COUNT,
        "actual_subspace_residual_reported": trace_state_count > 0,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "model_contract_version": CONTINUUM_MODEL_SPEC_VERSION,
        "model_contract_sha256": spec.contract_sha256(),
        "passed": all(checks.values()),
        "checks": checks,
        "declared_work_domain_rad": {
            "lower": spec.work_domain_lower_rad.tolist(),
            "upper": spec.work_domain_upper_rad.tolist(),
        },
        "configuration_count": int(configurations.shape[0]),
        "coverage_modes": [
            "straight",
            "single_axis",
            "two_axis",
            "inter_segment_counter_bend",
            "boundary",
            "seeded_uniform",
        ],
        "discrete_fk_vs_mujoco": {
            "position_error_m_max": discrete_position_max,
            "orientation_error_rad_max": discrete_orientation_max,
            "position_threshold_m": 1e-6,
            "orientation_threshold_rad": 1e-6,
        },
        "pcc_vs_discrete_full_arm": {
            "position_error_m": _summary(position_error),
            "orientation_error_rad": _summary(orientation_error),
            "per_arclength": per_arclength,
            "interpretation": (
                "model discrepancy between continuous PCC and the actual discrete chain; "
                "not a numerical implementation tolerance"
            ),
        },
        "pcc_derivative_audit": {
            "case_count": jacobian_case_count,
            "coordinate_checks": int(10 * jacobian_case_count),
            "finite_difference_step_rad": step,
            "position_jacobian_absolute_error": _summary(
                np.asarray(position_jacobian_errors)
            ),
            "rotation_jacobian_absolute_error": _summary(
                np.asarray(rotation_jacobian_errors)
            ),
        },
        "zero_curvature": {
            "finite": bool(zero_finite),
            "maximum_plus_minus_derivative_jump": zero_derivative_jump,
        },
        "actual_low_level_subspace_monitor": {
            "definition": "e_perp = q_a - B B_plus q_a",
            "trace_state_count": trace_state_count,
            "residual_l2_rad": _summary(np.asarray(residual_values_l2)),
            "residual_linf_rad": _summary(np.asarray(residual_values_linf)),
        },
        "controller_source_sha256": _source_hashes(),
    }


def _pcc_points_by_segment(
    shape: ContinuumShapeModel,
    configuration: np.ndarray,
    samples_per_segment: int,
) -> tuple[np.ndarray, ...]:
    values = []
    boundaries = shape.spec.segment_boundaries_m
    for segment in range(5):
        arclengths = boundaries[segment] + np.linspace(
            0.0,
            shape.spec.segment_lengths_m[segment],
            samples_per_segment,
        )
        points = [
            shape.evaluate(
                configuration, np.eye(4), float(value), with_jacobians=False
            ).position
            for value in arclengths
        ]
        values.append(np.asarray(points))
    return tuple(values)


def _capsule_axis_world_points(
    capsule: CapsuleEnvelope,
    transform: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    parameters = np.linspace(0.0, 1.0, sample_count)
    local = (
        (1.0 - parameters[:, None]) * capsule.local_start
        + parameters[:, None] * capsule.local_end
    )
    return transform[:3, 3] + local @ transform[:3, :3].T


def audit_geometry_envelope(
    configurations: np.ndarray,
    calibration_count: int,
    spec: ContinuumModelSpec,
    model: mujoco.MjModel,
) -> tuple[dict[str, Any], np.ndarray]:
    if calibration_count <= 0 or calibration_count >= configurations.shape[0]:
        raise ValueError("calibration count must leave a held-out set")
    shape = ContinuumShapeModel(spec)
    discrete = DiscreteContinuumKinematics(spec)
    envelopes = build_continuum_capsule_envelopes(model, spec)
    grouped: dict[int, list[CapsuleEnvelope]] = defaultdict(list)
    for capsule in envelopes.capsules:
        grouped[capsule.segment_index].append(capsule)
    required = np.zeros((configurations.shape[0], 5), dtype=np.float64)
    for configuration_index, configuration in enumerate(configurations):
        actual = spec.planner_to_actuated @ configuration
        transforms = discrete.body_transforms(actual, np.eye(4))
        pcc_points = _pcc_points_by_segment(
            shape, configuration, PCC_ENVELOPE_SAMPLES_PER_SEGMENT
        )
        for segment in range(5):
            maximum = 0.0
            for capsule in grouped[segment]:
                axis_points = _capsule_axis_world_points(
                    capsule,
                    transforms[capsule.body_name],
                    CAPSULE_AXIS_SAMPLES,
                )
                distances = np.linalg.norm(
                    axis_points[:, None, :] - pcc_points[segment][None, :, :],
                    axis=2,
                )
                maximum = max(
                    maximum,
                    float(np.max(np.min(distances, axis=1) + capsule.radius_m)),
                )
            required[configuration_index, segment] = maximum
        _progress("geometry-envelope", configuration_index, configurations.shape[0])

    axis_allowance = np.zeros(5, dtype=np.float64)
    for segment in range(5):
        axis_allowance[segment] = max(
            capsule.axis_length_m / (2.0 * (CAPSULE_AXIS_SAMPLES - 1))
            for capsule in grouped[segment]
        )
    pcc_allowance = spec.segment_lengths_m / (
        2.0 * (PCC_ENVELOPE_SAMPLES_PER_SEGMENT - 1)
    )
    numerical_margin = np.full(5, 1e-4, dtype=np.float64)
    calibration_maximum = np.max(required[:calibration_count], axis=0)
    tube_radii = calibration_maximum + axis_allowance + pcc_allowance + numerical_margin
    held_out = required[calibration_count:]
    excess = held_out - tube_radii
    uncovered = np.any(excess > 1e-12, axis=1)
    capsule_dicts = [item.to_dict() for item in envelopes.capsules]
    checks = {
        "configuration_count_at_least_10000": configurations.shape[0] >= 10_000,
        "calibration_and_held_out_sets_are_disjoint": calibration_count < configurations.shape[0],
        "every_continuum_geom_has_one_owner": (
            len(envelopes.continuum_geom_names)
            == len(envelopes.capsules) + len(envelopes.fallback_geom_names)
        ),
        "all_capsules_cover_source_vertices": max(
            item.maximum_vertex_excess_m for item in envelopes.capsules
        )
        <= 1e-10,
        "held_out_capsule_axes_inside_pcc_tubes": int(np.count_nonzero(uncovered)) == 0,
        "fallback_mount_remains_under_mujoco_policy": envelopes.fallback_geom_names
        == ("collision_0003",),
    }
    payload = {
        "audit_version": AUDIT_VERSION,
        "model_contract_sha256": spec.contract_sha256(),
        "passed": all(checks.values()),
        "checks": checks,
        "claim_boundary": (
            "Finite seeded regression evidence over the declared domain; not a "
            "global analytic enclosure certificate. The mount fallback remains "
            "under the original MuJoCo geometry constraint."
        ),
        "configuration_count": int(configurations.shape[0]),
        "calibration_configuration_count": int(calibration_count),
        "held_out_configuration_count": int(
            configurations.shape[0] - calibration_count
        ),
        "continuum_collision_geom_count": len(envelopes.continuum_geom_names),
        "capsule_count": len(envelopes.capsules),
        "fallback_geom_names": list(envelopes.fallback_geom_names),
        "ownership": {
            "capsule_geom_names": [item.geom_name for item in envelopes.capsules],
            "fallback_geom_names": list(envelopes.fallback_geom_names),
        },
        "capsules": capsule_dicts,
        "coverage_sampling": {
            "pcc_samples_per_segment": PCC_ENVELOPE_SAMPLES_PER_SEGMENT,
            "capsule_axis_samples": CAPSULE_AXIS_SAMPLES,
            "axis_sampling_allowance_m": axis_allowance.tolist(),
            "pcc_sampling_allowance_m": pcc_allowance.tolist(),
            "numerical_margin_m": numerical_margin.tolist(),
        },
        "calibrated_pcc_tube_radius_m": tube_radii.tolist(),
        "required_radius_distribution_by_segment": [
            _summary(required[:, segment]) for segment in range(5)
        ],
        "held_out": {
            "uncovered_configuration_count": int(np.count_nonzero(uncovered)),
            "maximum_radius_excess_m": float(np.max(excess)),
        },
    }
    return payload, tube_radii


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, quaternion)
    return flat.reshape(3, 3)


def _set_target_geom_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: slice,
    target_geom_id: int,
    center: np.ndarray,
    rotation: np.ndarray,
) -> None:
    geom_local_rotation = _rotation_from_mujoco_quaternion(
        model.geom_quat[target_geom_id]
    )
    body_rotation = rotation @ geom_local_rotation.T
    body_position = np.asarray(center) - body_rotation @ model.geom_pos[target_geom_id]
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, body_rotation.reshape(-1))
    data.qpos[target_qpos] = np.concatenate([body_position, quaternion])


def _rotation_from_mujoco_quaternion(quaternion: np.ndarray) -> np.ndarray:
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, np.asarray(quaternion, dtype=np.float64))
    return flat.reshape(3, 3)


def audit_shape_vs_geometry(
    configurations: np.ndarray,
    target_case_count: int,
    tube_radii: np.ndarray,
    spec: ContinuumModelSpec,
    model: mujoco.MjModel,
    seed: int,
) -> dict[str, Any]:
    robot = default_v6_lite_robot_spec()
    data = mujoco.MjData(model)
    qpos_ids, _ = joint_addresses(model, robot)
    base_qpos, _ = free_joint_slices(model, robot.base_joint_name)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    target_geom_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
        )
    )
    if target_geom_id < 0:
        raise ValueError("target satellite collision box is missing")
    shadow = ShapeClearanceShadow(
        model,
        target_geom_id,
        tube_radii,
        spec=spec,
        samples_per_segment=SHADOW_SAMPLES_PER_SEGMENT,
        query_distance_max_m=0.5,
    )
    rng = np.random.default_rng(seed + 3)
    categories = ("face", "edge", "corner", "rotated", "mid_arm", "penetration")
    capsule_false_safe = 0
    pcc_false_safe = 0
    capsule_conservatism_violations = 0
    query_truncation_count = 0
    category_counts: Counter[str] = Counter()
    category_unsafe: Counter[str] = Counter()
    actual_distances: list[float] = []
    capsule_distances: list[float] = []
    pcc_distances: list[float] = []
    gradient_cases: list[tuple[np.ndarray, np.ndarray]] = []

    candidates_by_category = {
        "mid_arm": [
            item
            for item in shadow.envelopes.capsules
            if item.segment_index in (1, 2, 3) and item.body_name.startswith("link_")
        ]
    }
    for case_index in range(target_case_count):
        configuration = configurations[case_index % configurations.shape[0]].copy()
        data.qpos[:] = model.qpos0
        full_configuration = robot.planner_zero.copy()
        full_configuration[:10] = configuration
        data.qpos[qpos_ids] = robot.encode_position(full_configuration)
        data.qpos[base_qpos] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        data.qpos[target_qpos] = [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        category = categories[case_index % len(categories)]
        category_counts[category] += 1
        if category == "mid_arm":
            capsule = candidates_by_category[category][
                case_index % len(candidates_by_category[category])
            ]
        else:
            capsule = shadow.envelopes.capsules[
                int(rng.integers(0, len(shadow.envelopes.capsules)))
            ]
        body_rotation = np.asarray(data.xmat[capsule.body_id]).reshape(3, 3)
        body_position = np.asarray(data.xpos[capsule.body_id])
        local_midpoint = 0.5 * (capsule.local_start + capsule.local_end)
        arm_point = body_position + body_rotation @ local_midpoint
        target_rotation = (
            np.eye(3) if category in ("face", "edge", "corner", "penetration") else _random_rotation(rng)
        )
        half_extents = np.asarray(model.geom_size[target_geom_id], dtype=np.float64)
        if category == "penetration":
            target_center = arm_point.copy()
        else:
            active_count = {
                "face": 1,
                "edge": 2,
                "corner": 3,
                "rotated": int(rng.integers(1, 4)),
                "mid_arm": 1,
            }[category]
            axes = rng.choice(3, size=active_count, replace=False)
            gap = float(rng.uniform(-0.02, 0.18))
            offset = np.zeros(3, dtype=np.float64)
            for axis in axes:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                offset[axis] = sign * (
                    half_extents[axis]
                    + (capsule.radius_m + gap) / np.sqrt(active_count)
                )
            target_center = arm_point - target_rotation @ offset
        _set_target_geom_pose(
            model,
            data,
            target_qpos,
            target_geom_id,
            target_center,
            target_rotation,
        )
        mujoco.mj_forward(model, data)
        comparison = shadow.evaluate(data, configuration, np.eye(4))
        exact = comparison.mujoco_geometry.signed_distance_m
        capsule_distance = comparison.capsule.signed_distance_m
        pcc_distance = comparison.pcc_tube.signed_distance_m
        query_truncation_count += int(comparison.mujoco_geometry.query_truncated)
        actual_distances.append(exact)
        capsule_distances.append(capsule_distance)
        pcc_distances.append(pcc_distance)
        unsafe = exact < SAFETY_GATE_M
        category_unsafe[category] += int(unsafe)
        capsule_false_safe += int(capsule_distance >= SAFETY_GATE_M and unsafe)
        pcc_false_safe += int(pcc_distance >= SAFETY_GATE_M and unsafe)
        if exact >= 0.0 and not comparison.mujoco_geometry.query_truncated:
            capsule_conservatism_violations += int(capsule_distance > exact + 2e-5)
        if (
            len(gradient_cases) < 32
            and exact > 0.01
            and capsule_distance > 0.0
            and pcc_distance > 0.0
        ):
            gradient_cases.append((configuration.copy(), data.qpos[target_qpos].copy()))
        _progress("shape-vs-geometry", case_index, target_case_count)

    pcc_gradient_errors: list[float] = []
    capsule_gradient_errors: list[float] = []
    pcc_gradient_stable = 0
    capsule_gradient_stable = 0
    finite_difference_step = 1e-6
    for configuration, target_state in gradient_cases:
        data.qpos[:] = model.qpos0
        full_configuration = robot.planner_zero.copy()
        full_configuration[:10] = configuration
        data.qpos[qpos_ids] = robot.encode_position(full_configuration)
        data.qpos[base_qpos] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        data.qpos[target_qpos] = target_state
        mujoco.mj_forward(model, data)
        box = target_box_from_mujoco(model, data, target_geom_id)
        center_pcc = pcc_tube_clearance(
            shadow.shape_model,
            configuration,
            np.eye(4),
            box,
            tube_radii,
            samples_per_segment=SHADOW_SAMPLES_PER_SEGMENT,
        )
        center_capsule = minimum_capsule_clearance(
            model, data, shadow.envelopes, box, spec=spec
        )
        numeric_pcc = np.zeros(10, dtype=np.float64)
        numeric_capsule = np.zeros(10, dtype=np.float64)
        pcc_sources_stable = True
        capsule_sources_stable = True
        for coordinate in range(10):
            distances_pcc = []
            distances_capsule = []
            source_pcc = []
            source_capsule = []
            for sign in (1.0, -1.0):
                perturbed = configuration.copy()
                perturbed[coordinate] += sign * finite_difference_step
                full_configuration[:10] = perturbed
                data.qpos[qpos_ids] = robot.encode_position(full_configuration)
                data.qpos[target_qpos] = target_state
                mujoco.mj_forward(model, data)
                current_box = target_box_from_mujoco(model, data, target_geom_id)
                value_pcc = pcc_tube_clearance(
                    shadow.shape_model,
                    perturbed,
                    np.eye(4),
                    current_box,
                    tube_radii,
                    samples_per_segment=SHADOW_SAMPLES_PER_SEGMENT,
                )
                value_capsule = minimum_capsule_clearance(
                    model, data, shadow.envelopes, current_box, spec=spec
                )
                distances_pcc.append(value_pcc.signed_distance_m)
                distances_capsule.append(value_capsule.signed_distance_m)
                source_pcc.append(value_pcc.source_index)
                source_capsule.append(value_capsule.source_name)
            numeric_pcc[coordinate] = (
                distances_pcc[0] - distances_pcc[1]
            ) / (2.0 * finite_difference_step)
            numeric_capsule[coordinate] = (
                distances_capsule[0] - distances_capsule[1]
            ) / (2.0 * finite_difference_step)
            pcc_sources_stable &= all(
                source == center_pcc.source_index for source in source_pcc
            )
            capsule_sources_stable &= all(
                source == center_capsule.source_name for source in source_capsule
            )
        if pcc_sources_stable:
            pcc_gradient_stable += 1
            pcc_gradient_errors.append(
                float(np.max(np.abs(numeric_pcc - center_pcc.planner_gradient)))
            )
        if capsule_sources_stable:
            capsule_gradient_stable += 1
            capsule_gradient_errors.append(
                float(
                    np.max(
                        np.abs(numeric_capsule - center_capsule.planner_gradient)
                    )
                )
            )

    formal_trace_false_safe = 0
    formal_trace_samples = 0
    for trace_path in sorted(Path("v6_lite/output/traces").glob("*.npz")):
        with np.load(trace_path, allow_pickle=False) as trace:
            task_qpos = np.asarray(trace["task_qpos"], dtype=np.float64)
        for state in task_qpos[::25]:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            projection = spec.project_actual_configuration(
                state[qpos_ids[:CONTINUUM_ACTUATED_DOF]]
            )
            comparison = shadow.evaluate(
                data,
                projection.planner_configuration,
                transform_from_free_qpos(state[base_qpos]),
            )
            formal_trace_false_safe += int(
                comparison.pcc_tube.signed_distance_m >= SAFETY_GATE_M
                and comparison.mujoco_geometry.signed_distance_m < SAFETY_GATE_M
            )
            formal_trace_samples += 1

    checks = {
        "target_case_count_at_least_10000": target_case_count >= 10_000,
        "all_required_target_case_types_present": set(category_counts) == set(categories),
        "capsule_false_safe_count_is_zero": capsule_false_safe == 0,
        "pcc_tube_false_safe_count_is_zero": pcc_false_safe == 0,
        "capsule_is_conservative_when_separated": capsule_conservatism_violations == 0,
        "formal_moving_target_trace_false_safe_count_is_zero": formal_trace_false_safe == 0,
        "pcc_gradient_matches_finite_difference": (
            pcc_gradient_stable > 0 and max(pcc_gradient_errors) <= 1e-4
        ),
        "capsule_gradient_matches_finite_difference": (
            capsule_gradient_stable > 0 and max(capsule_gradient_errors) <= 1e-4
        ),
    }
    return {
        "audit_version": AUDIT_VERSION,
        "model_contract_sha256": spec.contract_sha256(),
        "passed": all(checks.values()),
        "checks": checks,
        "safety_gate_m": SAFETY_GATE_M,
        "target_case_count": target_case_count,
        "target_case_categories": dict(category_counts),
        "unsafe_case_count_by_category": dict(category_unsafe),
        "query_truncation": {
            "query_distance_max_m": 0.5,
            "truncated_case_count": query_truncation_count,
            "rule": (
                "a value at distmax is marked truncated and is never described as "
                "an exact clearance"
            ),
        },
        "false_safe": {
            "definition": "proxy >= 5 mm while enveloped MuJoCo geometry < 5 mm",
            "capsule_count": capsule_false_safe,
            "pcc_tube_count": pcc_false_safe,
            "finite_random_zero_is_only_regression_evidence": True,
        },
        "distance_m": {
            "mujoco_geometry": _summary(np.asarray(actual_distances)),
            "capsule_proxy": _summary(np.asarray(capsule_distances)),
            "pcc_tube_proxy": _summary(np.asarray(pcc_distances)),
            "capsule_minus_geometry": _summary(
                np.asarray(capsule_distances) - np.asarray(actual_distances)
            ),
            "pcc_minus_geometry": _summary(
                np.asarray(pcc_distances) - np.asarray(actual_distances)
            ),
        },
        "gradient_audit": {
            "finite_difference_step_rad": finite_difference_step,
            "candidate_case_count": len(gradient_cases),
            "pcc_stable_nearest_feature_case_count": pcc_gradient_stable,
            "capsule_stable_nearest_feature_case_count": capsule_gradient_stable,
            "pcc_gradient_absolute_error": _summary(
                np.asarray(pcc_gradient_errors)
            ),
            "capsule_gradient_absolute_error": _summary(
                np.asarray(capsule_gradient_errors)
            ),
            "nonsmooth_nearest_feature_switches_excluded": True,
        },
        "formal_moving_target_trace_shadow": {
            "sample_count": formal_trace_samples,
            "pcc_false_safe_count": formal_trace_false_safe,
        },
        "claim_boundary": (
            "Shadow-only comparison. The PCC/capsule values do not alter the V6 QP "
            "or torque command. Zero false-safe cases are finite regression evidence, "
            "not a full-state safety certificate."
        ),
    }


def run_audit(
    output_dir: Path,
    *,
    configuration_count: int = DEFAULT_CONFIGURATION_COUNT,
    calibration_count: int = DEFAULT_CALIBRATION_COUNT,
    target_case_count: int = DEFAULT_TARGET_CASE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    start = time.perf_counter()
    robot = default_v6_lite_robot_spec()
    spec = default_continuum_model_spec(robot)
    model = robot.compile_dynamic_model()
    configurations = build_frozen_configurations(spec, configuration_count, seed)
    shape = audit_shape_model(configurations, spec, model, seed)
    geometry, tube_radii = audit_geometry_envelope(
        configurations, calibration_count, spec, model
    )
    comparison = audit_shape_vs_geometry(
        configurations[calibration_count:],
        target_case_count,
        tube_radii,
        spec,
        model,
        seed,
    )
    _write_json(output_dir / "shape_model_audit.json", shape)
    _write_json(output_dir / "geometry_envelope_audit.json", geometry)
    _write_json(output_dir / "shape_vs_geom_comparison.json", comparison)
    files = (
        output_dir / "shape_model_audit.json",
        output_dir / "geometry_envelope_audit.json",
        output_dir / "shape_vs_geom_comparison.json",
    )
    manifest = {
        "audit_version": AUDIT_VERSION,
        "passed": bool(shape["passed"] and geometry["passed"] and comparison["passed"]),
        "seed": seed,
        "model_contract": spec.to_dict(),
        "model_contract_sha256": spec.contract_sha256(),
        "configuration_count": configuration_count,
        "calibration_count": calibration_count,
        "held_out_count": configuration_count - calibration_count,
        "target_case_count": target_case_count,
        "controller_behavior": (
            "unchanged: V6.1-A modules are shadow-only and are not imported by the "
            "V6 QP/run command path"
        ),
        "artifacts": [
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
        "elapsed_wall_time_s": float(time.perf_counter() - start),
    }
    _write_json(output_dir / "v6_1a_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--configuration-count", type=int, default=DEFAULT_CONFIGURATION_COUNT
    )
    parser.add_argument("--calibration-count", type=int, default=DEFAULT_CALIBRATION_COUNT)
    parser.add_argument("--target-case-count", type=int, default=DEFAULT_TARGET_CASE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_audit(
        args.output_dir,
        configuration_count=args.configuration_count,
        calibration_count=args.calibration_count,
        target_case_count=args.target_case_count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "configuration_count": result["configuration_count"],
                "target_case_count": result["target_case_count"],
                "elapsed_wall_time_s": result["elapsed_wall_time_s"],
                "output_dir": args.output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
