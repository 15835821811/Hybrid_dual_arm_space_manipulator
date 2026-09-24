"""Generate V6.1-B PCC-aware CBF audit and comparison artifacts.

The audit deliberately separates three claims:

* differential correctness of the continuous PCC model;
* finite seeded false-safe evidence against the real MuJoCo collision meshes;
* online single-QP timing and command intervention at 50 Hz.

The 10,000-case geometry audit is regression evidence over the declared
``q_c in [-1, 1]^10`` domain.  It is not a global collision certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
)
from v6_lite.audit_v6_1a import (
    _random_rotation,
    _set_target_geom_pose,
    _summary,
    build_frozen_configurations,
)
from v6_lite.continuum_jacobian import compute_position_jacobian
from v6_lite.continuum_model_spec import (
    default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import (
    ContinuumShapeModel,
)
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M,
    HierarchicalQPConfig,
    HierarchicalVelocityQP,
    free_joint_slices,
    joint_addresses,
)
from v6_lite.pcc_clearance import PCCClearanceEvaluator
from v6_lite.pcc_monitor import PCCMonitor
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    OrientedBox,
    build_continuum_capsule_envelopes,
    minimum_capsule_clearance,
    minimum_mujoco_geom_clearance,
    target_box_from_mujoco,
)

AUDIT_VERSION = "v6.1-b.audit.1"
DEFAULT_SEED = 20260924
DEFAULT_CONFIGURATION_COUNT = 10_000
DEFAULT_JACOBIAN_CASE_COUNT = 1_000
DEFAULT_OUTPUT = Path("v6_lite/output/v6_1_b")
SAFETY_GATE_M = 0.005


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, np.ndarray):
            return sanitize(value.tolist())
        if isinstance(value, (float, np.floating)):
            scalar = float(value)
            return scalar if np.isfinite(scalar) else None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    with path.open("w", encoding="utf-8") as stream:
        json.dump(sanitize(payload), stream, ensure_ascii=False, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(label: str, index: int, total: int) -> None:
    stride = max(total // 10, 1)
    if index == 0 or (index + 1) % stride == 0 or index + 1 == total:
        print(f"[{label}] {index + 1}/{total}", flush=True)


def _position_jacobian_audit(
    shape: ContinuumShapeModel,
    case_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    step = 1e-6
    relative_errors = np.zeros(case_count, dtype=np.float64)
    absolute_errors = np.zeros(case_count, dtype=np.float64)
    for index in range(case_count):
        configuration = rng.uniform(-1.0, 1.0, size=10)
        arclength = float(rng.uniform(0.0, shape.spec.total_length_m))
        analytic = compute_position_jacobian(
            configuration, np.eye(4), arclength, model=shape
        )
        numeric = np.zeros((3, 10), dtype=np.float64)
        for coordinate in range(10):
            plus = configuration.copy()
            minus = configuration.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            numeric[:, coordinate] = (
                shape.evaluate(
                    plus, np.eye(4), arclength, with_jacobians=False
                ).position_world
                - shape.evaluate(
                    minus, np.eye(4), arclength, with_jacobians=False
                ).position_world
            ) / (2.0 * step)
        difference = analytic - numeric
        absolute_errors[index] = np.linalg.norm(difference)
        relative_errors[index] = absolute_errors[index] / max(
            float(np.linalg.norm(numeric)), 1e-12
        )
        _progress("position-jacobian", index, case_count)
    return {
        "case_count": case_count,
        "coordinate_checks": 10 * case_count,
        "finite_difference_step_rad": step,
        "relative_error": _summary(relative_errors),
        "absolute_error": _summary(absolute_errors),
        "threshold_relative": 0.05,
        "passed": bool(np.max(relative_errors) < 0.05),
    }


def _distance_gradient_audit(
    evaluator: PCCClearanceEvaluator,
    rng: np.random.Generator,
    required_cases: int = 64,
) -> dict[str, Any]:
    """Check the envelope-theorem gradient only on smooth exterior features."""

    box = OrientedBox(
        center=np.asarray([1.2, 0.326, 0.05]),
        rotation=np.eye(3),
        half_extents=np.asarray([0.2, 0.2, 0.2]),
    )
    step = 1e-6
    errors: list[float] = []
    attempts = 0
    feature_switches = 0
    while len(errors) < required_cases and attempts < 5_000:
        attempts += 1
        configuration = rng.uniform(-0.7, 0.7, size=10)
        center = evaluator.evaluate(configuration, np.eye(4), box)
        if center.distance <= SAFETY_GATE_M:
            continue
        numeric = np.zeros(10, dtype=np.float64)
        stable = True
        for coordinate in range(10):
            plus = configuration.copy()
            minus = configuration.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            value_plus = evaluator.evaluate(plus, np.eye(4), box)
            value_minus = evaluator.evaluate(minus, np.eye(4), box)
            stable &= (
                value_plus.segment_id == center.segment_id
                and value_minus.segment_id == center.segment_id
                and abs(value_plus.arc_length - center.arc_length) < 1e-3
                and abs(value_minus.arc_length - center.arc_length) < 1e-3
                and np.dot(value_plus.normal, center.normal) > 0.999
                and np.dot(value_minus.normal, center.normal) > 0.999
            )
            numeric[coordinate] = (
                value_plus.distance - value_minus.distance
            ) / (2.0 * step)
        if not stable:
            feature_switches += 1
            continue
        denominator = max(float(np.linalg.norm(numeric)), 1e-10)
        errors.append(float(np.linalg.norm(center.gradient - numeric) / denominator))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "attempt_count": attempts,
        "stable_smooth_case_count": len(errors),
        "excluded_feature_switch_count": feature_switches,
        "finite_difference_step_rad": step,
        "relative_error": _summary(values),
        "threshold_relative": 0.05,
        "nonsmooth_collision_and_feature_switches_excluded": True,
        "passed": bool(
            len(errors) >= required_cases
            and values.size > 0
            and np.max(values) < 0.05
        ),
    }


def _target_geometry_ids(
    model: mujoco.MjModel,
) -> tuple[int, tuple[int, ...]]:
    target_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
        )
    )
    if target_id < 0:
        raise ValueError("target satellite collision box is missing")
    envelopes = build_continuum_capsule_envelopes(model)
    return target_id, tuple(item.geom_id for item in envelopes.capsules)


def _false_safe_audit(
    configurations: np.ndarray,
    model: mujoco.MjModel,
    evaluator: PCCClearanceEvaluator,
    rng: np.random.Generator,
) -> dict[str, Any]:
    robot = default_v6_lite_robot_spec()
    spec = evaluator.shape_model.spec
    data = mujoco.MjData(model)
    qpos_ids, _ = joint_addresses(model, robot)
    base_qpos, _ = free_joint_slices(model, robot.base_joint_name)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    target_geom_id, enveloped_geom_ids = _target_geometry_ids(model)
    envelopes = build_continuum_capsule_envelopes(model, spec)
    categories = ("face", "edge", "corner", "rotated", "mid_arm", "penetration")
    category_counts: Counter[str] = Counter()
    unsafe_counts: Counter[str] = Counter()
    pcc_false_safe = 0
    capsule_false_safe = 0
    pcc_distance: list[float] = []
    capsule_distance: list[float] = []
    mujoco_distance: list[float] = []
    query_latency: list[float] = []
    middle_capsules = [
        item
        for item in envelopes.capsules
        if item.segment_index in (1, 2, 3) and item.body_name.startswith("link_")
    ]

    for case_index, configuration in enumerate(configurations):
        data.qpos[:] = model.qpos0
        planner = robot.planner_zero.copy()
        planner[:10] = configuration
        data.qpos[qpos_ids] = robot.encode_position(planner)
        data.qpos[base_qpos] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        data.qpos[target_qpos] = [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        category = categories[case_index % len(categories)]
        category_counts[category] += 1
        capsule = (
            middle_capsules[case_index % len(middle_capsules)]
            if category == "mid_arm"
            else envelopes.capsules[int(rng.integers(0, len(envelopes.capsules)))]
        )
        body_rotation = np.asarray(data.xmat[capsule.body_id]).reshape(3, 3)
        body_position = np.asarray(data.xpos[capsule.body_id])
        arm_point = body_position + body_rotation @ (
            0.5 * (capsule.local_start + capsule.local_end)
        )
        target_rotation = (
            np.eye(3)
            if category in ("face", "edge", "corner", "penetration")
            else _random_rotation(rng)
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
        box = target_box_from_mujoco(model, data, target_geom_id)
        started = time.perf_counter()
        pcc = evaluator.evaluate(configuration, np.eye(4), box)
        query_latency.append(time.perf_counter() - started)
        capsule_value = minimum_capsule_clearance(
            model,
            data,
            envelopes,
            box,
            spec=spec,
            compute_planner_gradient=False,
        )
        exact = minimum_mujoco_geom_clearance(
            model,
            data,
            enveloped_geom_ids,
            target_geom_id,
            query_distance_max_m=0.5,
        )
        pcc_distance.append(pcc.distance)
        capsule_distance.append(capsule_value.signed_distance_m)
        mujoco_distance.append(exact.signed_distance_m)
        unsafe = exact.signed_distance_m < SAFETY_GATE_M
        unsafe_counts[category] += int(unsafe)
        pcc_false_safe += int(pcc.distance >= SAFETY_GATE_M and unsafe)
        capsule_false_safe += int(
            capsule_value.signed_distance_m >= SAFETY_GATE_M and unsafe
        )
        _progress("false-safe", case_index, len(configurations))

    pcc_array = np.asarray(pcc_distance)
    capsule_array = np.asarray(capsule_distance)
    exact_array = np.asarray(mujoco_distance)
    return {
        "case_count": len(configurations),
        "categories": dict(category_counts),
        "unsafe_case_count_by_category": dict(unsafe_counts),
        "safety_gate_m": SAFETY_GATE_M,
        "pcc_count": pcc_false_safe,
        "capsule_count": capsule_false_safe,
        "definition": "proxy >= 5 mm while enveloped MuJoCo geometry < 5 mm",
        "distance_m": {
            "pcc": _summary(pcc_array),
            "capsule": _summary(capsule_array),
            "mujoco": _summary(exact_array),
            "pcc_minus_mujoco": _summary(pcc_array - exact_array),
            "capsule_minus_mujoco": _summary(capsule_array - exact_array),
        },
        "pcc_query_latency_ms": _summary(1e3 * np.asarray(query_latency)),
        "passed": pcc_false_safe == 0 and capsule_false_safe == 0,
    }


def _task_arguments(robot: Any, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    rigid_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot.rigid_tip_body_name
        )
    )
    continuum_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot.continuum_tip_body_name
        )
    )
    rigid_rotation = np.asarray(data.xmat[rigid_id]).reshape(3, 3).copy()
    continuum_rotation = np.asarray(data.xmat[continuum_id]).reshape(3, 3).copy()
    continuum_position = (
        np.asarray(data.xpos[continuum_id]).copy()
        + continuum_rotation @ CONTINUUM_EE_OFFSET_M
    )
    return {
        "rigid_target_position": np.asarray(data.xpos[rigid_id]).copy(),
        "rigid_target_velocity": np.zeros(3),
        "rigid_target_rotation": rigid_rotation,
        "rigid_target_angular_velocity": np.zeros(3),
        "continuum_target_position": continuum_position + [0.0, -0.15, 0.0],
        "continuum_target_velocity": np.zeros(3),
        "continuum_target_rotation": continuum_rotation,
        "continuum_target_angular_velocity": np.zeros(3),
    }


def _online_qp_audit(output_dir: Path) -> dict[str, Any]:
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        robot,
        (),
        WholeBodyVerificationConfig(
            include_target_satellite_pairs=True,
            adaptive_subdivisions=2,
        ),
    )
    model = verifier.model
    data = mujoco.MjData(model)
    qpos_ids, _ = joint_addresses(model, robot)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    data.qpos[qpos_ids] = robot.encode_position(robot.planner_zero)
    data.qpos[target_qpos.start : target_qpos.start + 3] = [1.2, 0.326, 0.05]
    data.qpos[target_qpos.start + 3 : target_qpos.stop] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    arguments = _task_arguments(robot, model, data)
    baseline_qp = HierarchicalVelocityQP(
        robot, model, verifier.pairs, HierarchicalQPConfig()
    )
    enabled_qp = HierarchicalVelocityQP(
        robot,
        model,
        verifier.pairs,
        HierarchicalQPConfig(enable_pcc_cbf=True, enable_capsule_cbf=True),
    )
    baseline = baseline_qp.solve(data, **arguments)
    monitor = PCCMonitor(scenario_id="v6_1_b_online_ab")
    results = []
    for index in range(200):
        result = enabled_qp.solve(data, **arguments)
        results.append(result)
        monitor.record(0.02 * index, result)
    comparison = monitor.write(output_dir)
    latencies = np.asarray([item.full_latency_s for item in results])
    successes = sum(item.success for item in results)
    command_delta = float(
        np.linalg.norm(results[0].planner_velocity - baseline.planner_velocity)
    )
    return {
        "sample_count": len(results),
        "success_count": int(successes),
        "qp_failure_count": int(len(results) - successes),
        "task_full_latency_ms": _summary(1e3 * latencies),
        "shape_clearance_latency_ms": _summary(
            1e3 * np.asarray([item.shape_clearance_latency_s for item in results])
        ),
        "pcc_constraint_active_count": int(
            sum(item.pcc_constraint_active_count for item in results)
        ),
        "pcc_constraint_binding_count": int(
            sum(item.pcc_binding_constraint_count for item in results)
        ),
        "capsule_constraint_active_count": int(
            sum(item.capsule_constraint_active_count for item in results)
        ),
        "pcc_avoidance_intervention_max": float(
            max(item.pcc_avoidance_intervention for item in results)
        ),
        "enabled_minus_baseline_command_norm": command_delta,
        "mujoco_continuum_target_clearance_m": float(
            min(item.mujoco_continuum_target_clearance_m for item in results)
        ),
        "comparison_sample_count": comparison["sample_count"],
        "diagnostic_latency_p95_below_20_ms": bool(
            np.percentile(latencies, 95.0) < 0.020
        ),
        "latency_gate_source": (
            "diagnostic only; the formal <20 ms gate uses the 6,750-tick "
            "five-scenario torque closed loop in regression_report.json"
        ),
        "passed": bool(
            successes == len(results)
            and sum(item.pcc_constraint_active_count for item in results) > 0
            and max(item.pcc_avoidance_intervention for item in results) > 1e-3
            and command_delta > 1e-2
            and min(
                item.mujoco_continuum_target_clearance_m for item in results
            )
            >= SAFETY_GATE_M
        ),
    }


def run_audit(
    output_dir: Path,
    *,
    configuration_count: int = DEFAULT_CONFIGURATION_COUNT,
    jacobian_case_count: int = DEFAULT_JACOBIAN_CASE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if configuration_count < 10_000:
        raise ValueError("formal V6.1-B audit requires at least 10,000 cases")
    if jacobian_case_count < 1_000:
        raise ValueError("formal V6.1-B audit requires at least 1,000 Jacobian cases")
    started = time.perf_counter()
    robot = default_v6_lite_robot_spec()
    spec = default_continuum_model_spec(robot)
    model = robot.compile_dynamic_model()
    shape = ContinuumShapeModel(spec)
    online_config = HierarchicalQPConfig()
    evaluator = PCCClearanceEvaluator(
        shape,
        coarse_samples_per_segment=online_config.pcc_coarse_samples_per_segment,
        active_segment_count=online_config.pcc_active_segment_count,
        refinement_max_iterations=online_config.pcc_refinement_max_iterations,
    )
    rng = np.random.default_rng(seed)
    jacobian = _position_jacobian_audit(shape, jacobian_case_count, rng)
    distance_gradient = _distance_gradient_audit(evaluator, rng)
    configurations = build_frozen_configurations(
        spec, configuration_count, seed + 101
    )
    false_safe = _false_safe_audit(
        configurations, model, evaluator, np.random.default_rng(seed + 202)
    )
    online = _online_qp_audit(output_dir)
    checks = {
        "configuration_count_at_least_10000": configuration_count >= 10_000,
        "jacobian_case_count_at_least_1000": jacobian_case_count >= 1_000,
        "position_jacobian_relative_error_below_5_percent": jacobian["passed"],
        "distance_gradient_relative_error_below_5_percent": distance_gradient[
            "passed"
        ],
        "pcc_false_safe_count_is_zero": false_safe["pcc_count"] == 0,
        "capsule_false_safe_count_is_zero": false_safe["capsule_count"] == 0,
        "single_qp_online_control_passes": online["passed"],
    }
    payload = {
        "audit_version": AUDIT_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "seed": seed,
        "configuration_count": configuration_count,
        "jacobian_case_count": jacobian_case_count,
        "model_contract_sha256": spec.contract_sha256(),
        "online_pcc_configuration": {
            "coarse_samples_per_segment": evaluator.coarse_samples_per_segment,
            "active_segment_count": evaluator.active_segment_count,
            "refinement_max_iterations": evaluator.refinement_max_iterations,
        },
        "position_jacobian": jacobian,
        "distance_gradient": distance_gradient,
        "false_safe": false_safe,
        "online_qp_ab": online,
        "claim_boundary": (
            "Finite seeded regression evidence over q_c in [-1,1]^10 and six "
            "target-pose categories; not a global continuous-time certificate."
        ),
        "elapsed_wall_time_s": float(time.perf_counter() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "pcc_audit.json"
    _write_json(audit_path, payload)
    manifest = {
        "audit_version": AUDIT_VERSION,
        "passed": payload["passed"],
        "artifacts": [
            {
                "path": audit_path.as_posix(),
                "sha256": _sha256(audit_path),
                "bytes": audit_path.stat().st_size,
            }
        ],
    }
    _write_json(output_dir / "audit_manifest.json", manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--configuration-count", type=int, default=DEFAULT_CONFIGURATION_COUNT
    )
    parser.add_argument(
        "--jacobian-case-count", type=int, default=DEFAULT_JACOBIAN_CASE_COUNT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_audit(
        args.output_dir,
        configuration_count=args.configuration_count,
        jacobian_case_count=args.jacobian_case_count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "configuration_count": result["configuration_count"],
                "jacobian_case_count": result["jacobian_case_count"],
                "false_safe": {
                    "pcc": result["false_safe"]["pcc_count"],
                    "capsule": result["false_safe"]["capsule_count"],
                },
                "elapsed_wall_time_s": result["elapsed_wall_time_s"],
                "output": (args.output_dir / "pcc_audit.json").as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
