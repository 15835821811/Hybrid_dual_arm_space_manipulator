"""Challenge capsule bound screening with target boxes on and near arm axes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_batched_capsule_bounds import BatchedCapsuleMidpointBounds
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, default_v6_lite_robot_spec, free_joint_slices,
)
from v6_lite.shape_clearance import (
    build_continuum_capsule_envelopes, minimum_capsule_clearance,
    target_box_from_mujoco,
)


TICKS = (0, 225, 450, 675, 900, 1125, 1350)
OFFSETS_M = (
    (0.0, 0.0, 0.0), (0.005, 0.0, 0.0), (-0.005, 0.0, 0.0),
    (0.0, 0.025, 0.0), (0.0, -0.025, 0.0),
    (0.0, 0.080, 0.0), (0.0, -0.080, 0.0),
    (0.0, 0.120, 0.0), (0.0, -0.120, 0.0),
)
NEAR_GATE_DIRECTIONS = ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
NEAR_GATE_DELTA_M = (-0.002, 0.0, 0.002)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, private_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = V6LiteRunConfig(**metrics["run_config"])
    saved = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == "v6_lite_scenario_01")
    robot = default_v6_lite_robot_spec()
    shape_spec = default_continuum_model_spec(robot)
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=run_config.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=run_config.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    envelopes = build_continuum_capsule_envelopes(model, shape_spec)
    batched = BatchedCapsuleMidpointBounds(envelopes)
    target_geom_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"))
    target_slice, _ = free_joint_slices(model, robot.target_free_joint_name)
    trace_path = (private_root / "domain_endpoint_cholesky_scene01_1350"
                  / "private_rollout_trace.npz")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos_states = trace["qpos_states"][::10].copy()
        qvel_states = trace["task_qvel_states"].copy()
    if len(qpos_states) != 1351 or len(qvel_states) != 1351:
        raise ValueError("private state trace is incomplete")
    rows = []
    capsule_indices = (0, len(envelopes.capsules) // 2,
                       len(envelopes.capsules) - 1)
    for tick in TICKS:
        original_qpos = qpos_states[tick]
        original_qvel = qvel_states[tick]
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)
        original_box_center = np.asarray(data.geom_xpos[target_geom_id]).copy()
        axis_midpoints = []
        for capsule_index in capsule_indices:
            capsule = envelopes.capsules[capsule_index]
            rotation = np.asarray(data.xmat[capsule.body_id]).reshape(3, 3)
            position = np.asarray(data.xpos[capsule.body_id])
            axis_midpoints.append(position + rotation @ (
                0.5 * (capsule.local_start + capsule.local_end)))
        for capsule_index, midpoint in zip(capsule_indices, axis_midpoints):
            def evaluate_target(offset, *, compare: bool, family: str):
                data.qpos[:] = original_qpos
                data.qvel[:] = original_qvel
                desired_center = midpoint + np.asarray(offset)
                data.qpos[target_slice.start:target_slice.start + 3] += (
                    desired_center - original_box_center)
                mujoco.mj_forward(model, data)
                box = target_box_from_mujoco(model, data, target_geom_id)
                reference = minimum_capsule_clearance(
                    model, data, envelopes, box, spec=shape_spec,
                    compute_planner_gradient=False)
                if not compare:
                    return reference.signed_distance_m
                trial = batched.minimum(model, data, box)
                distance_error = abs(reference.signed_distance_m
                                     - trial.signed_distance_m)
                witness_error = max(
                    float(np.max(np.abs(reference.point_on_arm - trial.point_on_arm))),
                    float(np.max(np.abs(reference.point_on_box - trial.point_on_box))),
                    float(np.max(np.abs(reference.normal_box_to_arm
                                        - trial.normal_box_to_arm))),
                )
                row = {
                    "tick": tick,
                    "family": family,
                    "targeted_capsule_index": capsule_index,
                    "target_center_offset_m": np.asarray(offset).tolist(),
                    "reference_source": reference.source_name,
                    "trial_source": trial.source_name,
                    "signed_clearance_m": reference.signed_distance_m,
                    "distance_error_m": distance_error,
                    "witness_error": witness_error,
                }
                rows.append(row)
                if (reference.source_name != trial.source_name
                        or distance_error > 1e-12 or witness_error > 1e-12):
                    raise ValueError(f"batched minimum differs at {tick}/"
                                     f"{capsule_index}/{offset}")
                return reference.signed_distance_m

            for offset in OFFSETS_M:
                evaluate_target(offset, compare=True, family="fixed_inside")
            for direction in NEAR_GATE_DIRECTIONS:
                direction_vector = np.asarray(direction)
                low, high = 0.0, 1.5
                if evaluate_target(high * direction_vector, compare=False,
                                   family="bracket") <= 0.005:
                    raise ValueError("near-gate target ray did not leave the arm")
                for _ in range(26):
                    midpoint_radius = 0.5 * (low + high)
                    clearance = evaluate_target(
                        midpoint_radius * direction_vector,
                        compare=False, family="bracket")
                    if clearance < 0.005:
                        low = midpoint_radius
                    else:
                        high = midpoint_radius
                gate_radius = 0.5 * (low + high)
                for delta in NEAR_GATE_DELTA_M:
                    evaluate_target((gate_radius + delta) * direction_vector,
                                    compare=True, family="near_five_mm")
    rows_path = output_dir / "adversarial_capsule_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    distances = [row["signed_clearance_m"] for row in rows]
    report = {
        "schema": "v6_2_b2_batched_capsule_adversarial_v1",
        "scope": "target_translation_to_capsule_axis_no_controller_execution",
        "predeclared_ticks": list(TICKS),
        "predeclared_offsets_m": OFFSETS_M,
        "near_gate_directions": NEAR_GATE_DIRECTIONS,
        "near_gate_delta_m": NEAR_GATE_DELTA_M,
        "case_count": len(rows),
        "near_gate_case_count": sum(row["family"] == "near_five_mm"
                                    for row in rows),
        "negative_clearance_count": sum(value < 0 for value in distances),
        "near_five_mm_count": sum(abs(value - 0.005) <= 0.005
                                  for value in distances),
        "minimum_signed_clearance_m": min(distances),
        "maximum_signed_clearance_m": max(distances),
        "source_mismatch_count": 0,
        "maximum_distance_error_m": max(row["distance_error_m"] for row in rows),
        "maximum_witness_error": max(row["witness_error"] for row in rows),
        "continuous_time_certified": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_batched_capsule_adversarial.py",
                              "b2_batched_capsule_bounds.py",
                              "shape_clearance.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path),
                           "private_trace": _sha(trace_path)},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "adversarial_capsule_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--private-root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.a1_root, args.private_root),
                     indent=2))


if __name__ == "__main__":
    main()
