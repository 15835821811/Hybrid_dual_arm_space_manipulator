"""Freeze and audit independent PCC cases deliberately near the 5 mm gate.

Placement uses B.1 bounds to *sample* proxy near-gate states, not as ground
truth. Once written, the input NPZ is immutable. MuJoCo comparison is separate
and does not influence which samples were admitted to the holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from v6_lite.audit_v6_1a import _set_target_geom_pose
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    OrientedBox, ShapeClearanceShadow, minimum_mujoco_geom_clearance,
)


NEAR_GATE_SEED = 20261001
GATE_M = 0.005


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()),
            "p50": float(np.quantile(array, 0.5)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(array.max())}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_near_gate(output_dir: Path, *, count: int = 128) -> Path:
    if count < 2 or count % 2:
        raise ValueError("near-gate count must be positive and even")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    model = robot.compile_dynamic_model()
    target_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
    ))
    extents = np.asarray(model.geom_size[target_id], dtype=np.float64).copy()
    bounded = PCCBoundedClearanceEvaluator()
    shape = bounded.shape_model
    spec = shape.spec
    rng = np.random.default_rng(NEAR_GATE_SEED)
    configurations, centers, rotations, desired_values = [], [], [], []
    selection_bounds = []
    side_counts = Counter()
    attempts = 0
    while len(configurations) < count and attempts < count * 100:
        attempts += 1
        desired = 0.003 if side_counts["below"] <= side_counts["above"] else 0.007
        side = "below" if desired < GATE_M else "above"
        q = rng.uniform(-0.7, 0.7, size=10)
        segment = int(rng.integers(1, 5))
        local = float(rng.uniform(0.2, 0.8) * spec.segment_lengths_m[segment])
        s = float(spec.segment_boundaries_m[segment] + local)
        point = shape.evaluate(q, np.eye(4), s, with_jacobians=False)
        axis = int(rng.integers(1, 3))
        normal = point.rotation_world[:, axis].copy()
        if rng.random() < 0.5:
            normal *= -1
        orientation = point.rotation_world.copy()
        center = (point.position_world
                  + normal * (extents[axis] + V61A_PCC_TUBE_RADII_M[segment] + desired))
        latest = None
        for _ in range(5):
            box = OrientedBox(center, orientation, extents)
            latest = bounded.evaluate(q, np.eye(4), box,
                                      max_evaluations=255,
                                      tolerance_m=0.0005,
                                      safety_gate_m=GATE_M)
            estimate = 0.5 * (latest.distance_lower_bound_m
                              + latest.distance_upper_bound_m)
            if abs(estimate - desired) <= 0.0008 and latest.bound_gap_m <= 0.0015:
                break
            center = center + normal * float(np.clip(desired - estimate,
                                                     -0.03, 0.03))
        assert latest is not None
        estimate = 0.5 * (latest.distance_lower_bound_m
                          + latest.distance_upper_bound_m)
        if (abs(estimate - desired) > 0.0008
                or latest.bound_gap_m > 0.0015
                or not latest.bounds_valid):
            continue
        configurations.append(q)
        centers.append(center)
        rotations.append(orientation)
        desired_values.append(desired)
        selection_bounds.append([latest.distance_lower_bound_m,
                                 latest.distance_upper_bound_m])
        side_counts[side] += 1
        if len(configurations) % 32 == 0:
            print(f"[b2-near] frozen {len(configurations)}/{count}", flush=True)
    if len(configurations) != count:
        failure = {"schema": "v6_2_b2_near_gate_generation_failure_v1",
                   "requested_count": count,
                   "generated_count": len(configurations),
                   "attempts": attempts, "side_counts": dict(side_counts)}
        (output_dir / "generation_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise RuntimeError("could not freeze enough near-gate cases")
    path = output_dir / "near_gate_inputs.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, configurations=np.asarray(configurations),
            centers=np.asarray(centers), rotations=np.asarray(rotations),
            half_extents=np.tile(extents, (count, 1)),
            desired_proxy_distances_m=np.asarray(desired_values),
            selection_bounds_m=np.asarray(selection_bounds),
            seed=np.asarray(NEAR_GATE_SEED),
        )
    metadata = {
        "schema": "v6_2_b2_near_gate_frozen_inputs_v1",
        "seed": NEAR_GATE_SEED,
        "count": count,
        "below_gate_target_count": side_counts["below"],
        "above_gate_target_count": side_counts["above"],
        "generator_attempt_count": attempts,
        "selection_method": "B.1 bounds used only to place frozen inputs near gate; not continuous truth",
        "generator_source_sha256": _sha(Path(__file__)),
        "input_sha256": _sha(path),
        "input_bytes": path.stat().st_size,
    }
    with (output_dir / "frozen_inputs_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def audit_near_gate(output_dir: Path) -> dict:
    input_path = output_dir / "near_gate_inputs.npz"
    metadata = json.loads((output_dir / "frozen_inputs_manifest.json").read_text(encoding="utf-8"))
    if _sha(input_path) != metadata["input_sha256"]:
        raise ValueError("frozen near-gate input hash changed")
    robot = default_v6_lite_robot_spec()
    model = robot.compile_dynamic_model()
    data = mujoco.MjData(model)
    qpos_ids, _ = joint_addresses(model, robot)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    target_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
    ))
    shadow = ShapeClearanceShadow(model, target_id, V61A_PCC_TUBE_RADII_M)
    bounded = PCCBoundedClearanceEvaluator(shadow.shape_model)
    counts = Counter()
    by_side = {"below": Counter(), "above": Counter()}
    unknown_examples = []
    gaps: list[float] = []
    times: list[float] = []
    with np.load(input_path, allow_pickle=False) as frozen:
        q_all = frozen["configurations"]
        for index, q in enumerate(q_all):
            center = frozen["centers"][index]
            rotation = frozen["rotations"][index]
            extents = frozen["half_extents"][index]
            desired = float(frozen["desired_proxy_distances_m"][index])
            side = "below" if desired < GATE_M else "above"
            model.geom_size[target_id] = extents
            data.qpos[:] = model.qpos0
            planner = robot.planner_zero.copy()
            planner[:10] = q
            data.qpos[qpos_ids] = robot.encode_position(planner)
            _set_target_geom_pose(model, data, target_qpos, target_id,
                                  center, rotation)
            mujoco.mj_forward(model, data)
            box = OrientedBox(center, rotation, extents)
            query = bounded.evaluate(q, np.eye(4), box, max_evaluations=255,
                                     tolerance_m=0.0005,
                                     safety_gate_m=GATE_M)
            actual = minimum_mujoco_geom_clearance(
                model, data, shadow.enveloped_geom_ids, target_id,
                query_distance_max_m=0.5,
            )
            counts["checked_count"] += 1
            counts["bounds_invalid"] += int(not query.bounds_valid)
            counts[query.proxy_clearance_status] += 1
            gaps.append(query.bound_gap_m)
            times.append(query.elapsed_ms)
            by_side[side][query.proxy_clearance_status] += 1
            safe_actual = actual.signed_distance_m >= GATE_M
            counts["actual_geometry_safe"] += int(safe_actual)
            counts["empirical_proxy_false_safe"] += int(
                query.proxy_clearance_status == "PROXY_CLEARANCE_AT_LEAST_GATE"
                and not safe_actual
            )
            counts["empirical_proxy_false_reject"] += int(
                query.proxy_clearance_status != "PROXY_CLEARANCE_AT_LEAST_GATE"
                and safe_actual
            )
            if query.proxy_clearance_status == "UNKNOWN_CROSSES_GATE" and len(unknown_examples) < 16:
                unknown_examples.append({
                    "index": index, "desired_m": desired,
                    "lower_m": query.distance_lower_bound_m,
                    "upper_m": query.distance_upper_bound_m,
                    "actual_mujoco_distance_m": actual.signed_distance_m,
                })
    report = {
        "schema": "v6_2_b2_near_gate_holdout_audit_v1",
        "frozen_input_sha256": metadata["input_sha256"],
        "gate_m": GATE_M,
        "counts": dict(counts),
        "bound_gap_m": _summary(gaps),
        "query_elapsed_ms": _summary(times),
        "by_generation_side": {k: dict(v) for k, v in by_side.items()},
        "unknown_examples": unknown_examples,
        "mujoco_comparator": "actual discrete-chain MuJoCo geomDistance at frozen state",
        "checks": {
            "all_frozen_inputs_compared": counts["checked_count"] == metadata["count"],
            "all_bounds_well_formed": counts["bounds_invalid"] == 0,
            "empirical_false_safe_zero": counts["empirical_proxy_false_safe"] == 0,
        },
    }
    report["passed"] = all(report["checks"].values())
    path = output_dir / "near_gate_audit.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": "v6_2_b2_near_gate_manifest_v1",
                "artifacts": [{"path": x.name, "sha256": _sha(x),
                               "bytes": x.stat().st_size}
                              for x in (input_path,
                                        output_dir / "frozen_inputs_manifest.json", path)]}
    with (output_dir / "near_gate_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    freeze_near_gate(args.output_dir, count=args.count)
    report = audit_near_gate(args.output_dir)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "counts": report["counts"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
