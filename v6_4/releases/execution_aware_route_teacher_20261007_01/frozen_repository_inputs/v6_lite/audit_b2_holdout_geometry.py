"""Add independent MuJoCo geometry comparison for B.1's frozen holdout."""

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


DEFAULT_INPUT = Path(
    "v6_lite/output/v6_2_b1/formal_audit_r02/independent_heldout_inputs.npz"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_holdout(input_path: Path, output_dir: Path) -> dict:
    input_path = Path(input_path)
    source_manifest = json.loads(
        (input_path.parent / "audit_manifest.json").read_text(encoding="utf-8")
    )
    entry = next(x for x in source_manifest["artifacts"]
                 if x["path"] == input_path.name)
    input_hash = _sha(input_path)
    if input_hash != entry["sha256"] or input_path.stat().st_size != entry["bytes"]:
        raise ValueError("frozen B.1 holdout hash or byte size mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    model = robot.compile_dynamic_model()
    data = mujoco.MjData(model)
    qpos_ids, _ = joint_addresses(model, robot)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    target_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
    ))
    shadow = ShapeClearanceShadow(
        model, target_id, V61A_PCC_TUBE_RADII_M,
    )
    bounded = PCCBoundedClearanceEvaluator(shadow.shape_model)
    counts: Counter[str] = Counter()
    near_cases = []
    with np.load(input_path, allow_pickle=False) as frozen:
        q_all = frozen["configurations"]
        centers, rotations = frozen["centers"], frozen["rotations"]
        extents = frozen["half_extents"]
        for index, q in enumerate(q_all):
            model.geom_size[target_id] = extents[index]
            data.qpos[:] = model.qpos0
            planner = robot.planner_zero.copy()
            planner[:10] = q
            data.qpos[qpos_ids] = robot.encode_position(planner)
            _set_target_geom_pose(
                model, data, target_qpos, target_id,
                centers[index], rotations[index],
            )
            mujoco.mj_forward(model, data)
            box = OrientedBox(centers[index], rotations[index], extents[index])
            result = bounded.evaluate(q, np.eye(4), box, max_evaluations=31,
                                      safety_gate_m=0.005)
            actual = minimum_mujoco_geom_clearance(
                model, data, shadow.enveloped_geom_ids, target_id,
                query_distance_max_m=0.5,
            )
            counts["checked_count"] += 1
            counts[result.proxy_clearance_status] += 1
            counts["bounds_invalid"] += int(not result.bounds_valid)
            counts["actual_geometry_safe"] += int(actual.signed_distance_m >= 0.005)
            counts["actual_geometry_below_gate"] += int(actual.signed_distance_m < 0.005)
            counts["empirical_proxy_false_safe"] += int(
                result.proxy_clearance_status == "PROXY_CLEARANCE_AT_LEAST_GATE"
                and actual.signed_distance_m < 0.005
            )
            counts["empirical_proxy_false_reject"] += int(
                result.proxy_clearance_status != "PROXY_CLEARANCE_AT_LEAST_GATE"
                and actual.signed_distance_m >= 0.005
            )
            if len(near_cases) < 16 and abs(result.distance_upper_bound_m - 0.005) <= 0.015:
                near_cases.append({
                    "index": index,
                    "lower_m": result.distance_lower_bound_m,
                    "upper_m": result.distance_upper_bound_m,
                    "actual_mujoco_distance_m": actual.signed_distance_m,
                    "proxy_status": result.proxy_clearance_status,
                })
    report = {
        "schema": "v6_2_b2_existing_heldout_mujoco_comparison_v1",
        "frozen_input_sha256": input_hash,
        "frozen_input_count": len(q_all),
        "target_obbs_from_frozen_inputs": True,
        "mujoco_target_geom_size_changed_only_in_offline_model": True,
        "query_max_evaluations": 31,
        "proxy_gate_m": 0.005,
        "mujoco_geometry_query_max_m": 0.5,
        "counts": dict(counts),
        "near_gate_examples": near_cases,
        "checks": {
            "all_1024_frozen_inputs_compared": counts["checked_count"] == 1024,
            "all_queries_well_formed": counts["bounds_invalid"] == 0,
            "empirical_false_safe_zero": counts["empirical_proxy_false_safe"] == 0,
        },
    }
    report["passed"] = all(report["checks"].values())
    report_path = output_dir / "heldout_geometry_report.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema": "v6_2_b2_holdout_geometry_manifest_v1",
        "frozen_input_sha256": input_hash,
        "report_sha256": _sha(report_path),
        "report_bytes": report_path.stat().st_size,
    }
    with (output_dir / "heldout_geometry_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_holdout(args.input, args.output_dir)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "counts": report["counts"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
