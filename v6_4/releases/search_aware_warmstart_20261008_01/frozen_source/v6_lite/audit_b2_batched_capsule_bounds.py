"""Compare batched capsule screening with the original exact minimum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_batched_capsule_bounds import BatchedCapsuleMidpointBounds
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    build_continuum_capsule_envelopes, minimum_capsule_clearance,
    target_box_from_mujoco,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values)
    return {"count": len(values),
            "p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)),
            "max_ms": float(np.max(data))}


def run(output_dir: Path, private_root: Path, a1_root: Path,
        scene_indices: tuple[int, ...]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    shape_spec = default_continuum_model_spec(robot)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = V6LiteRunConfig(**metrics["run_config"])
    frozen = {item["scenario"]["scenario_id"]: item
              for item in metrics["scenarios"]}
    rows = []
    sources = {}
    for scene_index in scene_indices:
        scene_id = f"v6_lite_scenario_{scene_index:02d}"
        saved = frozen[scene_id]
        trace_path = private_root / f"domain_endpoint_cholesky_scene{scene_index:02d}_1350" / "private_rollout_trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            qpos_states = trace["qpos_states"][::10].copy()
            qvel_states = trace["task_qvel_states"].copy()
        if len(qpos_states) != 1351 or len(qvel_states) != 1351:
            raise ValueError(f"private scene {scene_index} is incomplete")
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
        if target_geom_id < 0:
            raise ValueError("target OBB geom is absent")
        for tick, (qpos, qvel) in enumerate(zip(qpos_states, qvel_states)):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            box = target_box_from_mujoco(model, data, target_geom_id)
            if tick % 2:
                start = time.perf_counter()
                trial = batched.minimum(model, data, box)
                trial_ms = 1000.0 * (time.perf_counter() - start)
                start = time.perf_counter()
                reference = minimum_capsule_clearance(
                    model, data, envelopes, box,
                    spec=shape_spec, compute_planner_gradient=False)
                reference_ms = 1000.0 * (time.perf_counter() - start)
            else:
                start = time.perf_counter()
                reference = minimum_capsule_clearance(
                    model, data, envelopes, box,
                    spec=shape_spec, compute_planner_gradient=False)
                reference_ms = 1000.0 * (time.perf_counter() - start)
                start = time.perf_counter()
                trial = batched.minimum(model, data, box)
                trial_ms = 1000.0 * (time.perf_counter() - start)
            distance_error = abs(reference.signed_distance_m
                                 - trial.signed_distance_m)
            witness_error = max(
                float(np.max(np.abs(reference.point_on_arm - trial.point_on_arm))),
                float(np.max(np.abs(reference.point_on_box - trial.point_on_box))),
                float(np.max(np.abs(reference.normal_box_to_arm
                                    - trial.normal_box_to_arm))),
            )
            row = {"scenario_id": scene_id, "tick": tick,
                   "source_name": reference.source_name,
                   "trial_source_name": trial.source_name,
                   "distance_error_m": distance_error,
                   "witness_error": witness_error,
                   "reference_ms": reference_ms,
                   "batched_ms": trial_ms}
            rows.append(row)
            if (reference.source_name != trial.source_name
                    or distance_error > 1e-12 or witness_error > 1e-12):
                raise ValueError(f"capsule minimum changed at {scene_id}/{tick}")
        sources[scene_id] = {"trace_sha256": _sha(trace_path),
                             "states": len(qpos_states),
                             "capsule_count": len(envelopes.capsules)}

    rows_path = output_dir / "batched_capsule_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_batched_capsule_frozen_states_v1",
        "scope": "read_only_saved_private_50hz_states",
        "scene_indices": list(scene_indices),
        "state_count": len(rows),
        "source_mismatch_count": 0,
        "maximum_distance_error_m": max(row["distance_error_m"] for row in rows),
        "maximum_witness_error": max(row["witness_error"] for row in rows),
        "reference_timing": _stats([row["reference_ms"] for row in rows]),
        "batched_timing": _stats([row["batched_ms"] for row in rows]),
        "full_cycle_20ms_acceptance": False,
        "production_online_controller_changed": False,
        "sources": sources,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_batched_capsule_bounds.py",
                              "b2_batched_capsule_bounds.py",
                              "shape_clearance.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path)},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "batched_capsule_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.private_root, args.a1_root,
                         tuple(args.scenes)), indent=2))


if __name__ == "__main__":
    main()
