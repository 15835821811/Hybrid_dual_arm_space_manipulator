"""Compare five-section vectorization against fixed-arc PCC positions."""

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
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_fixed_arc_positions import FixedArcPCCPositions
from v6_lite.pcc_state_local_envelope import PCC_SAMPLES_PER_SEGMENT
from v6_lite.pcc_vectorized_fixed_arc_positions import VectorizedFixedArcPCCPositions
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, default_v6_lite_robot_spec, free_joint_slices,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(np.max(array))}


def run(output_dir: Path, private_root: Path, a1_root: Path,
        scene_indices: tuple[int, ...]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    scene = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == "v6_lite_scenario_01")
    robot = default_v6_lite_robot_spec()
    spec = default_continuum_model_spec(robot)
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(scene["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    qpos_ids = np.asarray([
        model.jnt_qposadr[int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name))]
        for name in spec.low_level_joint_names
    ], dtype=np.int32)
    base_slice, _ = free_joint_slices(model, robot.base_joint_name)
    boundaries = spec.segment_boundaries_m
    arclengths = np.concatenate([
        boundaries[index] + np.linspace(
            0.0, float(spec.segment_lengths_m[index]),
            PCC_SAMPLES_PER_SEGMENT)
        for index in range(len(spec.segment_lengths_m))
    ])
    reference = FixedArcPCCPositions(spec, arclengths)
    trial = VectorizedFixedArcPCCPositions(spec, arclengths)
    synthetic_q = (
        np.zeros(10),
        np.full(10, 1e-8),
        np.tile([1e-4 * (1.0 - 1e-8), 0.0], 5),
        np.tile([1e-4 * (1.0 + 1e-8), 0.0], 5),
        np.tile([0.4, -0.3], 5),
    )
    turned_base = np.eye(4)
    turned_base[:3, :3] = [[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]]
    turned_base[:3, 3] = [0.12, -0.08, 0.04]
    synthetic_errors = []
    for q in synthetic_q:
        for base in (np.eye(4), turned_base):
            error = float(np.max(np.abs(reference.evaluate(q, base)
                                            - trial.evaluate(q, base))))
            synthetic_errors.append(error)
            if error > 1e-12:
                raise ValueError("synthetic fixed-arc positions differ")
    rows = []
    traces = {}
    for scene_index in scene_indices:
        trace_path = (private_root
                      / f"domain_endpoint_cholesky_scene{scene_index:02d}_1350"
                      / "private_rollout_trace.npz")
        with np.load(trace_path, allow_pickle=False) as trace:
            qpos_states = trace["qpos_states"][::10].copy()
        if len(qpos_states) != 1351 or qpos_states.shape[1] != model.nq:
            raise ValueError("private trace has incompatible qpos states")
        for tick, qpos in enumerate(qpos_states):
            projection = spec.project_actual_configuration(qpos[qpos_ids[:60]])
            base = transform_from_free_qpos(qpos[base_slice])
            if tick % 2:
                start = time.perf_counter()
                candidate = trial.evaluate(projection.planner_configuration, base)
                trial_ms = (time.perf_counter() - start) * 1000.0
                start = time.perf_counter()
                original = reference.evaluate(projection.planner_configuration, base)
                reference_ms = (time.perf_counter() - start) * 1000.0
            else:
                start = time.perf_counter()
                original = reference.evaluate(projection.planner_configuration, base)
                reference_ms = (time.perf_counter() - start) * 1000.0
                start = time.perf_counter()
                candidate = trial.evaluate(projection.planner_configuration, base)
                trial_ms = (time.perf_counter() - start) * 1000.0
            error = float(np.max(np.abs(original - candidate)))
            rows.append({"scenario_id": scene_index, "tick": tick,
                         "max_abs_position_error_m": error,
                         "reference_ms": reference_ms,
                         "trial_ms": trial_ms})
            if error > 1e-12:
                raise ValueError(f"fixed-arc positions differ at {scene_index}/{tick}")
        traces[f"scene_{scene_index:02d}"] = _sha(trace_path)
    rows_path = output_dir / "fixed_arc_position_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_vectorized_fixed_arc_positions_v1",
        "scope": "read_only_saved_private_50hz_states",
        "state_count": len(rows),
        "synthetic_case_count": len(synthetic_errors),
        "synthetic_maximum_absolute_position_error_m": max(synthetic_errors),
        "maximum_absolute_position_error_m": max(
            row["max_abs_position_error_m"] for row in rows),
        "reference_timing": _stats([row["reference_ms"] for row in rows]),
        "trial_timing": _stats([row["trial_ms"] for row in rows]),
        "full_cycle_20ms_acceptance": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_vectorized_fixed_arc_positions.py",
                              "pcc_vectorized_fixed_arc_positions.py",
                              "pcc_fixed_arc_positions.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path),
                           **traces},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "fixed_arc_position_summary.json").write_text(
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
