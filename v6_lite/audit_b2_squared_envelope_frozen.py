"""Pair squared-distance PCC envelope against all five private frozen traces."""

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
from v6_lite import pcc_fixed_arc_state_envelope as fixed_module
from v6_lite.b2_compact_fixed_arc_envelope import (
    CompactFixedArcStateLocalPCCEnvelopeAudit,
)
from v6_lite.b2_squared_distance_envelope import SquaredDistanceFixedArcEnvelopeAudit
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import transform_from_free_qpos
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
    array = np.asarray(values)
    return {"p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(np.max(array))}


def run(output_dir: Path, private_root: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    spec = default_continuum_model_spec(robot)
    saved = {item["scenario"]["scenario_id"]: item
             for item in metrics["scenarios"]}
    rows = []
    trace_hashes = {}
    for scene_index in range(5):
        scene_id = f"v6_lite_scenario_{scene_index:02d}"
        trace_path = (private_root
                      / f"domain_endpoint_cholesky_scene{scene_index:02d}_1350"
                      / "private_rollout_trace.npz")
        with np.load(trace_path, allow_pickle=False) as trace:
            qpos_states = trace["qpos_states"][::10].copy()
            qvel_states = trace["task_qvel_states"].copy()
        if len(qpos_states) != 1351 or len(qvel_states) != 1351:
            raise ValueError(f"private trace incomplete: {scene_id}")
        verifier = WholeBodyCollisionVerifier(
            robot, _obstacles(saved[scene_id]["scenario"]),
            WholeBodyVerificationConfig(
                minimum_clearance=cfg.whole_body_minimum_clearance_m,
                query_distance_max=2.5,
                adaptive_subdivisions=cfg.verification_subdivisions,
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        data = mujoco.MjData(model)
        qpos_ids = np.asarray([
            model.jnt_qposadr[int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name))]
            for name in spec.low_level_joint_names
        ], dtype=np.int32)
        base_slice, _ = free_joint_slices(model, robot.base_joint_name)
        original_positions = fixed_module.FixedArcPCCPositions
        try:
            fixed_module.FixedArcPCCPositions = VectorizedFixedArcPCCPositions
            reference = CompactFixedArcStateLocalPCCEnvelopeAudit(model, spec)
            trial = SquaredDistanceFixedArcEnvelopeAudit(model, spec)
        finally:
            fixed_module.FixedArcPCCPositions = original_positions
        for tick, (qpos, qvel) in enumerate(zip(qpos_states, qvel_states)):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            projection = spec.project_actual_configuration(qpos[qpos_ids[:60]])
            base = transform_from_free_qpos(qpos[base_slice])
            if tick % 2:
                start = time.perf_counter()
                candidate = trial.evaluate(data, projection.planner_configuration,
                                           base)
                trial_ms = (time.perf_counter() - start) * 1000.0
                start = time.perf_counter()
                original = reference.evaluate(data, projection.planner_configuration,
                                              base)
                reference_ms = (time.perf_counter() - start) * 1000.0
            else:
                start = time.perf_counter()
                original = reference.evaluate(data, projection.planner_configuration,
                                              base)
                reference_ms = (time.perf_counter() - start) * 1000.0
                start = time.perf_counter()
                candidate = trial.evaluate(data, projection.planner_configuration,
                                           base)
                trial_ms = (time.perf_counter() - start) * 1000.0
            error = abs(original.min_margin_m - candidate.min_margin_m)
            if (original.status != candidate.status
                    or original.checked_capsule_count
                    != candidate.checked_capsule_count
                    or error > 1e-12):
                raise ValueError(f"envelope result changed at {scene_id}/{tick}")
            rows.append({"scenario_id": scene_id, "tick": tick,
                         "margin_error_m": error,
                         "reference_ms": reference_ms,
                         "trial_ms": trial_ms})
        trace_hashes[scene_id] = _sha(trace_path)
    rows_path = output_dir / "squared_envelope_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_squared_envelope_frozen_v1",
        "state_count": len(rows),
        "status_or_capsule_count_mismatches": 0,
        "maximum_margin_error_m": max(row["margin_error_m"] for row in rows),
        "reference_timing": _stats([row["reference_ms"] for row in rows]),
        "trial_timing": _stats([row["trial_ms"] for row in rows]),
        "full_cycle_20ms_acceptance": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_squared_envelope_frozen.py",
                              "b2_squared_distance_envelope.py",
                              "b2_compact_fixed_arc_envelope.py",
                              "pcc_vectorized_fixed_arc_positions.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path),
                           **trace_hashes},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "squared_envelope_summary.json").write_text(
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
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.private_root, args.a1_root),
                     indent=2))


if __name__ == "__main__":
    main()
