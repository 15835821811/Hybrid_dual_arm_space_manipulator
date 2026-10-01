"""Pair original and batched interval decisions on all private scene 01 states."""

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
from v6_lite.audit_b2_batched_query import _parity
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shape_clearance import target_box_from_mujoco


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, private_dir: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == "v6_lite_scenario_01")
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    data = mujoco.MjData(model)
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    original = PersistentIntervalDecisionQuery(evaluator.shape_model)
    batched = BatchedDistanceDecisionQuery(evaluator.shape_model)
    trace_path = private_dir / "private_rollout_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos_states = trace["qpos_states"][::10].copy()
        qvel_states = trace["task_qvel_states"].copy()
    if len(qpos_states) != 1351 or len(qvel_states) != 1351:
        raise ValueError("private trace incomplete")
    rows = []
    for tick, (qpos, qvel) in enumerate(zip(qpos_states, qvel_states)):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
        args = (projection.planner_configuration, base, box,
                IntervalPartition.uniform())
        if tick % 2:
            trial_result = batched.evaluate(*args, max_point_evaluations=255)
            reference = original.evaluate(*args, max_point_evaluations=255)
        else:
            reference = original.evaluate(*args, max_point_evaluations=255)
            trial_result = batched.evaluate(*args, max_point_evaluations=255)
        parity = _parity(reference, trial_result)
        if (reference.bounds_valid != trial_result.bounds_valid
                or reference.budget_exhausted != trial_result.budget_exhausted
                or reference.failure_reason != trial_result.failure_reason):
            raise ValueError(f"query status differs at tick {tick}")
        rows.append({"tick": tick,
                     "point_evaluations": reference.point_evaluation_count,
                     "interval_count": reference.interval_count,
                     "proxy_status": reference.proxy_clearance_status,
                     **parity})
    rows_path = output_dir / "private_query_cross_impl_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_private_query_cross_impl_v1",
        "scenario_id": "v6_lite_scenario_01",
        "state_count": len(rows),
        "decision_partition_or_status_mismatch_count": 0,
        "maximum_bound_error_m": max(
            max(row["lower_max_abs_error_m"],
                row["midpoint_upper_max_abs_error_m"],
                row["global_endpoint_max_abs_error_m"]) for row in rows),
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_private_query_cross_impl.py",
                              "pcc_persistent_interval_query.py",
                              "pcc_batched_distance_query.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path),
                           "private_trace": _sha(trace_path)},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "private_query_cross_impl_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.private_dir, args.a1_root),
                     indent=2))


if __name__ == "__main__":
    main()
