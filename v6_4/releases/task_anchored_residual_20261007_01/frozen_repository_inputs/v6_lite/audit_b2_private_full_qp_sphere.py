"""Re-solve both QP implementations at every saved private planning state.

This is a read-only full-trace timing and parity census. Both implementations
retain their own dual warm starts over each 400-tick scene. Saved torque traces
are never advanced or replaced by these newly computed candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_private_qp_sphere import (
    _ReferenceQP, _ScreenQP, _TELEMETRY, _parity, _sha, _source_sha,
)
from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _body_pose_and_twist, build_scenarios,
    default_v6_lite_robot_spec,
)
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


POINT_BUDGET = 255
METHODS = ("reference", "sphere_screen")


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data)),
            "over_20ms_count": int(np.count_nonzero(data > 20.0))}


def _field_digest(row: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(row["sources"], separators=(",", ":")).encode())
    for name in ("matrix", "lower", "drifts", "gains"):
        value = np.asarray(row[name], dtype=np.float64)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run(output_dir: Path, private_root: Path, a1_root: Path,
        *, max_ticks: int = 400, scene_limit: int = 5) -> dict:
    if not 1 <= max_ticks <= 400 or not 1 <= scene_limit <= 5:
        raise ValueError("trial subset outside the predeclared 400-tick five-scene trace")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    scenarios = {item.scenario_id: item for item in build_scenarios(robot, run_cfg)}
    saved = {item["scenario"]["scenario_id"]: item
             for item in metrics["scenarios"]}
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    report = {
        "schema": "v6_2_b2_private_full_qp_sphere_v1",
        "scope": "read_only_all_saved_private_planning_states",
        "max_ticks_per_scene": max_ticks, "scene_count": scene_limit,
        "point_budget": POINT_BUDGET,
        "method_order_by_scene": {},
        "input_metrics_sha256": _sha(metrics_path),
        "input_private_hashes": {},
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_private_full_qp_sphere.py",
                                       "audit_b2_private_qp_sphere.py",
                                       "audit_b2_weighted_qp_sphere_trial.py",
                                       "audit_b2_weighted_qp_probe.py",
                                       "hierarchical_qp.py", "pcc_interval_cbf.py",
                                       "safety_contract.py")},
        "production_online_controller_changed": False,
        "private_servo_commanded_by_trial": False,
        "full_cycle_timing_measured": False,
        "strict_online_domain_accepted": False,
    }
    all_rows = []
    failures = []
    for index in range(scene_limit):
        scene_id = f"v6_lite_scenario_{index:02d}"
        scene = scenarios[scene_id]
        frozen = saved[scene_id]
        folder = private_root / f"scene_{index:02d}"
        trace_path = folder / "private_rollout_trace.npz"
        records_path = folder / "private_rollout_records.jsonl"
        summary_path = folder / "private_rollout_summary.json"
        private = json.loads(summary_path.read_text(encoding="utf-8"))
        if (private["executed_ticks"] != 400
                or private["trace_sha256"] != _sha(trace_path)
                or private["records_sha256"] != _sha(records_path)):
            raise ValueError(f"private trace changed: {scene_id}")
        report["input_private_hashes"][scene_id] = {
            "trace_sha256": _sha(trace_path),
            "records_sha256": _sha(records_path),
            "summary_sha256": _sha(summary_path),
        }
        with np.load(trace_path, allow_pickle=False) as trace:
            qpos_states = trace["qpos_states"].copy()
            qvel_states = trace["task_qvel_states"].copy()
        saved_rows = [json.loads(line) for line in records_path.read_text(
            encoding="utf-8").splitlines()]
        order = METHODS if index % 2 == 0 else METHODS[::-1]
        report["method_order_by_scene"][scene_id] = list(order)
        raw = {}
        for method in order:
            verifier = WholeBodyCollisionVerifier(
                robot, _obstacles(frozen["scenario"]), WholeBodyVerificationConfig(
                    minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
                    query_distance_max=2.5,
                    adaptive_subdivisions=run_cfg.verification_subdivisions,
                    self_collision_ancestor_exclusion_depth=3,
                    include_target_satellite_pairs=True,
                ),
            )
            model = verifier.model
            model.geom_contype[:] = 0
            model.geom_conaffinity[:] = 0
            evaluator = FixedIntervalCBFEvaluator(robot, model)
            envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
            query = PersistentIntervalDecisionQuery(evaluator.shape_model)
            qp_cls = _ReferenceQP if method == "reference" else _ScreenQP
            qp = qp_cls(robot, model, verifier.pairs, no_legacy,
                        evaluator=evaluator)
            target_body = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite"))
            data = mujoco.MjData(model)
            for tick in range(max_ticks):
                data.qpos[:] = qpos_states[tick * 10]
                data.qvel[:] = qvel_states[tick]
                data.time = tick * run_cfg.task_period_s
                data.ctrl[:] = 0
                started = time.perf_counter()
                mujoco.mj_forward(model, data)
                previous = (np.zeros(17) if tick == 0 else
                            np.asarray(saved_rows[tick - 1]["selected_command"],
                                       dtype=np.float64))
                projection = evaluator.shape_spec.project_actual_configuration(
                    data.qpos[evaluator.qpos_ids[:60]])
                base = transform_from_free_qpos(
                    data.qpos[evaluator.base_qpos_slice])
                current_envelope = envelope.evaluate(
                    data, projection.planner_configuration, base)
                box = target_box_from_mujoco(model, data,
                                             evaluator.target_geom_id)
                decision = query.evaluate(
                    projection.planner_configuration, base, box,
                    IntervalPartition.uniform(),
                    max_point_evaluations=POINT_BUDGET,
                )
                planner_q = robot.low_level_to_planner @ data.qpos[
                    evaluator.qpos_ids]
                _lo, _hi, speed, box_valid = ramp_velocity_abs_bound(
                    robot, cfg, planner_q, previous)
                generalized_map, _ = qp.reaction_velocity_map(data)
                ids, _excluded = select_intervals_by_frozen_reach(
                    decision.lower_by_interval_id, decision.partition,
                    evaluator, data, cfg, generalized_map,
                    velocity_abs_bound=speed,
                )
                if (not box_valid or not decision.bounds_valid
                        or decision.proxy_clearance_status
                        != "PROXY_CLEARANCE_AT_LEAST_GATE"
                        or current_envelope.status != "COVERED_AT_THIS_STATE"):
                    raise ValueError(f"private state unsupported: {scene_id} {tick}")
                qp.partition = decision.partition
                qp.selected_ids = ids
                qp.pair_query_ms = 0.0
                _TELEMETRY.clear()
                (rigid_target, rigid_velocity, target_rotation,
                 target_angular_velocity) = _body_pose_and_twist(
                    model, data, target_body,
                    scene.grasp_point_target_frame_m)
                continuum_target, continuum_velocity = scene.continuum_target.sample(
                    float(data.time))
                result = qp.solve(
                    data,
                    rigid_target_position=rigid_target,
                    rigid_target_velocity=rigid_velocity,
                    rigid_target_rotation=(target_rotation
                                           @ scene.grasp_rotation_target_frame),
                    rigid_target_angular_velocity=target_angular_velocity,
                    continuum_target_position=continuum_target,
                    continuum_target_velocity=continuum_velocity,
                    continuum_target_rotation=scene.continuum_target_rotation_world,
                    continuum_target_angular_velocity=np.zeros(3),
                    state_timestamp_s=float(data.time),
                    target_timestamp_s=float(data.time),
                    ramp_start_velocity=previous,
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                telemetry = (_TELEMETRY[0] if method == "sphere_screen"
                             and len(_TELEMETRY) == 1 else None)
                if method == "sphere_screen" and telemetry is None:
                    raise ValueError(f"screen telemetry missing: {scene_id} {tick}")
                selected = (None if result.planner_velocity is None else
                            result.planner_velocity.tolist())
                item = {
                    "scenario_id": scene_id, "tick": tick,
                    "method": method, "preflight_plus_qp_ms": elapsed,
                    "qp_full_ms": result.full_latency_s * 1000.0,
                    "qp_solver_ms": result.solver_latency_s * 1000.0,
                    "mujoco_pair_block_ms": qp.pair_query_ms,
                    "exact_pair_calls": (telemetry["exact_call_count"]
                                         if telemetry is not None else
                                         len(verifier.pairs)),
                    "selected_interval_count": len(ids),
                    "subspace_residual_linf_rad": projection.residual_linf_rad,
                    "solver_status": result.solver_status,
                    "action_mode": result.action_validation.mode.value,
                    "failure_reason": result.action_validation.failure_reason.value,
                    "candidate": result.solver_candidate.tolist(),
                    "selected_command": selected,
                    "sources": list(result.clearance_sources),
                    "matrix": result.clearance_matrix.tolist(),
                    "lower": result.clearance_lower.tolist(),
                    "drifts": result.clearance_target_drift_m_s.tolist(),
                    "gains": result.clearance_barrier_gain_s_inv.tolist(),
                }
                item["constraint_row_sha256"] = _field_digest(item)
                expected = saved_rows[tick]["selected_command"]
                item["saved_command_max_error"] = (
                    float("inf") if selected is None else float(np.max(np.abs(
                        np.asarray(selected) - np.asarray(expected)))))
                raw[method, tick] = item
                all_rows.append({key: value for key, value in item.items()
                                 if key not in ("matrix", "lower", "drifts",
                                                "gains", "sources")})
            print(f"[b2-full-qp-sphere] {scene_id} {method}: "
                  f"{max_ticks} frozen states", flush=True)
        for tick in range(max_ticks):
            reference = raw["reference", tick]
            screen = raw["sphere_screen", tick]
            parity = _parity(reference, screen)
            if (not parity["same"]
                    or reference["constraint_row_sha256"]
                    != screen["constraint_row_sha256"]
                    or reference["saved_command_max_error"] > 1e-8
                    or screen["saved_command_max_error"] > 1e-8):
                failures.append({"scenario_id": scene_id, "tick": tick,
                                 "parity": parity,
                                 "row_hash_match": reference["constraint_row_sha256"]
                                 == screen["constraint_row_sha256"],
                                 "reference_saved_command_max_error":
                                 reference["saved_command_max_error"],
                                 "screen_saved_command_max_error":
                                 screen["saved_command_max_error"]})
    report["record_count"] = len(all_rows)
    report["failure_count"] = len(failures)
    report["summary"] = {
        method: {field: _stats([row[field] for row in all_rows
                                if row["method"] == method])
                 for field in ("preflight_plus_qp_ms", "qp_full_ms",
                               "qp_solver_ms", "mujoco_pair_block_ms",
                               "exact_pair_calls", "saved_command_max_error")}
        for method in METHODS
    }
    paths = {"records": output_dir / "private_full_qp_sphere_records.jsonl",
             "failures": output_dir / "private_full_qp_sphere_failures.jsonl"}
    for label, path in paths.items():
        values = all_rows if label == "records" else failures
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False) + "\n")
    report.update({f"{label}_sha256": _sha(path)
                   for label, path in paths.items()})
    summary = output_dir / "private_full_qp_sphere_summary.json"
    with summary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": "v6_2_b2_private_full_qp_sphere_manifest_v1",
                "summary_sha256": _sha(summary),
                **{f"{label}_sha256": _sha(path)
                   for label, path in paths.items()}}
    manifest_path = output_dir / "private_full_qp_sphere_manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/private_rollout_400"))
    parser.add_argument("--a1-root", type=Path, default=Path(
        "v6_lite/output/v6_2_a1"))
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--scene-limit", type=int, default=5)
    args = parser.parse_args()
    report = run(args.output_dir, args.private_root, args.a1_root,
                 max_ticks=args.max_ticks, scene_limit=args.scene_limit)
    print(json.dumps({"record_count": report["record_count"],
                      "failure_count": report["failure_count"],
                      "summary": report["summary"]}, indent=2))
    if report["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
