"""ABBA read-only QP sphere-screen trial on private torque-replayed states.

All four groups rebuild the unchanged 17-D weighted QP at the same thirty
predeclared private states. Sphere screening only skips original MuJoCo pairs
whose bounding-sphere lower bound exceeds the original query radius. No
candidate is sent to the private or production torque servo.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from v6_lite.audit_b2_weighted_qp_probe import _ReadOnlyIntervalQP
from v6_lite.audit_b2_weighted_qp_sphere_trial import (
    _SphereScreenIntervalQP, _TELEMETRY,
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


GROUPS = (("a1", "reference"), ("b1", "sphere_screen"),
          ("b2", "sphere_screen"), ("a2", "reference"))
TICKS = (1, 50, 100, 200, 300, 399)
POINT_BUDGET = 255


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        raise ValueError("empty timing group")
    return {"count": len(data), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data)),
            "over_20ms_count": int(np.count_nonzero(data > 20.0))}


class _TimedPairs:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pair_query_ms = 0.0

    def _mujoco_clearance_constraint_set(self, data, generalized_map):
        started = time.perf_counter()
        block = super()._mujoco_clearance_constraint_set(data, generalized_map)
        self.pair_query_ms += (time.perf_counter() - started) * 1000.0
        return block


class _ReferenceQP(_TimedPairs, _ReadOnlyIntervalQP):
    pass


class _ScreenQP(_TimedPairs, _SphereScreenIntervalQP):
    pass


def _parity(a: dict, b: dict) -> dict:
    if (a["sources"] != b["sources"]
            or a["action_mode"] != b["action_mode"]
            or a["failure_reason"] != b["failure_reason"]
            or a["solver_status"] != b["solver_status"]):
        return {"same": False, "reason": "SOURCE_OR_STATUS_CHANGED"}
    if (a["selected_command"] is None) != (b["selected_command"] is None):
        return {"same": False, "reason": "SELECTED_COMMAND_PRESENCE_CHANGED"}
    selected_error = (0.0 if a["selected_command"] is None else float(
        np.max(np.abs(np.asarray(a["selected_command"])
                      - np.asarray(b["selected_command"])))) )
    fields = ("candidate", "matrix", "lower", "drifts", "gains")
    errors = {name: float(np.max(np.abs(np.asarray(a[name])
                                         - np.asarray(b[name]))))
              if np.asarray(a[name]).size else 0.0 for name in fields}
    return {"same": selected_error <= 1e-8
            and all(error <= 1e-8 for error in errors.values()),
            "selected_error": selected_error, "field_errors": errors}


def run(output_dir: Path, private_root: Path, a1_root: Path) -> dict:
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
        "schema": "v6_2_b2_private_qp_sphere_abba_v1",
        "scope": "read_only_QP_on_private_torque_replayed_frozen_states",
        "groups_predeclared": [{"name": name, "method": method}
                               for name, method in GROUPS],
        "ticks_per_scene": list(TICKS),
        "point_budget": POINT_BUDGET,
        "input_metrics_sha256": _sha(metrics_path),
        "inputs": {},
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_private_qp_sphere.py",
            "audit_b2_weighted_qp_sphere_trial.py",
            "audit_b2_weighted_qp_probe.py", "hierarchical_qp.py",
            "pcc_interval_cbf.py", "safety_contract.py",
        )},
        "online_controller_changed": False,
        "private_servo_commanded_by_trial": False,
        "strict_online_domain_accepted": False,
        "full_cycle_timing_measured": False,
        "groups": [], "summary": {},
    }
    records = []
    failures = []
    reference = {}
    for group_name, method in GROUPS:
        group_records = []
        for index in range(5):
            scene_id = f"v6_lite_scenario_{index:02d}"
            scene = scenarios[scene_id]
            frozen = saved[scene_id]
            folder = private_root / f"scene_{index:02d}"
            trace_path = folder / "private_rollout_trace.npz"
            records_path = folder / "private_rollout_records.jsonl"
            summary_path = folder / "private_rollout_summary.json"
            private = json.loads(summary_path.read_text(encoding="utf-8"))
            if (private["executed_ticks"] != 400
                    or private["stop_reason"] != "HORIZON_COMPLETE"
                    or private["trace_sha256"] != _sha(trace_path)
                    or private["records_sha256"] != _sha(records_path)):
                raise ValueError(f"private source trace changed: {scene_id}")
            report["inputs"][scene_id] = {
                "trace_sha256": _sha(trace_path),
                "records_sha256": _sha(records_path),
                "summary_sha256": _sha(summary_path),
            }
            with np.load(trace_path, allow_pickle=False) as trace:
                qpos_states = trace["qpos_states"].copy()
                qvel_states = trace["task_qvel_states"].copy()
            saved_records = [json.loads(line) for line in records_path.read_text(
                encoding="utf-8").splitlines()]
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
            for tick in TICKS:
                data.qpos[:] = qpos_states[tick * 10]
                data.qvel[:] = qvel_states[tick]
                data.time = tick * run_cfg.task_period_s
                data.ctrl[:] = 0
                mujoco.mj_forward(model, data)
                previous = np.asarray(saved_records[tick - 1]["selected_command"],
                                      dtype=np.float64)
                started = time.perf_counter()
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
                    raise ValueError(f"private frozen state no longer supported: {scene_id} {tick}")
                qp.partition = decision.partition
                qp.selected_ids = ids
                qp.previous_velocity = previous.copy()
                qp._previous_constraint_dual.clear()
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
                    raise ValueError("sphere-screen telemetry missing")
                selected = (None if result.planner_velocity is None else
                            result.planner_velocity.tolist())
                payload = {
                    "group": group_name, "method": method,
                    "scenario_id": scene_id, "tick": tick,
                    "preflight_plus_qp_ms": elapsed,
                    "qp_full_ms": result.full_latency_s * 1000.0,
                    "qp_solver_ms": result.solver_latency_s * 1000.0,
                    "mujoco_pair_block_ms": qp.pair_query_ms,
                    "shape_block_ms": result.shape_clearance_latency_s * 1000.0,
                    "query_ms": decision.elapsed_ms,
                    "exact_pair_calls": (telemetry["exact_call_count"]
                                         if telemetry is not None else
                                         len(verifier.pairs)),
                    "query_points": decision.point_evaluation_count,
                    "selected_interval_count": len(ids),
                    "strict_subspace_domain": qp.last_batch.geometry_domain_status,
                    "candidate": result.solver_candidate.tolist(),
                    "selected_command": selected,
                    "solver_status": result.solver_status,
                    "action_mode": result.action_validation.mode.value,
                    "failure_reason": result.action_validation.failure_reason.value,
                    "sources": list(result.clearance_sources),
                    "matrix": result.clearance_matrix.tolist(),
                    "lower": result.clearance_lower.tolist(),
                    "drifts": result.clearance_target_drift_m_s.tolist(),
                    "gains": result.clearance_barrier_gain_s_inv.tolist(),
                }
                key = scene_id, tick
                if group_name == "a1":
                    reference[key] = payload
                else:
                    parity = _parity(payload, reference[key])
                    if not parity["same"]:
                        failures.append({"group": group_name,
                                         "scenario_id": scene_id, "tick": tick,
                                         "parity": parity})
                group_records.append(payload)
                records.append(payload)
        report["groups"].append({"name": group_name, "method": method,
                                 "record_count": len(group_records),
                                 "mismatch_count": sum(item["group"] == group_name
                                                       for item in failures)})
        print(f"[b2-private-qp-sphere] {group_name} {method}: "
              f"{len(group_records)} states, "
              f"{report['groups'][-1]['mismatch_count']} mismatches", flush=True)
    for method in ("reference", "sphere_screen"):
        own = [row for row in records if row["method"] == method]
        report["summary"][method] = {
            "record_count": len(own),
            **{name: _stats([row[name] for row in own]) for name in (
                "preflight_plus_qp_ms", "qp_full_ms", "qp_solver_ms",
                "mujoco_pair_block_ms", "shape_block_ms", "query_ms",
                "exact_pair_calls",
            )},
        }
    report["record_count"] = len(records)
    report["failure_count"] = len(failures)
    paths = {"records": output_dir / "private_qp_sphere_records.jsonl",
             "failures": output_dir / "private_qp_sphere_failures.jsonl"}
    for name, path in paths.items():
        values = records if name == "records" else failures
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":"),
                                        allow_nan=False) + "\n")
    report.update({f"{name}_sha256": _sha(path)
                   for name, path in paths.items()})
    report_path = output_dir / "private_qp_sphere_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 私有轨迹冻结状态的原 QP 包围球筛选试验", "",
        "在五场景各六个预声明规划状态，按全对→筛选→筛选→全对运行四组；"
        "每次清空 QP 对偶热启动，候选、约束行及失败状态逐状态对照。"
        "私有轨迹已另有原生力矩重放，这里只读取冻结状态，未执行候选。", "",
        "| 方法 | 记录 | 预检＋QP p95 / p99 / 最大 ms | 超 20 ms | "
        "QP 内原 MuJoCo 对处理 p95 ms | 精确对查询 p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("reference", "sphere_screen"):
        item = report["summary"][method]
        timing = item["preflight_plus_qp_ms"]
        lines.append(
            f"| {method} | {item['record_count']} | "
            f"{timing['p95']:.3f} / {timing['p99']:.3f} / "
            f"{timing['max']:.3f} | {timing['over_20ms_count']} | "
            f"{item['mujoco_pair_block_ms']['p95']:.3f} | "
            f"{item['exact_pair_calls']['p95']:.1f} |"
        )
    lines += [
        "", f"共 {report['record_count']} 条只读记录，"
        f"候选／行不一致 {report['failure_count']}。"
        "筛选只改变原 MuJoCo 精确对查询的候选集合；"
        "原安全距离、PCC 区间、胶囊、同一 QP 和动作验证保持。"
        "预检计时含当前状态包络、根区间查询和 QP，"
        "不含独立重算、十步力矩或在线调度。"
        "原严格子空间门禁仍不满足，不能据此接入在线。", "",
    ]
    doc_path = output_dir / "PRIVATE_QP_SPHERE.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_private_qp_sphere_manifest_v1",
                "summary_sha256": _sha(report_path),
                "document_sha256": _sha(doc_path),
                **{f"{name}_sha256": _sha(path) for name, path in paths.items()}}
    with (output_dir / "private_qp_sphere_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
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
    args = parser.parse_args()
    report = run(args.output_dir, args.private_root, args.a1_root)
    print(json.dumps({"record_count": report["record_count"],
                      "failure_count": report["failure_count"],
                      "summary": report["summary"]}, indent=2))
    if report["failure_count"] or report["record_count"] != 120:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
