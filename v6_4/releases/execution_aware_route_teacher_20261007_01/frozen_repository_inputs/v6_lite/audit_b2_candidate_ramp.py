"""Branch one ten-step servo ramp from each frozen early B.2 QP probe.

An old-command branch must reproduce every published A.1 torque and next
planning qpos. A second private MuJoCo branch applies the saved read-only B.2
candidate through the same 67-torque servo. It measures realized geometry and
rebuilds the new interval start rows at the next tick. No command is sent to
the historical run or admitted to the online interval mode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import (
    HierarchicalQPConfig, free_joint_slices, joint_addresses,
)
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import _model_based_servo_torque, default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


TICKS = (50, 100, 150)
STEPS = 10
POINT_BUDGET = 255


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _branch(model: mujoco.MjModel, robot, evaluator: FixedIntervalCBFEvaluator,
            envelope: StateLocalPCCEnvelopeAudit, qpos: np.ndarray,
            qvel: np.ndarray, start_time: float, reference_start: np.ndarray,
            previous_command: np.ndarray, endpoint_command: np.ndarray,
            *, physics_period_s: float, task_period_s: float) -> tuple[dict, mujoco.MjData]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.time = start_time
    data.ctrl[:] = 0.0
    qpos_ids, dof_ids = joint_addresses(model, robot)
    _base_qpos, base_dof = free_joint_slices(model, robot.base_joint_name)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    reference = reference_start.copy()
    torques = []
    states = []
    coverage_records = []
    saturation_counts = []
    for step in range(STEPS + 1):
        mujoco.mj_forward(model, data)
        states.append(data.qpos.copy())
        actual = data.qpos[evaluator.qpos_ids[:60]]
        projection = evaluator.shape_spec.project_actual_configuration(actual)
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        coverage = envelope.evaluate(data, projection.planner_configuration, base)
        coverage_records.append({
            "servo_substep": step, "time_s": float(data.time),
            "status": coverage.status,
            "minimum_margin_m": coverage.min_margin_m,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
        })
        if step == STEPS:
            break
        measured_q = robot.decode_position(data.qpos[qpos_ids])
        ref = advance_reference(
            reference, previous_command, endpoint_command, step + 1,
            measured_q, robot.planner_lower, robot.planner_upper,
            physics_period_s=physics_period_s, task_period_s=task_period_s,
        )
        reference = ref.position
        torque, _acc, diagnostics = _model_based_servo_torque(
            model, data, robot, qpos_ids, dof_ids, base_dof,
            ref.position, ref.velocity, ref.feedforward_acceleration, mass,
        )
        torques.append(torque.copy())
        saturation_counts.append(diagnostics.torque_saturation_count)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
    return {
        "torques": np.asarray(torques),
        "qpos_states": np.asarray(states),
        "coverage": coverage_records,
        "torque_saturation_count": sum(saturation_counts),
    }, data


def _next_start(model: mujoco.MjModel, robot, verifier,
                cfg: HierarchicalQPConfig, evaluator: FixedIntervalCBFEvaluator,
                data: mujoco.MjData, endpoint_command: np.ndarray) -> dict:
    projection = evaluator.shape_spec.project_actual_configuration(
        data.qpos[evaluator.qpos_ids[:60]])
    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
    box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
    query = PersistentIntervalDecisionQuery(evaluator.shape_model)
    decision = query.evaluate(
        projection.planner_configuration, base, box, IntervalPartition.uniform(),
        max_point_evaluations=POINT_BUDGET,
    )
    result = {
        "proxy_status": decision.proxy_clearance_status,
        "proxy_lower_m": decision.distance_lower_bound_m,
        "proxy_upper_m": decision.distance_upper_bound_m,
        "point_evaluations": decision.point_evaluation_count,
        "budget_exhausted": decision.budget_exhausted,
        "query_failure_reason": decision.failure_reason,
        "interval_count": decision.interval_count,
        "subspace_residual_linf_rad": projection.residual_linf_rad,
        "frozen_rows_status": "NOT_EVALUATED",
    }
    if not decision.bounds_valid:
        return result
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    independent = ReplayConstraintBuilder(robot, model, verifier.pairs, no_legacy)
    generalized_map, reaction_residual = independent._reaction_map(data)
    planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
    _lower, _upper, speed, _box_valid = ramp_velocity_abs_bound(
        robot, cfg, planner_q, endpoint_command,
    )
    ids, _excluded = select_intervals_by_frozen_reach(
        decision.lower_by_interval_id, decision.partition, evaluator,
        data, cfg, generalized_map, velocity_abs_bound=speed,
    )
    batch = evaluator.evaluate_state(
        data, decision.partition, generalized_map=generalized_map,
        derivative_interval_ids=ids,
    )
    result["selected_interval_count"] = len(ids)
    result["reaction_map_residual"] = reaction_residual
    result["interval_coverage_complete"] = batch.coverage_complete
    result["interval_well_formed"] = batch.interval_well_formed
    result["analytic_bound_assumptions_satisfied"] = (
        batch.analytic_bound_assumptions_satisfied)
    if not (batch.interval_well_formed and batch.coverage_complete
            and batch.analytic_bound_assumptions_satisfied
            and all(row.derivative_status == "SUPPORTED"
                    for row in batch.rows if row.interval_id in ids)):
        result["frozen_rows_status"] = "UNSUPPORTED"
        return result
    original = independent.build(data)
    matrix, lower, _drifts, _gains, sources = combined_rows(
        original, batch, ids, cfg,
    )
    slacks = matrix @ endpoint_command - lower
    index = int(np.argmin(slacks)) if len(slacks) else None
    minimum = float(slacks[index]) if index is not None else None
    result.update({
        "frozen_rows_status": (
            "START_ROWS_SATISFIED" if minimum is None
            or minimum >= -cfg.clearance_rate_tolerance_m_s
            else "START_CLEARANCE_VIOLATION"),
        "row_count": len(slacks),
        "worst_source": sources[index] if index is not None else None,
        "minimum_start_slack_m_s": minimum,
    })
    return result


def run(output_dir: Path, a1_root: Path, probe_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if (probe["probe_ticks"] != list(TICKS) or len(probe["records"]) != 30
            or not probe["passed_as_read_only_integrity"]
            or probe["online_control_changed"]):
        raise ValueError("frozen early QP probe changed")
    probes = {(row["mode"], row["scenario_id"], row["tick"]): row
              for row in probe["records"]}
    if len(probes) != 30:
        raise ValueError("duplicate frozen probe")
    report = {
        "schema": "v6_2_b2_one_ramp_candidate_branch_v1",
        "scope": "thirty_predeclared_early_old_A1_states_private_one_ramp_counterfactual",
        "probe_ticks": list(TICKS), "servo_steps_per_task": STEPS,
        "point_budget_for_realized_next_start": POINT_BUDGET,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_candidate_ramp.py", "execution_ramp.py", "run_v6_lite.py",
            "pcc_interval_cbf.py", "pcc_state_local_envelope.py",
            "b2_shadow_feasibility.py", "recompute_execution_constraints.py",
        )},
        "input_probe_sha256": _sha(probe_path),
        "new_interval_mode_executed": False,
        "private_counterfactual_servo_executed": True,
        "continuous_time_certified": False,
        "inputs": {}, "records": [], "modes": {},
    }
    robot = default_v6_lite_robot_spec()
    steps_path = output_dir / "candidate_ramp_states.jsonl"
    failures_path = output_dir / "candidate_ramp_failures.jsonl"
    with steps_path.open("x", encoding="utf-8", newline="\n") as steps_out, \
            failures_path.open("x", encoding="utf-8", newline="\n") as failures_out:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg = HierarchicalQPConfig(**metrics["qp_config"])
            run_cfg = metrics["run_config"]
            scenes = metrics["scenarios"]
            if len(scenes) != 5:
                raise ValueError("A.1 scenario count changed")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path), "traces": [],
            }
            for scene in scenes:
                scenario_id = scene["scenario"]["scenario_id"]
                verifier = WholeBodyCollisionVerifier(
                    robot, _obstacles(scene["scenario"]),
                    WholeBodyVerificationConfig(
                        minimum_clearance=run_cfg["whole_body_minimum_clearance_m"],
                        query_distance_max=2.5,
                        adaptive_subdivisions=run_cfg["verification_subdivisions"],
                        self_collision_ancestor_exclusion_depth=3,
                        include_target_satellite_pairs=True,
                    ),
                )
                model = verifier.model
                model.geom_contype[:] = 0
                model.geom_conaffinity[:] = 0
                replay = mujoco.MjData(model)
                evaluator = FixedIntervalCBFEvaluator(robot, model)
                envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
                trace_path = Path(scene["trace"]["path"])
                trace_sha = _sha(trace_path)
                if trace_sha != scene["trace"]["sha256"]:
                    raise ValueError("A.1 trace hash changed")
                report["inputs"][mode]["traces"].append({
                    "scenario_id": scenario_id, "path": trace_path.as_posix(),
                    "sha256": trace_sha,
                })
                with np.load(trace_path, allow_pickle=False) as trace:
                    initial_qpos = trace["initial_qpos"].copy()
                    initial_qvel = trace["initial_qvel"].copy()
                    torque = trace["torque"][:STEPS * (TICKS[-1] + 1)].copy()
                    task_qpos = trace["task_qpos"][:TICKS[-1] + 2].copy()
                    task_time = trace["task_time"][:TICKS[-1] + 1].copy()
                    reference_q = trace["reference_q"][:STEPS * TICKS[-1]].copy()
                    selected = trace["task_selected_command"][:TICKS[-1] + 1].copy()
                replay.qpos[:] = initial_qpos
                replay.qvel[:] = initial_qvel
                replay.ctrl[:] = 0.0
                mujoco.mj_forward(model, replay)
                replay_error = 0.0
                for step in range(STEPS * TICKS[-1] + 1):
                    if step % STEPS == 0:
                        tick = step // STEPS
                        mujoco.mj_forward(model, replay)
                        replay_error = max(
                            replay_error,
                            float(np.max(np.abs(replay.qpos - task_qpos[tick]))),
                            abs(float(replay.time - task_time[tick])),
                        )
                        if replay_error > 1e-8:
                            raise ValueError("frozen-state native torque replay diverged")
                        if tick in TICKS:
                            key = mode, scenario_id, tick
                            row = probes[key]
                            if not (row["candidate_valid"] and row["ramp_valid"]
                                    and row["selected_command"] is not None):
                                raise ValueError("frozen probe did not validate a candidate")
                            previous = selected[tick - 1].copy()
                            old_command = selected[tick].copy()
                            candidate = np.asarray(row["selected_command"], dtype=np.float64)
                            reference = reference_q[step - 1].copy()
                            common = (model, robot, evaluator, envelope,
                                      replay.qpos.copy(), replay.qvel.copy(),
                                      float(replay.time), reference, previous)
                            kwargs = {"physics_period_s": run_cfg["physics_period_s"],
                                      "task_period_s": run_cfg["task_period_s"]}
                            old, old_data = _branch(*common, old_command, **kwargs)
                            candidate_result, candidate_data = _branch(
                                *common, candidate, **kwargs)
                            old_torque_error = float(np.max(np.abs(
                                old["torques"] - torque[step:step + STEPS])))
                            old_next_qpos_error = float(np.max(np.abs(
                                old_data.qpos - task_qpos[tick + 1])))
                            if old_torque_error > 1e-8 or old_next_qpos_error > 1e-8:
                                failures_out.write(json.dumps({
                                    "mode": mode, "scenario_id": scenario_id,
                                    "tick": tick,
                                    "reason": "HISTORICAL_SERVO_PARITY_FAILED",
                                    "old_torque_max_abs_error": old_torque_error,
                                    "old_next_qpos_max_abs_error": old_next_qpos_error,
                                }) + "\n")
                                raise ValueError(f"historical servo parity failed: {key}")
                            next_start = _next_start(
                                model, robot, verifier, cfg, evaluator,
                                candidate_data, candidate,
                            )
                            geometry = verifier.verify_qpos_sequence(
                                candidate_result["qpos_states"])
                            coverage = candidate_result["coverage"]
                            record = {
                                "mode": mode, "scenario_id": scenario_id,
                                "tick": tick,
                                "command_l2_difference_rad_s": float(np.linalg.norm(
                                    candidate - old_command)),
                                "old_torque_max_abs_error": old_torque_error,
                                "old_next_qpos_max_abs_error": old_next_qpos_error,
                                "native_replay_max_state_error": replay_error,
                                "candidate_next_qpos_linf_difference_from_old": float(
                                    np.max(np.abs(candidate_data.qpos - old_data.qpos))),
                                "candidate_torque_saturation_count": candidate_result[
                                    "torque_saturation_count"],
                                "candidate_covered_microstate_count": sum(
                                    item["status"] == "COVERED_AT_THIS_STATE"
                                    for item in coverage),
                                "candidate_minimum_envelope_margin_m": min(
                                    item["minimum_margin_m"] for item in coverage),
                                "candidate_whole_body_minimum_distance_m": (
                                    geometry.minimum_clearance),
                                "candidate_whole_body_clearance_violation_count": (
                                    geometry.violation_count),
                                "realized_next_start": next_start,
                            }
                            report["records"].append(record)
                            for item in coverage:
                                steps_out.write(json.dumps({
                                    "mode": mode, "scenario_id": scenario_id,
                                    "tick": tick, **item,
                                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                            if (record["candidate_covered_microstate_count"] != STEPS + 1
                                    or geometry.violation_count
                                    or next_start["frozen_rows_status"]
                                    != "START_ROWS_SATISFIED"
                                    or next_start["proxy_status"]
                                    != "PROXY_CLEARANCE_AT_LEAST_GATE"):
                                failures_out.write(json.dumps(record, ensure_ascii=False,
                                                              separators=(",", ":")) + "\n")
                            print(f"[b2-candidate-ramp] {mode} {scenario_id} tick {tick}: "
                                  f"old torque error={old_torque_error:.2e}, "
                                  f"next={next_start['frozen_rows_status']}", flush=True)
                    if step < STEPS * TICKS[-1]:
                        replay.ctrl[:] = torque[step]
                        mujoco.mj_step(model, replay)
    if len(report["records"]) != 30:
        raise ValueError("candidate ramp population incomplete")
    for mode in ("baseline", "enabled"):
        own = [item for item in report["records"] if item["mode"] == mode]
        report["modes"][mode] = {
            "ramp_count": len(own),
            "covered_ramp_count": sum(
                item["candidate_covered_microstate_count"] == STEPS + 1 for item in own),
            "realized_next_start_satisfied_count": sum(
                item["realized_next_start"]["frozen_rows_status"]
                == "START_ROWS_SATISFIED" for item in own),
            "next_proxy_safe_count": sum(
                item["realized_next_start"]["proxy_status"]
                == "PROXY_CLEARANCE_AT_LEAST_GATE" for item in own),
            "whole_body_violation_ramp_count": sum(
                item["candidate_whole_body_clearance_violation_count"] > 0
                for item in own),
            "minimum_envelope_margin_m": min(
                item["candidate_minimum_envelope_margin_m"] for item in own),
            "maximum_old_torque_error": max(item["old_torque_max_abs_error"]
                                            for item in own),
            "maximum_old_next_qpos_error": max(item["old_next_qpos_max_abs_error"]
                                                for item in own),
            "next_start_status_counts": dict(Counter(
                item["realized_next_start"]["frozen_rows_status"] for item in own)),
        }
    report["microstate_records_sha256"] = _sha(steps_path)
    report["failure_records_sha256"] = _sha(failures_path)
    summary_path = output_dir / "candidate_ramp_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 早期 QP 候选的单周期私有力矩分支", "",
        "从旧 A.1 两组五场景 tick 50/100/150 的冻结状态分支。旧命令分支用"
        "原十步参考与 67 路力矩伺服逐步核对保存 torque，并核对下一规划 qpos；"
        "候选分支在私有 MuJoCo 状态中执行相同十步伺服。此操作不改变旧轨迹，"
        "也没有接入新区间在线模式。", "",
        "| 模式 | 斜坡 | 11 个状态均包含 | 下一起点区间行满足 | 下一起点代理安全 | 全身几何违例窗口 | 最小包络余量 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        lines.append(
            f"| {mode} | {item['ramp_count']} | {item['covered_ramp_count']} | "
            f"{item['realized_next_start_satisfied_count']} | "
            f"{item['next_proxy_safe_count']} | "
            f"{item['whole_body_violation_ramp_count']} | "
            f"{1000*item['minimum_envelope_margin_m']:.3f} |"
        )
    lines += [
        "", "全身几何验证会在保存的 11 个状态间再作原验证器的插值检查；"
        "这是有限离散/插值诊断，不构成连续时间证明。下一起点用候选执行后的"
        "真实 MuJoCo 状态重新查询 255 点根区间并独立重建原 MuJoCo、胶囊"
        "及新区间起点行；与冻结时的前瞻预测不可混为一谈。", "",
        "仅覆盖 30 个离线分支和一个规划周期，未测新区间闭环累计误差、"
        "全链 20 ms、未知时的停止行为或五场景新 trace。"
        "无论本表是否通过，阶段三接入门禁仍需单独判定。", "",
    ]
    doc_path = output_dir / "CANDIDATE_RAMP.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_one_ramp_candidate_branch_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("microstates", steps_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "candidate_ramp_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--probe", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json"))
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.probe)
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
