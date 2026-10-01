"""Private interval-QP rollout with pre-servo domain and ramp rejection.

This script starts from a published A.1 initial state, then uses its *own*
validated interval-QP commands in a private MuJoCo model. It does not switch
the production controller or confer online admission: the declared shape
subspace, future tube containment and full-cycle latency gates remain separate.
At any failed preflight, QP validation or independently recomputed next start,
the private model performs no further servo step and preserves the failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_candidate_ramp import _branch, _next_start
from v6_lite.audit_b2_weighted_qp_probe import _ReadOnlyIntervalQP, _row_parity
from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import advance_reference
from v6_lite.hierarchical_qp import HierarchicalQPConfig, joint_addresses
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _body_pose_and_twist, build_scenarios,
    default_v6_lite_robot_spec,
)
from v6_lite.safety_contract import command_is_current
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


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


def _run_impl(output_dir: Path, a1_root: Path, scenario_id: str,
              max_ticks: int) -> dict:
    if max_ticks < 1:
        raise ValueError("max_ticks must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    mode = "enabled"
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = next((item for item in metrics["scenarios"]
                  if item["scenario"]["scenario_id"] == scenario_id), None)
    if saved is None:
        raise ValueError(f"unknown frozen scenario: {scenario_id}")
    scenario = next(item for item in build_scenarios(robot, run_cfg)
                    if item.scenario_id == scenario_id)
    if scenario.seed != saved["scenario"]["seed"]:
        raise ValueError("scenario seed does not match published A.1 run")
    trace_path = Path(saved["trace"]["path"])
    trace_sha = _sha(trace_path)
    if trace_sha != saved["trace"]["sha256"]:
        raise ValueError("published A.1 trace hash changed")
    with np.load(trace_path, allow_pickle=False) as frozen:
        initial_qpos = frozen["initial_qpos"].copy()
        initial_qvel = frozen["initial_qvel"].copy()
        historical_task_qpos = frozen["task_qpos"][:max_ticks + 1].copy()
    if (len(historical_task_qpos) != max_ticks + 1
            or np.max(np.abs(initial_qpos - historical_task_qpos[0])) > 1e-8):
        raise ValueError("published initial state differs from task tick zero")
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
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
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
    no_legacy = replace(cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
    qp = _ReadOnlyIntervalQP(robot, model, verifier.pairs, no_legacy,
                             evaluator=evaluator)
    independent = ReplayConstraintBuilder(robot, model, verifier.pairs, no_legacy)
    query = PersistentIntervalDecisionQuery(evaluator.shape_model)
    target_body = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite"))
    qpos_ids, _dof_ids = joint_addresses(model, robot)
    previous = np.zeros(17, dtype=np.float64)
    reference = robot.planner_zero.copy()
    records = []
    torques = []
    qpos_states = [data.qpos.copy()]
    task_qvel_states = [data.qvel.copy()]
    stop_reason = "HORIZON_COMPLETE"
    failure = None
    for tick in range(max_ticks):
        started = time.perf_counter()
        mujoco.mj_forward(model, data)
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        current_envelope = envelope.evaluate(
            data, projection.planner_configuration, base)
        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
        decision = query.evaluate(
            projection.planner_configuration, base, box,
            IntervalPartition.uniform(), max_point_evaluations=POINT_BUDGET,
        )
        row = {
            "tick": tick, "time_s": float(data.time),
            "proxy_status": decision.proxy_clearance_status,
            "proxy_lower_m": decision.distance_lower_bound_m,
            "proxy_upper_m": decision.distance_upper_bound_m,
            "point_evaluations": decision.point_evaluation_count,
            "query_budget_exhausted": decision.budget_exhausted,
            "subspace_residual_linf_rad": projection.residual_linf_rad,
            "current_envelope_status": current_envelope.status,
            "current_envelope_margin_m": current_envelope.min_margin_m,
            "old_a1_state_reused_after_initialization": False,
        }
        if (not decision.bounds_valid or decision.proxy_clearance_status
                != "PROXY_CLEARANCE_AT_LEAST_GATE"):
            stop_reason = "PROXY_QUERY_UNSUPPORTED_OR_NOT_SAFE"
            failure = row
            records.append(row)
            break
        if current_envelope.status != "COVERED_AT_THIS_STATE":
            stop_reason = "ACTUAL_CHAIN_NOT_ENVELOPED_AT_CURRENT_STATE"
            failure = row
            records.append(row)
            break
        planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
        _lower, _upper, speed, box_valid = ramp_velocity_abs_bound(
            robot, cfg, planner_q, previous)
        generalized_map, _ = qp.reaction_velocity_map(data)
        ids, excluded = select_intervals_by_frozen_reach(
            decision.lower_by_interval_id, decision.partition, evaluator,
            data, cfg, generalized_map, velocity_abs_bound=speed,
        )
        row.update({"velocity_box_valid": box_valid,
                    "selected_interval_count": len(ids),
                    "excluded_interval_count": len(excluded)})
        if not box_valid:
            stop_reason = "RAMP_VELOCITY_BOX_INVALID"
            failure = row
            records.append(row)
            break
        qp.partition = decision.partition
        qp.selected_ids = ids
        (rigid_target, rigid_velocity, target_rotation,
         target_angular_velocity) = _body_pose_and_twist(
            model, data, target_body, scenario.grasp_point_target_frame_m)
        continuum_target, continuum_velocity = scenario.continuum_target.sample(
            float(data.time))
        result = qp.solve(
            data,
            rigid_target_position=rigid_target,
            rigid_target_velocity=rigid_velocity,
            rigid_target_rotation=(target_rotation
                                   @ scenario.grasp_rotation_target_frame),
            rigid_target_angular_velocity=target_angular_velocity,
            continuum_target_position=continuum_target,
            continuum_target_velocity=continuum_velocity,
            continuum_target_rotation=scenario.continuum_target_rotation_world,
            continuum_target_angular_velocity=np.zeros(3),
            state_timestamp_s=float(data.time),
            target_timestamp_s=float(data.time),
            ramp_start_velocity=previous,
        )
        preflight_qp_ms = (time.perf_counter() - started) * 1000.0
        recompute_started = time.perf_counter()
        original_rows = independent.build(data)
        parity = _row_parity(result, original_rows, qp.last_batch, ids, cfg)
        independent_recompute_ms = (time.perf_counter() - recompute_started) * 1000.0
        row.update({
            "interval_rows": len(ids), "row_parity": parity,
            "geometry_domain_status": qp.last_batch.geometry_domain_status,
            "analytic_bound_assumptions_satisfied":
                qp.last_batch.analytic_bound_assumptions_satisfied,
            "envelope_evidence_status": qp.last_batch.envelope_evidence_status,
            "strict_online_domain_met": (
                qp.last_batch.geometry_domain_status
                == "INSIDE_DECLARED_WORK_DOMAIN;ON_DECLARED_SHAPE_SUBSPACE"),
            "qp_solver_status": result.solver_status,
            "qp_iterations": result.solver_iterations,
            "action_mode": result.action_validation.mode.value,
            "failure_reason": result.action_validation.failure_reason.value,
            "candidate_valid": result.action_validation.candidate_valid,
            "ramp_valid": result.action_validation.ramp_valid,
            "selected_command": (None if result.planner_velocity is None
                                 else result.planner_velocity.tolist()),
            "qp_full_ms": result.full_latency_s * 1000.0,
            "preflight_plus_qp_ms": preflight_qp_ms,
            "independent_row_recompute_ms": independent_recompute_ms,
        })
        if not row["strict_online_domain_met"]:
            row.update({"action_mode": "UNCERTIFIED",
                        "failure_reason": "work_domain_unsupported",
                        "rejected_candidate_command": row["selected_command"],
                        "selected_command": None,
                        "next_servo_step_executed": False})
            stop_reason = "CURRENT_STATE_OUTSIDE_DECLARED_WORK_DOMAIN"
            failure = row
            records.append(row)
            break
        if (not qp.last_batch.interval_well_formed
                or not qp.last_batch.coverage_complete
                or not qp.last_batch.analytic_bound_assumptions_satisfied
                or any(parity[key] > 1e-7 for key in (
                    "matrix_max_abs_error", "lower_max_abs_error",
                    "drift_max_abs_error", "gain_max_abs_error"))):
            stop_reason = "INTERVAL_OR_INDEPENDENT_ROW_UNSUPPORTED"
            failure = row
            records.append(row)
            break
        if result.planner_velocity is None:
            stop_reason = "QP_CANDIDATE_NOT_VALIDATED"
            failure = row
            records.append(row)
            break
        if any(not command_is_current(
                result.action_validation, float(data.time) + i * run_cfg.physics_period_s)
               for i in range(STEPS)):
            stop_reason = "COMMAND_EXPIRED_BEFORE_SERVO_STEP"
            failure = row
            records.append(row)
            break
        endpoint = result.planner_velocity.copy()
        before_reference = reference.copy()
        branch, next_data = _branch(
            model, robot, evaluator, envelope,
            data.qpos.copy(), data.qvel.copy(), float(data.time),
            before_reference, previous, endpoint,
            physics_period_s=run_cfg.physics_period_s,
            task_period_s=run_cfg.task_period_s,
        )
        domain_failures = []
        for step, predicted_qpos in enumerate(branch["qpos_states"]):
            projected = evaluator.shape_spec.project_actual_configuration(
                predicted_qpos[evaluator.qpos_ids[:60]])
            q_shape = projected.planner_configuration
            lower = evaluator.shape_spec.work_domain_lower_rad
            upper = evaluator.shape_spec.work_domain_upper_rad
            if (not np.all(np.isfinite(q_shape))
                    or np.any(q_shape < lower)
                    or np.any(q_shape > upper)):
                domain_failures.append({
                    "servo_substep": step,
                    "maximum_upper_excess_rad": float(np.max(q_shape - upper)),
                    "maximum_lower_excess_rad": float(np.max(lower - q_shape)),
                })
        row["ramp_domain_failures"] = domain_failures
        row["minimum_ramp_envelope_margin_m"] = min(
            item["minimum_margin_m"] for item in branch["coverage"])
        row["ramp_covered_microstates"] = sum(
            item["status"] == "COVERED_AT_THIS_STATE"
            for item in branch["coverage"])
        if domain_failures or row["ramp_covered_microstates"] != STEPS + 1:
            row.update({"action_mode": "UNCERTIFIED",
                        "failure_reason": (
                            "ramp_work_domain_unsupported" if domain_failures
                            else "ramp_microstate_not_enveloped"),
                        "rejected_candidate_command": row["selected_command"],
                        "selected_command": None,
                        "next_servo_step_executed": False})
            stop_reason = ("RAMP_MICROSTATE_OUTSIDE_DECLARED_WORK_DOMAIN"
                           if domain_failures else "RAMP_MICROSTATE_NOT_ENVELOPED")
            failure = row
            records.append(row)
            break
        next_start = _next_start(model, robot, verifier, cfg,
                                 evaluator, next_data, endpoint)
        row["realized_next_start"] = next_start
        if (next_start["proxy_status"] != "PROXY_CLEARANCE_AT_LEAST_GATE"
                or next_start["frozen_rows_status"] != "START_ROWS_SATISFIED"):
            row.update({"action_mode": "UNCERTIFIED",
                        "failure_reason": "realized_next_start_unsupported",
                        "rejected_candidate_command": row["selected_command"],
                        "selected_command": None,
                        "next_servo_step_executed": False})
            stop_reason = "REALIZED_NEXT_START_UNSUPPORTED"
            failure = row
            records.append(row)
            break
        for step in range(STEPS):
            measured = robot.decode_position(branch["qpos_states"][step][qpos_ids])
            reference = advance_reference(
                reference, previous, endpoint, step + 1, measured,
                robot.planner_lower, robot.planner_upper,
                physics_period_s=run_cfg.physics_period_s,
                task_period_s=run_cfg.task_period_s,
            ).position
        torques.extend(branch["torques"])
        qpos_states.extend(branch["qpos_states"][1:])
        task_qvel_states.append(next_data.qvel.copy())
        row["next_qpos_linf_difference_from_old_a1"] = float(np.max(np.abs(
            next_data.qpos - historical_task_qpos[tick + 1])))
        records.append(row)
        data = next_data
        previous = endpoint
    records_path = output_dir / "private_rollout_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    trace_out = output_dir / "private_rollout_trace.npz"
    np.savez_compressed(trace_out, initial_qpos=initial_qpos,
                        initial_qvel=initial_qvel,
                        torque=np.asarray(torques),
                        qpos_states=np.asarray(qpos_states),
                        task_qvel_states=np.asarray(task_qvel_states))
    # Recreate the private trace from the saved 67-channel torques.
    replay = mujoco.MjData(model)
    replay.qpos[:] = initial_qpos
    replay.qvel[:] = initial_qvel
    replay.ctrl[:] = 0
    native_error = 0.0
    for step, torque in enumerate(torques):
        native_error = max(native_error, float(np.max(np.abs(
            replay.qpos - qpos_states[step]))))
        if step % STEPS == 0:
            native_error = max(native_error, float(np.max(np.abs(
                replay.qvel - task_qvel_states[step // STEPS]))))
        replay.ctrl[:] = torque
        mujoco.mj_step(model, replay)
    native_error = max(native_error, float(np.max(np.abs(
        replay.qpos - qpos_states[-1]))))
    native_error = max(native_error, float(np.max(np.abs(
        replay.qvel - task_qvel_states[-1]))))
    timed = [row["preflight_plus_qp_ms"] for row in records
             if "preflight_plus_qp_ms" in row]
    overrun = [value > 20.0 for value in timed]
    longest_overrun = 0
    current_overrun = 0
    for exceeded in overrun:
        current_overrun = current_overrun + 1 if exceeded else 0
        longest_overrun = max(longest_overrun, current_overrun)
    timing = ({"count": len(timed), "p95_ms": float(np.percentile(timed, 95)),
               "p99_ms": float(np.percentile(timed, 99)),
               "max_ms": max(timed), "over_20ms_count": sum(overrun),
               "longest_over_20ms_run": longest_overrun}
              if timed else None)
    report = {
        "schema": "v6_2_b2_strict_domain_private_multicycle_v1",
        "scope": "private_counterfactual_muJoCo_branch_with_pre_servo_domain_gate",
        "mode": mode, "scenario_id": scenario_id,
        "predeclared_horizon_ticks": max_ticks,
        "attempted_ticks": len(records),
        "executed_ticks": len(torques) // STEPS,
        "stop_reason": stop_reason, "failure_record": failure,
        "point_budget": POINT_BUDGET,
        "servo_steps_per_task": STEPS,
        "native_replay_max_qpos_error": native_error,
        "private_preflight_plus_qp_timing": timing,
        "strict_online_domain_all_executed_ticks": all(
            row.get("strict_online_domain_met", False)
            for row in records if row.get("selected_command") is not None),
        "first_tick_outside_strict_online_domain": next(
            (row["tick"] for row in records
             if row.get("strict_online_domain_met") is False), None),
        "first_preflight_domain_rejection_tick": next(
            (row["tick"] for row in records if row.get("ramp_domain_failures")),
            None),
        "maximum_subspace_residual_linf_rad": max(
            (row["subspace_residual_linf_rad"] for row in records), default=0.0),
        "minimum_ramp_envelope_margin_m": min(
            (row["minimum_ramp_envelope_margin_m"] for row in records
             if "minimum_ramp_envelope_margin_m" in row), default=None),
        "maximum_qpos_linf_difference_from_old_a1": max(
            (row["next_qpos_linf_difference_from_old_a1"] for row in records
             if "next_qpos_linf_difference_from_old_a1" in row), default=0.0),
        "minimum_realized_next_start_slack_m_s": min(
            (row["realized_next_start"]["minimum_start_slack_m_s"]
             for row in records if row.get("realized_next_start", {}).get(
                 "minimum_start_slack_m_s") is not None), default=None),
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_private_rollout.py", "audit_b2_weighted_qp_probe.py",
            "audit_b2_candidate_ramp.py", "hierarchical_qp.py",
            "pcc_interval_cbf.py", "safety_contract.py",
            "audit_b2_strict_domain_private_rollout.py",
        )},
        "inputs": {"metrics_path": metrics_path.as_posix(),
                   "metrics_sha256": _sha(metrics_path),
                   "published_trace_path": trace_path.as_posix(),
                   "published_trace_sha256": trace_sha},
        "records_sha256": _sha(records_path),
        "trace_sha256": _sha(trace_out),
    }
    report_path = output_dir / "private_rollout_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if failure is not None:
        failure_path = output_dir / "private_rollout_failure.json"
        failure_path.write_text(json.dumps({
            "stop_reason": stop_reason, "tick": failure["tick"],
            "last_record": failure, "next_servo_step_executed": False,
            "trace_sha256": report["trace_sha256"],
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")
    lines = [
        "# B.2 私有多周期区间 QP 力矩诊断", "",
        f"场景 `{scenario_id}` 从已发布 A.1 tick 0 状态出发；此后每周期用私有"
        "MuJoCo 模型上的新区间 QP 候选及原 67 路力矩伺服推进。"
        "历史 trace 只提供初始条件与场景目标。", "",
        f"预声明上限 {max_ticks} 周期；尝试 {len(records)}，执行 "
        f"{len(torques) // STEPS}；停止原因 `{stop_reason}`。"
        f"原生力矩重放最大状态误差 {native_error:.3e}。", "",
        "| 指标 | 结果 |", "| --- | ---: |",
        f"| 首次超出严格形状子空间的周期 | "
        f"{report['first_tick_outside_strict_online_domain']} |",
        f"| 最大形状子空间残差 rad | "
        f"{report['maximum_subspace_residual_linf_rad']:.3e} |",
        f"| 最小斜坡微状态 PCC 包络余量 m | "
        f"{report['minimum_ramp_envelope_margin_m']} |",
        f"| 相对旧 A.1 最大 qpos 差异 | "
        f"{report['maximum_qpos_linf_difference_from_old_a1']:.3e} |",
        f"| 重算下一起点最小约束松弛 m/s | "
        f"{report['minimum_realized_next_start_slack_m_s']} |",
        f"| 私有预检＋QP p95 / p99 / 最大 ms | "
        f"{timing['p95_ms']:.3f} / {timing['p99_ms']:.3f} / "
        f"{timing['max_ms']:.3f} |" if timing else
        "| 私有预检＋QP 计时 | 未运行 QP |",
        f"| 超 20 ms / 最长连续次数 | "
        f"{timing['over_20ms_count']} / {timing['longest_over_20ms_run']} |"
        if timing else "| 超 20 ms | 未运行 QP |",
        "", "私有分支的几何包络只在所测 500 Hz 状态成立；"
        "QP 诊断照样记录严格子空间不满足，未放宽原判据。"
        "上述预检计时含当前状态胶囊包络、区间查询和 QP，"
        "不含随后执行的独立约束重算、十步力矩伺服或下一起点诊断。"
        "本严格私有变体在提交力矩之前检查模拟斜坡的 11 个状态、"
        "工作域与下一起点，不合格候选明确拒绝且不执行下一伺服步。"
        "本试验没有接入在线控制，也不构成完整五场景、全链 20 ms "
        "或连续时间安全验收。", "",
    ]
    document_path = output_dir / "PRIVATE_ROLLOUT.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_private_rollout_manifest_v1",
        "summary_sha256": _sha(report_path),
        "records_sha256": _sha(records_path),
        "trace_sha256": _sha(trace_out),
        "document_sha256": _sha(document_path),
        "failure_sha256": (_sha(output_dir / "private_rollout_failure.json")
                           if failure is not None else None),
    }
    with (output_dir / "private_rollout_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    """Use the already selected private QP, query, envelope and servo variant."""
    from v6_lite import audit_b2_private_rollout as selected

    global _branch, _ReadOnlyIntervalQP
    global PersistentIntervalDecisionQuery, StateLocalPCCEnvelopeAudit
    _branch = selected._branch
    _ReadOnlyIntervalQP = selected._ReadOnlyIntervalQP
    PersistentIntervalDecisionQuery = selected.PersistentIntervalDecisionQuery
    StateLocalPCCEnvelopeAudit = selected.StateLocalPCCEnvelopeAudit
    return _run_impl(output_dir, a1_root, scenario_id, max_ticks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_00")
    parser.add_argument("--max-ticks", type=int, default=250)
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.scenario_id,
                 args.max_ticks)
    print(json.dumps({key: report[key] for key in (
        "scenario_id", "attempted_ticks", "executed_ticks", "stop_reason",
        "native_replay_max_qpos_error", "strict_online_domain_all_executed_ticks",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
