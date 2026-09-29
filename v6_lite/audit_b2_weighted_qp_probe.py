"""Read-only weighted-QP probe on predeclared early A.1 replay states.

The probe calls the existing 17-D weighted solver and action validator once
per frozen historical state. It never applies the resulting command. This is
stage-2 diagnostic evidence, not a new-mode closed-loop run or an admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_shadow_feasibility import combined_rows, ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import HierarchicalQPConfig, HierarchicalVelocityQP
from v6_lite.pcc_interval_cbf import (
    SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD, FixedIntervalCBFEvaluator,
    IntervalPartition,
)
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, _body_pose_and_twist, build_scenarios,
    default_v6_lite_robot_spec,
)
from v6_lite.shadow_b2_interval_cbf import _summary, select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


PROBE_TICKS = (50, 100, 150)
POINT_BUDGET = 63


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ReadOnlyIntervalQP(HierarchicalVelocityQP):
    """Use the unmodified weighted solve with independently selectable rows."""

    def __init__(self, *args, evaluator: FixedIntervalCBFEvaluator, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interval_evaluator = evaluator
        self.partition: IntervalPartition | None = None
        self.selected_ids: set[str] = set()
        self.last_batch = None

    def _build_all_clearance_constraints(self, data, generalized_map):
        original = super()._build_all_clearance_constraints(data, generalized_map)
        if self.partition is None:
            raise RuntimeError("the offline interval partition was not selected")
        batch = self.interval_evaluator.evaluate_state(
            data, self.partition, generalized_map=generalized_map,
            derivative_interval_ids=self.selected_ids,
        )
        self.last_batch = batch
        if (not batch.interval_well_formed or not batch.coverage_complete
                or not batch.analytic_bound_assumptions_satisfied):
            raise RuntimeError("the offline interval cover is unsupported")
        rows = [row for row in batch.rows if row.interval_id in self.selected_ids]
        if len(rows) != len(self.selected_ids) or any(
            row.derivative_status != "SUPPORTED" for row in rows
        ):
            raise RuntimeError("the offline QP lacks a required interval derivative")
        gain = self.config.pcc_clearance_barrier_gain
        return replace(
            original,
            matrix=np.vstack((original.matrix,
                              *(row.generalized_gradient.reshape(1, 17) for row in rows))),
            lower=np.concatenate((original.lower, np.asarray(
                [-gain * row.h_m - row.target_drift_m_s for row in rows]))),
            sources=(*original.sources, *(row.interval_id for row in rows)),
            distances_m=np.concatenate((original.distances_m, np.asarray(
                [row.distance_lower_bound_m for row in rows]))),
            target_drifts_m_s=np.concatenate((original.target_drifts_m_s, np.asarray(
                [row.target_drift_m_s for row in rows]))),
            barrier_gains_s_inv=np.concatenate((original.barrier_gains_s_inv,
                                                 np.full(len(rows), gain))),
            minimum_clearance_m=min(
                original.minimum_clearance_m,
                *(row.distance_lower_bound_m for row in rows),
            ),
        )


def _row_parity(result, original, batch, selected_ids, cfg) -> dict:
    matrix, lower, drifts, gains, sources = combined_rows(
        original, batch, selected_ids, cfg,
    )
    if len(set(sources)) != len(sources) or len(set(result.clearance_sources)) != len(
        result.clearance_sources
    ) or set(sources) != set(result.clearance_sources):
        raise ValueError("weighted QP and independent recomputation sources differ")
    positions = {name: i for i, name in enumerate(result.clearance_sources)}
    row_error = max((float(np.max(np.abs(row - result.clearance_matrix[positions[name]])))
                     for name, row in zip(sources, matrix)), default=0.0)
    lower_error = max((abs(float(value - result.clearance_lower[positions[name]]))
                       for name, value in zip(sources, lower)), default=0.0)
    drift_error = max((abs(float(value - result.clearance_target_drift_m_s[
        positions[name]])) for name, value in zip(sources, drifts)), default=0.0)
    gain_error = max((abs(float(value - result.clearance_barrier_gain_s_inv[
        positions[name]])) for name, value in zip(sources, gains)), default=0.0)
    return {"source_count": len(sources), "matrix_max_abs_error": row_error,
            "lower_max_abs_error": lower_error,
            "drift_max_abs_error": drift_error, "gain_max_abs_error": gain_error}


def run(output_dir: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    records = []
    native_checks = []
    inputs = {}
    for mode in ("baseline", "enabled"):
        root = a1_root / f"{mode}_root" / "output"
        metrics_path = root / "v6_lite_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = HierarchicalQPConfig(**metrics["qp_config"])
        run_cfg = V6LiteRunConfig(**metrics["run_config"])
        scenarios = {item.scenario_id: item for item in build_scenarios(robot, run_cfg)}
        inputs[mode] = {"metrics_path": metrics_path.as_posix(),
                        "metrics_sha256": _sha(metrics_path), "traces": []}
        for item in metrics["scenarios"]:
            scenario_id = item["scenario"]["scenario_id"]
            scenario = scenarios[scenario_id]
            if scenario.seed != item["scenario"]["seed"]:
                raise ValueError("generated scenario does not match frozen seed")
            verifier = WholeBodyCollisionVerifier(
                robot, _obstacles(item["scenario"]), WholeBodyVerificationConfig(
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
            evaluator = FixedIntervalCBFEvaluator(robot, model)
            no_legacy_pcc = replace(cfg, enable_pcc_cbf=False,
                                    enable_capsule_cbf=True)
            qp = _ReadOnlyIntervalQP(robot, model, verifier.pairs, no_legacy_pcc,
                                     evaluator=evaluator)
            independent = ReplayConstraintBuilder(
                robot, model, verifier.pairs, no_legacy_pcc,
            )
            query = PersistentIntervalDecisionQuery(evaluator.shape_model)
            target_body_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite"))
            trace_path = Path(item["trace"]["path"])
            trace_sha = _sha(trace_path)
            if trace_sha != item["trace"]["sha256"]:
                raise ValueError(f"A.1 trace hash mismatch: {trace_path}")
            inputs[mode]["traces"].append({"path": trace_path.as_posix(),
                                            "sha256": trace_sha})
            with np.load(trace_path, allow_pickle=False) as trace:
                initial_qpos = trace["initial_qpos"].copy()
                initial_qvel = trace["initial_qvel"].copy()
                torque = trace["torque"][:PROBE_TICKS[-1] * 10 + 1].copy()
                selected = trace["task_selected_command"].copy()
                task_qpos = trace["task_qpos"].copy()
                task_time = trace["task_time"].copy()
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            max_state_error = 0.0
            for step, control in enumerate(torque):
                if step % 10 == 0:
                    tick = step // 10
                    mujoco.mj_forward(model, data)
                    max_state_error = max(
                        max_state_error,
                        float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                        abs(float(data.time - task_time[tick])),
                    )
                    if tick in PROBE_TICKS:
                        started = time.perf_counter()
                        projection = evaluator.shape_spec.project_actual_configuration(
                            data.qpos[evaluator.qpos_ids[:60]])
                        base = transform_from_free_qpos(
                            data.qpos[evaluator.base_qpos_slice])
                        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
                        decision = query.evaluate(
                            projection.planner_configuration, base, box,
                            IntervalPartition.uniform(),
                            max_point_evaluations=POINT_BUDGET,
                        )
                        previous = np.zeros(17) if tick == 0 else selected[tick - 1]
                        planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
                        _vl, _vu, ramp_speed, box_valid = ramp_velocity_abs_bound(
                            robot, cfg, planner_q, previous,
                        )
                        generalized_map, _ = qp.reaction_velocity_map(data)
                        selected_ids, _ = select_intervals_by_frozen_reach(
                            decision.lower_by_interval_id, decision.partition,
                            evaluator, data, cfg, generalized_map,
                            velocity_abs_bound=ramp_speed,
                        )
                        qp.partition = decision.partition
                        qp.selected_ids = selected_ids
                        qp.previous_velocity = previous.copy()
                        qp._previous_constraint_dual.clear()
                        (rigid_target, rigid_velocity, target_rotation,
                         target_angular_velocity) = _body_pose_and_twist(
                            model, data, target_body_id,
                            scenario.grasp_point_target_frame_m,
                        )
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
                        probe_ms = (time.perf_counter() - started) * 1000.0
                        original_rows = independent.build(data)
                        parity = _row_parity(result, original_rows, qp.last_batch,
                                             selected_ids, cfg)
                        geometry_components_ms = (
                            decision.elapsed_ms
                            + qp.last_batch.total_query_time_ms
                        )
                        query_budget_ok = (
                            decision.point_evaluation_count
                            + qp.last_batch.point_evaluation_count <= 128
                            and qp.last_batch.jacobian_evaluation_count <= 96
                            and geometry_components_ms <= 20.0
                        )
                        proxy_safe = (decision.bounds_valid
                                      and decision.proxy_clearance_status
                                      == "PROXY_CLEARANCE_AT_LEAST_GATE"
                                      and qp.last_batch.all_intervals_safe)
                        envelope_supported = (
                            qp.last_batch.geometry_domain_status
                            == ("INSIDE_DECLARED_WORK_DOMAIN;"
                                "ON_DECLARED_SHAPE_SUBSPACE")
                            and qp.last_batch.envelope_evidence_status
                            != "NO_ENVELOPE_EVIDENCE_FOR_THIS_STATE"
                        )
                        admission_preconditions_met = (
                            box_valid and query_budget_ok and proxy_safe
                            and envelope_supported
                        )
                        records.append({
                            "mode": mode, "scenario_id": scenario_id, "tick": tick,
                            "query_status": decision.proxy_clearance_status,
                            "query_points": decision.point_evaluation_count,
                            "selected_interval_count": len(selected_ids),
                            "geometry_domain_status":
                                qp.last_batch.geometry_domain_status,
                            "envelope_evidence_status":
                                qp.last_batch.envelope_evidence_status,
                            "floating_point_certification":
                                qp.last_batch.floating_point_certification,
                            "subspace_residual_linf_rad":
                                qp.last_batch.subspace_residual_linf_rad,
                            "proxy_safe": proxy_safe,
                            "query_budget_ok": query_budget_ok,
                            "envelope_supported": envelope_supported,
                            "admission_preconditions_met": admission_preconditions_met,
                            "geometry_components_ms": geometry_components_ms,
                            "row_parity": parity,
                            "solver_status": result.solver_status,
                            "solver_iterations": result.solver_iterations,
                            "solver_candidate": result.solver_candidate.tolist(),
                            "action_mode": result.action_validation.mode.value,
                            "failure_reason": result.action_validation.failure_reason.value,
                            "candidate_valid": result.action_validation.candidate_valid,
                            "ramp_valid": result.action_validation.ramp_valid,
                            "selected_command": (None if result.planner_velocity is None
                                                 else result.planner_velocity.tolist()),
                            "interval_assembly_ms": qp.last_batch.total_query_time_ms,
                            "qp_full_ms": result.full_latency_s * 1000.0,
                            "qp_solver_ms": result.solver_latency_s * 1000.0,
                            "query_plus_qp_probe_ms": probe_ms,
                        })
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
            native_checks.append({"mode": mode, "scenario_id": scenario_id,
                                  "trace_sha256": trace_sha,
                                  "max_state_error": max_state_error})
            print(f"[b2-qp-probe] {mode} {scenario_id}: 3 frozen states")
    expected = 2 * 5 * len(PROBE_TICKS)
    if len(records) != expected or len(native_checks) != 10:
        raise ValueError("the predeclared QP probe matrix is incomplete")
    integrity = (all(item["max_state_error"] <= 1e-8 for item in native_checks)
                 and all(item["row_parity"][key] <= 1e-7
                         for item in records for key in (
                             "matrix_max_abs_error", "lower_max_abs_error",
                             "drift_max_abs_error", "gain_max_abs_error")))
    sources = {name: _sha(Path("v6_lite") / name) for name in (
        "audit_b2_weighted_qp_probe.py", "hierarchical_qp.py",
        "pcc_interval_cbf.py", "continuum_shape_model.py",
        "recompute_execution_constraints.py", "b2_shadow_feasibility.py",
        "safety_contract.py",
    )}
    summary = {}
    for mode in ("baseline", "enabled"):
        own = [item for item in records if item["mode"] == mode]
        summary[mode] = {
            "probe_count": len(own),
            "query_budget_ok_count": sum(item["query_budget_ok"] for item in own),
            "proxy_safe_count": sum(item["proxy_safe"] for item in own),
            "envelope_supported_count": sum(item["envelope_supported"] for item in own),
            "admission_preconditions_met_count": sum(
                item["admission_preconditions_met"] for item in own),
            "validated_command_count": sum(item["selected_command"] is not None
                                           for item in own),
            "failure_reasons": dict(Counter(item["failure_reason"] for item in own)),
            "subspace_residual_linf_rad": _summary(
                [item["subspace_residual_linf_rad"] for item in own]),
            "interval_assembly_ms": _summary(
                [item["interval_assembly_ms"] for item in own]),
            "qp_full_ms": _summary([item["qp_full_ms"] for item in own]),
            "qp_solver_ms": _summary([item["qp_solver_ms"] for item in own]),
            "query_plus_qp_probe_ms": _summary(
                [item["query_plus_qp_probe_ms"] for item in own]),
        }
    report = {
        "schema": "v6_2_b2_weighted_qp_probe_v1",
        "scope": "read_only_early_historical_states_no_new_execution",
        "online_control_changed": False,
        "new_mode_closed_loop_acceptance": False,
        "probe_ticks": list(PROBE_TICKS),
        "solver_dual_reused_between_probes": False,
        "ramp_start_source": "historical_previous_selected_command",
        "point_budget": POINT_BUDGET,
        "shape_subspace_membership_tolerance_rad":
            SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD,
        "passed_as_read_only_integrity": integrity,
        "source_sha256": sources,
        "input_refined_start_sha256": _sha(
            Path("v6_lite/output/v6_2_b2/refined_start/refined_start_report.json")),
        "inputs": inputs,
        "native_replay_checks": native_checks,
        "summary": summary,
        "records": records,
    }
    report_path = output_dir / "weighted_qp_probe.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 早期冻结状态：原 17 维加权 QP 只读探针", "",
        "旧 trace 原生 500 Hz 力矩重放至固定的规划 tick 50/100/150；"
        "在这些状态上构造根分区的区间 PCC 行，与原 MuJoCo 和胶囊行一起交给"
        "原加权 QP 求解与执行验证。候选和验证结果仅记录，不执行。", "",
        f"重放及独立逐行一致性：{'通过' if integrity else '未通过'}。"
        "这不是新区间模式闭环或 20 ms 全链验收。", "",
        "| 模式 | 冻结状态 | 查询预算合格 | 代理静态安全 | 包络证据支持 | 验证出命令 | 接入前提合格 | 查询加 QP 探针 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = summary[mode]
        lines.append(
            f"| {mode} | {item['probe_count']} | {item['query_budget_ok_count']} | "
            f"{item['proxy_safe_count']} | {item['envelope_supported_count']} | "
            f"{item['validated_command_count']} | "
            f"{item['admission_preconditions_met_count']} | "
            f"{item['query_plus_qp_probe_ms']['p95']:.3f} |"
        )
    lines += [
        "", "所有失败、候选、行对照及耗时保存在 JSON。探针只覆盖历史轨迹的"
        "早期冻结状态，且每次清空求解器对偶热启动。QP 候选通过验证"
        "不等于代理包络得到真实链支持；严格子空间判据保持原值。"
        "本探针也不改变后续旧轨迹上已观察到的代理低于门槛、"
        "起点违规和持久查询未知，也不构成第 3 阶段接入许可。", "",
    ]
    doc_path = output_dir / "WEIGHTED_QP_PROBE.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {"schema": "v6_2_b2_weighted_qp_probe_manifest_v1",
                "report_sha256": _sha(report_path), "report_bytes": report_path.stat().st_size,
                "document_sha256": _sha(doc_path), "document_bytes": doc_path.stat().st_size}
    with (output_dir / "weighted_qp_probe_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root)
    print(json.dumps({"passed_as_read_only_integrity": report[
        "passed_as_read_only_integrity"], "summary": report["summary"]}, indent=2))
    if not report["passed_as_read_only_integrity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
