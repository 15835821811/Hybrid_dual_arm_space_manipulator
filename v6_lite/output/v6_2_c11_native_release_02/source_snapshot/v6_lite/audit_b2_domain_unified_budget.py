"""Read-only unified budget audit bound to a new domain-QP private replay.

The audit rebuilds query partitions and interval derivatives from saved task
states. It does not send commands, replay torques, or alter the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_private_recompute import _obstacles
from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import ShadowBudget, select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


TICKS = 1350
QUERY_POINT_LIMIT = 255
FROZEN_BUDGET = ShadowBudget()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max())}


def assess_budget(branch_points: int, interval_points: int,
                  local_points: int, jacobians: int, elapsed_ms: float,
                  budget: ShadowBudget) -> list[str]:
    counts = (branch_points, interval_points, local_points, jacobians)
    if (any(not isinstance(value, int) or value < 0 for value in counts)
            or not math.isfinite(elapsed_ms) or elapsed_ms < 0):
        raise ValueError("budget use must be finite and nonnegative")
    failures = []
    if branch_points > budget.branch_point_evaluations:
        failures.append("BRANCH_POINTS")
    if branch_points + interval_points + local_points > budget.total_point_evaluations:
        failures.append("TOTAL_POINTS")
    if jacobians > budget.jacobian_evaluations:
        failures.append("JACOBIANS")
    if local_points > budget.local_refinement_evaluations:
        failures.append("LOCAL_REFINEMENT")
    if elapsed_ms > budget.total_query_time_ms:
        failures.append("QUERY_TIME")
    return failures


def run(root: Path, recompute_dir: Path, a1_root: Path,
        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = root / "private_rollout_summary.json"
    records_path = root / "private_rollout_records.jsonl"
    trace_path = root / "private_rollout_trace.npz"
    private = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_id = private["scenario_id"]
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    if (scene_id not in {f"v6_lite_scenario_{index:02d}" for index in range(5)}
            or private["executed_ticks"] != TICKS
            or private["stop_reason"] != "HORIZON_COMPLETE"
            or private["point_budget"] != QUERY_POINT_LIMIT
            or private["records_sha256"] != _sha(records_path)
            or private["trace_sha256"] != _sha(trace_path)
            or len(records) != TICKS):
        raise ValueError("private full-scene trace provenance changed")
    recompute_path = recompute_dir / "private_recompute_summary.json"
    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    if (not recompute["pass_recompute"] or recompute["failure_count"]
            or recompute["inputs"][0]["summary_sha256"] != _sha(summary_path)
            or recompute["inputs"][0]["trace_sha256"] != _sha(trace_path)
            or recompute["inputs"][0]["records_sha256"] != _sha(records_path)):
        raise ValueError("independent torque replay does not bind this trace")

    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if _sha(metrics_path) != private["inputs"]["metrics_sha256"]:
        raise ValueError("A.1 scenario metadata changed")
    cfg = HierarchicalQPConfig(**metrics["qp_config"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    saved = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == scene_id)
    robot = default_v6_lite_robot_spec()
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
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    query = PersistentIntervalDecisionQuery(evaluator.shape_model)
    with np.load(trace_path, allow_pickle=False) as trace:
        task_qpos = trace["qpos_states"][::10].copy()
        task_qvel = trace["task_qvel_states"].copy()
    if task_qpos.shape[0] != TICKS + 1 or task_qvel.shape[0] != TICKS + 1:
        raise ValueError("private task-state trace dimensions changed")

    rows = []
    for tick, saved_row in enumerate(records):
        data.qpos[:] = task_qpos[tick]
        data.qvel[:] = task_qvel[tick]
        data.time = float(saved_row["time_s"])
        mujoco.mj_forward(model, data)
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
        started = time.perf_counter()
        decision = query.evaluate(
            projection.planner_configuration, base, box,
            IntervalPartition.uniform(),
            max_point_evaluations=QUERY_POINT_LIMIT,
        )
        previous = (np.zeros(17, dtype=np.float64) if tick == 0 else
                    np.asarray(records[tick - 1]["selected_command"],
                               dtype=np.float64))
        planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
        _lo, _hi, speed, box_valid = ramp_velocity_abs_bound(
            robot, cfg, planner_q, previous)
        generalized_map = evaluator.reaction_map(data)
        selected, excluded = select_intervals_by_frozen_reach(
            decision.lower_by_interval_id, decision.partition,
            evaluator, data, cfg, generalized_map,
            velocity_abs_bound=speed,
        )
        batch = evaluator.evaluate_state(
            data, decision.partition, generalized_map=generalized_map,
            derivative_interval_ids=selected,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        branch_points = decision.point_evaluation_count
        interval_points = batch.point_evaluation_count
        local_points = 0  # PersistentIntervalDecisionQuery has no local refinement.
        all_points = branch_points + interval_points + local_points
        jacobians = batch.jacobian_evaluation_count
        parity_errors = []
        if (decision.proxy_clearance_status != saved_row["proxy_status"]
                or branch_points != saved_row["point_evaluations"]
                or len(selected) != saved_row["selected_interval_count"]
                or len(excluded) != saved_row["excluded_interval_count"]
                or not box_valid):
            parity_errors.append("QUERY_OR_SELECTION_CHANGED")
        if (not batch.coverage_complete or not batch.interval_well_formed
                or any(item.derivative_status != "SUPPORTED"
                       for item in batch.rows if item.interval_id in selected)):
            parity_errors.append("REQUIRED_INTERVAL_UNSUPPORTED")
        if batch.geometry_domain_status != saved_row["geometry_domain_status"]:
            parity_errors.append("GEOMETRY_DOMAIN_CHANGED")
        budget_failures = assess_budget(
            branch_points, interval_points, local_points, jacobians,
            elapsed_ms, FROZEN_BUDGET)
        rows.append({
            "tick": tick, "time_s": float(data.time),
            "branch_point_evaluation_count": branch_points,
            "interval_point_evaluation_count": interval_points,
            "local_refinement_evaluation_count": local_points,
            "total_point_evaluation_count": all_points,
            "shape_jacobian_evaluation_count":
                batch.shape_jacobian_evaluation_count,
            "mujoco_point_jacobian_count": batch.mujoco_point_jacobian_count,
            "jacobian_evaluation_count": jacobians,
            "query_and_interval_assembly_ms": elapsed_ms,
            "proxy_status": decision.proxy_clearance_status,
            "geometry_domain_status": batch.geometry_domain_status,
            "strict_online_domain_met": (
                batch.geometry_domain_status
                == "INSIDE_DECLARED_WORK_DOMAIN;ON_DECLARED_SHAPE_SUBSPACE"),
            "query_budget_exhausted": decision.budget_exhausted,
            "selected_interval_count": len(selected),
            "excluded_interval_count": len(excluded),
            "budget_failures": budget_failures,
            "parity_errors": parity_errors,
        })

    rows_path = output_dir / "unified_budget_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    budget_failures = [row for row in rows if row["budget_failures"]]
    parity_failures = [row for row in rows if row["parity_errors"]]
    report = {
        "schema": "v6_2_b2_domain_private_unified_budget_audit_v1",
        "scope": "saved_private_task_states_direct_mj_forward_not_torque_replay",
        "scenario_id": scene_id,
        "task_ticks": len(rows),
        "query_point_limit_used_by_private_controller": QUERY_POINT_LIMIT,
        "budget_protocol": "bounded_cold_shadow_default",
        "frozen_shadow_budget": FROZEN_BUDGET.__dict__,
        "budget_overrun_tick_count": len(budget_failures),
        "budget_overrun_ticks": [row["tick"] for row in budget_failures],
        "parity_failure_count": len(parity_failures),
        "query_budget_exhausted_count": sum(row["query_budget_exhausted"] for row in rows),
        "strict_domain_violation_tick_count": sum(
            not row["strict_online_domain_met"] for row in rows),
        "strict_domain_violation_ticks": [
            row["tick"] for row in rows if not row["strict_online_domain_met"]],
        "count_statistics": {key: _stats([row[key] for row in rows]) for key in (
            "branch_point_evaluation_count", "interval_point_evaluation_count",
            "local_refinement_evaluation_count", "total_point_evaluation_count",
            "shape_jacobian_evaluation_count", "mujoco_point_jacobian_count",
            "jacobian_evaluation_count", "query_and_interval_assembly_ms",
        )},
        "within_frozen_shadow_budget_on_saved_states": not budget_failures,
        "action_time_budget_enforced_by_private_controller": False,
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_control_time_evaluated": False,
        "continuous_time_certified": False,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_domain_unified_budget.py",
                                       "audit_b2_private_recompute.py",
                                       "pcc_persistent_interval_query.py",
                                       "pcc_interval_cbf.py",
                                       "shadow_b2_interval_cbf.py")},
        "inputs": {"private_summary_sha256": _sha(summary_path),
                   "private_records_sha256": _sha(records_path),
                   "private_trace_sha256": _sha(trace_path),
                   "independent_recompute_sha256": _sha(recompute_path),
                   "a1_metrics_sha256": _sha(metrics_path)},
        "rows_sha256": _sha(rows_path),
    }
    report_path = output_dir / "unified_budget_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    counts = report["count_statistics"]
    document = [
        "# B.2 私有完整场景的统一计算预算只读审计", "",
        "独立重建保存的 1,350 个规划状态；每状态从五段根区间重新查询，"
        "再按冻结可达筛选计算必要区间导数。"
        "输入力矩轨迹已由独立原生重放验证；本审计只读取保存状态，"
        "不执行新动作或重放力矩。", "",
        "| 指标 | p95 | 最大 |", "| --- | ---: | ---: |",
    ]
    for label, key, unit in (
        ("分支点", "branch_point_evaluation_count", "次"),
        ("区间重算点", "interval_point_evaluation_count", "次"),
        ("点总量", "total_point_evaluation_count", "次"),
        ("Jacobian 总量", "jacobian_evaluation_count", "次"),
        ("查询、筛选及区间装配", "query_and_interval_assembly_ms", "ms"),
    ):
        item = counts[key]
        document.append(f"| {label} | {item['p95']:.3f} {unit} | "
                        f"{item['max']:.3f} {unit} |")
    document += [
        "", f"冻结冷影子预算：分支点 {FROZEN_BUDGET.branch_point_evaluations}、"
        f"点总量 {FROZEN_BUDGET.total_point_evaluations}、"
        f"Jacobian {FROZEN_BUDGET.jacobian_evaluations}、"
        f"局部细化 {FROZEN_BUDGET.local_refinement_evaluations}、"
        f"查询及装配 {FROZEN_BUDGET.total_query_time_ms:.0f} ms。"
        f"逐状态超限 {len(budget_failures)}，重算选择不一致 {len(parity_failures)}。",
        f"严格工作域外规划周期 {report['strict_domain_violation_tick_count']}；"
        "工作域状态与私有控制记录逐状态核对。",
        "", "私有控制器只预先限制分支查询最多 255 点；本报告对总点数、"
        "Jacobian、局部细化和耗时作事后检查，未把这些阈值接入动作放行。"
        "计时从保存状态的 `mj_forward` 之后开始，不含完整 QP、"
        "力矩伺服或真实调度。数值属于本机此次独立诊断，"
        "不能推作硬实时时延保证。", "",
    ]
    doc_path = output_dir / "UNIFIED_BUDGET.md"
    doc_path.write_text("\n".join(document), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_domain_unified_budget_manifest_v1",
                "summary_sha256": _sha(report_path),
                "rows_sha256": _sha(rows_path),
                "document_sha256": _sha(doc_path)}
    manifest_path = output_dir / "unified_budget_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=2) + "\n", encoding="utf-8",
                             newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/domain_endpoint_cholesky_scene01_1350"))
    parser.add_argument("--recompute-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.recompute_dir, args.a1_root,
                 args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "task_ticks", "budget_overrun_tick_count", "parity_failure_count",
        "within_frozen_shadow_budget_on_saved_states",
    )}, indent=2))


if __name__ == "__main__":
    main()
