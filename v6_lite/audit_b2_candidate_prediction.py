"""Recompute fixed-interval lookahead after private candidate torque ramps.

Each early QP probe partition is reconstructed from its frozen old A.1 state.
Selected interval IDs are held fixed across the ten-step private candidate
servo branch. The resulting proxy h and CBF start row are compared with their
frozen affine predictions. This is an empirical one-cycle error audit, never
a continuous-time bound or a new-mode closed-loop acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_candidate_ramp import _branch
from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.execution_ramp import ramp_mean_weights
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder, _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
from v6_lite.shape_clearance import target_box_from_mujoco


TICKS = (50, 100, 150)
STEPS = 10


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(data), "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _rows(batch, selected_ids: set[str]) -> dict:
    result = {row.interval_id: row for row in batch.rows
              if row.interval_id in selected_ids}
    if set(result) != selected_ids:
        raise ValueError("fixed partition lost a selected interval")
    if any(row.derivative_status != "SUPPORTED" for row in result.values()):
        raise ValueError("required fixed interval derivative unsupported")
    return result


def run(output_dir: Path, a1_root: Path, probe_path: Path,
        ramp_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    ramp = json.loads(ramp_path.read_text(encoding="utf-8"))
    if (probe["probe_ticks"] != list(TICKS) or probe["point_budget"] != 63
            or len(probe["records"]) != 30
            or ramp["probe_ticks"] != list(TICKS)
            or len(ramp["records"]) != 30
            or ramp["input_probe_sha256"] != _sha(probe_path)
            or ramp["new_interval_mode_executed"]):
        raise ValueError("frozen candidate branch inputs changed")
    probes = {(r["mode"], r["scenario_id"], r["tick"]): r
              for r in probe["records"]}
    ramps = {(r["mode"], r["scenario_id"], r["tick"]): r
             for r in ramp["records"]}
    if len(probes) != 30 or set(probes) != set(ramps):
        raise ValueError("candidate branch keys changed")
    report = {
        "schema": "v6_2_b2_candidate_fixed_partition_prediction_v1",
        "scope": "thirty_predeclared_private_candidate_ramps_same_interval_ids_both_ends",
        "probe_ticks": list(TICKS), "servo_steps_per_task": STEPS,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_candidate_prediction.py", "audit_b2_candidate_ramp.py",
            "pcc_interval_cbf.py", "pcc_persistent_interval_query.py",
            "execution_ramp.py", "run_v6_lite.py",
            "recompute_execution_constraints.py",
        )},
        "input_probe_sha256": _sha(probe_path),
        "input_candidate_ramp_sha256": _sha(ramp_path),
        "new_interval_mode_executed": False,
        "continuous_time_certified": False,
        "margin_is_empirical_not_certified_bound": True,
        "inputs": {}, "cases": [], "modes": {},
    }
    robot = default_v6_lite_robot_spec()
    tolerance_by_mode = {}
    row_path = output_dir / "candidate_prediction_rows.jsonl"
    failure_path = output_dir / "candidate_prediction_failures.jsonl"
    with row_path.open("x", encoding="utf-8", newline="\n") as rows_out, \
            failure_path.open("x", encoding="utf-8", newline="\n") as failures_out:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg = HierarchicalQPConfig(**metrics["qp_config"])
            tolerance_by_mode[mode] = cfg.clearance_rate_tolerance_m_s
            run_cfg = metrics["run_config"]
            scenes = metrics["scenarios"]
            if len(scenes) != 5:
                raise ValueError("A.1 scene count changed")
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
                data = mujoco.MjData(model)
                evaluator = FixedIntervalCBFEvaluator(robot, model)
                envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
                no_legacy = replace(
                    cfg, enable_pcc_cbf=False, enable_capsule_cbf=True)
                independent = ReplayConstraintBuilder(
                    robot, model, verifier.pairs, no_legacy)
                query = PersistentIntervalDecisionQuery(evaluator.shape_model)
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
                    torque = trace["torque"][:STEPS*TICKS[-1]].copy()
                    task_qpos = trace["task_qpos"][:TICKS[-1]+2].copy()
                    task_time = trace["task_time"][:TICKS[-1]+1].copy()
                    reference_q = trace["reference_q"][:STEPS*TICKS[-1]].copy()
                    selected = trace["task_selected_command"][:TICKS[-1]+1].copy()
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                max_replay_error = 0.0
                for step in range(STEPS*TICKS[-1]+1):
                    if step % STEPS == 0:
                        tick = step // STEPS
                        mujoco.mj_forward(model, data)
                        max_replay_error = max(
                            max_replay_error,
                            float(np.max(np.abs(data.qpos-task_qpos[tick]))),
                            abs(float(data.time-task_time[tick])),
                        )
                        if max_replay_error > 1e-8:
                            raise ValueError("native frozen-state replay diverged")
                        if tick in TICKS:
                            key = mode, scenario_id, tick
                            probe_row, prior = probes[key], ramps[key]
                            previous = selected[tick-1].copy()
                            candidate = np.asarray(probe_row["selected_command"],
                                                   dtype=np.float64)
                            start_qpos = data.qpos.copy()
                            start_qvel = data.qvel.copy()
                            start_time = float(data.time)
                            projection = evaluator.shape_spec.project_actual_configuration(
                                data.qpos[evaluator.qpos_ids[:60]])
                            base = transform_from_free_qpos(
                                data.qpos[evaluator.base_qpos_slice])
                            box = target_box_from_mujoco(
                                model, data, evaluator.target_geom_id)
                            decision = query.evaluate(
                                projection.planner_configuration, base, box,
                                IntervalPartition.uniform(),
                                max_point_evaluations=probe["point_budget"],
                            )
                            if (decision.proxy_clearance_status
                                    != probe_row["query_status"]
                                    or decision.point_evaluation_count
                                    != probe_row["query_points"]):
                                raise ValueError("frozen query disagrees with QP probe")
                            start_map, _ = independent._reaction_map(data)
                            planner_q = robot.low_level_to_planner @ data.qpos[
                                evaluator.qpos_ids]
                            _lo, _hi, speed, _box_valid = ramp_velocity_abs_bound(
                                robot, cfg, planner_q, previous)
                            selected_ids, _ = select_intervals_by_frozen_reach(
                                decision.lower_by_interval_id, decision.partition,
                                evaluator, data, cfg, start_map,
                                velocity_abs_bound=speed,
                            )
                            if len(selected_ids) != probe_row["selected_interval_count"]:
                                raise ValueError("frozen selected interval set changed")
                            start_batch = evaluator.evaluate_state(
                                data, decision.partition, generalized_map=start_map,
                                derivative_interval_ids=selected_ids,
                            )
                            start_rows = _rows(start_batch, selected_ids)
                            result, end_data = _branch(
                                model, robot, evaluator, envelope,
                                start_qpos, start_qvel, start_time,
                                reference_q[step-1], previous, candidate,
                                physics_period_s=run_cfg["physics_period_s"],
                                task_period_s=run_cfg["task_period_s"],
                            )
                            candidate_qpos_difference = float(np.max(np.abs(
                                end_data.qpos-task_qpos[tick+1])))
                            if (abs(candidate_qpos_difference - prior[
                                    "candidate_next_qpos_linf_difference_from_old"])
                                    > 1e-8
                                    or min(x["minimum_margin_m"]
                                           for x in result["coverage"])
                                    < prior["candidate_minimum_envelope_margin_m"]-1e-8):
                                raise ValueError("candidate branch disagrees with prior audit")
                            end_map, _ = independent._reaction_map(end_data)
                            end_batch = evaluator.evaluate_state(
                                end_data, decision.partition, generalized_map=end_map,
                                derivative_interval_ids=selected_ids,
                            )
                            end_rows = _rows(end_batch, selected_ids)
                            old_weight, new_weight = ramp_mean_weights()
                            mean_velocity = old_weight*previous + new_weight*candidate
                            case_rows = []
                            for interval_id in sorted(selected_ids):
                                first, last = start_rows[interval_id], end_rows[interval_id]
                                predicted_h = first.h_m + cfg.task_period_s * (
                                    float(first.generalized_gradient @ mean_velocity)
                                    + first.target_drift_m_s)
                                realized_h = last.h_m
                                predicted_start_slack_before_margin = (
                                    float(first.generalized_gradient @ candidate)
                                    + first.target_drift_m_s
                                    + cfg.pcc_clearance_barrier_gain*predicted_h)
                                predicted_start_slack_after_margin = (
                                    predicted_start_slack_before_margin
                                    - cfg.lookahead_model_margin_m_s)
                                realized_start_slack = (
                                    float(last.generalized_gradient @ candidate)
                                    + last.target_drift_m_s
                                    + cfg.pcc_clearance_barrier_gain*realized_h)
                                item = {
                                    "mode": mode, "scenario_id": scenario_id,
                                    "tick": tick, "interval_id": interval_id,
                                    "partition_interval_count": decision.interval_count,
                                    "start_h_m": first.h_m,
                                    "predicted_next_h_m": predicted_h,
                                    "realized_next_h_m": realized_h,
                                    "h_prediction_error_rate_m_s": (
                                        (realized_h-predicted_h)/cfg.task_period_s),
                                    "predicted_start_slack_before_margin_m_s": (
                                        predicted_start_slack_before_margin),
                                    "predicted_start_slack_after_margin_m_s": (
                                        predicted_start_slack_after_margin),
                                    "realized_start_slack_m_s": realized_start_slack,
                                    "cbf_start_slack_model_error_m_s": (
                                        realized_start_slack
                                        - predicted_start_slack_before_margin),
                                    "frozen_margin_m_s": cfg.lookahead_model_margin_m_s,
                                    "derivative_status_start": first.derivative_status,
                                    "derivative_status_end": last.derivative_status,
                                }
                                case_rows.append(item)
                                rows_out.write(json.dumps(item, ensure_ascii=False,
                                                          separators=(",", ":")) + "\n")
                                if (item["cbf_start_slack_model_error_m_s"]
                                        < -cfg.lookahead_model_margin_m_s
                                        or item["realized_start_slack_m_s"]
                                        < -cfg.clearance_rate_tolerance_m_s):
                                    failures_out.write(json.dumps(item, ensure_ascii=False,
                                                                  separators=(",", ":")) + "\n")
                            report["cases"].append({
                                "mode": mode, "scenario_id": scenario_id,
                                "tick": tick, "selected_interval_count": len(case_rows),
                                "native_replay_max_state_error": max_replay_error,
                                "candidate_final_qpos_parity_error": abs(
                                    candidate_qpos_difference-prior[
                                        "candidate_next_qpos_linf_difference_from_old"]),
                                "maximum_h_optimism_rate_m_s": max(
                                    -r["h_prediction_error_rate_m_s"] for r in case_rows),
                                "maximum_cbf_slack_optimism_m_s": max(
                                    -r["cbf_start_slack_model_error_m_s"]
                                    for r in case_rows),
                                "minimum_realized_start_slack_m_s": min(
                                    r["realized_start_slack_m_s"] for r in case_rows),
                            })
                            print(f"[b2-candidate-prediction] {mode} {scenario_id} "
                                  f"tick {tick}: {len(case_rows)} rows, "
                                  f"max h optimism="
                                  f"{1000*max(-r['h_prediction_error_rate_m_s'] for r in case_rows):.3f} mm/s",
                                  flush=True)
                    if step < STEPS*TICKS[-1]:
                        data.ctrl[:] = torque[step]
                        mujoco.mj_step(model, data)
    if len(report["cases"]) != 30:
        raise ValueError("fixed-partition candidate population incomplete")
    all_rows = []
    with row_path.open(encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream]
    for mode in ("baseline", "enabled"):
        own = [r for r in all_rows if r["mode"] == mode]
        cases = [c for c in report["cases"] if c["mode"] == mode]
        report["modes"][mode] = {
            "case_count": len(cases), "selected_row_count": len(own),
            "h_prediction_error_rate_m_s": _summary([
                r["h_prediction_error_rate_m_s"] for r in own]),
            "h_optimism_rate_m_s": _summary([
                max(0.0, -r["h_prediction_error_rate_m_s"]) for r in own]),
            "cbf_start_slack_optimism_m_s": _summary([
                max(0.0, -r["cbf_start_slack_model_error_m_s"])
                for r in own]),
            "minimum_realized_start_slack_m_s": min(
                r["realized_start_slack_m_s"] for r in own),
            "cbf_optimism_over_frozen_margin_count": sum(
                -r["cbf_start_slack_model_error_m_s"] > r["frozen_margin_m_s"]
                for r in own),
            "realized_start_violation_count": sum(
                r["realized_start_slack_m_s"] < -tolerance_by_mode[mode]
                for r in own),
        }
    report["row_records_sha256"] = _sha(row_path)
    report["failure_records_sha256"] = _sha(failure_path)
    summary_path = output_dir / "candidate_prediction_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 候选单周期的同区间预测误差", "",
        "在旧 A.1 早期 30 个冻结起点重建原只读 QP 的区间划分与筛选 ID。"
        "通过相同 67 路力矩伺服执行候选一个 20 ms 斜坡后，在同一组固定材料"
        "弧长区间上重新计算 h、广义梯度和目标漂移。只比较对应区间 ID，"
        "不把重新分区的数值混作同一函数误差。", "",
        "| 模式 | 分支 | 固定区间行 | h 预测过高最大 mm/s | CBF 起点松弛预测过高最大 mm/s | CBF 过高超现有 5 mm/s 裕度 | 实现起点违例行 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        lines.append(
            f"| {mode} | {item['case_count']} | {item['selected_row_count']} | "
            f"{1000*item['h_optimism_rate_m_s']['max']:.3f} | "
            f"{1000*item['cbf_start_slack_optimism_m_s']['max']:.3f} | "
            f"{item['cbf_optimism_over_frozen_margin_count']} | "
            f"{item['realized_start_violation_count']} |"
        )
    lines += [
        "", "h 误差除以 20 ms 转成速度单位；CBF 起点松弛比较还包含"
        "执行后 Jacobian 与目标漂移变化。CBF 误差将未扣裕度的冻结预测"
        "与实际值比较，再单独与裕度对照；两种误差不能混为同一项。"
        "现有 5 mm/s 是原前瞻配置的经验裕度；样本未超出也不能"
        "推断它是所有状态的严格上界。", "",
        "本审计只使用私有单周期分支，不接入新区间在线闭环。"
        "不证明 2 ms 步间或连续时间安全，不测试 20 ms 全链时延。", "",
    ]
    document_path = output_dir / "CANDIDATE_PREDICTION.md"
    with document_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_candidate_fixed_partition_prediction_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("rows", row_path),
            ("failures", failure_path), ("document", document_path),
        )},
    }
    with (output_dir / "candidate_prediction_manifest.json").open(
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
    parser.add_argument("--candidate-ramp", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/candidate_ramp_early/candidate_ramp_summary.json"))
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.probe,
                 args.candidate_ramp)
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
