"""Independently replay a stopped private trace and audit its rejected command.

This checks saved torque execution in a fresh MuJoCo process. It does not
recompute interval CBF rows or turn a private diagnostic into online evidence.
"""

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
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(strict_dir: Path, baseline_dir: Path, a1_root: Path,
        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {name: strict_dir / filename for name, filename in {
        "summary": "private_rollout_summary.json",
        "manifest": "private_rollout_manifest.json",
        "records": "private_rollout_records.jsonl",
        "trace": "private_rollout_trace.npz",
        "failure": "private_rollout_failure.json",
    }.items()}
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    records = [json.loads(line) for line in paths["records"].read_text(
        encoding="utf-8").splitlines()]
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    executed = summary["executed_ticks"]
    attempted = summary["attempted_ticks"]
    if (summary["scenario_id"] != "v6_lite_scenario_01"
            or summary["stop_reason"]
            != "RAMP_MICROSTATE_OUTSIDE_DECLARED_WORK_DOMAIN"
            or not summary["strict_gate_rejected_before_servo"]
            or executed != 887 or attempted != executed + 1
            or len(records) != attempted
            or records[-1] != summary["failure_record"]
            or failure["last_record"] != records[-1]
            or failure["trace_sha256"] != _sha(paths["trace"])
            or any(manifest[f"{name}_sha256"] != _sha(paths[name])
                   for name in ("summary", "records", "trace", "failure"))):
        raise ValueError("strict stop protocol or input hash changed")
    rejected = records[-1]
    failures = rejected["ramp_domain_failures"]
    if (rejected["tick"] != executed
            or rejected["action_mode"] != "UNCERTIFIED"
            or rejected["failure_reason"] != "ramp_work_domain_unsupported"
            or rejected["selected_command"] is not None
            or rejected["next_servo_step_executed"] is not False
            or len(rejected["rejected_candidate_command"]) != 17
            or not failures or failures[0]["servo_substep"] != 4):
        raise ValueError("rejected command was not explicitly separated")

    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = next(item for item in metrics["scenarios"] if
                 item["scenario"]["scenario_id"] == summary["scenario_id"])
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
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
    with np.load(paths["trace"], allow_pickle=False) as trace:
        initial_qpos = trace["initial_qpos"].copy()
        initial_qvel = trace["initial_qvel"].copy()
        torques = trace["torque"].copy()
        qpos = trace["qpos_states"].copy()
        task_qvel = trace["task_qvel_states"].copy()
    baseline_trace_path = baseline_dir / "private_rollout_trace.npz"
    baseline_summary_path = baseline_dir / "private_rollout_summary.json"
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    with np.load(baseline_trace_path, allow_pickle=False) as trace:
        baseline_torques = trace["torque"][:len(torques)].copy()
        baseline_qpos = trace["qpos_states"][:len(qpos)].copy()
    if (baseline["scenario_id"] != summary["scenario_id"]
            or baseline["trace_sha256"] != _sha(baseline_trace_path)
            or torques.shape != (executed * 10, 67)
            or qpos.shape != (executed * 10 + 1, model.nq)
            or task_qvel.shape != (executed + 1, model.nv)
            or not np.array_equal(torques, baseline_torques)
            or not np.array_equal(qpos, baseline_qpos)):
        raise ValueError("executed trace differs from verified baseline prefix")
    if np.any(np.abs(torques) > robot.torque_limits[None, :] + 1e-10):
        raise ValueError("saved torque exceeds existing limit")

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0
    max_error = 0.0
    minimum_work_domain_margin = float("inf")
    for step in range(len(torques) + 1):
        max_error = max(max_error, float(np.max(np.abs(data.qpos - qpos[step]))))
        if step % 10 == 0:
            max_error = max(max_error, float(np.max(np.abs(
                data.qvel - task_qvel[step // 10]))))
        projection = evaluator.shape_spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        planner_q = projection.planner_configuration
        lower = evaluator.shape_spec.work_domain_lower_rad
        upper = evaluator.shape_spec.work_domain_upper_rad
        margin = min(float(np.min(planner_q - lower)),
                     float(np.min(upper - planner_q)))
        minimum_work_domain_margin = min(minimum_work_domain_margin, margin)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError(f"executed microstate {step} outside work domain")
        if step < len(torques):
            data.ctrl[:] = torques[step]
            mujoco.mj_step(model, data)
    if max_error > 1e-8:
        raise ValueError("native torque replay changed saved states")

    report = {
        "schema": "v6_2_b2_strict_domain_independent_replay_v1",
        "scenario_id": summary["scenario_id"],
        "executed_ticks": executed,
        "native_torque_steps": len(torques),
        "checked_native_states": len(qpos),
        "native_replay_max_state_error": max_error,
        "minimum_executed_work_domain_margin_rad": minimum_work_domain_margin,
        "baseline_prefix_torques_exact": True,
        "baseline_prefix_qpos_exact": True,
        "first_rejected_candidate_tick": rejected["tick"],
        "first_predicted_outside_servo_substep": failures[0]["servo_substep"],
        "rejected_candidate_excluded_from_torque_trace": True,
        "independent_interval_rows_recomputed_here": False,
        "production_online_admitted": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "inputs_sha256": {name: _sha(path) for name, path in paths.items()},
        "a1_metrics_sha256": _sha(metrics_path),
        "baseline_summary_sha256": _sha(baseline_summary_path),
        "baseline_trace_sha256": _sha(baseline_trace_path),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(
            b"\r\n", b"\n")).hexdigest(),
    }
    report_path = output_dir / "strict_domain_replay.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8",
                           newline="\n")
    document_path = output_dir / "STRICT_DOMAIN_REPLAY.md"
    document_path.write_text(
        "# B.2 严格域门禁的独立力矩重放\n\n"
        f"场景 01 执行 {executed} 个任务周期、{len(torques)} 个力矩步；"
        f"独立原生重放 {len(qpos)} 个状态，最大状态误差 {max_error:.3e}。"
        f"执行状态距工作域边界最小 {minimum_work_domain_margin:.6g} rad。\n\n"
        "与先前完整私有诊断相比，已执行力矩与 qpos 前缀逐值完全一致。"
        f"tick {rejected['tick']} 的候选预测在第 {failures[0]['servo_substep']} "
        "个 2 ms 微步越界，因而整段候选未写入保存的力矩轨迹。\n\n"
        "这里只独立重放力矩和检查工作域，没有重新计算区间 CBF 行。"
        "这条停止轨迹不能满足五场景完整新模式验收、真实在线时限或连续时间证明。\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.strict_dir, args.baseline_dir, args.a1_root,
                 args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "executed_ticks", "native_torque_steps", "checked_native_states",
        "native_replay_max_state_error",
        "minimum_executed_work_domain_margin_rad",
        "first_rejected_candidate_tick", "first_predicted_outside_servo_substep",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
