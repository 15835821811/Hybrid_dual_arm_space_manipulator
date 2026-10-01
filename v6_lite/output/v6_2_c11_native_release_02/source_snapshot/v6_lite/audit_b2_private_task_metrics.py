"""Recompute original task thresholds on a new private 67-torque trace.

This is a scenario metric audit, not the original 26/11 online acceptance.
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
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M, rotation_error_angle_rad,
)
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(private_dir: Path, a1_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = private_dir / "private_rollout_summary.json"
    trace_path = private_dir / "private_rollout_trace.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["stop_reason"] != "HORIZON_COMPLETE"
            or summary["executed_ticks"] != 1350
            or summary["trace_sha256"] != _sha(trace_path)):
        raise ValueError("this metric audit requires a full private 27 s trace")
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_cfg = V6LiteRunConfig(**metrics["run_config"])
    robot = default_v6_lite_robot_spec()
    scenario = next(item for item in build_scenarios(robot, run_cfg)
                    if item.scenario_id == summary["scenario_id"])
    saved = next(item for item in metrics["scenarios"] if
                 item["scenario"]["scenario_id"] == summary["scenario_id"])
    if scenario.seed != saved["scenario"]["seed"]:
        raise ValueError("published scenario seed changed")
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
    data = mujoco.MjData(model)
    rigid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                robot.rigid_tip_body_name)
    continuum_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    robot.continuum_tip_body_name)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 "target_satellite")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = trace["qpos_states"].copy()
        torques = trace["torque"].copy()
    if qpos.shape != (13501, model.nq) or torques.shape != (13500, 67):
        raise ValueError("private trace dimensions changed")
    rigid_error = np.empty(13500)
    continuum_error = np.empty(13500)
    rigid_orientation = np.empty(13500)
    continuum_orientation = np.empty(13500)
    times = np.arange(1, 13501, dtype=np.float64) * run_cfg.physics_period_s
    for step, state in enumerate(qpos[1:]):
        data.qpos[:] = state
        data.qvel[:] = 0
        data.time = float(times[step])
        mujoco.mj_forward(model, data)
        target_rotation = np.asarray(data.xmat[target_id]).reshape(3, 3)
        rigid_target = (np.asarray(data.xpos[target_id]) + target_rotation
                        @ scenario.grasp_point_target_frame_m)
        rigid_target_rotation = (
            target_rotation @ scenario.grasp_rotation_target_frame)
        rigid_rotation = np.asarray(data.xmat[rigid_id]).reshape(3, 3)
        continuum_rotation = np.asarray(data.xmat[continuum_id]).reshape(3, 3)
        continuum_tip = (np.asarray(data.xpos[continuum_id])
                         + continuum_rotation @ CONTINUUM_EE_OFFSET_M)
        continuum_target, _ = scenario.continuum_target.sample(float(times[step]))
        rigid_error[step] = np.linalg.norm(
            rigid_target - np.asarray(data.xpos[rigid_id]))
        continuum_error[step] = np.linalg.norm(continuum_target - continuum_tip)
        rigid_orientation[step] = np.rad2deg(rotation_error_angle_rad(
            rigid_target_rotation, rigid_rotation))
        continuum_orientation[step] = np.rad2deg(rotation_error_angle_rad(
            scenario.continuum_target_rotation_world, continuum_rotation))
    steady = times >= run_cfg.duration_s - run_cfg.steady_window_s
    path = ((times >= scenario.continuum_target.path_start_s)
            & (times <= scenario.continuum_target.path_end_s))
    metric = {
        "rigid_final_error_m": float(rigid_error[-1]),
        "rigid_steady_rmse_m": float(np.sqrt(np.mean(rigid_error[steady] ** 2))),
        "continuum_active_path_rmse_m": float(np.sqrt(np.mean(
            continuum_error[path] ** 2))),
        "rigid_orientation_full_max_deg": float(np.max(rigid_orientation)),
        "continuum_orientation_full_max_deg": float(
            np.max(continuum_orientation)),
    }
    thresholds = {
        "rigid_final_error_m": run_cfg.rigid_final_error_threshold_m,
        "rigid_steady_rmse_m": run_cfg.rigid_steady_rmse_threshold_m,
        "continuum_active_path_rmse_m": (
            run_cfg.continuum_irregular_path_rmse_threshold_m),
        "rigid_orientation_full_max_deg": (
            run_cfg.orientation_error_threshold_deg),
        "continuum_orientation_full_max_deg": (
            run_cfg.orientation_error_threshold_deg),
    }
    checks = {key: metric[key] <= threshold
              for key, threshold in thresholds.items()}
    tracking_passed = all(checks.values())
    whole_body = verifier.verify_qpos_sequence(qpos[::10])
    checks["whole_body_dense_discrete_clearance"] = bool(whole_body.feasible)
    checks["continuum_target_dense_discrete_clearance"] = bool(
        whole_body.minimum_by_class.get("continuum_target", float("-inf"))
        >= run_cfg.whole_body_minimum_clearance_m)
    report = {
        "schema": "v6_2_b2_private_original_task_metrics_v1",
        "scenario_id": summary["scenario_id"],
        "servo_states_evaluated": len(times),
        "metric": metric,
        "original_thresholds": thresholds,
        "checks": checks,
        "task_tracking_thresholds_passed": tracking_passed,
        "original_task_and_clearance_thresholds_passed": all(checks.values()),
        "whole_body_clearance": whole_body.to_dict(),
        "whole_body_clearance_evaluated_here": True,
        "original_26_11_acceptance_completed": False,
        "production_online_admitted": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(
            b"\r\n", b"\n")).hexdigest(),
        "inputs_sha256": {
            "private_summary": _sha(summary_path),
            "private_trace": _sha(trace_path),
            "a1_metrics": _sha(metrics_path),
        },
    }
    report_path = output_dir / "private_task_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8",
                           newline="\n")
    document_path = output_dir / "PRIVATE_TASK_METRICS.md"
    lines = ["# B.2 新私有轨迹上的原任务跟踪指标", "",
             f"场景 `{summary['scenario_id']}` 共核对 {len(times)} 个 2 ms 状态。"
             "指标公式和阈值沿用原 A.1 配置。", "",
             "| 指标 | 新轨迹 | 原阈值 | 通过 |",
             "| --- | ---: | ---: | :---: |"]
    for key, value in metric.items():
        lines.append(f"| {key} | {value:.9g} | {thresholds[key]:.9g} | "
                     f"{checks[key]} |")
    lines += ["", f"按原任务状态和 {run_cfg.verification_subdivisions} 次"
              "插值核对整机净空："
              f"{whole_body.minimum_clearance:.6g} m，"
              f"阈值 {run_cfg.whole_body_minimum_clearance_m:.6g} m；"
              f"continuum-target 最小净空 "
              f"{whole_body.minimum_by_class.get('continuum_target', float('nan')):.6g} m。",
              "本程序没有完成原 26/11 项合同或在线时限验收。", ""]
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.private_dir, args.a1_root, args.output_dir)
    print(json.dumps({"scenario_id": report["scenario_id"],
                      "metric": report["metric"],
                      "checks": report["checks"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
