"""Native-torque read-only envelope audit inside selected ten-step A.1 ramps.

Windows are fixed by the early QP probe tick and the first *sampled* old-speed
interval CBF start violation in each scene. Commands and torques come from the
published A.1 traces. No new interval-QP command is executed. Coverage is
checked at the eleven 500 Hz states bounding each ten-step ramp; no claim is
made about the continuous path between those states.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


EARLY_TICK = 50
SERVO_STEPS_PER_TASK = 10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _first_sampled_start_violations(refined: dict) -> dict[tuple[str, str], int]:
    first: dict[tuple[str, str], int] = {}
    for record in refined["records"]:
        if record["feasibility"]["status"] == "START_CLEARANCE_VIOLATION":
            key = record["mode"], record["scenario_id"]
            first[key] = min(first.get(key, record["tick"]), record["tick"])
    if (len(first) != 10 or any(tick <= EARLY_TICK or tick % 50
                                 for tick in first.values())):
        raise ValueError("frozen first sampled start-violation protocol changed")
    return first


def run(output_dir: Path, a1_root: Path, refined_path: Path,
        sweep_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    sweep_path = sweep_dir / "envelope_sweep_summary.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if (sweep["scope"] != "all_13500_saved_A1_task_qpos_direct_mj_forward_not_torque_replay"
            or any(sweep["modes"][mode]["state_count"] != 6750
                   for mode in ("baseline", "enabled"))):
        raise ValueError("full saved-state envelope input changed")
    states_path = sweep_dir / "envelope_sweep_states.jsonl"
    if _sha(states_path) != sweep["state_records_sha256"]:
        raise ValueError("saved-state envelope records changed")
    first = _first_sampled_start_violations(refined)
    selected_ticks = {
        key: (EARLY_TICK, violation_tick) for key, violation_tick in first.items()
    }
    endpoint_keys = {
        (mode, scenario, tick + offset)
        for (mode, scenario), ticks in selected_ticks.items()
        for tick in ticks for offset in (0, 1)
    }
    saved_margin = {}
    with states_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            key = item["mode"], item["scenario_id"], item["tick"]
            if key in endpoint_keys:
                saved_margin[key] = item["minimum_margin_m"]
    if set(saved_margin) != endpoint_keys:
        raise ValueError("selected ramp endpoint missing from full sweep")

    robot = default_v6_lite_robot_spec()
    report = {
        "schema": "v6_2_b2_native_torque_microstep_envelope_v1",
        "scope": "two_predeclared_old_A1_ten_step_ramps_per_mode_and_scene",
        "early_tick": EARLY_TICK,
        "servo_steps_per_task": SERVO_STEPS_PER_TASK,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_microstep_envelope.py", "pcc_state_local_envelope.py",
            "pcc_clearance.py", "shape_clearance.py", "continuum_model_spec.py",
            "continuum_shape_model.py", "recompute_execution_constraints.py",
        )},
        "input_refined_start_sha256": _sha(refined_path),
        "input_full_sweep_sha256": _sha(sweep_path),
        "input_full_sweep_states_sha256": _sha(states_path),
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "continuous_time_certified": False,
        "new_interval_mode_executed": False,
        "inputs": {},
        "native_replay_checks": [],
        "windows": [],
        "modes": {},
    }
    records_path = output_dir / "microstep_envelope_states.jsonl"
    failures_path = output_dir / "microstep_envelope_failures.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as records, \
            failures_path.open("x", encoding="utf-8", newline="\n") as failures:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_cfg = metrics["run_config"]
            scenes = metrics["scenarios"]
            if len(scenes) != 5 or len({item["scenario"]["scenario_id"] for item in scenes}) != 5:
                raise ValueError("A.1 must contain five unique scenes per mode")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path), "traces": [],
            }
            for scene in scenes:
                scenario_id = scene["scenario"]["scenario_id"]
                ticks = selected_ticks[mode, scenario_id]
                windows = {
                    "early_probe": ticks[0],
                    "first_sampled_old_start_violation": ticks[1],
                }
                step_to_window = {}
                for label, tick in windows.items():
                    for substep in range(SERVO_STEPS_PER_TASK + 1):
                        step = SERVO_STEPS_PER_TASK * tick + substep
                        if step in step_to_window:
                            raise ValueError("selected ramp windows overlap")
                        step_to_window[step] = label
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
                if envelope.envelopes.fallback_geom_names != ("collision_0003",):
                    raise ValueError("original MuJoCo mount fallback changed")
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
                    torque = trace["torque"].copy()
                    task_qpos = trace["task_qpos"].copy()
                    task_time = trace["task_time"].copy()
                if (len(torque) != len(task_time) * SERVO_STEPS_PER_TASK
                        or task_qpos.shape != (len(task_time) + 1, model.nq)):
                    raise ValueError("A.1 torque/task state cadence changed")
                stop_step = max(step_to_window)
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                max_replay_error = 0.0
                window_records: dict[str, list[dict]] = {label: [] for label in windows}
                for step in range(stop_step + 1):
                    if step % SERVO_STEPS_PER_TASK == 0:
                        mujoco.mj_forward(model, data)
                        tick = step // SERVO_STEPS_PER_TASK
                        max_replay_error = max(
                            max_replay_error,
                            float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                            abs(float(data.time - task_time[tick])),
                        )
                    label = step_to_window.get(step)
                    if label is not None:
                        if step % SERVO_STEPS_PER_TASK:
                            mujoco.mj_forward(model, data)
                        actual = data.qpos[evaluator.qpos_ids[:60]]
                        projection = evaluator.shape_spec.project_actual_configuration(actual)
                        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                        started = time.perf_counter()
                        coverage = envelope.evaluate(
                            data, projection.planner_configuration, base,
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        worst = min(coverage.capsules, key=lambda row: row.margin_m)
                        window_tick = windows[label]
                        item = {
                            "mode": mode, "scenario_id": scenario_id,
                            "window": label, "task_tick": window_tick,
                            "servo_substep": step - SERVO_STEPS_PER_TASK * window_tick,
                            "physics_step": step, "time_s": float(data.time),
                            "status": coverage.status,
                            "minimum_margin_m": coverage.min_margin_m,
                            "worst_geom_name": worst.geom_name,
                            "subspace_residual_linf_rad": projection.residual_linf_rad,
                            "check_ms": elapsed_ms,
                        }
                        records.write(json.dumps(item, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n")
                        if coverage.status != "COVERED_AT_THIS_STATE":
                            failures.write(json.dumps({
                                **item,
                                "all_capsules": [asdict(row) for row in coverage.capsules],
                            }, ensure_ascii=False, separators=(",", ":")) + "\n")
                        window_records[label].append(item)
                    if step < stop_step:
                        data.ctrl[:] = torque[step]
                        mujoco.mj_step(model, data)
                report["native_replay_checks"].append({
                    "mode": mode, "scenario_id": scenario_id,
                    "through_task_tick": stop_step // SERVO_STEPS_PER_TASK,
                    "max_state_error": max_replay_error,
                })
                if max_replay_error > 1e-8:
                    raise ValueError("native torque replay diverged from saved task state")
                for label, values in window_records.items():
                    if len(values) != SERVO_STEPS_PER_TASK + 1:
                        raise ValueError("microstep window is incomplete")
                    tick = windows[label]
                    start_margin = values[0]["minimum_margin_m"]
                    end_margin = values[-1]["minimum_margin_m"]
                    if (abs(start_margin - saved_margin[mode, scenario_id, tick]) > 1e-8
                            or abs(end_margin - saved_margin[mode, scenario_id, tick + 1]) > 1e-8):
                        raise ValueError("native ramp endpoints disagree with saved-state sweep")
                    minimum = min(values, key=lambda item: item["minimum_margin_m"])
                    report["windows"].append({
                        "mode": mode, "scenario_id": scenario_id,
                        "window": label, "task_tick": tick,
                        "start_margin_m": start_margin,
                        "end_margin_m": end_margin,
                        "minimum_margin_m": minimum["minimum_margin_m"],
                        "minimum_at_servo_substep": minimum["servo_substep"],
                        "interior_below_both_endpoints": any(
                            item["minimum_margin_m"] < min(start_margin, end_margin)
                            for item in values[1:-1]
                        ),
                        "covered_microstates": sum(
                            item["status"] == "COVERED_AT_THIS_STATE" for item in values
                        ),
                    })
                print(f"[b2-microstep-envelope] {mode} {scenario_id}: "
                      f"2 windows, replay error {max_replay_error:.2e}", flush=True)
    if len(report["windows"]) != 20 or len(report["native_replay_checks"]) != 10:
        raise ValueError("microstep audit did not cover the declared scenes")
    for mode in ("baseline", "enabled"):
        own = [item for item in report["windows"] if item["mode"] == mode]
        report["modes"][mode] = {
            "window_count": len(own),
            "microstate_count": (SERVO_STEPS_PER_TASK + 1) * len(own),
            "covered_microstate_count": sum(item["covered_microstates"] for item in own),
            "minimum_margin_m": _summary([item["minimum_margin_m"] for item in own]),
            "interior_below_both_endpoints_count": sum(
                item["interior_below_both_endpoints"] for item in own
            ),
        }
    report["state_records_sha256"] = _sha(records_path)
    report["failure_records_sha256"] = _sha(failures_path)
    report_path = output_dir / "microstep_envelope_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 旧 A.1 十步斜坡的 500 Hz 包络审计", "",
        "每组每场景选 tick 50 与第一次抽样观察到旧速度区间 CBF 起点违例的规划 tick。"
        "按原 trace 力矩逐步重放，每个十步斜坡检查两端及九个内部 500 Hz 状态；"
        "每个规划边界与保存 qpos 逐状态核对。旧命令和原 MuJoCo/胶囊配置不变。", "",
        "| 模式 | 斜坡窗口 | 500 Hz 状态 | 当前状态包含 | 窗口最小余量 mm | 内部余量低于两端的窗口 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        lines.append(
            f"| {mode} | {item['window_count']} | {item['microstate_count']} | "
            f"{item['covered_microstate_count']} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['interior_below_both_endpoints_count']} |"
        )
    lines += [
        "", "完整逐步记录及失败快照分别保存在 JSONL。"
        "所选窗口由已有抽样反例决定，不能代表所有斜坡。"
        "500 Hz 离散检查不证明 2 ms 步间或连续时间安全；"
        "也不证明新区间 QP 闭环可行、20 ms 全链预算或接入门禁。", "",
    ]
    doc_path = output_dir / "MICROSTEP_ENVELOPE.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_native_torque_microstep_envelope_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", report_path), ("states", records_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "microstep_envelope_manifest.json").open(
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
    parser.add_argument("--refined-start", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/refined_start/refined_start_report.json"))
    parser.add_argument("--full-sweep-dir", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/full_state_envelope"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.refined_start,
                 args.full_sweep_dir)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
