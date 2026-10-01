"""Read-only state-local PCC envelope audit at every old A.1 torque state.

The saved torques are replayed in native MuJoCo. Every 500 Hz state, including
the final endpoint, is checked. This is discrete evidence for the historical
controller, not execution or acceptance of the new interval controller.
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
from v6_lite.audit_b2_microstep_envelope import _summary
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


STEPS_PER_TICK = 10


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _indexed_margins(path: Path, step_key: str, multiplier: int) -> dict:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["mode"], item["scenario_id"],
                   multiplier * item[step_key])
            if key in result:
                raise ValueError(f"duplicate reference state: {key}")
            result[key] = item
    return result


def run(output_dir: Path, a1_root: Path, sweep_dir: Path,
        selected_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    sweep_path = sweep_dir / "envelope_sweep_summary.json"
    sweep_states_path = sweep_dir / "envelope_sweep_states.jsonl"
    selected_path = selected_dir / "microstep_envelope_summary.json"
    selected_states_path = selected_dir / "microstep_envelope_states.jsonl"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if (sweep["state_records_sha256"] != _sha(sweep_states_path)
            or selected["state_records_sha256"] != _sha(selected_states_path)
            or sweep["scope"]
            != "all_13500_saved_A1_task_qpos_direct_mj_forward_not_torque_replay"
            or selected["scope"]
            != "two_predeclared_old_A1_ten_step_ramps_per_mode_and_scene"):
        raise ValueError("frozen state references changed")
    saved = _indexed_margins(sweep_states_path, "tick", STEPS_PER_TICK)
    sampled = _indexed_margins(selected_states_path, "physics_step", 1)
    if len(saved) != 13500 or len(sampled) != 220:
        raise ValueError("reference populations changed")

    report = {
        "schema": "v6_2_b2_full_old_torque_envelope_v1",
        "scope": "all_500hz_states_of_ten_historical_A1_native_torque_replays",
        "servo_steps_per_task": STEPS_PER_TICK,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_full_torque_envelope.py", "pcc_state_local_envelope.py",
            "pcc_clearance.py", "shape_clearance.py", "continuum_model_spec.py",
            "continuum_shape_model.py", "recompute_execution_constraints.py",
        )},
        "input_full_sweep_summary_sha256": _sha(sweep_path),
        "input_full_sweep_states_sha256": _sha(sweep_states_path),
        "input_selected_summary_sha256": _sha(selected_path),
        "input_selected_states_sha256": _sha(selected_states_path),
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "continuous_time_certified": False,
        "new_interval_mode_executed": False,
        "inputs": {}, "scenes": [], "modes": {},
        "cross_checks": {"saved_task_states": 0, "selected_microstates": 0},
    }
    seen_saved = set()
    seen_sampled = set()
    states_path = output_dir / "full_torque_envelope_states.jsonl"
    failures_path = output_dir / "full_torque_envelope_failures.jsonl"
    robot = default_v6_lite_robot_spec()
    with states_path.open("x", encoding="utf-8", newline="\n") as states, \
            failures_path.open("x", encoding="utf-8", newline="\n") as failures:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg = metrics["run_config"]
            scenes = metrics["scenarios"]
            if len(scenes) != 5 or len({s["scenario"]["scenario_id"]
                                        for s in scenes}) != 5:
                raise ValueError("expected five distinct A.1 scenes")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path), "traces": [],
            }
            for scene in scenes:
                scenario_id = scene["scenario"]["scenario_id"]
                verifier = WholeBodyCollisionVerifier(
                    robot, _obstacles(scene["scenario"]),
                    WholeBodyVerificationConfig(
                        minimum_clearance=cfg["whole_body_minimum_clearance_m"],
                        query_distance_max=2.5,
                        adaptive_subdivisions=cfg["verification_subdivisions"],
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
                    raise ValueError("original mount fallback changed")
                trace_path = Path(scene["trace"]["path"])
                trace_sha = _sha(trace_path)
                if trace_sha != scene["trace"]["sha256"]:
                    raise ValueError("published A.1 trace changed")
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
                if (len(torque) != STEPS_PER_TICK * len(task_time)
                        or task_qpos.shape != (len(task_time) + 1, model.nq)):
                    raise ValueError("A.1 torque/task cadence changed")
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                margins, timings, residuals = [], [], []
                statuses = Counter()
                worst_item = None
                max_replay_error = 0.0
                for step in range(len(torque) + 1):
                    mujoco.mj_forward(model, data)
                    key = mode, scenario_id, step
                    if step % STEPS_PER_TICK == 0:
                        tick = step // STEPS_PER_TICK
                        expected_time = (task_time[tick] if tick < len(task_time)
                                         else task_time[-1] + cfg["task_period_s"])
                        max_replay_error = max(
                            max_replay_error,
                            float(np.max(np.abs(data.qpos - task_qpos[tick]))),
                            abs(float(data.time - expected_time)),
                        )
                        if max_replay_error > 1e-8:
                            raise ValueError(f"native replay diverged at {key}")
                    actual = data.qpos[evaluator.qpos_ids[:60]]
                    projection = evaluator.shape_spec.project_actual_configuration(actual)
                    base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
                    started = time.perf_counter()
                    coverage = envelope.evaluate(data, projection.planner_configuration, base)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    worst = min(coverage.capsules, key=lambda row: row.margin_m)
                    item = {
                        "mode": mode, "scenario_id": scenario_id,
                        "physics_step": step, "time_s": float(data.time),
                        "status": coverage.status,
                        "minimum_margin_m": coverage.min_margin_m,
                        "worst_geom_name": worst.geom_name,
                        "subspace_residual_linf_rad": projection.residual_linf_rad,
                        "check_ms_excluding_mj_forward": elapsed_ms,
                    }
                    states.write(json.dumps(item, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
                    if coverage.status != "COVERED_AT_THIS_STATE":
                        failures.write(json.dumps({
                            **item, "all_capsules": [asdict(row) for row in coverage.capsules],
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    statuses[coverage.status] += 1
                    margins.append(coverage.min_margin_m)
                    timings.append(elapsed_ms)
                    residuals.append(projection.residual_linf_rad)
                    if worst_item is None or coverage.min_margin_m < worst_item["minimum_margin_m"]:
                        worst_item = item
                    if key in saved:
                        ref = saved[key]
                        if (step % STEPS_PER_TICK
                                or coverage.status != ref["status"]
                                or abs(coverage.min_margin_m - ref["minimum_margin_m"]) > 1e-8):
                            raise ValueError(f"full saved-state cross-check failed: {key}")
                        seen_saved.add(key)
                    if key in sampled:
                        ref = sampled[key]
                        if (coverage.status != ref["status"]
                                or abs(coverage.min_margin_m - ref["minimum_margin_m"]) > 1e-8):
                            raise ValueError(f"selected microstate cross-check failed: {key}")
                        seen_sampled.add(key)
                    if step < len(torque):
                        data.ctrl[:] = torque[step]
                        mujoco.mj_step(model, data)
                result = {
                    "mode": mode, "scenario_id": scenario_id,
                    "torque_steps": len(torque), "checked_states": len(margins),
                    "native_replay_max_state_error": max_replay_error,
                    "status_counts": dict(statuses),
                    "minimum_margin_m": _summary(margins),
                    "worst_state": worst_item,
                    "subspace_residual_linf_rad": _summary(residuals),
                    "check_ms_excluding_mj_forward": _summary(timings),
                }
                report["scenes"].append(result)
                print(f"[b2-full-torque] {mode} {scenario_id}: "
                      f"{len(margins)} states, min={1000*min(margins):.3f} mm, "
                      f"replay error={max_replay_error:.2e}", flush=True)
    if seen_saved != set(saved) or seen_sampled != set(sampled):
        raise ValueError("reference cross-check coverage incomplete")
    report["cross_checks"] = {
        "saved_task_states": len(seen_saved),
        "selected_microstates": len(seen_sampled),
    }
    for mode in ("baseline", "enabled"):
        own = [s for s in report["scenes"] if s["mode"] == mode]
        report["modes"][mode] = {
            "scene_count": len(own),
            "torque_steps": sum(s["torque_steps"] for s in own),
            "checked_states": sum(s["checked_states"] for s in own),
            "covered_states": sum(s["status_counts"].get("COVERED_AT_THIS_STATE", 0)
                                  for s in own),
            "minimum_margin_m": min(s["minimum_margin_m"]["min"] for s in own),
            "maximum_native_replay_error": max(s["native_replay_max_state_error"]
                                               for s in own),
        }
    report["state_records_sha256"] = _sha(states_path)
    report["failure_records_sha256"] = _sha(failures_path)
    summary_path = output_dir / "full_torque_envelope_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 旧 A.1 完整力矩轨迹的 500 Hz 状态局部包络审计", "",
        "两组五场景按原 trace 力矩逐步原生重放。每个物理步起点和最终端点均检查"
        "实际 MuJoCo 胶囊是否包含于原 PCC 管；全部规划边界逐状态核对保存 qpos 和时间。"
        "另与既有 50 Hz 全量扫描和 20 个选定斜坡窗口逐状态核对。", "",
        "| 模式 | 场景 | 力矩步 | 检查状态 | 当前状态包含 | 最小余量 mm | 最大重放误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        lines.append(
            f"| {mode} | {item['scene_count']} | {item['torque_steps']} | "
            f"{item['checked_states']} | {item['covered_states']} | "
            f"{1000*item['minimum_margin_m']:.3f} | "
            f"{item['maximum_native_replay_error']:.2e} |"
        )
    lines += [
        "", f"交叉核对：保存规划状态 {len(seen_saved)} 个；既有选定物理状态"
        f" {len(seen_sampled)} 个。逐状态记录、未包含状态的完整胶囊快照及哈希清单"
        "分别保存。", "",
        "这是历史旧控制轨迹的离散状态几何检查。它没有执行新区间 QP 命令，"
        "不证明相邻 2 ms 状态之间的连续时间包含，也不满足新区间模式的"
        "接入门禁或 20 ms 全链时延验收。包络检查耗时不含 MuJoCo 正运动学、"
        "区间查询、Jacobian、QP 和执行。", "",
    ]
    doc_path = output_dir / "FULL_TORQUE_ENVELOPE.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_full_old_torque_envelope_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("states", states_path),
            ("failures", failures_path), ("document", doc_path),
        )},
    }
    with (output_dir / "full_torque_envelope_manifest.json").open(
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
    parser.add_argument("--full-sweep-dir", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/full_state_envelope"))
    parser.add_argument("--selected-dir", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/microstep_envelope"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root, args.full_sweep_dir,
                 args.selected_dir)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
