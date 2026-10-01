"""Sweep every saved A.1 task state for state-local PCC capsule containment.

This directly re-evaluates saved qpos with MuJoCo forward kinematics. The
separate B.2 shadow audit already verifies those saved states by native torque
replay. This sweep does not replay torques, execute commands, or certify motion
between task ticks.
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

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def run(output_dir: Path, a1_root: Path, warm_shadow_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    warm_shadow = json.loads(warm_shadow_path.read_text(encoding="utf-8"))
    if (not warm_shadow["passed_as_read_only_audit"]
            or any(warm_shadow["modes"][mode]["counts"].get(
                "native_replay_state_mismatch", 0) for mode in ("baseline", "enabled"))):
        raise ValueError("saved-state provenance lacks passing native replay")
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(robot)
    model, data = verifier.model, verifier.data
    evaluator = FixedIntervalCBFEvaluator(robot, model)
    envelope = StateLocalPCCEnvelopeAudit(model, evaluator.shape_spec)
    if envelope.envelopes.fallback_geom_names != ("collision_0003",):
        raise ValueError("original MuJoCo mount fallback changed")
    state_path = output_dir / "envelope_sweep_states.jsonl"
    failure_path = output_dir / "envelope_sweep_failures.jsonl"
    sources = {name: _source_sha(Path("v6_lite") / name) for name in (
        "audit_b2_full_trace_envelope.py", "pcc_state_local_envelope.py",
        "pcc_clearance.py", "shape_clearance.py", "continuum_model_spec.py",
        "continuum_shape_model.py",
    )}
    report = {
        "schema": "v6_2_b2_full_saved_task_state_envelope_v1",
        "scope": "all_13500_saved_A1_task_qpos_direct_mj_forward_not_torque_replay",
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": sources,
        "source_urdf_sha256": evaluator.shape_spec.source_urdf_sha256,
        "warm_shadow_sha256": _sha(warm_shadow_path),
        "geometry_model": "WholeBodyCollisionVerifier_without_scenario_obstacles",
        "original_tube_radii_m": V61A_PCC_TUBE_RADII_M.tolist(),
        "capsule_count_per_state": len(envelope.envelopes.capsules),
        "fallback_geom_names": list(envelope.envelopes.fallback_geom_names),
        "numerical_certification": "NOT_FORMALLY_CERTIFIED",
        "online_control_changed": False,
        "modes": {},
        "inputs": {},
    }
    with state_path.open("x", encoding="utf-8", newline="\n") as states, \
            failure_path.open("x", encoding="utf-8", newline="\n") as failures:
        for mode in ("baseline", "enabled"):
            metrics_path = a1_root / f"{mode}_root" / "output" / "v6_lite_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            scenes = metrics["scenarios"]
            if len(scenes) != 5 or len({x["scenario"]["scenario_id"] for x in scenes}) != 5:
                raise ValueError("A.1 metrics must contain five unique scenarios")
            warm_scenes = {item["scenario_id"]: item for item in
                           warm_shadow["modes"][mode]["scenarios"]}
            if len(warm_scenes) != 5:
                raise ValueError("native replay audit scenario count changed")
            report["inputs"][mode] = {
                "metrics_path": metrics_path.as_posix(),
                "metrics_sha256": _sha(metrics_path),
                "traces": [],
            }
            mode_margins: list[float] = []
            mode_residuals: list[float] = []
            mode_check_ms: list[float] = []
            status_counts: Counter[str] = Counter()
            first_uncovered: dict[str, int] = {}
            for scene in scenes:
                scenario_id = scene["scenario"]["scenario_id"]
                trace_path = Path(scene["trace"]["path"])
                trace_sha = _sha(trace_path)
                if trace_sha != scene["trace"]["sha256"]:
                    raise ValueError(f"A.1 trace hash mismatch: {trace_path}")
                if (scenario_id not in warm_scenes
                        or not warm_scenes[scenario_id]["trace_hash_matches"]
                        or warm_scenes[scenario_id]["trace_sha256"] != trace_sha
                        or warm_scenes[scenario_id]["max_native_replay_state_error"] > 1e-8):
                    raise ValueError("full sweep trace differs from native replay audit")
                report["inputs"][mode]["traces"].append({
                    "scenario_id": scenario_id,
                    "path": trace_path.as_posix(),
                    "sha256": trace_sha,
                })
                with np.load(trace_path, allow_pickle=False) as trace:
                    task_qpos = trace["task_qpos"].copy()
                    task_time = trace["task_time"].copy()
                if (task_qpos.shape != (1351, model.nq)
                        or task_time.shape != (1350,)):
                    raise ValueError("A.1 saved task-state protocol changed")
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
                for tick, time_s in enumerate(task_time):
                    data.qpos[:] = task_qpos[tick]
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
                    record = {
                        "mode": mode,
                        "scenario_id": scenario_id,
                        "tick": tick,
                        "time_s": float(time_s),
                        "task_qpos_sha256": hashlib.sha256(task_qpos[tick].tobytes()).hexdigest(),
                        "subspace_residual_linf_rad": projection.residual_linf_rad,
                        "status": coverage.status,
                        "minimum_margin_m": coverage.min_margin_m,
                        "worst_capsule": asdict(worst),
                        "check_ms": elapsed_ms,
                    }
                    states.write(json.dumps(record, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
                    if coverage.status != "COVERED_AT_THIS_STATE":
                        first_uncovered.setdefault(scenario_id, tick)
                        failures.write(json.dumps({
                            **record,
                            "all_capsules": [asdict(row) for row in coverage.capsules],
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    mode_margins.append(coverage.min_margin_m)
                    mode_residuals.append(projection.residual_linf_rad)
                    mode_check_ms.append(elapsed_ms)
                    status_counts[coverage.status] += 1
                print(f"[b2-envelope-sweep] {mode} {scenario_id}: {len(task_time)} states",
                      flush=True)
            report["modes"][mode] = {
                "state_count": len(mode_margins),
                "status_counts": dict(status_counts),
                "first_uncovered_tick_by_scenario": first_uncovered,
                "minimum_margin_m": _summary(mode_margins),
                "subspace_residual_linf_rad": _summary(mode_residuals),
                "check_ms_excluding_mj_forward": _summary(mode_check_ms),
            }
    if any(report["modes"][mode]["state_count"] != 6750
           for mode in ("baseline", "enabled")):
        raise ValueError("full A.1 task-state count changed")
    report["state_records_sha256"] = _sha(state_path)
    report["failure_records_sha256"] = _sha(failure_path)
    report_path = output_dir / "envelope_sweep_summary.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 全部旧 A.1 规划状态的实际胶囊包络审计", "",
        "两组五场景，每场景 1350 个保存的规划状态，总计 13500 个。"
        "逐状态以 MuJoCo `mj_forward` 重算实际胶囊轴，再检查现有 PCC 管半径。"
        "旧 trace 的原生力矩重放一致性由已哈希绑定的影子审计另行完成；"
        "此扫描自身不重放力矩、不执行新区间命令。", "",
        "| 模式 | 状态数 | 当前状态几何包含 | 未包含 | 最小余量 mm | 检查 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"][mode]
        lines.append(
            f"| {mode} | {item['state_count']} | "
            f"{item['status_counts'].get('COVERED_AT_THIS_STATE', 0)} | "
            f"{item['status_counts'].get('NOT_COVERED_AT_THIS_STATE', 0)} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['check_ms_excluding_mj_forward']['p95']:.3f} |"
        )
    lines += [
        "", "保存状态逐行记录与所有未包含时刻的完整胶囊快照分别保存在两个 JSONL。"
        "检查耗时不含 MuJoCo `mj_forward`、原区间查询和 QP，不能用于证明 20 ms 全链。"
        "本检查只证明所测时刻的模型几何包含；无浮点形式认证、跨 tick 保持、"
        "十步斜坡或连续时间安全结论。安装块继续由原 MuJoCo 约束处理。", "",
    ]
    doc_path = output_dir / "ENVELOPE_SWEEP.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_full_saved_task_state_envelope_manifest_v1",
        "summary_sha256": _sha(report_path),
        "summary_bytes": report_path.stat().st_size,
        "states_sha256": _sha(state_path),
        "states_bytes": state_path.stat().st_size,
        "failures_sha256": _sha(failure_path),
        "failures_bytes": failure_path.stat().st_size,
        "document_sha256": _sha(doc_path),
        "document_bytes": doc_path.stat().st_size,
    }
    with (output_dir / "envelope_sweep_manifest.json").open(
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
    parser.add_argument("--warm-shadow", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/shadow_warm/shadow_report.json"))
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.warm_shadow)
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
