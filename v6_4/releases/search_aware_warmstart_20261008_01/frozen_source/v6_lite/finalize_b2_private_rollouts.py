"""Hash-check five private interval-QP rollout diagnostics and summarize limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCENES = tuple(f"v6_lite_scenario_{index:02d}" for index in range(5))
HORIZON = 400


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _mm(value: float | None) -> str:
    return "NOT_EVALUATED" if value is None else f"{1000 * value:.3f}"


def run(root: Path, refined_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    first_old_violation = {}
    for row in refined["records"]:
        if (row["mode"] == "enabled" and row["feasibility"]["status"]
                == "START_CLEARANCE_VIOLATION"):
            scene = row["scenario_id"]
            first_old_violation[scene] = min(
                row["tick"], first_old_violation.get(scene, row["tick"]))
    if set(first_old_violation) != set(SCENES):
        raise ValueError("old first-violation comparison is incomplete")
    rows = []
    for index, scene in enumerate(SCENES):
        folder = root / f"scene_{index:02d}"
        report_path = folder / "private_rollout_summary.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads((folder / "private_rollout_manifest.json")
                              .read_text(encoding="utf-8"))
        files = {"summary": report_path,
                 "records": folder / "private_rollout_records.jsonl",
                 "trace": folder / "private_rollout_trace.npz",
                 "document": folder / "PRIVATE_ROLLOUT.md"}
        for name, path in files.items():
            if manifest[f"{name}_sha256"] != _sha(path):
                raise ValueError(f"{scene} {name} hash changed")
        failure_path = folder / "private_rollout_failure.json"
        if manifest["failure_sha256"] is None:
            if failure_path.exists():
                raise ValueError(f"{scene} has unmanifested failure")
        elif manifest["failure_sha256"] != _sha(failure_path):
            raise ValueError(f"{scene} failure hash changed")
        if (report["scenario_id"] != scene or report["mode"] != "enabled"
                or report["predeclared_horizon_ticks"] != HORIZON
                or report["point_budget"] != 255
                or report["servo_steps_per_task"] != 10
                or report["source_hash_newline_policy"] != "LF_NORMALIZED"
                or report["production_online_controller_changed"]
                or report["stage3_admission"]
                or report["full_cycle_20ms_acceptance"]
                or report["continuous_time_certified"]):
            raise ValueError(f"{scene} protocol or scope changed")
        for name, digest in report["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"{scene} source changed: {name}")
        inputs = report["inputs"]
        if (_sha(Path(inputs["metrics_path"])) != inputs["metrics_sha256"]
                or _sha(Path(inputs["published_trace_path"]))
                != inputs["published_trace_sha256"]):
            raise ValueError(f"{scene} A.1 input changed")
        records = [json.loads(line) for line in files["records"].read_text(
            encoding="utf-8").splitlines()]
        if (report["records_sha256"] != _sha(files["records"])
                or report["trace_sha256"] != _sha(files["trace"])
                or report["attempted_ticks"] != len(records)
                or [row["tick"] for row in records] != list(range(len(records)))):
            raise ValueError(f"{scene} raw record sequence changed")
        with np.load(files["trace"], allow_pickle=False) as trace:
            torques = trace["torque"]
            qpos = trace["qpos_states"]
            qvel = trace["task_qvel_states"]
            if (torques.shape != (10 * report["executed_ticks"], 67)
                    or qpos.shape[0] != 10 * report["executed_ticks"] + 1
                    or qvel.shape[0] != report["executed_ticks"] + 1
                    or not np.all(np.isfinite(torques))
                    or not np.all(np.isfinite(qpos))
                    or not np.all(np.isfinite(qvel))):
                raise ValueError(f"{scene} private native trace shape changed")
        if report["native_replay_max_qpos_error"] > 1e-8:
            raise ValueError(f"{scene} native torque replay diverged")
        executed = [row for row in records if "realized_next_start" in row]
        if len(executed) != report["executed_ticks"]:
            raise ValueError(f"{scene} executed tick count changed")
        for row in records:
            if "row_parity" in row and any(row["row_parity"][field] > 1e-7
                    for field in ("matrix_max_abs_error", "lower_max_abs_error",
                                  "drift_max_abs_error", "gain_max_abs_error")):
                raise ValueError(f"{scene} independent row mismatch")
        old_tick = first_old_violation[scene]
        old_tick_reached = report["executed_ticks"] >= old_tick
        private_at_old_tick = (executed[old_tick - 1]["realized_next_start"]
                               if old_tick_reached else None)
        rows.append({
            "scenario_id": scene,
            "folder": folder.as_posix(),
            "summary_sha256": _sha(report_path),
            "manifest_sha256": _sha(folder / "private_rollout_manifest.json"),
            "attempted_ticks": report["attempted_ticks"],
            "executed_ticks": report["executed_ticks"],
            "stop_reason": report["stop_reason"],
            "old_first_sampled_start_violation_tick": old_tick,
            "old_violation_tick_reached_privately": old_tick_reached,
            "private_start_status_at_old_violation_tick": (
                private_at_old_tick["frozen_rows_status"]
                if private_at_old_tick is not None else "NOT_REACHED"),
            "private_start_slack_at_old_violation_tick_m_s": (
                private_at_old_tick["minimum_start_slack_m_s"]
                if private_at_old_tick is not None else None),
            "minimum_next_start_slack_m_s":
                report["minimum_realized_next_start_slack_m_s"],
            "minimum_ramp_envelope_margin_m":
                report["minimum_ramp_envelope_margin_m"],
            "first_tick_outside_strict_online_domain":
                report["first_tick_outside_strict_online_domain"],
            "maximum_subspace_residual_linf_rad":
                report["maximum_subspace_residual_linf_rad"],
            "maximum_qpos_linf_difference_from_old_a1":
                report["maximum_qpos_linf_difference_from_old_a1"],
            "private_preflight_plus_qp_timing":
                report["private_preflight_plus_qp_timing"],
        })
    summary = {
        "schema": "v6_2_b2_private_five_scene_400_tick_summary_v1",
        "scope": "private_counterfactual_not_stage3_online_acceptance",
        "predeclared_scenarios": list(SCENES),
        "predeclared_horizon_ticks_per_scene": HORIZON,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "finalize_b2_private_rollouts.py", "audit_b2_private_rollout.py",
        )},
        "input_refined_start_sha256": _sha(refined_path),
        "scenes": rows,
        "complete_horizon_count": sum(row["stop_reason"] == "HORIZON_COMPLETE"
                                      for row in rows),
        "old_violation_tick_reached_count": sum(
            row["old_violation_tick_reached_privately"] for row in rows),
        "old_violation_tick_start_satisfied_count": sum(
            row["private_start_status_at_old_violation_tick"]
            == "START_ROWS_SATISFIED" for row in rows),
        "strict_online_admission": False,
        "new_mode_five_scene_acceptance": False,
        "full_cycle_20ms_acceptance": False,
    }
    summary_path = output_dir / "private_five_scene_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 私有五场景多周期诊断", "",
        "从发布的 A.1 五个初始状态分别启动私有 MuJoCo 模型，"
        f"每场景预声明 {HORIZON} 个规划周期。后续状态由新区间 QP 候选"
        "经原 67 路力矩伺服产生，按保存力矩原生重放核对。"
        "这是离线反事实分支，不是生产控制器的开关式在线接入。", "",
        "| 场景 | 私有执行周期 | 旧速度首次违例抽样 tick | 私有对应起点 | "
        "最小下一起点松弛 mm/s | 最小 500 Hz 包络余量 mm | "
        "预检＋QP p95 ms | 超 20 ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        timing = row["private_preflight_plus_qp_timing"]
        lines.append(
            f"| {row['scenario_id'][-2:]} | {row['executed_ticks']} | "
            f"{row['old_first_sampled_start_violation_tick']} | "
            f"{row['private_start_status_at_old_violation_tick']} | "
            f"{_mm(row['minimum_next_start_slack_m_s'])} | "
            f"{_mm(row['minimum_ramp_envelope_margin_m'])} | "
            f"{timing['p95_ms']:.3f} | {timing['over_20ms_count']} |"
            if timing else
            f"| {row['scenario_id'][-2:]} | {row['executed_ticks']} | "
            f"{row['old_first_sampled_start_violation_tick']} | "
            f"{row['private_start_status_at_old_violation_tick']} | "
            f"{_mm(row['minimum_next_start_slack_m_s'])} | "
            f"{_mm(row['minimum_ramp_envelope_margin_m'])} | "
            "NOT_EVALUATED | NOT_EVALUATED |"
        )
    lines += [
        "", f"完整到预声明时域：{summary['complete_horizon_count']}/5；"
        f"到达旧轨迹首次速度违例抽样 tick："
        f"{summary['old_violation_tick_reached_count']}/5；"
        f"其中私有路径起点行满足："
        f"{summary['old_violation_tick_start_satisfied_count']}/5。", "",
        "各场景严格形状子空间判据的首次失败、数值残差、原 MuJoCo／胶囊行"
        "独立重算和新区间行的同批次装配核对均在原始报告中；"
        "新区间函数尚未在独立进程从力矩 trace 重算，"
        "也未调整 `1e-10 rad` 判据。"
        "实际胶囊包含只在所测 500 Hz 状态成立，不构成连续时间证明。"
        "所报计时不含独立重算、力矩伺服及下一起点，不能当作全链验收。"
        "当前门禁仍未批准 stage 3。", "",
    ]
    document_path = output_dir / "PRIVATE_FIVE_SCENE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "v6_2_b2_private_five_scene_manifest_v1",
                "summary_sha256": _sha(summary_path),
                "document_sha256": _sha(document_path)}
    with (output_dir / "private_five_scene_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/private_rollout_400"))
    parser.add_argument("--refined-start", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/refined_start/refined_start_report.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.refined_start, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "complete_horizon_count", "old_violation_tick_reached_count",
        "old_violation_tick_start_satisfied_count", "strict_online_admission",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
