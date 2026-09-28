"""Build V6.2-A.1 evidence from this version's traces and replay reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "v6_lite" / "output" / "v6_2_a1"
BASE_COMMIT = "a77ca2c89d0f8fa56e6ab1304588501a8bafa1cf"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path, evidence_type: str) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "evidence_type": evidence_type,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _short_summary(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        count = len(trace["time"])
        task_count = len(trace["task_time"])
        failures = int(np.sum(trace["task_failure_reason"] != "none"))
        ramp_min = float(np.min(trace["task_ramp_clearance_min_slack_m_s"]))
        next_min = float(np.min(trace["task_lookahead_min_slack_m_s"]))
        checks = {
            "1000_native_physics_steps": count == 1000,
            "100_task_ticks": task_count == 100,
            "no_uncertified_command": failures == 0,
            "current_ramp_feasible": ramp_min >= -1e-4,
            "predicted_next_start_feasible": next_min >= -1e-4,
            "selected_matches_command": np.array_equal(
                trace["task_selected_command"], trace["command_velocity"][::10]
            ),
            "execution_diagnostics_present": all(
                key in trace for key in (
                    "measured_planner_q_before_servo", "measured_velocity_before_servo",
                    "reference_joint_limit_clip_count", "acceleration_clip_count",
                    "torque_saturation_count", "wall_time_since_start_s",
                )
            ),
        }
        return {
            "evidence_type": "new_version_two_second_mujoco_closed_loop_trace_check",
            "passed": all(checks.values()),
            "checks": checks,
            "physics_steps": count,
            "task_ticks": task_count,
            "minimum_ramp_residual_m_s": ramp_min,
            "minimum_next_start_residual_m_s": next_min,
            "trace_sha256": _sha(path),
        }


def build() -> dict:
    frozen_json = EVIDENCE_ROOT / "frozen_failure" / "traces" / "v6_lite_scenario_00_counterexample.json"
    frozen_trace = EVIDENCE_ROOT / "frozen_failure" / "traces" / "v6_lite_scenario_00_partial_trace.npz"
    margin_json = EVIDENCE_ROOT / "lookahead_margin_diagnostic" / "baseline_scenario_00_9p32s.json"
    margin_trace = EVIDENCE_ROOT / "lookahead_margin_diagnostic" / "baseline_scenario_00_partial_trace.npz"
    short_trace = EVIDENCE_ROOT / "short_enabled" / "traces" / "v6_lite_scenario_00.npz"
    required = [frozen_json, frozen_trace, margin_json, margin_trace, short_trace]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"missing required local evidence: {missing}")
    frozen = _load(frozen_json)
    margin_diagnostic = _load(margin_json)
    if frozen["partial_trace"]["sha256"] != _sha(frozen_trace):
        raise ValueError("frozen failure snapshot and partial trace do not match")
    if margin_diagnostic["partial_trace"]["sha256"] != _sha(margin_trace):
        raise ValueError("second failure snapshot and partial trace do not match")
    short = _short_summary(short_trace)
    short_path = EVIDENCE_ROOT / "short_enabled" / "short_validation.json"
    short_path.write_text(json.dumps(short, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    artifacts = [
        _entry(frozen_json, "frozen_three_cycle_failure_snapshot"),
        _entry(frozen_trace, "frozen_partial_native_trace"),
        _entry(margin_json, "frozen_baseline_margin_failure_snapshot"),
        _entry(margin_trace, "frozen_baseline_margin_partial_native_trace"),
        _entry(short_trace, "new_two_second_native_closed_loop_trace"),
        _entry(short_path, "new_two_second_trace_validation"),
    ]
    groups = {}
    for name in ("baseline", "enabled"):
        output = EVIDENCE_ROOT / f"{name}_root" / "output"
        paths = {
            "metrics": output / "v6_lite_metrics.json",
            "original_26_replay": output / "validation.json",
            "execution_contract_validation": output / "execution_validation.json",
            "run_manifest": output / "artifact_manifest.json",
        }
        if not all(path.is_file() for path in paths.values()):
            groups[name] = {"status": "missing", "missing": [
                key for key, path in paths.items() if not path.is_file()
            ]}
            continue
        metrics = _load(paths["metrics"])
        replay = _load(paths["original_26_replay"])
        execution = _load(paths["execution_contract_validation"])
        manifest = _load(paths["run_manifest"])
        config_bytes = json.dumps(
            {"run": metrics["run_config"], "qp": metrics["qp_config"]},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        configuration_sha256 = hashlib.sha256(config_bytes).hexdigest()
        model_identity_matches = (
            metrics["robot_model_identity"]["runtime_contract_sha256"]
            == frozen["model_runtime_contract_sha256"]
        )
        trace_entries = []
        for item in manifest["traces"]:
            path = ROOT / item["path"]
            if not path.is_file() or _sha(path) != item["sha256"]:
                raise ValueError(f"missing or changed full trace: {path}")
            trace_entries.append(_entry(path, f"new_{name}_native_closed_loop_trace"))
        artifacts.extend(_entry(path, f"new_{name}_{key}") for key, path in paths.items())
        artifacts.extend(trace_entries)
        groups[name] = {
            "status": "verified" if (
                len(trace_entries) == 5 and metrics["passed"]
                and replay["passed"] and replay["total_count"] == 26
                and execution["passed"] and model_identity_matches
            ) else "failed",
            "scenario_count": len(trace_entries),
            "run_passed": metrics["passed"],
            "original_26_replay": f"{replay['passed_count']}/{replay['total_count']}",
            "execution_validation": f"{execution['passed_count']}/{execution['total_count']}",
            "execution_passed": execution["passed"],
            "aggregate_metrics": metrics["aggregate_metrics"],
            "configuration_sha256": configuration_sha256,
            "model_identity_matches_frozen": model_identity_matches,
            "instantaneous_binding_total": sum(
                scenario["metrics"]["avoidance"]["instantaneous_binding_constraint_total"]
                for scenario in metrics["scenarios"]
            ),
            "lookahead_binding_total": sum(
                scenario["metrics"]["avoidance"]["lookahead_binding_constraint_total"]
                for scenario in metrics["scenarios"]
            ),
        }
    report = {
        "version": "v6_2_a1_ramp_aware_qp",
        "evidence_type": "new_version_evidence_manifest",
        "base_commit": BASE_COMMIT,
        "model_runtime_contract_sha256": frozen["model_runtime_contract_sha256"],
        "frozen_configuration_sha256": frozen["configuration_sha256"],
        "margin_diagnostic_configuration_sha256": margin_diagnostic["configuration_sha256"],
        "code_sha256": {
            relative: _sha(ROOT / relative)
            for relative in (
                "v6_lite/hierarchical_qp.py", "v6_lite/execution_ramp.py",
                "v6_lite/safety_contract.py", "v6_lite/run_v6_lite.py",
            )
        },
        "short_enabled": short,
        "groups": groups,
        "artifacts": artifacts,
        "complete": short["passed"] and all(
            groups[name]["status"] == "verified" for name in ("baseline", "enabled")
        ),
    }
    manifest_path = EVIDENCE_ROOT / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _render_markdown(report, frozen, margin_diagnostic)
    return report


def _render_markdown(report: dict, frozen: dict, margin_diagnostic: dict) -> None:
    root_cause = frozen["cross_cycle_report"]
    row_action_change = (
        root_cause["residual_change_m_s"] + root_cause["lower_change_m_s"]
    )
    second_root = margin_diagnostic["cross_cycle_report"]
    second_gradient_change = (
        second_root["residual_change_m_s"] + second_root["lower_change_m_s"]
    )
    lines = [
        "# V6.2-A.1 斜坡感知 QP 证据",
        "",
        "> 由 `python -m v6_lite.finalize_v6_2_a1` 从本版本状态快照、trace、",
        "重放 JSON 自动生成；历史 V6.1-B 报告不作为本版本通过证据。",
        "",
        f"开发基点：`{BASE_COMMIT}`；当前证据完整：`{report['complete']}`。",
        "",
        "## 冻结反例与根因",
        "",
        f"保存了 `{len(frozen['planning_snapshots'])}` 次规划状态，",
        f"时刻分别为 `{', '.join(f'{item['time_s']:.3f}' for item in frozen['planning_snapshots'])} s`。",
        f"主导行：`{root_cause['dominant_source']}`。",
        f"上一周期候选残差 `{root_cause['previous_candidate_residual_m_s']:.6g} m/s`；",
        f"当前旧命令残差 `{root_cause['current_old_command_residual_m_s']:.6g} m/s`；",
        f"变化 `{root_cause['residual_change_m_s']:.6g} m/s`，",
        f"其中约束下界变化 `+{root_cause['lower_change_m_s']:.6g} m/s`，",
        f"同一旧动作的梯度项变化 `{row_action_change:+.6g} m/s`。",
        "原快照的斜坡起点仍被拒绝；不能由重新选择终点修复。",
        f"首次关闭组在 `9.320 s` 另一次正确拒绝：`{second_root['dominant_source']}`；",
        f"跨周期残差变化 `{second_root['residual_change_m_s']:.6g} m/s`，",
        f"其中几何梯度作用变化 `{second_gradient_change:+.6g} m/s`。",
        "",
        "## 新增前瞻裕度",
        "",
        "单个 17 维 QP 保留原 MuJoCo、PCC、胶囊行；额外行采用",
        "10 个 500 Hz 插值速度的精确均值（旧端点权重 0.45、新端点权重 0.55），",
        "预测下一周期起点的冻结线性化 CBF 残差。",
        "`lookahead_model_margin_m_s = 0.005 m/s` 是经验裕度，",
        "来源于原 0.460 s 反例和首次关闭组 9.320 s 反例的跨周期几何行变化；",
        "有效域仅限当前 MuJoCo 模型、20 ms 任务周期与已测场景，",
        "证据类别为有限场景经验值，不是严格误差上界或连续时间证明。",
        "现有安全距离、PCC 半径与数值可行性容差未减小或扩大。",
        "",
        "## 新版本运行与真实重放",
        "",
        f"2 s 开启版短程：`{'通过' if report['short_enabled']['passed'] else '失败'}`，",
        f"`{report['short_enabled']['physics_steps']}` 次物理步，",
        f"`{report['short_enabled']['task_ticks']}` 次规划。",
        "",
        "原有 26 项的门槛与检查名称保持；其中“避障已介入”统计的是",
        "当前 QP 全部安全行的绑定次数。报告另列瞬时行与前瞻行分项，",
        "避免把提前避障误写成旧瞬时行绑定。",
        "",
        "| 模式 | 运行 | 原 26 项 MuJoCo 力矩重放 | 新执行合同验证 | 瞬时/前瞻绑定 | 状态 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for name in ("baseline", "enabled"):
        group = report["groups"][name]
        if group["status"] == "missing":
            lines.append(f"| {name} | 缺失 | 未运行 | 未运行 | — | 缺少 {', '.join(group['missing'])} |")
        else:
            lines.append(
                f"| {name} | {group['run_passed']} | {group['original_26_replay']} | "
                f"{group['execution_validation']} | "
                f"{group['instantaneous_binding_total']}/{group['lookahead_binding_total']} | "
                f"{group['status']} |"
            )
    lines += [
        "",
        "本阶段验证的是有限 MuJoCo 场景、冻结行的离散前瞻及保存力矩的真实重放；",
        "不构成采样间连续时间碰撞证明。未通过的项目按 manifest 保留，",
        "不能用历史产物或仿真中止替代。",
        "",
    ]
    (ROOT / "docs" / "V6_2A1_RAMP_AWARE_EVIDENCE.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    result = build()
    print(json.dumps({"complete": result["complete"], "groups": result["groups"]},
                     ensure_ascii=False, indent=2))
