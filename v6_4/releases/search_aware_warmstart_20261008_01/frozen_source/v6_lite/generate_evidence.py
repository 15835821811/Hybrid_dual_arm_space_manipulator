"""Render evidence numbers from JSON; never convert artifact checks into replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "v6_lite" / "output" / "v6_1_b"
BASE_COMMIT = "ce619ddc9512c2f0e4d88b35484e7da14a72317a"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_historical_artifacts() -> dict:
    manifest = _load(HISTORY / "v6_1_b_manifest.json")
    checks = {}
    for entry in manifest["artifacts"]:
        path = ROOT / entry["path"]
        checks[entry["path"]] = (
            path.is_file() and path.stat().st_size == entry["bytes"]
            and _hash(path) == entry["sha256"]
        )
    return {
        "evidence_type": "historical_artifact_integrity_only",
        "source_commit": BASE_COMMIT,
        "passed": all(checks.values()),
        "checks": checks,
        "native_mujoco_replay_performed": False,
    }


def render_markdown() -> str:
    integrity = check_historical_artifacts()
    audit_path = HISTORY / "pcc_audit.json"
    regression_path = HISTORY / "regression_report.json"
    audit = _load(audit_path)
    regression = _load(regression_path)
    baseline = regression["baseline"]
    enabled = regression["enabled_control"]
    gradient = audit["distance_gradient"]["relative_error"]["maximum"]
    lines = [
        "# V6.2-A 证据与验收边界",
        "",
        "> 本页由 `python -m v6_lite.generate_evidence` 从仓库 JSON 自动生成。",
        "",
        f"历史基点：`v6.1-b@{BASE_COMMIT[:7]}`。以下数字为该提交的已提交产物，",
        "不是本分支的真实物理重放。历史产物完整性检查仅核对文件大小与 SHA-256。",
        f"完整性结果：`{'通过' if integrity['passed'] else '失败'}`；",
        f"已核对 `{len(integrity['checks'])}` 个文件。",
        "",
        f"- 审计 JSON SHA-256：`{_hash(audit_path)}`",
        f"- 回归 JSON SHA-256：`{_hash(regression_path)}`",
        f"- PCC 距离梯度最大相对误差：`{gradient:.10g}`（来自 `pcc_audit.json`，",
        "  替代旧文档中与审计产物不一致的数字）。",
        "",
        "| 历史模式 | 已提交独立验证 | QP 失败 | 50 Hz 全链 p95 最差值 | 连续体—卫星 500 Hz 最小间隙 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in (("默认关闭", baseline), ("PCC+胶囊", enabled)):
        lines.append(
            f"| {name} | {item['validation_checks']} | {item['qp_failure_count']} | "
            f"{item['task_full_latency_p95_ms']:.6f} ms | "
            f"{1000 * item['continuum_target_500hz_minimum_clearance_m']:.6f} mm |"
        )
    lines += [
        "",
        "## 当前分支真实重放",
        "",
        "当前分支仍沿用原五场景、26 项 MuJoCo 独立重放检查及其门槛。",
        "`python -m v6_lite.validate_v6_lite` 会从保存的力矩和初态逐步执行",
        "`data.ctrl + mj_step`，重新计算状态、任务误差与碰撞间隙。",
        "历史产物检查命令 `python -m v6_lite.generate_evidence --historical-check`",
        "不会运行 MuJoCo，也不能代替该重放。V6.1-B 启用版原始 trace 未在 Git 中；",
        "其已提交 26/26 报告只能作为历史记录。",
        "",
        "执行合同只验证当前线性化 QP 行及命令速度斜坡；它不提供",
        "采样间连续时间安全证明，也不覆盖目标预测误差和力矩伺服跟踪误差。",
        "BACKUP 状态保留给未来经恢复域验证的控制策略；本阶段没有这样的策略。",
        "",
    ]
    historical_replay_path = (
        ROOT / "v6_lite" / "output" / "v6_2_a" / "historical_baseline_replay.json"
    )
    if historical_replay_path.is_file():
        historical_replay = _load(historical_replay_path)
        if (historical_replay.get("evidence_type") == "native_mujoco_torque_replay"
                and historical_replay.get("contract_version") == "v6_lite_6"):
            lines += [
                "本机额外真实重放了已提交的 `v6_lite_6` 历史基线 trace：",
                f"`{historical_replay['passed_count']}/{historical_replay['total_count']}`；",
                f"结果 SHA-256：`{_hash(historical_replay_path)}`。",
                "这验证的是历史基线，不是当前执行合同或 V6.1-B 启用版的重放。",
                "",
            ]
    failure_path = (
        ROOT / "v6_lite" / "output" / "v6_2_a" / "counterexamples"
        / "enabled_scenario_00_0p46s.json"
    )
    if failure_path.is_file():
        failure = _load(failure_path)
        lines += [
            "短程 MuJoCo 执行门诊断（PCC+胶囊开启，2 s 配置，",
            "**非五场景验收**）：",
            f"在 `{failure['time_s']:.3f} s` 返回 `{failure['failure_reason']}`；",
            f"候选距离行最小残差 `{failure['candidate_clearance_min_slack_m_s']:.6g} m/s`，",
            f"旧命令到候选斜坡的最小残差 `{failure['ramp_clearance_min_slack_m_s']:.6g} m/s`。",
            "下一次力矩更新未执行，未将该场景记作成功。",
            f"事件文件：`{failure_path.relative_to(ROOT)}`。",
            "",
        ]
    validation_path = ROOT / "v6_lite" / "output" / "v6_2_a" / "validation.json"
    if validation_path.is_file():
        current = _load(validation_path)
        if current.get("evidence_type") == "native_mujoco_torque_replay":
            lines += [
                f"当前分支重放结果：`{current['passed_count']}/{current['total_count']}`；",
                f"通过：`{current['passed']}`。源文件：`{validation_path.relative_to(ROOT)}`。",
                "",
            ]
    else:
        lines += ["当前分支完整重放：尚无可核对的 26 项结果。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-check", action="store_true")
    args = parser.parse_args()
    if args.historical_check:
        result = check_historical_artifacts()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        path = ROOT / "docs" / "V6_2A_EVIDENCE.md"
        path.write_text(render_markdown(), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
