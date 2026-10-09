"""Run the strict private torque loop with work-domain rows in the same QP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_cholesky_private_rollout as cholesky_module
from v6_lite.audit_b2_strict_cholesky_private_rollout import run as strict_run
from v6_lite.b2_work_domain_velocity_box import WorkDomainBoundedCholeskyQP


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    original_qp = cholesky_module.CholeskyUnconstrainedScreenQP
    try:
        cholesky_module.CholeskyUnconstrainedScreenQP = WorkDomainBoundedCholeskyQP
        report = strict_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        cholesky_module.CholeskyUnconstrainedScreenQP = original_qp
    report["schema"] = "v6_2_b2_work_domain_qp_private_v1"
    report["scope"] = (
        "private_17d_weighted_qp_with_shape_domain_ramp_endpoint_rows_"
        "and_pre_servo_11_state_domain_gate"
    )
    report["work_domain_endpoint_rows_in_same_qp"] = True
    report["work_domain_box_is_only_kinematic_necessary_condition"] = True
    report["work_domain_microstate_preflight_still_required"] = True
    report["source_sha256"].update({
        name: _source_sha(Path("v6_lite") / name) for name in (
            "b2_work_domain_velocity_box.py",
            "audit_b2_work_domain_qp_private_rollout.py",
        )
    })
    summary_path = output_dir / "private_rollout_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    document_path = output_dir / "PRIVATE_ROLLOUT.md"
    with document_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "\n本变体将声明形状工作域的十步平均速度端点不等式作为独立终点行，"
            "仍由同一个 17 维加权 QP 求解；原速度盒与斜坡起点验证保持原义。"
            "这些行只约束冻结运动学预测；实际 11 个"
            "MuJoCo 状态仍需逐个预检，失败时不得提交下一力矩步。"
            "原物理半径、安全距离、执行器限制和生产在线模式不变。\n"
        )
    manifest_path = output_dir / "private_rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"] = _sha(summary_path)
    manifest["document_sha256"] = _sha(document_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                             + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--max-ticks", type=int, default=900)
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.scenario_id,
                 args.max_ticks)
    print(json.dumps({key: report[key] for key in (
        "scenario_id", "attempted_ticks", "executed_ticks", "stop_reason",
        "first_preflight_domain_rejection_tick", "native_replay_max_qpos_error",
        "private_preflight_plus_qp_timing",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
