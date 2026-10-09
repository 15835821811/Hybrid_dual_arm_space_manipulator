"""Run the existing private Cholesky QP through a pre-servo domain gate.

This is a private stage-2 diagnostic. It preserves the old diagnostic traces
and leaves the production online controller unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v6_lite import audit_b2_private_rollout as private_loop
from v6_lite import audit_b2_strict_domain_private_rollout as strict_loop
from v6_lite.audit_b2_cholesky_private_rollout import run as cholesky_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, a1_root: Path, scenario_id: str,
        max_ticks: int) -> dict:
    original_run = private_loop.run
    try:
        private_loop.run = strict_loop.run
        report = cholesky_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        private_loop.run = original_run
    report["schema"] = "v6_2_b2_strict_domain_cholesky_private_v1"
    report["scope"] = (
        "private_interval_qp_with_predicted_11_state_domain_gate_before_torque_commit"
    )
    report["source_sha256"]["audit_b2_strict_cholesky_private_rollout.py"] = (
        _source_sha(Path("v6_lite/audit_b2_strict_cholesky_private_rollout.py")))
    report["strict_gate_rejected_before_servo"] = (
        report["stop_reason"] == "RAMP_MICROSTATE_OUTSIDE_DECLARED_WORK_DOMAIN"
        and report["failure_record"] is not None
        and report["failure_record"]["selected_command"] is None
        and report["failure_record"]["next_servo_step_executed"] is False
        and report["attempted_ticks"] == report["executed_ticks"] + 1
    )
    if report["stop_reason"] == "RAMP_MICROSTATE_OUTSIDE_DECLARED_WORK_DOMAIN":
        if not report["strict_gate_rejected_before_servo"]:
            raise ValueError("domain rejection did not precede servo execution")
        with np.load(output_dir / "private_rollout_trace.npz", allow_pickle=False) as trace:
            if (len(trace["torque"]) != 10 * report["executed_ticks"]
                    or len(trace["qpos_states"]) != 10 * report["executed_ticks"] + 1):
                raise ValueError("rejected candidate leaked into torque trace")
    summary_path = output_dir / "private_rollout_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    document_path = output_dir / "PRIVATE_ROLLOUT.md"
    with document_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "\n严格私有门禁从同一候选模拟十步力矩斜坡，检查 11 个实际构型的"
            "声明工作域，再决定是否提交力矩。"
            f"本次拒绝发生在 tick {report['first_preflight_domain_rejection_tick']}；"
            f"已执行 {report['executed_ticks']} 个周期，"
            "拒绝候选没有写入下一伺服步。"
            "这不是生产在线接入或连续时间保证。\n"
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
        "first_preflight_domain_rejection_tick",
        "strict_gate_rejected_before_servo",
        "native_replay_max_qpos_error",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
