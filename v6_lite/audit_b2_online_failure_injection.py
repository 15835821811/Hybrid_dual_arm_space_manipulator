"""Inject new-mode failures and verify no uncertified servo step executes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from v6_lite import b2_interval_runtime, run_v6_lite
from v6_lite.b2_interval_online import IntervalPreflightFailure
from v6_lite.b2_interval_online_optimized import (
    OptimizedBoundedIntervalAdmission, OptimizedBoundedIntervalVelocityQP,
)
from v6_lite.hierarchical_qp import HierarchicalQPConfig


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    spec = run_v6_lite.default_v6_lite_robot_spec()
    cfg = run_v6_lite.V6LiteRunConfig(
        duration_s=1.6, pcc_mode="bounded_interval_pcc")
    qp_cfg = HierarchicalQPConfig(enable_capsule_cbf=True)
    scenario = run_v6_lite.build_scenarios(spec, cfg)[0]
    original_preview = b2_interval_runtime.preview_ramp

    def invalid_query(_self, data, previous):
        raise IntervalPreflightFailure("INJECTED_QUERY_UNKNOWN", {
            "time_s": float(data.time), "injection": "invalid_query"})

    def failed_qp(_self, data, *args, **kwargs):
        raise RuntimeError("injected QP infeasibility")

    def uncovered_preview(*args, **kwargs):
        branch, next_data = original_preview(*args, **kwargs)
        branch["coverage"][0]["status"] = "NOT_COVERED_AT_THIS_STATE"
        return branch, next_data

    cases = (
        ("query_unknown", OptimizedBoundedIntervalAdmission,
         "prepare", invalid_query, "INJECTED_QUERY_UNKNOWN", "interval"),
        ("qp_failure", OptimizedBoundedIntervalVelocityQP,
         "solve", failed_qp, "INTERVAL_QP_EXCEPTION", "interval"),
        ("ramp_uncovered", b2_interval_runtime,
         "preview_ramp", uncovered_preview,
         "RAMP_MICROSTATE_NOT_ENVELOPED", "interval"),
        ("expired_command", run_v6_lite,
         "command_is_current", lambda *_: False,
         "expired_command", "execution"),
    )
    records = []
    for name, target, attribute, replacement, expected, family in cases:
        trace_dir = output_dir / name / "traces"
        trace_dir.mkdir(parents=True, exist_ok=False)
        thrown = None
        with patch.object(target, attribute, replacement):
            try:
                run_v6_lite.run_scenario(
                    spec, cfg, qp_cfg, scenario, trace_dir)
            except run_v6_lite.UncertifiedExecutionError as error:
                thrown = str(error)
        failure_dir = output_dir / name / "failures"
        failure_path = failure_dir / (
            f"{scenario.scenario_id}_interval_failure.json" if family == "interval"
            else f"{scenario.scenario_id}_execution_failure.json")
        partial_path = failure_dir / (
            f"{scenario.scenario_id}_interval_partial_trace.npz" if family == "interval"
            else f"{scenario.scenario_id}_partial_trace.npz")
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        with np.load(partial_path, allow_pickle=False) as trace:
            torque_steps = len(trace["torque"])
        actual_reason = failure["failure_reason"]
        record = {
            "case": name, "expected_reason": expected,
            "actual_reason": actual_reason,
            "uncertified_error_thrown": thrown is not None,
            "next_servo_step_executed": failure["next_servo_step_executed"],
            "saved_torque_steps": torque_steps,
            "failure_sha256": _sha(failure_path),
            "partial_trace_sha256": _sha(partial_path),
        }
        record["passed"] = (
            actual_reason == expected and thrown is not None
            and not record["next_servo_step_executed"]
            and torque_steps == 0)
        records.append(record)
    report = {
        "schema": "v6_2_b2_online_failure_injection_v1",
        "passed": all(item["passed"] for item in records),
        "records": records,
    }
    (output_dir / "failure_injection_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
