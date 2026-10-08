"""Measure QP stages inside the latest exact-output private B.2 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v6_lite import audit_b2_current_qp_breakdown as probe
from v6_lite.audit_b2_vectorized_positions_private_timing import (
    run as vectorized_run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(output_dir: Path, reference_dir: Path, batched_reference_dir: Path,
        compact_reference_dir: Path, screened_dir: Path, fixed_dir: Path,
        baseline_dir: Path, a1_root: Path, scenario_id: str,
        ticks: int) -> dict:
    original = probe.prepared_run

    def latest(output_dir_arg, _screened_dir, _fixed_dir, _baseline_dir,
               _a1_root, _scenario_id, _ticks):
        return vectorized_run(
            output_dir_arg, batched_reference_dir, compact_reference_dir,
            screened_dir, fixed_dir, baseline_dir, a1_root,
            scenario_id, ticks)

    try:
        probe.prepared_run = latest
        detailed = probe.run(output_dir, reference_dir, screened_dir,
                             fixed_dir, baseline_dir, a1_root,
                             scenario_id, ticks)
    finally:
        probe.prepared_run = original
    report = {
        "schema": "v6_2_b2_latest_private_qp_breakdown_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "all_non_timing_records_exact": detailed["all_non_timing_records_exact"],
        "trajectory_exact_to_reference": detailed["trajectory_exact_to_reference"],
        "qp_iterations_p95": detailed["qp_iterations_p95"],
        "metrics": detailed["metrics"],
        "full_online_cycle_evaluated": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_latest_qp_breakdown.py",
                              "audit_b2_current_qp_breakdown.py",
                              "audit_b2_vectorized_positions_private_timing.py",
                              "b2_screened_next_start_rows.py",
                              "recompute_execution_constraints.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                reference_dir / "private_rollout_records.jsonl"),
            "reference_trace": _sha(
                reference_dir / "private_rollout_trace.npz"),
            "new_records": _sha(
                output_dir / "private_rollout_records.jsonl"),
            "new_trace": _sha(
                output_dir / "private_rollout_trace.npz"),
            "breakdown_rows": _sha(output_dir / "qp_breakdown_rows.jsonl"),
        },
    }
    if report["inputs_sha256"]["reference_trace"] != report[
            "inputs_sha256"]["new_trace"]:
        raise ValueError("instrumentation changed the 67-channel torque trace")
    (output_dir / "latest_qp_breakdown_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--batched-reference-dir", type=Path, required=True)
    parser.add_argument("--compact-reference-dir", type=Path, required=True)
    parser.add_argument("--screened-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    result = run(args.output_dir, args.reference_dir,
                 args.batched_reference_dir, args.compact_reference_dir,
                 args.screened_dir, args.fixed_dir, args.baseline_dir,
                 args.a1_root, args.scenario_id, args.ticks)
    print(json.dumps({"qp_iterations_p95": result["qp_iterations_p95"],
                      "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
