"""Compare two saved B.2 online runs while excluding wall-clock fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TIMING_KEYS = {
    "torque_latency", "task_full_latency", "task_solver_latency",
    "task_shape_clearance_latency", "task_wall_time_since_start_s",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(reference_dir: Path, candidate_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for scene in range(5):
        name = f"v6_lite_scenario_{scene:02d}.npz"
        reference_path = reference_dir / "traces" / name
        candidate_path = candidate_dir / "traces" / name
        mismatches = []
        maximum_error = 0.0
        with np.load(reference_path, allow_pickle=False) as reference, \
                np.load(candidate_path, allow_pickle=False) as candidate:
            if set(reference.files) != set(candidate.files):
                raise ValueError(f"trace key set changed for {name}")
            for key in reference.files:
                if (key in TIMING_KEYS or "latency" in key
                        or "wall_time" in key):
                    continue
                old, new = reference[key], candidate[key]
                if old.shape != new.shape or old.dtype != new.dtype:
                    mismatches.append({"key": key, "reason": "shape_or_dtype"})
                    continue
                if np.issubdtype(old.dtype, np.number):
                    if not np.allclose(old, new, rtol=0.0, atol=1e-9,
                                       equal_nan=True):
                        mismatches.append({"key": key, "reason": "value"})
                    finite = np.isfinite(old) & np.isfinite(new)
                    if np.any(finite):
                        maximum_error = max(maximum_error, float(np.max(
                            np.abs(old[finite] - new[finite]))))
                elif not np.array_equal(old, new):
                    mismatches.append({"key": key, "reason": "value"})
            torque_error = float(np.max(np.abs(
                reference["torque"] - candidate["torque"])))
            selected_error = float(np.max(np.abs(
                reference["task_selected_command"]
                - candidate["task_selected_command"])))
            task_ticks = len(candidate["task_selected_command"])
            torque_steps = len(candidate["torque"])
        records.append({
            "scenario_id": name[:-4],
            "task_ticks": task_ticks,
            "torque_steps": torque_steps,
            "reference_trace_sha256": _sha(reference_path),
            "candidate_trace_sha256": _sha(candidate_path),
            "maximum_nontiming_numeric_error": maximum_error,
            "torque_linf_error_nm": torque_error,
            "selected_command_linf_error_rad_s": selected_error,
            "mismatches": mismatches,
            "passed": not mismatches,
        })
    report = {
        "schema": "v6_2_b2_online_trace_nontiming_parity_v1",
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "all_passed": all(item["passed"] for item in records),
        "records": records,
    }
    (output_dir / "trace_parity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.reference_dir, args.candidate_dir, args.output_dir)
    print(json.dumps({"all_passed": report["all_passed"],
                      "records": [{key: item[key] for key in (
                          "scenario_id", "maximum_nontiming_numeric_error",
                          "torque_linf_error_nm", "mismatches")}
                          for item in report["records"]]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
