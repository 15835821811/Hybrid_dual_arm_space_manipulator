"""Build the portable B.2 viewer from already sealed results; no simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "v6_4/output/task_anchored_residual_20261007_01"
DEFAULT_OUTPUT = ROOT / "v6_4/visualization/task_anchored_residual_20261007_01"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("visualization output must be separate from sealed evidence")
    analysis = json.loads((source / "delivery_analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((source / "verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS" or sha(source / "manifest.json") != verification["manifest_sha256"]:
        raise ValueError("sealed evidence identity does not match original verification")
    fields = (
        "slot_id", "task_id", "method", "status", "full_task_success", "actual_steps",
        "actual_path_length_m", "actual_joint_path_length_rad", "whole_body_minimum_m",
        "minimum_by_class_m", "command_intervention_rad_s", "consumed_QP_offset_peak_m",
        "nonzero_reference_consumed", "requirements", "runtime_reported_passed",
    )
    data = {
        "schema": "v64_b2_portable_dashboard_v1",
        "verdict": summary["verdict"],
        "TEST_rows": [{key: row.get(key) for key in fields} for row in analysis["TEST_rows"]],
        "cost": analysis["cost"],
        "paired_completed_task_quality": analysis["paired_completed_task_quality"],
        "provenance": {
            "branch": "v6.4-b2-task-anchored-residual",
            "source_producer_commit": analysis["source_producer"],
            "sealed_manifest_sha256": verification["manifest_sha256"],
            "source_delivery_analysis_sha256": sha(source / "delivery_analysis.json"),
            "complete_evidence_is_in_git": False,
            "source_manifest": "../../releases/task_anchored_residual_20261007_01/snapshot/manifest.json",
            "geometry_scope": "robot-target native500Hz; whole-body50Hz + configuration subdivisions4",
            "historical_nonzero_full_curve_runtime_gate": "FAILED_ALL_30_completed_nonzero_attempts",
            "new_physics_steps": 0,
            "new_optimizer_updates": 0,
            "new_samples": 0,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dashboard_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name in ("index.html", "style.css", "app.js"):
        shutil.copyfile(Path(__file__).parent / "web" / name, output / name)
    return {"output": str(output), "TEST_rows": len(data["TEST_rows"]), "new_physics_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
