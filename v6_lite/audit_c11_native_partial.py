"""Audit retained native records without converting a failed trial to a pass."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from v6_lite.audit_c11_wall_evidence import packet_replay, timeline_audit


def run(run_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    trial = json.loads((run_dir/"wall_trial_report.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir/"run_metadata.json").read_text(encoding="utf-8"))
    metrics = {"scenarios": trial["completed_scenes"], "run_config": metadata["run_config"]}
    with threadpool_limits(limits=1):
        timelines = timeline_audit(run_dir, metrics)
        replay = packet_replay(run_dir, metrics)
    failures = []
    for path in sorted((run_dir/"failures").glob("*_interval_failure.json")):
        failure = json.loads(path.read_text(encoding="utf-8"))
        trace = Path(failure["partial_trace"]["path"])
        with np.load(trace, allow_pickle=False) as native:
            count = len(native["applied_torques"])
            previous = float(native["timings"][-1, 3]) if count else None
            last_time = float(native["integration_states"][-1, 0])
        attempt = failure["rejected_servo_attempt"]
        failures.append({"scenario_id": failure["scenario_id"],
            "reason": failure["failure_reason"], "physical_steps": count,
            "physical_time_s": last_time, "previous_step_finished": previous,
            "rejected_attempt": attempt,
            "wake_offset_ms": (attempt["actual_start"]-attempt["scheduled"])*1e3,
            "no_next_physics_step": count == failure["physics_steps_executed"]
                and abs(last_time-count*.002) <= 1e-9,
            "failure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "native_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
    payload = {"schema": "c11_native_partial_audit_v1",
        "source_trial": run_dir.as_posix(), "source_commit": metadata["source"]["git_commit"],
        "completed_scene_count": len(metrics["scenarios"]), "retained_failures": failures,
        "complete_trial_passed": False, "whole_26_and_11_acceptance": "NOT_MET; incomplete five-scene trial",
        "repeated_native_wall_acceptance": "NOT_STARTED; five-scene qualification failed",
        "complete_scene_timelines": timelines, "complete_scene_native_packet_replay": replay,
        "evidence_consistent": timelines["passed"] and replay["passed"]
            and all(f["no_next_physics_step"] for f in failures),
        "safe_backup_established": False, "hard_realtime_certified": False}
    (output/"report.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Retained complete scenes:", len(metrics["scenarios"]),
          "evidence consistent:", payload["evidence_consistent"], flush=True)
    return payload["evidence_consistent"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.run_dir, args.output_dir) else 1)
