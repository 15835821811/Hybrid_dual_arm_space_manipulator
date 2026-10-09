"""Audit retained native records without converting a failed trial to a pass."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from v6_lite.audit_c11_wall_evidence import auditor_source_identity, packet_replay, timeline_audit


def native_failure_audit(path):
    """Check the stopped live state, not only the recorder's step counter."""
    failure = json.loads(path.read_text(encoding="utf-8"))
    trace = Path(failure["partial_trace"]["path"])
    trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    with np.load(trace, allow_pickle=False) as native:
        count = len(native["applied_torques"])
        timings, states = native["timings"], native["integration_states"]
        previous = float(timings[-1, 3]) if count else None
        last_time = float(states[-1, 0])
        final = native["final_actual_integration_state"] if "final_actual_integration_state" in native else None
        final_matches = bool(final is not None and np.all(np.isfinite(final))
            and np.array_equal(final, states[-1]))
        actual_time = float(final[0]) if final is not None and final.size else None
        complete_capture = len(states) == count+1 and len(timings) == count
    attempt = failure["rejected_servo_attempt"]
    hash_matches = trace_hash == failure["partial_trace"]["sha256"]
    started, scheduled = attempt.get("actual_start"), attempt.get("scheduled")
    attempt_timing_available = (isinstance(started, (float, int))
        and isinstance(scheduled, (float, int))
        and np.isfinite(started) and np.isfinite(scheduled))
    consumed = attempt.get("torque_consumed")
    not_attempted = attempt.get("not_attempted") is True or attempt.get("attempted") is False
    attempted_index = attempt.get("physics_step")
    if not_attempted:
        attempt_count_consistent = attempted_index is None and consumed is False
    else:
        attempt_count_consistent = ((consumed is True and count > 0 and attempted_index == count-1)
            or (consumed is False and attempted_index == count))
    return {"scenario_id": failure["scenario_id"],
        "reason": failure["failure_reason"], "physical_steps": count,
        "physical_time_s": last_time, "previous_step_finished": previous,
        "final_actual_physics_time_s": actual_time,
        "final_actual_matches_last_capture": final_matches,
        "state_capture_complete": complete_capture,
        "native_trace_hash_matches_failure": hash_matches,
        "rejected_attempt": attempt,
        "rejected_operation_consumed": consumed is True,
        "rejected_operation_attempted": not not_attempted,
        "attempted_step_count_consistent": bool(attempt_count_consistent),
        "rejected_attempt_timing_available": bool(attempt_timing_available),
        "wake_offset_ms": (started-scheduled)*1e3 if attempt_timing_available else None,
        "no_next_physics_step": bool(complete_capture and final_matches and hash_matches
            and count == failure["physics_steps_executed"]
            and abs(last_time-count*.002) <= 1e-9
            and failure["next_servo_step_executed"] is False
            and attempt_count_consistent),
        "failure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_trace_sha256": trace_hash}


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
        failures.append(native_failure_audit(path))
    payload = {"schema": "c11_native_partial_audit_v1",
        "source_trial": run_dir.as_posix(), "source_commit": metadata["source"]["git_commit"],
        "source_trial_commit": metadata["source"]["git_commit"], **auditor_source_identity(),
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
