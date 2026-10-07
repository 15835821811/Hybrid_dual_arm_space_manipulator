"""Preserve and diagnose the five C.1 first wall-clock rejections."""
import hashlib
import json
from pathlib import Path


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(source.glob("*/timing/*.jsonl")):
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        row = rows[0]
        c = row["certificate"]
        failure_path = path.parent.parent / "failures" / f"{path.stem}_interval_failure.json"
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        diagnostic = failure["diagnostic"]
        rejection = row.get("later_dispatch_rejection", diagnostic)
        checks = row["servo_dispatch_checks"]
        phases = {p["name"]: p for p in row["phases"]}
        acquired = row["state_acquisition_monotonic_ns"] * 1e-9
        published = row["actual_dispatch_monotonic_ns"] * 1e-9
        # C.1 did not have a separate release or publication deadline.
        # Record the same-path release and expiry, and label that limitation.
        events = {
            "state_acquisition": acquired,
            "planning_release": acquired,
            "solve_start": phases["qp_assembly_and_solve"]["start_ns"] * 1e-9,
            "solve_end": c["solve_finished_time"],
            "validation_end": c["validation_finished_time"],
            "planned_execution_start": c["planned_execution_start"],
            "publish": published,
            "first_servo_application": checks[0]["dispatch_time"],
            "rejected_servo_substep": rejection["servo_substep"],
            "rejection": diagnostic["dispatch_time"],
            "expiry": c["valid_until"],
        }
        copy = output / path.name
        copy.write_bytes(raw)
        records.append({
            "scenario_id": path.stem,
            "source": path.as_posix(), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "preserved_timeline": copy.as_posix(), "events_monotonic_s": events,
            "publish_slack_s": c["valid_until"] - published,
            "application_start_offset_s": events["first_servo_application"] - c["planned_execution_start"],
            "remaining_execution_coverage_at_rejection_s": c["valid_until"] - events["rejection"],
            "evidence_limit": "planning release shares acquisition; C.1 publish deadline was the single valid_until; first dispatch recorded immediately before mj_step",
            "confirmed_mechanisms": [
                "expiry during the first ramp, after successful first dispatch",
                "acquisition+20ms covers serial planning and subsequent servo work together",
                "physics did not advance during planning; there was no concurrent next-segment planner",
            ],
            "not_established_by_this_short_trace": ["long-run phase drift", "steady-state scheduling tails"],
            "raw_record": row, "raw_failure": failure,
            "failure_source_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
        })
    if len(records) != 5:
        raise RuntimeError(f"expected five original timelines, found {len(records)}")
    (output / "first_rejection_audit.json").write_text(json.dumps({
        "schema": "c11_first_rejection_audit_v1", "clock": "perf_counter monotonic seconds",
        "records": records, "continuous_execution_passed": False,
    }, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for row in run(args.source, args.output):
        print(row["scenario_id"], row["publish_slack_s"], row["events_monotonic_s"]["rejected_servo_substep"])
