"""Predeclare single -> five -> all three repeats -> independent acceptance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from v6_lite.c11_wall_qualification import qualify_directory
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def source_check(root, declared):
    comparison = source_provenance(declared, current_source_snapshot(root))
    comparison["tracked_worktree_clean"] = comparison["after"]["tracked_worktree_dirty"] is False
    comparison["passed"] = comparison["source_unchanged"] and comparison["tracked_worktree_clean"]
    return comparison


def run(output, policy):
    root = Path(__file__).resolve().parent.parent
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    declared_source = current_source_snapshot(root)
    commit = declared_source["git_commit"]
    if declared_source["capture_errors"] or declared_source["tracked_worktree_dirty"] is not False:
        raise ValueError("freeze available tracked source before acceptance: " +
                         str(declared_source["capture_errors"] or declared_source["status_porcelain"]))
    prefix = [sys.executable, "-m"]
    commands = [
        ("test_profiles", prefix+["v6_lite.run_test_profiles", "--profile", "all",
            "--output-dir", str(output/"test_profiles")]),
        ("single", prefix+["v6_lite.run_c11_wall_single", "--wall-scheduler-policy", policy,
            "--output-dir", str(output/"single")]),
        ("five", prefix+["v6_lite.run_v6_lite", "--pcc-mode", "bounded_interval_pcc",
            "--enable-capsule-cbf", "--continue-failed-scenarios", "--dispatch-clock-policy", "wall_deadline",
            "--scenario-count", "5", "--duration", "27", "--seed", "20260801", "--wall-scheduler-policy", policy,
            "--output-dir", str(output/"five")]),
        ("three_by_five", prefix+["v6_lite.audit_c11_wall_repeats",
            "--wall-scheduler-policy", policy, "--output-dir", str(output/"three_by_five")]),
        ("independent_acceptance", prefix+["v6_lite.audit_c11_wall_evidence", "--run-dir",
            str(output/"five"), *(str(output/"three_by_five"/f"round_{n:02d}") for n in range(1,4)),
            "--output-dir", str(output/"independent_acceptance")]),
    ]
    plan = {"schema": "c11_qualified_acceptance_plan_v1", "source_commit": commit,
        "source_snapshot": declared_source,
        "created_utc": datetime.now(timezone.utc).isoformat(), "scheduler_policy": policy,
        "steps": [{"name": name, "command": command} for name, command in commands],
        "qualification": {"single_and_five_full_horizons_required": True,
            "dispatch_p95_ms_max": 20, "acquisition_to_first_application_p95_ms_max": 20,
            "startup_retained_in_percentiles": True, "physics_steps_per_scene": 13500,
            "planning_ticks_per_scene": 1350},
        "retention": "unique output; retain each failure; after prerequisites pass, execute all three declared rounds despite scenario/metric failures while source stays fixed; source changes fail the cohort and remaining rounds are NOT_STARTED_SOURCE_CHANGED",
        "hardware_and_global_OS_changes": False, "hard_realtime_certified": False,
        "required_privilege_when_realtime": "existing SeIncreaseBasePriorityPrivilege in the invoking token",
        "system_priority_settings_persist_after_run": False}
    write(output/"plan.json", plan)
    from v6_lite.runtime_scheduler_environment import ThreadScheduling
    scheduling = ThreadScheduling("supervisor", policy)
    condition = dict(scheduling.record)
    scheduling.restore()
    available = (condition.get("applied") and condition.get("process_priority_set")
        and condition.get("actual_process_priority_class") == (0x100 if policy == "realtime" else 0x80))
    write(output/"scheduler_preflight.json", {"available": bool(available),
        "condition": condition, "settings_restored": True})
    if not available:
        write(output/"report.json", {"plan": "plan.json", "passed": False,
            "reason": "DECLARED_SCHEDULER_CONDITIONS_UNAVAILABLE",
            "steps": [{"name": name, "status": "NOT_STARTED_FAILED_PREFLIGHT"} for name, _ in commands],
            "hard_realtime_certified": False})
        print("Declared scheduler conditions unavailable; no physics started.", flush=True)
        return False
    records, passed = [], True
    for name, command in commands:
        if not passed:
            records.append({"name": name, "status": "NOT_STARTED_FAILED_PREREQUISITE"})
        else:
            source_path = output/f"{name}_source_provenance.json"
            source_record = {"before_step": source_check(root, declared_source)}
            write(source_path, source_record)
            if not source_record["before_step"]["passed"]:
                passed = False
                records.append({"name": name, "status": "NOT_STARTED_SOURCE_CHANGED",
                    "reason": "SOURCE_IDENTITY_CHANGED_AFTER_DECLARATION",
                    "source_provenance": source_path.name})
                write(output/"report.json", {"plan": "plan.json", "steps": records,
                    "passed": False, "hard_realtime_certified": False})
                continue
            print(f"Acceptance step: {name}", flush=True)
            with (output/f"{name}.log").open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            source_record["after_step"] = source_check(root, declared_source)
            write(source_path, source_record)
            passed = result.returncode == 0 and source_record["after_step"]["passed"]
            record = {"name": name, "exit_code": result.returncode,
                      "source_provenance": source_path.name}
            if not source_record["after_step"]["passed"]:
                record["reason"] = "SOURCE_IDENTITY_CHANGED_AFTER_DECLARATION"
            if name in ("single", "five"):
                qualification = qualify_directory(output/name, 1 if name == "single" else 5)
                write(output/f"{name}_qualification.json", qualification)
                passed &= qualification["passed"]
                record["qualification"] = f"{name}_qualification.json"
            record["status"] = "PASSED" if passed else "FAILED"
            records.append(record)
        write(output/"report.json", {"plan": "plan.json", "steps": records,
            "passed": passed and len(records) == len(commands), "hard_realtime_certified": False})
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-scheduler-policy", choices=("high", "realtime"), default="high")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir, args.wall_scheduler_policy) else 1)
