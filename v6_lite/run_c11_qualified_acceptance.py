"""Predeclare single -> five -> all three repeats -> independent acceptance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def run(output, policy):
    root = Path(__file__).resolve().parent.parent
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root).decode().splitlines()
    if dirty:
        raise ValueError(f"freeze tracked source before acceptance: {dirty}")
    prefix = [sys.executable, "-m"]
    commands = [
        ("test_profiles", prefix+["v6_lite.run_test_profiles", "--profile", "all",
            "--output-dir", str(output/"test_profiles")]),
        ("single", prefix+["v6_lite.run_c11_wall_single", "--wall-scheduler-policy", policy,
            "--output-dir", str(output/"single")]),
        ("five", prefix+["v6_lite.run_v6_lite", "--pcc-mode", "bounded_interval_pcc",
            "--enable-capsule-cbf", "--continue-failed-scenarios", "--wall-scheduler-policy", policy,
            "--output-dir", str(output/"five")]),
        ("three_by_five", prefix+["v6_lite.audit_c11_wall_repeats",
            "--wall-scheduler-policy", policy, "--output-dir", str(output/"three_by_five")]),
        ("independent_acceptance", prefix+["v6_lite.audit_c11_wall_evidence", "--run-dir",
            str(output/"five"), *(str(output/"three_by_five"/f"round_{n:02d}") for n in range(1,4)),
            "--output-dir", str(output/"independent_acceptance")]),
    ]
    plan = {"schema": "c11_qualified_acceptance_plan_v1", "source_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(), "scheduler_policy": policy,
        "steps": [{"name": name, "command": command} for name, command in commands],
        "qualification": "single and five must pass before all three repeats are attempted",
        "retention": "unique output; retain each failure; all predeclared repeat rounds attempted",
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
            print(f"Acceptance step: {name}", flush=True)
            with (output/f"{name}.log").open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            passed = result.returncode == 0
            records.append({"name": name, "status": "PASSED" if passed else "FAILED",
                "exit_code": result.returncode})
        write(output/"report.json", {"plan": "plan.json", "steps": records,
            "passed": passed and len(records) == len(commands), "hard_realtime_certified": False})
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-scheduler-policy", choices=("high", "realtime"), default="high")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir, args.wall_scheduler_policy) else 1)
