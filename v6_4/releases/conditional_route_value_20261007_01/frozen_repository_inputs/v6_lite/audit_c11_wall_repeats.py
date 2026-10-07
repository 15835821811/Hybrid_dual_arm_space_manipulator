"""Predeclare and retain three sequential full five-scene wall trials."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from v6_lite.c11_wall_qualification import qualify_directory
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_check(root, declared):
    comparison = source_provenance(declared, current_source_snapshot(root))
    comparison["tracked_worktree_clean"] = comparison["after"]["tracked_worktree_dirty"] is False
    comparison["passed"] = comparison["source_unchanged"] and comparison["tracked_worktree_clean"]
    return comparison


def run(output, scheduler_policy="high"):
    root = Path(__file__).resolve().parent.parent
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    declared_source = current_source_snapshot(root)
    commit = declared_source["git_commit"]
    if declared_source["capture_errors"] or declared_source["tracked_worktree_dirty"] is not False:
        raise ValueError("freeze available tracked source before formal trials: " +
                         str(declared_source["capture_errors"] or declared_source["status_porcelain"]))
    commands = [[sys.executable, "-m", "v6_lite.run_v6_lite", "--pcc-mode",
        "bounded_interval_pcc", "--enable-capsule-cbf", "--dispatch-clock-policy",
        "wall_deadline", "--scenario-count", "5", "--duration", "27",
        "--seed", "20260801", "--continue-failed-scenarios", "--output-dir",
        str(output / f"round_{number:02d}"), "--wall-scheduler-policy", scheduler_policy] for number in range(1, 4)]
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "source_snapshot": declared_source,
        "round_count": 3, "scenario_count_per_round": 5,
        "duration_per_scene_s": 27, "seed": 20260801, "commands": commands,
        "acceptance": {"all_15_full_scenes_required": True,
            "dispatch_p95_ms_max": 20, "acquisition_to_first_application_p95_ms_max": 20,
            "startup_first_cycle_retained_in_percentiles": True,
            "startup_scope": "separately validated unarmed simulation before predetermined task epoch",
            "original_physics_and_contract_required": True},
        "retention": "all declared rounds and rejected scenes retained; scenario/metric failures with fixed source still execute all three rounds; source changes fail the cohort and remaining rounds are NOT_STARTED_SOURCE_CHANGED; no selection or overwrite",
        "runtime_conditions": "Windows HIGH_PRIORITY_CLASS; disjoint physical P-core sets for planner/native actor; BLAS one thread; native no-GIL absolute 2 ms QPC grid; native full-state capture and model/payload SHA256 checks; planner directly releases two bounded native command slots; supervisor outside critical path; cyclic GC disabled during task",
        "hard_realtime_certified": False}
    plan["scheduler_policy"] = scheduler_policy
    plan["runtime_conditions"] = plan["runtime_conditions"].replace(
        "HIGH_PRIORITY_CLASS", "REALTIME_PRIORITY_CLASS" if scheduler_policy == "realtime" else "HIGH_PRIORITY_CLASS")
    write(output / "plan.json", plan)
    rounds, source_changed = [], False
    for number, command in enumerate(commands, 1):
        source_path = output/f"round_{number:02d}_source_provenance.json"
        source_record = {"before_round": source_check(root, declared_source)}
        write(source_path, source_record)
        source_changed |= not source_record["before_round"]["passed"]
        if source_changed:
            rounds.append({"round": number, "status": "NOT_STARTED_SOURCE_CHANGED",
                "reason": "SOURCE_IDENTITY_CHANGED_AFTER_DECLARATION", "passed": False,
                "complete_five_scenes": False, "source_provenance": source_path.name,
                "run_directory": (output/f"round_{number:02d}").as_posix()})
            write(output/"report.json", {"plan": "plan.json", "rounds": rounds,
                "complete": False, "all_declared_rounds_accounted_for": len(rounds) == 3,
                "passed": False, "physical_validation_pending": True,
                "hard_realtime_certified": False})
            continue
        print(f"Starting declared round {number}/3", flush=True)
        with (output / f"round_{number:02d}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        run_dir = output / f"round_{number:02d}"
        source_record["after_round"] = source_check(root, declared_source)
        write(source_path, source_record)
        source_changed |= not source_record["after_round"]["passed"]
        qualification = qualify_directory(run_dir, 5)
        write(output/f"round_{number:02d}_qualification.json", qualification)
        qualified_scenes = qualification.get("scenes", [])
        dispatch_passed = bool(qualified_scenes) and all(
            scene["checks"]["dispatch_p95"] for scene in qualified_scenes)
        application_passed = bool(qualified_scenes) and all(
            scene["checks"]["acquisition_to_first_application_p95"] for scene in qualified_scenes)
        rounds.append({"round": number, "exit_code": completed.returncode,
            "status": "FAILED_SOURCE_CHANGED" if source_changed else "EXECUTED",
            "source_provenance": source_path.name,
            "complete_five_scenes": qualification.get("complete_scene_horizons", False),
            "dispatch_p95_passed": dispatch_passed,
            "actual_application_p95_passed": application_passed,
            "passed": completed.returncode == 0 and qualification["passed"] and not source_changed,
            "qualification": f"round_{number:02d}_qualification.json",
            "run_directory": run_dir.as_posix(), "log": f"round_{number:02d}.log"})
        write(output / "report.json", {"plan": "plan.json", "rounds": rounds,
            "complete": len(rounds) == 3 and all("exit_code" in row for row in rounds),
            "all_declared_rounds_accounted_for": len(rounds) == 3,
            "passed": len(rounds) == 3 and all(x["passed"] for x in rounds),
            "physical_validation_pending": True, "hard_realtime_certified": False})
        print(f"Declared round {number} exit={completed.returncode}", flush=True)
    return all(x["passed"] for x in rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-scheduler-policy", choices=("high", "realtime"), default="high")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir, args.wall_scheduler_policy) else 1)
