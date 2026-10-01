"""Predeclare and retain three sequential full five-scene wall trials."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(output):
    root = Path(__file__).resolve().parent.parent
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root).decode().splitlines()
    if dirty:
        raise ValueError(f"freeze tracked source before formal trials: {dirty}")
    commands = [[sys.executable, "-m", "v6_lite.run_v6_lite", "--pcc-mode",
        "bounded_interval_pcc", "--enable-capsule-cbf", "--dispatch-clock-policy",
        "wall_deadline", "--scenario-count", "5", "--duration", "27",
        "--seed", "20260801", "--continue-failed-scenarios", "--output-dir",
        str(output / f"round_{number:02d}")] for number in range(1, 4)]
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "round_count": 3, "scenario_count_per_round": 5,
        "duration_per_scene_s": 27, "seed": 20260801, "commands": commands,
        "acceptance": {"all_15_full_scenes_required": True,
            "dispatch_p95_ms_max": 20, "acquisition_to_first_application_p95_ms_max": 20,
            "startup_first_cycle_retained_in_percentiles": True,
            "startup_scope": "separately validated unarmed simulation before predetermined task epoch",
            "original_physics_and_contract_required": True},
        "retention": "all declared rounds and all rejected scenes retained; no selection or overwrite",
        "runtime_conditions": "Windows HIGH_PRIORITY_CLASS; disjoint physical P-core sets for planner/native actor; BLAS one thread; native no-GIL absolute 2 ms QPC grid; native full-state capture and model/payload SHA256 checks; planner directly releases two bounded native command slots; supervisor outside critical path; cyclic GC disabled during task",
        "hard_realtime_certified": False}
    write(output / "plan.json", plan)
    rounds = []
    for number, command in enumerate(commands, 1):
        print(f"Starting declared round {number}/3", flush=True)
        with (output / f"round_{number:02d}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        run_dir = output / f"round_{number:02d}"
        metrics_path = run_dir / "v6_lite_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        dispatch_passed = metrics is not None and all(
            scene["metrics"]["rates_and_latency"]["dispatch"]["p95_ms"] <= 20
            for scene in metrics["scenarios"])
        application_passed = metrics is not None and all(
            scene["metrics"]["rates_and_latency"]["acquisition_to_first_application"]["p95_ms"] <= 20
            for scene in metrics["scenarios"])
        rounds.append({"round": number, "exit_code": completed.returncode,
            "complete_five_scenes": metrics is not None and len(metrics["scenarios"]) == 5,
            "dispatch_p95_passed": dispatch_passed,
            "actual_application_p95_passed": application_passed,
            "passed": completed.returncode == 0 and metrics is not None and metrics["passed"] and dispatch_passed and application_passed,
            "run_directory": run_dir.as_posix(), "log": f"round_{number:02d}.log"})
        write(output / "report.json", {"plan": "plan.json", "rounds": rounds,
            "complete": len(rounds) == 3, "passed": len(rounds) == 3 and all(x["passed"] for x in rounds),
            "physical_validation_pending": True, "hard_realtime_certified": False})
        print(f"Declared round {number} exit={completed.returncode}", flush=True)
    return all(x["passed"] for x in rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir) else 1)
