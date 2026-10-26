"""Reproducible, resumable staged experiments for the frozen local task domain."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .environment import ROOT, Task, TrackingEnv
from .pipeline import run_episode, write_json
from .task_domain import CONFIG, DOMAIN, profile, profile_hash, sample_task

DEFAULT_OUTPUT = ROOT/"tracking_output/local10mm_v1"


def code_hash():
    h = hashlib.sha256()
    for name in ("environment.py", "trajectory.py", "pipeline.py", "task_domain.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def freeze(output):
    if CONFIG.exists():
        raise FileExistsError("Frozen domain already exists; do not silently redefine it")
    env = TrackingEnv(Task(20261012, 27., 2, "legacy-wide-v1"))
    value = {"domain": DOMAIN, "nominal_seed": 20261012,
        "model_runtime_contract_sha256": env.spec.runtime_contract_sha256(),
        "waypoints_m": env.path.waypoint_points_w.tolist(),
        "segment_durations_s": env.path.segment_durations_s.tolist(),
        "continuum_home": env.pose(0)[0].tolist(),
        "continuum_rotation": env.pose(0)[1].tolist(),
        "initial_qpos": env.data.qpos.tolist(), "initial_qvel": env.data.qvel.tolist(),
        "base_qpos": env.data.qpos[env.bp].tolist(), "base_qvel": env.data.qvel[env.bv].tolist(),
        "low_level_qpos": env.data.qpos[env.qids].tolist(),
        "target_shift": env.target_shift.tolist(), "target_velocity": env.target_velocity.tolist(),
        "target_omega_local": env.target_omega_local.tolist(),
        "grasp_local": env.grasp_local.tolist(), "approach_offset": env.approach_offset.tolist(),
        "obstacles": [{"name": o.name, "center": o.center.tolist(), "radius": o.radius} for o in env.obstacles]}
    write_json(CONFIG, value)
    tasks = [Task(20261012, 27., 2, DOMAIN, True)] + [
        Task(31000000+i*104729, 27., 2, DOMAIN, False) for i in range(19)]
    manifest = {"domain": DOMAIN, "profile_sha256": profile_hash(),
        "purpose": "development only; not held-out final test", "sampling_seeds": [73, 137, 211],
        "tasks": [{"id": f"pilot_{i:02d}", "task": asdict(t),
                   "parameters": sample_task(t.seed, t.nominal)} for i,t in enumerate(tasks)]}
    write_json(output/"pilot_manifest.json", manifest)
    print(f"Frozen profile: {CONFIG}\nManifest: {output/'pilot_manifest.json'}", flush=True)


def verify_domain(output):
    manifest = json.loads((output/"pilot_manifest.json").read_text(encoding="utf-8"))
    checks = []
    for entry in manifest["tasks"]:
        t = Task(**entry["task"])
        a, b = sample_task(t.seed, t.nominal), sample_task(t.seed, t.nominal)
        assert a == b == entry["parameters"]
        assert np.max(np.abs(a["curve_delta_m"])) <= .010
        for stream, changed in (("curve", ["curve_delta_m"]),
                                ("target_position", ["target_shift"]),
                                ("target_motion", ["target_velocity", "target_omega_local"])):
            x = sample_task(t.seed, False)
            y = sample_task(t.seed, False, {stream: t.seed+1})
            for key in ("curve_delta_m", "target_shift", "target_velocity", "target_omega_local"):
                assert (x[key] == y[key]) == (key not in changed)
        checks.append(entry["id"])
    p = profile()
    initial_checks = []
    for entry in manifest["tasks"][:3]:
        env = TrackingEnv(Task(**entry["task"]))
        np.testing.assert_array_equal(env.data.qpos[env.bp], p["base_qpos"])
        np.testing.assert_array_equal(env.data.qvel[env.bv], p["base_qvel"])
        np.testing.assert_array_equal(env.data.qpos[env.qids], p["low_level_qpos"])
        np.testing.assert_array_equal(env.data.qvel[env.vids], 0.)
        np.testing.assert_array_equal(env.path.segment_durations_s, p["segment_durations_s"])
        for o, expected in zip(env.obstacles, p["obstacles"]):
            np.testing.assert_array_equal(o.center, expected["center"])
            assert o.radius == expected["radius"]
        initial_checks.append(entry["id"])
    write_json(output/"domain_checks.json", {"passed": True, "sampling_tasks": checks,
        "physics_initializations": initial_checks, "profile_sha256": profile_hash(),
        "checks": ["same-seed reproducibility", "independent streams", "10mm bounds",
                   "fixed initial robot/base state", "fixed obstacles", "fixed segment times"]})
    print("Domain checks passed", flush=True)


def teachers(output, name, count):
    manifest = json.loads((output/"pilot_manifest.json").read_text(encoding="utf-8"))
    if manifest["profile_sha256"] != profile_hash():
        raise ValueError("Manifest profile mismatch")
    folder = output/name
    version_file = folder/"code_version.json"
    version = code_hash()
    if version_file.exists():
        if json.loads(version_file.read_text())["sha256"] != version:
            raise ValueError("Code changed; use a new experiment name instead of mixing runs")
    else:
        write_json(version_file, {"sha256": version})
        snapshots = folder/"source_snapshot"
        snapshots.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob("*.py"):
            (snapshots/source.name).write_bytes(source.read_bytes())
        (snapshots/CONFIG.name).write_bytes(CONFIG.read_bytes())
    results = []
    for entry in manifest["tasks"][:count]:
        run = folder/entry["id"]
        if (run/"result.json").exists():
            result = json.loads((run/"result.json").read_text(encoding="utf-8"))
        else:
            result, _, _ = run_episode(Task(**entry["task"]), run, progress=True)
        results.append({"id": entry["id"], "accepted": result["accepted_for_teacher_dataset"],
                        "failure": result["failure"], "metrics": result["metrics"],
                        "execution": result["execution"]})
        write_json(folder/"summary.json", {"attempted": len(results),
            "accepted": sum(r["accepted"] for r in results), "episodes": results,
            "profile_sha256": profile_hash(), "code_sha256": version})
        # First five tasks are a gate: do not silently advance on a failed pilot.
        if len(results) <= 5 and not result["accepted_for_teacher_dataset"]:
            raise RuntimeError(f"M0 gate failed: {entry['id']}; inspect saved metrics before continuing")
    print(f"Teacher gate: {sum(r['accepted'] for r in results)}/{len(results)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify-domain", "teachers"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="teacher_a01")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "verify-domain":
        verify_domain(args.output)
    elif args.stage == "teachers":
        if not 1 <= args.count <= 20:
            parser.error("count must be 1..20")
        teachers(args.output, args.name, args.count)


if __name__ == "__main__":
    main()
