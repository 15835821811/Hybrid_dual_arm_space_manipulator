"""Reuse verified complete observations after a confirmed pause; no physics replay."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / ".git").exists() and (p / "model_test").is_dir())
sys.path.insert(0, str(ROOT))
from v6_lite import run_research_inertia_shadow as producer

SOURCE = ROOT / "v6_lite/output/runs/research_inertia_shadow_01"
OUTPUT = ROOT / "v6_lite/output/runs/research_inertia_shadow_resumed_01"
PIN = "a8543ecf3a62c7adef87e7a698a0536860ab4a8a1800ed95905c870eef256a0a"
HEAD = "19ad85c7db81a2e254fd176077b6aae39333a439"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(directory):
    return {p.relative_to(directory).as_posix(): {
                "sha256": producer.sha(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}


def validate_observation(row, plan):
    directory = SOURCE / "observations" / row["key"]
    report = read(directory / "observation.json")
    observer_plan = read(directory / "observer_plan.json")
    require(report["key"] == row["key"] and report["scenario_index"] == row["scenario_index"]
            and report["alpha"] == row["alpha"] and report["status"] == "OBSERVATION_COMPLETED",
            "observation identity mismatch")
    provenance = report["source_provenance"]
    require(provenance["source_unchanged"] and provenance["capture_complete"]
            and provenance["before"]["git_commit"] == HEAD
            and provenance["after"]["files"] == plan["current_source"]["files"],
            "old observation source mismatch")
    require(observer_plan["input_trajectory_sha256"] == row["trajectory_sha256"]
            and observer_plan["pair_policy_sha256"] == report["pair_policy_sha256"]
            and observer_plan["source"]["files"] == plan["current_source"]["files"],
            "observer input/source mismatch")
    pairs = observer_plan["pairs"]
    require(len(pairs) == 2927
            and len({(p["geom_a_id"], p["geom_b_id"]) for p in pairs}) == 2927
            and sum(p["pair_class"].endswith("_target") for p in pairs) == 75,
            "pair policy scope mismatch")
    for key, filename, count, pair_count, phase_step in (
            ("robot_target_500hz", "robot_target_500hz.npz", 13501, 75, 1.),
            ("whole_body_dense_discrete", "whole_body_dense.npz", 5401, 2927, 2.5)):
        geometry = report[key]
        require(producer.sha(directory / filename) == geometry["arrays_sha256"],
                "observation raw array hash mismatch")
        with np.load(directory / filename, allow_pickle=False) as arrays:
            minimum = arrays["minimum_m"]
            require(bool(arrays["complete"]) and int(arrays["completed_state_count"]) == count
                    and int(arrays["planned_state_count"]) == count
                    and int(arrays["completed_query_count"]) == count * pair_count,
                    "incomplete old geometry array")
            require(minimum.shape == (count,) and np.all(np.isfinite(minimum))
                    and np.allclose(arrays["physics_grid_phase"], np.arange(count) * phase_step,
                                    rtol=0., atol=1e-9), "old geometry state clock mismatch")
            require(geometry["checked_state_count"] == count
                    and geometry["pair_count"] == pair_count
                    and geometry["query_count"] == count * pair_count
                    and geometry["minimum_clearance_m"] == float(np.min(minimum))
                    and geometry["below_required_clearance_state_count"] == int(np.sum(minimum < .005))
                    and geometry["negative_native_signed_state_count"] == int(np.sum(minimum < 0.)),
                    "old geometry report differs from saved arrays")
    require(producer.sha(directory / "fresh_kinematics.npz") ==
            report["fresh_kinematics"]["arrays_sha256"], "old kinematics hash mismatch")
    with np.load(directory / "fresh_kinematics.npz", allow_pickle=False) as arrays:
        require(len(arrays["time"]) == 13501
                and all(np.all(np.isfinite(arrays[k])) for k in arrays.files),
                "incomplete/nonfinite old kinematics")
    return report


def main():
    require(not OUTPUT.exists(), "exclusive resume output already exists")
    require(not (SOURCE / "report.json").exists() and not (SOURCE / "manifest.json").exists(),
            "source is not the recorded interrupted observer")
    pause = read(SOURCE / "goal_pause_receipt.json")
    require(pause["observer_terminal_exit_code"] == 1
            and pause["physics_replay_complete"] and pause["physics_steps_completed"] == 202500,
            "pause was not a confirmed terminal observer interruption")
    plan = read(SOURCE / "plan.json")
    require(plan["current_source"]["git_commit"] == HEAD, "wrong frozen producer source")
    producer.identity_check(SOURCE, plan)
    require(producer.sha(SOURCE / "replay_manifest.json") == PIN, "replay pin mismatch")
    replay_manifest = read(SOURCE / "replay_manifest.json")
    require(len(replay_manifest) == 64, "replay manifest scope mismatch")
    for name, expected in replay_manifest.items():
        require(producer.sha(SOURCE / name) == expected, "replay file changed: " + name)
    replay = read(SOURCE / "replay_report.json")
    expected = [(i, a, producer.run_key(i, a)) for a in producer.ALPHAS for i in range(5)]
    require(replay["complete"] and replay["physics_steps_completed"] == 202500
            and [(r["scenario_index"], r["alpha"], r["key"]) for r in replay["runs"]] == expected,
            "replay qualification mismatch")
    require(all(r["full_27s_complete"] and r["status"] == "REPLAY_COMPLETED" for r in replay["runs"])
            and all(r["nominal_parity"]["passed"] for r in replay["runs"] if r["alpha"] == 1.),
            "full nominal-first replay qualification failed")
    inherited = {}
    for row in replay["runs"]:
        if (SOURCE / "observations" / row["key"] / "observation.json").exists():
            inherited[row["key"]] = validate_observation(row, plan)
    missing = [r["key"] for r in replay["runs"] if r["key"] not in inherited]
    require(len(inherited) == 14 and missing == ["scene_04_alpha_1.05"],
            "resume scope differs from recorded pause")
    source_inventory = inventory(SOURCE)
    OUTPUT.mkdir(exist_ok=False)
    try:
        shutil.copyfile(__file__, OUTPUT / "resume_observations.py")
        producer.write(OUTPUT / "interrupted_source_inventory.json", source_inventory)
        producer.write(OUTPUT / "resumption_plan.json", {
            "schema": "inertia_shadow_observation_resumption_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(SOURCE), "source_head": HEAD,
            "pinned_replay_manifest_sha256": PIN,
            "resume_script_sha256": producer.sha(__file__),
            "inherited_complete_keys": list(inherited), "new_observation_keys": missing,
            "physics_steps_replayed_by_resumption": 0,
            "interrupted_source_inventory_sha256": producer.sha(OUTPUT / "interrupted_source_inventory.json"),
            "limitations": producer.LIMITATIONS})
        names = set(replay_manifest) | {"replay_manifest.json", "test_qualification.json"}
        for key in inherited:
            names.update((SOURCE / "observations" / key / filename).relative_to(SOURCE).as_posix()
                         for filename in ("observer_plan.json", "robot_target_500hz.npz",
                                          "whole_body_dense.npz", "fresh_kinematics.npz", "observation.json"))
        for name in sorted(names):
            destination = OUTPUT / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, destination)
            require(producer.sha(destination) == source_inventory[name]["sha256"], "copied byte identity failed")
        shutil.copy2(SOURCE / "goal_pause_receipt.json", OUTPUT / "original_goal_pause_receipt.json")
        rows = []
        for row in replay["runs"]:
            rows.append(inherited[row["key"]] if row["key"] in inherited
                        else producer.observe_one(OUTPUT, plan, row))
        deltas = producer.paired_deltas(OUTPUT, plan["inputs"])
        _, _, provenance = producer.identity_check(OUTPUT, plan)
        require(inventory(SOURCE) == source_inventory, "interrupted source changed during resumption")
        report = {"schema": "inertia_shadow_observation_v1", "complete": len(rows) == 15,
                  "evidence_valid": len(rows) == 15 and provenance["source_unchanged"],
                  "status": "DIAGNOSTIC_COMPLETE", "runs": rows, "paired_vs_nominal": deltas,
                  "physics_steps_completed": 202500, "source_provenance": provenance,
                  "caller_pinned_replay_manifest_sha256": PIN,
                  "resumption": {"complete_observations_reused": 14, "new_observations_computed": 1,
                                 "physics_steps_recomputed": 0, "source_directory": str(SOURCE),
                                 "original_interrupted_source_unchanged": True,
                                 "original_observer_terminal_report_existed": False},
                  "limits_and_guard_status": "original runtime guards unchanged; shadow never submitted to them",
                  "limitations": producer.LIMITATIONS}
        manifest = {p.relative_to(OUTPUT).as_posix(): producer.sha(p) for p in sorted(OUTPUT.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"}
        raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        manifest["report.json"] = hashlib.sha256(raw).hexdigest()
        producer.write(OUTPUT / "manifest.json", manifest)
        producer.write(OUTPUT / "report.json", report)
        print(json.dumps({"status": report["status"], "complete": report["complete"],
                          "evidence_valid": report["evidence_valid"], "resumption": report["resumption"]}))
        return 0
    except BaseException as error:
        producer.write(OUTPUT / "resumption_failure.json", {
            "complete": False, "evidence_valid": False, "type": type(error).__name__,
            "message": str(error), "original_interrupted_source_retained": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
