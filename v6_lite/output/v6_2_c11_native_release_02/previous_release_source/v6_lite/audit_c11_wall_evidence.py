"""Run original acceptance plus independent wall timeline and packet replay."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np
from threadpoolctl import threadpool_limits
from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.validate_v6_lite import validate_delivery, validate_execution_contract, _obstacles
from v6_lite.audit_b2_online_interval_recompute import run as recompute


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, value, kind)
    return value


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def timeline_audit(run_dir, metrics):
    records = []
    for scene in metrics["scenarios"]:
        scene_id = scene["scenario"]["scenario_id"]
        path = run_dir / "timing" / f"{scene_id}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        errors = []
        if len(rows) != 1350:
            errors.append("INCOMPLETE_PLANNING_SEGMENTS")
        previous_end = None
        for i, row in enumerate(rows):
            c, phases = row["certificate"], row["phases"]
            if c["command_id"] != i or c["predecessor_command_id"] != i - 1:
                errors.append(f"{i}:COMMAND_ORDER")
            times = [c[k] for k in ("source_acquisition_time", "planning_release", "solve_started",
                "solve_finished", "validation_finished")]
            times.extend([row["actual_publish_monotonic_s"], c["publish_deadline"], c["execution_start"]])
            if times != sorted(times) or abs(c["execution_end"] - c["execution_start"] - .020) > 1e-9:
                errors.append(f"{i}:CLOCK_OR_COVERAGE")
            if previous_end is not None and abs(previous_end - c["execution_start"]) > 1e-9:
                errors.append(f"{i}:NONCONTIGUOUS_SEGMENTS")
            previous_end = c["execution_end"]
            if not all(a["end_ns"] == b["start_ns"] for a, b in zip(phases, phases[1:])):
                errors.append(f"{i}:PHASE_OVERLAP_OR_GAP")
            if abs(sum(p["wall_s"] for p in phases) - row["dispatch_latency_s"]) > 1e-6:
                errors.append(f"{i}:PHASE_COVERAGE")
            if not row["publication_check"]["accepted"] or not row["handoff_check"]["accepted"]:
                errors.append(f"{i}:REJECTED_SEGMENT")
            if len(row["servo_dispatch_checks"]) != 10 or not all(x["accepted"] for x in row["servo_dispatch_checks"]):
                errors.append(f"{i}:INCOMPLETE_SERVO_CHECKS")
            applied = row["first_servo_application_monotonic_s"]
            if not c["execution_start"] <= applied < c["execution_start"] + .002:
                errors.append(f"{i}:ACTUAL_APPLICATION_WINDOW")
            if c["controller_config_hash"] != scene["execution_contract"]["controller_config_hash"]:
                errors.append(f"{i}:CONFIG_IDENTITY")
            if "acquisition_to_first_application" in scene["metrics"]["rates_and_latency"]:
                mid_source = scene["execution_contract"].get("planning_source_phase") == "after_first_native_microstep"
                expected_source = max(i - 1, 0) * .020 + (.002 if i and mid_source else 0)
                if abs(row["source_simulation_time_s"] - expected_source) > 1e-9:
                    errors.append(f"{i}:OBSERVED_SOURCE_TIME")
        records.append({"scenario_id": scene_id, "segments": len(rows),
            "timeline_sha256": sha(path), "errors": errors, "passed": not errors})
    return {"schema": "c11_independent_wall_timeline_audit_v1", "passed": all(r["passed"] for r in records),
        "records": records, "clock_serialization_tolerance_s": 1e-6,
        "control_admission_tolerance_modified": False}


def packet_replay(run_dir, metrics):
    records = []
    robot = default_v6_lite_robot_spec()
    cfg = metrics["run_config"]
    for scene in metrics["scenarios"]:
        scene_id = scene["scenario"]["scenario_id"]
        verifier = WholeBodyCollisionVerifier(robot, _obstacles(scene["scenario"]),
            WholeBodyVerificationConfig(minimum_clearance=cfg["whole_body_minimum_clearance_m"],
                query_distance_max=2.5, adaptive_subdivisions=cfg["verification_subdivisions"],
                self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
        model = verifier.model
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data = mujoco.MjData(model)
        rows = [json.loads(line) for line in (run_dir / "timing" / f"{scene_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        native_info = scene["execution_contract"].get("native_raw_trace")
        native = np.load(Path(native_info["path"]), allow_pickle=False) if native_info else None
        native_states = native["integration_states"] if native is not None else None
        native_errors = []
        if native_info and sha(Path(native_info["path"])) != native_info["sha256"]:
            native_errors.append("NATIVE_RAW_TRACE_HASH")
        with np.load(Path(scene["trace"]["path"]), allow_pickle=False) as trace:
            data.qpos[:] = trace["initial_qpos"]
            data.qvel[:] = trace["initial_qvel"]
            mujoco.mj_forward(model, data)
            torques, selected, reference = trace["torque"], trace["task_selected_command"], trace["reference_q"]
            if native is not None:
                if not np.array_equal(torques, native["applied_torques"]):
                    native_errors.append("NATIVE_APPLIED_TORQUES_DIFFER_FROM_OBSERVER_LOG")
                times = native["timings"]
                scheduled = float(native["epoch"]) + np.arange(len(torques))*.002
                if (len(times) != len(torques) or len(native_states) != len(torques)+1
                        or not np.all(np.isfinite(times))
                        or not np.all(np.abs(times[:, 0]-scheduled) <= 1e-9)
                        or not np.all((times[:, 1] >= scheduled) & (times[:, 1] < scheduled+.002))
                        or not np.all((times[:, 2] >= times[:, 1]) & (times[:, 2] < scheduled+.002))
                        or not np.all(times[:, 3] >= times[:, 2])):
                    native_errors.append("NATIVE_ALL_STEP_CLOCK_CONTRACT")
                if not np.array_equal(state(model, data), native_states[0]):
                    native_errors.append("NATIVE_FRESH_INITIAL_STATE")
            boundary_ids, mid_ids, errors = [], [], []
            for tick, row in enumerate(rows):
                states = [state(model, data)]
                boundary_ids.append(array_sha(states[0]))
                c = row["certificate"]
                mid_source = scene["execution_contract"].get("planning_source_phase") == "after_first_native_microstep"
                expected_source = mid_ids[tick - 1] if tick and mid_source else boundary_ids[max(tick - 1, 0)]
                if c["source_state_id"] != expected_source:
                    errors.append(f"{tick}:ACTUAL_SOURCE_FULL_STATE_HASH")
                if c["predicted_start_state_id"] != boundary_ids[-1]:
                    errors.append(f"{tick}:PREDICTED_START_FULL_STATE_HASH")
                for step in range(tick * 10, tick * 10 + 10):
                    data.ctrl[:] = torques[step]
                    mujoco.mj_step(model, data)
                    states.append(state(model, data))
                    if native is not None and not np.array_equal(states[-1], native_states[step+1]):
                        native_errors.append(f"{step}:NATIVE_CAPTURED_FULL_STATE")
                mid_ids.append(array_sha(states[1]))
                digest = hashlib.sha256()
                for value in (states, torques[tick * 10:(tick + 1) * 10], selected[tick], reference[(tick + 1) * 10 - 1]):
                    digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
                if c["payload_sha256"] != digest.hexdigest():
                    errors.append(f"{tick}:NATIVE_REPLAY_PACKET_PAYLOAD_HASH")
        if native is not None:
            native.close()
        records.append({"scenario_id": scene_id, "segments": len(rows), "native_steps": len(torques),
            "full_state_hash_mismatches": len(errors), "errors": errors,
            "raw_native_trace_available": native_info is not None,
            "raw_native_trace_errors": native_errors, "passed": not errors and not native_errors})
    return {"schema": "c11_independent_native_packet_replay_v1", "passed": all(x["passed"] for x in records),
        "scope": "fresh MjData, original saved torques, full integration source/predicted/payload hashes",
        "records": records}


def run(run_dirs, output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with threadpool_limits(limits=1):
        for number, run_dir in enumerate(run_dirs, 1):
            print(f"Verifying {run_dir}", flush=True)
            item = {"run_dir": run_dir.as_posix(), "reports": {}, "passed": True}
            metrics = json.loads((run_dir / "v6_lite_metrics.json").read_text(encoding="utf-8"))
            for name, action in (
                ("native_validation", lambda: validate_delivery(Path("v6_lite"), run_dir)),
                ("execution_validation", lambda: validate_execution_contract(run_dir)),
                ("timeline_validation", lambda: timeline_audit(run_dir, metrics)),
                ("packet_replay", lambda: packet_replay(run_dir, metrics)),
                ("interval_validation", lambda: recompute(run_dir, run_dir / "c11_independent_recompute"))):
                try:
                    result = action()
                    item["reports"][name] = result
                    item["passed"] &= result["passed"]
                    print(name, result["passed"], flush=True)
                except Exception as error:
                    item["passed"] = False
                    item["reports"][name] = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
            (output / f"run_{number:02d}.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
            records.append(item)
            (output / "summary.json").write_text(json.dumps({"passed": all(x["passed"] for x in records),
                "records": records}, indent=2) + "\n", encoding="utf-8")
    return all(x["passed"] for x in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.run_dir, args.output_dir) else 1)
