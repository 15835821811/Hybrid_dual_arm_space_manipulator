"""Run original acceptance plus independent wall timeline and packet replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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


def auditor_source_identity():
    directory = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory.parent).decode().strip()
    return {"auditor_source_commit": commit, "auditor_source_raw_sha256": {
        f"v6_lite/{name}": sha(directory/name)
        for name in ("audit_c11_wall_evidence.py", "audit_c11_native_partial.py")}}


def state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, value, kind)
    return value


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite_values(values):
    try:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))
    except (TypeError, ValueError):
        return False


def _clock_summary(samples, deadline_s, *, steady=True):
    """Recompute statistics from saved clocks, without the runtime reporter."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "p95_ms": None, "passed": False}
    late = values > deadline_s
    longest = current = 0
    for missed in late:
        current = current + 1 if missed else 0
        longest = max(longest, current)
    return {
        "count": int(values.size),
        "p50_ms": float(np.median(values)*1000),
        "p95_ms": float(np.percentile(values, 95)*1000),
        "p99_ms": float(np.percentile(values, 99)*1000),
        "max_ms": float(np.max(values)*1000),
        "first_cycle_ms": float(values[0]*1000),
        "over_deadline_count": int(np.count_nonzero(late)),
        "longest_consecutive_over_deadline": int(longest),
        "passed": bool(np.percentile(values, 95) <= deadline_s),
        "steady_after_first_cycle": _clock_summary(values[1:], deadline_s, steady=False)
            if steady and values.size > 1 else None,
    }


def _compare_clock_summary(expected, reported, prefix, errors):
    if not isinstance(reported, dict):
        errors.append(f"{prefix}:MISSING_REPORTED_CLOCK_STATISTICS")
        return
    for key, value in expected.items():
        actual = reported.get(key)
        if isinstance(value, dict):
            _compare_clock_summary(value, actual, f"{prefix}.{key}", errors)
        elif key.endswith("_ms") and value is not None:
            if (not isinstance(actual, (float, int)) or not np.isfinite(actual)
                    or abs(actual-value) > 1e-6):
                errors.append(f"{prefix}.{key}:REPORTED_CLOCK_STATISTIC_MISMATCH")
        elif actual != value:
            errors.append(f"{prefix}.{key}:REPORTED_CLOCK_STATISTIC_MISMATCH")


def native_clock_audit(rows, native, reported_rates):
    """Bind JSON evidence to captured QPC times and independently recompute it."""
    errors = []
    result = {
        "schema": "c11_independent_native_clock_audit_v1",
        "source_and_application_timestamp_tolerance_s": 0,
        "reported_statistic_tolerance_ms": 1e-6,
        "startup_retained_in_all_sample_statistics": True,
        "statistics": {}, "errors": errors, "passed": False,
        "clock_sources": {
            "source": "native acquired for startup; native source_times afterwards",
            "actual_application": "native timings[:, 2], after all 67 ctrl assignments",
            "publication": "saved actual_publish_monotonic_s from atomic slot release",
            "servo": "native timings: scheduled, started, consumed, finished",
        },
    }
    required = ("acquired", "epoch", "source_times", "timings")
    if any(key not in native for key in required):
        errors.append("MISSING_NATIVE_CLOCK_FIELDS")
        return result
    acquired, epoch = float(native["acquired"]), float(native["epoch"])
    source_times, timings = np.asarray(native["source_times"]), np.asarray(native["timings"])
    count = len(rows)
    if (not count or timings.shape != (count*10, 4)
            or source_times.shape != (count,)
            or not np.isfinite(acquired) or not np.isfinite(epoch)
            or acquired > epoch or not np.all(np.isfinite(source_times))
            or not np.all(np.isfinite(timings))):
        errors.append("INCOMPLETE_OR_INVALID_NATIVE_CLOCK_CAPTURE")
        return result
    sources = np.concatenate(([acquired], source_times[:-1]))
    first_applications = timings[::10, 2]
    publications = np.asarray([row["actual_publish_monotonic_s"] for row in rows])
    scheduled = epoch + np.arange(count*10)*.002
    if (not np.all(np.abs(timings[:, 0]-scheduled) <= 1e-9)
            or not np.all((timings[:, 1] >= scheduled) & (timings[:, 1] < scheduled+.002))
            or not np.all((timings[:, 2] >= timings[:, 1]) & (timings[:, 2] < scheduled+.002))
            or not np.all(timings[:, 3] >= timings[:, 2])):
        errors.append("NATIVE_ALL_STEP_CLOCK_CONTRACT")
    if not np.array_equal(source_times, timings[::10, 3]):
        errors.append("SOURCE_CAPTURE_TIME_DIFFERS_FROM_COMPLETED_NATIVE_STEP")
    for i, row in enumerate(rows):
        certificate = row["certificate"]
        certificate_times = [certificate.get(name) for name in (
            "source_acquisition_time", "publish_deadline", "execution_start",
            "execution_end", "execution_start_simulation_s")]
        row_times = [row.get(name) for name in (
            "first_servo_application_monotonic_s", "actual_publish_monotonic_s",
            "dispatch_latency_s", "source_simulation_time_s")]
        if not _finite_values(certificate_times+row_times):
            errors.append(f"{i}:NONFINITE_OR_MISSING_NATIVE_TIMELINE_CLOCK")
            return result
        if certificate["source_acquisition_time"] != float(sources[i]):
            errors.append(f"{i}:SOURCE_ACQUISITION_TIME_DIFFERS_FROM_NATIVE")
        if row["first_servo_application_monotonic_s"] != float(first_applications[i]):
            errors.append(f"{i}:FIRST_APPLICATION_TIME_DIFFERS_FROM_NATIVE")
        if abs(certificate["execution_start"]-scheduled[i*10]) > 1e-9:
            errors.append(f"{i}:EXECUTION_START_DIFFERS_FROM_NATIVE_GRID")
        if (not np.isfinite(publications[i]) or publications[i] < sources[i]
                or publications[i] > certificate["publish_deadline"]):
            errors.append(f"{i}:INVALID_CAPTURED_PUBLICATION_TIME")
        dispatch = publications[i]-sources[i]
        if abs(row["dispatch_latency_s"]-dispatch) > 1e-9:
            errors.append(f"{i}:DISPATCH_LATENCY_DIFFERS_FROM_CAPTURED_CLOCKS")
    samples = {
        "dispatch": (publications-sources, .020),
        "acquisition_to_first_application": (first_applications-sources, .020),
        "publication_to_first_application": (first_applications-publications, .020),
        "application_start_offset": (first_applications-scheduled[::10], .002),
        "servo_jitter": (timings[:, 1]-timings[:, 0], .002),
        "full_servo_cycle": (timings[:, 3]-timings[:, 1], .002),
    }
    for name, (values, deadline) in samples.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            errors.append(f"{name}:INVALID_CAPTURED_LATENCY")
            continue
        summary = _clock_summary(values, deadline)
        result["statistics"][name] = summary
        _compare_clock_summary(summary, reported_rates.get(name), name, errors)
    result["passed"] = not errors
    return result


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
            other_times = [c["execution_end"], c["execution_start_simulation_s"],
                row["source_simulation_time_s"], row["dispatch_latency_s"],
                row["first_servo_application_monotonic_s"]]
            phase_times = [p[key] for p in phases for key in ("start_ns", "end_ns", "wall_s")]
            phase_times.extend(p["thread_cpu_s"] for p in phases if p.get("thread_cpu_s") is not None)
            if not _finite_values(times+other_times+phase_times):
                errors.append(f"{i}:NONFINITE_TIMELINE_CLOCK")
                continue
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
        clock_report = None
        native_info = scene["execution_contract"].get("native_raw_trace")
        if native_info:
            native_path = Path(native_info["path"])
            if sha(native_path) != native_info["sha256"]:
                errors.append("NATIVE_RAW_TRACE_HASH")
            with np.load(native_path, allow_pickle=False) as native:
                clock_report = native_clock_audit(
                    rows, native, scene["metrics"]["rates_and_latency"])
            errors.extend(clock_report["errors"])
        elif scene["execution_contract"].get("controller_version") == "v6_2_c11_native_wall_handoff":
            errors.append("MISSING_NATIVE_RAW_TRACE")
        records.append({"scenario_id": scene_id, "segments": len(rows),
            "timeline_sha256": sha(path), "native_clock_validation": clock_report,
            "errors": errors, "passed": not errors})
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
    auditor_identity = auditor_source_identity()
    records = []
    with threadpool_limits(limits=1):
        for number, run_dir in enumerate(run_dirs, 1):
            print(f"Verifying {run_dir}", flush=True)
            metadata_path = run_dir/"run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            item = {"run_dir": run_dir.as_posix(),
                "source_trial_commit": metadata.get("source", {}).get("git_commit"),
                **auditor_identity, "reports": {}, "passed": True}
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
                **auditor_identity, "records": records}, indent=2) + "\n", encoding="utf-8")
    return all(x["passed"] for x in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.run_dir, args.output_dir) else 1)
