"""Independent saved-evidence audit; no controller imports or MuJoCo stepping.

Only writes new audit products beside this script. The trial, baseline and
runtime sources are read-only. Statistics are recomputed from raw samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import traceback
import xml.etree.ElementTree as ET

import numpy as np

SCENE_CHECKS = set("""continuum_initial_position_contract rigid_grasp_final_error
rigid_grasp_steady_rmse continuum_irregular_waypoint_path_rmse
rigid_orientation_error_below_limit continuum_orientation_error_below_limit
whole_body_dense_discrete_clearance continuum_target_dense_discrete_clearance
obstacle_avoidance_engaged single_qp_every_task_tick all_qp_solves_succeeded
dynamic_execution_only no_degenerate_active_clearance_gradients
reaction_map_satisfies_zero_momentum_constraint physics_and_task_rate_exact""".split())
DELIVERY_CHECKS = set("""contract_version authoritative_metrics_hash five_seeded_scenarios
exact_control_rates requested_acceptance_thresholds_locked learning_free_architecture
original_irregular_waypoint_contract continuum_end_effector_offset_contract
no_learning_runtime_imports reported_summary_passed trace_hashes
continuum_initial_position_is_1930_626_0_mm metrics_recomputed_from_traces
native_torque_replay_exact one_qp_per_50hz_tick obstacle_constraints_materially_engaged
dynamic_ctrl_mj_step_only honest_collision_claim_scope moving_target_continuum_clearance
continuum_target_all_500hz_replay_states_clear rigid_grasp_point_error
continuum_irregular_waypoint_tracking_error both_arm_orientation_error
whole_body_minimum_clearance base_pose_drift_recomputed""".split())
EXECUTION_CHECKS = set("""new_contract_version five_scenarios selected_endpoint_equals_servo_command
no_uncertified_execution current_ramp_rows_feasible predicted_next_start_rows_feasible
shared_ten_step_reference_exact reference_measured_error_recorded
clip_and_saturation_diagnostics_recorded sim_and_wall_clock_separate trace_hashes_match""".split())
PERFORMANCE_CHECKS = {"task_controller_runs_within_50hz_p95", "torque_controller_runs_within_500hz_p95"}
STEPS = ["current_and_historical_tests", "complete_five_scene_simulation", "simulation_qualification",
         "independent_torque_delivery", "independent_execution_contract", "independent_interval_recompute",
         "exact_c1_nontiming_parity", "raw_timing_audit"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b, tolerance=1e-9):
    return finite(a) and finite(b) and abs(a - b) <= tolerance


def equal_stats(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equal_stats(actual[key], value) for key, value in expected.items())
    if isinstance(expected, float):
        return near(actual, expected, 1e-10)
    return actual == expected


def stats(raw, deadline, steady=True):
    samples = np.asarray(raw, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("invalid raw latency samples")
    longest = streak = 0
    for exceeded in samples > deadline:
        streak = streak + 1 if exceeded else 0
        longest = max(longest, streak)
    return {"count": len(samples), "p50_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000),
            "p99_ms": float(np.percentile(samples, 99) * 1000),
            "max_ms": float(np.max(samples) * 1000),
            "over_deadline_count": int(np.count_nonzero(samples > deadline)),
            "longest_consecutive_over_deadline": longest,
            "first_cycle_ms": float(samples[0] * 1000),
            "steady_after_first_cycle": stats(samples[1:], deadline, False) if steady and len(samples) > 1 else None,
            "passed": bool(np.percentile(samples, 95) <= deadline)}


def audit_cycle(row, index, trace, scene):
    checks = {}
    cert, guards = row["certificate"], row["servo_dispatch_checks"]
    start, end = row["state_acquisition_monotonic_ns"], row["actual_dispatch_monotonic_ns"]
    phases = row["phases"]
    source = float(trace["task_time"][index])
    raw_boundary = 0.0 if index == 0 else float(trace["time"][index * 10 - 1])
    clocks = [start, end, row["source_simulation_time_s"], row["dispatch_latency_s"], row["thread_cpu_s"]]
    clocks.extend(value for phase in phases for value in (
        phase["start_ns"], phase["end_ns"], phase["wall_s"], phase["thread_cpu_s"]))
    cert_clocks = [cert[name] for name in ("state_acquisition_time", "target_acquisition_time",
        "planned_execution_start", "planned_execution_start_simulation_s", "solve_finished_time",
        "validation_finished_time", "valid_until", "maximum_supported_state_age", "simulation_valid_until")]
    checks["finite_nonnegative_clocks"] = all(finite(value) and value >= 0 for value in clocks + cert_clocks)
    if not checks["finite_nonnegative_clocks"]:
        return {"scene": scene["scenario"]["scenario_id"], "cycle": index,
                "passed": False, "checks": checks, "errors": ["finite_nonnegative_clocks"]}
    checks["source_time_raw_trace_and_grid"] = (near(source, index * .020) and near(source, raw_boundary)
        and near(row["source_simulation_time_s"], source)
        and near(cert["planned_execution_start_simulation_s"], source))
    checks["phase_clock_chain"] = bool(phases) and phases[0]["start_ns"] == start and phases[-1]["end_ns"] == end \
        and end >= start and all(p["end_ns"] >= p["start_ns"] and
            near(p["wall_s"], (p["end_ns"] - p["start_ns"]) * 1e-9, 1e-12) for p in phases) \
        and all(a["end_ns"] == b["start_ns"] for a, b in zip(phases, phases[1:])) \
        and near(sum(p["thread_cpu_s"] for p in phases), row["thread_cpu_s"], 1e-12)
    checks["raw_dispatch_latency"] = near((end - start) * 1e-9, row["dispatch_latency_s"], 1e-12)
    phase_ends = {p["name"]: p["end_ns"] * 1e-9 for p in phases}
    checks["certificate_raw_wall_clock_binding"] = (near(cert["state_acquisition_time"], start * 1e-9)
        and cert["target_acquisition_time"] == cert["state_acquisition_time"]
        and cert["planned_execution_start"] == cert["state_acquisition_time"]
        and near(cert["solve_finished_time"], phase_ends.get("qp_assembly_and_solve"))
        and near(cert["validation_finished_time"], phase_ends.get("execution_validation"))
        and cert["state_acquisition_time"] <= cert["solve_finished_time"] <= cert["validation_finished_time"] <= end * 1e-9)
    checks["simulation_validity_20ms_unextended"] = (cert["validity_clock"] == "simulation_time"
        and cert["maximum_supported_state_age"] == .020
        and cert["valid_until"] == cert["state_acquisition_time"] + .020
        and near(cert["simulation_valid_until"], source + .020, 1e-12))
    checks["ordered_command_and_selected_payload"] = (row["command_id"] == index == cert["command_id"]
        and cert["command_sha256"] == hashlib.sha256(trace["task_selected_command"][index].tobytes()).hexdigest()
        and np.array_equal(trace["command_velocity"][index * 10:index * 10 + 10],
            np.repeat(trace["task_selected_command"][index][None, :], 10, axis=0)))
    checks["recorded_source_model_partition_ids"] = (cert["source_model_hash"] ==
        scene["execution_contract"]["source_compiled_model_sha256"]
        and all(isinstance(cert[key], str) and re.fullmatch(r"[0-9a-f]{64}", cert[key]) is not None
                for key in ("source_state_id", "source_model_hash", "source_partition_id", "command_sha256")))
    checks["all_ten_guard_records"] = len(guards) == 10 and row["accepted"] is True \
        and row["dispatch_check"] == guards[0] if guards else False
    guard_errors = []
    previous = -math.inf
    for substep, guard in enumerate(guards):
        scalar_values = [guard.get(key) for key in ("dispatch_time", "simulation_state_age_s",
            "simulation_valid_until", "state_age_s", "target_age_s")]
        if not all(finite(value) and value >= -1e-9 for value in scalar_values):
            guard_errors.append(f"nonfinite_guard_{substep}")
            continue
        observed_simulation = source + guard["simulation_state_age_s"]
        raw_time = 0.0 if index * 10 + substep == 0 else float(trace["time"][index * 10 + substep - 1])
        if not (guard["accepted"] is True and guard["reason"] is None
                and guard["policy"] == "research_simulation" and guard["validity_clock"] == "simulation_time"
                and guard["servo_substep"] == substep and substep < 10
                and near(guard["simulation_state_age_s"], .002 * substep)
                and near(observed_simulation, raw_time) and near(observed_simulation, index * .020 + substep * .002)
                and guard["simulation_valid_until"] == cert["simulation_valid_until"]
                and observed_simulation < cert["simulation_valid_until"]
                and cert["validation_finished_time"] <= guard["dispatch_time"] and previous <= guard["dispatch_time"]
                and near(guard["state_age_s"], guard["dispatch_time"] - cert["state_acquisition_time"])
                and near(guard["target_age_s"], guard["dispatch_time"] - cert["target_acquisition_time"])
                and guard["wall_clock_current"] == (guard["dispatch_time"] <= cert["valid_until"])
                and guard["wall_deployment_certified"] is False
                and guard["continuation_guaranteed_on_reject"] is False):
            guard_errors.append(f"invalid_guard_{substep}")
        if substep == 0 and not (start * 1e-9 <= guard["dispatch_time"] <= end * 1e-9):
            guard_errors.append("first_guard_outside_raw_timeline")
        previous = guard["dispatch_time"]
    checks["accepted_ordered_finite_guards_on_raw_simulation_grid"] = not guard_errors
    errors = [key for key, value in checks.items() if not value] + guard_errors
    return {"scene": scene["scenario"]["scenario_id"], "cycle": index,
            "certificate_id": cert["command_id"], "guard_count": len(guards),
            "accepted_after_wall_valid_until_count": sum(g["wall_clock_current"] is False for g in guards),
            "passed": not errors, "checks": checks, "errors": errors}


def run(root, trial, output):
    audit_plan = load(output / "audit_plan.json")
    plan, report = load(trial / "plan.json"), load(trial / "report.json")
    suite = trial / "simulation"
    metrics = load(suite / "v6_lite_metrics.json")
    metadata = load(suite / "run_metadata.json")
    artifact = load(suite / "artifact_manifest.json")
    native = load(suite / "validation.json")
    execution = load(suite / "execution_validation.json")
    interval = load(trial / "interval_recompute/online_recompute_summary.json")
    parity = load(trial / "parity/parity.json")
    tests = load(trial / "test_profiles/report.json")
    items = []

    def check(name, passed, detail=None):
        items.append({"name": name, "passed": bool(passed), "detail": detail})

    check("trial_finished_and_passed", report.get("passed") is True and report.get("complete") is True)
    check("declared_trial_plan_preserved", sha(trial / "plan.json") == audit_plan["trial_plan_sha256"])
    check("predeclared_steps_all_complete", [s["name"] for s in report["steps"]] == STEPS
        and plan["steps"] == STEPS and all(s["status"] == "COMPLETED" and s["source_unchanged"] is True for s in report["steps"]))
    step_results = {item["name"]: item["result"] for item in report["steps"]}
    check("saved_step_results_match_aggregate_reports", step_results["current_and_historical_tests"] is True
        and step_results["complete_five_scene_simulation"] == metrics
        and step_results["independent_torque_delivery"] == native
        and step_results["independent_execution_contract"] == execution
        and step_results["independent_interval_recompute"] == interval
        and step_results["exact_c1_nontiming_parity"] == parity
        and step_results["raw_timing_audit"] == report["computational_performance"]
        and report["algorithm_simulation"] == {"qualification": step_results["simulation_qualification"],
            "delivery": native, "execution": execution, "interval": interval, "c1_exact_parity": parity})
    check("current_and_immutable_historical_regressions", tests["passed"] is True
        and tests["current"]["tests"] == 200 and tests["current"]["failures"] == 0
        and tests["current"]["errors"] == 0 and tests["current"]["passed"] is True
        and tests["historical"]["tests_selected"] == 16 and tests["historical"]["exit_code"] == 0
        and tests["historical"]["passed"] is True and tests["historical_goldens_modified"] is False,
        {"current": tests["current"]["tests"], "historical": tests["historical"]["tests_selected"]})
    check("qualification_each_item", step_results["simulation_qualification"]["passed"] is True
        and all(value is True for value in step_results["simulation_qualification"]["checks"].values()))
    for kind, value, expected in (("delivery", native, DELIVERY_CHECKS), ("execution", execution, EXECUTION_CHECKS)):
        check(kind + "_locked_check_set", set(value["checks"]) == expected
            and value["total_count"] == len(expected) and value["passed_count"] == len(expected) and value["passed"] is True)
        for name in sorted(expected):
            check(kind + ":" + name, value["checks"].get(name) is True)
    check("delivery_only_removes_performance_gate", set(native["observed_checks"]) == DELIVERY_CHECKS | {"measured_rate_deadlines"}
        and native["acceptance_profile"] == "research_simulation" and native["performance_is_research_gate"] is False
        and native["wall_continuation_or_hardware_validated"] is False)
    config = metrics["run_config"]
    locked = {"scenario_count": 5, "seed": 20260801, "duration_s": 27.0, "physics_period_s": .002,
        "task_period_s": .020, "whole_body_minimum_clearance_m": .005, "verification_subdivisions": 4,
        "rigid_final_error_threshold_m": .0001, "rigid_steady_rmse_threshold_m": .00015,
        "continuum_irregular_path_rmse_threshold_m": .00018, "orientation_error_threshold_deg": .25,
        "task_latency_p95_threshold_s": .020, "torque_latency_p95_threshold_s": .002,
        "steady_window_s": 1.5, "pcc_mode": "bounded_interval_pcc", "dispatch_clock_policy": "research_simulation"}
    check("unchanged_task_thresholds_rates_and_research_mode", all(config.get(k) == v for k, v in locked.items())
        and config == plan["run_config"] == metadata["run_config"]
        and metrics["qp_config"] == plan["qp_config"] == metadata["qp_config"]
        and metrics["qp_config"]["enable_capsule_cbf"] is True)
    source = report["source_provenance"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    expected_commit = audit_plan["source_trial_commit"]
    check("source_trial_commit_exact", plan["source"]["git_commit"] == source["before"]["git_commit"]
        == source["after"]["git_commit"] == metadata["source"]["git_commit"] == expected_commit)
    check("frozen_source_before_after_and_current", source["source_unchanged"] is True
        and source["capture_complete"] is True and source["git_commit_unchanged"] is True
        and not source["before"]["capture_errors"] and not source["after"]["capture_errors"]
        and source["before"]["tracked_worktree_dirty"] is False and source["after"]["tracked_worktree_dirty"] is False
        and source["before"]["files"] == source["after"]["files"] == plan["source"]["files"]
        and commit == expected_commit and not status)
    current_hashes = {name: sha(root / name) for name in plan["source"]["files"]}
    check("current_raw_source_inventory_matches_frozen_trial", all(current_hashes[name] == record["sha256_raw"]
        for name, record in plan["source"]["files"].items()), {"source_file_count": len(current_hashes)})
    check("suite_source_inventory_matches_plan", all(metadata["source"]["source_normalized_lf_sha256"].get(name)
        == record["sha256_lf_normalized"] for name, record in plan["source"]["files"].items()))
    write_new(output / "source_inventory.json", {"source_trial_commit": expected_commit,
        "auditor_source_commit": commit, "auditor_script_sha256_raw": sha(__file__), "raw_source_sha256": current_hashes})
    manifest = load(trial / "manifest.json")
    produced = {path.relative_to(trial).as_posix(): sha(path) for path in trial.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    check("trial_manifest_complete_and_every_hash_matches", manifest == produced,
        {"file_count": len(produced), "manifest_sha256": sha(trial / "manifest.json")})
    check("authoritative_metrics_and_trace_manifest", artifact["metrics"]["sha256"] == sha(suite / "v6_lite_metrics.json")
        and len(artifact["traces"]) == 5 and all(entry["sha256"] == sha(root / entry["path"]) for entry in artifact["traces"]))
    reference = Path(audit_plan["reference_dir"])
    baseline_manifest = load(reference / "artifact_manifest.json")
    check("historical_manifest_preserved_since_audit_predeclaration", sha(reference / "artifact_manifest.json")
        == audit_plan["reference_artifact_manifest_sha256"])
    check("historical_metrics_hash_matches_original_manifest", sha(reference / "v6_lite_metrics.json")
        == baseline_manifest["metrics"]["sha256"] == audit_plan["reference_metrics_sha256"])
    check("parity_reference_directory_and_all_five", Path(parity["reference_dir"]).resolve() == reference.resolve()
        and Path(parity["candidate_dir"]).resolve() == suite.resolve() and parity["all_passed"] is True and len(parity["records"]) == 5)
    baseline_hashes = {Path(entry["path"]).name: entry["sha256"] for entry in baseline_manifest["traces"]}
    scenes, cycles, performance = [], [], []
    for i, scene in enumerate(metrics["scenarios"]):
        name = f"v6_lite_scenario_{i:02d}"
        check(name + ":identity_seed_and_functional_check_set", scene["scenario"]["scenario_id"] == name
            and scene["scenario"]["seed"] == 20260801 + 104729 * i
            and set(scene["checks"]) == SCENE_CHECKS and scene["passed"] is True
            and all(value is True for value in scene["checks"].values()))
        for key in sorted(SCENE_CHECKS):
            check(name + ":" + key, scene["checks"].get(key) is True)
        trace_path, old_path = suite / "traces" / (name + ".npz"), reference / "traces" / (name + ".npz")
        trace_hash, old_hash = sha(trace_path), sha(old_path)
        record = parity["records"][i]
        check(name + ":parity_hashes_bound_to_original_historical_manifest", record["scenario"] == name + ".npz"
            and old_hash == baseline_hashes[name + ".npz"] == audit_plan["reference_trace_sha256"][name + ".npz"]
            == record["reference_sha256"] and trace_hash == record["candidate_sha256"]
            and record["exact_equal"] is True and not record["mismatches"])
        timeline_path = suite / "timing" / (name + ".jsonl")
        timelines = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines()]
        with np.load(trace_path, allow_pickle=False) as trace, np.load(old_path, allow_pickle=False) as old:
            check(name + ":full_trace_shapes_and_time_grid", trace["torque"].shape == (13500, 67)
                and trace["task_selected_command"].shape == (1350, 17) and trace["task_qpos"].shape == (1351, 81)
                and trace["task_time"].shape == (1350,) and trace["time"].shape == (13500,)
                and np.all(np.isfinite(trace["time"])) and np.all(np.isfinite(trace["task_time"]))
                and np.max(np.abs(trace["time"] - np.arange(1, 13501) * .002)) <= 1e-9
                and np.max(np.abs(trace["task_time"] - np.arange(1350) * .020)) <= 1e-9)
            keys = sorted(set(trace.files) | set(old.files))
            selected_keys = [key for key in keys if "latency" not in key and "wall_time" not in key]
            mismatches = [key for key in selected_keys if key not in trace or key not in old
                or trace[key].shape != old[key].shape or trace[key].dtype != old[key].dtype
                or not (np.array_equal(trace[key], old[key], equal_nan=True) if np.issubdtype(trace[key].dtype, np.number)
                        else np.array_equal(trace[key], old[key]))]
            check(name + ":independent_all_nontiming_array_parity", not mismatches
                and len(selected_keys) == record["checked_array_count"], {"checked_arrays": len(selected_keys), "mismatches": mismatches})
            check(name + ":all_1350_certificate_timelines", len(timelines) == 1350)
            scene_cycles = [audit_cycle(row, index, trace, scene) for index, row in enumerate(timelines)]
            cycles.extend(scene_cycles)
            check(name + ":all_certificate_and_guard_items", all(row["passed"] for row in scene_cycles),
                {"certificates": len(scene_cycles), "guards": sum(row.get("guard_count", 0) for row in scene_cycles),
                 "failed_cycles": sum(not row["passed"] for row in scene_cycles)})
            measured = {"algorithm": stats(trace["task_full_latency"], .020),
                "dispatch": stats([(row["actual_dispatch_monotonic_ns"] - row["state_acquisition_monotonic_ns"]) * 1e-9
                                   for row in timelines], .020), "torque": stats(trace["torque_latency"], .002)}
            rates = scene["metrics"]["rates_and_latency"]
            scalar_map = {"task_full_latency_median_ms": measured["algorithm"]["p50_ms"],
                "task_full_latency_p95_ms": measured["algorithm"]["p95_ms"],
                "task_full_latency_max_ms": measured["algorithm"]["max_ms"],
                "torque_latency_p95_ms": measured["torque"]["p95_ms"],
                "qp_solver_latency_p95_ms": float(np.percentile(trace["task_solver_latency"], 95) * 1000),
                "shape_clearance_latency_p95_ms": float(np.percentile(trace["task_shape_clearance_latency"], 95) * 1000)}
            check(name + ":all_original_latency_scalars_raw_consistent", all(near(rates.get(key), value, 1e-10)
                for key, value in scalar_map.items()), scalar_map)
            check(name + ":all_algorithm_dispatch_reported_statistics_raw_consistent", all(
                equal_stats(rates[key], measured[key]) for key in ("algorithm", "dispatch")))
            perf_expected = {"task_controller_runs_within_50hz_p95": measured["algorithm"]["passed"],
                             "torque_controller_runs_within_500hz_p95": measured["torque"]["passed"]}
            check(name + ":performance_boolean_consistency_without_functional_gate", scene["performance_checks"] == perf_expected
                and set(scene["performance_checks"]) == PERFORMANCE_CHECKS
                and scene["performance_passed"] == all(perf_expected.values())
                and scene["performance_is_acceptance_gate"] is False)
            root_timing = report["computational_performance"]["scenes"][i]
            check(name + ":runner_timing_report_matches_independent_raw_statistics", root_timing["scenario_id"] == name
                and root_timing["passed"] is True and not root_timing["errors"]
                and root_timing["trace_sha256"] == trace_hash and root_timing["timeline_sha256"] == sha(timeline_path)
                and all(equal_stats(root_timing[key], measured[key]) for key in measured)
                and near(root_timing["initialization_latency_s"], rates["initialization_latency_s"], 1e-12))
            performance.append(all(measured[key]["passed"] for key in measured))
            scenes.append({"scenario_id": name, "trace_sha256": trace_hash, "timeline_sha256": sha(timeline_path),
                "reference_trace_sha256": old_hash, "nontiming_array_count": len(selected_keys), **measured})
    measured_original_perf = all(s["algorithm"]["passed"] and s["torque"]["passed"] for s in scenes)
    check("original_delivery_performance_booleans_match_raw_samples", native["performance_checks"] == {
        "measured_rate_deadlines": measured_original_perf}
        and native["observed_checks"]["measured_rate_deadlines"] == measured_original_perf)
    check("full_raw_certificate_guard_population", len(cycles) == 6750 and sum(row.get("guard_count", 0) for row in cycles) == 67500)
    check("overall_timing_classification_matches_raw_samples", report["computational_performance"]["performance_target_met"] == all(performance)
        and report["computational_performance"]["evidence_valid"] is True
        and report["computational_performance"]["startup_samples_excluded"] is False
        and report["computational_performance"]["performance_is_functional_gate"] is False)
    check("complete_independent_interval_scope", interval["passed"] is True and interval["scenario_count"] == 5
        and interval["task_state_checks"] == 6755 and interval["interval_rows_recomputed"] == 12350
        and interval["failure_count"] == 0 and interval["input_metrics_sha256"] == sha(suite / "v6_lite_metrics.json")
        and interval["continuous_time_certified"] is False)
    interval_paths = {"checks": "online_recompute_checks.jsonl", "interval_rows": "online_recompute_interval_rows.jsonl",
                      "failures": "online_recompute_failures.jsonl"}
    interval_rows = {}
    for key, name in interval_paths.items():
        path = trial / "interval_recompute" / name
        check("interval_" + key + "_raw_hash", sha(path) == interval["output_sha256"][key])
        interval_rows[key] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    boundaries = {(row["scenario_id"], row["tick"]) for row in interval_rows["checks"]}
    check("all_interval_boundary_records_and_zero_failures", len(interval_rows["checks"]) == 6755
        and boundaries == {(f"v6_lite_scenario_{i:02d}", j) for i in range(5) for j in range(1351)}
        and all(not row["errors"] for row in interval_rows["checks"]) and not interval_rows["failures"]
        and len(interval_rows["interval_rows"]) == 12350)
    check("interval_input_trace_hashes_bound", len(interval["inputs"]) == 5 and all(
        row["scenario_id"] == scenes[i]["scenario_id"] and row["trace_sha256"] == scenes[i]["trace_sha256"]
        and row["task_ticks"] == 1350 and row["torque_steps"] == 13500 and row["failure_count"] == 0
        for i, row in enumerate(interval["inputs"])))
    check("supplement_does_not_complete_prior_wall_or_hardware_goal", plan["supplements_previous_c11_goal"] is True
        and report["wall_continuation"] == {"status": "NOT_MET", "retested": False, "previous_goal_completed": False}
        and report["hardware_deployment"]["status"] == "NOT_ESTABLISHED" and report["hard_realtime_certified"] is False
        and metrics["wall_deployment_certified"] is False and metadata["timing_protocol"]["wall_deadline_enforced"] is False)
    with (output / "certificate_guard_checks.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in cycles:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    write_new(output / "by_item_checks.json", items)
    failures = [row for row in items if not row["passed"]]
    return {"schema": "research_saved_evidence_independent_audit_v1", "evidence_valid": not failures,
        "independent_research_acceptance_confirmed": not failures,
        "trial_directory": str(trial), "source_trial_commit": expected_commit, "auditor_source_commit": commit,
        "auditor_script_sha256_raw": sha(__file__), "audit_plan_sha256": sha(output / "audit_plan.json"),
        "trial_report_sha256": sha(trial / "report.json"), "trial_manifest_sha256": sha(trial / "manifest.json"),
        "counts": {"by_item_checks": len(items), "failed_checks": len(failures), "scenarios": len(scenes),
            "physics_steps": 67500, "certificates": len(cycles), "dispatch_guards": sum(row.get("guard_count", 0) for row in cycles),
            "interval_boundaries": len(interval_rows["checks"]), "interval_rows": len(interval_rows["interval_rows"])},
        "performance_target_met": all(performance), "performance_is_research_gate": False,
        "accepted_guard_records_after_wall_valid_until": sum(row.get("accepted_after_wall_valid_until_count", 0) for row in cycles),
        "wall_continuation_status": "NOT_MET", "previous_c11_goal_completed": False,
        "hardware_deployment": "NOT_ESTABLISHED", "continuous_time_certified": False, "hard_realtime_certified": False,
        "verification_scope": "saved reports, hashes, all non-timing arrays, raw clocks/statistics and certificate/guard records; no repeated physics",
        "source_state_hash_limit": "certificate source_state_id presence checked; integration-state hash not independently recreated from qpos-only task states",
        "failures": failures, "scenes": scenes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    if (output / "report.json").exists():
        raise FileExistsError("audit report already exists; preserve existing audit")
    try:
        result = run(args.root.resolve(), args.trial_dir.resolve(), output)
    except Exception as error:
        result = {"schema": "research_saved_evidence_independent_audit_v1", "evidence_valid": False,
            "independent_research_acceptance_confirmed": False, "auditor_script_sha256_raw": sha(__file__),
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "previous_c11_goal_completed": False, "wall_continuation_status": "NOT_MET"}
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_new(output / "report.json", result)
    write_new(output / "manifest.json", {path.name: sha(path) for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"})
    print(json.dumps({"evidence_valid": result["evidence_valid"], "report": str(output / "report.json"),
        "counts": result.get("counts"), "error": result.get("error"), "failures": result.get("failures")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["evidence_valid"] else 1)


if __name__ == "__main__":
    main()
