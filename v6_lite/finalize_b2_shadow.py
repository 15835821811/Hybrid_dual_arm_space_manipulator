"""Generate a hash-bound B.2 stage-2 gate report from immutable audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _gate_blockers(frontier: dict, refined: dict) -> dict[str, dict]:
    """Separate witnessed proxy penetration from old-velocity CBF failures."""

    if frontier["budgets"][-1] != 511 or frontier["gate_m"] != .005:
        raise ValueError("the frozen high-budget witness protocol changed")
    blockers = {}
    for mode in ("baseline", "enabled"):
        entries = [item for item in frontier["entries"] if item["mode"] == mode]
        records = [item for item in refined["records"] if item["mode"] == mode]
        witnessed_below = []
        for entry in entries:
            last = entry["budget_ladder"][-1]
            if last["point_budget"] != 511:
                raise ValueError("missing high-budget witness result")
            if last["proxy_status"] == "PROXY_CLEARANCE_BELOW_GATE":
                if not last["upper_m"] < frontier["gate_m"]:
                    raise ValueError("below-gate status lacks a point witness")
                witnessed_below.append(entry)
        start_bad = [item for item in records if item["feasibility"]["status"]
                     == "START_CLEARANCE_VIOLATION"]
        initial = [item for item in records if item["tick"] == 0]
        positive_h_bad = [item for item in start_bad
                          if item["worst_interval_start_terms"] is not None
                          and item["worst_interval_start_terms"]["h_m"] >= 0.0]
        if (len(witnessed_below) != frontier["modes"][mode]["status_by_budget"]
                ["511"].get("PROXY_CLEARANCE_BELOW_GATE", 0)
                or len(start_bad) != refined["modes"][mode]
                ["diagnostic_refined_start_violation_count"]
                or len(positive_h_bad) != refined["modes"][mode]
                ["start_violations_with_positive_interval_h"]
                or len(initial) != 5
                or len({item["scenario_id"] for item in initial}) != 5):
            raise ValueError(f"gate blocker totals disagree with source: {mode}")

        def earliest_by_scenario(items: list[dict]) -> dict[str, int]:
            first = {}
            for item in items:
                name = item["scenario_id"]
                first[name] = min(first.get(name, item["tick"]), item["tick"])
            return dict(sorted(first.items()))

        blockers[mode] = {
            "initial_proxy_safe_and_frozen_rows_feasible": sum(
                item["proxy_status"] == "PROXY_CLEARANCE_AT_LEAST_GATE"
                and item["feasibility"]["status"] == "FROZEN_ROWS_FEASIBLE"
                for item in initial),
            "sampled_proxy_below_gate_with_point_witness": len(witnessed_below),
            "first_witnessed_below_tick_by_scenario": earliest_by_scenario(
                witnessed_below),
            "sampled_old_start_clearance_violations_after_refinement": len(start_bad),
            "first_old_start_violation_tick_by_scenario": earliest_by_scenario(
                start_bad),
            "old_start_violations_with_nonnegative_interval_h": len(positive_h_bad),
        }
    return blockers


def finalize(root: Path, output_dir: Path) -> dict:
    root = Path(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "cold_shadow": root / "shadow_a1" / "shadow_report.json",
        "warm_shadow": root / "shadow_warm" / "shadow_report.json",
        "near_gate": root / "near_gate_audit" / "near_gate_audit.json",
        "b1_holdout_geometry": root / "heldout_geometry" / "heldout_geometry_report.json",
        "budget_frontier": root / "budget_frontier" / "budget_frontier.json",
        "refined_start": root / "refined_start" / "refined_start_report.json",
        "repartition_counterfactual": root / "repartition_counterfactual" / "repartition_counterfactual.json",
        "repartition_handoff": root / "repartition_handoff" / "repartition_handoff.json",
        "weighted_qp_probe": root / "qp_probe_early" / "weighted_qp_probe.json",
        "initial_qp_probe": root / "qp_probe_initial255" / "weighted_qp_probe.json",
        "initial_qp_probe_failure": root / "qp_probe_initial255_failures" / "attempt1.json",
        "candidate_ramp_early": root / "candidate_ramp_early" / "candidate_ramp_summary.json",
        "candidate_prediction": root / "candidate_prediction_early" / "candidate_prediction_summary.json",
        "candidate_prediction_failure": root / "candidate_prediction_early_failures" / "attempt1.json",
        "repeated_qp_probe": root / "repeated_qp_probe_early" / "repeated_probe_summary.json",
        "blas_thread_trial": root / "blas_thread_trial" / "abba_summary.json",
        "full_state_envelope": root / "full_state_envelope" / "envelope_sweep_summary.json",
        "microstep_envelope": root / "microstep_envelope" / "microstep_envelope_summary.json",
        "full_torque_envelope": root / "full_torque_envelope" / "full_torque_envelope_summary.json",
        "batched_point_trial": root / "batched_query_trial" / "batched_query_summary.json",
        "prepared_cold_trial": root / "prepared_query_trial" / "prepared_query_summary.json",
        "prepared_full_trace": root / "prepared_full_trace" / "prepared_full_trace_summary.json",
        "root_rescue_frontier": root / "root_rescue_frontier" / "root_rescue_summary.json",
        "full_root255_census": root / "full_root255_census" / "full_root255_summary.json",
    }
    values = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in sources.items()}
    cold, warm, near, heldout, frontier, refined, repartition, handoff, probe, initial_probe, initial_failure, candidate_ramp, candidate_prediction, prediction_failure, repeated_qp, blas_trial, sweep, micro, full_torque, batched, prepared_cold, prepared_full, root_rescue, full_root = (
        values[name] for name in sources
    )
    if not cold["passed_as_read_only_audit"] or not warm["passed_as_read_only_audit"]:
        raise ValueError("native replay shadow integrity failed")
    if not near["passed"] or not heldout["passed"]:
        raise ValueError("frozen geometry audit failed")
    if frontier["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("budget frontier is not tied to the current warm shadow")
    if refined["input_frontier"]["sha256"] != _sha(sources["budget_frontier"]):
        raise ValueError("refined start is not tied to the current budget frontier")
    if repartition["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("repartition counterfactual is not tied to the current warm shadow")
    if (repartition["point_budget"] != 64 or repartition["gate_m"] != .005
            or repartition["repartition_applied_to_execution"]):
        raise ValueError("repartition counterfactual changed the frozen comparison")
    for mode in ("baseline", "enabled"):
        counts = repartition["modes"][mode]["counts"]
        expected = warm["modes"][mode]["counts"].get("warm_all_task_unknown", 0)
        expected_ticks = warm["modes"][mode]["summaries"]["warm_all_task_query_ms"]["count"]
        if (counts.get("warm_UNKNOWN_CROSSES_GATE", 0) != expected
                or counts.get("task_ticks", 0) != expected_ticks):
            raise ValueError(f"repartition counterfactual warm count changed: {mode}")
    if handoff["input_repartition"]["sha256"] != _sha(sources["repartition_counterfactual"]):
        raise ValueError("repartition handoff is not tied to the current counterfactual")
    if (handoff["repartition_admitted_online"]
            or len(handoff["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in handoff["native_replay_checks"])):
        raise ValueError("repartition handoff native replay integrity failed")
    for mode in ("baseline", "enabled"):
        expected_samples = warm["modes"][mode]["counts"].get("UNKNOWN_CROSSES_GATE", 0)
        if handoff["modes"][mode]["sample_count"] != expected_samples:
            raise ValueError(f"repartition handoff sampled count changed: {mode}")
    if (len(refined["native_replay_checks"]) != 10
            or any(x["max_state_error"] > 1e-8
                   for x in refined["native_replay_checks"])):
        raise ValueError("refined native torque replay integrity failed")
    if (not probe["passed_as_read_only_integrity"]
            or probe["online_control_changed"]
            or probe["new_mode_closed_loop_acceptance"]
            or probe["probe_ticks"] != [50, 100, 150]
            or len(probe["native_replay_checks"]) != 10
            or len(probe["records"]) != 30
            or probe["input_refined_start_sha256"] != _sha(sources["refined_start"])):
        raise ValueError("weighted QP probe integrity or scope failed")
    if probe["source_hash_newline_policy"] != "LF_NORMALIZED":
        raise ValueError("weighted QP probe source hash policy changed")
    for name, digest in probe["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"weighted QP probe source changed: {name}")
    for item in probe["inputs"].values():
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("weighted QP probe metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("weighted QP probe trace changed")
    if (not initial_probe["passed_as_read_only_integrity"]
            or initial_probe["online_control_changed"]
            or initial_probe["new_mode_closed_loop_acceptance"]
            or initial_probe["probe_ticks"] != [0]
            or initial_probe["probe_protocol"] != "initial_preflight"
            or initial_probe["point_budget"] != 255
            or initial_probe["source_hash_newline_policy"] != "LF_NORMALIZED"
            or initial_probe["input_refined_start_sha256"]
            != _sha(sources["refined_start"])
            or len(initial_probe["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in initial_probe["native_replay_checks"])
            or len(initial_probe["records"]) != 10
            or initial_probe["source_sha256"] != probe["source_sha256"]):
        raise ValueError("initial weighted QP probe integrity or scope failed")
    for item in initial_probe["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("initial weighted QP five-scenario metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("initial weighted QP trace changed")
    for mode in ("baseline", "enabled"):
        item = initial_probe["summary"][mode]
        if (item["probe_count"] != 5
                or item["query_budget_ok_count"] != 5
                or item["proxy_safe_count"] != 5
                or item["envelope_supported_count"] != 5
                or item["validated_command_count"] != 5
                or item["admission_preconditions_met_count"] != 5):
            raise ValueError("initial QP preflight did not retain five validated starts")
        rows = [row for row in initial_probe["records"] if row["mode"] == mode]
        if (len(rows) != 5 or len({row["scenario_id"] for row in rows}) != 5
                or any(row["tick"] != 0 or row["selected_interval_count"] != 0
                       or row["query_points"] != 5
                       or row["selected_command"] is None for row in rows)):
            raise ValueError("initial QP records changed")
    initial_dir = root / "qp_probe_initial255"
    initial_manifest = json.loads((initial_dir / "weighted_qp_probe_manifest.json")
                                  .read_text(encoding="utf-8"))
    if (initial_manifest["report_sha256"] != _sha(sources["initial_qp_probe"])
            or initial_manifest["document_sha256"]
            != _sha(initial_dir / "WEIGHTED_QP_PROBE.md")):
        raise ValueError("initial weighted QP output hash changed")
    if (initial_failure["status"] != "SCRIPT_ERROR_BEFORE_QP_SOLVE"
            or initial_failure["exception"] != "TypeError: 'float' object is not iterable"
            or initial_failure["failure_output_file_count"] != 0):
        raise ValueError("initial probe first-attempt failure record changed")
    if (candidate_ramp["input_probe_sha256"] != _sha(sources["weighted_qp_probe"])
            or candidate_ramp["source_hash_newline_policy"] != "LF_NORMALIZED"
            or candidate_ramp["new_interval_mode_executed"]
            or candidate_ramp["continuous_time_certified"]
            or not candidate_ramp["private_counterfactual_servo_executed"]
            or candidate_ramp["probe_ticks"] != [50, 100, 150]
            or candidate_ramp["servo_steps_per_task"] != 10
            or len(candidate_ramp["records"]) != 30):
        raise ValueError("one-ramp candidate branch provenance failed")
    for name, digest in candidate_ramp["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"one-ramp candidate source changed: {name}")
    for item in candidate_ramp["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("one-ramp candidate metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("one-ramp candidate trace changed")
    candidate_dir = root / "candidate_ramp_early"
    candidate_outputs = {
        "summary": "candidate_ramp_summary.json",
        "microstates": "candidate_ramp_states.jsonl",
        "failures": "candidate_ramp_failures.jsonl",
        "document": "CANDIDATE_RAMP.md",
    }
    candidate_manifest = json.loads((candidate_dir / "candidate_ramp_manifest.json")
                                    .read_text(encoding="utf-8"))
    for name, filename in candidate_outputs.items():
        if candidate_manifest[f"{name}_sha256"] != _sha(candidate_dir / filename):
            raise ValueError(f"one-ramp candidate {name} hash changed")
    if (candidate_ramp["microstate_records_sha256"]
            != _sha(candidate_dir / candidate_outputs["microstates"])
            or candidate_ramp["failure_records_sha256"]
            != _sha(candidate_dir / candidate_outputs["failures"])):
        raise ValueError("one-ramp candidate records changed")
    if (candidate_prediction["input_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or candidate_prediction["input_candidate_ramp_sha256"]
            != _sha(sources["candidate_ramp_early"])
            or candidate_prediction["source_hash_newline_policy"] != "LF_NORMALIZED"
            or candidate_prediction["new_interval_mode_executed"]
            or candidate_prediction["continuous_time_certified"]
            or not candidate_prediction["margin_is_empirical_not_certified_bound"]
            or candidate_prediction["probe_ticks"] != [50, 100, 150]
            or len(candidate_prediction["cases"]) != 30):
        raise ValueError("candidate fixed-partition prediction provenance failed")
    for name, digest in candidate_prediction["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"candidate prediction source changed: {name}")
    for item in candidate_prediction["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("candidate prediction metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("candidate prediction trace changed")
    prediction_dir = root / "candidate_prediction_early"
    prediction_outputs = {
        "summary": "candidate_prediction_summary.json",
        "rows": "candidate_prediction_rows.jsonl",
        "failures": "candidate_prediction_failures.jsonl",
        "document": "CANDIDATE_PREDICTION.md",
    }
    prediction_manifest = json.loads((prediction_dir / "candidate_prediction_manifest.json")
                                     .read_text(encoding="utf-8"))
    for name, filename in prediction_outputs.items():
        if prediction_manifest[f"{name}_sha256"] != _sha(prediction_dir / filename):
            raise ValueError(f"candidate prediction {name} hash changed")
    if (candidate_prediction["row_records_sha256"]
            != _sha(prediction_dir / prediction_outputs["rows"])
            or candidate_prediction["failure_records_sha256"]
            != _sha(prediction_dir / prediction_outputs["failures"])):
        raise ValueError("candidate prediction records changed")
    partial_dir = root / "candidate_prediction_early_failures" / "attempt1_partial"
    if (prediction_failure["status"] != "SCRIPT_ERROR_AFTER_TWO_COMPLETED_PROBES"
            or prediction_failure["full_result_generated"]
            or prediction_failure["partial_row_count"] != 5
            or prediction_failure["partial_rows_sha256"]
            != _sha(partial_dir / "candidate_prediction_rows.jsonl")
            or prediction_failure["partial_failure_rows_sha256"]
            != _sha(partial_dir / "candidate_prediction_failures.jsonl")):
        raise ValueError("candidate prediction first-attempt failure changed")
    if (repeated_qp["input_original_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or repeated_qp["source_hash_newline_policy"] != "LF_NORMALIZED"
            or repeated_qp["new_interval_mode_executed"]
            or repeated_qp["full_control_cycle_measured"]
            or repeated_qp["round_count"] != 5
            or repeated_qp["complete_round_count"] != 5
            or repeated_qp["probe_ticks"] != [50, 100, 150]
            or repeated_qp["point_budget"] != 63
            or repeated_qp["candidate_mismatch_count"] != 0):
        raise ValueError("repeated read-only QP timing provenance failed")
    for name, digest in repeated_qp["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"repeated QP probe source changed: {name}")
    repeated_dir = root / "repeated_qp_probe_early"
    repeated_outputs = {
        "summary": "repeated_probe_summary.json",
        "records": "repeated_probe_records.jsonl",
        "document": "REPEATED_QP_PROBE.md",
    }
    repeated_manifest = json.loads((repeated_dir / "repeated_probe_manifest.json")
                                   .read_text(encoding="utf-8"))
    for name, filename in repeated_outputs.items():
        if repeated_manifest[f"{name}_sha256"] != _sha(repeated_dir / filename):
            raise ValueError(f"repeated QP probe {name} hash changed")
    if repeated_qp["records_sha256"] != _sha(repeated_dir / repeated_outputs["records"]):
        raise ValueError("repeated QP probe records changed")
    for round_index, item in enumerate(repeated_qp["rounds"], 1):
        if (item["round"] != round_index or item["status"] != "COMPLETE"
                or item["candidate_mismatch_count"] != 0
                or item["record_count"] != 30
                or item["report_sha256"] != _sha(Path(item["report_path"]))):
            raise ValueError("repeated QP probe round changed")
        round_dir = Path(item["report_path"]).parent
        if (item["document_sha256"] != _sha(round_dir / "WEIGHTED_QP_PROBE.md")
                or item["manifest_sha256"]
                != _sha(round_dir / "weighted_qp_probe_manifest.json")):
            raise ValueError("repeated QP probe round outputs changed")
    if (blas_trial["input_original_probe_sha256"]
            != _sha(sources["weighted_qp_probe"])
            or blas_trial["source_hash_newline_policy"] != "LF_NORMALIZED"
            or blas_trial["new_interval_mode_executed"]
            or blas_trial["full_control_cycle_measured"]
            or blas_trial["complete_group_count"] != 4
            or blas_trial["candidate_mismatch_count"] != 0
            or blas_trial["probe_ticks"] != [50, 100, 150]
            or blas_trial["point_budget"] != 63
            or [item["requested_openblas_threads"] for item in blas_trial["groups"]]
            != [2, 1, 1, 2]):
        raise ValueError("OpenBLAS ABBA trial provenance failed")
    for name, digest in blas_trial["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"OpenBLAS ABBA trial source changed: {name}")
    blas_dir = root / "blas_thread_trial"
    blas_manifest = json.loads((blas_dir / "abba_manifest.json").read_text(encoding="utf-8"))
    for label, filename in (("summary", "abba_summary.json"),
                            ("records", "abba_records.jsonl"),
                            ("document", "BLAS_THREAD_TRIAL.md")):
        if blas_manifest[f"{label}_sha256"] != _sha(blas_dir / filename):
            raise ValueError(f"OpenBLAS ABBA trial {label} hash changed")
    if blas_trial["records_sha256"] != _sha(blas_dir / "abba_records.jsonl"):
        raise ValueError("OpenBLAS ABBA trial records changed")
    for item in blas_trial["groups"]:
        folder = blas_dir / item["name"]
        if (item["status"] != "COMPLETE" or item["record_count"] != 30
                or item["candidate_mismatch_count"] != 0
                or item["report_sha256"] != _sha(folder / "weighted_qp_probe.json")
                or item["document_sha256"] != _sha(folder / "WEIGHTED_QP_PROBE.md")
                or item["manifest_sha256"]
                != _sha(folder / "weighted_qp_probe_manifest.json")
                or item["stdout_sha256"] != _sha(blas_dir / f"{item['name']}_stdout.txt")
                or item["stderr_sha256"] != _sha(blas_dir / f"{item['name']}_stderr.txt")):
            raise ValueError("OpenBLAS ABBA trial group outputs changed")
    if (sweep["warm_shadow_sha256"] != _sha(sources["warm_shadow"])
            or sweep["source_hash_newline_policy"] != "LF_NORMALIZED"
            or sweep["online_control_changed"]
            or sweep["fallback_geom_names"] != ["collision_0003"]):
        raise ValueError("full task-state envelope sweep provenance failed")
    for name, digest in sweep["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full envelope sweep source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = sweep["inputs"][mode]
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("full envelope sweep metrics changed")
        if len(item["traces"]) != 5:
            raise ValueError("full envelope sweep lacks five scenarios")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full envelope sweep trace changed")
    sweep_dir = root / "full_state_envelope"
    if (_sha(sweep_dir / "envelope_sweep_states.jsonl")
            != sweep["state_records_sha256"]
            or _sha(sweep_dir / "envelope_sweep_failures.jsonl")
            != sweep["failure_records_sha256"]):
        raise ValueError("full envelope sweep records changed")
    if (micro["input_refined_start_sha256"] != _sha(sources["refined_start"])
            or micro["input_full_sweep_sha256"] != _sha(sources["full_state_envelope"])
            or micro["input_full_sweep_states_sha256"]
            != sweep["state_records_sha256"]
            or micro["source_hash_newline_policy"] != "LF_NORMALIZED"
            or micro["new_interval_mode_executed"]
            or micro["continuous_time_certified"]
            or len(micro["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in micro["native_replay_checks"])):
        raise ValueError("microstep envelope native replay provenance failed")
    for name, digest in micro["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"microstep envelope source changed: {name}")
    for item in micro["inputs"].values():
        if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
            raise ValueError("microstep envelope metrics changed")
        if len(item["traces"]) != 5:
            raise ValueError("microstep envelope lacks five scenarios")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("microstep envelope trace changed")
    micro_dir = root / "microstep_envelope"
    if (_sha(micro_dir / "microstep_envelope_states.jsonl")
            != micro["state_records_sha256"]
            or _sha(micro_dir / "microstep_envelope_failures.jsonl")
            != micro["failure_records_sha256"]):
        raise ValueError("microstep envelope records changed")
    if (full_torque["input_full_sweep_summary_sha256"]
            != _sha(sources["full_state_envelope"])
            or full_torque["input_full_sweep_states_sha256"]
            != sweep["state_records_sha256"]
            or full_torque["input_selected_summary_sha256"]
            != _sha(sources["microstep_envelope"])
            or full_torque["input_selected_states_sha256"]
            != micro["state_records_sha256"]
            or full_torque["source_hash_newline_policy"] != "LF_NORMALIZED"
            or full_torque["new_interval_mode_executed"]
            or full_torque["continuous_time_certified"]
            or full_torque["cross_checks"] != {
                "saved_task_states": 13500, "selected_microstates": 220,
            }
            or len(full_torque["scenes"]) != 10):
        raise ValueError("full old-torque envelope provenance failed")
    for name, digest in full_torque["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full old-torque source changed: {name}")
    for item in full_torque["inputs"].values():
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("full old-torque metrics changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full old-torque trace changed")
    full_torque_dir = root / "full_torque_envelope"
    full_torque_outputs = {
        "summary": "full_torque_envelope_summary.json",
        "states": "full_torque_envelope_states.jsonl",
        "failures": "full_torque_envelope_failures.jsonl",
        "document": "FULL_TORQUE_ENVELOPE.md",
    }
    full_torque_manifest = json.loads((full_torque_dir / "full_torque_envelope_manifest.json")
                                      .read_text(encoding="utf-8"))
    for name, filename in full_torque_outputs.items():
        if full_torque_manifest[f"{name}_sha256"] != _sha(full_torque_dir / filename):
            raise ValueError(f"full old-torque {name} hash changed")
    trials = (
        (batched, root / "batched_query_trial", {
            "summary": "batched_query_summary.json",
            "document": "BATCHED_QUERY_TRIAL.md",
        }, "batched_query_manifest.json", "input_refined_start_sha256",
         sources["refined_start"]),
        (prepared_cold, root / "prepared_query_trial", {
            "summary": "prepared_query_summary.json",
            "document": "PREPARED_QUERY_TRIAL.md",
        }, "prepared_query_manifest.json", "input_warm_shadow_sha256",
         sources["warm_shadow"]),
        (prepared_full, root / "prepared_full_trace", {
            "summary": "prepared_full_trace_summary.json",
            "states": "prepared_full_trace_states.jsonl",
            "failures": "prepared_full_trace_failures.jsonl",
            "document": "PREPARED_FULL_TRACE.md",
        }, "prepared_full_trace_manifest.json", "input_warm_shadow_sha256",
         sources["warm_shadow"]),
    )
    for trial, directory, outputs, manifest_name, input_key, input_path in trials:
        if (trial[input_key] != _sha(input_path)
                or trial["source_hash_newline_policy"] != "LF_NORMALIZED"
                or trial["online_control_changed"]
                or trial["new_interval_mode_executed"]):
            raise ValueError("read-only performance trial provenance failed")
        for name, digest in trial["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"performance trial source changed: {name}")
        for item in trial["inputs"].values():
            if _sha(Path(item["metrics_path"])) != item["metrics_sha256"]:
                raise ValueError("performance trial metrics changed")
            if len(item["traces"]) != 5:
                raise ValueError("performance trial lacks five scenarios")
            for trace in item["traces"]:
                if _sha(Path(trace["path"])) != trace["sha256"]:
                    raise ValueError("performance trial trace changed")
        trial_manifest = json.loads((directory / manifest_name).read_text(
            encoding="utf-8"))
        for name, filename in outputs.items():
            if trial_manifest[f"{name}_sha256"] != _sha(directory / filename):
                raise ValueError(f"performance trial {name} hash changed")
    if (len(batched["records"]) != 60
            or len(prepared_cold["records"]) != 136
            or prepared_full["state_records_sha256"] != _sha(
                root / "prepared_full_trace" / "prepared_full_trace_states.jsonl")
            or prepared_full["failure_records_sha256"] != _sha(
                root / "prepared_full_trace" / "prepared_full_trace_failures.jsonl")
            or (root / "prepared_full_trace" /
                "prepared_full_trace_failures.jsonl").stat().st_size != 0):
        raise ValueError("performance trial records or parity changed")
    for mode in ("baseline", "enabled"):
        item = prepared_full["modes"][mode]
        if (item["counts"]["task_states"] != 6750
                or item["counts"].get("parity_failures", 0)
                or item["counts"]["UNKNOWN_CROSSES_GATE"]
                != warm["modes"][mode]["counts"]["warm_all_task_unknown"]
                or item["maximum_parity_error_m"] > 1e-12):
            raise ValueError("full saved-state prepared query parity failed")
    prediction_margins = {}
    for mode in ("baseline", "enabled"):
        metrics_path = Path(probe["inputs"][mode]["metrics_path"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_config = metrics["run_config"]
        if (run_config["task_period_s"] != .02
                or run_config["physics_period_s"] != .002
                or run_config["task_latency_p95_threshold_s"] != .02
                or metrics["qp_config"]["task_period_s"] != .02):
            raise ValueError("A.1 task cadence or latency threshold changed")
        prediction_margins[mode] = metrics["qp_config"]["lookahead_model_margin_m_s"]
        if prediction_margins[mode] != .005:
            raise ValueError("original lookahead margin changed")
        summaries = warm["modes"][mode]["summaries"]
        h = summaries["next_h_prediction_abs_error_m"]
        residual = summaries["next_start_residual_abs_error_m_s"]
        if (not 0 < residual["count"] == h["count"]
                or any(not (0 <= series["p50"] <= series["p95"]
                            <= series["p99"] <= series["max"])
                       for series in (h, residual))):
            raise ValueError("fixed-partition prediction summary invalid")
    rescue_dir = root / "root_rescue_frontier"
    if (root_rescue["input_repartition_counterfactual_sha256"]
            != _sha(sources["repartition_counterfactual"])
            or root_rescue["source_hash_newline_policy"] != "LF_NORMALIZED"
            or root_rescue["online_control_changed"]
            or root_rescue["new_interval_mode_executed"]
            or root_rescue["budgets"] != [127, 255]
            or root_rescue["max_leaves"] != 256):
        raise ValueError("root rescue frontier provenance or scope failed")
    for name, digest in root_rescue["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"root rescue source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = root_rescue["inputs"][mode]
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("root rescue five-scenario input changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("root rescue trace changed")
        result = root_rescue["modes"][mode]
        counts = result["counts"]
        frozen = repartition["modes"][mode]["counts"]["cold_UNKNOWN_CROSSES_GATE"]
        if (counts["frozen_states"] != frozen
                or result["maximum_parity_error_m"] > 1e-12):
            raise ValueError("root rescue parity or frozen population failed")
        for budget in root_rescue["budgets"]:
            prefix = str(budget)
            classified = sum(counts.get(f"{prefix}_{status}", 0) for status in (
                "PROXY_CLEARANCE_AT_LEAST_GATE", "PROXY_CLEARANCE_BELOW_GATE",
                "UNKNOWN_CROSSES_GATE"))
            if (classified != frozen
                    or result["budgets"][prefix]["prepared_query_ms"]["count"]
                    != frozen):
                raise ValueError("root rescue status totals changed")
    rescue_outputs = {
        "summary": "root_rescue_summary.json",
        "states": "root_rescue_states.jsonl",
        "failures": "root_rescue_failures.jsonl",
        "document": "ROOT_RESCUE_FRONTIER.md",
    }
    rescue_manifest = json.loads((rescue_dir / "root_rescue_manifest.json").read_text(
        encoding="utf-8"))
    for name, filename in rescue_outputs.items():
        if rescue_manifest[f"{name}_sha256"] != _sha(rescue_dir / filename):
            raise ValueError(f"root rescue {name} hash changed")
    if (rescue_dir / rescue_outputs["failures"]).stat().st_size != 0:
        raise ValueError("root rescue parity failures present")
    full_root_dir = root / "full_root255_census"
    if (full_root["input_warm_shadow_sha256"] != _sha(sources["warm_shadow"])
            or full_root["input_root_rescue_summary_sha256"]
            != _sha(sources["root_rescue_frontier"])
            or full_root["input_root_rescue_states_sha256"]
            != _sha(rescue_dir / "root_rescue_states.jsonl")
            or full_root["source_hash_newline_policy"] != "LF_NORMALIZED"
            or full_root["point_budget"] != 255
            or full_root["max_leaves"] != 256
            or full_root["online_control_changed"]
            or full_root["new_interval_mode_executed"]):
        raise ValueError("full root-255 census provenance or scope failed")
    for name, digest in full_root["source_sha256"].items():
        if _source_sha(Path("v6_lite") / name) != digest:
            raise ValueError(f"full root-255 source changed: {name}")
    for mode in ("baseline", "enabled"):
        item = full_root["inputs"][mode]
        if (_sha(Path(item["metrics_path"])) != item["metrics_sha256"]
                or len(item["traces"]) != 5):
            raise ValueError("full root-255 five-scenario input changed")
        for trace in item["traces"]:
            if _sha(Path(trace["path"])) != trace["sha256"]:
                raise ValueError("full root-255 trace changed")
        result = full_root["modes"][mode]
        counts = result["counts"]
        classified = sum(counts.get(status, 0) for status in (
            "PROXY_CLEARANCE_AT_LEAST_GATE", "PROXY_CLEARANCE_BELOW_GATE",
            "UNKNOWN_CROSSES_GATE"))
        if (counts["task_states"] != 6750
                or classified != counts["task_states"]
                or counts["root_rescue_matches"]
                != root_rescue["modes"][mode]["counts"]["frozen_states"]
                or counts.get("parity_failures", 0)
                or result["maximum_parity_error_m"] > 1e-12):
            raise ValueError("full root-255 census parity or population failed")
    full_outputs = {
        "summary": "full_root255_summary.json",
        "states": "full_root255_states.jsonl",
        "failures": "full_root255_failures.jsonl",
        "document": "FULL_ROOT255_CENSUS.md",
    }
    full_manifest = json.loads((full_root_dir / "full_root255_manifest.json").read_text(
        encoding="utf-8"))
    for name, filename in full_outputs.items():
        if full_manifest[f"{name}_sha256"] != _sha(full_root_dir / filename):
            raise ValueError(f"full root-255 {name} hash changed")
    if (full_root_dir / full_outputs["failures"]).stat().st_size != 0:
        raise ValueError("full root-255 parity failures present")
    gate = warm["online_admission_gate"]
    blockers = _gate_blockers(frontier, refined)
    remaining = sum(refined["modes"][mode]["old_bad_remains_unexecutable"]
                    for mode in ("baseline", "enabled"))
    probe_ready = all(
        probe["summary"][mode]["probe_count"] == 15
        and probe["summary"][mode]["admission_preconditions_met_count"] == 15
        and probe["summary"][mode]["validated_command_count"] == 15
        for mode in ("baseline", "enabled")
    )
    sweep_ready = all(
        sweep["modes"][mode]["state_count"] == 6750
        and sweep["modes"][mode]["status_counts"].get(
            "COVERED_AT_THIS_STATE", 0) == 6750
        for mode in ("baseline", "enabled")
    )
    micro_ready = all(
        micro["modes"][mode]["window_count"] == 10
        and micro["modes"][mode]["microstate_count"] == 110
        and micro["modes"][mode]["covered_microstate_count"] == 110
        for mode in ("baseline", "enabled")
    )
    candidate_ramp_ready = all(
        candidate_ramp["modes"][mode]["ramp_count"] == 15
        and candidate_ramp["modes"][mode]["covered_ramp_count"] == 15
        and candidate_ramp["modes"][mode]["realized_next_start_satisfied_count"] == 15
        and candidate_ramp["modes"][mode]["next_proxy_safe_count"] == 15
        and candidate_ramp["modes"][mode]["whole_body_violation_ramp_count"] == 0
        and candidate_ramp["modes"][mode]["maximum_old_torque_error"] <= 1e-8
        and candidate_ramp["modes"][mode]["maximum_old_next_qpos_error"] <= 1e-8
        for mode in ("baseline", "enabled")
    )
    candidate_prediction_ready = all(
        candidate_prediction["modes"][mode]["case_count"] == 15
        and candidate_prediction["modes"][mode]["selected_row_count"] == 50
        and candidate_prediction["modes"][mode]["cbf_optimism_over_frozen_margin_count"] == 0
        and candidate_prediction["modes"][mode]["realized_start_violation_count"] == 0
        for mode in ("baseline", "enabled")
    )
    repeated_partial_budget_ready = all(
        repeated_qp["modes"][mode]["record_count"] == 75
        and repeated_qp["modes"][mode]["query_plus_qp_probe_ms"]["p95"] <= 20.0
        for mode in ("baseline", "enabled")
    )
    full_torque_ready = all(
        full_torque["modes"][mode]["scene_count"] == 5
        and full_torque["modes"][mode]["torque_steps"] == 67500
        and full_torque["modes"][mode]["checked_states"] == 67505
        and full_torque["modes"][mode]["covered_states"] == 67505
        and full_torque["modes"][mode]["maximum_native_replay_error"] <= 1e-8
        for mode in ("baseline", "enabled")
    )
    status = ("GATE_NOT_MET" if gate["status"] == "NOT_MET" or remaining
              or not probe_ready or not sweep_ready or not micro_ready
              or not candidate_ramp_ready or not candidate_prediction_ready
              or not repeated_partial_budget_ready
              or not full_torque_ready
              else "NEEDS_TRUE_ONLINE_TIMING")
    lines = [
        "# V6.2-B.2 第二阶段：影子评估与在线接入门禁", "",
        f"当前门禁：**{status}**。因此本证据不作为新区间模式的五场景闭环验收。",
        "", "## 20 ms 门槛的来源", "",
        "旧 A.1 两组已发布配置均固定 50 Hz 规划周期 (`task_period_s=0.02`)，"
        "500 Hz MuJoCo/力矩周期 (`physics_period_s=0.002`)，"
        "即每个规划命令对应共享十步斜坡。原任务控制全链 p95 时延"
        "验收阈值也设为 0.02 s；B.2 沿用它作为周期预算，"
        "没有从 CBF 数学性质推得这个数值。只读查询或查询加 QP 探针"
        "不是全链计时；即使全链 p95 达标，也不构成每周期无超时的硬实时证明。",
        "", "## A.1 原生力矩重放上的只读几何", "",
        "两组旧控制 trace 均按 500 Hz 力矩原生重放；每个规划边界先调用 `mj_forward`"
        " 更新空间几何量，再与保存的 50 Hz 状态及哈希核对。"
        "每组五场景，每场景预定抽取 27 个规划状态；持久查询则在每组 "
        f"{warm['modes']['baseline']['summaries']['warm_all_task_query_ms']['count']:,} "
        "个规划 tick 上更新。", "",
        "| 模式 | 冷查询 p95 ms | 持久查询采样 p95 ms | 持久查询全部 tick 未知 | 最大叶数 | 配对容量估计 p95 ms | 真实几何误拒绝 | 子空间外状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c, w = cold["modes"][mode], warm["modes"][mode]
        lines.append(
            f"| {mode} | {c['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['counts'].get('warm_all_task_unknown', 0)} | "
            f"{w['summaries']['warm_all_task_partition_size']['max']:.0f} | "
            f"{w['summaries']['paired_replacement_estimate_ms']['p95']:.3f} | "
            f"{w['counts'].get('proxy_false_reject_vs_mujoco', 0)} | "
            f"{w['counts'].get('off_shape_subspace', 0)} |"
        )
    lines += [
        "", "持久区间未合并时可增长到预算边界，许多贴近 5 mm 门槛的状态保持未知。"
        "排除区间在所测下一 tick 进入激活距离的计数为 0，但此有限重放结果不是一般动态保证。",
        "配对容量估计使用旧全链耗时减旧形状查询耗时再加影子查询耗时；"
        "它不是新模式实测，也不能单独证明 20 ms 全链 p95。", "",
        "区间导数对每个状态的基座和目标刚体原点各调用一次 MuJoCo Jacobian，"
        "再将刚体速度精确平移至各中点及目标 witness；"
        "下面的计数包含形状 Jacobian 与这两次 MuJoCo 调用。", "",
        "| 模式 | 抽样 Jacobian 次数 p95 | 统一预算超限 | 持久热查询 p95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = warm["modes"][mode]
        lines.append(
            f"| {mode} | {item['summaries']['jacobian_evaluations']['p95']:.0f} | "
            f"{item['counts'].get('unified_budget_exceeded', 0)} | "
            f"{item['summaries']['query_time_ms']['p95']:.3f} |"
        )
    lines += [
        "", "调用次数下降不等于墙钟时延同比例下降；这里仍无新区间模式的实测全链时延。", "",
        "## 固定区间的下一周期预测误差", "",
        "在旧 A.1 力矩原生重放中，用当前冻结区间和共享十步斜坡预测下一规划边界的"
        "区间安全函数及起点残差，再于下一边界重算同一固定区间。"
        "下表只统计具备配对下一状态和所需导数的行。", "",
        "| 模式 | 配对区间行 | 安全函数绝对误差 p95 / 最大 (µm) | 起点残差绝对误差 p95 / 最大 (mm/s) | 原前瞻裕度 (mm/s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        summaries = warm["modes"][mode]["summaries"]
        h = summaries["next_h_prediction_abs_error_m"]
        residual = summaries["next_start_residual_abs_error_m_s"]
        lines.append(
            f"| {mode} | {residual['count']} | "
            f"{1e6 * h['p95']:.3f} / {1e6 * h['max']:.3f} | "
            f"{1e3 * residual['p95']:.3f} / {1e3 * residual['max']:.3f} | "
            f"{1e3 * prediction_margins[mode]:.3f} |"
        )
    lines += [
        "", "所测两组最大绝对残差均低于原 5 mm/s 前瞻裕度，但这只是旧轨迹、"
        "抽样状态和固定区间上的经验观察。绝对误差不保留偏差方向，"
        "本审计也没有分离区间切换与新命令执行误差；不能把该裕度称为严格误差上界，"
        "更不能用它放行当前起点违反。", "",
        "持久未知状态另以保存的 A.1 规划位置做同状态对照：保持 64 点预算不变，"
        "只从五段根区间重新查询，未重放力矩，也未将新分区用于动作。", "",
        "| 模式 | 持久未知 | 从根区间后确定安全 | 确定低于门槛 | 仍未知 | 额外冷查询 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = repartition["modes"][mode]
        c = item["counts"]
        lines.append(
            f"| {mode} | {c.get('warm_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{c.get('cold_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{item['counterfactual_cold_query_time_ms'].get('p95', float('nan')):.3f} |"
        )
    lines += [
        "", "重分区能解释部分持久未知，但会改变区间安全函数集合。"
        "这项额外耗时不能与热查询并列当作新模式全链计时；"
        "分区切换后的斜坡起点、必要约束和预算仍须单独验证。", "",
        "对持久未知的抽样 tick，又以 500 Hz 旧力矩原生重放，从根分区重建区间行，"
        "加入独立 MuJoCo／胶囊行，按原速度盒与十步斜坡做冻结 LP。", "",
        "| 模式 | 抽样未知 | 根分区确定安全 | 安全且旧起点可执行 | 安全但旧起点不可执行 | Jacobian 超预算 | 下一 tick 漏选激活 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = handoff["modes"][mode]
        lines.append(
            f"| {mode} | {item['sample_count']} | "
            f"{item['cold_status'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{item['safe_repartition_executable_candidate_count']} | "
            f"{item['safe_repartition_not_executable_count']} | "
            f"{item['jacobian_budget_exceeded']} | "
            f"{item['excluded_reached_activation_next_tick']} |"
        )
    lines += [
        "", "根分区代理安全只完成几何判定；旧斜坡起点仍可能不合格，"
        "且此计算时间不含独立原约束装配、LP 或新控制器全链。"
        "这个有限旧轨迹审计不能批准在线重分区或证明新控制轨迹可行。", "",
        "区间激活筛选从现有速度、加速度和关节限位得到候选盒，"
        "并将十步斜坡起点速度纳入逐轴绝对上界；"
        "空盒退回全局速度上界并单独使门禁失败。"
        "以下计数仅比较两种冻结模型筛选，均未修改安全距离或真实执行动作。", "",
        "| 查询 | 模式 | 比全局速度筛选少选区间 | 多选区间 | 空候选盒 | 排除区间下一 tick 激活 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for query_name, report in (("cold", cold), ("persistent", warm)):
        for mode in ("baseline", "enabled"):
            counts = report["modes"][mode]["counts"]
            lines.append(
                f"| {query_name} | {mode} | "
                f"{counts.get('ramp_box_newly_excluded', 0)} | "
                f"{counts.get('ramp_box_newly_selected', 0)} | "
                f"{counts.get('empty_candidate_velocity_box', 0)} | "
                f"{counts.get('excluded_reached_activation_next_tick', 0)} |"
            )
    lines += [
        "", "速度盒只约束声明的斜坡起点和候选终点；反作用映射、目标漂移"
        "和几何灵敏度在筛选中仍冻结。零漏选是有限重放观察，不是跨周期证明。", "",
        "## 同一冻结行的动作可行性", "",
        "在抽样 A.1 状态上，独立重算 MuJoCo 和实际链胶囊行后加入必要的区间行；"
        "只读 LP 使用原 17 维速度盒、十步斜坡起点、原容差和冻结前瞻约束。"
        "LP 可行仅表示存在终点，不表示历史斜坡起点或任何实际命令通过。", "",
        "| 模式 | 冷查询起点违反 | 持久查询起点违反 | 持久查询无可行终点 | 持久查询历史动作被拒 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c = cold["modes"][mode]["counts"]
        w = warm["modes"][mode]["counts"]
        lines.append(
            f"| {mode} | {c.get('frozen_START_CLEARANCE_VIOLATION', 0) + c.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_START_CLEARANCE_VIOLATION', 0) + w.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_NO_FEASIBLE_ENDPOINT', 0)} | "
            f"{w.get('historical_endpoint_rejected_by_new_rows', 0)} |"
        )
    failures = [
        record for report in (cold, warm) for mode in ("baseline", "enabled")
        for record in report["modes"][mode]["frozen_feasibility_records"]
        if record["feasibility"]["status"].startswith("START_")
    ]
    if failures and all(
        record["feasibility"].get("worst_start_source", "").startswith("pcc_interval:")
        for record in failures
    ):
        lines.append(
            "四组所列起点反例的最差行均来自 PCC 区间；该代理残差不能解释为实际链碰撞。"
        )
    lines += [
        "", "逐状态失败、区间 ID、起点残差和 LP 结果保留在两份影子 JSON 中。"
        "这些是旧轨迹的反例，不可引用为新控制器已经失败或已完成闭环。", "",
        "## 预算阶梯与细分后起点", "",
        "固定预算阶梯 31/63/127/255/511 从完整根分区重新查询保存状态。"
        "随后在原生力矩重放中，以 255 点诊断预算重建新区间行与斜坡起点。"
        "高预算只用于辨别原因，不是在线接入配置。", "",
        "| 模式 | 511 点确定代理低于 5 mm | 511 点仍未知 | 旧起点违例经细分消除 | 细分后旧违例仍不可执行 | 细分计算超过 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        front = frontier["modes"][mode]
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {front['status_by_budget']['511'].get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{front['status_by_budget']['511'].get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{ref['old_bad_cured_by_diagnostic_refinement']} | "
            f"{ref['old_bad_remains_unexecutable']} | "
            f"{ref['interval_20ms_exceeded']} |"
        )
    lines += [
        "", "代理低于门槛不是实际链碰撞；细分后区间函数与梯度必须重新计算，"
        "且分区切换还可能产生新的起点违例。逐状态预算、失败原因及运行耗时均保留在"
        "预算阶梯和细分起点 JSON 中。", "",
        "最差 PCC 起点行进一步拆为静态区间 h、形变速度、基座反作用、目标漂移"
        "和屏障项，各项和须重组为保存的起点残差：", "",
        "| 模式 | 细分后起点违反且最差区间 h >= 0 | 细分后起点违反且最差区间 h < 0 |",
        "| --- | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {ref['start_violations_with_positive_interval_h']} | "
            f"{ref['start_violations_with_negative_interval_h']} |"
        )
    enabled_bad_terms = [
        item["worst_interval_start_terms"] for item in refined["records"]
        if item["mode"] == "enabled"
        and item["feasibility"]["status"] == "START_CLEARANCE_VIOLATION"
        and item["worst_interval_start_terms"] is not None
    ]
    if enabled_bad_terms:
        def mean_rate_mm_s(name: str) -> float:
            return (1000.0 * sum(item[name] for item in enabled_bad_terms)
                    / len(enabled_bad_terms))

        lines.append(
            f"启用组这 {len(enabled_bad_terms)} 条最差行的平均形变速率 "
            f"{mean_rate_mm_s('shape_rate_m_s'):.2f} mm/s、基座反作用 "
            f"{mean_rate_mm_s('base_reaction_rate_m_s'):.2f} mm/s、目标漂移 "
            f"{mean_rate_mm_s('target_drift_m_s'):.2f} mm/s、屏障项 "
            f"{mean_rate_mm_s('barrier_rate_m_s'):.2f} mm/s。"
        )
    lines += [
        "", "启用组静态下界仍达到 5 mm 门槛的起点也可能因旧动作的相对逼近速率"
        "违反新区间 CBF；进一步提高静态查询预算本身不能使这些旧斜坡起点合格。"
        "这不证明新区间模式一定无法找到不同轨迹，也不批准在当前门禁下接入。", "",
        "## 门禁障碍的首次出现", "",
        "511 点诊断中的代理低于门槛状态都有 PCC 曲线中点 witness：该点的"
        " PCC 净空上界小于 5 mm。继续细分不能把同一旧状态变成代理安全；"
        "这仍不等于实际离散链碰撞。下表的 tick 是 50 Hz 规划 tick，"
        "来自每 50 tick 抽样，"
        "只是所测首次出现位置。", "",
        "| 模式 | 五场景初始可行 | 代理低于门槛的 witness 状态 | 细分后旧起点违反 | 其中最差行 h >= 0 | 各场景首次 witness tick | 各场景首次起点违反 tick |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for mode in ("baseline", "enabled"):
        item = blockers[mode]

        def ticks(name: str) -> str:
            values = item[name]
            return ", ".join(f"{scenario[-2:]}:{tick}"
                             for scenario, tick in values.items()) or "无"

        lines.append(
            f"| {mode} | "
            f"{item['initial_proxy_safe_and_frozen_rows_feasible']}/5 | "
            f"{item['sampled_proxy_below_gate_with_point_witness']} | "
            f"{item['sampled_old_start_clearance_violations_after_refinement']} | "
            f"{item['old_start_violations_with_nonnegative_interval_h']} | "
            f"{ticks('first_witnessed_below_tick_by_scenario')} | "
            f"{ticks('first_old_start_violation_tick_by_scenario')} |"
        )
    lines += [
        "", "开启组的静态 witness 未发现低于门槛，但旧起点仍可因相对逼近过快"
        "而失效。要验证不同的控制轨迹，必须先解决在线接入门禁；"
        "旧 trace 的继续重放不能代替新区间闭环。", "",
        "## 原加权 QP 的早期只读探针", "",
        "在旧力矩 trace 的每场景 tick 50/100/150，用 63 点根分区构造区间行，"
        "与原 MuJoCo 和胶囊行放进同一个 17 维加权 QP。逐行结果与独立重算对照；"
        "候选仅做原动作验证，不发送给执行器。每个探针清空对偶热启动，"
        "因此下面的耗时既不是连续新模式全链，也不能代替其 20 ms 验收。", "",
        "| 模式 | 冻结状态 | 查询预算合格 | 静态代理安全 | 包络证据支持 | QP 验证命令 | 接入前提合格 | 查询加 QP 探针 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = probe["summary"][mode]
        lines.append(
            f"| {mode} | {item['probe_count']} | {item['query_budget_ok_count']} | "
            f"{item['proxy_safe_count']} | {item['envelope_supported_count']} | "
            f"{item['validated_command_count']} | "
            f"{item['admission_preconditions_met_count']} | "
            f"{item['query_plus_qp_probe_ms']['p95']:.3f} |"
        )
    lines += [
        "", "早期探针证明原求解及动作验证接口能够处理这些冻结新区间行；"
        "但现有严格形状子空间判据未支持真实链包络，不能把 QP 命令验证"
        "改写成新区间安全接入。"
        f"子空间阈值 {probe['shape_subspace_membership_tolerance_rad']:.1e} rad；"
        f"两组残差 p95 分别为 "
        f"{probe['summary']['baseline']['subspace_residual_linf_rad']['p95']:.2e}、"
        f"{probe['summary']['enabled']['subspace_residual_linf_rad']['p95']:.2e} rad。"
        "实际与投影 URDF 离散骨架的解析位移上界 p95 分别为 "
        f"{probe['summary']['baseline']['discrete_backbone_residual_upper_m']['p95']:.2e}、"
        f"{probe['summary']['enabled']['discrete_backbone_residual_upper_m']['p95']:.2e} m；"
        "该只读诊断没有建立 PCC 到投影离散链的包络，也不覆盖物理碰撞几何。"
        "另以实际 MuJoCo 状态逐个检查胶囊是否包含于现有 PCC 管："
        f"baseline {probe['summary']['baseline']['state_local_capsule_covered_count']}/15、"
        f"enabled {probe['summary']['enabled']['state_local_capsule_covered_count']}/15，"
        "最小半径余量分别为 "
        f"{probe['summary']['baseline']['state_local_capsule_envelope_min_margin_m']['min']:.2e}、"
        f"{probe['summary']['enabled']['state_local_capsule_envelope_min_margin_m']['min']:.2e} m。"
        "该证据只覆盖冻结时刻的实际几何；未证明命令执行期间的包络保持。"
        "检查耗时单列，未计入上表 QP 探针：两组 p95 分别为 "
        f"{probe['summary']['baseline']['state_local_capsule_envelope_check_ms']['p95']:.3f}、"
        f"{probe['summary']['enabled']['state_local_capsule_envelope_check_ms']['p95']:.3f} ms。"
        "所测耗时尾部和后续旧轨迹反例也仍存在。", "",
        "## 早期只读 QP 的五轮重复计时", "",
        "在相同两组五场景 tick 50/100/150 冻结状态预先固定五轮，每轮重新原生重放"
        "旧力矩状态、建立模型并清空 QP 对偶热启动。150 条候选命令及区间选择"
        "与首次探针逐条一致；逐轮原始记录和哈希保留。", "",
        "| 模式 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | 再加当前状态包络 p95 / 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = repeated_qp["modes"][mode]
        partial = item["query_plus_qp_probe_ms"]
        envelope = item["probe_plus_envelope_ms"]
        lines.append(
            f"| {mode} | {item['record_count']} | "
            f"{partial['p95']:.3f} / {partial['p99']:.3f} / {partial['max']:.3f} | "
            f"{partial['over_20ms_count']} | "
            f"{envelope['p95']:.3f} / {envelope['max']:.3f} |"
        )
    lines += [
        "", "两组的查询加 QP p95 均已超过既有 20 ms 周期。该探针只测部分链路；"
        "没有计入全部状态采样、监控及 500 Hz 力矩计算和执行，"
        "包络列也只是同冻结状态计时相加。它不能替代新区间模式真实全链计时，"
        "超 20 ms 的只读探针记录也不能称为在线漏期。", "",
        "## 线性代数线程数的 ABBA 只读对照", "",
        "在相同 30 个冻结状态上，按请求 OpenBLAS 线程数 2→1→1→2 "
        "分别启动独立进程；所有候选与首次探针核对，保留每组原始报告。", "",
        "| 模式 | 请求线程数 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | QP 求解 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for threads in (2, 1):
            item = blas_trial["modes"][mode][str(threads)]
            partial = item["query_plus_qp_probe_ms"]
            lines.append(
                f"| {mode} | {threads} | {item['record_count']} | "
                f"{partial['p95']:.3f} / {partial['p99']:.3f} / "
                f"{partial['max']:.3f} | {partial['over_20ms_count']} | "
                f"{item['qp_solver_ms']['p95']:.3f} |"
            )
    lines += [
        "", "只改变子进程的 `OPENBLAS_NUM_THREADS` 请求值；未固定操作系统调度，"
        "也未核实每个库的实际线程数。这是本机性能敏感性诊断，"
        "不能与旧影子的容量估计拼接成全链 20 ms 验收。"
        "同一次 ABBA 对照中 1 与 2 的 p95 差异很小，且两者均未复现前一次"
        "五轮重复探针的耗时尾部；因此不能把跨次差异归因于线程数，"
        "也不能丢弃先前超 20 ms 的原始记录。"
        "新区间接入门禁仍需独立满足几何、动作与真实闭环要求。", "",
        "## 五场景初始状态的加权 QP 只读预检", "",
        "另在两组五场景的 tick 0 从根区间以最多 255 点查询，并将所需区间行"
        "送入原单个 17 维加权 QP；候选只经原动作验证，未驱动力矩伺服。"
        "本次 10 个初始状态都只用五个根中点完成静态判定，冻结可达筛选"
        "不要求任何 PCC 导数行，仍保留原 MuJoCo 和胶囊行。"
        "首次试运行因只读 QP 包装器在空区间行集合上调用 `min` 而在求解前报错；"
        "失败快照已保存，修复后重新生成下表。", "",
        "| 模式 | 初始状态 | 代理安全 | 包络证据支持 | 验证出命令 | 接入前提合格 | 查询加 QP 探针 p95 / 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = initial_probe["summary"][mode]
        lines.append(
            f"| {mode} | {item['probe_count']} | {item['proxy_safe_count']} | "
            f"{item['envelope_supported_count']} | "
            f"{item['validated_command_count']} | "
            f"{item['admission_preconditions_met_count']} | "
            f"{item['query_plus_qp_probe_ms']['p95']:.3f} / "
            f"{item['query_plus_qp_probe_ms']['max']:.3f} |"
        )
    lines += [
        "", "初始预检说明原求解器能够在当前冻结起点生成并验证候选；"
        "空区间行并不意味着取消完整覆盖或未来的区间约束。"
        "这些计时不含状态包络检查和连续执行；enabled 的所测尾部还超过"
        "20 ms，故不能当作全链性能通过。后续状态的子空间残差、"
        "旧速度起点违例和预算未知仍阻止在线门禁。", "",
        "## 早期候选命令的单周期私有力矩分支", "",
        "对上面的 tick 50/100/150 只读 QP 候选，在旧 A.1 冻结状态另开"
        "私有 MuJoCo 分支。先以旧命令逐步复现保存的十步力矩和下一规划 qpos；"
        "再用同一参考斜坡及 67 路力矩伺服执行候选，并在实际到达的下一起点"
        "重新查询 255 点根区间、独立装配原 MuJoCo/胶囊及新区间行。"
        "这检验冻结预测之后的实际一步结果，不是新模式在线闭环。", "",
        "| 模式 | 单周期分支 | 11 状态均包含 | 实现下一起点满足 | 代理安全 | 全身几何违例窗口 | 最小包络余量 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = candidate_ramp["modes"][mode]
        lines.append(
            f"| {mode} | {item['ramp_count']} | {item['covered_ramp_count']} | "
            f"{item['realized_next_start_satisfied_count']} | "
            f"{item['next_proxy_safe_count']} | "
            f"{item['whole_body_violation_ramp_count']} | "
            f"{1000*item['minimum_envelope_margin_m']:.3f} |"
        )
    lines += [
        "", "候选和旧命令的差异在这 30 个分支中确实非零；旧命令分支"
        "的力矩及下一 qpos 最大误差均为零。"
        "全身几何采用原验证器有限插值检查，仍不是连续时间认证。"
        "这只是从历史状态出发的单周期反事实；不能覆盖新区间闭环累计误差、"
        "停止策略、完整五场景或 20 ms 全链时延。", "",
        "## 候选分支的同区间前瞻误差", "",
        "对同一 30 个私有候选分支，重建原只读 QP 的区间划分与筛选 ID，"
        "在执行后一规划起点冻结相同 ID 重算 h、17 维广义梯度及目标漂移。"
        "因此下表的误差来自同一材料区间函数的状态变化。首次运行因下一"
        "qpos 数组切片少一项而报错；部分行和错误快照已单独保留，修复后"
        "在新目录完整重跑。", "",
        "| 模式 | 单周期分支 | 固定区间行 | h 最大预测高估 mm/s | CBF 起点松弛最大预测高估 mm/s | 超 5 mm/s 经验裕度 | 实际起点违例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = candidate_prediction["modes"][mode]
        lines.append(
            f"| {mode} | {item['case_count']} | {item['selected_row_count']} | "
            f"{1000*item['h_optimism_rate_m_s']['max']:.3f} | "
            f"{1000*item['cbf_start_slack_optimism_m_s']['max']:.4f} | "
            f"{item['cbf_optimism_over_frozen_margin_count']} | "
            f"{item['realized_start_violation_count']} |"
        )
    lines += [
        "", "h 残差除以原 20 ms 周期转成速度单位；CBF 松弛残差还包含"
        "广义梯度与目标漂移变化。这里 5 mm/s 仍是原经验配置，不是"
        "通用或严格误差上界；30 个单周期样本不能推出连续闭环可靠性。", "",
        "## 全部保存规划状态的实际胶囊包络", "",
        "对旧 A.1 两组五场景全部保存的规划 qpos 直接执行 MuJoCo 正运动学，"
        "逐状态检查实际胶囊是否包含于原 PCC 管；所用 trace 与上述原生重放审计逐场景哈希一致。"
        "这不是第二次力矩重放，也不是新区间模式动作。", "",
        "| 模式 | 保存状态 | 当前状态包含 | 未包含 | 最小余量 mm | 包络检查 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = sweep["modes"][mode]
        lines.append(
            f"| {mode} | {item['state_count']} | "
            f"{item['status_counts'].get('COVERED_AT_THIS_STATE', 0)} | "
            f"{item['status_counts'].get('NOT_COVERED_AT_THIS_STATE', 0)} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['check_ms_excluding_mj_forward']['p95']:.3f} |"
        )
    lines += [
        "", "全量余量均为正，但这只证明保存时刻的模型几何包含。"
        "检查耗时不含 MuJoCo 正运动学、区间查询和 QP；无法单独证明 20 ms 全链。"
        "本节全量扫描不包括斜坡内部；下节另检查预定的 20 个斜坡窗口。"
        "仍无跨规划周期的包络保持证明。", "",
        "## 原生力矩重放的十步斜坡中间状态", "",
        "每场景预定 tick 50 和首次抽样旧速度 CBF 起点违例 tick 两个窗口，"
        "原生重放旧 trace 的 500 Hz 力矩，对十步斜坡两端及九个内部状态"
        "分别检查实际胶囊包络。规划边界与保存状态逐步一致；未执行新区间 QP 命令。", "",
        "| 模式 | 窗口 | 500 Hz 状态 | 当前状态包含 | 最小余量 mm | 内部余量低于两端的窗口 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = micro["modes"][mode]
        lines.append(
            f"| {mode} | {item['window_count']} | {item['microstate_count']} | "
            f"{item['covered_microstate_count']} | "
            f"{1000 * item['minimum_margin_m']['min']:.3f} | "
            f"{item['interior_below_both_endpoints_count']} |"
        )
    lines += [
        "", "所选 20 个窗口不能代表全部 13,500 个斜坡；500 Hz 离散观察"
        "不证明两次观测之间、连续时间或新区间闭环安全。", "",
        "## 旧 A.1 全轨迹每个物理步的实际胶囊包络", "",
        "对原两组五场景的全部力矩按 MuJoCo 原生重放，在每个 500 Hz 物理步"
        "起点和最终端点检查实际胶囊与原 PCC 管。全部 13,500 个保存规划状态"
        "和先前 220 个选定斜坡状态逐状态交叉核对；逐步记录和失败快照保存在"
        "独立的不可覆盖审计目录。", "",
        "| 模式 | 力矩步 | 500 Hz 检查状态 | 当前状态包含 | 最小余量 mm | 最大重放误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = full_torque["modes"][mode]
        lines.append(
            f"| {mode} | {item['torque_steps']} | {item['checked_states']} | "
            f"{item['covered_states']} | {1000*item['minimum_margin_m']:.3f} | "
            f"{item['maximum_native_replay_error']:.2e} |"
        )
    lines += [
        "", "这是历史旧控制轨迹的离散时刻几何证据，仍不能推断相邻 2 ms"
        "状态之间的连续时间包含，也不能替代新区间模式的五场景闭环和独立重放。"
        "包络检查耗时不含 MuJoCo 正运动学、区间查询、Jacobian、QP 和执行，"
        "不能作为 20 ms 全链验收。", "",
        "## 只读查询计算复用试验", "",
        "首次批量点模型试验在 30 个冻结状态上没有获得稳定提速，"
        "因此未用于控制或正式影子结果。随后将同一状态的五段完整变换仅计算一次，"
        "并保持参考查询的分区细分、区间下界及终止规则。"
        "冷查询试验覆盖 58 个已保存的持久热查询抽样未知状态和 10 个早期状态；"
        "以下全量对照使用旧 A.1 的 13,500 个保存规划 qpos。", "",
        "| 模式 | 全量状态 | 分区/判定不一致 | 最大距离偏差 m | 参考查询 p95 ms | 前缀复用 p95 ms | 单状态配对加速比 p50 | 持久未知 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = prepared_full["modes"][mode]
        lines.append(
            f"| {mode} | {item['counts']['task_states']} | "
            f"{item['counts'].get('parity_failures', 0)} | "
            f"{item['maximum_parity_error_m']:.2e} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} | "
            f"{item['per_state_speedup']['p50']:.3f} | "
            f"{item['counts']['UNKNOWN_CROSSES_GATE']} |"
        )
    lines += [
        "", "全量试验以参考结果推进持久分区；每状态另用试验实现检查同一输入。"
        "查询时间只在本次配对运行内比较，不能与先前报告的 p95 直接拼接。"
        "该改进降低了计算时间，没有消除未知或旧起点违例；"
        "也未计入 MuJoCo 正运动学、广义 Jacobian、原约束、QP 和力矩执行。"
        "没有新区间模式全链 20 ms 验收或连续时间认证。", "",
        "## 根区间未知状态的预算敏感性", "",
        "针对 64 点根区间重查询后仍未知的 998 个冻结 A.1 状态，"
        "分别以 127 和 255 点预算从五段根区间独立重查。"
        "下表仅统计这些冻结状态；每个查询均与参考实现逐状态核对分区和判定。", "",
        "| 模式 | 冻结未知 | 点预算 | 确定安全 | 确定低于门槛 | 仍未知 | 前缀复用查询 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = root_rescue["modes"][mode]
        counts = item["counts"]
        for budget in root_rescue["budgets"]:
            prefix = str(budget)
            lines.append(
                f"| {mode} | {counts['frozen_states']} | {budget} | "
                f"{counts.get(prefix + '_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
                f"{counts.get(prefix + '_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
                f"{counts.get(prefix + '_UNKNOWN_CROSSES_GATE', 0)} | "
                f"{item['budgets'][prefix]['prepared_query_ms']['p95']:.3f} |"
            )
    lines += [
        "", "255 点预算使 enabled 组这 874 个冻结未知状态全部判为安全，"
        "baseline 组仍有 17 个未知。该结果仅说明预算敏感性；"
        "不覆盖其他已确定低于门槛的状态、旧速度起点违例或全任务动态演化。"
        "查询耗时不含 MuJoCo 正运动学、广义 Jacobian、原约束、QP 与力矩伺服；"
        "较大预算尚未进入在线控制，不能由此宣称 20 ms 全链通过。", "",
        "## 全部保存状态的 255 点根区间查询", "",
        "为避免只看先前未知状态带来的选择偏差，对两组五场景全部 13,500 个"
        "保存规划状态分别从五段根区间重新查询；每状态参考实现与前缀复用实现"
        "逐项比较，先前 998 个未知状态还与独立预算阶梯交叉核对。", "",
        "| 模式 | 状态 | 代理安全 | 代理低于 5 mm | 未知 | 预算耗尽 | 分区/判定不一致 | 参考查询 p95 ms | 前缀复用 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = full_root["modes"][mode]
        counts = item["counts"]
        lines.append(
            f"| {mode} | {counts['task_states']} | "
            f"{counts.get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{counts.get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{counts.get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{counts['budget_exhausted']} | "
            f"{counts.get('parity_failures', 0)} | "
            f"{item['reference_query_ms']['p95']:.3f} | "
            f"{item['prepared_query_ms']['p95']:.3f} |"
        )
    lines += [
        "", "enabled 旧轨迹的保存状态在此高预算根区间诊断中全部确定代理安全；"
        "baseline 则有大量确定低于门槛的旧状态，继续增加查询预算无法使同一"
        "状态的真实代理净空变大。代理分类不等于 MuJoCo 实际碰撞。"
        "这些是冷查询计时，不含 MuJoCo 正运动学、广义 Jacobian、原约束、"
        "QP 和力矩执行；也没有解决旧速度起点违例或证明新模式全链 20 ms。", "",
        "## 冻结留出与真实几何", "",
        f"旧 B.1 独立留出 {heldout['counts']['checked_count']} 例完成新补的 MuJoCo 离散链对照："
        f"代理假安全 {heldout['counts']['empirical_proxy_false_safe']}、"
        f"代理误拒绝 {heldout['counts']['empirical_proxy_false_reject']}。",
        f"新冻结的 5 mm 附近 {near['counts']['checked_count']} 例分布于门槛两侧："
        f"代理确定低于门槛 {near['counts'].get('PROXY_CLEARANCE_BELOW_GATE', 0)}、"
        f"确定达到门槛 {near['counts'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)}、"
        f"未知 {near['counts'].get('UNKNOWN_CROSSES_GATE', 0)}；"
        f"独立 MuJoCo 对照中的代理假安全 {near['counts']['empirical_proxy_false_safe']}。",
        "样本由 B.1 区间查询定位到代理门槛附近，冻结后才计算真实几何；"
        "密采样或 B.1 区间中点均不被当作连续真值。", "",
        "## 证据边界与下一动作", "",
        "`interval_well_formed`、完整覆盖、精确模型下 K=1 假设、双精度认证状态、"
        "实际链子空间残差与有限样本包络证据仍分开报告。"
        "当前在线接入门禁未通过，应继续处理持久分区未知、预算与保守性；"
        "不接入新区间 QP，也不引用旧 trace 冒充新控制运行。", "",
    ]
    doc = output_dir / "STAGE2_EVIDENCE.md"
    with doc.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_stage2_hash_manifest_v1",
        "status": status,
        "gate_blockers": blockers,
        "sources": {name: {"path": path.as_posix(), "sha256": _sha(path),
                           "bytes": path.stat().st_size}
                    for name, path in sources.items()},
        "generated_document": {"path": doc.as_posix(),
                               "sha256": _sha(doc), "bytes": doc.stat().st_size},
    }
    with (output_dir / "stage2_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.root, args.output_dir)
    print(json.dumps({"status": result["status"],
                      "document": result["generated_document"]}, indent=2))


if __name__ == "__main__":
    main()
