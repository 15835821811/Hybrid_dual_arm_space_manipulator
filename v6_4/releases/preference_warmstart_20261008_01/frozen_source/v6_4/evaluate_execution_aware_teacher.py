"""Reduce the frozen B.3.1 slots using saved files only; never execute a model.

The teacher is a finite development-set oracle, not a learned policy. Failures
remain in success-rate denominators and never receive an invented scalar cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("quality_matrix.json", "teacher_records.json", "report.json", "REPORT.md", "all_candidates.csv")
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return value is not None and np.isfinite(float(value))


def vector_stats(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim in (1, 2) and np.isfinite(values).all(), "invalid saved diagnostic array")
    norm = np.linalg.norm(values, axis=1) if values.ndim == 2 else values
    return {"samples": len(norm), "rms": float(np.sqrt(np.mean(norm**2))) if len(norm) else None,
            "peak": float(np.max(norm)) if len(norm) else None,
            "mean": float(np.mean(norm)) if len(norm) else None}


def window(times, interval):
    return (times >= interval[0]-1e-10) & (times <= interval[1]+1e-10)


def rank_candidates(rows, band):
    """Non-transitive groups anchored at their minimum; fixed within-group tie break."""
    require(finite(band) and band >= 0., "invalid frozen tie band")
    pending = sorted((row for row in rows if row["eligible"]),
        key=lambda row: (row["I_route_rad_s"], row["mode"]))
    for row in pending:
        require(finite(row["I_route_rad_s"]) and row["I_route_rad_s"] >= 0., "invalid eligible route cost")
        require(finite(row["continuum_path_length_m"]) and finite(row["coefficient_norm_m"]), "invalid tie-break fields")
    groups = []
    while pending:
        minimum = pending[0]["I_route_rad_s"]
        group = [row for row in pending if row["I_route_rad_s"] <= minimum+band]
        ids = {row["slot_id"] for row in group}
        pending = [row for row in pending if row["slot_id"] not in ids]
        ordered = sorted(group, key=lambda row: (row["continuum_path_length_m"], row["coefficient_norm_m"], row["mode"]))
        groups.append({"anchor_minimum_I_route_rad_s": minimum,
            "indistinguishable_within_frozen_resolution": len(group) > 1,
            "slot_ids_cost_order": [row["slot_id"] for row in sorted(group, key=lambda row: (row["I_route_rad_s"], row["mode"]))],
            "slot_ids_tie_break_order": [row["slot_id"] for row in ordered]})
    minimum_row = min((row for row in rows if row["eligible"]),
        key=lambda row: (row["I_route_rad_s"], row["mode"]), default=None)
    return {"groups": groups, "minimum_cost_slot_id": minimum_row["slot_id"] if minimum_row else None,
        "minimum_I_route_rad_s": minimum_row["I_route_rad_s"] if minimum_row else None,
        "tie_selected_slot_id": groups[0]["slot_ids_tie_break_order"][0] if groups else None,
        "ranking_slot_ids": [slot for group in groups for slot in group["slot_ids_tie_break_order"]]}


def conditional_value(rows, task_ids, modes):
    """Only a rectangular matrix of complete, eligible results gets a scalar V_cond."""
    lookup = {(row["task_id"], row["mode"]): row for row in rows}
    require(len(lookup) == len(rows), "duplicate task/mode cells")
    complete = [task for task in task_ids if all(lookup.get((task, mode), {}).get("eligible") is True for mode in modes)]
    def evaluate(subset):
        if not subset:
            return {"status": "UNAVAILABLE_NO_COMMON_COMPLETE_TASKS", "task_ids": [], "V_cond_rad_s": None}
        matrix = np.asarray([[lookup[task, mode]["I_route_rad_s"] for mode in modes] for task in subset])
        means = matrix.mean(axis=0)
        best = min(range(len(modes)), key=lambda index: (means[index], modes[index]))
        oracle = float(matrix.min(axis=1).mean())
        return {"status": "AVAILABLE", "task_ids": subset, "task_count": len(subset),
            "constant_mean_costs_rad_s": dict(zip(modes, map(float, means))),
            "best_constant_mode": modes[best], "best_constant_mean_I_route_rad_s": float(means[best]),
            "mean_taskwise_minimum_I_route_rad_s": oracle, "V_cond_rad_s": float(means[best]-oracle),
            "failure_cost_imputation": False}
    common = evaluate(complete)
    full = evaluate(task_ids) if len(complete) == len(task_ids) else {
        "status": "UNAVAILABLE_INCOMPLETE_FULL_MATRIX", "task_ids": task_ids, "V_cond_rad_s": None,
        "incomplete_task_ids": [task for task in task_ids if task not in complete], "failure_cost_imputation": False}
    return {"formula": "min_z mean_c J(z,c) - mean_c min_z J(z,c)", "modes": modes,
        "full_development_matrix": full, "common_complete_task_subset": common,
        "scope": "finite fixed candidate set on development tasks, with one mother scene; offline conditional selection potential only; not a policy, continuous optimum, generalization TEST, or statistical guarantee"}


class Inputs:
    def __init__(self):
        self.files = {}
        self.state_cache = {}
    def bind(self, path, expected=None):
        path = Path(path).resolve()
        if str(path) not in self.files:
            self.files[str(path)] = sha(path)
        if expected is not None:
            require(self.files[str(path)] == expected, "source digest differs: "+str(path))
        return path
    def json(self, path, expected=None):
        return read(self.bind(path, expected))
    def unchanged(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, "input changed during pure-file reduction: "+path)


def guard(identity_path, inputs):
    identity = inputs.json(identity_path)
    require(identity.get("schema") == "v64_b31_source_identity_v1", "wrong B.3.1 source identity")
    for relative, digest in identity["source_sha256"].items():
        inputs.bind(ROOT/relative, digest)
    for path, digest in identity["protected_artifacts"].items():
        inputs.bind(path, digest)
    return identity


def array(trace, key, count, width=None):
    value = np.asarray(trace[key], dtype=float)[:count]
    shape = (count,) if width is None else (count, width)
    require(value.shape == shape and np.isfinite(value).all(), "invalid consumed saved field: "+key)
    return value


def obstacle_activity(encoded_rows, times, interval, period):
    active, binding, lookahead = [], [], []
    sources, residual, distances = Counter(), [], []
    per_source = {}
    for index, encoded in enumerate(encoded_rows):
        rows = json.loads(str(encoded))
        require(isinstance(rows, list), "invalid saved obstacle-row ledger")
        flags = {"active": False, "selected_binding": False, "selected_lookahead_binding": False}
        source_flags = {}
        for row in rows:
            source = row["source"]
            tick_flags = source_flags.setdefault(source, {key: False for key in flags})
            for key in flags:
                yes = row.get(key) is True
                if key != "active": yes = yes and row.get("active") is True
                flags[key] |= yes
                tick_flags[key] |= yes
            if row.get("active") is True:
                sources[source] += 1
                if finite(row.get("selected_residual_m_s")): residual.append(float(row["selected_residual_m_s"]))
                if finite(row.get("distance_m")): distances.append(float(row["distance_m"]))
        for source, tick_flags in source_flags.items():
            state = per_source.setdefault(source, {"active_ticks": 0, "selected_binding_ticks": 0,
                "selected_lookahead_binding_ticks": 0, "route_active_ticks": 0,
                "route_selected_binding_ticks": 0, "route_selected_lookahead_binding_ticks": 0})
            in_route = interval[0]-1e-10 <= times[index] <= interval[1]+1e-10
            for key, yes in tick_flags.items():
                state[key+"_ticks"] += int(yes)
                state["route_"+key+"_ticks"] += int(yes and in_route)
        active.append(flags["active"]); binding.append(flags["selected_binding"]); lookahead.append(flags["selected_lookahead_binding"])
    route = window(times, interval)
    def durations(mask):
        mask = np.asarray(mask, dtype=bool)
        return {"full_consumed_duration_s": float(mask.sum()*period),
            "route_consumed_duration_s": float(mask[route].sum()*period), "full_consumed_ticks": int(mask.sum()),
            "route_consumed_ticks": int(mask[route].sum())}
    for state in per_source.values():
        for key, count in list(state.items()): state[key.replace("_ticks", "_duration_s")] = count*period
    return {"status": "AVAILABLE", "active_any_related_row": durations(active),
        "selected_binding_any_related_row": durations(binding),
        "selected_lookahead_binding_any_related_row": durations(lookahead), "per_source": per_source,
        "active_row_source_counts": dict(sources),
        "minimum_existing_selected_residual_m_s": min(residual) if residual else None,
        "minimum_existing_active_row_distance_m": min(distances) if distances else None,
        "binding_threshold_m_s": 2e-5,
        "scope": "union of existing related-sphere QP row flags over consumed 20ms commands; not a single-obstacle causal attribution; inactive rows have no distance witness; no new query"}


def diagnostic_summary(record, slot, interval, inputs):
    result = record["source_result"]
    if slot["old_source_slot"]:
        return {"status": "NOT_MEASURED_OLD_EVIDENCE", "source_vectors_available": False,
            "reason": "old I_route is a scalar norm; no 10D/7D vectors or related-sphere row activity are reconstructed"}
    count = int(result.get("actual_steps", 0))
    if not count:
        return {"status": "NOT_MEASURED_NO_CONSUMED_TRACE", "source_vectors_available": False}
    require(count % 10 == 0, "saved actual prefix is not complete ten-step ramps")
    ticks = count//10
    trace_path = inputs.bind(result["trace_path"], result["trace_sha256"])
    with np.load(trace_path, allow_pickle=False) as trace:
        times = array(trace, "task_time", ticks)
        route = window(times, interval)
        nominal = array(trace, "task_qp_box_nominal_velocity", ticks, 17)
        selected = array(trace, "task_qp_selected_velocity", ticks, 17)
        raw = array(trace, "task_qp_raw_unconstrained_velocity", ticks, 17)
        frozen = inputs.json(slot["plan_path"], slot["plan_file_sha256"])
        for key, expected in (("task_consumed_reference_version", frozen["representation_version"]),
                ("task_consumed_reference_plan_sha256", slot["plan_sha256"]),
                ("task_consumed_reference_definition_sha256", frozen["definition"]["definition_sha256"])):
            observed = np.asarray(trace[key])[:ticks]
            require(observed.shape == (ticks,) and np.all(observed == expected), "saved consumer identity differs: "+key)
        z = np.asarray(trace["task_consumed_reference_z_m"], dtype=float)[:ticks]
        require(z.shape == (ticks, 6, 2) and np.array_equal(z, np.broadcast_to(np.asarray(frozen["z_m"]), z.shape)), "consumed residual coefficients differ")
        for name, value in (("box_nominal_velocity", nominal), ("selected_velocity", selected), ("raw_unconstrained_velocity", raw)):
            require(np.array_equal(array(trace, "task_qp_"+name+"_continuum", ticks, 10), value[:, :10])
                and np.array_equal(array(trace, "task_qp_"+name+"_rigid", ticks, 7), value[:, 10:]), "saved 10D/7D split differs")
        intervention = selected-nominal
        norm = np.linalg.norm(intervention, axis=1)
        saved_norm = array(trace, "task_avoidance_intervention", ticks)
        require(np.allclose(norm, saved_norm, atol=1e-12, rtol=0.), "new vectors differ from original intervention scalar")
        values = {"I_full_17D_rad_s": vector_stats(intervention)["rms"],
            "I_route_17D_rad_s": vector_stats(intervention[route])["rms"],
            "I_full_continuum_10D_rad_s": vector_stats(intervention[:, :10])["rms"],
            "I_route_continuum_10D_rad_s": vector_stats(intervention[route, :10])["rms"],
            "I_full_rigid_7D_rad_s": vector_stats(intervention[:, 10:])["rms"],
            "I_route_rigid_7D_rad_s": vector_stats(intervention[route, 10:])["rms"],
            "box_clip_velocity_change_rad_s": vector_stats(nominal-raw)}
        require(np.isclose(values["I_full_17D_rad_s"]**2,
            values["I_full_continuum_10D_rad_s"]**2+values["I_full_rigid_7D_rad_s"]**2, atol=1e-14, rtol=0.), "component RMS identity failed")
        if record["quality"].get("full_metrics"):
            require(np.isclose(values["I_route_17D_rad_s"], record["quality"]["full_metrics"]["I_route_rad_s"], atol=1e-12, rtol=0.), "vector I_route differs from qualified quality")
        continuum = {}
        for key in ("target_position_m", "actual_position_m", "position_error_m", "reference_velocity_m_s",
                    "desired_velocity_m_s", "actual_velocity_m_s", "selected_task_velocity_m_s"):
            value = array(trace, "task_qp_continuum_"+key, ticks, 3)
            continuum[key] = {"full": vector_stats(value), "route": vector_stats(value[route])}
        desired = array(trace, "task_qp_continuum_desired_velocity_m_s", ticks, 3)
        actual = array(trace, "task_qp_continuum_actual_velocity_m_s", ticks, 3)
        continuum["desired_minus_actual_velocity_m_s"] = {"full": vector_stats(desired-actual), "route": vector_stats((desired-actual)[route])}
        saturation = array(trace, "torque_saturation_count", count)
        torque = array(trace, "torque_unclipped_max_nm", count)
        native_times = array(trace, "time", count)
        native_route = window(native_times, interval)
        torque_summary = {"saturated_actuator_step_count_full": int(saturation.sum()),
            "saturated_actuator_step_count_route": int(saturation[native_route].sum()),
            "any_saturation_native_steps_full": int((saturation > 0).sum()),
            "any_saturation_duration_s_full": float((saturation > 0).sum()*.002),
            "required_torque_abs_peak_nm_full": float(torque.max()),
            "required_torque_abs_peak_nm_route": float(torque[native_route].max()) if native_route.any() else None}
        activity = obstacle_activity(np.asarray(trace["task_qp_obstacle_rows_json"])[:ticks], times, interval, .020)
        require(len(np.asarray(trace["task_qp_obstacle_rows_json"])[:ticks]) == ticks, "missing consumed obstacle rows")
    return {"status": "AVAILABLE_FULL" if record["full_task_success"] else "AVAILABLE_FAILED_PREFIX_ONLY",
        "source_vectors_available": True, "source_trace": str(trace_path), "consumed_ticks": ticks,
        "sample_scope": "only actually consumed prefix; rejected/unconsumed proposed rows excluded",
        "intervention": values, "continuum_task": continuum, "related_sphere_rows": activity,
        "torque": torque_summary, "selected_task_velocity_is_actual_velocity": False,
        "new_qp_solves": 0, "new_geometry_queries": 0}


def saved_state(record, inputs):
    if not (record["source_result"].get("full_task_success") is True
            and record["quality"].get("safety", {}).get("full_task_and_original_safety_passed") is True):
        return None
    evaluation = record["source_result"].get("evaluation_path")
    require(evaluation is not None, "qualified slot has no independent evaluation")
    report = inputs.json(evaluation, record["source_result"]["evaluation_sha256"])
    require(report == record["source_result"]["evaluation"], "embedded independent evaluation differs from bound report")
    directory = Path(evaluation).resolve().parent
    manifest_path = directory/"manifest.json"
    manifest = inputs.json(manifest_path, record["evidence_bindings"].get(str(manifest_path)))
    require(manifest.get("report.json") == record["source_result"]["evaluation_sha256"], "evaluation manifest differs from bound report")
    require(manifest.get("fresh_replay.npz") == report["fresh_replay_sha256"], "fresh replay manifest differs from original bound evaluation digest")
    for name, digest in manifest.items():
        artifact = (directory/name).resolve()
        require(artifact.is_relative_to(directory), "evaluation manifest path escapes producer directory")
        inputs.bind(artifact, digest)
    path = inputs.bind(directory/"fresh_replay.npz", report["fresh_replay_sha256"])
    if str(path) not in inputs.state_cache:
        with np.load(path, allow_pickle=False) as state:
            times = np.asarray(state["time"], dtype=float)
            positions = np.asarray(state["continuum_position"], dtype=float)
        times.setflags(write=False); positions.setflags(write=False)
        inputs.state_cache[str(path)] = (times, positions)
    times, positions = inputs.state_cache[str(path)]
    require(times.shape == (13501,) and positions.shape == (13501, 3)
        and np.isfinite(times).all() and np.isfinite(positions).all(), "invalid complete saved response state")
    return times, positions, str(path)


def quality_path_for_slot(slot, record):
    if slot["old_source_slot"]:
        return Path(slot["old_quality_path"])
    name = "route_quality_failure.json" if record["quality"].get("quality_stage_status") == "QUALITY_POSTPROCESS_FAILED" else "route_quality.json"
    path = Path(slot["plan_path"]).parents[2]/"quality"/slot["slot_id"]/name
    require(str(path) in record["evidence_bindings"], "terminal record does not bind its stated quality stage artifact")
    return path


def route_minimum_saved_time(record, slot, interval, inputs):
    if not record["quality"].get("quality_label_eligible"):
        return {"status": "UNAVAILABLE_NO_COMPLETE_QUALIFIED_ROUTE_CLEARANCE"}
    directory = quality_path_for_slot(slot, record).parent
    manifest = inputs.json(directory/"manifest.json")
    inputs.bind(directory/"route_quality.json", manifest["route_quality.json"])
    path = inputs.bind(directory/"route_clearance.npz", manifest["route_clearance.npz"])
    with np.load(path, allow_pickle=False) as state:
        times = np.asarray(state["time"], dtype=float)
        minima = np.asarray(state["continuum_obstacle_minimum_m"], dtype=float)
    require(times.shape == (13501,) and minima.shape == times.shape and np.isfinite(times).all()
        and np.isfinite(minima).all(), "invalid saved local-clearance series")
    selected = np.flatnonzero(window(times, interval))
    require(len(selected) > 0, "route clearance has no frozen-window samples")
    index = int(selected[np.argmin(minima[selected])])
    expected = record["quality"]["full_metrics"]["continuum_route_obstacle_clearance"]["route_window_minimum_m"]
    require(np.isclose(minima[index], expected, atol=1e-12, rtol=0.), "saved route clearance minimum differs from original quality")
    return {"status": "AVAILABLE_SAVED_SERIES", "time_s": float(times[index]), "state_index": index,
        "minimum_m": float(minima[index]), "saved_series_source": str(path),
        "pair_witness": "NOT_SAVED_FOR_ROUTE_WINDOW_MINIMUM",
        "global_full_horizon_witness_used_as_route_witness": False, "new_geometry_queries": 0}


def response_summary(record, slot, zero_record, task_info, inputs):
    """Frozen analytic reference evaluated on saved times, never a new rollout."""
    from .task_protocol import TaskSpec
    from .task_anchored_reference import TaskAnchoredResidualPlan, _target, _base_kinematics
    state = saved_state(record, inputs)
    if state is None:
        return {"status": "UNAVAILABLE_INCOMPLETE_OR_UNQUALIFIED", "complete_comparable_response": False}
    times, actual, source = state
    task = TaskSpec.from_dict(inputs.json(slot["task_path"], task_info["task_file_sha256"]))
    residual = TaskAnchoredResidualPlan.from_dict(inputs.json(slot["plan_path"], slot["plan_file_sha256"]))
    interval = task_info["route_interval_s"]
    route = window(times, interval)
    basis = np.asarray(residual.definition["transverse_bases"][2], dtype=float)
    base = _base_kinematics(_target(task), times)[0]
    offset = residual.offset_kinematics(times)[0]
    actual_offset = actual-base
    stats = {"reference_minus_base_m": vector_stats(offset[route]),
        "actual_minus_base_m": vector_stats(actual_offset[route]),
        "actual_minus_reference_m": vector_stats((actual_offset-offset)[route])}
    full_stats = {"reference_minus_base_m": vector_stats(offset),
        "actual_minus_base_m": vector_stats(actual_offset),
        "actual_minus_reference_m": vector_stats(actual_offset-offset)}
    zero_state = saved_state(zero_record, inputs)
    difference, zero_source = None, None
    if zero_state is not None:
        zero_times, zero_positions, zero_source = zero_state
        require(np.array_equal(times, zero_times), "same-task zero reference response clocks differ; no interpolation permitted")
        difference = actual-zero_positions
        stats["actual_minus_same_task_zero_actual_m"] = vector_stats(difference[route])
        full_stats["actual_minus_same_task_zero_actual_m"] = vector_stats(difference)
    else:
        stats["actual_minus_same_task_zero_actual_m"] = {"status": "UNAVAILABLE_ZERO_NOT_COMPLETE_QUALIFIED"}
        full_stats["actual_minus_same_task_zero_actual_m"] = stats["actual_minus_same_task_zero_actual_m"]
    indices = np.flatnonzero(route)
    selected = np.unique(np.r_[indices[::10], indices[-1]])
    midpoint = int(np.argmin(np.abs(times-float(np.mean(interval)))))
    return {"status": "AVAILABLE", "complete_comparable_response": zero_state is not None,
        "saved_state_source": source, "same_task_zero_saved_source": zero_source,
        "route_interval_s": interval, "native_grid_metrics": stats, "full_horizon_native_grid_metrics": full_stats,
        "native_grid_metrics_scope": "complete frozen route interval; full 27s metrics separately preserved",
        "reference_evaluation": "frozen declared analytic p on exact saved 2ms times; zero geometry, zero physics; fixed world E",
        "route_curve": {"source_native_indices": selected.tolist(), "times_s": times[selected].tolist(),
            "reference_minus_base_transverse_m": (offset[selected]@basis).tolist(),
            "actual_minus_base_transverse_m": (actual_offset[selected]@basis).tolist(),
            "actual_minus_zero_actual_transverse_m": (difference[selected]@basis).tolist() if difference is not None else None,
            "thinning": "every tenth existing route-native row plus last; metrics use every native row; no interpolation"},
        "nearest_saved_midpoint_slice": {"time_s": float(times[midpoint]),
            "reference_minus_base_transverse_m": (offset[midpoint]@basis).tolist(),
            "actual_minus_base_transverse_m": (actual_offset[midpoint]@basis).tolist(),
            "actual_minus_zero_actual_transverse_m": (difference[midpoint]@basis).tolist() if difference is not None else None},
        "scope": "descriptive whole-window closed-loop response and exact same-task zero comparison; no identified controller transfer gain or obstacle-specific causal claim"}


def comparison(candidate, baseline):
    if not candidate["eligible"] or not baseline["eligible"]:
        return {"status": "UNAVAILABLE_NOT_BOTH_COMPLETE_SAFE", "candidate_slot": candidate["slot_id"],
            "baseline_slot": baseline["slot_id"], "failure_cost_imputed": False}
    reduction = baseline["I_route_rad_s"]-candidate["I_route_rad_s"]
    fields = ("continuum_path_length_m", "route_clearance_m", "base_translation_peak_m", "base_rotation_peak_rad")
    new_vectors = {"status": "NOT_MEASURED_BOTH_CANDIDATES_WITH_NEW_VECTORS"}
    a, b = candidate.get("new_diagnostics", {}), baseline.get("new_diagnostics", {})
    if a.get("source_vectors_available") and b.get("source_vectors_available"):
        new_vectors = {"status": "AVAILABLE", "candidate_minus_baseline": {
            "torque_saturated_actuator_step_count_full": a["torque"]["saturated_actuator_step_count_full"]-b["torque"]["saturated_actuator_step_count_full"],
            "continuum_desired_minus_actual_velocity_RMS_m_s": a["continuum_task"]["desired_minus_actual_velocity_m_s"]["full"]["rms"]-b["continuum_task"]["desired_minus_actual_velocity_m_s"]["full"]["rms"],
            "I_route_continuum_10D_rad_s": a["intervention"]["I_route_continuum_10D_rad_s"]-b["intervention"]["I_route_continuum_10D_rad_s"],
            "I_route_rigid_7D_rad_s": a["intervention"]["I_route_rigid_7D_rad_s"]-b["intervention"]["I_route_rigid_7D_rad_s"]}}
    return {"status": "AVAILABLE", "candidate_slot": candidate["slot_id"], "baseline_slot": baseline["slot_id"],
        "I_route_reduction_rad_s": reduction,
        "I_route_relative_reduction": reduction/baseline["I_route_rad_s"] if baseline["I_route_rad_s"] > 0. else None,
        "candidate_minus_baseline": {key: candidate[key]-baseline[key] for key in fields},
        "additional_new_vector_cost_differences": new_vectors,
        "cost_conflicts": {"longer_path": candidate["continuum_path_length_m"] > baseline["continuum_path_length_m"],
            "smaller_related_sphere_clearance": candidate["route_clearance_m"] < baseline["route_clearance_m"],
            "larger_base_translation": candidate["base_translation_peak_m"] > baseline["base_translation_peak_m"],
            "larger_base_rotation": candidate["base_rotation_peak_rad"] > baseline["base_rotation_peak_rad"]}}


def old_threshold_checks(rows, plan):
    by_mode = {row["mode"]: row for row in rows}
    gates = plan["teacher"]["old_B3_gate"]
    checks = []
    for plus, minus in (("v1_plus12", "v1_minus12"), ("v2_plus12", "v2_minus12"), ("v2_plus20", "v2_minus20")):
        for mode, opposite in ((plus, minus), (minus, plus)):
            row = by_mode[mode]
            comparisons = []
            for baseline in ("z0", opposite):
                comp = comparison(row, by_mode[baseline])
                passes = comp["status"] == "AVAILABLE" and comp["I_route_reduction_rad_s"] >= gates["absolute_reduction_rad_s"]-1e-12 \
                    and comp["I_route_relative_reduction"] is not None and comp["I_route_relative_reduction"] >= gates["relative_reduction"]-1e-12
                metric_unavailable = by_mode[baseline]["full_task_and_original_safety_passed"] and not by_mode[baseline]["eligible"]
                comparisons.append({"comparator_mode": baseline, "quality_threshold_passed": bool(passes),
                    "complete_safe_vs_failure": row["eligible"] and not by_mode[baseline]["eligible"] and not metric_unavailable,
                    "comparator_metric_unavailable_after_safe_actual": metric_unavailable, **comp})
            checks.append({"mode": mode, "comparison_evidence": comparisons,
                "legacy_OR_advantage_vs_zero_or_opposite": any(c["quality_threshold_passed"] or c["complete_safe_vs_failure"] for c in comparisons),
                "stricter_supplemental_AND_quality_threshold_passed": all(c["quality_threshold_passed"] for c in comparisons)})
    return {"thresholds": gates, "candidates": checks,
        "favorable_comparator_semantics": "zero OR opposite; report both, then original preferred-direction and cross-side reversal checks",
        "old_B3_sealed_result_changed": False, "scope": "same predeclared thresholds reported on this supplementary development set; AND is explicitly a stricter supplementary summary; cannot rewrite the old B.3 decision"}


def original_gate_groups(rows):
    from .route_quality_dataset import pilot_discriminability
    groups = []
    for distance in (.043, .055):
        for family, plus, minus in (("v1_12mm", "v1_plus12", "v1_minus12"),
                ("v2_12mm", "v2_plus12", "v2_minus12"), ("v2_20mm", "v2_plus20", "v2_minus20")):
            chosen = [row for row in rows if row["distance_m"] == distance and row["mode"] in ("z0", plus, minus)]
            task_ids = list(dict.fromkeys(row["task_id"] for row in chosen))
            normalized = [{"slot_id": row["slot_id"], "task_id": row["task_id"],
                "candidate_name": {"z0": "z0", plus: "z+", minus: "z-"}[row["mode"]],
                "quality_label_eligible": row["eligible"],
                "full_metrics": {"I_route_rad_s": row["I_route_rad_s"]} if row["eligible"] else None}
                for row in chosen]
            original = pilot_discriminability(normalized, task_ids)
            unavailable = [row["slot_id"] for row in chosen if row["full_task_and_original_safety_passed"] and not row["eligible"]]
            groups.append({"distance_m": distance, "reference_family": family,
                "status": "UNAVAILABLE_QUALITY_METRIC_AFTER_SAFE_ACTUAL" if unavailable else "AVAILABLE",
                "route_value_identifiable_under_original_rule": None if unavailable else original["route_value_identifiable"],
                "metric_unavailable_slot_ids": unavailable, "exact_original_function_result": original,
                "unavailable_scope": "original valid=false would label missing metric as failure; raw function retained for fidelity but no physical-failure advantage is inferred" if unavailable else None,
                "raw_original_function_training_authorized_field_is_historical_only": True, "B31_training_authorized": False})
    original_six = next(group for group in groups if group["distance_m"] == .043 and group["reference_family"] == "v1_12mm")
    require(original_six["route_value_identifiable_under_original_rule"] is False, "old six-slot decision changed unexpectedly")
    return groups


def task_teachers(rows, plan):
    result = []
    band = plan["teacher"]["tie_band_absolute_rad_s"]
    for task in plan["tasks"]:
        cells = [row for row in rows if row["task_id"] == task["task_id"]]
        by_mode = {row["mode"]: row for row in cells}
        ranking = rank_candidates(cells, band)
        minimum = next((row for row in cells if row["slot_id"] == ranking["minimum_cost_slot_id"]), None)
        chosen = next((row for row in cells if row["slot_id"] == ranking["tie_selected_slot_id"]), None)
        positive = min((row for row in cells if row["eligible"] and row["amplitude_m"] > 0.), key=lambda row: (row["I_route_rad_s"], row["mode"]), default=None)
        negative = min((row for row in cells if row["eligible"] and row["amplitude_m"] < 0.), key=lambda row: (row["I_route_rad_s"], row["mode"]), default=None)
        geometric = by_mode[plan["teacher"]["geometric_rule_side_mapping"]["+1" if task["side"] == 1 else "-1"]]
        summary = {"task_id": task["task_id"], "distance_m": task["distance_m"], "side": task["side"],
            "attempt_slots": len(cells), "complete_safe_count": sum(row["full_task_and_original_safety_passed"] for row in cells),
            "quality_comparable_count": sum(row["eligible"] for row in cells),
            "failed_or_unqualified_slot_ids": [row["slot_id"] for row in cells if not row["eligible"]],
            **ranking, "tie_selected_I_route_rad_s": chosen["I_route_rad_s"] if chosen else None,
            "tie_selected_excess_over_minimum_rad_s": chosen["I_route_rad_s"]-minimum["I_route_rad_s"] if chosen else None,
            "zero_slot_id": by_mode["z0"]["slot_id"], "geometric_rule_slot_id": geometric["slot_id"],
            "best_positive_in_finite_set_slot_id": positive["slot_id"] if positive else None,
            "best_negative_in_finite_set_slot_id": negative["slot_id"] if negative else None,
            "positive_negative_selection_scope": "task-local finite-set minima, not a preimplemented constant policy",
            "old_B3_threshold_checks": old_threshold_checks(cells, plan),
            "same_z_v2_vs_v1": [comparison(by_mode["v2_"+direction+"12"], by_mode["v1_"+direction+"12"]) for direction in ("plus", "minus")],
            "teacher_minimum_vs_zero": comparison(minimum, by_mode["z0"]) if minimum else None,
            "teacher_minimum_vs_geometric": comparison(minimum, geometric) if minimum else None,
            "teacher_minimum_vs_best_positive": comparison(minimum, positive) if minimum and positive else None,
            "teacher_minimum_vs_best_negative": comparison(minimum, negative) if minimum and negative else None,
            "tie_selected_vs_minimum": comparison(chosen, minimum) if chosen else None}
        result.append(summary)
    return result


def policy_summary(rows, plan, teacher_tasks, value):
    modes = plan["teacher"]["constant_policy_candidate_set"]
    tasks = [task["task_id"] for task in plan["tasks"]]
    lookup = {(row["task_id"], row["mode"]): row for row in rows}
    policies = []
    selected = {task["task_id"]: task for task in teacher_tasks}
    common_ids = value["common_complete_task_subset"]["task_ids"]
    for name in [*modes, "geometric_rule_v2_away12", "finite_teacher_minimum", "finite_teacher_tie_selected"]:
        choices = []
        for task in tasks:
            key = {"geometric_rule_v2_away12": "geometric_rule_slot_id", "finite_teacher_minimum": "minimum_cost_slot_id", "finite_teacher_tie_selected": "tie_selected_slot_id"}.get(name)
            row = next((row for row in rows if row["slot_id"] == selected[task][key]), None) if key else lookup[task, name]
            choices.append(row)
        complete = [row for row in choices if row is not None and row["eligible"]]
        successful = [row for row in choices if row is not None and row["full_task_and_original_safety_passed"]]
        common_choices = [row for task, row in zip(tasks, choices) if task in common_ids]
        policies.append({"policy": name, "successful_safe_tasks": len(successful), "total_tasks": len(tasks),
            "success_rate": len(successful)/len(tasks), "quality_comparable_tasks": len(complete),
            "selected_slot_ids": [row["slot_id"] if row else None for row in choices],
            "all_tasks_mean_I_route_rad_s": float(np.mean([row["I_route_rad_s"] for row in complete])) if len(complete) == len(tasks) else None,
            "successful_subset_mean_I_route_rad_s": float(np.mean([row["I_route_rad_s"] for row in complete])) if complete else None,
            "successful_subset_task_ids": [row["task_id"] for row in complete],
            "successful_subset_mean_is_cross_policy_comparable": len(complete) == len(tasks),
            "common_complete_task_subset_task_ids": common_ids,
            "common_complete_task_subset_mean_I_route_rad_s": float(np.mean([row["I_route_rad_s"] for row in common_choices])) if common_choices else None,
            "failure_cost_imputed": False})
    common = value["common_complete_task_subset"]
    constants = common.get("constant_mean_costs_rad_s", {})
    best_signed = {}
    for name, token in (("positive", "plus"), ("negative", "minus")):
        applicable = [(cost, mode) for mode, cost in constants.items() if token in mode]
        best = min(applicable, default=None)
        best_signed[name] = {"mode": best[1], "mean_I_route_rad_s": best[0], "task_ids": common["task_ids"]} if best else {"status": "UNAVAILABLE_NO_COMMON_COMPLETE_TASKS"}
    common_means = {row["policy"]: row["common_complete_task_subset_mean_I_route_rad_s"] for row in policies}
    best_constant = common.get("best_constant_mean_I_route_rad_s")
    same_subset = {"task_ids": common_ids, "best_constant_mode": common.get("best_constant_mode"),
        "best_constant_mean_I_route_rad_s": best_constant,
        "zero_mean_I_route_rad_s": common_means["z0"],
        "geometric_rule_mean_I_route_rad_s": common_means["geometric_rule_v2_away12"],
        "finite_teacher_minimum_mean_I_route_rad_s": common_means["finite_teacher_minimum"],
        "finite_teacher_tie_selected_mean_I_route_rad_s": common_means["finite_teacher_tie_selected"],
        "geometric_rule_minus_teacher_minimum_rad_s": common_means["geometric_rule_v2_away12"]-common_means["finite_teacher_minimum"] if common_ids else None,
        "best_constant_minus_teacher_tie_selected_rad_s": best_constant-common_means["finite_teacher_tie_selected"] if common_ids else None,
        "all_means_use_identical_tasks": True, "failure_cost_imputed": False}
    return {"policies": policies, "common_complete_subset_direct_comparisons": same_subset,
        "best_constant_signed_modes_on_common_complete_subset": best_signed,
        "best_constant_among_all_seven": value["full_development_matrix"].get("best_constant_mode"),
        "selection_scope": "constant policies use one mode for every task; finite teacher is a retrospective development-set oracle; geometric rule is frozen and performs no outcome selection"}


def flatten_record(record, slot, task_info, inputs):
    require(record.get("schema") == "v64_b31_fixed_slot_record_v1" and record["frozen_slot"] == slot,
        "terminal slot differs from frozen declaration")
    require(record["slot_id"] == slot["slot_id"], "slot alias mismatch")
    for path, digest in record["evidence_bindings"].items(): inputs.bind(path, digest)
    source, quality = record["source_result"], record["quality"]
    attempted = Path(slot["old_attempt_path"])/"attempt_result.json" if slot["old_source_slot"] else Path(slot["plan_path"]).parents[2]/"attempts"/slot["slot_id"]/"attempt_result.json"
    require(inputs.json(attempted, record["evidence_bindings"][str(attempted)]) == source, "embedded attempt result differs from original bound file")
    quality_path = quality_path_for_slot(slot, record)
    require(inputs.json(quality_path, record["evidence_bindings"][str(quality_path)]) == quality, "embedded quality differs from original bound file")
    source_slot = slot["old_source_slot"] or slot["slot_id"]
    require(source["slot_id"] == quality["slot_id"] == source_slot, "source slot identity mismatch")
    require(source["task_sha256"] == quality["task_sha256"] == slot["task_sha256"], "source Task identity mismatch")
    require(record["status"] == source["status"] == quality["status"], "source status mismatch")
    require(record["full_task_success"] == source["full_task_success"], "source success mismatch")
    require(quality["route_intervals_s"] == [task_info["route_interval_s"]], "quality used a different route window")
    metrics = quality.get("full_metrics")
    full_safe = bool(source["full_task_success"] is True and source["actual_steps"] == 13500
        and quality["safety"]["full_task_and_original_safety_passed"] is True
        and all(quality["safety"]["gates"].get(name) is True for name in GATES))
    eligible = full_safe and quality.get("quality_label_eligible") is True and metrics is not None
    require(not quality.get("quality_label_eligible") or eligible, "quality label eligibility contradicts original full gates")
    metrics = metrics or {}
    clearance = metrics.get("continuum_route_obstacle_clearance", {})
    plan_data = inputs.json(slot["plan_path"], slot["plan_file_sha256"])
    norm = float(np.linalg.norm(np.asarray(plan_data["z_m"])))
    row = {"slot_id": slot["slot_id"], "task_id": slot["task_id"], "mode": slot["mode"],
        "reference_version": plan_data["representation_version"], "distance_m": slot["distance_m"], "side": slot["side"],
        "amplitude_m": slot["amplitude_m"], "coefficient_norm_m": norm,
        "source_role": record["source_role"], "old_source_slot": slot["old_source_slot"], "status": record["status"],
        "eligible": eligible, "full_task_success": record["full_task_success"],
        "full_task_and_original_safety_passed": full_safe, "actual_source_steps": source["actual_steps"],
        "quality_stage_status": quality.get("quality_stage_status", "QUALITY_METRICS_AVAILABLE" if eligible else "QUALITY_NOT_QUALIFIED"),
        "I_route_rad_s": metrics.get("I_route_rad_s"), "I_full_rad_s": metrics.get("I_full_rad_s"),
        "continuum_path_length_m": metrics.get("continuum_path_length_m"),
        "continuum_route_window_path_length_m": metrics.get("continuum_route_window_path_length_m"),
        "route_clearance_m": clearance.get("route_window_minimum_m"),
        "full_clearance_m": clearance.get("full_saved_horizon_minimum_m"),
        "base_translation_peak_m": metrics.get("base_translation_peak_m"), "base_rotation_peak_rad": metrics.get("base_rotation_peak_rad"),
        "original_full_quality_vector": quality.get("full_metrics"), "failed_prefix_quality_vector": quality.get("failed_prefix_metrics"),
        "original_safety": quality["safety"], "metric_unavailable": quality.get("metric_unavailable"),
        "execution_failure": source.get("execution_failure"), "pipeline_failure": source.get("pipeline_failure"),
        "evaluation_errors": (source.get("evaluation") or {}).get("errors", []),
        "original_evaluation_performance_metrics": {key: (source.get("evaluation") or {}).get("metrics", {}).get(key)
            for key in ("planning_algorithm_wall_latency_s", "torque_preparation_wall_latency_s")},
        "original_dispatch_timing": (source.get("evaluation") or {}).get("dispatch_timing"),
        "independent_safety_and_task_metrics": quality.get("independent_safety_and_task_metrics"),
        "independent_task_requirement_errors": quality.get("independent_safety_and_task_metrics", {}).get("task_requirements"),
        "new_cost": record["new_cost"], "source_evidence_bindings": record["evidence_bindings"]}
    if eligible:
        require(all(finite(row[key]) for key in ("I_route_rad_s", "I_full_rad_s", "continuum_path_length_m", "route_clearance_m", "base_translation_peak_m", "base_rotation_peak_rad")), "qualified full metrics are nonfinite")
    row["new_diagnostics"] = diagnostic_summary(record, slot, task_info["route_interval_s"], inputs)
    row["route_minimum_saved_state"] = route_minimum_saved_time(record, slot, task_info["route_interval_s"], inputs)
    row["route_minimum_time_s"] = row["route_minimum_saved_state"].get("time_s")
    return row


def csv_text(rows):
    columns = ["slot_id", "task_id", "mode", "reference_version", "distance_m", "side", "amplitude_m",
        "source_role", "old_source_slot", "status", "quality_stage_status", "full_task_and_original_safety_passed", "eligible", "actual_source_steps", "I_route_rad_s", "I_full_rad_s",
        "continuum_path_length_m", "continuum_route_window_path_length_m", "route_clearance_m", "full_clearance_m",
        "route_minimum_time_s", "base_translation_peak_m", "base_rotation_peak_rad"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def markdown_report(report, rows, teacher, policies, value):
    text = ["# V6.4-B.3.1 控制器感知参考与有限质量教师", "",
        f"固定开发槽位 {len(rows)}；完整 Task 与原五门禁通过 {report['complete_safe_slots']}，质量指标可比 {report['quality_comparable_slots']}；旧原件复用 {report['reused_slots']}；新固定槽位 {report['new_slots']}。",
        "本轮只评价一个母场景上的四个开发任务、七种固定参考。没有训练、模型采样或独立泛化 TEST；部署仍为 `NOT_MET`。", "",
        "v2 为 r=0.25、f=0.75 的 C² 提前平台参考。解析位置、速度和加速度经过原参考门禁；实际消费版本与 z 以独立参考绑定为准。旧 B.3 封存结论保持不变。", "",
        "| 槽位 | 任务 | 模式 | 来源 | actual状态 | 质量阶段 | 完整安全 | 质量可比 | I_route rad/s | 全程路径 m | 路线球净空 mm |", "|---|---|---|---|---|---|---|---|---:|---:|---:|"]
    def fmt(value, scale=1.): return f"{value*scale:.9g}" if value is not None else "NOT_AVAILABLE"
    for row in rows:
        text.append(f"| {row['slot_id']} | {row['task_id']} | {row['mode']} | {row['source_role']} | {row['status']} | {row['quality_stage_status']} | {row['full_task_and_original_safety_passed']} | {row['eligible']} | {fmt(row['I_route_rad_s'])} | {fmt(row['continuum_path_length_m'])} | {fmt(row['route_clearance_m'],1000.)} |")
    text += ["", "失败、预检拒绝和执行拒绝保留在全部槽位和成功率分母内；其前缀指标单列，不进入成功轨迹质量均值。", "",
        "## 固定有限教师", "", "按原 I_route 排序，使用冻结的 0.001 rad/s tie-band。每组锚定其最低代价，不通过链式近邻扩大并列组；组内按全程路径、系数范数和候选 ID 排序。并列表示固定工程分辨率，不表示统计等价。", "",
        "| 任务 | 合格/7 | 严格最低槽位 | 最低 I_route | tie-selected | selected I_route | 几何规则槽位 |", "|---|---:|---|---:|---|---:|---|"]
    for task in teacher:
        text.append(f"| {task['task_id']} | {task['quality_comparable_count']}/7 | {task['minimum_cost_slot_id']} | {fmt(task['minimum_I_route_rad_s'])} | {task['tie_selected_slot_id']} | {fmt(task['tie_selected_I_route_rad_s'])} | {task['geometric_rule_slot_id']} |")
    text += ["", "严格最低代价与 tie-selected 分开保存；所有同 z 的 v1/v2 差异、原 10%/0.001 标准、路径/净空/基座代价冲突见 `teacher_records.json`。", "",
        "## 零残差、几何规则与最佳常量", "", "| 规则 | 完整安全/4 | 质量可比/4 | 四任务均值 I_route | 成功子集均值（分母可能不同） | 七模式共同完整任务子集均值 |", "|---|---:|---:|---:|---:|---:|"]
    for policy in policies["policies"]:
        text.append(f"| {policy['policy']} | {policy['successful_safe_tasks']}/4 | {policy['quality_comparable_tasks']}/4 | {fmt(policy['all_tasks_mean_I_route_rad_s'])} | {fmt(policy['successful_subset_mean_I_route_rad_s'])} | {fmt(policy['common_complete_task_subset_mean_I_route_rad_s'])} |")
    text += ["", "`V_cond = min_z mean_c J(z,c) - mean_c min_z J(z,c)`。它只反映固定开发任务和有限候选上的离线条件选择潜力。", ""]
    for name in ("full_development_matrix", "common_complete_task_subset"):
        current = value[name]
        text.append(f"- {name}: `{current['status']}`；V_cond = {fmt(current['V_cond_rad_s'])} rad/s；任务数 {len(current['task_ids'])}。")
    text += ["", "缺失或失败没有被填入惩罚代价。全部七种模式共同完成的任务子集与完整四任务矩阵分别报告；最佳常量包含零残差，不只比较最差固定方向。", "",
        "## 原 B.3 门槛", "", "原函数对每方向与 zero、opposite 的比较采用 OR：任一比较满足完整成功对失败的 A 条件，或 10% 且 0.001 rad/s 的 B 条件，可进入 favorable；再要求两侧实际代价优选方向合格且反转。两种比较都报告，不把原规则改成 AND。", "",
        "| 距离 mm | 参考族 | 可解释状态 | 原 OR 与两侧反转判定 |", "|---:|---|---|---|"]
    for group in report["original_OR_gate_by_distance_and_same_amplitude_reference_family"]:
        text.append(f"| {group['distance_m']*1000.:.0f} | {group['reference_family']} | {group['status']} | {group['route_value_identifiable_under_original_rule']} |")
    text += ["", "43mm/v1 六条原件仍不满足原门槛；任何新组结果只属于本补充，不改变旧 B.3 停止结论。安全 actual 后质量后处理缺测时相关判定 unavailable，不能把缺测当作碰撞失败。另报的 AND 字段明确属于更严格的 supplementary 检查。", "",
        "## 向量与路线响应", "", "新运行的 10D/7D 干预分量、连续体参考/反馈期望/实际速度与误差、已有相关球行活动和绑定时长、力矩饱和来自保存数组。selected task velocity 是命令诊断，不能替代实际物理速度；I_route 是全部约束共同作用的 17D 指标，不能称作单球贡献。旧六槽的缺失向量和球行活动为 `NOT_MEASURED_OLD_EVIDENCE`。", "",
        "完整合格结果按保存的 2ms 时刻报告 reference-base、actual-base、actual-reference 和同任务非零 actual-zero actual 的全窗口响应；不插值重建轨迹，不推断全域传递增益。见 `quality_matrix.json`。", "",
        "## 成本与结论边界", "", "```json", json.dumps(report["new_costs"], ensure_ascii=False, indent=2), "```", "",
        "actual、private preview、独立保存力矩 replay、原生几何、附加路线净空查询分别计账。旧六条复用不算新 actual；本 reducer 新物理、几何、QP、DDIM、optimizer 均为 0。", "",
        f"后续判断：`{report['learning_followup_status']}`。即使参考或有限教师改善，也不能由本轮宣称 Diffusion 优势、生成模型必要性、独立泛化、硬实时或连续时间安全。独立审阅与 GitHub 发布由后续交付步骤完成。", ""]
    return "\n".join(text)


def reduce_study(source):
    source = Path(source).resolve()
    require(not any((source/name).exists() for name in OUTPUTS), "exclusive reducer outputs already exist")
    started = time.perf_counter()
    inputs = Inputs()
    identity = guard(source/"source_identity.json", inputs)
    inputs.bind(Path(__file__))
    plan = inputs.json(source/"plan.json")
    terminal = inputs.json(source/"execution_complete.json")
    manifest = inputs.json(source/"task_manifest.json", plan["task_manifest_sha256"])
    require(plan["schema"] == "v64_b31_frozen_plan_v1" and len(plan["slots"]) == 28 and len(plan["tasks"]) == 4, "wrong frozen 4x7 protocol")
    require(manifest["slots"] == plan["slots"] and manifest["tasks"] == plan["tasks"], "frozen Task manifest differs from plan")
    require(terminal["terminal_slots"] == 28 and terminal["reused_slots"] == 6 and terminal["new_attempt_slots"] == 22, "execution is not the frozen complete 28-slot study")
    modes = plan["teacher"]["constant_policy_candidate_set"]
    require(len(modes) == len(set(modes)) == 7 and plan["teacher"]["tie_group_rule"] == "anchor at group minimum; no transitive chaining", "unsupported frozen teacher rules")
    records, rows = [], []
    task_info = {task["task_id"]: task for task in plan["tasks"]}
    for slot in plan["slots"]:
        record = inputs.json(source/"slots"/slot["slot_id"]/"slot_result.json")
        records.append(record)
        rows.append(flatten_record(record, slot, task_info[slot["task_id"]], inputs))
    lookup = {(row["task_id"], row["mode"]): record for row, record in zip(rows, records)}
    require(len(lookup) == 28 and all((task, mode) in lookup for task in task_info for mode in modes), "missing/duplicate fixed comparison cells")
    for row, record, slot in zip(rows, records, plan["slots"]):
        row["reference_actual_response"] = response_summary(record, slot, lookup[slot["task_id"], "z0"], task_info[slot["task_id"]], inputs)
    teacher = task_teachers(rows, plan)
    original_groups = original_gate_groups(rows)
    value = conditional_value(rows, list(task_info), modes)
    policies = policy_summary(rows, plan, teacher, value)
    costs = {}
    for name in ("actual_physics_steps", "private_preview_physics_steps", "independent_saved_torque_replay_steps",
                 "native_geometry_query_calls", "additional_route_quality_geometry_queries", "qp_solve_calls", "preview_calls"):
        costs[name] = sum(int(record["new_cost"].get(name, 0)) for record in records)
    costs["initial_and_anchor_geometry_queries"] = manifest["new_initial_precheck_geometry_queries"]
    costs["phase_counts"] = {}
    costs["route_quality_phase_counts"] = {}
    for record in records:
        for phase, operations in record["new_cost"].get("phase_counts", {}).items():
            target = costs["phase_counts"].setdefault(phase, {})
            for operation, counts in operations.items():
                summed = target.setdefault(operation, {})
                for state, count in counts.items(): summed[state] = summed.get(state, 0)+count
        for phase, operations in record["new_cost"].get("route_quality_cost_ledger", {}).get("phase_counts", {}).items():
            target = costs["route_quality_phase_counts"].setdefault(phase, {})
            for operation, counts in operations.items():
                summed = target.setdefault(operation, {})
                for state, count in counts.items(): summed[state] = summed.get(state, 0)+count
    costs["total_new_native_geometry_query_calls_including_initial_and_route_quality"] = (
        costs["native_geometry_query_calls"]+costs["additional_route_quality_geometry_queries"]+costs["initial_and_anchor_geometry_queries"])
    costs["route_quality_phase_counts_are_separate_from_execution_phase_counts"] = True
    costs.update(training_runs=0, model_sampling=0, seed_search=0, reducer_physics_steps=0,
        reducer_geometry_queries=0, reducer_qp_solves=0, reducer_DDIM_calls=0, reducer_optimizer_updates=0)
    improvement = any(task["teacher_minimum_vs_zero"] and task["teacher_minimum_vs_zero"].get("I_route_reduction_rad_s", 0.) > 0. for task in teacher)
    report = {"schema": "v64_b31_execution_aware_teacher_report_v1", "run_id": plan["run_id"],
        "source_producer_commit": identity["algorithm_producer_commit"], "postprocessing_source_sha256": sha(Path(__file__)),
        "postprocessing_source_path": "v6_4/evaluate_execution_aware_teacher.py",
        "plan_sha256": inputs.files[str((source/"plan.json").resolve())],
        "research_protocol_execution_complete": True, "research_delivery_complete": False,
        "independent_review_status": "PENDING_SEPARATE_REVIEW", "development_tasks": 4, "mother_scenes": 1,
        "total_slots": 28, "reused_slots": 6, "new_slots": 22,
        "complete_safe_slots": sum(row["full_task_and_original_safety_passed"] for row in rows),
        "quality_comparable_slots": sum(row["eligible"] for row in rows),
        "failed_original_Task_or_safety_slots": [row["slot_id"] for row in rows if not row["full_task_and_original_safety_passed"]],
        "quality_unavailable_after_safe_actual_slots": [row["slot_id"] for row in rows if row["full_task_and_original_safety_passed"] and not row["eligible"]],
        "failed_or_unqualified_slots": [row["slot_id"] for row in rows if not row["eligible"]],
        "source_status_counts": dict(Counter(row["status"] for row in rows)),
        "quality_stage_status_counts": dict(Counter(row["quality_stage_status"] for row in rows)), "new_costs": costs,
        "policy_comparisons": policies, "conditional_selection_potential": value,
        "v2_actual_reference_binding": [{"slot_id": row["slot_id"], "eligible": row["eligible"],
            "binding": record["quality"].get("independent_safety_and_task_metrics", {}).get("reference_binding")}
            for row, record in zip(rows, records) if "_v2" in row["reference_version"]],
        "same_z_v1_v2_comparisons": {task["task_id"]: task["same_z_v2_vs_v1"] for task in teacher},
        "old_B3_thresholds_on_supplement": {task["task_id"]: task["old_B3_threshold_checks"] for task in teacher},
        "original_OR_gate_by_distance_and_same_amplitude_reference_family": original_groups,
        "old_B3_sealed_decision": plan["old_B3_route_value_result_immutable"], "old_B3_decision_rewritten": False,
        "training_executed": False, "model_sampling_executed": False, "independent_generalization_TEST": "NOT_RUN_DEVELOPMENT_STUDY",
        "learning_followup_status": "REFERENCE_VALUE_REQUIRES_COST_AND_GEOMETRIC_RULE_REVIEW_DIFFUSION_NECESSITY_UNESTABLISHED" if improvement else "NO_REFERENCE_GAIN_TO_MOTIVATE_EXPANDED_LEARNING",
        "deployment": "NOT_MET", "20ms_wall_is_research_gate": False, "continuous_time_certified": False,
        "causal_obstacle_specific_intervention_claim": False,
        "reducer_scope": "saved arrays and frozen analytic reference evaluated on saved clocks; no model allocation/forward/geometry/physics/solver/inference",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "reducer_wall_s": time.perf_counter()-started}
    try:
        report["postprocessing_git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["postprocessing_git_head"] = None
    matrix = {"schema": "v64_b31_quality_matrix_v1", "run_id": plan["run_id"], "plan_sha256": report["plan_sha256"],
        "rows": rows, "failures_excluded_from_quality_means": True, "failures_retained_in_success_denominators": True,
        "new_physics_steps": 0, "new_geometry_queries": 0}
    teachers = {"schema": "v64_b31_finite_teacher_records_v1", "plan_sha256": report["plan_sha256"],
        "frozen_teacher_rules": plan["teacher"], "tasks": teacher, "policies": policies,
        "conditional_selection_potential": value, "original_OR_gate_groups": original_groups, "labels_used_for_learning": False}
    text, table = markdown_report(report, rows, teacher, policies, value), csv_text(rows)
    inputs.unchanged()
    matrix["input_sha256"] = dict(sorted(inputs.files.items()))
    for name, content in (("quality_matrix.json", matrix), ("teacher_records.json", teachers), ("report.json", report),
                           ("REPORT.md", text), ("all_candidates.csv", table)):
        with (source/name).open("x", encoding="utf8", newline="\n") as handle:
            handle.write(content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return {"status": "PASS", "terminal_slots": len(rows), "complete_safe_slots": report["complete_safe_slots"],
        "outputs": {name: sha(source/name) for name in OUTPUTS}, "physics_steps": 0, "geometry_queries": 0, "DDIM_calls": 0}


def self_test():
    """Synthetic comparison-rule examples only; never an experimental result."""
    rows = [{"slot_id": str(i), "mode": chr(97+i), "eligible": True, "I_route_rad_s": value,
        "continuum_path_length_m": path, "coefficient_norm_m": .012}
        for i, (value, path) in enumerate(((1., 3.), (1.0009, 1.), (1.0018, .5)))]
    ranking = rank_candidates(rows, .001)
    require(len(ranking["groups"]) == 2 and ranking["tie_selected_slot_id"] == "1"
        and ranking["minimum_cost_slot_id"] == "0", "tie groups chained or selected overwritten minimum")
    matrix = [{"task_id": task, "mode": mode, "eligible": True, "I_route_rad_s": cost}
        for task, mode, cost in (("a", "x", 1.), ("a", "y", 2.), ("b", "x", 2.), ("b", "y", 1.))]
    values = conditional_value(matrix, ["a", "b"], ["x", "y"])
    require(values["full_development_matrix"]["V_cond_rad_s"] == .5, "wrong conditional value")
    matrix[-1]["eligible"] = False
    values = conditional_value(matrix, ["a", "b"], ["x", "y"])
    require(values["full_development_matrix"]["V_cond_rad_s"] is None
        and values["common_complete_task_subset"]["task_ids"] == ["a"], "failure imputed into matrix")
    require(rank_candidates([{**rows[0], "eligible": False}], .001)["tie_selected_slot_id"] is None, "failed candidate ranked")
    examples = []
    for mode, cost in (("z0", 2.), ("v1_plus12", 1.), ("v1_minus12", 1.05), ("v2_plus12", 1.),
                       ("v2_minus12", 1.05), ("v2_plus20", 1.), ("v2_minus20", 1.05)):
        examples.append({"slot_id": mode, "mode": mode, "eligible": True, "full_task_and_original_safety_passed": True,
            "I_route_rad_s": cost, "continuum_path_length_m": 1., "route_clearance_m": .02,
            "base_translation_peak_m": .001, "base_rotation_peak_rad": .001})
    checks = old_threshold_checks(examples, {"teacher": {"old_B3_gate": {"absolute_reduction_rad_s": .001, "relative_reduction": .1}}})
    positive = next(row for row in checks["candidates"] if row["mode"] == "v1_plus12")
    require(positive["legacy_OR_advantage_vs_zero_or_opposite"]
        and not positive["stricter_supplemental_AND_quality_threshold_passed"], "legacy OR silently became AND")
    encoded = [json.dumps([{"source": "pair", "active": True, "selected_binding": True},
                           {"source": "pair", "active": True, "selected_binding": False}]),
               json.dumps([{"source": "pair", "active": False, "selected_binding": False}])]
    activity = obstacle_activity(encoded, np.array([0., .02]), [0., .02], .02)
    require(activity["active_any_related_row"]["full_consumed_duration_s"] == .02
        and activity["per_source"]["pair"]["active_duration_s"] == .02, "row durations double-counted simultaneous/duplicate rows")
    return {"schema": "v64_b31_reducer_rule_self_test_v1", "status": "PASS",
        "checks": ["anchor_at_min_nontransitive_ties", "minJ_separate_from_tie_selected", "finite_V_cond",
            "failure_no_imputation_common_subset", "failure_not_ranked", "original_OR_not_AND", "obstacle_duration_union_not_row_sum"],
        "synthetic_rule_examples_only": True, "experiment_results_created": False,
        "physics_steps": 0, "geometry_queries": 0, "DDIM_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.source: parser.error("self-test must not target real study outputs")
        result = self_test()
    else:
        if args.source is None: parser.error("--source is required")
        result = reduce_study(args.source)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
