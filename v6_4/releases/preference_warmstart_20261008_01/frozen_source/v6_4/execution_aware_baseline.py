"""Static B.3.1 T0 diagnostics and exact-input reuse; no simulator imports.

The published B.3 manifest authenticates local raw files omitted from Git.
Reference, actual, and paired actual differences use separate time grids and
never stand in for a controller transfer gain or a causal obstacle attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

B3_RELEASE_COMMIT = "bdd6df6b4c68a6c95872f4639b8b8269fb6856c7"
B3_PRODUCER = "f5f1687f5ba0b119c91bbc385dfdc841b56d091e"
B3_MANIFEST_SHA256 = "57ba4bbf31ca3ff106d93d1ade0d1700bb4ab50840075847d5299878cf429033"
DEFAULT_OLD = Path(r"E:\v64b3work_20261007_01\v6_4\output\conditional_route_value_20261007_01")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "v6_4/output/execution_aware_route_teacher_20261007_01"
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")
ROOT_ALIASES = {"PILOT_00": "EA_00", "PILOT_01": "EA_01", "PILOT_02": "EA_02",
                "PILOT_03": "EA_07", "PILOT_04": "EA_08", "PILOT_05": "EA_09"}
MODES = {"z0": "z0", "z+": "v1_plus12", "z-": "v1_minus12"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class _Sources:
    def __init__(self, old_output, expected_manifest_sha256=B3_MANIFEST_SHA256):
        self.root = Path(old_output).resolve()
        manifest_path = self.root / "manifest.json"
        if _sha(manifest_path) != expected_manifest_sha256:
            raise ValueError("B.3 manifest differs from the published release anchor")
        self.manifest = _read(manifest_path)
        if self.manifest.get("status") != "FINAL" or self.manifest.get("algorithm_producer_commit") != B3_PRODUCER:
            raise ValueError("not the sealed B.3 producer manifest")
        self.records = {"manifest.json": {"relative_path": "manifest.json", "path": str(manifest_path),
            "sha256": expected_manifest_sha256, "bytes": manifest_path.stat().st_size,
            "verification": "PUBLISHED_B3_RELEASE_MANIFEST_SHA256"}}

    def bind(self, name, expected=None, *, optional=False):
        given = Path(name)
        path = given.resolve() if given.is_absolute() else (self.root / given).resolve()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("B.3 source path escapes the immutable source OUT") from error
        entry = self.manifest["payload"].get(relative)
        if not entry:
            raise ValueError("B.3 source is not authenticated by the original manifest: " + relative)
        if expected is not None and entry["sha256"] != expected:
            raise ValueError("nested source digest differs from the published B.3 manifest: " + relative)
        record = {"relative_path": relative, "path": str(path), "sha256": entry["sha256"],
                  "bytes": entry["bytes"], "verification": "ORIGINAL_B3_MANIFEST_SHA256"}
        if not path.is_file():
            if not optional:
                raise FileNotFoundError(path)
            record["verification"] = "LOCAL_RAW_NOT_AVAILABLE"
            self.records[relative] = record
            return None
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise ValueError("local B.3 bytes differ from the original manifest: " + relative)
        self.records[relative] = record
        return path

    def read(self, name, expected=None):
        return _read(self.bind(name, expected))

    def unchanged(self):
        for record in self.records.values():
            path = Path(record["path"])
            if record["verification"] == "LOCAL_RAW_NOT_AVAILABLE":
                if path.exists():
                    raise ValueError("source availability changed during T0")
            elif _sha(path) != record["sha256"]:
                raise ValueError("immutable B.3 source changed during T0: " + record["relative_path"])


def minimum_jerk_base(task, times):
    """Evaluate the saved polynomial declaration, without provider/model calls."""
    source = task["scenario"]["continuum_target"]
    if (source.get("mode"), source.get("reference_profile")) != ("irregular_waypoints", "minimum_jerk_c2"):
        raise ValueError("unsupported frozen base reference")
    times = np.asarray(times, dtype=float)
    points = np.asarray(source["waypoint_points_m"], dtype=float)
    durations = np.asarray(source["segment_durations_s"], dtype=float)
    initial = np.asarray(source["initial_position_w"], dtype=float)
    transition = float(source["transition_duration_s"])
    cumulative = np.r_[0., np.cumsum(durations)]
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < -1e-9)
            or np.any(times > 27 + 1e-9) or points.shape != (len(durations) + 1, 3)
            or not np.isfinite(points).all() or np.any(durations <= 0)
            or abs(durations.sum() - source["path_duration_s"]) > 1e-9):
        raise ValueError("invalid saved reference declaration/time grid")
    result = np.empty((len(times), 3))
    for index, time in enumerate(np.clip(times, 0., 27.)):
        if time < transition:
            lower, upper, u = initial, points[0], time / transition
        elif time - transition >= source["path_duration_s"]:
            result[index] = points[-1]
            continue
        else:
            elapsed = time - transition
            segment = min(max(int(np.searchsorted(cumulative, elapsed, side="right") - 1), 0), len(durations) - 1)
            lower, upper = points[segment:segment + 2]
            u = (elapsed - cumulative[segment]) / durations[segment]
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        result[index] = lower + progress * (upper - lower)
    return result


def v1_reference_offset(plan, times):
    times = np.asarray(times, dtype=float)
    if plan["representation_version"] != "task_anchored_cartesian_residual_v1":
        raise ValueError("T0 must retain the old v1 representation")
    result = np.zeros((len(times), 3))
    definition = plan["definition"]
    for interval, basis, z, active in zip(definition["intervals_s"], definition["transverse_bases"],
                                         plan["z_m"], definition["interval_mask"]):
        if not active:
            if np.any(z):
                raise ValueError("nonzero inactive v1 coefficient")
            continue
        lower, upper = interval
        inside = (times > lower) & (times < upper)
        u = np.where(inside, (times - lower) / (upper - lower), 0.)
        result += np.where(inside, 64 * u**3 * (1 - u)**3, 0.)[:, None] * (np.asarray(basis) @ z)
    return result


def _window(times, interval):
    lower, upper = interval
    return (times >= lower - 1e-10) & (times <= upper + 1e-10)


def _displacement_summary(delta, times, interval, first_axis):
    delta, times = np.asarray(delta), np.asarray(times)
    if delta.shape != (len(times), 3) or not np.isfinite(delta).all():
        raise ValueError("invalid saved displacement array")
    result = {}
    for name, mask in (("full_task", np.ones(len(times), dtype=bool)), ("route_window", _window(times, interval))):
        if not mask.any():
            raise ValueError("empty fixed comparison window")
        values = delta[mask]
        norms = np.linalg.norm(values, axis=1)
        signed = values @ first_axis
        result[name] = {"sample_count": len(values), "norm_rms_m": float(np.sqrt(np.mean(norms**2))),
            "norm_peak_m": float(np.max(norms)), "norm_mean_m": float(np.mean(norms)),
            "first_transverse_signed_min_m": float(np.min(signed)),
            "first_transverse_signed_max_m": float(np.max(signed)),
            "first_transverse_rms_m": float(np.sqrt(np.mean(signed**2))),
            "first_transverse_peak_absolute_m": float(np.max(np.abs(signed)))}
    return result


def _load_slot(sources, declared):
    slot, task_id = declared["slot_id"], declared["task_id"]
    directory = "pilot/attempts/" + slot
    attempt = sources.read(directory + "/attempt_result.json")
    task = sources.read(directory + "/task.json")
    plan_path = sources.bind(directory + "/plan.json", attempt["plan_file_sha256"])
    plan = _read(plan_path)
    definition = sources.read(f"tasks/{task_id}/definition.json")
    original_task = sources.read(f"tasks/{task_id}/task.json")
    fixed_plan = sources.read(declared["plan_path"])
    evaluation_path = sources.bind(attempt["evaluation_path"], attempt["evaluation_sha256"])
    evaluation = _read(evaluation_path)
    evaluation_manifest = sources.read(evaluation_path.parent / "manifest.json")
    for name, digest in evaluation_manifest.items():
        sources.bind(evaluation_path.parent / name, digest, optional=name == "fresh_replay.npz")
    trace = sources.bind(attempt["trace_path"], attempt["trace_sha256"], optional=True)
    fresh = sources.bind(evaluation_path.parent / "fresh_replay.npz", evaluation_manifest["fresh_replay.npz"], optional=True)
    quality = sources.read(f"pilot/quality/{slot}/route_quality.json")
    clearance = sources.bind(f"pilot/quality/{slot}/route_clearance.npz", optional=True)
    config = sources.bind("frozen_execution_config.json", attempt["qp_config_sha256"])
    sources.bind("source_identity.json", attempt["source_identity_sha256"])
    sources.bind(directory + "/cost_ledger.json")
    expected_z = np.zeros((6, 2))
    if declared["candidate_name"] != "z0":
        expected_z[2, 0] = .012 if declared["candidate_name"] == "z+" else -.012
    checks = {
        "slot_identity": (attempt["slot_id"], attempt["task_id"]) == (slot, task_id),
        "original_TaskSpec_exact_object": task == original_task,
        "original_TaskSpec_exact_bytes": _sha(sources.root / f"tasks/{task_id}/task.json") == _sha(sources.root / directory / "task.json"),
        "task_canonical_identity": _canonical_hash(task) == attempt["task_sha256"] == definition["task_sha256"],
        "original_v1_plan_exact_object": plan == fixed_plan and plan["definition"] == definition,
        "original_v1_plan_exact_bytes": _sha(plan_path) == _sha(sources.root / declared["plan_path"]),
        "plan_canonical_identity": _canonical_hash(plan) == attempt["plan_content_sha256"],
        "definition_canonical_identity": _canonical_hash({k: v for k, v in definition.items() if k != "definition_sha256"}) == definition["definition_sha256"],
        "v1_version": plan["schema"] == "task_anchored_residual_plan_v1" and plan["representation_version"] == "task_anchored_cartesian_residual_v1",
        "expected_frozen_z": bool(np.array_equal(plan["z_m"], expected_z)),
        "route_interval_unchanged": declared["T_route_s"] == definition["intervals_s"][2] == quality["route_intervals_s"][0],
        "complete_original_task_and_safety": bool(attempt["actual_steps"] == 13500 and attempt["full_task_success"] is True
            and evaluation["complete"] is True and evaluation["full_task_success"] is True
            and evaluation["evidence_valid"] is True and all(evaluation[name]["passed"] is True for name in GATES)),
        "no_repair_or_fallback": attempt["fallback_used"] is False and attempt["postprocessing"] == [],
    }
    mid = np.mean(declared["T_route_s"])
    basis = np.asarray(definition["transverse_bases"][2])
    center = np.asarray(task["scenario"]["workspace_obstacles"][1]["center_w"])
    side = 1 if task_id.endswith("c_plus") else -1
    checks["exact_43mm_declaration"] = bool(np.allclose(center, minimum_jerk_base(task, [mid])[0] + side * .043 * basis[:, 0], atol=1e-14, rtol=0.))
    if not all(checks.values()):
        raise ValueError("strict original B.3 reuse check failed: " + str({k: v for k, v in checks.items() if not v}))
    return {"declared": declared, "attempt": attempt, "task": task, "plan": plan,
            "definition": definition, "evaluation": evaluation, "quality": quality,
            "trace_path": trace, "fresh_path": fresh, "clearance_path": clearance,
            "config_path": config, "checks": checks, "raw_available": trace is not None and fresh is not None}


def validate_reuse_inputs(old_output, source_slot, *, task_path=None, plan_path=None, slot_alias=None):
    """Verify original evidence and optional prospective exact-byte inputs.

    The new alias is wrapper metadata only. Supplying either prospective file
    requires both. A renamed TaskSpec, reserialized plan, or changed version is
    refused; this function never rewrites a source or starts an actual run.
    """
    sources = _Sources(old_output)
    old_plan = sources.read("plan.json")
    declared = next((row for row in old_plan["pilot_slots"] if row["slot_id"] == source_slot), None)
    if declared is None:
        raise ValueError("unknown old source slot")
    data = _load_slot(sources, declared)
    if (task_path is None) != (plan_path is None):
        raise ValueError("prospective task_path and plan_path must be supplied together")
    prospective_verified = False
    if task_path is not None:
        original = sources.root / "pilot/attempts" / source_slot
        if Path(task_path).read_bytes() != (original / "task.json").read_bytes():
            raise ValueError("prospective TaskSpec is not the original exact bytes")
        if Path(plan_path).read_bytes() != (original / "plan.json").read_bytes():
            raise ValueError("prospective v1 plan is not the original exact bytes")
        prospective_verified = True
    sources.unchanged()
    return {"source_slot_id": source_slot, "slot_alias": slot_alias or ROOT_ALIASES[source_slot],
        "canonical_mode": MODES[declared["candidate_name"]], "task_id_retained": declared["task_id"],
        "eligible_to_reuse_as_original_43mm_v1": data["raw_available"],
        "prospective_exact_input_bytes_verified": prospective_verified,
        "new_actual_run": False, "new_physics_steps": 0, "new_geometry_queries": 0,
        "strict_checks": data["checks"], "objects": list(sources.records.values())}


def build_baseline_diagnosis(old_output=DEFAULT_OLD, output=DEFAULT_OUTPUT):
    sources = _Sources(old_output)
    output = Path(output).resolve()
    if output == sources.root or sources.root in output.parents:
        raise ValueError("T0 output must be outside immutable B.3 OUT")
    for name in ("baseline_diagnosis.json", "baseline_inputs_manifest.json"):
        if (output / name).exists():
            raise FileExistsError("T0 output already exists; no silent overwrite: " + name)
    old_plan = sources.read("plan.json")
    paired = sources.read("paired_metrics.json")
    old_report = sources.read("report.json")
    old_decision = sources.read("pilot/decision.json")
    if len(old_plan["pilot_slots"]) != 6 or old_decision["route_value_identifiable"] is not False:
        raise ValueError("T0 requires the unchanged negative six-slot B.3 result")
    loaded = [_load_slot(sources, declared) for declared in old_plan["pilot_slots"]]
    zero_states = {}
    for data in loaded:
        if data["declared"]["candidate_name"] == "z0" and data["fresh_path"] is not None:
            with np.load(data["fresh_path"], allow_pickle=False) as state:
                zero_states[data["declared"]["task_id"]] = (state["time"].copy(), state["continuum_position"].copy())
    rows = []
    for data in loaded:
        declared, definition = data["declared"], data["definition"]
        slot, task_id = declared["slot_id"], declared["task_id"]
        interval = declared["T_route_s"]
        axis = np.asarray(definition["transverse_bases"][2])[:, 0]
        row = {"source_slot_id": slot, "new_slot_alias": ROOT_ALIASES[slot], "canonical_mode": MODES[declared["candidate_name"]],
            "task_id_retained": task_id, "obstacle_offset_m": .043, "reference_version": data["plan"]["representation_version"],
            "strict_checks": data["checks"], "strictly_reusable_as_original_input": data["raw_available"],
            "reused_evidence_is_new_actual": False, "T_route_s": interval,
            "unmeasured": {name: {"status": "NOT_MEASURED", "reason": reason} for name, reason in (
                ("original_unconstrained_velocity_17d", "not saved in the B.3 trace"),
                ("velocity_box_clipped_nominal_17d", "not saved in the B.3 trace"),
                ("I_route_continuum_10d", "cannot decompose a scalar norm without the nominal source vector"),
                ("I_route_rigid_7d", "cannot decompose a scalar norm without the nominal source vector"),
                ("single_sphere_intervention_or_constraint_activity", "no per-row sphere-source activation/residual log"),
                ("controller_transfer_gain_or_causal_attribution", "paired closed-loop trajectory differences are descriptive, not an identified transfer function"))}}
        if not data["raw_available"]:
            row.update(status="LOCAL_FULL_RAW_NOT_AVAILABLE", full_window_metrics="NOT_MEASURED",
                reported_midpoint_slice_only=next((r for r in paired["pilot"].get("actual_route_midpoint_diagnostic", []) if r["slot_id"] == slot), None),
                reconstructed_or_interpolated_actual=False)
            rows.append(row)
            continue
        with np.load(data["fresh_path"], allow_pickle=False) as state:
            times, actual = state["time"].copy(), state["continuum_position"].copy()
        if (len(times) != 13501 or actual.shape != (len(times), 3) or not np.isfinite(actual).all()
                or not np.allclose(times, np.arange(13501) * .002, atol=1e-8, rtol=0.)):
            raise ValueError("invalid full original saved-state grid")
        with np.load(data["trace_path"], allow_pickle=False) as trace:
            planning_times = trace["task_time"].copy()
            intervention = trace["task_avoidance_intervention"].copy()
            consumed = trace["task_input_reference_continuum_target_position"].copy()
            selected = trace["task_selected_command"]
            if trace["torque"].shape != (13500, 67) or selected.shape != (1350, 17) or not np.isfinite(selected).all():
                raise ValueError("old selected-command/actual-step shape invalid")
        if planning_times.shape != (1350,) or intervention.shape != (1350,) or not np.isfinite(intervention).all():
            raise ValueError("old consumed planning grid/diagnostic invalid")
        base = minimum_jerk_base(data["task"], times)
        analytic = v1_reference_offset(data["plan"], times)
        consumed_offset = consumed - minimum_jerk_base(data["task"], planning_times)
        consumed_error = np.linalg.norm(consumed_offset - v1_reference_offset(data["plan"], planning_times), axis=1)
        if np.max(consumed_error) > 1e-12:
            raise ValueError("saved consumed input differs from the original analytic v1 reference")
        mask = _window(planning_times, interval)
        scalar_I = float(np.sqrt(np.mean(intervention[mask]**2)))
        if abs(scalar_I - data["quality"]["full_metrics"]["I_route_rad_s"]) > 1e-14:
            raise ValueError("old route scalar diagnostic differs from sealed quality")
        midpoint = float(np.mean(interval))
        index = int(np.argmin(np.abs(times - midpoint)))
        zeros = zero_states.get(task_id)
        paired_summary = {"status": "NOT_MEASURED", "reason": "same-task zero actual native data unavailable"}
        midpoint_vs_zero = None
        if zeros is not None:
            zero_time, zero_actual = zeros
            if not np.array_equal(times, zero_time):
                raise ValueError("same-task actual grids differ; T0 must not interpolate")
            paired_summary = {"status": "MEASURED_FROM_SAVED_PAIRED_ACTUAL", "zero_source_slot_id": "PILOT_00" if task_id.endswith("c_plus") else "PILOT_03",
                "sampling": "same physical time; exact common saved 2ms grid; no interpolation",
                **_displacement_summary(actual - zero_actual, times, interval, axis)}
            midpoint_vs_zero = float((actual[index] - zero_actual[index]) @ axis)
        row.update(status="FULL_LOCAL_RAW_VERIFIED", selected_command_source={"status": "SAVED", "field": "task_selected_command", "dimension": 17},
            input_reference_minus_base={"sampling": "actual consumed input positions at original 50Hz planning ticks",
                "source_field": "task_input_reference_continuum_target_position", "max_analytic_consistency_error_m": float(np.max(consumed_error)),
                **_displacement_summary(consumed_offset, planning_times, interval, axis)},
            declared_input_minus_base_on_native_actual_grid={"sampling": "analytic v1 reference evaluated on saved native actual times; no synthetic actual",
                **_displacement_summary(analytic, times, interval, axis)},
            actual_minus_base={"sampling": "original independent fresh saved 2ms states minus analytic base at the same physical times",
                **_displacement_summary(actual - base, times, interval, axis)},
            actual_minus_same_task_zero_actual=paired_summary,
            original_scalar_intervention={"I_route_rad_s": scalar_I, "I_full_rad_s": float(np.sqrt(np.mean(intervention**2))),
                "route_sample_count": int(mask.sum()), "full_sample_count": len(planning_times),
                "scope": "17D velocity-box-clipped nominal to selected norm; all constraints jointly; not single-obstacle contribution"},
            midpoint_slice={"frozen_midpoint_s": midpoint, "nearest_saved_time_s": float(times[index]), "source_state_index": index,
                "declared_reference_first_transverse_offset_m": float(analytic[index] @ axis),
                "actual_first_transverse_offset_from_base_m": float((actual[index] - base[index]) @ axis),
                "actual_first_transverse_difference_from_same_task_zero_m": midpoint_vs_zero,
                "interpretation": "local slice only; not full-window transfer gain, required avoidance distance, or causal proof"})
        rows.append(row)
    sources.unchanged()
    diagnosis = {"schema": "v64_b31_baseline_diagnosis_v1", "stage": "T0_STATIC_REUSE_ONLY",
        "old_output": str(sources.root), "old_release_commit": B3_RELEASE_COMMIT, "old_actual_producer": B3_PRODUCER,
        "old_manifest_sha256": B3_MANIFEST_SHA256, "old_B3_status_unchanged": old_report["status"],
        "old_B3_route_value_identifiable_unchanged": False, "old_B3_new_training_and_TEST": "NOT_RUN_PILOT_STOP",
        "source_slot_count": 6, "independent_task_count": 2, "mother_scene_count": 1,
        "strictly_reusable_slot_count": sum(r["strictly_reusable_as_original_input"] for r in rows),
        "reuse_contract": "exact original 43mm TaskSpec bytes and v1 plan bytes; original task_id retained; new alias is wrapper metadata; no new actual credit",
        "rows": rows, "observations": [
            "The original zero references already have substantial signed actual-base displacement near the fixed midpoint.",
            "Same-side nonzero-vs-zero differences are measured separately over the full route window and full task.",
            "Midpoint slices are descriptive; no transfer gain, minimum necessary avoidance or causal attribution is inferred.",
            "Original 17D scalar intervention cannot provide 10D/7D or single-sphere decomposition."],
        "new_costs": {"actual_slots": 0, "physics_steps": 0, "private_preview_steps": 0, "replay_steps": 0,
            "geometry_queries": 0, "DDIM_calls": 0, "optimizer_updates": 0},
        "old_sources_unchanged_after_read": True, "deployment": "NOT_MET"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "baseline_diagnosis.json").open("x", encoding="utf-8") as stream:
        json.dump(diagnosis, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"schema": "v64_b31_baseline_inputs_manifest_v1", "published_B3_manifest_sha256": B3_MANIFEST_SHA256,
        "published_B3_manifest_path": "v6_4/releases/conditional_route_value_20261007_01/snapshot/manifest.json",
        "published_B3_release_commit": B3_RELEASE_COMMIT, "inputs": list(sources.records.values()),
        "generator": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
        "outputs": {"baseline_diagnosis.json": _sha(output / "baseline_diagnosis.json")},
        "new_physics_steps": 0, "new_geometry_queries": 0, "new_DDIM_calls": 0, "new_optimizer_updates": 0}
    with (output / "baseline_inputs_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_baseline_diagnosis(args.old_output, args.output)
    print(json.dumps({"stage": result["stage"], "strictly_reusable_slot_count": result["strictly_reusable_slot_count"],
                      "new_costs": result["new_costs"]}))


if __name__ == "__main__":
    main()
