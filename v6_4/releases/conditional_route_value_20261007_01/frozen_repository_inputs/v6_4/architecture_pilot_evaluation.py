"""Frozen 48-slot B.1 raw comparison and twelve attributed K1 attempts.

DDIM is an offline proposal calculation before execution, never a new 50 Hz
planning call.  The two models share seeds, tasks, codec and unchanged gates.
All files are additive.  An unfinished actual attempt is never restarted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from v6_4.contracts import TaskSpec, TrajectoryProposal
from v6_4.proposal_gate import gate_proposal, requirement_results
from v6_4.run_planning import load_task_suite, prepare_provider, save_reference
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_v6_lite import default_v6_lite_robot_spec

MODELS = ("M0", "M1")
CANDIDATE_SEEDS = (2026100611, 2026100612, 2026100613, 2026100614)
CONFIG_SCHEMA = "v6_4_b1_paired_candidate_protocol_v1"
A1_EXECUTION_CONFIG_SHA256 = "25a8f4aa060fcc3bc1f8aa6e5c791541d1f30a92b32565b8868f24e7da9f291e"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(values):
    """Identity of finite little-endian float64 physical controls, not metadata."""
    values = np.asarray(values, dtype="<f8")
    if not np.all(np.isfinite(values)):
        raise ValueError("control identity requires finite values")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _sources():
    from v6_4.reference_execution_repair import record_sources
    return record_sources()


def _safe_id(task_id):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id) or task_id in (".", ".."):
        raise ValueError("task ID is not a safe single directory name")
    return task_id


def _validate_six_tasks(tasks):
    if (len(tasks) != 6 or any(t.split != "test" for t in tasks)
            or len({t.task_id for t in tasks}) != 6
            or Counter(t.family for t in tasks) != Counter({
                "end_effector_detour": 2, "mid_arm_detour": 2, "multiple_routes": 2})):
        raise ValueError("pilot requires six independent TEST tasks, two in each frozen family")
    if len({t.group_id for t in tasks}) != 6:
        raise ValueError("the six pilot tasks must have six independent source groups")
    for task in tasks:
        _safe_id(task.task_id)


def freeze_candidate_config(tasks_path, output):
    """Freeze the complete paired latent and K1/K4 policy before training."""
    tasks_path = Path(tasks_path).resolve()
    tasks = load_task_suite(tasks_path)
    _validate_six_tasks(tasks)
    config = {"schema": CONFIG_SCHEMA, "tasks_path": str(tasks_path),
        "tasks_file_sha256": _sha(tasks_path),
        "task_ids": [t.task_id for t in tasks],
        "task_hashes": {t.task_id: t.sha256() for t in tasks},
        "models": list(MODELS), "candidate_seeds": list(CANDIDATE_SEEDS),
        "candidate_slots_per_model": 24, "candidate_slots_total": 48,
        "K1_candidate_index": 0, "K4_candidate_indices": [0, 1, 2, 3],
        "latent_shape": [30, 17], "latent_dtype": "float32",
        "latent_rng": "numpy.random.Generator(PCG64).standard_normal(dtype=float32)",
        "paired_latent_bytes_shared_between_models": True,
        "ddim_steps": 20, "output_full_controls": [32, 17],
        "start_boundary": "original codec algebraic C0/C1 from exact TaskSpec q0/dq0",
        "raw_clip_repair_fallback": False,
        "raw_gate": "original gate_proposal and complete private free-base 1351-state prediction",
        "geometry_short_circuit": "NOT_RUN when the unchanged gate does not query native geometry",
        "checkpoint_selection": "predeclared VAL metric only; no TEST checkpoint selection",
        "frozen_before_formal_training": True,
        "maximum_actual_attempts": 12, "actual_candidate_index": 0,
        "actual_denominator_per_model": 6, "K4_actual": "NOT_RUN",
        "raw_closed_loop": "NOT_RUN", "engineering_repair_version": "v2",
        "terminal_progress_repair": True, "execution_clock": "research_simulation",
        "simulation_task_period_s": .020, "physics_period_s": .002, "duration_s": 27.,
        "wall_performance_is_research_gate": False, "wall_deployment": "NOT_MET",
        "frozen_execution_config_sha256": A1_EXECUTION_CONFIG_SHA256,
        "inference_scope": "offline DDIM proposals before a rollout, not online 50Hz planning",
        "diagnostic_candidate_count": 2,
        "diagnostics_count_in_48_slots_or_checkpoint_selection": False,
        "diagnostic_shadow_native_evaluations": 3,
        "diagnostic_shadow_native_cases": ["original_M1_K1", "task_goal", "obstacle"],
        "diagnostic_shadow_geometry_scope": "original nominal free-base 1351 states, 5mm, adaptive1, original whole-body pair policy; unscored even when task gate rejects",
        "diagnostic_shadow_geometry_overrides_main_raw_gate": False}
    _write(output, config)
    return config


def load_candidate_config(path):
    config = _read(path)
    if (config.get("schema") != CONFIG_SCHEMA or config.get("models") != list(MODELS)
            or config.get("candidate_seeds") != list(CANDIDATE_SEEDS)
            or config.get("K1_candidate_index") != 0
            or config.get("K4_candidate_indices") != [0, 1, 2, 3]
            or config.get("candidate_slots_total") != 48
            or config.get("maximum_actual_attempts") != 12
            or config.get("engineering_repair_version") != "v2"
            or config.get("terminal_progress_repair") is not True
            or config.get("execution_clock") != "research_simulation"
            or config.get("latent_shape") != [30, 17] or config.get("ddim_steps") != 20
            or config.get("output_full_controls") != [32, 17]
            or config.get("simulation_task_period_s") != .020
            or config.get("physics_period_s") != .002 or config.get("duration_s") != 27.
            or config.get("raw_clip_repair_fallback") is not False
            or config.get("frozen_execution_config_sha256") != A1_EXECUTION_CONFIG_SHA256):
        raise ValueError("candidate protocol differs from the finite frozen pilot")
    task_path = Path(config["tasks_path"])
    if _sha(task_path) != config["tasks_file_sha256"]:
        raise ValueError("frozen TEST suite bytes changed")
    tasks = load_task_suite(task_path)
    _validate_six_tasks(tasks)
    if (config["task_ids"] != [t.task_id for t in tasks]
            or config["task_hashes"] != {t.task_id: t.sha256() for t in tasks}):
        raise ValueError("frozen TEST task identities changed")
    return config, tasks


def _latent(seed):
    return np.random.default_rng(seed).standard_normal((30, 17), dtype=np.float32)


def _candidate_dir(root, model_name, task_id, index):
    if model_name not in MODELS or index not in range(4):
        raise ValueError("unknown model or candidate slot")
    return Path(root) / model_name / _safe_id(task_id) / f"candidate_{index:03d}"


def _load_sampler(checkpoint, device):
    from v6_4.architecture_pilot_training import load_pilot_sampler
    return load_pilot_sampler(checkpoint, device=device)


def sample_candidates(config_path, training_root, output, *, device="cpu"):
    config, tasks = load_candidate_config(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = _sources()
    started = time.perf_counter()
    rows, models = [], {}
    for model_name in MODELS:
        checkpoint = Path(training_root) / "models" / model_name / "checkpoint.pt"
        initialized = time.perf_counter()
        sampler = _load_sampler(checkpoint, device)
        models[model_name] = {"checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha(checkpoint),
            "initialization_wall_s": time.perf_counter()-initialized,
            "architecture": sampler.model.architecture_identity()}
        for task in tasks:
            codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
            for index, seed in enumerate(CANDIDATE_SEEDS):
                directory = _candidate_dir(output, model_name, task.task_id, index)
                directory.mkdir(parents=True, exist_ok=False)
                latent = _latent(seed)
                np.save(directory / "latent.npy", latent, allow_pickle=False)
                row = {"model": model_name, "task_id": task.task_id,
                    "task_sha256": task.sha256(), "candidate_index": index, "seed": seed,
                    "latent_sha256": hashlib.sha256(latent.tobytes()).hexdigest(),
                    "latent_file_sha256": _sha(directory / "latent.npy"),
                    "status": "STARTED", "proposal_path": None,
                    "raw_closed_loop": "NOT_RUN", "postprocessing": [], "fallback_used": False}
                _write(directory / "slot_started.json", row)
                tick = time.perf_counter()
                try:
                    free, metadata = sampler.sample(task, latent)
                    full = codec.decode_free(free)
                    metadata = {**metadata, "model": model_name,
                        "candidate_index": index, "candidate_budget": 4,
                        "latent_sha256": row["latent_sha256"],
                        "paired_latent_protocol": config["latent_rng"],
                        "checkpoint_file_sha256": models[model_name]["checkpoint_sha256"],
                        "offline_proposal_before_execution": True,
                        "inference_is_online_50Hz_control_call": False,
                        "postprocessing": [], "fallback_used": False}
                    proposal = TrajectoryProposal.from_controls(task, free, origin="diffusion",
                                                                 seed=seed, metadata=metadata)
                    save_tick = time.perf_counter()
                    _write(directory / "proposal.json", proposal.to_dict())
                    np.save(directory / "controls_free.npy", proposal.free_controls, allow_pickle=False)
                    np.save(directory / "controls_full.npy", full, allow_pickle=False)
                    row.update(status="RAW_GENERATED", proposal_path=str((directory / "proposal.json").resolve()),
                        proposal_file_sha256=_sha(directory / "proposal.json"),
                        proposal_sha256=proposal.sha256(), controls_sha256=_array_sha(full),
                        free_controls_sha256=_array_sha(proposal.free_controls),
                        fixed_C0_C1_exact=bool(np.array_equal(full[:2], codec.fixed_controls)),
                        inference_metadata=metadata, artifact_save_wall_s=time.perf_counter()-save_tick)
                except Exception as error:
                    row.update(status="GENERATION_FAILED", failure={"type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc()})
                row["slot_wall_s"] = time.perf_counter()-tick
                _write(directory / "slot_result.json", row)
                rows.append(row)
    after = _sources()
    report = {"schema": "v6_4_b1_paired_sampling_v1", "config_sha256": _sha(config_path),
        "models": models, "records": rows, "slot_count": len(rows),
        "generated_count": sum(r["status"] == "RAW_GENERATED" for r in rows),
        "sources_before": before, "sources_after": after, "sources_unchanged": before == after,
        "wall_s": time.perf_counter()-started, "new_actual_steps": 0,
        "inference_scope": config["inference_scope"], "clipping_or_repair": False}
    _write(output / "sampling.json", report)
    if before != after:
        raise RuntimeError("source changed during paired sampling; retained outputs are not a frozen comparison")
    return report


def _load_candidate(sampling, model_name, task, index):
    directory = _candidate_dir(sampling, model_name, task.task_id, index)
    row = _read(directory / "slot_result.json")
    if (row["model"] != model_name or row["task_sha256"] != task.sha256()
            or row["candidate_index"] != index or row["seed"] != CANDIDATE_SEEDS[index]):
        raise ValueError("candidate slot identity changed")
    if row["status"] != "RAW_GENERATED":
        return None, row
    path = directory / "proposal.json"
    if _sha(path) != row["proposal_file_sha256"]:
        raise ValueError("raw proposal bytes changed")
    proposal = TrajectoryProposal.from_dict(_read(path))
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    full = codec.decode_free(proposal.free_controls)
    if (proposal.sha256() != row["proposal_sha256"] or proposal.postprocessing
            or proposal.origin != "diffusion" or proposal.task_sha256 != task.sha256()
            or _array_sha(full) != row["controls_sha256"]
            or not np.array_equal(np.load(directory / "controls_full.npy", allow_pickle=False), full)
            or not np.array_equal(np.load(directory / "controls_free.npy", allow_pickle=False), proposal.free_controls)):
        raise ValueError("raw controls or attribution changed")
    latent = np.load(directory / "latent.npy", allow_pickle=False)
    if (latent.shape != (30, 17) or latent.dtype != np.float32
            or not np.array_equal(latent, _latent(CANDIDATE_SEEDS[index]))
            or hashlib.sha256(latent.tobytes()).hexdigest() != row["latent_sha256"]):
        raise ValueError("paired candidate latent bytes changed")
    return proposal, row


def _raw_metrics(task, controls, prediction, spec):
    from v6_lite.continuum_model_spec import default_continuum_model_spec
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    samples = codec.sample(controls, np.arange(1351)*.02)
    q, dq, ddq = (samples[k] for k in ("q", "dq", "ddq"))
    domain = default_continuum_model_spec(spec)
    position_excess = np.maximum(np.maximum(spec.planner_lower-q, q-spec.planner_upper), 0.)
    velocity_excess = np.maximum(np.abs(dq)-spec.planner_velocity_limits, 0.)
    pcc_excess = np.maximum(np.maximum(domain.work_domain_lower_rad-q[:, :10],
                                      q[:, :10]-domain.work_domain_upper_rad), 0.)
    requirements = requirement_results(task, prediction)
    return {"initial_boundary_passed": bool(np.array_equal(controls[:2], codec.fixed_controls)
                and np.allclose(q[0], task.initial_planner_q, atol=1e-12, rtol=0.)
                and np.allclose(dq[0], task.initial_planner_dq, atol=1e-12, rtol=0.)),
        "control_point_range_passed": bool(np.all(controls >= spec.planner_lower) and np.all(controls <= spec.planner_upper)),
        "reconstructed_range_passed": bool(np.all(position_excess == 0.)),
        "pcc_work_domain_passed": bool(np.all(pcc_excess == 0.)),
        "velocity_passed": bool(np.all(velocity_excess <= 1e-12)),
        "task_passed": requirements["passed"], "task_requirements": requirements,
        "q_coordinate_min_rad": q.min(axis=0).tolist(), "q_coordinate_max_rad": q.max(axis=0).tolist(),
        "dq_coordinate_max_abs_rad_s": np.abs(dq).max(axis=0).tolist(),
        "ddq_coordinate_max_abs_rad_s2": np.abs(ddq).max(axis=0).tolist(),
        "position_violation_max_rad": float(position_excess.max()),
        "pcc_work_domain_violation_max_rad": float(pcc_excess.max()),
        "velocity_violation_max_rad_s": float(velocity_excess.max()),
        "velocity_violation_state_count": int(np.count_nonzero(np.any(velocity_excess > 1e-12, axis=1))),
        "velocity_violation_coordinate_count": int(np.count_nonzero(velocity_excess > 1e-12)),
        "nominal_prediction_scope": "original 1351-state free-base reconstruction; no terminal progress repair"}


def evaluate_raw_candidate(task, proposal, directory, *, spec=None):
    """Preserve complete nominal prediction even if original raw gate short-circuits."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row = {"task_id": task.task_id, "task_sha256": task.sha256(),
        "raw_passed": False, "geometry_status": "NOT_RUN", "geometry_passed": None,
        "raw_closed_loop": "NOT_RUN", "new_actual_steps": 0,
        "postprocessing": [], "fallback_used": False}
    if proposal is None:
        row.update(status="GENERATION_FAILED_NO_PROPOSAL", rejection_reasons=["generation"], metrics=None)
    else:
        spec = default_v6_lite_robot_spec() if spec is None else spec
        tick = time.perf_counter()
        try:
            codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
            controls = codec.decode_free(proposal.free_controls)
            provider, _ = prepare_provider(task, controls, spec)
            row["prediction_wall_s"] = time.perf_counter()-tick
            save_reference(provider, directory / "nominal_reference.npz")
            _write(directory / "nominal_reference_identity.json", provider.metadata)
            metrics = _raw_metrics(task, controls, provider.prediction, spec)
            gate = gate_proposal(task, proposal, spec, provider=provider, output_path=directory / "gate.json")
            screen = gate.get("nominal_screen") or {}
            geometry = screen.get("geometry")
            reasons = [name for name, key in (("initial_boundary", "initial_boundary_passed"),
                ("control_point_range", "control_point_range_passed"), ("reconstructed_range", "reconstructed_range_passed"),
                ("pcc_work_domain", "pcc_work_domain_passed"), ("velocity", "velocity_passed"),
                ("task_points", "task_passed")) if metrics[key] is not True]
            if geometry is not None:
                row["geometry_status"] = "RUN"
                row["geometry_passed"] = bool(geometry.get("feasible") is True and geometry["minimum_clearance"] >= .005)
                row["geometry"] = geometry
                if not row["geometry_passed"]:
                    reasons.append("nominal_native_geometry")
            if gate["errors"] or screen.get("errors"):
                reasons.append("gate_evidence")
            if not gate["raw_passed"] and not reasons:
                reasons.append("other_original_gate_check")
            row.update(status="RAW_PASSED" if gate["raw_passed"] else "RAW_REJECTED",
                raw_passed=bool(gate["raw_passed"]), metrics=metrics, rejection_reasons=reasons,
                controls_sha256=_array_sha(controls), proposal_sha256=proposal.sha256(),
                gate_path=str((directory / "gate.json").resolve()), gate_wall_s=gate["elapsed_wall_s"])
        except Exception as error:
            row.update(status="RAW_EVIDENCE_FAILURE", metrics=None, rejection_reasons=["prediction_or_evidence"],
                failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    row["wall_s"] = time.perf_counter()-started
    _write(directory / "raw_result.json", row)
    return row


def summarize_raw(rows):
    summaries = {}
    for model_name in MODELS:
        selected = [r for r in rows if r["model"] == model_name]
        by_task = {}
        for row in selected:
            by_task.setdefault(row["task_id"], []).append(row)
        hashes = [r["controls_sha256"] for r in selected if r.get("controls_sha256")]
        reasons = Counter(reason for r in selected for reason in r["rejection_reasons"])
        metric_counts = {name: sum(r.get("metrics") is not None and r["metrics"][key] is True for r in selected)
            for name, key in (("start", "initial_boundary_passed"), ("task", "task_passed"),
                ("control_point_range", "control_point_range_passed"), ("reconstructed_range", "reconstructed_range_passed"),
                ("pcc_work_domain", "pcc_work_domain_passed"), ("velocity", "velocity_passed"))}
        summaries[model_name] = {"task_denominator": 6, "slot_denominator": 24,
            "evaluated_slots": len(selected), "unique_full_controls": len(set(hashes)),
            "generated_slots": len(hashes), "duplicate_slots": len(hashes)-len(set(hashes)),
            "duplicate_fraction_of_generated": ((len(hashes)-len(set(hashes)))/len(hashes) if hashes else None),
            "raw_comprehensive_passed": sum(r["raw_passed"] for r in selected),
            "individual_metric_passed": metric_counts,
            "geometry_run_count": sum(r["geometry_status"] == "RUN" for r in selected),
            "geometry_passed_count": sum(r["geometry_passed"] is True for r in selected),
            "geometry_NOT_RUN_count": sum(r["geometry_status"] == "NOT_RUN" for r in selected),
            "rejection_reason_counts": dict(reasons),
            "tasks": {task_id: {"K1_raw_passed": next(r["raw_passed"] for r in group if r["candidate_index"] == 0),
                "any_of_K4_raw_passed": any(r["raw_passed"] for r in group), "candidate_slots": len(group),
                "unique_full_controls": len({r["controls_sha256"] for r in group if r.get("controls_sha256")}),
                "K4_actual": "NOT_RUN"} for task_id, group in by_task.items()}}
    return summaries


def evaluate_raw(config_path, sampling, output):
    _, tasks = load_candidate_config(config_path)
    sampling_report = _read(Path(sampling) / "sampling.json")
    if sampling_report["config_sha256"] != _sha(config_path) or sampling_report["sources_unchanged"] is not True:
        raise ValueError("sampling configuration or source freeze invalid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before, started = _sources(), time.perf_counter()
    spec = default_v6_lite_robot_spec()
    rows = []
    for model_name in MODELS:
        for task in tasks:
            proposals = []
            for index in range(4):
                proposal, slot = _load_candidate(sampling, model_name, task, index)
                row = evaluate_raw_candidate(task, proposal, _candidate_dir(output, model_name, task.task_id, index), spec=spec)
                row.update(model=model_name, candidate_index=index, seed=CANDIDATE_SEEDS[index],
                           sampling_slot_status=slot["status"])
                rows.append(row)
                if proposal is not None:
                    proposals.append(proposal.free_controls)
            if len(proposals) > 1:
                distances = [float(np.linalg.norm(a-b)) for i, a in enumerate(proposals) for b in proposals[i+1:]]
                _write(output / model_name / task.task_id / "diversity.json", {
                    "definition": "pairwise L2 distance of raw free30x17 controls in radians",
                    "pair_count": len(distances), "distances_rad": distances,
                    "mean_rad": float(np.mean(distances)), "minimum_rad": min(distances), "maximum_rad": max(distances)})
    after = _sources()
    report = {"schema": "v6_4_b1_raw_comparison_v1", "config_sha256": _sha(config_path),
        "sampling_sha256": _sha(Path(sampling) / "sampling.json"), "records": rows,
        "summary": summarize_raw(rows), "slot_count": len(rows),
        "sources_before": before, "sources_after": after, "sources_unchanged": before == after,
        "wall_s": time.perf_counter()-started, "new_actual_steps": 0,
        "raw_closed_loop": "NOT_RUN", "trajectory_optimization_repair_or_fallback": False}
    _write(output / "raw_comparison.json", report)
    if before != after:
        raise RuntimeError("source changed during raw evaluation")
    return report


def _actual_steps(actual_directory):
    records = []
    for subdir in ("traces", "failures"):
        for path in sorted((Path(actual_directory)/subdir).glob("*.npz")):
            with np.load(path, allow_pickle=False) as trace:
                if "torque" in trace:
                    torque = trace["torque"]
                    if torque.ndim != 2 or torque.shape[1] != 67 or not np.all(np.isfinite(torque)):
                        raise ValueError("actual trace torque shape or finiteness invalid")
                    records.append({"path": str(path.resolve()), "sha256": _sha(path), "steps": len(torque)})
    return max((r["steps"] for r in records), default=0), records


def _actual_result_row(task, model_name, raw, repair_report, result, actual_directory, wall_s):
    steps, traces = _actual_steps(actual_directory)
    evaluation = result.get("evaluation") or {}
    success = (steps == 13500 and result.get("complete") is True and result.get("task_success") is True
               and evaluation.get("complete") is True and evaluation.get("evidence_valid") is True
               and evaluation.get("task_success") is True
               and (evaluation.get("execution_contract") or {}).get("passed") is True
               and (evaluation.get("independent_interval") or {}).get("passed") is True
               and (evaluation.get("native_geometry") or {}).get("passed") is True)
    row = {"model": model_name, "task_id": task.task_id, "task_sha256": task.sha256(),
        "candidate_index": 0, "task_denominator_per_model": 6,
        "status": result["status"], "actual_steps": steps, "actual_trace_records": traces,
        "entered_actual": steps > 0, "full_27s_success": bool(success),
        "actual_runner_started": Path(actual_directory).exists(),
        "actual_admission_marker_saved": (Path(actual_directory).parent / "actual_admission_started.json").exists(),
        "raw_closed_loop": "NOT_RUN", "K4_actual": "NOT_RUN", "fallback_used": False,
        "repair_version": "v2", "terminal_progress_repair": True,
        "raw_proposal_sha256": raw.sha256() if raw else None,
        "repair_selected_K": 1, "repair_report_status": (repair_report.get("records") or [{}])[0].get("status"),
        "repair_modification": (repair_report.get("records") or [{}])[0].get("modification"),
        "repair_wall_s": repair_report.get("total_invocation_wall_s"),
        "attempt_wall_s": wall_s, "evaluation_metrics": evaluation.get("metrics"),
        "execution_contract": evaluation.get("execution_contract"),
        "independent_interval": evaluation.get("independent_interval"),
        "native_geometry": evaluation.get("native_geometry"),
        "failure": result.get("failure"), "complete": result.get("complete", False),
        "evaluation_evidence_valid": evaluation.get("evidence_valid", False),
        "success_attribution": "model + frozen v2 projection + frozen terminal progress + frozen controller",
        "wall_deployment": "NOT_MET", "hardware": "NOT_ESTABLISHED"}
    return row


def execute_k1_attempt(task, model_name, sampling, output, execution_config_path):
    """One retained K1; finished rows reuse evidence, unfinished physics never reruns."""
    from v6_4.engineering_repair import repair_proposal, load_selected_proposal
    from v6_4.reference_execution_repair import execute
    output = Path(output)
    if _sha(execution_config_path) != A1_EXECUTION_CONFIG_SHA256:
        raise ValueError("K1 execution configuration differs from the byte-frozen A1 configuration")
    proposal, slot = _load_candidate(sampling, model_name, task, 0)
    identity = {"model": model_name, "task_id": task.task_id, "task_sha256": task.sha256(),
        "raw_proposal_sha256": proposal.sha256() if proposal else None,
        "execution_config_sha256": _sha(execution_config_path), "candidate_index": 0}
    result_path = output / "attempt_result.json"
    if result_path.exists():
        if _read(output / "attempt_started.json")["identity"] != identity:
            raise ValueError("existing K1 attempt differs from frozen inputs")
        saved = _read(result_path)
        if saved.get("sources_unchanged") is not True:
            raise ValueError("existing K1 source-freeze failure remains invalid")
        return saved
    if output.exists():
        raise FileExistsError("unfinished K1 attempt retained; inspect its live process and evidence, never automatically restart")
    output.mkdir(parents=True, exist_ok=False)
    started, before = time.perf_counter(), _sources()
    _write(output / "attempt_started.json", {"identity": identity, "sources_before": before,
        "status": "STARTED", "never_automatically_restart_actual": True,
        "sampling_slot_status": slot["status"]})
    _write(output / "task.json", task.to_dict())
    config = HierarchicalQPConfig(**_read(execution_config_path))
    config.validate()
    try:
        repair = repair_proposal(task, proposal, output / "repair", version="v2")
        selected = load_selected_proposal(repair, output / "repair", K=1)
        if selected is None:
            result = {"status": "REPAIR_REJECTED", "task_success": False, "complete": False,
                      "failure": {"phase": "engineering_repair", "records": repair["records"]}}
        else:
            selected_path = output / "selected_repaired_proposal.json"
            _write(selected_path, selected.to_dict())
            _write(output / "actual_admission_started.json", {"proposal_sha256": selected.sha256(),
                "from_task_initial_state": True, "terminal_progress_repair": True})
            result = execute(output / "task.json", output / "actual", proposal_path=selected_path,
                config=config, terminal_progress_repair=True, method="diffusion",
                attribution={"model": model_name, "candidate_index": 0, "K1_is_slot_zero": True,
                    "raw_proposal_sha256": proposal.sha256(), "repaired_proposal_sha256": selected.sha256(),
                    "raw_credit": False, "fallback_used": False, "new_TEST_used_to_adjust_repair": False,
                    "controller_config_sha256": identity["execution_config_sha256"]})
        row = _actual_result_row(task, model_name, proposal, repair, result, output / "actual",
                                 time.perf_counter()-started)
        after = _sources()
        row.update(sources_unchanged=before == after, sources_after=after)
        _write(result_path, row)
        if before != after:
            raise RuntimeError("source changed during K1 attempt")
        return row
    except Exception as error:
        # This preserves the true partial state.  It does not invent a terminal
        # zero-step result when a pipeline may already have executed physics.
        _write(output / "wrapper_failure.json", {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "actual_directory_exists": (output / "actual").exists(),
            "attempt_requires_inspection": True, "automatic_reexecution_permitted": False})
        raise


def summarize_actual(rows):
    gate_rejections = {"REPAIR_REJECTED", "PROPOSAL_REJECTED", "REFERENCE_REPAIR_REJECTED"}
    return {name: {"task_denominator": 6, "retained_task_attempts": sum(r["model"] == name for r in rows),
        "entered_actual": sum(r["model"] == name and r["entered_actual"] for r in rows),
        "actual_runner_started": sum(r["model"] == name and r.get("actual_runner_started") is True for r in rows),
        "full_27s_success": sum(r["model"] == name and r["full_27s_success"] for r in rows),
        "repair_or_gate_rejections": sum(r["model"] == name and r["status"] in gate_rejections for r in rows),
        "zero_step_other_failures": sum(r["model"] == name and r["actual_steps"] == 0
                                         and r["status"] not in gate_rejections for r in rows),
        "actual_steps": sum(r["actual_steps"] for r in rows if r["model"] == name),
        "raw_closed_loop": "NOT_RUN", "K4_actual": "NOT_RUN",
        "status_counts": dict(Counter(r["status"] for r in rows if r["model"] == name))} for name in MODELS}


def execute_k1(config_path, sampling, output, execution_config_path, *, only=None):
    _, tasks = load_candidate_config(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    valid = {f"{model}:{task.task_id}" for model in MODELS for task in tasks}
    if only is not None and only not in valid:
        raise ValueError("--only must name an existing model:task_id pair")
    for model_name in MODELS:
        for task in tasks:
            if only is None or only == f"{model_name}:{task.task_id}":
                execute_k1_attempt(task, model_name, sampling, output / model_name / task.task_id, execution_config_path)
    rows = [_read(path) for path in sorted(output.glob("*/*/attempt_result.json"))]
    report = {"schema": "v6_4_b1_K1_engineering_comparison_v1", "config_sha256": _sha(config_path),
        "execution_config_sha256": _sha(execution_config_path), "records": rows,
        "summary": summarize_actual(rows), "maximum_task_attempts": 12,
        "raw_closed_loop": "NOT_RUN", "K4_actual": "NOT_RUN"}
    # Every snapshot is additive; a later --only invocation never replaces one.
    _write(output / "summary_snapshots" / f"summary_{time.time_ns()}.json", report)
    return report


def condition_diagnostics(config_path, training_root, sampling, output, *, device="cpu"):
    """Two disclosed condition interventions plus an unordered-context check."""
    import torch
    _, tasks = load_candidate_config(config_path)
    task = tasks[0]
    sampler = _load_sampler(Path(training_root) / "models" / "M1" / "checkpoint.pt", device)
    latent = _latent(CANDIDATE_SEEDS[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original, _ = _load_candidate(sampling, "M1", task, 0)
    if original is None:
        raise ValueError("condition diagnostic requires the frozen M1 K1 output")
    baseline_geometry = diagnostic_shadow_geometry(task, original, output / "original_M1_K1_geometry")
    rows = []
    for intervention in ("task_goal", "obstacle"):
        changed = task.to_dict()
        if intervention == "task_goal":
            point = next(p for p in changed["requirements"] if p["arm"] == "continuum" and p["kind"] == "terminal")
            point["position_m"][0] += .015
        else:
            changed["scenario"]["workspace_obstacles"][0]["center_w"][1] += .015
        variant = TaskSpec.from_dict(changed)
        free, metadata = sampler.sample(variant, latent)
        proposal = TrajectoryProposal.from_controls(variant, free, origin="diffusion", seed=CANDIDATE_SEEDS[0],
            metadata={**metadata, "diagnostic_only": True, "intervention": intervention,
                      "count_in_candidate_slots": False, "checkpoint_selection_used": False})
        directory = output / intervention
        directory.mkdir()
        _write(directory / "task.json", variant.to_dict())
        _write(directory / "proposal.json", proposal.to_dict())
        result = evaluate_raw_candidate(variant, proposal, directory / "raw")
        shadow_geometry = diagnostic_shadow_geometry(variant, proposal, directory / "shadow_geometry")
        delta = proposal.free_controls-original.free_controls
        rows.append({"intervention": intervention, "changed_task_sha256": variant.sha256(),
            "control_delta_L2_rad": float(np.linalg.norm(delta)),
            "control_delta_max_abs_rad": float(np.max(np.abs(delta))),
            "output_changed": bool(np.any(delta != 0.)), "raw_result": result,
            "shadow_geometry": shadow_geometry,
            "output_changed_establishes_obstacle_avoidance": False})
    condition = sampler.condition(task)
    permuted = {key: value.clone() for key, value in condition.items()}
    order = list(range(64)); order[16:24] = list(reversed(order[16:24]))
    for field in ("token_features", "token_types", "token_mask"):
        permuted[field] = permuted[field][:, order]
    initial_noise = torch.as_tensor(latent[None, :], device=condition["global"].device)
    standardized = sampler.model.sample_ddim(condition, initial_noise=initial_noise)
    reordered = sampler.model.sample_ddim(permuted, initial_noise=initial_noise)
    residual = float(torch.max(torch.abs(standardized-reordered)).cpu())
    report = {"schema": "v6_4_b1_condition_intervention_v1", "model": "M1",
        "base_task_sha256": task.sha256(), "latent_sha256": hashlib.sha256(latent.tobytes()).hexdigest(),
        "extra_condition_proposal_count": 2, "count_in_formal_48_slots": False,
        "used_for_checkpoint_or_architecture_selection": False, "new_actual_steps": 0,
        "shadow_geometry_case_count": 3, "baseline_shadow_geometry": baseline_geometry,
        "shadow_geometry_is_raw_acceptance_or_actual_safety": False,
        "rows": rows, "obstacle_joint_permutation": {"standardized_control_max_abs": residual,
            "threshold": 1e-5, "passed": residual <= 1e-5,
            "meaning": "same trained model, same latent and geometry; only typed context order changes"}}
    _write(output / "condition_diagnostics.json", report)
    return report


def diagnostic_shadow_geometry(task, proposal, output, *, spec=None):
    """Unscored geometric observation, preserving rejected main raw gates.

    This always labels its scope as a shadow query.  Native FK can expose
    geometric changes in a task-invalid or range-invalid proposal, but cannot
    make that proposal admissible or establish successful obstacle avoidance.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    spec = default_v6_lite_robot_spec() if spec is None else spec
    controls = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq).decode_free(proposal.free_controls)
    provider, verifier = prepare_provider(task, controls, spec)
    geometry = verifier.verify_qpos_sequence(provider.prediction["full_qpos"]).to_dict()
    report = {"schema": "v6_4_b1_unscored_geometry_intervention_v1",
        "task_sha256": task.sha256(), "proposal_sha256": proposal.sha256(),
        "controls_sha256": _array_sha(controls), "geometry": geometry,
        "state_count": len(provider.prediction["time"]), "grid_period_s": .02,
        "adaptive_subdivisions": 1, "minimum_clearance_threshold_m": .005,
        "uses_original_pair_policy": True, "uses_fixed_base_fk": False,
        "main_raw_geometry_gate_unchanged": True, "changes_raw_acceptance": False,
        "query_pass_establishes_task_or_closed_loop_success": False,
        "new_actual_steps": 0, "wall_s": time.perf_counter()-started}
    _write(output / "shadow_geometry.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze-candidates")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for action in ("sample", "raw", "actual", "diagnostics"):
        command = sub.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if action in ("sample", "diagnostics"):
            command.add_argument("--training-root", type=Path, required=True)
            command.add_argument("--device", default="cpu")
        if action in ("raw", "actual", "diagnostics"):
            command.add_argument("--sampling", type=Path, required=True)
        if action == "actual":
            command.add_argument("--execution-config", type=Path, required=True)
            command.add_argument("--only")
    args = parser.parse_args()
    started, status, failure = time.perf_counter(), 0, None
    try:
        if args.action == "freeze-candidates":
            result = freeze_candidate_config(args.tasks, args.output)
        elif args.action == "sample":
            result = sample_candidates(args.config, args.training_root, args.output, device=args.device)
        elif args.action == "raw":
            result = evaluate_raw(args.config, args.sampling, args.output)
        elif args.action == "actual":
            result = execute_k1(args.config, args.sampling, args.output, args.execution_config, only=args.only)
        else:
            result = condition_diagnostics(args.config, args.training_root, args.sampling, args.output, device=args.device)
        print(json.dumps({"event": "PILOT_COMMAND_TERMINAL", "action": args.action,
            "cli_exit_status": 0, "status_meaning": "saved finite results, not necessarily task success",
            "slots": result.get("slot_count"), "summary": result.get("summary")}), flush=True)
    except Exception as error:
        status, failure = 1, {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        receipt_root = args.output.parent if args.action == "freeze-candidates" else args.output
        _write(receipt_root / "cli_receipts" / f"{args.action}_{time.time_ns()}.json", {
            "argv": sys.argv, "cli_exit_status": status, "wall_s": time.perf_counter()-started,
            "failure": failure, "interruption_or_partial_actual_permits_restart": False})


if __name__ == "__main__":
    main()
