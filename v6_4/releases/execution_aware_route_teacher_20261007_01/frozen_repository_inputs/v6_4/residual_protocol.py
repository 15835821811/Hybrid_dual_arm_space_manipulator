"""Freeze the bounded B.2 task, teacher and comparison protocol before labels.

This module declares references and budgets.  It never executes a controller,
searches for a feasible task, reads an actual trajectory as a condition, or
replaces a failed task.  The old Cartesian positive control is a separate
interface check; none of its local perturbations are new TEST tasks.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from v6_4.task_protocol import TaskSpec, canonical_json, task_from_scenario, validate_task_splits


PROTOCOL_SCHEMA = "v6_4_task_anchored_residual_protocol_v1"
DEFAULT_SEED = 2026100701
SPLIT_COUNTS = (("train", 6), ("val", 2), ("test", 4))
PAIRED_SAMPLE_SEEDS = (2026100711, 2026100712, 2026100713, 2026100714)
COEFFICIENT_BOUND_M = .020
TEACHER_AMPLITUDE_M = .010
CARTESIAN_SPEED_LIMIT_M_S = .24
OLD_A1_OUTPUT = Path("E:/v64a1work_20261004_01/v6_4/output/reference_execution_repair_20261004_01")
OLD_B1_OUTPUT = Path("E:/v64b1work_20261006_01/v6_4/output/architecture_pilot_20261006_01")
POSITIVE_TASK = OLD_A1_OUTPUT / "frozen_inputs/positive_control_task.json"
POSITIVE_RESULT = OLD_A1_OUTPUT / "positive_control/R1_01/result.json"
POSITIVE_EVALUATION = OLD_A1_OUTPUT / "positive_control/R1_01/evaluation/report.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _input_record(path: Path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def fixed_training_protocol():
    """The one predeclared run; exposures are not independent references."""
    return {
        "seed": 64201,
        "draw_seed": 64202,
        "validation_seed": 64203,
        "optimizer_updates": 4000,
        "batch_size": 32,
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01, "gradient_norm_clip": 1.},
        "learning_rate": 1e-4,
        "weight_decay": .01,
        "gradient_norm_clip": 1.,
        "validate_every": 250,
        "curve_every": 50,
        "validation_draws_per_successful_VAL_reference": 16,
        "checkpoint_selection": "minimum_fixed_VAL_denoising_loss_then_earliest_update",
        "normalizer_fit": "successful_TRAIN_references_only",
        "seed_search": False,
        "TEST_selects_checkpoint_or_configuration": False,
        "condition_uses_future_actual": False,
        "eligibility": {
            "required_successful_TRAIN_references": 1,
            "required_successful_VAL_references": 1,
            "required_successful_nonzero_teacher_references": 1,
            "missing_required_success": "NOT_RUN_no_checkpoint",
            "labels": "only_complete_Task_execution_interval_and_declared_whole_body_passed_consumed_z",
        },
        "recommended_DATA_LIMITED_thresholds": {
            "TRAIN_reference_count": 8, "TRAIN_task_count": 4,
            "VAL_reference_count": 2, "VAL_task_count": 2,
        },
        "diffusion": {
            "hidden_layers": 2, "hidden_dim": 128,
            "diffusion_steps": 100, "noise_schedule": "cosine",
            "cosine_s": .008, "beta_cap": .999,
            "parameterization": "epsilon_residual", "objective": "v_mse",
            "ddim_steps": 20, "ddim_eta": 0.,
            "inference_clip_or_project": False,
            "inactive_dimensions": "zero_in_clean_noise_loss_and_generation",
        },
    }


def fixed_retrieval_protocol():
    return {
        "library": "successful_TRAIN_consumed_nonzero_z_only",
        "condition_distance": "Euclidean_after_TRAIN_only_feature_standardization",
        "normalizer_fit": "successful_TRAIN_conditions_only",
        "standard_deviation_floor": 1e-6,
        "compatible_fields": ["interval_mask", "frame", "representation_version", "coefficient_norm_bound_m"],
        "tie_break": ["distance", "task_id", "candidate_index", "plan_sha256"],
        "candidate_transfer": "copy_z_into_query_Task_frozen_reference_definition",
        "zero_residual_fallback": False,
        "TEST_actual_is_condition_or_library_input": False,
        "empty_compatible_library": "PRECHECK_REJECTED_keep_task_denominator",
    }


def _legacy_tasks():
    records = []
    tasks = []
    for name in ("D0", "D1", "D2", "D3"):
        path = OLD_A1_OUTPUT / "frozen_inputs" / name / "actual_task.json"
        task = TaskSpec.from_dict(_read(path))
        records.append({"label": name, **_input_record(path), "task_sha256": task.sha256()})
        tasks.append((name, task))
    path = OLD_B1_OUTPUT / "test_tasks.json"
    payload = _read(path)
    records.append({"label": "B1_TEST_suite", **_input_record(path)})
    for raw in payload["tasks"]:
        task = TaskSpec.from_dict(raw)
        if task.split != "test":
            raise ValueError("old B1 exclusion source must contain TEST only")
        tasks.append(("B1_TEST", task))
    return tasks, records


def _waypoints(task):
    points = np.asarray(task.scenario["continuum_target"]["waypoint_points_m"], dtype=float)
    if points.shape != (7, 3) or not np.all(np.isfinite(points)):
        raise ValueError("B.2 requires seven finite original Cartesian waypoints")
    return points


def legacy_distance_diagnostics(tasks, legacy):
    """Report ancestry and distances without filtering, searching or replacing."""
    rows = []
    for task in tasks:
        points = _waypoints(task)
        for label, old in legacy:
            distances = np.linalg.norm(points - _waypoints(old), axis=1)
            same_numeric_waypoints = bool(np.array_equal(points, _waypoints(old)))
            if task.seed == old.seed or same_numeric_waypoints:
                raise ValueError("new task reuses an excluded seed or numerical waypoint template")
            rows.append({
                "new_task_id": task.task_id, "new_split": task.split,
                "old_label": label, "old_task_id": old.task_id,
                "old_task_sha256": old.sha256(),
                "same_seed": False, "same_numeric_waypoints": False,
                "waypoint_rms_distance_m": float(np.sqrt(np.mean(distances**2))),
                "waypoint_max_distance_m": float(np.max(distances)),
                "waypoint_min_distance_m": float(np.min(distances)),
                "target_pose_translation_distance_m": float(np.linalg.norm(
                    np.asarray(task.target_pose[:3])-np.asarray(old.target_pose[:3]))),
                "target_linear_twist_distance_m_s": float(np.linalg.norm(
                    np.asarray(task.target_twist[:3])-np.asarray(old.target_twist[:3]))),
                "target_angular_twist_distance_rad_s": float(np.linalg.norm(
                    np.asarray(task.target_twist[3:])-np.asarray(old.target_twist[3:]))),
            })
    return {
        "schema": "v6_4_b2_prior_task_exclusion_diagnostics_v1",
        "definition": "fresh_full_seeded_scene_generation_not_jitter_of_old_Task_arrays",
        "old_task_values_used_to_generate_new_tasks": False,
        "old_task_distances_used_to_select_or_replace_new_tasks": False,
        "distance_is_physical_feasibility_or_statistical_independence_proof": False,
        "records": rows,
    }


def generate_residual_tasks(seed=DEFAULT_SEED):
    """Exactly twelve independent seeded declarations; zero actual steps."""
    from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec
    from v6_4.reference_adapter import scenario_from_task
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("protocol seed must be a nonnegative integer")
    spec = default_v6_lite_robot_spec()
    scenes = build_scenarios(spec, V6LiteRunConfig(scenario_count=12, seed=seed))
    if len(scenes) != 12:
        raise ValueError("scene generator must produce exactly twelve scenes")
    tasks = []
    index = 0
    for split, count in SPLIT_COUNTS:
        for local in range(count):
            task_id = f"b2_continuum_route_{split}_{local:03d}"
            scenario = replace(scenes[index], scenario_id=task_id)
            task = task_from_scenario(spec, scenario, task_id=task_id, group_id=task_id,
                family="end_effector_detour", split=split, layout_diagnostics={
                    "definition": "B2_fresh_seeded_original_Cartesian_end_effector_sphere_family",
                    "source_template": "original_Cartesian_generator_formulas_not_old_Task_numeric_values",
                    "source_generator": "v6_lite.run_v6_lite.build_scenarios",
                    "source_seed": seed, "source_scene_index": index,
                    "source_scene_seed": scenario.seed,
                    "ancestry_task_ids": [],
                    "trajectory_feasibility_established": False,
                    "intermediate_route_side_is_not_a_task_requirement": True,
                })
            reconstructed = scenario_from_task(task)
            if canonical_json(reconstructed.to_dict()) != canonical_json(task.scenario):
                raise ValueError("new task scenario failed canonical roundtrip")
            if TaskSpec.from_dict(task.to_dict()).sha256() != task.sha256():
                raise ValueError("new TaskSpec failed canonical roundtrip")
            if task.path_freedom != "free_intermediate_path_between_fixed_requirements":
                raise ValueError("B.2 cannot silently alter the original task path contract")
            if len(task.scenario["workspace_obstacles"]) != 2 or [
                    float(x["radius_m"]) for x in task.scenario["workspace_obstacles"]] != [.035, .025]:
                raise ValueError("B.2 must retain the two supported original sphere radii")
            tasks.append(task)
            index += 1
    validate_task_splits(tasks)
    if len({t.seed for t in tasks}) != 12:
        raise ValueError("new scene seeds must be distinct")
    return tuple(tasks)


def teacher_coefficients(definition, candidate_index):
    """Three immutable modes, each ten millimetres on every active interval."""
    if not isinstance(candidate_index, int) or isinstance(candidate_index, bool) or candidate_index not in (0, 1, 2):
        raise ValueError("there are exactly three fixed teacher candidates")
    mask = np.asarray(definition["interval_mask"], dtype=bool)
    if mask.shape != (6,):
        raise ValueError("reference definition needs six interval mask entries")
    z = np.zeros((6, 2), dtype=float)
    coordinate, sign = ((0, 1.), (0, -1.), (1, 1.))[candidate_index]
    z[mask, coordinate] = sign * TEACHER_AMPLITUDE_M
    return z


def _validate_positive_control(task, result, evaluation):
    if (result.get("task_sha256") != task.sha256()
            or evaluation.get("task_sha256") != task.sha256()
            or result.get("task_success") is not True
            or evaluation.get("task_success") is not True
            or evaluation.get("complete") is not True
            or evaluation.get("evidence_valid") is not True
            or evaluation.get("metrics", {}).get("physics_steps") != 13500):
        raise ValueError("old Cartesian interface source is not a bound complete success")
    for key in ("execution_contract", "independent_interval", "native_geometry"):
        if evaluation.get(key, {}).get("passed") is not True:
            raise ValueError("old Cartesian interface source lacks " + key)


def freeze_residual_protocol(output, seed=DEFAULT_SEED):
    """Exclusively freeze declarations. Root must explicitly invoke this stage.

    ``output`` may already contain its bootstrap identity and frozen execution
    configuration. Any existing protocol-owned path refuses regeneration,
    including a partial freeze; it is never silently resumed or overwritten.
    """
    from v6_4.task_anchored_reference import build_reference_definition, TaskAnchoredResidualPlan
    output = Path(output).resolve()
    bootstrap_path = output / "bootstrap_identity.json"
    execution_path = output / "frozen_execution_config.json"
    reserved = ("plan.json", "tasks.json", "definitions.json", "legacy_diagnostics.json",
                "tasks", "definitions", "teacher_candidates", "test_candidates", "zero_interface")
    if any((output / name).exists() for name in reserved):
        raise FileExistsError("B.2 protocol already started; preserve the existing declaration")
    bootstrap = _read(bootstrap_path)
    config = _read(execution_path)
    if bootstrap.get("frozen_config_sha256") != sha256_file(execution_path):
        raise ValueError("frozen execution config does not match bootstrap")
    if (config.get("task_period_s") != .020
            or config.get("continuum_speed_limit_m_s") != CARTESIAN_SPEED_LIMIT_M_S
            or config.get("rigid_speed_limit_m_s") != CARTESIAN_SPEED_LIMIT_M_S
            or config.get("enable_pcc_braking_guard") is not True):
        raise ValueError("B.2 execution config differs from the frozen timing, speed or braking contract")
    tasks = generate_residual_tasks(seed)
    splits = validate_task_splits(tasks)
    legacy, legacy_inputs = _legacy_tasks()
    diagnostics = legacy_distance_diagnostics(tasks, legacy)
    positive_task = TaskSpec.from_dict(_read(POSITIVE_TASK))
    _validate_positive_control(positive_task, _read(POSITIVE_RESULT), _read(POSITIVE_EVALUATION))
    definitions = {task.task_id: build_reference_definition(task) for task in tasks}
    positive_definition = build_reference_definition(positive_task)
    records, test_records, payloads = [], [], {}
    payloads["tasks.json"] = {
        "schema": "v6_4_task_protocol_v1", "protocol_schema": PROTOCOL_SCHEMA,
        "tasks": [t.to_dict() for t in tasks], "task_hashes": {t.task_id: t.sha256() for t in tasks},
        "splits": splits, "family_counts": {"end_effector_detour": 12},
        "declared_not_feasibility_certified": True,
    }
    payloads["definitions.json"] = definitions
    payloads["legacy_diagnostics.json"] = {**diagnostics, "inputs": legacy_inputs}
    for task in tasks:
        task_path = f"tasks/{task.task_id}.json"
        definition_path = f"definitions/{task.task_id}.json"
        payloads[task_path] = task.to_dict()
        definition = definitions[task.task_id]
        payloads[definition_path] = definition
        if float(definition["coefficient_norm_bound_m"]) != COEFFICIENT_BOUND_M:
            raise ValueError("reference representation changed its frozen coefficient bound")
        if definition.get("frame") != "world" or definition.get("time_mapping") != "identity_physical_time":
            raise ValueError("reference representation changed its frozen coordinate/time contract")
        if task.split in ("train", "val"):
            for candidate_index, mode in enumerate(("positive_first_transverse", "negative_first_transverse", "positive_second_transverse")):
                plan_path = f"teacher_candidates/{task.task_id}/candidate_{candidate_index:02d}/plan.json"
                if definition.get("applicable") is True:
                    candidate = TaskAnchoredResidualPlan.from_definition(
                        definition, teacher_coefficients(definition, candidate_index))
                    raw_plan = candidate.to_dict()
                    candidate_sha = candidate.sha256()
                    status = "DECLARED_NOT_EXECUTED"
                else:
                    # An inapplicable declaration keeps its slot and denominator.
                    raw_plan = {"schema": "v6_4_b2_inapplicable_candidate_slot_v1",
                        "task_id": task.task_id, "definition": definition,
                        "candidate_index": candidate_index, "status": "NOT_APPLICABLE",
                        "intended_z_m": teacher_coefficients(definition, candidate_index).tolist()}
                    candidate_sha = object_sha256(raw_plan)
                    status = "NOT_APPLICABLE"
                payloads[plan_path] = raw_plan
                records.append({
                    "stage": "teacher", "task_id": task.task_id, "split": task.split,
                    "task_path": task_path, "task_sha256": task.sha256(),
                    "definition_path": definition_path,
                    "definition_sha256": definition["definition_sha256"],
                    "candidate_index": candidate_index, "mode": mode,
                    "amplitude_m": TEACHER_AMPLITUDE_M, "plan_path": plan_path,
                    "plan_sha256": candidate_sha, "status": status,
                    "actual_attempt_budget": 1, "replacement_or_fallback": False,
                })
        else:
            zero_path = f"test_candidates/{task.task_id}/E0/plan.json"
            if definition.get("applicable") is True:
                payloads[zero_path] = TaskAnchoredResidualPlan.from_definition(definition, np.zeros((6, 2))).to_dict()
            else:
                payloads[zero_path] = {"schema": "v6_4_b2_inapplicable_candidate_slot_v1",
                    "task_id": task.task_id, "definition": definition, "status": "NOT_APPLICABLE"}
            test_records.append({"task_id": task.task_id, "task_path": task_path,
                "task_sha256": task.sha256(), "definition_path": definition_path,
                "definition_sha256": definition["definition_sha256"], "E0_plan_path": zero_path,
                "method_order": ["E0", "E1", "E2"], "E2_K1_slot": 0,
                "E2_sample_seeds": list(PAIRED_SAMPLE_SEEDS), "actual_attempt_budget": 3,
                "new_task_feasibility": "FEASIBILITY_NOT_ESTABLISHED"})
    if len(records) != 24 or len(test_records) != 4:
        raise AssertionError("fixed B.2 slot counts changed")
    if positive_definition.get("applicable") is not True:
        raise ValueError("declared old Cartesian interface task is unexpectedly inapplicable")
    positive_plan = TaskAnchoredResidualPlan.from_definition(positive_definition, np.zeros((6, 2)))
    payloads["zero_interface/task.json"] = positive_task.to_dict()
    payloads["zero_interface/definition.json"] = positive_definition
    payloads["zero_interface/plan.json"] = positive_plan.to_dict()
    positive_inputs = [_input_record(p) for p in (POSITIVE_TASK, POSITIVE_RESULT, POSITIVE_EVALUATION)]
    plan = {
        "schema": PROTOCOL_SCHEMA, "seed": seed,
        "task_count": 12, "splits": splits, "single_family": "end_effector_detour",
        "task_generation": {"generator": "v6_lite.run_v6_lite.build_scenarios", "scenario_count": 12,
            "scene_seed_rule": "seed+104729*scene_index", "numerical_old_Task_augmentation": False,
            "feasibility_search_or_task_replacement": False, "initial_feasibility": "FEASIBILITY_NOT_ESTABLISHED"},
        "bootstrap": _input_record(bootstrap_path), "frozen_execution_config": _input_record(execution_path),
        "historical_inputs": legacy_inputs + positive_inputs,
        "historical_base_git_head": bootstrap.get("base_git_head"),
        "source_identity": "source_identity.json_frozen_by_driver_before_execution",
        "representation": {"schema": "task_anchored_cartesian_residual_v1", "frame": "world",
            "time_mapping": "identity_physical_time", "maximum_intervals": 6,
            "maximum_dimensions": 12, "coefficient_norm_bound_m": COEFFICIENT_BOUND_M,
            "teacher_amplitude_m": TEACHER_AMPLITUDE_M,
            "support_cutoff_s": float(positive_definition["support_cutoff_s"]),
            "inference_clamp_project_repair_or_fallback": False,
            "strict_path_or_hold_requirements_may_be_removed": False,
            "terminal_progress_repair": "not_applied_Cartesian_original_physical_time_and_25.5_to_27_hold"},
        "simulation": {"dispatch_clock_policy": "research_simulation", "task_period_s": .020,
            "physics_period_s": .002, "duration_s": 27., "wall_20ms_is_gate": False,
            "deployment": "NOT_MET", "administrator_scheduling_trials": False},
        "budget": {"zero_interface_actual": 1, "teacher_slots": 24, "teacher_actual_max": 24,
            "TEST_E2_candidate_slots": 16, "TEST_actual_max": 12, "total_actual_max": 37,
            "rejected_slots_replaced": False, "private_preview_replay_and_queries_counted_separately": True},
        "phase_order": ["T1_representation_unit_checks", "T1_zero_interface_actual",
            "T2_fixed_teacher_24_slots_including_first_nonzero_development_attempt",
            "T3_one_training_run_if_eligible", "T4_E2_fixed_16_reference_candidates",
            "T4_TEST_E0_E1_E2_K1_only", "final_evidence_and_report"],
        "zero_interface": {"task_path": "zero_interface/task.json", "plan_path": "zero_interface/plan.json",
            "definition_path": "zero_interface/definition.json", "task_sha256": positive_task.sha256(),
            "plan_sha256": positive_plan.sha256(), "actual_attempt_budget": 1,
            "old_success_inputs": positive_inputs, "included_in_new_TRAIN_VAL_TEST": False,
            "original_and_zero_QP_input_parity": "same_current_state_and_physical_time",
            "all_old_actual_traces_are_conditions": False},
        "teacher": {"candidate_modes": ["positive_first_transverse", "negative_first_transverse", "positive_second_transverse"],
            "candidates_per_TRAIN_VAL_task": 3, "pattern": "10mm_in_every_active_interval_no_candidate_dependent_mask",
            "actual_order": [r["plan_path"] for r in records], "failed_candidates_retained": True,
            "labels_are_actual_q_fits": False, "extra_candidates_after_failure": False},
        "records": records, "test_records": test_records,
        "training": fixed_training_protocol(), "retrieval": fixed_retrieval_protocol(),
        "comparison": {"methods": {"E0": "zero_residual_Cartesian", "E1": "nearest_successful_TRAIN_residual", "E2": "residual_diffusion"},
            "TEST_task_denominator_per_method": 4, "E2_slots_per_task": 4,
            "E2_sample_seeds": list(PAIRED_SAMPLE_SEEDS), "E2_K1_slot": 0,
            "E2_K4_contains_K1": True, "E2_K4_actual": "NOT_RUN",
            "E1_E2_zero_residual_automatic_fallback": False,
            "TEST_actual_can_update_training_inference_or_selection": False,
            "nonlearning_failure_interpretation": "FEASIBILITY_NOT_ESTABLISHED_not_physical_infeasibility"},
        "reference_precheck": {"continuum_speed_limit_m_s": CARTESIAN_SPEED_LIMIT_M_S,
            "grid_period_s": .002, "amplitude_and_mask": "reject_no_clamp_or_projection",
            "task_anchor_checks": "original_frozen_TaskSpec_and_analytic_reference",
            "old_joint_codec_range": "N/A/new-representation", "nominal_q_ref_whole_body": "N/A/no_q_ref",
            "actual_constraints": "unchanged_QP_braking_preview_guards_interval_and_declared_whole_body",
            "point_clearance_is_whole_body_safety": False},
        "artifact_paths": {"tasks": "tasks.json", "definitions": "definitions.json",
            "legacy_diagnostics": "legacy_diagnostics.json", "source_identity": "source_identity.json"},
        "delivery_verdict_fields": ["research_delivery_complete", "zero_residual_parity_passed",
            "task_anchor_representation_verified", "nonzero_residual_full_task_count",
            "teacher_successful_task_count", "training_executed", "DATA_LIMITED",
            "diffusion_full_task_success_count", "advantage_over_retrieval", "deployment"],
    }
    # All source-dependent declarations are built before the first write. Files
    # are exclusive; their exact bytes are bound into the final frozen plan.
    for relative, payload in payloads.items():
        _write(output / relative, payload)
    plan["artifact_sha256"] = {relative: sha256_file(output / relative) for relative in sorted(payloads)}
    plan["task_hashes"] = {task.task_id: task.sha256() for task in tasks}
    plan["definition_hashes"] = {task_id: d["definition_sha256"] for task_id, d in definitions.items()}
    plan["protocol_sha256"] = object_sha256(plan)
    _write(output / "plan.json", plan)
    return plan


def validate_frozen_protocol(output):
    """Read back exact declarations before any explicit reuse or execution."""
    output = Path(output).resolve()
    plan = _read(output / "plan.json")
    if plan.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported B.2 protocol schema")
    expected = dict(plan)
    claimed = expected.pop("protocol_sha256", None)
    if claimed != object_sha256(expected):
        raise ValueError("frozen protocol content hash differs")
    for relative, digest in plan["artifact_sha256"].items():
        path = (output / relative).resolve()
        if output not in path.parents or sha256_file(path) != digest:
            raise ValueError("frozen protocol artifact differs: " + relative)
    for key in ("bootstrap", "frozen_execution_config"):
        record = plan[key]
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("frozen protocol input differs: " + key)
    for record in plan.get("historical_inputs", []):
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("frozen historical input differs: " + record["path"])
    tasks = tuple(TaskSpec.from_dict(t) for t in _read(output / "tasks.json")["tasks"])
    if ({t.task_id: t.sha256() for t in tasks} != plan["task_hashes"]
            or validate_task_splits(tasks) != plan["splits"]):
        raise ValueError("frozen task identities or splits differ")
    return plan
