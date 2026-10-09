"""Predeclared B.3 obstacle-side pairs and finite route libraries.

Only declared references, initial-state geometry, and scene compilation are
used here. No controller, physics step, inverse kinematics, learned candidate,
actual trajectory, retry, or obstacle-position search is used to make a pair.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .task_protocol import TaskSpec, canonical_json, task_from_scenario, validate_task_splits
from .task_anchored_reference import (
    TaskAnchoredResidualPlan, _target, build_reference_definition, reference_precheck,
)

SCHEMA = "v6_4_b3_route_pair_protocol_v1"
DEFAULT_SEED = 2026100731
KEY_INTERVAL_SLOT = 2
OBSTACLE_INDEX = 1
PAIR_ROLES = ("pilot", "train", "train", "train", "val", "test", "test")
PILOT_MODES = ("z0", "z_plus", "z_minus")
TEACHER_MODES = (*PILOT_MODES, "z_perp")
TEST_METHODS = ("Z0", "R0", "U0", "D_true", "D_swap")


def object_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def fixed_route_protocol(seed=DEFAULT_SEED):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("the mother-scene seed must be a nonnegative integer")
    return {
        "schema": SCHEMA, "mother_seed": seed, "mother_count": 7,
        "mother_seed_rule": "seed + 104729 * mother_index",
        "source_generator": "v6_lite.run_v6_lite.build_scenarios",
        "mother_roles": list(PAIR_ROLES),
        "split_task_counts": {"train": 6, "val": 2, "test": 4},
        "pilot_tasks_separate_from_formal_dataset": True,
        "transfer_scope": "local_single_task_family_pilot_seven_seeded_mother_scenes",
        "key_interval_slot": KEY_INTERVAL_SLOT,
        "route_window": "complete_predeclared_slot_2_interval_open_endpoints",
        "obstacle_index": OBSTACLE_INDEX, "obstacle_radius_m": .025,
        "obstacle_center_rule": "base_reference_at_interval_midpoint + side * offset * first_transverse_basis",
        "obstacle_offset_rule_m": ".043 + .001 * (mother_index % 3)",
        "obstacle_offset_range_m": [.043, .045],
        "side_order": [1, -1], "candidate_modes": list(TEACHER_MODES),
        "z_first_basis_amplitude_m": .012, "z_second_basis_amplitude_m": .010,
        "coefficient_norm_bound_m": .020,
        "unchanged_fields": ["initial_state", "target_motion", "requirements_and_windows",
            "duration", "obstacle_sizes", "other_obstacle", "robot_and_controller"],
        "side_is_task_success_requirement": False,
        "route_direction_label_is_condition_input": False,
        "position_search_or_failed_task_replacement": False,
        "geometry_checks": {"initial": "one_original_policy_whole_body_query_at_declared_qpos",
            "anchors": "continuum_world_anchor_point_to_sphere_surface_distance_only",
            "reference": "original_analytic_Task_and_velocity_precheck",
            "insufficient_as_full_Task_feasibility_proof": True,
            "failed_geometry_keeps_frozen_task_and_consumes_its_slots": True},
        "route_value": {"success_and_independent_safety_required": True,
            "relative_I_route_reduction_min": .10, "absolute_I_route_reduction_min_rad_s": .001,
            "I_route_vectors": "QP_unconstrained_nominal_velocity_minus_selected_velocity",
            "paired_opposite_direction_benefit_required": True,
            "failed_prefix_eligible_for_complete_mean_comparison": False},
        "budget": {"old_checkpoint_DDIM_max": 32, "pilot_actual_slots": 6,
            "teacher_actual_slots": 32, "training_runs_max": 1, "optimizer_updates_max": 4000,
            "new_TEST_DDIM_max": 32, "TEST_actual_slots": 20, "total_new_actual_slots_max": 58,
            "private_preview_independent_replay_geometry_charged_separately": True},
        "pilot_failure_stop": "ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION",
        "deployment": "NOT_MET", "wall_20ms_is_research_gate": False,
    }


def route_coefficients(definition, mode):
    """One fixed interval, no changing mask or searching alternate intervals."""
    if mode not in TEACHER_MODES:
        raise ValueError("unknown fixed route candidate mode")
    mask = np.asarray(definition["interval_mask"], dtype=bool)
    if mask.shape != (6,):
        raise ValueError("the inherited reference needs six mask entries")
    z = np.zeros((6, 2), dtype=float)
    if mode != "z0":
        if not mask[KEY_INTERVAL_SLOT]:
            raise ValueError("NOT_APPLICABLE: frozen key interval slot 2 is disabled")
        coordinate, amplitude = {"z_plus": (0, .012), "z_minus": (0, -.012),
                                 "z_perp": (1, .010)}[mode]
        z[KEY_INTERVAL_SLOT, coordinate] = amplitude
    return z


def _without_pair_changes(task):
    value = task.to_dict()
    value.pop("task_id")
    value["scenario"].pop("scenario_id")
    value["scenario"]["workspace_obstacles"][OBSTACLE_INDEX].pop("center_w")
    return value


def validate_pair_contract(positive, negative):
    if canonical_json(_without_pair_changes(positive)) != canonical_json(_without_pair_changes(negative)):
        raise ValueError("paired Tasks differ outside obstacle center and identity metadata")
    if positive.group_id != negative.group_id:
        raise ValueError("paired Tasks need one mother-scene group")
    first, second = build_reference_definition(positive), build_reference_definition(negative)
    for field in ("intervals_s", "transverse_bases", "interval_mask"):
        if first[field] != second[field]:
            raise ValueError("obstacle-side substitution changed the residual geometry")
    return True


def _geometry_check(spec, task, definition):
    import mujoco
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
    from .reference_adapter import scenario_from_task
    from v6_lite.runtime_command import model_id
    scene = scenario_from_task(task)
    verifier = WholeBodyCollisionVerifier(spec, scene.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    initial = verifier.verify_qpos_sequence(np.asarray([task.initial_qpos]))
    model, data = verifier.model, verifier.data
    obstacle = scene.obstacles[OBSTACLE_INDEX]
    obstacle_name = f"v5_workspace_sphere_{OBSTACLE_INDEX:03d}_{obstacle.name}"
    continuum_bodies = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
    related = [p for p in verifier.pairs if p.pair_class == "arm_obstacle"
        and p.geom_b_name == obstacle_name and int(model.geom_bodyid[p.geom_a]) in continuum_bodies]
    related_distances = [float(mujoco.mj_geomDistance(model, data, p.geom_a, p.geom_b, 2.5, np.zeros(6)))
                         for p in related]
    anchors = [{"point_id": point.point_id, "time_s": point.time_s,
        "point_sphere_surface_distance_m": float(np.linalg.norm(np.asarray(point.position_m)-obstacle.center)-obstacle.radius)}
        for point in task.requirements if point.arm == "continuum" and point.frame == "world"]
    anchor_passed = bool(anchors) and all(a["point_sphere_surface_distance_m"] >= .005 for a in anchors)
    result = {
        "schema": "v6_4_b3_pair_geometry_precheck_v1", "task_id": task.task_id,
        "task_sha256": task.sha256(), "initial_whole_body": initial.to_dict(),
        "initial_related_continuum_obstacle_clearance_m": min(related_distances) if related_distances else None,
        "initial_related_pairs": len(related), "continuum_world_anchor_point_checks": anchors,
        "anchor_point_passed": anchor_passed, "initial_geometry_passed": initial.feasible,
        "key_interval_active": bool(definition["interval_mask"][KEY_INTERVAL_SLOT]),
        "passed": initial.feasible and anchor_passed and bool(definition["interval_mask"][KEY_INTERVAL_SLOT]),
        "compiled_scene_model_sha256": model_id(model, spec.runtime_contract_sha256()),
        "scene_declaration_sha256": object_sha256(scene.to_dict()),
        "robot_model_contract_sha256": spec.runtime_contract_sha256(),
        "native_distance_queries": initial.query_count + len(related),
        "physics_steps_executed": 0, "private_preview_steps": 0, "IK_or_control_rollouts": 0,
        "initial_state_checked_count": 1, "no_position_search": True,
        "anchor_check_scope": "points_only_not_arm_geometry_or_reachability",
        "full_Task_feasibility_established": False, "continuous_time_certified": False,
    }
    return result


def generate_route_pairs(seed=DEFAULT_SEED, *, geometry_checks=True, reference_checks=True):
    """Build all seven mother declarations once, before any B.3 actual result.

    The pilot pair has TaskSpec split=train for schema compatibility, but its
    role is pilot and it is excluded from the twelve formal tasks returned.
    No failed check changes the fixed seed, offset, key interval, or pair.
    """
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec, build_scenarios, V6LiteRunConfig
    protocol = fixed_route_protocol(seed)
    spec = default_v6_lite_robot_spec()
    mothers = build_scenarios(spec, V6LiteRunConfig(scenario_count=7, seed=seed))
    tasks, pilot_tasks, pair_records, definitions, checks, candidates = [], [], [], {}, {}, []
    for mother_index, (role, mother) in enumerate(zip(PAIR_ROLES, mothers)):
        group_id = f"b3_mother_{mother_index:02d}"
        split = "train" if role == "pilot" else role
        common_layout = {"definition": "B3_frozen_reference_midpoint_single_sphere_side_pair",
            "role": role, "mother_index": mother_index, "mother_seed": mother.seed,
            "source_generator": protocol["source_generator"], "key_interval_slot": KEY_INTERVAL_SLOT,
            "sphere_offset_m": .043+.001*(mother_index % 3), "ancestry_task_ids": [],
            "local_transfer_pilot": True, "trajectory_feasibility_established": False,
            "intermediate_route_side_is_not_a_task_requirement": True}
        base_task = task_from_scenario(spec, mother, task_id=group_id+"_source", group_id=group_id,
            family="end_effector_detour", split=split, layout_diagnostics=common_layout)
        base_definition = build_reference_definition(base_task)
        if not base_definition["interval_mask"][KEY_INTERVAL_SLOT]:
            raise ValueError(f"{group_id}: NOT_APPLICABLE_frozen_key_interval_disabled_no_replacement")
        interval = base_definition["intervals_s"][KEY_INTERVAL_SLOT]
        midpoint = float(np.mean(interval))
        point, _ = _target(base_task).sample(midpoint)
        basis = np.asarray(base_definition["transverse_bases"][KEY_INTERVAL_SLOT])
        offset = common_layout["sphere_offset_m"]
        pair = []
        for side, side_name in ((1, "c_plus"), (-1, "c_minus")):
            task_id = group_id+"_"+side_name
            spheres = list(mother.obstacles)
            original = spheres[OBSTACLE_INDEX]
            if original.radius != .025:
                raise ValueError("the moved obstacle must keep the original 25mm sphere radius")
            spheres[OBSTACLE_INDEX] = replace(original, center=point+side*offset*basis[:, 0])
            scene = replace(mother, scenario_id=task_id, obstacles=tuple(spheres))
            task = task_from_scenario(spec, scene, task_id=task_id, group_id=group_id,
                family="end_effector_detour", split=split, requirements=base_task.requirements,
                layout_diagnostics=common_layout)
            definition = build_reference_definition(task)
            pair.append(task)
            definitions[task_id] = definition
            if geometry_checks:
                checks[task_id] = _geometry_check(spec, task, definition)
            modes = PILOT_MODES if role == "pilot" else TEACHER_MODES if role in ("train", "val") else ("z0",)
            for mode in modes:
                residual = TaskAnchoredResidualPlan.from_definition(definition, route_coefficients(definition, mode))
                record = {"task_id": task_id, "role": role, "mode": mode,
                    "plan": residual.to_dict(), "plan_sha256": residual.sha256(),
                    "T_route_s": interval, "key_interval_slot": KEY_INTERVAL_SLOT}
                if reference_checks:
                    record["reference_precheck"] = reference_precheck(task, residual)
                candidates.append(record)
            (pilot_tasks if role == "pilot" else tasks).append(task)
        validate_pair_contract(*pair)
        pair_records.append({"group_id": group_id, "role": role, "split": split,
            "mother_index": mother_index, "mother_seed": mother.seed,
            "mother_scene_sha256": object_sha256(mother.to_dict()),
            "task_ids": [t.task_id for t in pair], "task_sha256": [t.sha256() for t in pair],
            "key_interval_slot": KEY_INTERVAL_SLOT, "T_route_s": interval,
            "midpoint_s": midpoint, "base_reference_midpoint_m": point.tolist(),
            "transverse_basis_world": basis.tolist(), "sphere_offset_m": offset,
            "sphere_centers_m": [t.scenario["workspace_obstacles"][OBSTACLE_INDEX]["center_w"] for t in pair],
            "pair_contract_passed": True, "side_order": [1, -1],
            "scene_identities": [checks[t.task_id]["compiled_scene_model_sha256"] for t in pair] if geometry_checks else [],
            "full_Task_feasibility_established": False})
    split_ids = validate_task_splits(tasks)
    if {key: len(value) for key, value in split_ids.items()} != protocol["split_task_counts"]:
        raise ValueError("the frozen pair split cardinalities changed")
    if len({t.seed for t in [*pilot_tasks, *tasks]}) != 7:
        raise ValueError("the seven mother seeds must remain distinct")
    return {"protocol": protocol, "pilot_tasks": tuple(pilot_tasks), "tasks": tuple(tasks),
        "pair_records": pair_records, "definitions": definitions, "geometry_checks": checks,
        "candidate_records": candidates, "splits": split_ids,
        "physics_steps_executed": 0, "actual_attempts": 0,
        "geometry_query_count": sum(c["native_distance_queries"] for c in checks.values())}


def freeze_route_pair_inputs(output, seed=DEFAULT_SEED):
    """Exclusive new inputs only; plan.json and execution remain the caller's."""
    output = Path(output).resolve()
    suite = generate_route_pairs(seed)
    def save(relative, value):
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if (output/"route_pair_inputs.json").exists():
        raise FileExistsError("frozen pair inputs already exist; no replacement is permitted")
    artifacts = []
    for task in [*suite["pilot_tasks"], *suite["tasks"]]:
        root = f"tasks/{task.task_id}"
        artifacts.extend([save(root+"/task.json", task.to_dict()),
            save(root+"/definition.json", suite["definitions"][task.task_id]),
            save(root+"/geometry_precheck.json", suite["geometry_checks"][task.task_id])])
    candidate_records = []
    for record in suite["candidate_records"]:
        record = dict(record)
        root = f"fixed_candidates/{record['task_id']}/{record['mode']}"
        plan = record.pop("plan")
        precheck = record.pop("reference_precheck")
        plan_record, check_record = save(root+"/plan.json", plan), save(root+"/reference_precheck.json", precheck)
        artifacts.extend([plan_record, check_record])
        candidate_records.append({**record, "plan_path": plan_record["path"],
            "plan_file_sha256": plan_record["sha256"], "precheck_path": check_record["path"],
            "precheck_sha256": check_record["sha256"], "reference_precheck_passed": precheck["passed"]})
    manifest = {"schema": SCHEMA, "protocol": suite["protocol"],
        "pair_records": suite["pair_records"], "splits": suite["splits"],
        "pilot_task_ids": [t.task_id for t in suite["pilot_tasks"]],
        "task_ids": [t.task_id for t in suite["tasks"]], "candidate_records": candidate_records,
        "artifacts": artifacts, "geometry_query_count": suite["geometry_query_count"],
        "physics_steps_executed": 0, "actual_attempts": 0,
        "actual_outcomes_read_to_construct_tasks": False,
        "learned_outputs_read_to_construct_tasks": False}
    save("route_pair_inputs.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = freeze_route_pair_inputs(args.output, args.seed)
    print(json.dumps({"event": "B3_PAIR_INPUTS_FROZEN", "pairs": len(result["pair_records"]),
        "geometry_query_count": result["geometry_query_count"], "physics_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
