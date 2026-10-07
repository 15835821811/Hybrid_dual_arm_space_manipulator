"""Fixed B.3.1 declarations and execution; no learning or candidate search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np

from .residual_execution import sha, source_guard, write, execute_residual_attempt
from .conditional_execution import ExecutionCostLedger
from .task_protocol import TaskSpec, canonical_json
from .task_anchored_reference import (
    TaskAnchoredResidualPlan, build_reference_definition, reference_precheck,
    REPRESENTATION_VERSION, PLATEAU_REPRESENTATION_VERSION, _target,
)
from .route_pair_protocol import _geometry_check
from .route_quality_dataset import build_route_quality, safety_eligibility, GATE_NAMES


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("E:/v64b3work_20261007_01")
OLD = OLD_ROOT/"v6_4/output/conditional_route_value_20261007_01"
BASE = "bdd6df6b4c68a6c95872f4639b8b8269fb6856c7"
BRANCH = "v6.4-b3-1-execution-aware-route-teacher"
TARGET = Path("C:/Users/admin/.codex/attachments/030cd1e3-81d3-4bed-9d63-c5ae79d53278/pasted-text-1.txt")
MODES = (("z0", "v1", 0.), ("v1_plus12", "v1", .012),
         ("v1_minus12", "v1", -.012), ("v2_plus12", "v2", .012),
         ("v2_minus12", "v2", -.012), ("v2_plus20", "v2", .020),
         ("v2_minus20", "v2", -.020))
ALLOWED_INHERITED_CHANGES = {
    "v6_4/task_anchored_reference.py", "v6_4/residual_execution.py",
    "v6_4/conditional_execution.py", "v6_lite/hierarchical_qp.py",
    "v6_lite/run_v6_lite.py",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def event(name, **fields):
    print(json.dumps({"event": name, **fields}, ensure_ascii=False), flush=True)


def _invariants(value):
    value = json.loads(canonical_json(value))
    value.pop("task_id")
    value["scenario"].pop("scenario_id")
    value["scenario"]["workspace_obstacles"][1].pop("center_w")
    value.pop("layout")
    return value


def prepare(output):
    output = Path(output).resolve()
    if (output/"plan.json").exists():
        raise FileExistsError("Plan already frozen; do not regenerate or tune it")
    output.mkdir(parents=True, exist_ok=True)
    (output/"inputs").mkdir(exist_ok=False)
    shutil.copyfile(TARGET, output/"inputs/EXECUTION_TARGET.md")
    shutil.copyfile(OLD/"frozen_execution_config.json", output/"frozen_execution_config.json")
    old_identity = read(OLD/"source_identity.json")
    old_seal = read(OLD/"manifest.json")
    assert sha(OLD/"manifest.json") == read(OLD/"verification.json")["manifest_sha256"]
    spec = __import__("v6_lite.run_v6_lite", fromlist=["default_v6_lite_robot_spec"]).default_v6_lite_robot_spec()
    tasks, slots, artifacts = [], [], []
    initial_query_count = 0
    for distance in (.043, .055):
        for side in (1, -1):
            old_id = "b3_mother_00_c_"+("plus" if side == 1 else "minus")
            source_task = OLD/"tasks"/old_id/"task.json"
            task_data = read(source_task)
            task = TaskSpec.from_dict(task_data)
            original = build_reference_definition(task)
            interval = original["intervals_s"][2]
            basis = np.asarray(original["transverse_bases"][2])
            midpoint = float(np.mean(interval))
            position = _target(task).sample(midpoint)[0]
            center = position+side*distance*basis[:, 0]
            if distance == .043:
                assert np.array_equal(center, task_data["scenario"]["workspace_obstacles"][1]["center_w"])
            else:
                task_data["task_id"] = "b31_mother_00_d055_c_"+("plus" if side == 1 else "minus")
                task_data["scenario"]["scenario_id"] = task_data["task_id"]
                task_data["scenario"]["workspace_obstacles"][1]["center_w"] = center.tolist()
                task_data["layout"] = {**task_data["layout"], "role": "development_teacher",
                    "sphere_offset_m": distance, "definition": "B31_fixed_reference_midpoint_offsets_43_55mm",
                    "trajectory_feasibility_established": False}
                assert _invariants(task_data) == _invariants(read(source_task))
                task = TaskSpec.from_dict(task_data)
            directory = output/"tasks"/task.task_id
            directory.mkdir(parents=True, exist_ok=False)
            if distance == .043:
                shutil.copyfile(source_task, directory/"task.json")
                shutil.copyfile(OLD/"tasks"/old_id/"geometry_precheck.json", directory/"geometry_precheck.json")
                geometry = read(directory/"geometry_precheck.json")
                geometry_role = "EXACT_OLD_DECLARATION_PRECHECK_REUSED_ZERO_NEW_QUERIES"
            else:
                write(directory/"task.json", task.to_dict())
                geometry = _geometry_check(spec, task, build_reference_definition(task))
                initial_query_count += geometry["native_distance_queries"]
                write(directory/"geometry_precheck.json", geometry)
                geometry_role = "NEW_DECLARED_INITIAL_AND_ANCHOR_CHECK_NO_ROLLOUT"
            task_record = {"task_id": task.task_id, "task_sha256": task.sha256(),
                "distance_m": distance, "side": side, "mother_scene": "b3_mother_00",
                "task_path": str((directory/"task.json").resolve()), "task_file_sha256": sha(directory/"task.json"),
                "route_interval_s": interval, "key_interval_slot": 2,
                "sphere_center_w": center.tolist(), "geometry_precheck_passed": geometry["passed"],
                "geometry_precheck_role": geometry_role,
                "obstacle_name": task.scenario["workspace_obstacles"][1]["name"],
                "development_only_not_independent_TEST": True}
            tasks.append(task_record)
            for mode_index, (mode, version, amplitude) in enumerate(MODES):
                slot_id = f"EA_{len(slots):02d}"
                definition = build_reference_definition(task, version=(
                    REPRESENTATION_VERSION if version == "v1" else PLATEAU_REPRESENTATION_VERSION))
                z = np.zeros((6, 2)); z[2, 0] = amplitude
                candidate = TaskAnchoredResidualPlan.from_definition(definition, z)
                candidate_dir = output/"candidates"/slot_id
                candidate_dir.mkdir(parents=True, exist_ok=False)
                old_slot = None
                if distance == .043 and mode_index < 3:
                    old_slot = f"PILOT_{(0 if side == 1 else 3)+mode_index:02d}"
                    old_plan = OLD/"pilot/attempts"/old_slot/"plan.json"
                    assert read(old_plan) == candidate.to_dict(), "Old v1 serialization changed"
                    shutil.copyfile(old_plan, candidate_dir/"plan.json")
                else:
                    write(candidate_dir/"plan.json", candidate.to_dict())
                precheck = reference_precheck(task, candidate)
                write(candidate_dir/"reference_precheck.json", precheck)
                slots.append({"slot_id": slot_id, "task_id": task.task_id,
                    "mode": mode, "reference_version": version, "amplitude_m": amplitude,
                    "distance_m": distance, "side": side, "plan_sha256": candidate.sha256(),
                    "plan_path": str((candidate_dir/"plan.json").resolve()),
                    "plan_file_sha256": sha(candidate_dir/"plan.json"),
                    "task_path": task_record["task_path"], "task_sha256": task.sha256(),
                    "reference_precheck_passed": precheck["passed"],
                    "geometry_precheck_passed": geometry["passed"],
                    "old_source_slot": old_slot,
                    "old_attempt_path": str(OLD/"pilot/attempts"/old_slot) if old_slot else None,
                    "old_quality_path": str(OLD/"pilot/quality"/old_slot/"route_quality.json") if old_slot else None,
                    "source_role": "REUSE_ORIGINAL_EVIDENCE" if old_slot else "NEW_FIXED_ACTUAL_SLOT"})
    for directory in (output/"tasks", output/"candidates"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                artifacts.append({"path": path.relative_to(output).as_posix(), "sha256": sha(path)})
    task_manifest = {"schema": "v64_b31_task_manifest_v1", "tasks": tasks, "slots": slots,
        "frozen_artifacts": artifacts, "new_initial_precheck_geometry_queries": initial_query_count,
        "new_physics_steps": 0, "old_six_reuse_expected": 6, "new_actual_slots_expected": 22,
        "formal_old_B3_TEST_used": False}
    write(output/"task_manifest.json", task_manifest)
    plan = {"schema": "v64_b31_frozen_plan_v1", "run_id": output.name,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "base_publication_commit": BASE,
        "branch": BRANCH, "user_target_sha256": sha(TARGET),
        "old_B3_algorithm_producer": old_identity["algorithm_producer_commit"],
        "old_B3_manifest_sha256": sha(OLD/"manifest.json"),
        "task_manifest_sha256": sha(output/"task_manifest.json"),
        "old_B3_route_value_result_immutable": "ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION",
        "reference_versions": {"v1": REPRESENTATION_VERSION, "v2": PLATEAU_REPRESENTATION_VERSION},
        "plateau": {"rise_fraction": .25, "fall_start_fraction": .75},
        "tasks": tasks, "slots": slots,
        "budget": {"slot_limit": 28, "expected_reused_old_slots": 6,
            "expected_new_actual_slots": 22, "maximum_new_actual_slots": 22,
            "training": 0, "model_sampling": 0, "seed_search": 0,
            "replacement_or_retry": False, "new_mass_video_or_PDF": False},
        "teacher": {"eligible": "complete27s_and_original_five_independent_gates",
            "primary_cost": "I_route_rad_s_17D_clipped_nominal_to_selected_RMS",
            "tie_band_absolute_rad_s": .001,
            "tie_band_scope": "fixed engineering preference resolution, not a statistical confidence bound",
            "tie_group_rule": "anchor at group minimum; no transitive chaining",
            "within_tie_break": ["full_continuum_path_length_m", "coefficient_norm_m", "candidate_ID"],
            "report_minimum_cost_and_tie_selected_separately": True,
            "geometric_rule": "v2 +/-12mm, choose direction opposite sphere side",
            "geometric_rule_side_mapping": {"+1": "v2_minus12", "-1": "v2_plus12"},
            "constant_policy_candidate_set": [row[0] for row in MODES],
            "V_cond": "min_z mean_c J(z,c) - mean_c min_z J(z,c)",
            "failure_cost_imputation": False,
            "incomplete_matrix": "report success rates and common fully complete task subset; value on full matrix unavailable",
            "old_B3_gate": {"relative_reduction": .10, "absolute_reduction_rad_s": .001,
                "require_comparison_with_zero_and_opposite": True}},
        "invariants": {"controller_actions_weights_safety_tolerances": "UNCHANGED",
            "source_changes": "v2 reference and optional diagnostic logging/binding only",
            "residual_dim": 12, "per_segment_coefficient_norm_m": .020,
            "reference_cutoff_s": 23.98, "horizon_s": 27., "planning_period_s": .020,
            "physics_period_s": .002, "wall_20ms_research_gate": False,
            "deployment": "NOT_MET", "continuous_time_certified": False,
            "robot_target_native_hz": 500, "whole_body_boundary_hz": 50,
            "whole_body_configuration_subdivisions": 4},
        "T0": "saved_trajectory_diagnosis_only_no_replay_no_vector_reconstruction",
        "outputs": ["baseline_diagnosis.json", "reference_tests.json", "task_manifest.json",
            "attempts", "quality_matrix.json", "teacher_records.json", "REPORT.md",
            "manifest.json", "two_reference_actual_figures", "all_candidates_table",
            "independent_review", "separate_GitHub_branch_and_refreshed_current_visualizations"]}
    write(output/"plan.json", plan)
    event("B31_PLAN_FROZEN", slots=len(slots), expected_new_actual=22,
          reused_old=6, plan_sha256=sha(output/"plan.json"))
    return plan


def freeze_source(output, checks_path):
    output, checks_path = Path(output).resolve(), Path(checks_path).resolve()
    assert read(checks_path)["passed"] is True
    assert not git("status", "--porcelain"), "Commit reviewed source before freeze"
    plan = read(output/"plan.json")
    old = read(OLD/"source_identity.json")
    parity = []
    for relative, digest in old["source_sha256"].items():
        assert sha(OLD_ROOT/relative) == digest
        first, current = (OLD_ROOT/relative).read_bytes(), (ROOT/relative).read_bytes()
        same = first.replace(b"\r\n", b"\n") == current.replace(b"\r\n", b"\n")
        assert same or relative in ALLOWED_INHERITED_CHANGES, relative
        parity.append({"path": relative, "old_sha256": digest, "new_sha256": sha(ROOT/relative),
            "change": "UNCHANGED_OR_LINE_ENDING_ONLY" if same else "DECLARED_REFERENCE_OR_DIAGNOSTIC_ONLY_CHANGE"})
    names = set(old["source_sha256"])
    # New reducers/plots have a separate publication producer and never enter
    # the actual path. All inherited executable source remains frozen above.
    names.update(("v6_4/execution_aware_route_teacher.py", "v6_4/execution_aware_baseline.py"))
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"v6_4/tests").glob("*execution_aware*.py"))
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"v6_4/tests").glob("*execution_diagnostics*.py"))
    protected = {str((output/row["path"]).resolve()): row["sha256"]
        for row in read(output/"task_manifest.json")["frozen_artifacts"]}
    for name in ("plan.json", "task_manifest.json", "frozen_execution_config.json", "inputs/EXECUTION_TARGET.md",
                 "baseline_diagnosis.json", "baseline_inputs_manifest.json", "protocol_clarifications.json",
                 "reference_tests.json", "execution_diagnostics_checks.json", "execution_diagnostics_interface.md"):
        protected[str(output/name)] = sha(output/name)
    protected[str(checks_path)] = sha(checks_path)
    for name, row in read(OLD/"manifest.json")["payload"].items():
        protected[str(OLD/name)] = row["sha256"]
    for name in ("manifest.json", "verification.json", "source_identity.json"):
        protected[str(OLD/name)] = sha(OLD/name)
    identity = {"schema": "v64_b31_source_identity_v1", "git_head": git("rev-parse", "HEAD"),
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "base_publication_commit": BASE, "old_B3_algorithm_producer": old["algorithm_producer_commit"],
        "source_sha256": {name: sha(ROOT/name) for name in sorted(names)},
        "protected_artifacts": protected, "inherited_source_comparison": parity,
        "declared_changed_inherited_files": sorted(ALLOWED_INHERITED_CHANGES),
        "checks_sha256": sha(checks_path), "python": sys.version, "platform": platform.platform(),
        "control_decisions_changed": False, "new_reference_version": True,
        "diagnostic_logging_only_for_QP": True, "training_authorized": False}
    write(output/"source_identity.json", identity)
    source_guard(output/"source_identity.json")
    event("B31_SOURCE_FROZEN", head=identity["git_head"], source_files=len(names),
          protected_files=len(protected), plan_sha256=sha(output/"plan.json"))


def _reused_slot(slot):
    from .execution_aware_baseline import validate_reuse_inputs
    validate_reuse_inputs(OLD, slot["old_source_slot"], task_path=slot["task_path"],
        plan_path=slot["plan_path"], slot_alias=slot["slot_id"])
    old_dir = Path(slot["old_attempt_path"])
    result = read(old_dir/"attempt_result.json")
    for relative, name in (("task.json", "task_path"), ("plan.json", "plan_path")):
        assert (old_dir/relative).read_bytes() == Path(slot[name]).read_bytes(), relative
    assert result["full_task_success"] and result["actual_steps"] == 13500
    for key in ("trace", "evaluation"):
        assert sha(result[key+"_path"]) == result[key+"_sha256"]
    quality_path = Path(slot["old_quality_path"])
    quality = read(quality_path)
    assert quality["quality_label_eligible"] and quality["safety"]["full_task_and_original_safety_passed"]
    for path, digest in quality["sources"].items():
        assert sha(path) == digest
    return result, quality


def _quality_with_retained_failure(output, directory, attempt, task, residual, slot, result):
    """Keep a failed quality stage in its fixed slot, including spent queries."""
    identity_path = output/"source_identity.json"
    source_guard(identity_path)
    bound = {str(attempt/"attempt_result.json"): sha(attempt/"attempt_result.json"),
        str(attempt/"task.json"): sha(attempt/"task.json")}
    for key in ("trace", "evaluation"):
        if result.get(key+"_path"):
            assert sha(result[key+"_path"]) == result[key+"_sha256"]
            bound[result[key+"_path"]] = result[key+"_sha256"]
    if (attempt/"plan.json").exists():
        bound[str(attempt/"plan.json")] = sha(attempt/"plan.json")
    quality_dir = output/"quality"/slot["slot_id"]
    ledger = ExecutionCostLedger()
    failed = None
    try:
        with ledger.installed(), ledger.scope("route_quality"):
            quality = build_route_quality(attempt, [residual.definition["intervals_s"][2]],
                task.scenario["workspace_obstacles"][1]["name"], quality_dir,
                candidate_name=slot["mode"])
        quality_path = quality_dir/"route_quality.json"
    except Exception as error:
        failed = {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "retry_or_replacement": False}
    # Source/input integrity failures are still hard stops, never quality labels.
    source_guard(identity_path)
    for path, digest in bound.items():
        assert sha(path) == digest, "Actual evidence changed during quality stage: "+path
    costs = ledger.to_dict()
    write(directory/"route_quality_cost_ledger.json", costs)
    if failed is not None:
        quality_dir.mkdir(parents=True, exist_ok=True)
        evaluation = result.get("evaluation") or {}
        quality = {"schema": "v64_b31_failed_quality_stage_v1", "slot_id": slot["slot_id"],
            "task_id": task.task_id, "task_sha256": task.sha256(), "status": result["status"],
            "quality_stage_status": "QUALITY_POSTPROCESS_FAILED",
            "route_intervals_s": [residual.definition["intervals_s"][2]],
            "route_obstacle_name": task.scenario["workspace_obstacles"][1]["name"],
            "candidate_name": slot["mode"], "z_m": residual.z_m.tolist(),
            "quality_label_eligible": False, "full_metrics": None, "failed_prefix_metrics": None,
            "safety": {**safety_eligibility(result, evaluation), "path_quality_comparison_eligible": False},
            "independent_safety_and_task_metrics": {name: evaluation.get(name) for name in GATE_NAMES},
            "metric_unavailable": {"reason": "QUALITY_POSTPROCESS_FAILED", "error": failed},
            "sources": bound, "costs": {"additional_route_quality_geometry_queries": costs["native_geometry_query_calls"]},
            "deployment": "NOT_MET"}
        quality_path = quality_dir/"route_quality_failure.json"
        write(quality_path, quality)
        event("B31_QUALITY_FAILURE_RETAINED", slot_id=slot["slot_id"], error=failed["type"])
    else:
        assert quality["costs"]["additional_route_quality_geometry_queries"] == costs["native_geometry_query_calls"]
    assert all(costs[key] == 0 for key in ("actual_physics_steps", "private_preview_physics_steps",
        "independent_saved_torque_replay_steps", "qp_solve_calls"))
    return quality, quality_path, costs


def run(output):
    output = Path(output).resolve(); plan = read(output/"plan.json")
    source_guard(output/"source_identity.json")
    if (output/"execution_complete.json").exists():
        raise FileExistsError("Study already terminal; do not restart")
    for slot in plan["slots"]:
        directory = output/"slots"/slot["slot_id"]
        if directory.exists():
            terminal = directory/"slot_result.json"
            if not terminal.exists():
                raise RuntimeError("Unfinished fixed slot needs inspection; never retry automatically: "+slot["slot_id"])
            retained = read(terminal)
            assert retained["frozen_slot"] == slot
            for path, digest in retained["evidence_bindings"].items():
                assert sha(path) == digest
            event("B31_RETAIN_TERMINAL_SLOT", slot_id=slot["slot_id"])
            continue
        directory.mkdir(parents=True, exist_ok=False)
        write(directory/"started.json", {"slot": slot, "source_identity_sha256": sha(output/"source_identity.json"),
            "argv": [sys.executable, *sys.argv]})
        if slot["old_source_slot"]:
            result, quality = _reused_slot(slot)
            bindings = {str(Path(slot["old_attempt_path"])/"attempt_result.json"): sha(Path(slot["old_attempt_path"])/"attempt_result.json"),
                slot["old_quality_path"]: sha(slot["old_quality_path"]), **quality["sources"]}
            new_cost = {"actual_physics_steps": 0, "private_preview_physics_steps": 0,
                "independent_saved_torque_replay_steps": 0, "native_geometry_query_calls": 0,
                "additional_route_quality_geometry_queries": 0, "old_evidence_reused": True}
        else:
            task = TaskSpec.from_dict(read(slot["task_path"]))
            residual = TaskAnchoredResidualPlan.from_dict(read(slot["plan_path"]))
            ledger = ExecutionCostLedger()
            attempt = output/"attempts"/slot["slot_id"]
            # Every rejection occupies its frozen slot; no candidate is replaced.
            if not slot["geometry_precheck_passed"]:
                from .task_anchored_residual import reject_slot
                result = reject_slot(attempt, task, residual.to_dict(),
                    "DECLARED_INITIAL_OR_ANCHOR_GEOMETRY_PRECHECK_REJECTED",
                    output/"source_identity.json", slot["slot_id"])
            else:
                with ledger.installed():
                    result = execute_residual_attempt(task, residual, attempt,
                        qp_config_path=output/"frozen_execution_config.json",
                        identity_path=output/"source_identity.json", slot_id=slot["slot_id"],
                        execution_diagnostics=True,
                        diagnostic_obstacle_name=task.scenario["workspace_obstacles"][1]["name"])
            costs = ledger.to_dict()
            write(directory/"execution_cost_ledger.json", costs)
            assert costs["actual_physics_steps"] == result["actual_steps"]
            quality, quality_path, quality_costs = _quality_with_retained_failure(
                output, directory, attempt, task, residual, slot, result)
            new_cost = {**costs, "additional_route_quality_geometry_queries": quality_costs["native_geometry_query_calls"],
                "route_quality_cost_ledger": quality_costs,
                "old_evidence_reused": False}
            bindings = {str(attempt/"attempt_result.json"): sha(attempt/"attempt_result.json"),
                str(quality_path): sha(quality_path),
                **quality["sources"]}
        record = {"schema": "v64_b31_fixed_slot_record_v1", "slot_id": slot["slot_id"],
            "frozen_slot": slot, "source_role": slot["source_role"], "source_result": result,
            "quality": quality, "new_cost": new_cost, "evidence_bindings": bindings,
            "new_diagnostic_vectors": "NOT_MEASURED_OLD_EVIDENCE" if slot["old_source_slot"] else "SAVED_IF_CONSUMED",
            "status": result["status"], "full_task_success": result["full_task_success"]}
        write(directory/"slot_result.json", record)
        event("B31_SLOT_TERMINAL", slot_id=slot["slot_id"], mode=slot["mode"],
            reused=bool(slot["old_source_slot"]), status=result["status"],
            full_task_success=result["full_task_success"],
            I_route=(quality.get("full_metrics") or {}).get("I_route_rad_s"))
    records = [read(output/"slots"/s["slot_id"]/"slot_result.json") for s in plan["slots"]]
    assert len(records) == 28
    source_guard(output/"source_identity.json")
    write(output/"execution_complete.json", {"schema": "v64_b31_execution_complete_v1", "terminal_slots": 28,
        "reused_slots": sum(bool(r["frozen_slot"]["old_source_slot"]) for r in records),
        "new_attempt_slots": sum(not bool(r["frozen_slot"]["old_source_slot"]) for r in records),
        "full_safe_successful_slots": sum(r["full_task_success"] for r in records),
        "training_runs": 0, "model_samples": 0, "additional_slots_authorized": False})
    event("B31_EXECUTION_COMPLETE", terminal_slots=28)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze-source", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.output)
    elif args.command == "freeze-source":
        if not args.checks: parser.error("freeze-source requires --checks")
        freeze_source(args.output, args.checks)
    else: run(args.output)


if __name__ == "__main__":
    main()
