"""Freeze and execute the finite B.3 pilot on the unmodified B.2 executor.

Later stages are authorized only by the stored pilot decision. This entry
point never silently trains, searches positions, retries, or replaces a slot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .residual_execution import sha, source_guard, write, execute_residual_attempt
from .task_anchored_residual import reject_slot
from .task_anchored_reference import TaskAnchoredResidualPlan
from .task_protocol import TaskSpec
from .conditional_execution import ExecutionCostLedger
from .route_quality_dataset import build_route_quality, pilot_discriminability

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("E:/v64b2work_20261007_01")
OLD_OUTPUT = OLD_ROOT / "v6_4/output/task_anchored_residual_20261007_01"
DEFAULT_OUTPUT = ROOT / "v6_4/output/conditional_route_value_20261007_01"
TARGET = Path("C:/Users/admin/.codex/attachments/34215c89-ed30-4cff-99d5-25a33d555c2d/pasted-text-1.txt")
PUBLISHED_B2 = "28ef7889be16b4a504d9449f52c50e3a99e82a31"
B2_ALGORITHM_PRODUCER = "7d0a3fd4bd17f41b99b388c30c5ac4897ac4d31d"
EXPECTED_REPORT_SHA = "2664eec9104eff627dfccce3b04201db64134634d5fbaed5f595a713111539f8"
NOISE_SEEDS = [2026100711, 2026100712, 2026100713, 2026100714]
MODES = {"z0": "z0", "z_plus": "z+", "z_minus": "z-", "z_perp": "z_perp"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def event(name, **values):
    print(json.dumps({"event": name, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def prepare(output):
    output = Path(output).resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("plan already frozen; no replacement")
    inputs = read(output / "route_pair_inputs.json")
    if sha(OLD_OUTPUT / "REPORT.md") != EXPECTED_REPORT_SHA:
        raise ValueError("B.2 report differs from the user-provided identity")
    for record in inputs["artifacts"]:
        if sha(output / record["path"]) != record["sha256"]:
            raise ValueError("fixed input changed: " + record["path"])
    source = output / "inputs/EXECUTION_TARGET.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        raise FileExistsError(source)
    shutil.copyfile(TARGET, source)
    config = ROOT / "v6_4/releases/task_anchored_residual_20261007_01/snapshot/frozen_execution_config.json"
    if sha(config) != "25a8f4aa060fcc3bc1f8aa6e5c791541d1f30a92b32565b8868f24e7da9f291e":
        raise ValueError("frozen QP configuration changed")
    shutil.copyfile(config, output / "frozen_execution_config.json")
    old_tests = sorted((t for t in read(OLD_OUTPUT / "tasks.json")["tasks"] if t["split"] == "test"), key=lambda t: t["task_id"])
    old_ids = [t["task_id"] for t in old_tests]
    if len(old_ids) != 4:
        raise ValueError("old diagnostic task count changed")
    pilot = [r for r in inputs["candidate_records"] if r["role"] == "pilot"]
    if len(pilot) != 6:
        raise ValueError("pilot must have exactly six predeclared candidates")
    slots = [{"slot_id": f"PILOT_{i:02d}", "task_id": r["task_id"], "candidate_name": MODES[r["mode"]],
              "plan_path": r["plan_path"], "T_route_s": r["T_route_s"]} for i, r in enumerate(pilot)]
    plan = {
        "schema": "v64_b3_conditional_route_value_plan_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name, "user_target_file_sha256": sha(source),
        "B2_algorithm_producer": B2_ALGORITHM_PRODUCER, "published_B2_base": PUBLISHED_B2,
        "development_base_commit": git("rev-parse", "HEAD"),
        "route_pair_inputs_sha256": sha(output / "route_pair_inputs.json"), "protocol": inputs["protocol"],
        "frozen_task_artifacts": inputs["artifacts"], "pilot_slots": slots,
        "old_checkpoint_probe": {
            "enabled": True, "checkpoint_sha256": "9ace53f482253216cea7baabed7479d84f37a9e80e71231219c7e2f9d3c94ca1",
            "selected_update": 250, "selected_training_exposures": 8000,
            "noise_pairs_per_task": 4, "maximum_ddim_calls": 32, "noise_seeds": NOISE_SEEDS,
            "task_pairs": [[old_ids[0], old_ids[1]], [old_ids[2], old_ids[3]]],
            "obstacle_swap": {"mode": "matching_obstacle_fields", "name_prefix": "continuum_side", "fields": ["center_w"]},
            "signed_probe": {"interval_selection": "first_active", "basis_axis": 0, "time_fraction": .5,
                             "world_direction_rule": "selected_task_world_transverse_basis_column_0"},
            "observed_response_threshold_m": 1e-10,
            "reference_time_grid_s": {"start": 0, "stop": 27, "period": .02},
            "old_public_TEST_role": "DIAGNOSTIC_ONLY_NOT_NEW_INDEPENDENT_TEST",
            "condition_change_alone_proves_correct_adaptation": False,
        },
        "route_quality": {
            "T_route": "the complete fixed key-slot-2 interval of each task, identically applied to every method",
            "sampling": "closed [start,end] 50Hz consumed planning ticks",
            "descriptive_input_route_window_string_correction": "route_pair_inputs.protocol says open_endpoints; this plan fixes CLOSED sampling before outcomes; numerical intervals and Task definitions unchanged",
            "primary": "sqrt(mean(task_avoidance_intervention**2))",
            "producer_scope": "norm of selected 17D QP command minus original Hessian-optimum nominal command after existing velocity-bound clipping; source vectors not logged",
            "raw_proposal_minus_selected_used": False,
            "local_clearance": "all continuum geoms vs the single moved sphere, fresh saved 2ms states; full task and route window reported separately",
            "full_task_and_all_independent_gates_required_for_quality_comparison": True,
            "failed_prefix_excluded_from_full_means": True,
            "pilot_direction_rule": "minimum I_route among full successful safe nonzero candidates; <=1e-12 absolute I tie is unidentified; preferred must satisfy A or B versus zero or opposite; pair preferred qualifying directions must differ",
            "pilot_A": "preferred nonzero full success and zero or opposite fails",
            "pilot_B": {"relative_reduction_min": .10, "absolute_reduction_min_rad_s": .001},
            "pilot_negative_stop": "ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION",
        },
        "teacher": {
            "formal_train_tasks": 6, "formal_val_tasks": 2, "candidates_per_task": ["z0", "z+", "z-", "z_perp"],
            "actual_slots": 32, "precheck_rejection_consumes_slot": True, "pilot_labels_allowed": False,
            "quality_filter": "I_route <= task_I_min + max(0.1*task_I_min,0.001)",
            "zero_labels_allowed": True, "different_modes_are_not_averaged": True,
            "path_length": "secondary description and same-I tie break only",
            "data_qualification": {"minimum_TRAIN_tasks_with_labels": 4, "required_VAL_tasks_with_labels": 2,
                                   "reason": "fixed validation samples must be evaluable against each of the two VAL task elite sets"},
        },
        "training": {
            "runs_max": 1, "seed": 64201, "draw_seed": 64202, "validation_seed": 64203,
            "optimizer_updates_max": 4000, "batch_size": 32, "validate_every": 250, "curve_every": 50,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01, "gradient_norm_clip": 1.},
            "network": {"condition_dim": 878, "latent_dim": 12, "hidden_layers": 2, "hidden_dim": 128, "parameters": 150028},
            "diffusion": {"steps": 100, "schedule": "cosine", "s": .008, "beta_cap": .999, "objective": "v_mse", "ddim_steps": 20, "eta": 0},
            "scalers": "original normalizer classes, refit exclusively on formal high-quality TRAIN labels/unique TRAIN tasks",
            "draw_distribution": "uniform eligible TRAIN task then uniform elite reference of that task",
            "checkpoint_selection": ["maximum legal VAL outputs out of 2 tasks x 4 fixed noises", "minimum task-mean decoded-route distance to closest verified elite VAL route", "minimum fixed VAL denoising loss", "earliest update"],
            "VAL_route_distance": "RMS world Cartesian residual difference on the common 2ms physical-time grid, minimum over that Task elite set without averaging modes",
            "VAL_noise_seeds": NOISE_SEEDS, "VAL_actual_rollouts": 0, "illegal_raw_repair": False,
            "selected_checkpoint_exposures_reported_separately": True, "TEST_used_for_selection": False,
        },
        "TEST": {
            "task_count": 4, "methods": ["Z0", "R0", "U0", "D_true", "D_swap"], "actual_slots": 20,
            "D_raw_candidates_per_task_method": 4, "D_actual_slot": 0, "K4_actual": "NOT_RUN",
            "noise_seeds": NOISE_SEEDS, "D_true_D_swap_random_streams_identical": True,
            "D_swap_fields": "only moved continuum sphere center from the paired task; all derived condition features re-encoded; true execution Task remains unchanged",
            "R0": {"distance": "Euclidean in same TRAIN-only standardized conditions", "compatible_fields": ["interval_mask", "frame", "representation_version", "coefficient_norm_bound_m"],
                   "tie_break": ["distance", "task_id", "candidate_name", "plan_sha256"]},
            "U0": {"seed": 64303, "distribution": "uniform elite TRAIN task then uniform elite label within task"},
            "failed_precheck_or_actual_keeps_denominator": True, "retries_fallback_or_candidate_replacements": False,
            "nonlearning_witness": "Z0 or R0 full success, otherwise TASK_FEASIBILITY_NOT_ESTABLISHED",
            "comparison": "same-task jointly complete-safe pairs only, while all failures remain in denominator",
        },
        "timing": {"research_wall_gate": False, "planning_period_s": .020, "physics_period_s": .002, "task_horizon_s": 27.,
                   "proposal_components": ["construct_or_load", "condition_encode", "sampling_or_retrieval", "decode", "reference_precheck", "end_to_end"],
                   "preparation_first_load_steady_separate": True, "nested_quantiles_added": False,
                   "cost_wrapper_overhead_included": True},
        "budget": inputs["protocol"]["budget"],
        "invariants": {"controller_provider_safety_backend_edited": False, "coefficient_bound_m": .020,
                       "robot_target_native_hz": 500, "whole_body_boundary_hz": 50, "configuration_subdivisions": 4,
                       "scene_hash_recorded_per_layout": True, "continuous_time_certified": False, "deployment": "NOT_MET"},
        "outputs": ["old_checkpoint_probe", "pilot", "teacher_if_pilot_passes", "dataset_if_qualified", "model_if_qualified", "test_if_trained",
                    "paired_metrics.json", "report.json", "REPORT.md", "manifest.json", "two_paired_route_figures", "method_table", "independent_review", "separate_GitHub_branch"],
        "publication": {"branch": "v6.4-b3-conditional-route-value", "push_only_after_finite_study_and_review": True,
                        "refresh_current_visualizations": True, "new_mass_video_generation": False},
    }
    write(output / "plan.json", plan)
    event("B3_PLAN_FROZEN", path=str(output / "plan.json"), sha256=sha(output / "plan.json"), pilot_slots=6)
    return plan


def freeze_source(output, checks_path):
    output, checks_path = Path(output).resolve(), Path(checks_path).resolve()
    plan = read(output / "plan.json")
    checks = read(checks_path)
    if checks.get("passed") is not True:
        raise ValueError("preflight checks must pass before source freeze")
    if git("status", "--porcelain"):
        raise ValueError("commit the reviewed B.3 implementation before source freeze")
    old_identity = read(OLD_OUTPUT / "source_identity.json")
    parity = []
    for relative, digest in old_identity["source_sha256"].items():
        old, current = OLD_ROOT / relative, ROOT / relative
        if sha(old) != digest:
            raise ValueError("old frozen source changed: " + relative)
        original, imported = old.read_bytes(), current.read_bytes()
        if original.replace(b"\r\n", b"\n") != imported.replace(b"\r\n", b"\n"):
            raise ValueError("B.2 executor source changed substantively: " + relative)
        parity.append({"path": relative, "old_sha256": digest, "current_sha256": sha(current),
                       "difference": "IDENTICAL" if original == imported else "CRLF_LF_ONLY"})
    names = set(old_identity["source_sha256"])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "v6_4").glob("*.py"))
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "v6_4/tests").glob("test_*conditional*.py"))
    names.update(("v6_4/tests/test_route_pair_protocol.py", "v6_4/tests/test_route_quality_dataset.py"))
    protected = {str((output / r["path"]).resolve()): r["sha256"] for r in plan["frozen_task_artifacts"]}
    for name in ("plan.json", "route_pair_inputs.json", "frozen_execution_config.json", "inputs/EXECUTION_TARGET.md"):
        protected[str((output / name).resolve())] = sha(output / name)
    protected[str(checks_path)] = sha(checks_path)
    for name in ("REPORT.md", "manifest.json", "verification.json", "source_identity.json", "training/model/selected.pt", "training/condition_normalizer.json", "training/residual_normalizer.json"):
        protected[str((OLD_OUTPUT / name).resolve())] = sha(OLD_OUTPUT / name)
    identity = {"schema": "v64_b3_frozen_source_identity_v1", "git_head": git("rev-parse", "HEAD"),
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "published_B2_base": PUBLISHED_B2,
        "B2_original_algorithm_producer": B2_ALGORITHM_PRODUCER,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "source_sha256": {p: sha(ROOT / p) for p in sorted(names)}, "protected_artifacts": protected,
        "B2_executor_semantic_byte_parity": parity, "B2_executor_only_CRLF_LF_differences": True,
        "controller_or_safety_changes": False, "wall_20ms_is_research_gate": False, "deployment": "NOT_MET"}
    write(output / "source_identity.json", identity)
    source_guard(output / "source_identity.json")
    event("B3_SOURCE_FROZEN", producer=identity["git_head"], source_files=len(names))
    return identity


def run_pilot(output):
    output = Path(output).resolve()
    plan = read(output / "plan.json")
    source_guard(output / "source_identity.json")
    probe = read(output / "old_checkpoint_probe/report.json")
    if probe["status"] != "COMPLETED" or probe["ddim_calls"] != 32 or probe["actual_attempts"] != 0:
        raise ValueError("the bounded old-checkpoint diagnostic must finish first")
    records = []
    for declared in plan["pilot_slots"]:
        slot, task_id = declared["slot_id"], declared["task_id"]
        task = TaskSpec.from_dict(read(output / f"tasks/{task_id}/task.json"))
        raw = read(output / declared["plan_path"])
        geometry = read(output / f"tasks/{task_id}/geometry_precheck.json")
        directory = output / "pilot/attempts" / slot
        if directory.exists():
            if not (directory / "attempt_result.json").exists() or not (directory / "cost_ledger.json").exists():
                raise ValueError("unfinished pilot slot cannot be retried: " + slot)
            if geometry["passed"]:
                result = execute_residual_attempt(task, TaskAnchoredResidualPlan.from_dict(raw), directory,
                    qp_config_path=output / "frozen_execution_config.json", identity_path=output / "source_identity.json",
                    slot_id=slot, reuse_completed=True)
            else:
                result = reject_slot(directory, task, raw, "FROZEN_INITIAL_OR_ANCHOR_GEOMETRY_PRECHECK_REJECTED",
                    output / "source_identity.json", slot, reuse=True)
            cost = read(directory / "cost_ledger.json")
            if (cost["slot_id"] != slot or cost["actual_physics_steps"] != result["actual_steps"]
                    or cost["saved_actual_trace_steps"] != result["actual_steps"]):
                raise ValueError("retained cost ledger differs from actual evidence")
        else:
            ledger = ExecutionCostLedger()
            with ledger.installed():
                if not geometry["passed"]:
                    result = reject_slot(directory, task, raw, "FROZEN_INITIAL_OR_ANCHOR_GEOMETRY_PRECHECK_REJECTED", output / "source_identity.json", slot)
                else:
                    residual = TaskAnchoredResidualPlan.from_dict(raw)
                    result = execute_residual_attempt(task, residual, directory,
                        qp_config_path=output / "frozen_execution_config.json", identity_path=output / "source_identity.json", slot_id=slot)
            cost = ledger.to_dict()
            cost["slot_id"] = slot
            cost["saved_actual_trace_steps"] = result["actual_steps"]
            cost["actual_step_count_matches_saved_trace"] = cost["actual_physics_steps"] == result["actual_steps"]
            write(directory / "cost_ledger.json", cost)
            if cost["actual_physics_steps"] != result["actual_steps"]:
                raise ValueError("native integration counter differs from saved actual trace; retain slot without retry")
        quality_dir = output / "pilot/quality" / slot
        if (quality_dir / "route_quality.json").exists():
            for name, digest in read(quality_dir / "manifest.json").items():
                if sha(quality_dir / name) != digest:
                    raise ValueError("retained quality artifact differs: " + name)
            quality = read(quality_dir / "route_quality.json")
            if (quality["slot_id"], quality["task_sha256"], quality["candidate_name"], quality["route_intervals_s"]) != (
                    slot, task.sha256(), declared["candidate_name"], [declared["T_route_s"]]):
                raise ValueError("retained quality input identity differs")
            for path, digest in quality["sources"].items():
                if sha(path) != digest:
                    raise ValueError("retained quality source differs: " + path)
        else:
            obstacle = task.scenario["workspace_obstacles"][1]["name"]
            quality = build_route_quality(directory, [declared["T_route_s"]], obstacle, quality_dir,
                                          candidate_name=declared["candidate_name"])
        records.append(quality)
        event("B3_PILOT_SLOT_RECORDED", slot=slot, status=result["status"], actual_steps=result["actual_steps"],
              full_task_success=result["full_task_success"], I_route=(quality.get("full_metrics") or {}).get("I_route_rad_s"))
    decision = pilot_discriminability(records, read(output / "route_pair_inputs.json")["pilot_task_ids"])
    write(output / "pilot/decision.json", decision)
    write(output / "pilot/quality_records.json", records)
    event("B3_PILOT_TERMINAL", **decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze-source", "pilot"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    elif args.action == "freeze-source":
        if args.checks is None:
            parser.error("--checks is required")
        freeze_source(args.output, args.checks)
    else:
        run_pilot(args.output)


if __name__ == "__main__":
    main()
