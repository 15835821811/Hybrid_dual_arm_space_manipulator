"""Final held-out C.2 teacher update; evidence export only, no new evaluation.

Prediction repetitions across methods remain distinct cost/lineage receipts,
while identical physical reference identities share one candidate fact. These
TEST records are explicitly forbidden as inputs to the current C.2 model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from v6_4.route_optimizer_protocol import read, sha, write, digest, VERSIONS
from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import TaskAnchoredResidualPlan
from v6_4.preference_warmstart_protocol import search_interval_mask
from v6_4.preference_teacher_dataset import physical_candidate_key, classify_evidence

SCHEMA = "v64_c2_final_heldout_teacher_update_v1"
METHODS = ("R", "N", "D")
ENDPOINTS = ("R8", "R12", "N8", "D8")
METRICS = ("I_support", "I_support_10D", "I_support_7D", "I_full", "L_full", "L_support",
    "d_support", "clearance_status", "clearance_is_censored_lower_bound",
    "base_translation_peak_m", "base_rotation_peak_rad", "saved_horizon_s", "native_state_count")
TIERS = ("VALIDATED_EXECUTION", "PREDICTED_COMPLETE", "FAILED_OR_INCOMPLETE", "MISSING_OR_UNBOUND")


def _finite(value):
    return type(value) in (float, int) and np.isfinite(value)


class Inventory:
    """Run-relative paths and SHA bind every exported fact to original bytes."""
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.files = {}

    def add(self, path):
        path = Path(path).resolve()
        if self.run not in path.parents:
            raise ValueError("teacher evidence escapes its immutable run")
        relative = path.relative_to(self.run).as_posix()
        value = {"path": relative, "sha256": sha(path), "size_bytes": path.stat().st_size}
        previous = self.files.get(relative)
        if previous is not None and previous != value:
            raise ValueError("evidence changed during export: " + relative)
        self.files[relative] = value
        return value

    def read(self, path):
        self.add(path)
        return read(path)

    def sealed_read(self, directory, relative):
        directory = Path(directory)
        manifest = self.read(directory / "manifest.json")
        path = directory / relative
        record = self.add(path)
        if manifest.get(Path(relative).as_posix()) != record["sha256"]:
            raise ValueError("actual independent evidence seal differs: " + str(path))
        return read(path)


def _gate(run, inventory):
    from v6_4.evaluate_preference_warmstart import verify_selections
    verify_selections(run)
    sealed = inventory.read(run / "sealed_selections/all_selections.json")
    complete = inventory.read(run / "actual_complete.json")
    validation = inventory.sealed_read(run / "validation", "validation.json")
    if (validation.get("all_terminal") is not True or validation.get("tool_error_count") != 0
            or validation.get("logical_actual_slots") != 32 or complete.get("logical_slots") != 32):
        raise ValueError("final teacher update requires the complete terminal 32-slot protocol")
    if sealed.get("logical_actual_slots") != 32 or sealed.get("selection_reads_actual") is not False:
        raise ValueError("selected endpoints were not sealed independently of actual")
    freeze = inventory.add(run / "model_freeze.json")
    if freeze["sha256"] != sealed["model_freeze_sha256"]:
        raise ValueError("model freeze changed after TEST selections")
    plan = inventory.read(run / "plan.json")
    tasks = [r for r in plan["tasks"] if r["split"] == "test"]
    if len(tasks) != 4 or len({r["mother_source_sha256"] for r in tasks}) != 2:
        raise ValueError("teacher update requires the four held-out tasks from two frozen mothers")
    return tasks, complete, freeze["sha256"]


def _actual_evidence(run, complete, tasks, inventory):
    slots = {}
    expected = {(t["task_id"], e + "_" + p) for t in tasks for e in ENDPOINTS for p in ("A", "B")}
    for relative, expected_sha in complete["slot_hashes"].items():
        path = run / relative
        bound = inventory.add(path)
        if bound["sha256"] != expected_sha:
            raise ValueError("final actual completion receipt differs")
        slot = read(path)
        key = (slot["task_id"], slot["method"])
        if key in slots or key not in expected:
            raise ValueError("unexpected or repeated final logical slot")
        selection = run / "sealed_selections" / key[0] / (slot["endpoint"] + ".json")
        if inventory.add(selection)["sha256"] != slot["selection_sha256"]:
            raise ValueError("actual slot does not bind its sealed selection")
        slots[key] = slot
    if set(slots) != expected:
        raise ValueError("not all 32 final logical methods have evidence or NO_PLAN")

    evidence = {}
    for key, slot in slots.items():
        if slot.get("plan_sha256") is None:
            continue
        origin = slot
        seen = set()
        while origin.get("alias_of_method"):
            method = origin["alias_of_method"]
            if method in seen:
                raise ValueError("cyclic actual alias")
            seen.add(method)
            source = slots.get((key[0], method))
            if source is None or source["alias_identity"] != slot["alias_identity"]:
                raise ValueError("actual alias lacks complete matching execution identity")
            origin = source
        identity = digest({"task_sha256": slot["task_sha256"], "plan_sha256": slot["plan_sha256"],
            "source_identity_sha256": sha(run / "source_identity.json"),
            "config_sha256": sha(run / "frozen_execution_config.json"),
            "run_config_sha256": sha(run / "frozen_run_config.json")})
        if slot["alias_identity"] != identity:
            raise ValueError("actual alias execution configuration differs")
        source_key = (key[0], origin["method"])
        if source_key not in evidence:
            directory = run / "actual" / key[0] / origin["method"]
            report_path = directory / "attempt/actual/evaluation/report.json"
            if not report_path.exists():
                evidence[source_key] = {"report": None, "attempt": None, "plan": None}
            else:
                evidence[source_key] = {
                    "report": inventory.sealed_read(directory, "attempt/actual/evaluation/report.json"),
                    "attempt": inventory.sealed_read(directory, "attempt/attempt_result.json"),
                    "plan": inventory.sealed_read(directory, "selected_plan.json"),
                    "source_actual_method": origin["method"]}
        slot["_source_evidence"] = evidence[source_key]
    return slots


def bind_actual(slot, row, task, plan):
    """Qualification is per prediction run, including aliases of actual evidence."""
    source = slot.get("_source_evidence") or {}
    report, attempt = source.get("report") or {}, source.get("attempt") or {}
    quality = slot.get("quality") or {}
    predicted = row.get("prediction_metrics") or {}
    reference = report.get("reference_binding") or {}
    consumed = reference.get("consumed_reference_identity") or {}
    gates = {k: (report.get(k) or {}).get("passed") is True for k in
        ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")}
    source_plan = TaskAnchoredResidualPlan.from_dict(source["plan"]) if source.get("plan") else None
    checks = {"task_sha256": slot.get("task_sha256") == task.sha256() == report.get("task_sha256"),
        "plan_sha256": slot.get("plan_sha256") == plan.sha256() == reference.get("plan_sha256"),
        "executed_source_plan": source_plan is not None and source_plan.sha256() == plan.sha256(),
        "definition": reference.get("definition_sha256") == plan.definition["definition_sha256"],
        "consumed_reference": consumed.get("available") is True and consumed.get("passed") is True
            and consumed.get("plan_sha256") == plan.sha256() and np.array_equal(np.asarray(consumed.get("z_m")), plan.z_m),
        "full13500": slot.get("actual_steps") == 13500 == report.get("actual_physics_steps"),
        "full27s": _finite(report.get("actual_saved_horizon_s")) and abs(report["actual_saved_horizon_s"] - 27.) < 1e-8,
        "independent_replay": report.get("replayed_physics_steps") == 13500 and bool(report.get("fresh_replay_sha256")),
        "original_five_gates": all(gates.values()) and slot.get("original_independent_gates_passed") is True,
        "full_task": slot.get("full_task_success") is True and report.get("evidence_valid") is True
            and report.get("full_task_success") is True and attempt.get("full_task_success") is True,
        "attempt_binding": attempt.get("plan_content_sha256") == plan.sha256()
            and attempt.get("task_sha256") == task.sha256(),
        "attempt_full27_and_steps": attempt.get("full_27s_success") is True and attempt.get("actual_steps") == 13500,
        "attempt_execution_config": attempt.get("qp_config_sha256") == row["execution_identity"]["config_sha256"]
            and attempt.get("source_identity_sha256") == row["execution_identity"]["source_identity_sha256"],
        "slot_own_prediction_consistency": slot.get("prediction_actual_consistent") is True,
        "this_prediction_numeric_consistency": all(_finite(quality.get(k)) and _finite(predicted.get(k))
            and abs(quality[k] - predicted[k]) <= 1e-9 for k in ("I_support", "L_full", "d_support"))}
    return {"logical_method": slot["method"], "endpoint": slot["endpoint"], "preference": slot["preference"],
        "source_actual_method": source.get("source_actual_method"), "alias_of_method": slot.get("alias_of_method"),
        "independently_bound_and_validated": all(checks.values()), "checks": checks,
        "actual_quality": {k: quality.get(k) for k in METRICS},
        "B_clearance_30mm": bool(_finite(quality.get("d_support")) and quality["d_support"] >= .030)}


def compact_costs(costs):
    costs = costs or {}
    return {k: v for k, v in costs.items() if k != "preview_records"}


def merge_prediction(facts, row, task, frozen, method, actual_bindings, registry_source):
    plan = TaskAnchoredResidualPlan.from_dict(row["plan"])
    family = next((f for f, version in VERSIONS.items() if version == plan.representation_version), None)
    if family is None:
        raise ValueError("unknown TEST reference family")
    key = physical_candidate_key(task.sha256(), row["execution_identity"], plan.representation_version, plan.sha256())
    fact = facts.setdefault(key, {"physical_candidate_id": key, "task_id": task.task_id,
        "task_sha256": task.sha256(), "mother_id": frozen["mother_id"],
        "mother_source_sha256": frozen["mother_source_sha256"], "split": "test",
        "learning_role": "heldout_for_C2", "forbidden_current_C2_training": True,
        "reference_family": family, "reference_version": plan.representation_version,
        "plan_sha256": plan.sha256(), "execution_identity": row["execution_identity"],
        "z_m": plan.z_m.tolist(), "reference_interval_mask": plan.interval_mask.tolist(),
        "search_interval_mask": search_interval_mask(plan.definition).tolist(),
        "prediction_runs": [], "actual_bindings": [], "evidence_tier": "MISSING_OR_UNBOUND"})
    if fact["z_m"] != plan.z_m.tolist() or fact["reference_family"] != family:
        raise ValueError("same physical identity has inconsistent reference")
    run_id = f"{task.task_id}/{method}/{row['candidate_id']}"
    if any(r["prediction_run_id"] == run_id for r in fact["prediction_runs"]):
        raise ValueError("prediction run exported twice")
    tier = classify_evidence(row, task, plan, actual_bindings)
    fact["prediction_runs"].append({"prediction_run_id": run_id, "method": method,
        "candidate_id": row["candidate_id"], "source": row.get("source"),
        "origin_source": row.get("origin_source"), "parent_candidate_id": row.get("parent_candidate_id"),
        "proposal_lineage": row.get("proposal_lineage"), "proposal_index": row.get("proposal_index"),
        "status": row.get("status"), "evidence_tier": tier,
        "prediction_steps": row.get("prediction_steps", 0),
        "prediction_rollout_started": row.get("prediction_rollout_started") is True,
        "prediction_metrics": {k: (row.get("prediction_metrics") or {}).get(k) for k in METRICS},
        "elapsed_wall_s": row.get("elapsed_wall_s"), "costs": compact_costs(row.get("costs")),
        "full_costs_sha256": digest(row.get("costs")), "registry_source": registry_source,
        "actual_binding_methods_validated": [b["logical_method"] for b in actual_bindings if b["independently_bound_and_validated"]]})
    for binding in actual_bindings:
        fact["actual_bindings"].append({**binding, "prediction_run_id": run_id})
    order = {name: i for i, name in enumerate(TIERS)}
    fact["evidence_tier"] = min([r["evidence_tier"] for r in fact["prediction_runs"]], key=order.get)
    return key


def preference_views(facts, model_freeze_sha256):
    """Two evidence views per physical candidate; neither is a training label."""
    rows = []
    for fact in facts:
        complete = [r for r in fact["prediction_runs"] if r["evidence_tier"] in TIERS[:2]]
        primary = sorted(complete or fact["prediction_runs"], key=lambda r: r["prediction_run_id"])[0]
        metrics = primary["prediction_metrics"]
        for preference in ("A", "B"):
            B = bool(_finite(metrics.get("d_support")) and metrics["d_support"] >= .030
                and metrics.get("clearance_status") == "MEASURED")
            rows.append({"physical_candidate_id": fact["physical_candidate_id"],
                "task_id": fact["task_id"], "task_sha256": fact["task_sha256"], "mother_id": fact["mother_id"],
                "split": "test", "learning_role": "heldout_for_C2", "forbidden_current_C2_training": True,
                "preference": preference, "family": fact["reference_family"], "reference_version": fact["reference_version"],
                "plan_sha256": fact["plan_sha256"], "z_m": fact["z_m"],
                "evidence_tier": fact["evidence_tier"], "prediction_eligible": bool(complete),
                "validated_execution_example": fact["evidence_tier"] == "VALIDATED_EXECUTION",
                "preference_qualified": bool(complete and (preference == "A" and _finite(metrics.get("I_support")) or preference == "B" and B)),
                "B_clearance_30mm_independent_of_actual_validation": B,
                "prediction_metrics": metrics, "quality_source_prediction_run_id": primary["prediction_run_id"],
                "prediction_run_ids": [r["prediction_run_id"] for r in fact["prediction_runs"]],
                "actual_logical_methods": sorted({b["logical_method"] for b in fact["actual_bindings"]}),
                "model_freeze_sha256": model_freeze_sha256,
                "label_scope": "final_TEST_evidence_view_only_not_family_near_optimal_supervision"})
    return rows


def _reuse(run, output):
    manifest = read(output / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown final teacher update schema")
    for relative, expected in manifest["output_sha256"].items():
        if sha(run / relative) != expected:
            raise ValueError("final teacher output changed: " + relative)
    inventory = read(output / "source_inventory.json")
    for source in inventory["files"]:
        if sha(run / source["path"]) != source["sha256"]:
            raise ValueError("teacher source changed after immutable export: " + source["path"])
    if sha(run / "model_freeze.json") != manifest["model_freeze_sha256"]:
        raise ValueError("current C.2 model freeze changed")
    return manifest


def export_teacher_update(run):
    run = Path(run).resolve()
    output = run / "teacher_update"
    if (output / "manifest.json").exists():
        return _reuse(run, output)
    if output.exists() or (run / "teacher_records.jsonl").exists():
        raise FileExistsError("unfinished final teacher export retained; no overwrite")
    inventory = Inventory(run)
    frozen_tasks, complete, frozen_model_sha = _gate(run, inventory)
    for name in ("source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
        inventory.add(run / name)
    slots = _actual_evidence(run, complete, frozen_tasks, inventory)
    facts, rejected, streams = {}, [], []
    for frozen in frozen_tasks:
        tid = frozen["task_id"]
        task = TaskSpec.from_dict(inventory.read(run / "frozen_tasks" / tid / "task.json"))
        if task.sha256() != frozen["task_sha256"]:
            raise ValueError("held-out TEST Task SHA changed")
        for method in METHODS:
            planning = run / "benchmark_search" / tid / method / "planning" / tid
            registry_source = inventory.add(planning / "candidate_registry.json")
            rows = read(planning / "candidate_registry.json")
            proposals = inventory.read(planning / "proposals.json")
            inventory.add(planning / "selection.json")
            inventory.add(planning.parents[1] / "planning_cost.json")
            started, steps = 0, 0
            for row in rows:
                started += int(row.get("prediction_rollout_started") is True)
                steps += row.get("prediction_steps", 0)
                if row.get("plan") is None:
                    if row.get("prediction_rollout_started") or row.get("prediction_steps", 0) != 0:
                        raise ValueError("planless rejection unexpectedly has physical work")
                    rejected.append({"task_id": tid, "task_sha256": task.sha256(), "mother_id": frozen["mother_id"],
                        "method": method, "candidate_id": row["candidate_id"], "proposal_index": row.get("proposal_index"),
                        "source": row.get("source"), "status": row.get("status"), "raw_z_m": row.get("raw_z_m"),
                        "raw_seed_diagnostics": row.get("raw_seed_diagnostics"),
                        "initializer_rejection": row.get("initializer_rejection"), "plan": None, "plan_sha256": None,
                        "evaluation_slot_consumed": True, "physics_steps": 0, "registry_source": registry_source,
                        "learning_role": "heldout_for_C2", "forbidden_current_C2_training": True})
                    continue
                plan = TaskAnchoredResidualPlan.from_dict(row["plan"])
                actual = [bind_actual(s, row, task, plan) for s in slots.values()
                    if s["task_id"] == tid and s.get("plan_sha256") == plan.sha256()]
                merge_prediction(facts, row, task, frozen, method, actual, registry_source)
            streams.append({"task_id": tid, "method": method, "evaluation_slots": len(rows),
                "parameter_proposals": len(proposals), "exact_cache_hits": sum("cache_hit_candidate_id" in p for p in proposals),
                "prediction_rollouts_started": started, "prediction_main_steps": steps,
                "rejected_planless_slots": sum(r.get("plan") is None for r in rows)})
    if sum(s["evaluation_slots"] for s in streams) > 112 or len(streams) != 12:
        raise ValueError("final TEST prediction budget/stream inventory differs")
    unique = sorted(facts.values(), key=lambda f: (f["task_sha256"], f["plan_sha256"], f["physical_candidate_id"]))
    views = preference_views(unique, frozen_model_sha)
    if sha(run / "model_freeze.json") != frozen_model_sha:
        raise ValueError("model freeze changed during final evidence export")
    output.mkdir()
    write(output / "candidate_facts.json", unique)
    write(output / "rejected_proposals.json", rejected)
    write(output / "source_inventory.json", {"schema": SCHEMA, "path_base": "run_root",
        "files": sorted(inventory.files.values(), key=lambda r: r["path"]),
        "model_freeze_sha256": frozen_model_sha})
    with (run / "teacher_records.jsonl").open("x", encoding="utf8") as stream:
        for row in views:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "learning_role": "heldout_for_C2", "forbidden_current_C2_training": True,
        "current_C2_model_retrained": False, "model_freeze_sha256": frozen_model_sha,
        "model_freeze_unchanged": True, "physics_steps_added": 0, "inference_samples_added": 0,
        "task_denominator": 4, "mother_denominator": 2, "logical_actual_slots": 32,
        "unique_actual_runs": complete["unique_actual_runs"], "streams": streams,
        "evaluation_slots": sum(s["evaluation_slots"] for s in streams),
        "physical_candidates_unique": len(unique), "preference_view_rows": len(views),
        "distinct_prediction_runs": sum(len(f["prediction_runs"]) for f in unique),
        "rejected_planless_slots": len(rejected),
        "evidence_tiers_unique_candidates": dict(Counter(f["evidence_tier"] for f in unique)),
        "row_count_is_not_independent_trajectory_count": True,
        "output_sha256": {p.relative_to(run).as_posix(): sha(p) for p in
            (output / "candidate_facts.json", output / "rejected_proposals.json", output / "source_inventory.json", run / "teacher_records.jsonl")}}
    write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(export_teacher_update(args.run), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
