"""C.1 frozen search orchestration, final independent execution and reporting."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from .route_optimizer_protocol import (ROOT, HISTORY, METHODS, PreferenceSpec, TaskSpec,
    initial_candidates, parameter_plan, read, write, sha, digest, verify_frozen)
from .route_candidate_evaluator import NominalCandidateEvaluator, fresh_quality, seal, verify_seal
from .continuous_route_optimizer import optimize, rank_candidates, admissible


def analyze_history(run):
    """Only the 28 published quality rows; zero new physics or old replay."""
    output = run / "history_frontier"
    if output.exists():
        verify_seal(output); return
    path = ROOT / "v6_4/visualization/execution_aware_route_teacher_20261007_01/all_candidates.csv"
    with path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 28:
        raise ValueError("published history is not the frozen 28-row matrix")
    output.mkdir()
    groups = {}
    for row in rows:
        groups.setdefault(row["task_id"], []).append(row)
    result = []
    for tid, records in groups.items():
        def vector(r):
            return np.array([float(r["I_route_rad_s"]), float(r["continuum_path_length_m"]), -float(r["route_clearance_m"])])
        frontier = [r["slot_id"] for r in records if not any(np.all(vector(o) <= vector(r)) and np.any(vector(o) < vector(r)) for o in records)]
        result.append({"task_id": tid, "finite_non_dominated_ids": frontier, "rows": records})
    write(output / "history_frontier.json", {"source_sha256": sha(path), "tasks": result,
        "scope": "published finite development quality only; no missing raw assumed or replay claimed",
        "used_to_query_new_task_optimum": False, "physics_steps": 0, "geometry_queries": 0})
    seal(output)


def optimize_tasks(run):
    protocol = verify_frozen(run)
    if (run / "all_selections.json").exists():
        verify_selections(run); return
    selections = {}
    for frozen in protocol["tasks"]:
        task = TaskSpec.from_dict(read(run / "frozen_tasks" / frozen["task_id"] / "task.json"))
        prefs = [PreferenceSpec(**p) for p in read(run / "frozen_tasks" / task.task_id / "preferences.json")]
        evaluator = NominalCandidateEvaluator(run, task, frozen)
        selection = optimize(task, prefs, evaluator.execution_identity, evaluator, run / "planning" / task.task_id)
        if selection["budget"]["stop_reason"] == "TOOL_ERROR":
            raise RuntimeError("tool error retained in consumed candidate; protocol incomplete, no actual phase: " + task.task_id)
        selections[task.task_id] = {"selection_sha256": sha(run / "planning" / task.task_id / "selection.json"),
            "registry_sha256": sha(run / "planning" / task.task_id / "candidate_registry.json"),
            "proposals_sha256": sha(run / "planning" / task.task_id / "proposals.json")}
    write(run / "all_selections.json", {"sealed_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": sha(run / "source_identity.json"), "selections": selections,
        "final_actual_started": False, "selection_may_read_final_actual": False})
    print(json.dumps({"event": "C1_ALL_SELECTIONS_SEALED", "tasks": len(selections)}), flush=True)


def verify_selections(run):
    protocol = verify_frozen(run)
    seal_record = read(run / "all_selections.json")
    if seal_record["source_identity_sha256"] != sha(run / "source_identity.json") or set(seal_record["selections"]) != {t["task_id"] for t in protocol["tasks"]}:
        raise ValueError("all-selection seal identity mismatch")
    for tid, row in seal_record["selections"].items():
        for name, key in (("selection.json", "selection_sha256"), ("candidate_registry.json", "registry_sha256"), ("proposals.json", "proposals_sha256")):
            if sha(run / "planning" / tid / name) != row[key]:
                raise ValueError("selection changed after freeze: " + tid)
    return protocol


def selected_methods(task, selection):
    from .task_anchored_reference import TaskAnchoredResidualPlan, build_reference_definition
    seeds = initial_candidates(task)
    if seeds:
        z0 = parameter_plan(task, seeds[0]["family"], seeds[0]["x_m"])
        g0 = parameter_plan(task, seeds[2]["family"], seeds[2]["x_m"])
    else:
        d = build_reference_definition(task)
        z0 = TaskAnchoredResidualPlan.from_definition(d, np.zeros((6, 2))) if d["applicable"] else None
        g0 = None
    return {"Z0": z0, "G0": g0,
        **{method: TaskAnchoredResidualPlan.from_dict(selection["preferences"][name]["selected_plan"])
           if selection["preferences"][name]["selected_plan"] else None for method, name in (("OI", "A"), ("OC", "B"))}}


def execute_selected(run):
    from dataclasses import asdict
    from v6_lite.run_v6_lite import V6LiteRunConfig
    from .conditional_execution import ExecutionCostLedger
    from .residual_execution import execute_residual_attempt
    protocol = verify_selections(run)
    inherited_config = asdict(V6LiteRunConfig(pcc_mode="bounded_interval_pcc", dispatch_clock_policy="research_simulation"))
    if inherited_config != read(run / "frozen_run_config.json"):
        raise ValueError("unchanged final executor run configuration differs from frozen actual configuration")
    if (run / "actual_complete.json").exists():
        for directory in (run / "actual").glob("*/*"):
            verify_seal(directory)
        return
    slots = []; unique = 0
    for frozen in protocol["tasks"]:
        tid = frozen["task_id"]
        task = TaskSpec.from_dict(read(run / "frozen_tasks" / tid / "task.json"))
        selection = read(run / "planning" / tid / "selection.json")
        chosen = selected_methods(task, selection); aliases = {}
        # First actual is the first task Z0. All four selections already sealed.
        for method in METHODS:
            directory = run / "actual" / tid / method
            if directory.exists():
                if not (directory / "slot.json").exists():
                    raise RuntimeError("unfinished consumed final method slot retained; never retry: " + str(directory))
                verify_seal(directory)
                slot = read(directory / "slot.json")
                if slot["selection_sha256"] != sha(run / "planning" / tid / "selection.json"):
                    raise ValueError("retained final slot selection differs")
                if slot.get("unique_run"):
                    aliases[slot["plan_sha256"]] = slot
                    unique += 1
                slots.append(slot); continue
            directory.mkdir(parents=True)
            plan = chosen[method]
            slot = {"task_id": tid, "method": method, "logical_slot": len(slots), "unique_run": False,
                "selection_sha256": sha(run / "planning" / tid / "selection.json"),
                "task_sha256": task.sha256(), "status": "NO_PLAN", "entered_actual": False,
                "full_task_success": False, "original_independent_gates_passed": False,
                "clearance_30mm_met": None, "quality": None, "actual_steps": 0,
                "plan_sha256": plan.sha256() if plan else None,
                "source_candidate_id": selection["preferences"]["A" if method == "OI" else "B"]["source_candidate_id"] if method in ("OI", "OC") else ("C00" if method == "Z0" else "C02"),
                "prediction_actual_consistent": None, "costs": None, "deployment": "NOT_MET"}
            if plan is not None:
                write(directory / "selected_plan.json", plan.to_dict())
                if not frozen["geometry_precheck_passed"]:
                    slot["status"] = "TASK_INITIAL_OR_ANCHOR_PRECHECK_REJECTED"
                elif plan.sha256() in aliases:
                    source = aliases[plan.sha256()]
                    for key in ("status", "entered_actual", "full_task_success", "original_independent_gates_passed", "clearance_30mm_met", "quality", "actual_steps", "evidence_method", "prediction_actual_consistent", "prediction_actual_difference"):
                        slot[key] = source.get(key)
                    slot["alias_of_method"] = source["method"]
                    slot["costs"] = {"alias_zero_new_work": True}
                else:
                    ledger = ExecutionCostLedger()
                    with ledger.installed():
                        result = execute_residual_attempt(task, plan, directory / "attempt",
                            qp_config_path=run / "frozen_execution_config.json", identity_path=run / "source_identity.json",
                            slot_id=tid + "_" + method, execution_diagnostics=True,
                            diagnostic_obstacle_name=frozen["obstacle_name"])
                        slot.update(unique_run=bool(result.get("actual_runner_started")), entered_actual=result["entered_actual"], status=result["status"],
                            full_task_success=result["full_task_success"], actual_steps=result["actual_steps"], evidence_method=method)
                        evaluation = result.get("evaluation") or {}
                        slot["original_independent_gates_passed"] = bool(evaluation.get("evidence_valid") and all(
                            evaluation.get(k, {}).get("passed") is True for k in
                            ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")))
                        replay = directory / "attempt/actual/evaluation/fresh_replay.npz"
                        if result.get("trace_path") and result["actual_steps"] and replay.exists():
                            with ledger.scope("actual_quality"):
                                quality, requirements = fresh_quality(task, plan, result["trace_path"], frozen,
                                    directory / "quality", evidence_root=directory / "attempt/actual", replay_path=replay)
                            slot["quality"] = quality
                            slot["clearance_30mm_met"] = quality["d_support"] >= .030
                        slot["costs"] = ledger.to_dict()
                    if result.get("pipeline_failure"):
                        slot["tool_error"] = result["pipeline_failure"]
                    # Compare against the exact selected plan prediction, including baseline cache seeds.
                    rows = read(run / "planning" / tid / "candidate_registry.json")
                    predicted = next((r for r in rows if r["plan_sha256"] == plan.sha256()), None)
                    if predicted and slot["quality"] and predicted.get("prediction_metrics"):
                        differences = {k: slot["quality"][k] - predicted["prediction_metrics"][k]
                                       for k in ("I_support", "I_route_key_legacy", "I_full", "L_full", "d_support")}
                        slot["prediction_actual_difference"] = differences
                        # Numeric evidence comparison, never a relaxed safety or preference threshold.
                        slot["prediction_actual_consistent"] = bool(
                            slot["full_task_success"] == predicted["prediction_admissible"] and
                            all(abs(v) <= 1e-9 for v in differences.values()))
                        slot["consistency_metric_threshold"] = 1e-9
                    elif predicted:
                        slot["prediction_actual_consistent"] = bool(not predicted["prediction_admissible"] and not slot["full_task_success"])
                    unique += int(slot["unique_run"])
                    aliases[plan.sha256()] = slot
            write(directory / "slot.json", slot); seal(directory)
            slots.append(slot)
            print(json.dumps({"event": "C1_FINAL_SLOT", "task": tid, "method": method,
                "status": slot["status"], "unique_run": slot["unique_run"], "steps": slot["actual_steps"],
                "original_gates": slot["original_independent_gates_passed"], "clearance_30mm": slot["clearance_30mm_met"]}), flush=True)
            if slot.get("tool_error"):
                raise RuntimeError("final tool error retained; no silent retry")
    verify_selections(run)
    write(run / "actual_complete.json", {"logical_method_slots": len(slots), "unique_actual_runs": unique,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "slot_hashes": {
            f'{s["task_id"]}/{s["method"]}': sha(run / "actual" / s["task_id"] / s["method"] / "slot.json") for s in slots}})


def validate(run):
    protocol = verify_selections(run)
    output = run / "validation"
    if output.exists():
        verify_seal(output); return read(output / "validation.json")
    complete = read(run / "actual_complete.json")
    rows = []
    for frozen in protocol["tasks"]:
        for method in METHODS:
            directory = run / "actual" / frozen["task_id"] / method
            verify_seal(directory)
            row = read(directory / "slot.json")
            relative = f'{row["task_id"]}/{method}'
            if sha(directory / "slot.json") != complete["slot_hashes"][relative]:
                raise ValueError("actual slot seal differs")
            if row["full_task_success"] and (row["actual_steps"] != 13500 or not row["original_independent_gates_passed"] or not row["quality"]):
                raise ValueError("successful actual lacks full steps/gates/quality")
            rows.append(row)
    predictions = [row for frozen in protocol["tasks"] for row in read(run / "planning" / frozen["task_id"] / "candidate_registry.json")]
    for frozen in protocol["tasks"]:
        for directory in (run / "planning" / frozen["task_id"] / "predictions").glob("*"):
            verify_seal(directory)
    slots = len(predictions); rollouts = sum(r["prediction_rollout_started"] for r in predictions)
    if slots > 48 or rollouts > 48 or len(rows) != 16 or sum(s["actual_steps"] for s in rows if s["unique_run"]) > 216000:
        raise ValueError("hard experiment budget violated")
    result = {"original_actual_validation": "PERFORMED_BY_UNCHANGED_FINAL_EXECUTOR",
        "validation_command_role": "read-only digest/gate/quality/budget reconciliation; no extra replay",
        "logical_slots": len(rows), "candidate_slots": slots, "prediction_rollouts": rollouts,
        "all_slots_terminal": True, "new_physics_steps_by_this_validation": 0,
        "method_denominator": 4, "all_sources_and_selections_unchanged": True,
        "actual_metric_missing_count": sum(s["unique_run"] and s["actual_steps"] > 0 and not s["quality"] for s in rows),
        "tool_error_count": sum(bool(r.get("tool_error")) for r in [*predictions, *rows])}
    write(output / "validation.json", result); seal(output)
    return result


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader(); writer.writerows(rows)


def report(run):
    validation = validate(run); protocol = verify_selections(run)
    if (run / "summary.json").exists():
        verify_seal(run); return read(run / "summary.json")
    actual = [read(run / "actual" / t["task_id"] / m / "slot.json") for t in protocol["tasks"] for m in METHODS]
    selected = {t["task_id"]: read(run / "planning" / t["task_id"] / "selection.json") for t in protocol["tasks"]}
    candidates = {t["task_id"]: read(run / "planning" / t["task_id"] / "candidate_registry.json") for t in protocol["tasks"]}
    states = []
    for method in METHODS:
        rows = [a for a in actual if a["method"] == method]
        states.append({"method": method, "denominator": 4,
            "precheck_rejected": sum("PRECHECK_REJECTED" in a["status"] for a in rows),
            "NO_PLAN": sum(a["status"] == "NO_PLAN" for a in rows),
            "entered_actual": sum(a["entered_actual"] for a in rows),
            "full_task_success": sum(a["full_task_success"] for a in rows),
            "original_independent_gates_passed": sum(a["original_independent_gates_passed"] for a in rows),
            "clearance_30mm_met_full_success": sum(a["full_task_success"] and a["clearance_30mm_met"] is True for a in rows)})
    _csv(run / "result_tables/method_status.csv", states)
    metrics = ("I_support", "I_route_key_legacy", "I_full", "L_full", "L_support", "d_support",
               "base_translation_peak_m", "base_rotation_peak_rad", "torque_saturation_count")
    quality_rows = [{"task_id": a["task_id"], "method": a["method"], "status": a["status"],
        "full_success": a["full_task_success"], "alias_of": a.get("alias_of_method"),
        **{k: (a.get("quality") or {}).get(k) for k in metrics}} for a in actual]
    _csv(run / "result_tables/paired_quality.csv", quality_rows)
    pairing = []
    for method in ("OI", "OC"):
        for baseline in ("Z0", "G0"):
            paired = []
            for t in protocol["tasks"]:
                first = next(a for a in actual if a["task_id"] == t["task_id"] and a["method"] == method)
                second = next(a for a in actual if a["task_id"] == t["task_id"] and a["method"] == baseline)
                if first["full_task_success"] and second["full_task_success"] and first["quality"] and second["quality"]:
                    paired.append({"task_id": t["task_id"], **{k: first["quality"][k] - second["quality"][k] for k in metrics}})
            pairing.append({"method": method, "baseline": baseline, "paired_complete_count": len(paired),
                "method_full_success_over_4": next(s["full_task_success"] for s in states if s["method"] == method),
                "baseline_full_success_over_4": next(s["full_task_success"] for s in states if s["method"] == baseline),
                "paired_mean_difference": {k: float(np.mean([p[k] for p in paired])) if paired else None for k in metrics}, "tasks": paired})
    write(run / "result_tables/paired_comparisons.json", pairing)
    search_rows = []
    for tid, selection in selected.items():
        rows = candidates[tid]; ranks = rank_candidates(rows); seeds = rank_candidates(rows[:4])
        search_rows.append({"task_id": tid, "seed_strict": seeds["strict"]["candidate_id"] if seeds["strict"] else None,
            "seed_I_support": seeds["strict"]["prediction_metrics"]["I_support"] if seeds["strict"] else None,
            "all_strict": selection["strict_winner"], "tie_selected": selection["tie_selected_winner"],
            "OI_source": selection["preferences"]["A"]["source_candidate_id"],
            "OC_source": selection["preferences"]["B"]["source_candidate_id"],
            "strict_I_support": ranks["strict"]["prediction_metrics"]["I_support"] if ranks["strict"] else None,
            "strict_gain_vs_four_seeds": seeds["strict"]["prediction_metrics"]["I_support"] - ranks["strict"]["prediction_metrics"]["I_support"] if seeds["strict"] and ranks["strict"] else None,
            "selected_OI_is_seed": selection["preferences"]["A"]["source_candidate_id"] in ("C00", "C01", "C02", "C03"),
            **selection["budget"]})
    _csv(run / "result_tables/search_contribution.csv", search_rows)
    costs = []
    for tid, rows in candidates.items():
        for row in rows:
            c = row.get("costs") or {}
            costs.append({"task_id": tid, "stage": "prediction", "id": row["candidate_id"],
                "slots": 1, "main_physics_steps": c.get("prediction_physics_steps", 0),
                **{k: c.get(k, 0) for k in ("private_preview_physics_steps", "preview_calls", "independent_saved_torque_replay_steps", "native_geometry_query_calls", "qp_solve_calls", "elapsed_wall_s")}})
    for a in actual:
        c = a.get("costs") or {}
        costs.append({"task_id": a["task_id"], "stage": "actual", "id": a["method"], "slots": 1,
            "unique_run": a["unique_run"], "alias_of": a.get("alias_of_method"), "main_physics_steps": c.get("actual_physics_steps", 0),
            **{k: c.get(k, 0) for k in ("private_preview_physics_steps", "preview_calls", "independent_saved_torque_replay_steps", "native_geometry_query_calls", "qp_solve_calls", "elapsed_wall_s")}})
    _csv(run / "result_tables/costs.csv", costs)
    cost_total = {key: sum(c[key] for c in costs) for key in ("main_physics_steps", "private_preview_physics_steps", "preview_calls", "independent_saved_torque_replay_steps", "native_geometry_query_calls", "qp_solve_calls", "elapsed_wall_s")}
    cost_total["initial_geometry_queries"] = sum(t["initial_geometry_query_count"] for t in protocol["tasks"])
    cost_total["phase_nesting"] = "main prediction/actual, nested private ten-step preview, independent final torque replay, and quality geometry are separate integrations/queries; native geometry total includes quality"
    write(run / "result_tables/cost_totals.json", cost_total)
    teacher = []
    for task in protocol["tasks"]:
        tid = task["task_id"]; s = selected[tid]
        for row in candidates[tid]:
            bound = [a for a in actual if a["task_id"] == tid and a["plan_sha256"] == row["plan_sha256"]]
            validated = any(a["full_task_success"] and a["original_independent_gates_passed"] and a["prediction_actual_consistent"] is True for a in bound)
            for preference in ("A", "B"):
                teacher.append({"task_id": tid, "group_id": task["group_id"], "task_sha256": task["task_sha256"],
                    "environment": {k: task[k] for k in ("seed", "mother_source_sha256", "obstacle_name", "W_support", "related_pair_ids")},
                    "preference": preference, "family": row["family"], "z_m": row["plan"]["z_m"],
                    "candidate_id": row["candidate_id"], "parent_candidate_id": row["parent_candidate_id"],
                    "source": row["source"], "prediction_eligible": admissible(row), "prediction_metrics": row.get("prediction_metrics"),
                    "actual_bindings": [{"method": a["method"], "status": a["status"], "quality": a["quality"],
                                         "original_gates": a["original_independent_gates_passed"], "alias_of": a.get("alias_of_method")} for a in bound],
                    "actual_status": "VALIDATED" if validated else "NOT_RUN" if not bound else "NOT_VALIDATED",
                    "qualification": "validated_execution_example" if validated else "prediction_only",
                    "strict_A_member": row["candidate_id"] == s["strict_winner"],
                    "tie_A_member": row["candidate_id"] in s["tie_group_ids"], "tie_A_selected": row["candidate_id"] == s["tie_selected_winner"],
                    "B_available_member": row["candidate_id"] in s["B_available_ids"],
                    "finite_non_dominated_member": row["candidate_id"] in s["finite_non_dominated_ids"],
                    "plan_sha256": row["plan_sha256"], "future_split_unit": task["group_id"]})
    with (run / "teacher_records.jsonl").open("x", encoding="utf8") as f:
        for row in teacher:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    all_rows = [r for rows in candidates.values() for r in rows]
    summary = {"research_delivery_complete": validation["all_slots_terminal"] and validation["tool_error_count"] == 0 and validation["actual_metric_missing_count"] == 0,
        "optimizer_implemented": True, "continuous_candidates_constructed": sum(s["budget"]["continuous_candidates_constructed"] for s in selected.values()),
        "continuous_candidates_evaluated": sum(s["budget"]["non_historical_continuous_candidates_evaluated"] for s in selected.values()),
        "candidate_slots_consumed": len(all_rows), "prediction_rollouts_started": sum(r["prediction_rollout_started"] for r in all_rows),
        "predicted_admissible_plan_found": {tid: s["strict_winner"] is not None for tid, s in selected.items()},
        "predicted_preference_plan_found": {tid: {p: v["selected_plan"] is not None for p, v in s["preferences"].items()} for tid, s in selected.items()},
        "actual_method_slots": len(actual), "actual_unique_runs": sum(a["unique_run"] for a in actual),
        "full_task_success_by_method": {s["method"]: s["full_task_success"] for s in states},
        "original_independent_gate_pass_by_method": {s["method"]: s["original_independent_gates_passed"] for s in states},
        "preferred_clearance_30mm_met_by_method": {s["method"]: s["clearance_30mm_met_full_success"] for s in states},
        "actual_preference_condition_confirmed": {tid: {"A_prediction_reproduced": next(a["prediction_actual_consistent"] for a in actual if a["task_id"] == tid and a["method"] == "OI"),
            "B_full_success_and_30mm": any(a["full_task_success"] and a["clearance_30mm_met"] is True and a["prediction_actual_consistent"] is True for a in actual if a["task_id"] == tid and a["method"] == "OC")} for tid in selected},
        "improvement_over_zero": [p for p in pairing if p["baseline"] == "Z0"],
        "improvement_over_geometric_rule": [p for p in pairing if p["baseline"] == "G0"],
        "selected_seed_vs_optimized": search_rows,
        "prediction_actual_consistency": {f'{a["task_id"]}/{a["method"]}': a["prediction_actual_consistent"] for a in actual},
        "finite_teacher_available": bool(teacher), "teacher_rows": len(teacher),
        "validated_execution_example_rows": sum(r["qualification"] == "validated_execution_example" for r in teacher),
        "costs": cost_total, "new_training_runs": 0, "neural_sampling_calls": 0, "optimizer_updates": 0,
        "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED", "hardware_safety": "NOT_ESTABLISHED",
        "task_structure": "four tasks paired from two mother scenes; no population/generalization guarantee",
        "algorithm_producer_commit": protocol["algorithm_producer_commit"], "base_publication_commit": protocol["base_publication_commit"]}
    _figures(run, protocol, actual, selected, costs)
    _markdown(run, summary, states, quality_rows, pairing)
    write(run / "summary.json", summary)
    seal(run)
    return summary


def _figures(run, protocol, actual, selected, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .task_anchored_reference import TaskAnchoredResidualPlan, _target
    output = run / "figures"; output.mkdir(exist_ok=True)
    colors = {"Z0": "#637687", "G0": "#d68e36", "OI": "#177f80", "OC": "#9b4da0"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, frozen in zip(axes.flat, protocol["tasks"]):
        tid = frozen["task_id"]; task = TaskSpec.from_dict(read(run / "frozen_tasks" / tid / "task.json"))
        times = np.linspace(0., 27., 2701); base = np.array([_target(task).sample(t)[0] for t in times])
        basis = np.asarray(__import__("v6_4.task_anchored_reference", fromlist=["build_reference_definition"]).build_reference_definition(task)["transverse_bases"][2])
        origin = base[0]
        # Fixed world e1 and base route chord projection, same for all methods.
        chord = base[-1] - base[0]; chord /= np.linalg.norm(chord)
        for a in [r for r in actual if r["task_id"] == tid]:
            method = a["method"]; directory = run / "actual" / tid / a.get("evidence_method", method)
            if a["quality"]:
                with np.load(directory / "quality/fresh_states.npz", allow_pickle=False) as f:
                    p = f["continuum_position"]
                ax.plot((p - origin) @ chord, (p - origin) @ basis[:, 0], color=colors[method], label=method + " actual", linewidth=1.5)
            plan_path = run / "actual" / tid / method / "selected_plan.json"
            if plan_path.exists():
                plan = TaskAnchoredResidualPlan.from_dict(read(plan_path))
                p = base + plan.offset_kinematics(times)[0]
                ax.plot((p - origin) @ chord, (p - origin) @ basis[:, 0], color=colors[method], linestyle="--", alpha=.55, linewidth=.8)
        center = np.asarray(task.scenario["workspace_obstacles"][1]["center_w"]) - origin
        from matplotlib.patches import Circle
        ax.add_patch(Circle((center @ chord, center @ basis[:, 0]), .025, color="#c64848", alpha=.25))
        ax.set(title=tid, xlabel="World chord projection (m)", ylabel="World e1 projection (m)")
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.suptitle("Frozen new tasks: actual routes (solid), selected references (dashed)\n2D projection; native body-pair clearance is evaluated separately")
    fig.savefig(output / "routes.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = [t["task_id"].replace("c1_mother_", "M") for t in protocol["tasks"]]
    for column, (key, title, scale) in enumerate((("I_support", "Fixed support intervention (rad/s)", 1.), ("L_full", "Complete tip path (m)", 1.), ("d_support", "Related body/sphere clearance (mm)", 1000.))):
        ax = axes[0, column]
        for j, method in enumerate(METHODS):
            values = [next(a for a in actual if a["task_id"] == t["task_id"] and a["method"] == method) for t in protocol["tasks"]]
            y = [(a["quality"][key] * scale if a["full_task_success"] and a["quality"] else np.nan) for a in values]
            ax.bar(np.arange(4) + (j - 1.5) * .19, y, .18, color=colors[method], label=method)
        if key == "d_support": ax.axhline(30., color="#b84848", linestyle="--", linewidth=1)
        ax.set(title=title, xticks=range(4), xticklabels=labels); ax.grid(axis="y", alpha=.15)
    ax = axes[1, 0]
    ax.bar(METHODS, [sum(a["full_task_success"] for a in actual if a["method"] == m) for m in METHODS], color=[colors[m] for m in METHODS])
    ax.set(title="Full Task + original gates (denominator 4)", ylim=(0, 4.5), yticks=range(5))
    ax = axes[1, 1]
    names = ("main_physics_steps", "private_preview_physics_steps", "independent_saved_torque_replay_steps")
    for j, stage in enumerate(("prediction", "actual")):
        ax.bar(np.arange(3) + (j - .5) * .35, [sum(c[k] for c in costs if c["stage"] == stage) for k in names], .34, label=stage)
    ax.set(title="Integration work (nested preview charged)", xticks=range(3), xticklabels=("main", "preview", "replay")); ax.legend()
    ax = axes[1, 2]
    for tid, selection in selected.items():
        rows = read(run / "planning" / tid / "candidate_registry.json")
        ax.plot([int(r["candidate_id"][1:]) + 1 for r in rows if admissible(r)],
                [r["prediction_metrics"]["I_support"] for r in rows if admissible(r)], "o-", label=tid.replace("c1_mother_", "M"), markersize=3)
    ax.axvline(4.5, color="gray", linestyle="--"); ax.set(title="Shared candidate pool: prediction quality", xlabel="Consumed slot", ylabel="I_support rad/s"); ax.legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("C.1 quality, full-success denominator and computation\nMissing/failed prefixes excluded from quality bars; four tasks come from two mother scenes")
    fig.savefig(output / "quality_cost.png", dpi=170); plt.close(fig)


def _markdown(run, summary, states, quality_rows, pairing):
    text = ["# V6.4-C.1 连续路线优化与多目标质量教师", "",
        "实现可调用的非学习连续路线规划器；四个冻结新任务来自两个母场景。两偏好共享每任务最多12候选槽。完整预测与最终独立执行分别记账。", "",
        f"Producer `{summary['algorithm_producer_commit']}`；开发基点 `{summary['base_publication_commit']}`。历史 B.2/B.3/B.3.1 结论保持。", "",
        "## 完整方法状态（各分母4）", "", "|方法|进入actual|完整Task|原独立门禁|完整且30mm|NO_PLAN|", "|---|---:|---:|---:|---:|---:|"]
    for r in states:
        text.append(f"|{r['method']}|{r['entered_actual']}|{r['full_task_success']}/4|{r['original_independent_gates_passed']}/4|{r['clearance_30mm_met_full_success']}/4|{r['NO_PLAN']}|")
    text += ["", "## 同任务质量", "", "|任务|方法|完整成功|I_support rad/s|I_key legacy|I_full|L_full m|d_support mm|", "|---|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value, scale=1.): return "null" if value is None else f"{value*scale:.8g}"
    for r in quality_rows:
        text.append(f"|{r['task_id']}|{r['method']}|{r['full_success']}|{fmt(r['I_support'])}|{fmt(r['I_route_key_legacy'])}|{fmt(r['I_full'])}|{fmt(r['L_full'])}|{fmt(r['d_support'],1000)}|")
    text += ["", "完整质量配对只使用双方同任务均完整成功的子集；缺失不填惩罚常数。源向量、10D/7D分量、最小几何witness、基座漂移和力矩饱和见质量原件与CSV。", "",
        "## 连续搜索贡献", "", "|任务|四初值strict|全池strict|OI输出|OC输出|strict新增收益 rad/s|OI来自初值|", "|---|---|---|---|---|---:|---|"]
    for r in summary["selected_seed_vs_optimized"]:
        text.append(f"|{r['task_id']}|{r['seed_strict']}|{r['all_strict']}|{r['OI_source']}|{r['OC_source']}|{fmt(r['strict_gain_vs_four_seeds'])}|{r['selected_OI_is_seed']}|")
    text += ["", "A的工程并列组锚定strict最低值+0.001rad/s，再按完整路径、系数范数、family、ID选取；strict与实际输出分别保存。B严格要求离散相关连续体—路线球净空≥0.030m，再最小完整路径；未满足返回NO_PLAN。30mm是新增质量偏好，不替换任何原硬安全距离。", "",
        "## 配对效果—成本", "", "|方法|基线|完整配对数|平均ΔI_support|平均ΔL_full m|平均Δd_support mm|", "|---|---|---:|---:|---:|---:|"]
    for p in pairing:
        m = p["paired_mean_difference"]
        text.append(f"|{p['method']}|{p['baseline']}|{p['paired_complete_count']}|{fmt(m['I_support'])}|{fmt(m['L_full'])}|{fmt(m['d_support'],1000)}|")
    text += ["", "差值为优化方法−基线。优化器拥有额外私有模型预测，不能宣称同计算预算优势。I_support是固定前驱/关键两段并集的新指标，不能重算或改判旧key-only 3.27%/1.03%结论。", "",
        f"候选槽 {summary['candidate_slots_consumed']}/48；内部预测 {summary['prediction_rollouts_started']}；非历史连续点评价 {summary['continuous_candidates_evaluated']}。最终逻辑方法槽 {summary['actual_method_slots']}/16，唯一实际运行 {summary['actual_unique_runs']}。", "",
        "```json", json.dumps(summary["costs"], indent=2, ensure_ascii=False), "```", "",
        "搜索阶段复用原完整控制链与全部在线守卫；原runner自带全身检查的成本也计入。搜索未追加独立力矩/区间重放，状态为NOT_RUN_SEARCH_SCREENING。最终actual从初态重新计算反馈力矩，执行原五项独立门禁；whole-body为50Hz/subdivisions4，robot-target为原生500Hz，不能称作连续时间安全。", "",
        "![新任务路线](figures/routes.png)", "", "![质量与成本](figures/quality_cost.png)", "",
        "## 结论边界与有限教师", "",
        f"研究执行完成：`{summary['research_delivery_complete']}`。工程能力、预测偏好、actual确认与相对基线收益按summary.json独立列示；正结果缺失不追加预算。教师共{summary['teacher_rows']}行，其中{summary['validated_execution_example_rows']}行关联独立通过且预测复现的最终执行；其余prediction_only，不作成功示范。左右任务与全部候选保留同一group_id。", "",
        "新训练、神经采样、学习optimizer updates均为0。本轮效果如有，来自计费的非学习搜索。Diffusion优势、总体泛化、模型误差鲁棒性、连续时间或硬件安全均未建立。部署保持NOT_MET。", "",
        "## 实际命令", "", "```sh",
        f"python -B -X utf8 -m v6_4.continuous_route_optimizer prepare --output v6_4/output/{run.name}",
        f"python -B -X utf8 -m v6_4.continuous_route_optimizer run-all --run v6_4/output/{run.name}", "```", "",
        "环境、sys.argv、cwd、开始/结束时间与各阶段状态见source_identity、candidate started/result、actual slot和command_receipts.jsonl。run-all完成阶段仅验哈希后跳过。validate只对已执行的原验证原件作一致性核对，不额外产生重放。"]
    (run / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf8")


def dispatch(command, run):
    start = datetime.now(timezone.utc).isoformat()
    verify_frozen(run)
    if (run / "summary.json").exists():
        verify_seal(run); return
    status = "TOOL_ERROR"
    try:
        if command in ("analyze-history", "run-all"): analyze_history(run)
        if command in ("optimize", "run-all"): optimize_tasks(run)
        if command in ("execute-selected", "run-all"): execute_selected(run)
        if command in ("validate", "run-all"): validate(run)
        # Receipts must be included in the final seal, so report is dispatched
        # after its successful receipt is appended below.
        status = "COMPLETED"
    finally:
        with (run / "command_receipts.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"command": command, "argv": [sys.executable, *sys.argv],
                "cwd": str(Path.cwd()), "started_utc": start, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "status": status, "exit_code": 0 if status == "COMPLETED" else 1}) + "\n")
    if command in ("report", "run-all"):
        result = report(run)
        print(json.dumps({"event": "C1_REPORT_COMPLETE", "delivery": result["research_delivery_complete"],
                          "candidate_slots": result["candidate_slots_consumed"], "actual_unique_runs": result["actual_unique_runs"]}), flush=True)
