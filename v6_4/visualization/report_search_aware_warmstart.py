"""Reporting-only C.3 evidence aggregation; no physics, training or sampling.

Missing terminal evidence remains missing. This exporter never changes a
selection, checkpoint, candidate registry, trace, or acceptance result.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ENDPOINTS = ("R8", "R12", "N8", "S8", "D8")
PREFERENCES = ("A", "B")
METRICS = ("I_support", "L_full", "d_support", "base_translation_peak_m", "base_rotation_peak_rad")
WORK = ("prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls",
        "independent_saved_torque_replay_steps", "qp_solve_calls")
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def _csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()})


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sum_complete(values):
    values = list(values)
    return sum(values) if values and all(_finite(v) for v in values) else None


def _full(slot):
    return bool(slot and slot.get("report_evidence_verified") is not False and
        slot.get("full_task_success") is True and slot.get("original_independent_gates_passed") is True
        and slot.get("actual_steps") == 13500 and all(slot.get("five_gates", {}).get(g) is True for g in GATES))


def _reference(slot, preference):
    keys = ("L_full", "I_support") if preference == "A" else ("L_full", "d_support")
    return bool(_full(slot) and all(_finite((slot.get("quality") or {}).get(k)) for k in keys))


def _near(slot, reference, preference):
    if not _reference(reference, preference):
        return None
    if not _full(slot):
        return False
    if not _reference(slot, preference):
        return None
    q, r = slot["quality"], reference["quality"]
    return bool(q["L_full"] <= r["L_full"] + .005 and
        (q["I_support"] <= r["I_support"] + .001 if preference == "A" else q["d_support"] >= .030))


def _pred_full(row):
    q = row.get("prediction_metrics") or {}
    return bool(row.get("prediction_admissible") is True and row.get("prediction_task_passed") is True and
        row.get("online_guards_passed") is True and row.get("prediction_steps") == 13500 and
        q.get("native_state_count") == 13501 and _finite(q.get("saved_horizon_s")) and abs(q["saved_horizon_s"] - 27.) < 1e-8 and
        all(_finite(q.get(k)) for k in ("I_support", "L_full")))


def _pred_b30(row):
    q = row.get("prediction_metrics") or {}
    return bool(_pred_full(row) and q.get("clearance_status") == "MEASURED" and _finite(q.get("d_support")) and q["d_support"] >= .030)


def _pred_near(row, reference, preference):
    if not _reference(reference, preference):
        return None
    if not (_pred_full(row) if preference == "A" else _pred_b30(row)):
        return False
    q, r = row["prediction_metrics"], reference["quality"]
    return bool(q["L_full"] <= r["L_full"] + .005 and
        (q["I_support"] <= r["I_support"] + .001 if preference == "A" else q["d_support"] >= .030))


def _hit(rows, predicate, budget, complete, *, applicable=True):
    if not applicable:
        return {"status": "N/A_R12_NO_QUALIFIED_ACTUAL", "slot": None, "sort_encoding": None}
    if rows is None:
        return {"status": "NOT_RUN", "slot": None, "sort_encoding": None}
    first = next((i for i, row in enumerate(rows[:budget], 1) if predicate(row)), None)
    if first is not None:
        return {"status": "HIT", "slot": first, "sort_encoding": first}
    return {"status": "RIGHT_CENSORED" if complete else "TECHNICAL_INCOMPLETE", "slot": None,
        "right_censored_budget": budget if complete else None, "slots_observed": len(rows[:budget]),
        "sort_encoding": budget + 1 if complete else None, "encoding_is_observed_hit": False}


class Evidence:
    def __init__(self, run):
        self.run = Path(run).resolve(); self.inputs = {}; self.unverified = []

    def record(self, path):
        path = Path(path).resolve()
        self.inputs[str(path)] = _sha(path)

    def pending(self, path, reason):
        self.unverified.append({"path": str(Path(path).resolve()), "reason": reason})

    def read(self, path, default=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.run / path
        if not path.exists():
            return default
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        self.record(path)
        return value

    def phase(self, phase, *, bindings, tasks, endpoints):
        from v6_4.closed_loop_warmstart_validation import verify_phase_selections
        path = self.run / phase / "sealed_selections/all_selections.json"
        manifest = self.read(path)
        if manifest is None:
            self.pending(path, "phase selections have not been sealed")
            return None
        if (manifest.get("schema") != "v64_c3_all_phase_selections_sealed_v1" or not manifest.get("files") or
                manifest.get("phase") != ("VAL" if phase == "closed_loop_val" else "TEST") or
                manifest.get("selection_reads_actual") is not False):
            raise ValueError("phase manifest lacks the original sealed evidence schema")
        verify_phase_selections(path.parents[1], bindings=bindings)
        expected = {(t, e) for t in tasks for e in endpoints}
        entries = manifest.get("entries", [])
        if (set(manifest.get("task_ids", [])) != set(tasks) or set(manifest.get("endpoints", [])) != set(endpoints) or
                len(entries) != len(expected) or {(e["task_id"], e["endpoint"]) for e in entries} != expected):
            raise ValueError("sealed phase selection schedule differs from the frozen plan")
        for filename in manifest["files"]:
            self.record(filename)
        return manifest

    def model_freeze(self):
        from v6_4.search_aware_warmstart_experiment import verify_model_freeze
        path = self.run / "model_freeze.json"; frozen = self.read(path)
        if frozen is None:
            self.pending(path, "models and retrieval pool have not been frozen")
            return None
        if (frozen.get("schema") != "v64_c3_frozen_models_and_retrieval_v1" or not frozen.get("artifacts") or
                set(frozen.get("selected_checkpoints", {})) != {"D", "S"}):
            raise ValueError("model freeze lacks the original artifact/checkpoint bindings")
        verify_model_freeze(self.run)
        for filename in frozen["artifacts"]:
            self.record(filename)
        return frozen

    def slots(self, phase, manifest=None):
        from v6_4.closed_loop_warmstart_validation import verify_actual_slot
        result = {}
        for path in sorted((self.run / phase / "actual").glob("*/*/slot.json")):
            slot = self.read(path)
            key = (slot["task_id"], slot["endpoint"], slot["preference"])
            if path.parent.name != key[1] + "_" + key[2] or path.parent.parent.name != key[0]:
                raise ValueError("actual slot path and Task/endpoint/preference binding differ")
            if key in result:
                raise ValueError("duplicate logical actual slot")
            seal = self.read(path.parent / "manifest.json")
            verified = False
            if seal is None:
                self.pending(path, "actual slot has not been sealed")
            else:
                if "slot.json" not in seal:
                    raise ValueError("actual manifest does not bind its slot.json")
                verify_actual_slot(path.parent)
                for relative in seal:
                    self.record(path.parent / relative)
                if slot.get("alias_of_slot"):
                    source = Path(slot["alias_of_slot"])
                    original_seal = self.read(source.parent / "manifest.json")
                    if "slot.json" not in original_seal:
                        raise ValueError("actual alias source manifest does not bind its slot.json")
                    for relative in original_seal:
                        self.record(source.parent / relative)
                if manifest is None:
                    self.pending(path, "actual slot has no verified phase selection manifest")
                else:
                    entry = next((e for e in manifest["entries"] if (e["task_id"], e["endpoint"]) == key[:2]), None)
                    binding = slot.get("actual_binding") or {}
                    if (entry is None or key[2] not in PREFERENCES or
                            slot.get("phase") != manifest["phase"] or
                            slot.get("selection_sha256") != entry["selection_sha256"] or
                            slot.get("task_sha256") != entry["task_sha256"] or
                            any(binding.get(k) != slot.get(k) for k in
                                ("task_sha256", "selection_sha256", "plan_sha256", "alias_identity"))):
                        raise ValueError("actual slot differs from its sealed phase Task/selection binding")
                    verified = True
            slot = {**slot, "report_evidence_verified": verified}
            result[key] = slot
        return result

    def stream(self, stage, task, method, budget):
        directory = self.run / {"teacher": "teacher_search", "val": "closed_loop_val/search", "test": "test_search"}[stage] / task / method
        planning = directory / "planning" / task
        rows = self.read(planning / "candidate_registry.json")
        proposals = self.read(planning / "proposals.json")
        selection = self.read(planning / "selection.json")
        cost = self.read(directory / "planning_cost.json")
        outer = self.read(directory / "outer_process.json")
        initialization = self.read(directory / "initializer_proposals.json", {})
        generated = list(initialization.get("proposals", {}).values())
        complete = bool(selection and cost and rows is not None and proposals is not None and
            (selection.get("budget") or {}).get("stop_reason") != "TOOL_ERROR")
        if complete:
            consumed = (selection.get("budget") or {}).get("slots_consumed")
            digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if consumed != len(rows) or consumed > budget or selection.get("registry_content_sha256") != digest:
                raise ValueError("search selection and consumed candidate registry differ")
            if stage in ("val", "test"):
                complete = bool(outer and outer.get("exit_code") == 0 and _finite(outer.get("elapsed_wall_s")))
                if not complete:
                    self.pending(directory / "outer_process.json", "formal search has no successful process receipt")
        statuses = sorted({r.get("status", "MISSING_STATUS") for r in rows}) if rows is not None else []
        result = {"stage": stage, "task_id": task, "stream": method, "budget_limit": budget,
            "protocol_completed": complete, "rows": rows, "proposals": proposals, "selection": selection,
            "cost": cost, "stream_path": str(directory), "parameter_constructions": len(proposals) if proposals is not None else None,
            "consumed_candidate_slots": len(rows) if rows is not None else None,
            "prediction_rollouts_started": sum(r.get("prediction_rollout_started") is True for r in rows) if rows is not None else None,
            "raw_rejected": sum(r.get("status") == "INITIALIZER_RAW_REJECTED" for r in rows) if rows is not None else None,
            "nominal_status_counts": {status: sum(r.get("status", "MISSING_STATUS") == status for r in rows) for status in statuses} if rows is not None else None,
            "nominal_failed_rollouts": sum(r.get("prediction_rollout_started") is True and not _pred_full(r) for r in rows) if rows is not None else None,
            "nominal_positive_step_incomplete_rollouts": sum(0 < r.get("prediction_steps", 0) < 13500 for r in rows) if rows is not None else None,
            "legal_exact_cache_hits": sum(bool(p.get("cache_hit_candidate_id")) for p in proposals) if proposals is not None else None,
            "end_to_end_cold_planning_s": (cost or {}).get("end_to_end_cold_planning_s"),
            "optimizer_elapsed_s": (selection or {}).get("elapsed_wall_s"),
            "outer_process_wall_s": (outer or {}).get("elapsed_wall_s"),
            "outer_process_exit_code": (outer or {}).get("exit_code"),
            "initializer_setup_s": initialization.get("timing", {}).get("total_setup_s"),
            "DDIM_samples_generated": sum(p.get("ddim_sample_units", 0) for p in generated),
            "regression_forward_units": sum(p.get("regression_forward_units", 0) for p in generated)}
        for name in WORK:
            result[name] = _sum_complete((r.get("costs") or {}).get(name, 0 if r.get("status") == "INITIALIZER_RAW_REJECTED" else None) for r in rows) if rows is not None else None
        return result


def _prefix(evidence, stream, budget):
    path = Path(stream["stream_path"]) / "planning" / stream["task_id"] / f"prefix_{budget:02d}.json"
    value = evidence.read(path)
    rows = stream["rows"]
    if value is None:
        return {"exists": False, "rows": None, "protocol_completed": False, "budget": budget}
    consumed = value["budget"]["slots_consumed"]
    if rows is None or consumed > len(rows):
        raise ValueError("sealed prefix has no matching registry")
    pool = rows[:consumed]
    digest = hashlib.sha256(json.dumps(pool, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != value["registry_content_sha256"]:
        raise ValueError("sealed prefix candidate pool changed")
    return {"exists": True, "rows": pool, "protocol_completed": value.get("protocol_completed") is True,
            "budget": budget, "selection": value, "optimizer_elapsed_s": value.get("elapsed_wall_s")}


def _endpoint_cost(stream, endpoint, prefix):
    # R8 is an actually saved prefix of one R12 flow. Only its optimizer timer
    # was measured. Assigning the R12 total cold timer to R8 would be false.
    rows = prefix["rows"] if endpoint == "R8" else stream["rows"]
    result = {"end_to_end_cold_planning_s": None if endpoint == "R8" else stream["end_to_end_cold_planning_s"],
        "outer_process_wall_s": None if endpoint == "R8" else stream["outer_process_wall_s"],
        "optimizer_elapsed_s": prefix.get("optimizer_elapsed_s") if endpoint == "R8" else stream["optimizer_elapsed_s"],
        "planning_time_scope": "measured_optimizer_prefix_only; cold_request_total_not_measured" if endpoint == "R8" else "measured_cold_request_total",
        "consumed_candidate_slots": len(rows) if rows is not None else None,
        "raw_rejected": sum(r.get("status") == "INITIALIZER_RAW_REJECTED" for r in rows) if rows is not None else None}
    for name in WORK:
        result[name] = _sum_complete((r.get("costs") or {}).get(name, 0 if r.get("status") == "INITIALIZER_RAW_REJECTED" else None) for r in rows) if rows is not None else None
    return result


def _comparison(method, comparator, tasks, slots, costs):
    pairs = [(t, p) for t in tasks for p in PREFERENCES]
    complete = all((t, m, p) in slots for t, p in pairs for m in (method, comparator))
    capable = [(t, p) for t, p in pairs if _full(slots.get((t, comparator, p)))]
    qualified = [(t, p) for t, p in pairs if _reference(slots.get((t, comparator, p)), p)]
    missing = [(t, p) for t, p in capable if not _full(slots.get((t, method, p)))]
    quality_missing = [(t, p) for t, p in capable if (t, p) not in qualified]
    method_quality_missing = [(t, p) for t, p in qualified if _full(slots.get((t, method, p))) and
        not _reference(slots.get((t, method, p)), p)]
    quality_fail = [(t, p) for t, p in qualified if _near(slots.get((t, method, p)), slots[(t, comparator, p)], p) is False]
    b30_lost = [t for t in tasks if _full(slots.get((t, comparator, "B"))) and
        slots[(t, comparator, "B")].get("clearance_30mm_met") is True and not
        (_full(slots.get((t, method, "B"))) and slots[(t, method, "B")].get("clearance_30mm_met") is True)]
    r12_set = [(t, p) for t, p in pairs if _reference(slots.get((t, "R12", p)), p)]
    r12_quality_missing = [(t, p) for t, p in pairs if _full(slots.get((t, "R12", p))) and (t, p) not in r12_set]
    r12_method_quality_missing = [(t, p) for t, p in r12_set if _full(slots.get((t, method, p))) and
        not _reference(slots.get((t, method, p)), p)]
    r12_preserved = bool(r12_set) and not r12_quality_missing and all(
        _near(slots.get((t, method, p)), slots[(t, "R12", p)], p) is True for t, p in r12_set)
    means, metric_pairs = {}, []
    for t, p in pairs:
        a, b = slots.get((t, method, p)), slots.get((t, comparator, p))
        if not (_full(a) and _full(b)):
            continue
        qa, qb = a.get("quality") or {}, b.get("quality") or {}
        row = {"task_id": t, "preference": p}
        for k in METRICS:
            row[k + "_difference"] = qa[k] - qb[k] if _finite(qa.get(k)) and _finite(qb.get(k)) else None
        metric_pairs.append(row)
    for p in PREFERENCES:
        means[p] = {k: _sum_complete(r[k + "_difference"] for r in metric_pairs if r["preference"] == p) for k in METRICS}
        n = sum(r["preference"] == p for r in metric_pairs)
        means[p] = {k: value / n if n and value is not None else None for k, value in means[p].items()}
        means[p]["complete_paired_tasks"] = n
    reductions = {}
    for key in ("end_to_end_cold_planning_s", "prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls"):
        a = _sum_complete(costs.get((t, method), {}).get(key) for t in tasks)
        b = _sum_complete(costs.get((t, comparator), {}).get(key) for t in tasks)
        reductions[key] = {"method_total": a, "comparator_total": b,
            "reduction_fraction": (b - a) / b if a is not None and b is not None and b > 0 else None}
    raw_method = _sum_complete(costs.get((t, method), {}).get("raw_rejected") for t in tasks)
    raw_comparator = _sum_complete(costs.get((t, comparator), {}).get("raw_rejected") for t in tasks)
    illegal_confounded = raw_method is None or raw_comparator is None or raw_method > raw_comparator
    observed = bool(complete and capable)
    quality_observed = bool(observed and qualified and not quality_missing and not method_quality_missing)
    quality_preserved = quality_observed and not missing and not quality_fail and not b30_lost
    cost_goal = quality_preserved and r12_preserved and not illegal_confounded and any(
        reductions[k]["reduction_fraction"] is not None and reductions[k]["reduction_fraction"] >= .10
        for k in ("end_to_end_cold_planning_s", "prediction_physics_steps"))
    method_full = sum(_full(slots.get((t, method, p))) for t, p in pairs)
    comparator_full = sum(_full(slots.get((t, comparator, p))) for t, p in pairs)
    return {"method": method, "comparator": comparator, "all_logical_slots_present": complete,
        "comparator_qualified_reference_endpoints": len(qualified),
        "comparator_full_actual_endpoints": len(capable),
        "comparator_quality_missing": [{"task_id": t, "preference": p} for t, p in quality_missing],
        "method_quality_missing": [{"task_id": t, "preference": p} for t, p in method_quality_missing],
        "capability_preserved_same_tasks": not missing if observed else None,
        "B30_preserved_same_tasks": not b30_lost if observed else None,
        "near_quality_preserved_against_comparator": not quality_fail if quality_observed else None,
        "R12_near_quality_preserved": r12_preserved if r12_set and complete and not r12_quality_missing and not r12_method_quality_missing else None,
        "R12_quality_missing": [{"task_id": t, "preference": p} for t, p in r12_quality_missing],
        "coverage_method": method_full, "coverage_comparator": comparator_full,
        "coverage_gain_without_lost_comparator_task": bool(observed and not missing and method_full > comparator_full),
        "capability_lost": [{"task_id": t, "preference": p} for t, p in missing],
        "near_quality_lost": [{"task_id": t, "preference": p} for t, p in quality_fail], "B30_lost_tasks": b30_lost,
        "metric_differences_on_complete_pairs": metric_pairs, "mean_metric_differences_on_complete_pairs": means,
        "cost_reductions_all_four_tasks_including_failed_requests": reductions,
        "raw_illegal_method": raw_method, "raw_illegal_comparator": raw_comparator,
        "cost_saving_confounded_by_more_raw_rejections": illegal_confounded,
        "engineering_10percent_goal_met": bool(cost_goal), "statistical_noninferiority_established": False}


def _plots(table_dir, prefixes, endpoints, per_task):
    if not prefixes and not per_task:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"R": "#4c566a", "N": "#d08770", "S": "#5e81ac", "D": "#a34c86"}
    files = []
    if any(r["prefix_complete_task_count"] == 4 for r in prefixes):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        panels = (("predicted_full_A", "A: complete nominal coverage", "full_27s_five_gates"),
                  ("predicted_B30", "B: nominal 30 mm coverage", "actual_B30"),
                  ("near_R12_A", "A: near actual R12 reference", "near_R12_count"),
                  ("near_R12_B", "B: near actual R12 reference", "near_R12_count"))
        for ax, (field, title, actual_key) in zip(axes.flat, panels):
            for method in ("R", "N", "S", "D"):
                data = [r for r in prefixes if r["method"] == method and r["prefix_complete_task_count"] == 4 and
                    (not field.startswith("near_R12") or r["R12_" + field[-1] + "_reference_available"] > 0)]
                if data:
                    ax.plot([r["budget"] for r in data], [r[field] for r in data], "o--", color=colors[method], label=method + " predicted")
                pref = "A" if field.endswith("A") else "B"
                for row in endpoints:
                    if row["endpoint"].startswith(method) and row["preference"] == pref and row["logical_slots_present"] == 4:
                        if actual_key == "near_R12_count" and row["R12_qualified_reference_count"] == 0:
                            continue
                        ax.scatter(int(row["endpoint"][1:]), row[actual_key], marker="*", s=135, color=colors[method], edgecolor="white", zorder=5)
            ax.set_title(title); ax.set_xticks([4, 8, 12]); ax.set_xlim(3.5, 12.5); ax.set_ylim(-.1, 4.3)
            ax.set_yticks(range(5)); ax.set_xlabel("Consumed-slot budget"); ax.set_ylabel("Endpoints / 4"); ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(handles, labels, fontsize=8, loc="lower right")
        fig.suptitle("Frozen TEST: predicted curves; stars = final actual logical endpoints (strict aliases retained)\nNo actual result exists at 4 slots; identical curves or points may overlap", fontsize=11)
        path = table_dir / "budget_coverage_near_quality.png"; fig.savefig(path, dpi=300); plt.close(fig); files.append(path)
    data = [r for r in per_task if r["full_27s_five_gates"] and _finite(r.get("end_to_end_cold_planning_s"))]
    if data:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        for ax, pref, metric, label in ((axes[0], "A", "actual_I_support", "Actual I_support [rad/s]"),
                                       (axes[1], "B", "actual_L_full", "Actual L_full [m]")):
            for endpoint in ENDPOINTS:
                rows = [r for r in data if r["endpoint"] == endpoint and r["preference"] == pref and _finite(r.get(metric))]
                summary = next((r for r in endpoints if r["endpoint"] == endpoint and r["preference"] == pref), {})
                if rows:
                    ax.scatter([r["end_to_end_cold_planning_s"] for r in rows], [r[metric] for r in rows],
                        color=colors[endpoint[0]], marker="s" if endpoint == "R12" else "o", s=55,
                        label=f"{endpoint}: {summary.get('full_27s_five_gates', 0)}/4 complete")
            ax.set_xlabel("Measured cold planning cost per shared A/B request [s]"); ax.set_ylabel(label)
            ax.set_title(pref + ": full 27 s + five gates only"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle("Actual quality vs planning cost; failures remain in tables and denominators\nR8 has no separately measured cold-request total and is omitted here", fontsize=11)
        path = table_dir / "actual_quality_planning_cost.png"; fig.savefig(path, dpi=300); plt.close(fig); files.append(path)
    return files


def _verify_selection(evidence, selection, freeze, manifest, val_slots):
    path = evidence.run / "closed_loop_val/model_selection.json"
    retained = evidence.read(path)
    if not selection or retained is None or manifest is None or freeze is None:
        evidence.pending(path, "closed-loop selection or its phase/model bindings are incomplete")
        return False
    if selection != retained or selection.get("phase_manifest_sha256") != _sha(
            evidence.run / "closed_loop_val/sealed_selections/all_selections.json"):
        raise ValueError("root model selection differs from its sealed VAL evidence")
    expected = {str((evidence.run / "closed_loop_val/actual" / t / (e + "_" + p) / "slot.json").resolve())
        for t in manifest["task_ids"] for e in manifest["endpoints"] for p in PREFERENCES}
    hashes = selection.get("actual_slot_files", {})
    if set(hashes) != expected:
        raise ValueError("model selection does not bind every VAL actual slot")
    if len(val_slots) != len(expected) or not all(s.get("report_evidence_verified") for s in val_slots.values()):
        evidence.pending(path, "scored VAL actual evidence is missing or not sealed")
        return False
    for filename, expected_sha in hashes.items():
        if _sha(filename) != expected_sha:
            raise ValueError("scored VAL actual evidence changed")
        evidence.record(filename)
    if (selection.get("closed_loop_val_completed") is not True or selection.get("test_read") is not False or
            freeze["selected_checkpoints"] != selection.get("selected_checkpoint_files")):
        raise ValueError("frozen selected checkpoints differ from the closed-loop VAL selection")
    for model, filename in freeze["selected_checkpoints"].items():
        if (filename not in freeze["artifacts"] or
                selection.get("selected_checkpoint_sha256", {}).get(model) != _sha(filename)):
            raise ValueError("selected model checkpoint lacks the frozen VAL byte identity")
    return True


def _verify_terminal_accounting(evidence, validation, streams, slots, val_slots, reserved, plan):
    """Recheck the saved validator's provenance without rerunning or writing it."""
    phases_complete = all(sum(s["protocol_completed"] for s in streams if s["stage"] == stage) == count
        for stage, count in (("teacher", 12), ("val", 10), ("test", 16)))
    if (not validation or not phases_complete or len(slots) != 40 or len(val_slots) != 20 or
            not all(s.get("report_evidence_verified") for s in [*slots.values(), *val_slots.values()])):
        evidence.pending(evidence.run / "validation/protocol_validation.json", "terminal search/actual evidence is incomplete")
        return False
    totals = {k: reserved.get(k, 0) for k in plan.get("budget_limits", {})}
    totals.update(candidate_slots_total=sum(totals.get(k, 0) for k in
        ("teacher_candidate_slots", "val_candidate_slots", "test_candidate_slots")),
        actual_slots_total=totals.get("val_actual_slots", 0) + totals.get("test_actual_slots", 0),
        ddim_samples_total=sum(totals.get(k, 0) for k in
            ("val_ddim_samples", "test_ddim_samples", "smoke_ddim_samples", "diagnostic_ddim_samples")))
    rows = [r for s in streams for r in (s["rows"] or [])]
    actuals = [*val_slots.values(), *slots.values()]
    errors = sum(bool(r.get("tool_error")) for r in [*rows, *actuals])
    current = {"candidate_slots": {stage: sum(len(s["rows"]) for s in streams if s["stage"] == stage)
            for stage in ("teacher", "val", "test")},
        "actual_logical_slots": {"val": len(val_slots), "test": len(slots)}, "budget_reservations": totals,
        "main_prediction_physics_steps": sum(r.get("prediction_steps", 0) for r in rows),
        "main_actual_physics_steps": sum(s.get("actual_steps", 0) for s in actuals if s.get("unique_run")),
        "tool_error_count": errors, "all_terminal": not errors}
    if any(validation.get(key) != value for key, value in current.items()):
        raise ValueError("saved terminal validation no longer matches current sealed evidence/accounting")
    receipt = evidence.read("actual_complete.json")
    if receipt is None:
        evidence.pending(evidence.run / "actual_complete.json", "final actual completion receipt is missing")
        return False
    hashes = {p.relative_to(evidence.run).as_posix(): _sha(p)
        for p in sorted((evidence.run / "frozen_test/actual").glob("*/*/slot.json"))}
    if (receipt.get("slot_hashes") != hashes or receipt.get("logical_slots") != 40 or
            receipt.get("unique_actual") != sum(s.get("unique_run") is True for s in slots.values())):
        raise ValueError("final actual completion receipt differs from current sealed slots")
    return bool(validation.get("all_terminal") is True)


def build_report(run):
    evidence = Evidence(run); run = evidence.run; tables = run / "result_tables"; tables.mkdir(parents=True, exist_ok=True)
    plan = evidence.read("plan.json", {}); identity = evidence.read("source_identity.json", {})
    manifest = evidence.read("dataset/manifest.json", {}); teacher = evidence.read("teacher_search/results.json", {})
    history = evidence.read("historical_import/history.json", {})
    selection = evidence.read("model_selection.json", {}); freeze = evidence.model_freeze()
    validation = evidence.read("validation/protocol_validation.json", {})
    tasks = [r["task_id"] for r in plan.get("tasks", []) if r["split"] == "test"]
    val_tasks = [r["task_id"] for r in plan.get("tasks", []) if r["split"] == "val"]
    source_sha = evidence.inputs.get(str((run / "source_identity.json").resolve()))
    val_endpoints = ("R12", "D250", "D4000", "S250", "S4000")
    val_manifest = evidence.phase("closed_loop_val", bindings={"source_identity_sha256": source_sha},
        tasks=val_tasks, endpoints=val_endpoints)
    phase_manifest = evidence.phase("frozen_test", bindings={"source_identity_sha256": source_sha,
        "model_freeze_sha256": evidence.inputs.get(str((run / "model_freeze.json").resolve()))},
        tasks=tasks, endpoints=ENDPOINTS)
    slots = evidence.slots("frozen_test", phase_manifest); val_slots = evidence.slots("closed_loop_val", val_manifest)
    selection_verified = _verify_selection(evidence, selection, freeze, val_manifest, val_slots)
    streams, test_streams, costs, prefix_rows, first_rows = [], {}, {}, [], []
    for tid in tasks:
        for method in ("R", "N", "S", "D"):
            stream = evidence.stream("test", tid, method, 12 if method == "R" else 8)
            streams.append(stream); test_streams[(tid, method)] = stream
            for endpoint in (("R8", "R12") if method == "R" else (method + "8",)):
                prefix = _prefix(evidence, stream, int(endpoint[1:]))
                costs[(tid, endpoint)] = _endpoint_cost(stream, endpoint, prefix)
            for budget in ((4, 8, 12) if method == "R" else (4, 8)):
                prefix = _prefix(evidence, stream, budget); pool = prefix["rows"]
                first = {"task_id": tid, "method": method, "budget": budget,
                    "prefix_protocol_completed": prefix["protocol_completed"], "slots_observed": len(pool) if pool is not None else None}
                tests = {"first_complete": lambda r: _pred_full(r), "first_B30": lambda r: _pred_b30(r)}
                for pref in PREFERENCES:
                    reference = slots.get((tid, "R12", pref)) if phase_manifest else None
                    tests["first_near_R12_" + pref] = lambda r, ref=reference, p=pref: _pred_near(r, ref, p)
                for name, predicate in tests.items():
                    applicable = not name.startswith("first_near") or _reference(slots.get((tid, "R12", name[-1])), name[-1])
                    hit = _hit(pool, predicate, budget, prefix["protocol_completed"], applicable=applicable)
                    first[name] = hit
                first_rows.append(first)
                prefix_rows.append({"task_id": tid, "method": method, "budget": budget,
                    "prefix_complete": prefix["protocol_completed"], "slots_observed": len(pool) if pool is not None else None,
                    "predicted_full_A": bool(pool is not None and any(_pred_full(r) for r in pool)),
                    "predicted_B30": bool(pool is not None and any(_pred_b30(r) for r in pool)),
                    "near_R12_A": any(_pred_near(r, slots.get((tid, "R12", "A")), "A") is True for r in pool) if pool is not None else False,
                    "near_R12_B": any(_pred_near(r, slots.get((tid, "R12", "B")), "B") is True for r in pool) if pool is not None else False,
                    "R12_A_reference_available": _reference(slots.get((tid, "R12", "A")), "A"),
                    "R12_B_reference_available": _reference(slots.get((tid, "R12", "B")), "B"),
                    "actual_execution_for_this_prefix": budget != 4 and (tid, method + str(budget), "A") in slots})
    per_task, endpoint_rows, near_summary = [], [], {}
    for endpoint in ENDPOINTS:
        near_summary[endpoint] = {}
        for pref in PREFERENCES:
            group = [slots[(t, endpoint, pref)] for t in tasks if (t, endpoint, pref) in slots]
            near_values = [_near(slots.get((t, endpoint, pref)), slots.get((t, "R12", pref)), pref) for t in tasks]
            refs = sum(_reference(slots.get((t, "R12", pref)), pref) for t in tasks); passed = sum(v is True for v in near_values)
            method_quality_na = sum(_reference(slots.get((t, "R12", pref)), pref) and
                _full(slots.get((t, endpoint, pref))) and not _reference(slots.get((t, endpoint, pref)), pref) for t in tasks)
            near_summary[endpoint][pref] = {"preserved": passed == refs if refs and len(group) == 4 and not method_quality_na else None,
                "passed": passed, "qualified_R12_reference_count": refs, "missing_R12_references": 4 - refs,
                "method_quality_missing": method_quality_na}
            raw = _sum_complete(costs.get((t, endpoint), {}).get("raw_rejected") for t in tasks)
            row = {"endpoint": endpoint, "preference": pref, "denominator_tasks": 4,
                "logical_slots_present": len(group), "missing_slots": 4 - len(group),
                "full_27s_five_gates": sum(_full(s) for s in group),
                "actual_B30": sum(_full(s) and s.get("clearance_30mm_met") is True for s in group) if pref == "B" else None,
                "NO_PLAN": sum(s.get("diagnostic_category") == "NO_PLAN" for s in group),
                "raw_rejected_shared_stream": raw, "actual_pre_rejected": sum(s.get("diagnostic_category") in ("ACTUAL_PRECHECK_REJECTED", "ACTUAL_ZERO_STEP_REFUSAL") for s in group),
                "actual_failed": sum(s.get("diagnostic_category") == "ACTUAL_FAILURE" for s in group),
                "tool_error": sum(s.get("diagnostic_category") == "TOOL_ERROR" for s in group),
                "unique_actual_attempts": sum(s.get("unique_run") is True for s in group),
                "unique_positive_step_actual": sum(s.get("unique_run") is True and s.get("actual_steps", 0) > 0 for s in group),
                "actual_aliases": sum(bool(s.get("alias_of_slot")) for s in group),
                "near_R12_count": passed, "R12_qualified_reference_count": refs,
                "near_R12_NA": 4 - refs,
                "near_R12_method_quality_NA": method_quality_na,
                "direct_learning_seed": sum(s.get("source_attribution") == "direct_learning_seed" for s in group),
                "learning_seed_descendant": sum(s.get("source_attribution") == "learning_seed_descendant" for s in group),
                "rule_seed_or_descendant": sum(s.get("source_attribution") == "rule_seed_or_descendant" for s in group)}
            endpoint_rows.append(row)
            for tid in tasks:
                s = slots.get((tid, endpoint, pref)); q = (s or {}).get("quality") or {}
                stream = test_streams[(tid, endpoint[0])]
                selected = ((stream.get("selection") or {}).get("preferences") or {}).get(pref) or {}
                if endpoint == "R8":
                    selected = (((_prefix(evidence, stream, 8).get("selection") or {}).get("preferences")) or {}).get(pref) or {}
                predicted = selected.get("prediction_metrics") or {}
                task_row = {"task_id": tid, "endpoint": endpoint, "preference": pref,
                    "status": (s or {}).get("diagnostic_category", "NOT_RUN"), "full_27s_five_gates": _full(s),
                    "quality_scope": "full_27s" if _full(s) else "full_27s_not_accepted" if q and (s or {}).get("actual_steps") == 13500 else "failed_or_incomplete_prefix" if q else "NOT_MEASURED",
                    "actual_steps": (s or {}).get("actual_steps"), "B30": (s or {}).get("clearance_30mm_met") if pref == "B" else None,
                    "near_R12": _near(s, slots.get((tid, "R12", pref)), pref),
                    "unique_run": (s or {}).get("unique_run"), "alias_of_slot": (s or {}).get("alias_of_slot"),
                    "plan_sha256": (s or {}).get("plan_sha256"), "selected_candidate_id": (s or {}).get("source_candidate_id"),
                    "source_attribution": (s or {}).get("source_attribution"), "selected_origin_source": (s or {}).get("selected_origin_source"),
                    "selected_lineage": (s or {}).get("selected_lineage"), **costs.get((tid, endpoint), {})}
                for metric in METRICS:
                    task_row["actual_" + metric] = q.get(metric)
                    task_row["predicted_" + metric] = predicted.get(metric)
                    task_row["actual_minus_prediction_" + metric] = q[metric] - predicted[metric] if _finite(q.get(metric)) and _finite(predicted.get(metric)) else None
                for gate in GATES:
                    task_row[gate + "_passed"] = (s or {}).get("five_gates", {}).get(gate)
                per_task.append(task_row)
    prefix_aggregate = []
    for method in ("R", "N", "S", "D"):
        for budget in ((4, 8, 12) if method == "R" else (4, 8)):
            group = [r for r in prefix_rows if r["method"] == method and r["budget"] == budget]
            row = {"method": method, "budget": budget, "denominator_tasks": 4,
                "prefix_complete_task_count": sum(r["prefix_complete"] for r in group)}
            for key in ("predicted_full_A", "predicted_B30", "near_R12_A", "near_R12_B", "R12_A_reference_available", "R12_B_reference_available"):
                row[key] = sum(r[key] for r in group)
            prefix_aggregate.append(row)
    comparisons = [_comparison(e, "R12", tasks, slots, costs) for e in ("R8", "N8", "S8", "D8")]
    comparisons += [_comparison("D8", e, tasks, slots, costs) for e in ("R8", "N8", "S8")]
    train_reports = {m: evidence.read(f"models/{m}/training_report.json", {}) for m in ("D", "S")}
    train_statuses = {m: evidence.read(f"models/{m}/training_status.json", {}) for m in ("D", "S")}
    training_config = evidence.read("models/training_config.json", {})
    for row in plan.get("tasks", []):
        tid = row["task_id"]
        if row["split"] == "train":
            for method in ("T_local", "T_transfer"):
                streams.append(evidence.stream("teacher", tid, method, 8))
        elif row["split"] == "val":
            for method in ("R12", "D250", "D4000", "S250", "S4000"):
                streams.append(evidence.stream("val", tid, "R" if method == "R12" else method, 12 if method == "R12" else 8))
    reservations = [evidence.read(p) for p in sorted((run / "budget_ledger").glob("*.json"))]
    reserved = {}
    for r in reservations:
        reserved[r["category"]] = reserved.get(r["category"], 0) + r["count"]
    accounting = {"reservations": reserved, "limits": plan.get("budget_limits", {}), "stages": {},
        "timing_note": "Shared A/B request counted once; summed service time is not makespan; absent measurements remain null.",
        "R8_timing": "only sealed optimizer-prefix timer; no separately measured end-to-end cold total", "data_preparation_elapsed_s": None,
        "stage_makespan_s": None, "outer_process_start_to_finish_s": None}
    for stage in ("teacher", "val", "test"):
        group = [s for s in streams if s["stage"] == stage]
        out = {"streams_expected": {"teacher": 12, "val": 10, "test": 16}[stage],
            "streams_complete": sum(s["protocol_completed"] for s in group),
            "parameter_constructions": _sum_complete(s["parameter_constructions"] for s in group),
            "consumed_candidate_slots": _sum_complete(s["consumed_candidate_slots"] for s in group),
            "prediction_rollouts_started": _sum_complete(s["prediction_rollouts_started"] for s in group),
            "legal_exact_cache_hits": _sum_complete(s["legal_exact_cache_hits"] for s in group),
            "raw_rejected": _sum_complete(s["raw_rejected"] for s in group),
            "nominal_failed_rollouts": _sum_complete(s["nominal_failed_rollouts"] for s in group),
            "nominal_positive_step_incomplete_rollouts": _sum_complete(s["nominal_positive_step_incomplete_rollouts"] for s in group),
            "nominal_status_counts": {status: sum((s["nominal_status_counts"] or {}).get(status, 0) for s in group)
                for status in sorted({status for s in group for status in (s["nominal_status_counts"] or {})})}
                if group and all(s["nominal_status_counts"] is not None for s in group) else None,
            "summed_cold_planning_service_s": _sum_complete(s["end_to_end_cold_planning_s"] for s in group),
            "summed_outer_process_wall_s": _sum_complete(s["outer_process_wall_s"] for s in group),
            "DDIM_samples_generated": sum(s["DDIM_samples_generated"] for s in group),
            "regression_forward_units": sum(s["regression_forward_units"] for s in group)}
        for k in WORK:
            out[k] = _sum_complete(s[k] for s in group)
        actuals = list((val_slots if stage == "val" else slots if stage == "test" else {}).values())
        out.update(actual_logical_slots_present=len(actuals), unique_actual_attempts=sum(s.get("unique_run") is True for s in actuals),
            actual_aliases=sum(bool(s.get("alias_of_slot")) for s in actuals), actual_NO_PLAN=sum(s.get("diagnostic_category") == "NO_PLAN" for s in actuals),
            main_actual_physics_steps=sum(s.get("actual_steps", 0) for s in actuals if s.get("unique_run")),
            summed_unique_actual_service_s=_sum_complete(s.get("elapsed_wall_s") for s in actuals if s.get("unique_run")))
        out["actual_auxiliary_work"] = {k: _sum_complete((s.get("costs") or {}).get(k) for s in actuals if s.get("unique_run"))
            for k in ("private_preview_physics_steps", "native_geometry_query_calls", "independent_saved_torque_replay_steps", "qp_solve_calls")}
        accounting["stages"][stage] = out
    accounting["training"] = {m: {k: train_reports[m].get(k) for k in ("optimizer_updates_total", "sample_exposures_total", "elapsed_s", "effective_parameter_count", "initialization_seed", "paired_reference_indices_sha256", "ddim_sample_units")}
        for m in ("D", "S")}
    accounting["training"]["summed_offline_training_service_s"] = _sum_complete(train_reports[m].get("elapsed_s") for m in ("D", "S"))
    complete_training = {m: train_reports[m].get("status") == "COMPLETED" and train_reports[m].get("optimizer_updates_total") == 4000 for m in ("D", "S")}
    expected_slots = {(t, e, p) for t in tasks for e in ENDPOINTS for p in PREFERENCES}
    terminal_verified = _verify_terminal_accounting(evidence, validation, streams, slots, val_slots, reserved, plan)
    verification = {"complete": bool(freeze and val_manifest and phase_manifest and selection_verified and terminal_verified and not evidence.unverified),
        "actual_slot_and_alias_seals_verified": all(s.get("report_evidence_verified") for s in [*slots.values(), *val_slots.values()]) and len(slots) == 40 and len(val_slots) == 20,
        "phase_selections_verified": bool(val_manifest and phase_manifest), "model_freeze_verified": bool(freeze),
        "closed_loop_selection_verified": selection_verified, "current_terminal_accounting_verified": terminal_verified,
        "missing_or_pending": evidence.unverified, "read_only": True, "no_new_physics_training_or_sampling": True}
    independent = bool(verification["complete"] and len(tasks) == 4 and set(slots) == expected_slots and
        not any(s.get("diagnostic_category") == "TOOL_ERROR" for s in slots.values()))
    d_comparisons = {r["comparator"]: r for r in comparisons if r["method"] == "D8"}
    d_participated = any(_full(s) and s.get("selected_origin_source") == "diffusion" for s in slots.values() if s["endpoint"] == "D8")
    benefit = {}
    for key, comparators in (("rules", ("R8", "R12")), ("retrieval", ("N8",)), ("simple_regression", ("S8",))):
        benefit[key] = bool(independent and d_participated and any(d_comparisons[c]["engineering_10percent_goal_met"] or
            (d_comparisons[c]["coverage_gain_without_lost_comparator_task"] and d_comparisons[c]["near_quality_preserved_against_comparator"] and d_comparisons[c]["R12_near_quality_preserved"])
            for c in comparators))
    d_next = bool(independent and d_participated and d_comparisons["R12"]["capability_preserved_same_tasks"] and
        d_comparisons["R12"]["R12_near_quality_preserved"] and all(
        d_comparisons[c]["capability_preserved_same_tasks"] and d_comparisons[c]["engineering_10percent_goal_met"] for c in ("N8", "S8")))
    statuses = {"research_execution_completed": bool(independent and teacher.get("protocol_completed") is True and all(complete_training.values()) and selection.get("closed_loop_val_completed") is True and freeze),
        "implementation_completed": True, "search_effect_teacher_evidence_available": manifest.get("search_effect_teacher_evidence_available"),
        "training_executed_D": bool(complete_training["D"] or train_statuses["D"].get("training_executed")),
        "training_executed_S": bool(complete_training["S"] or train_statuses["S"].get("training_executed")),
        "training_completed_D": complete_training["D"], "training_completed_S": complete_training["S"],
        "selected_checkpoint_D": selection.get("selected_checkpoint_D"), "selected_checkpoint_S": selection.get("selected_checkpoint_S"),
        "closed_loop_val_completed": selection_verified,
        "independent_test_completed": independent,
        "full_actual_task_by_method_and_preference": {e: {p: {"passed": next(r["full_27s_five_gates"] for r in endpoint_rows if r["endpoint"] == e and r["preference"] == p), "denominator": 4}
            for p in PREFERENCES} for e in ENDPOINTS},
        "near_quality_preserved_by_method": near_summary,
        "learning_benefit_over_rules": benefit["rules"] if independent else "NOT_ESTABLISHED",
        "learning_benefit_over_retrieval": benefit["retrieval"] if independent else "NOT_ESTABLISHED",
        "learning_benefit_over_simple_regression": benefit["simple_regression"] if independent else "NOT_ESTABLISHED",
        "D_next_round_candidate_engineering_criteria_met": d_next,
        "default_initializer_decision": "KEEP_C1_RULE", "deployment": "NOT_MET",
        "continuous_time_safety": "NOT_ESTABLISHED", "hardware_safety": "NOT_ESTABLISHED", "data_status": "DATA_LIMITED"}
    summary = {"schema": "v64_c3_truthful_result_summary_v1", "status": statuses,
        "run": str(run), "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": plan.get("base_publication_commit", "1758e13b01735b80c5a512b81cbc1d5e47a07ec9"),
        "implementation_producer_commit": identity.get("algorithm_producer_commit"),
        "publication_commit": None, "publication_commit_note": "Final publication identity is separate; this report does not relabel the experiment producer.",
        "external_split_counts": {s: {"tasks": sum(r["split"] == s for r in plan.get("tasks", [])),
            "mothers": len({r["mother_id"] for r in plan.get("tasks", []) if r["split"] == s})} for s in ("train", "val", "test")},
        "endpoint_summary": endpoint_rows, "paired_comparisons": comparisons,
        "teacher": {"historical_physical_candidates": history.get("candidate_count_unique", len(history.get("candidates", [])) if history else None),
            "historical_evidence_tiers": {tier: sum(r.get("evidence_tier") == tier for r in history.get("candidates", [])) for tier in
                ("VALIDATED_EXECUTION", "PREDICTED_COMPLETE", "FAILED_OR_INCOMPLETE", "MISSING_OR_UNBOUND")},
            "route_quality_label_count": manifest.get("route_quality_label_count"),
            "initializer_effect_label_count": manifest.get("initializer_effect_label_count"),
            "unique_unified_supervision_labels": manifest.get("label_count_views"), "zero_label_views": manifest.get("zero_label_views"),
            "search_effect_status": manifest.get("search_effect_status"), "slots_consumed": teacher.get("slots_consumed"),
            "pair_count": len(teacher.get("summaries", [])), "selected_pair_ids": teacher.get("selected_teacher_pair_ids"),
            "pair_effects": [{"task_id": r.get("task_id"), "combination": r.get("combination"), "seed_pair_id": r.get("seed_pair_id"),
                "selected_for_supervision": r.get("seed_pair_id") in (teacher.get("selected_teacher_pair_ids") or {}).get(r.get("task_id"), []),
                "endpoints": r.get("endpoints", {})} for r in teacher.get("summaries", [])],
            "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH", "shared_seed_pair_is_independent_counterfactual": False},
        "model_selection": selection, "training": train_reports, "cost_accounting": accounting,
        "TEST_actual_logical_slots": len(slots), "TEST_unique_actual_attempts": sum(s.get("unique_run") is True for s in slots.values()),
        "TEST_actual_aliases": sum(bool(s.get("alias_of_slot")) for s in slots.values()),
        "D_successful_endpoint_with_seed_or_descendant_participation": d_participated,
        "protocol_validation": validation, "evidence_verification": verification,
        "scope": "frozen pilot; 3 TRAIN mothers, 1 VAL mother, 2 TEST mothers, one initialization seed per model; no statistical noninferiority",
        "geometry_scope": "original native-related-pair saved 2ms state scope; missing metrics null/NOT_RUN; no continuous-time upgrade",
        "benefit_rule": "Same-task ability and near-quality first; >=10% measured cold planning cost or prediction work only after preservation; extra raw-rejection saving not learning benefit; lineage is not causal necessity",
        "training_shared_condition_dim": training_config.get("models", {}).get("S", {}).get("condition_dim")}
    stream_export = [{k: v for k, v in s.items() if k not in ("rows", "proposals", "selection", "cost")} for s in streams]
    _csv(tables / "endpoint_summary.csv", endpoint_rows)
    _csv(tables / "per_task_endpoints.csv", per_task)
    _csv(tables / "first_hits.csv", first_rows)
    _csv(tables / "predicted_prefixes.csv", prefix_rows)
    _csv(tables / "predicted_prefix_aggregate.csv", prefix_aggregate)
    _csv(tables / "stream_accounting.csv", stream_export)
    _write(tables / "paired_comparisons.json", comparisons)
    _write(tables / "cost_accounting.json", accounting)
    _write(tables / "status.json", statuses)
    _write(tables / "first_hits.json", first_rows)
    figures = _plots(tables, prefix_aggregate, endpoint_rows, per_task)
    summary["core_figures"] = [p.relative_to(run).as_posix() for p in figures]
    _write(run / "summary.json", summary)
    lines = _markdown(summary, per_task, first_rows, stream_export)
    (run / "REPORT.md").write_text(lines, encoding="utf8")
    exporter = Path(__file__).resolve()
    artifacts = {str(p.relative_to(run).as_posix()): _sha(p) for p in sorted(tables.rglob("*")) if p.is_file() and p.name != "report_artifact_identity.json"}
    _write(tables / "report_artifact_identity.json", {"schema": "v64_c3_report_artifact_identity_v1",
        "reporter_source_path": str(exporter), "reporter_source_sha256": _sha(exporter),
        "input_files": evidence.inputs, "derived_artifacts": artifacts,
        "root_reports": {name: _sha(run / name) for name in ("summary.json", "REPORT.md")},
        "no_new_physics": True, "no_training": True, "no_DDIM": True, "changes_execution_evidence": False})
    return summary


def _markdown(summary, per_task, hits, streams):
    status = summary["status"]; teacher = summary["teacher"]; accounting = summary["cost_accounting"]
    def cell(x, digits=6):
        return "N/A" if x is None else "true" if x is True else "false" if x is False else f"{x:.{digits}g}" if isinstance(x, float) else str(x)
    lines = ["# V6.4-C.3 搜索导向初值、闭环 VAL 与简单回归对照", "",
        f"研究执行完成：**{cell(status['research_execution_completed'])}**；独立 TEST 完成：**{cell(status['independent_test_completed'])}**。",
        f"默认决策：**{status['default_initializer_decision']}**。deployment=NOT_MET；DATA_LIMITED。", "",
        "## 边界、身份与样本范围", "",
        "C.2 维持研究完成、总体学习收益未建立和默认 C.1；B.2/B.3/B.3.1/C.1/C.2 的记录、失败和权重均只读。C.3 是新方法包，跨轮差异不能拆成教师、模型选择或回归对照的独立因果贡献。",
        f"固定基点 `{summary['base_commit']}`；本轮实现及实验 producer `{summary['implementation_producer_commit']}`。发布提交在最终发布记录中单列，不能冒充早期 producer。",
        f"外部分组：`{json.dumps(summary['external_split_counts'], ensure_ascii=False)}`。原 TaskSpec split 未据此重写。新母场景按来源 seed 一次性冻结，失败不重抽。",
        "三个 TRAIN、一个 VAL、两个 TEST 母场景，以及每模型一个初始化 seed，只支持冻结先导；128000 次曝光并非独立参考或母场景数。无统计非劣、连续时间安全、误差鲁棒性或硬件安全结论。", "",
        "## 教师与统一监督池", "",
        f"历史去重物理候选：{cell(teacher['historical_physical_candidates'])}；证据层级 `{json.dumps(teacher['historical_evidence_tiers'], ensure_ascii=False)}`。A/B 视图和 actual alias 不增加物理候选。",
        f"有限教师组合完成 {teacher['pair_count']}/12，消耗槽 {cell(teacher['slots_consumed'])}/96；formal_actual_validation=NOT_RUN_TEACHER_SEARCH。",
        f"统一唯一标签 {cell(teacher['unique_unified_supervision_labels'])}；具有 route_quality 来源 {cell(teacher['route_quality_label_count'])}，具有 initializer_effect 来源 {cell(teacher['initializer_effect_label_count'])}，零残差视图 {cell(teacher['zero_label_views'])}。双来源可能重叠，不能相加冒充样本总量。",
        f"搜索效果证据状态：`{teacher['search_effect_status']}`。新增效果标签保存原送入 slot1/3 的 raw 初值，带固定伙伴和 seed_pair_id；最终最优参数不自动取得初值效果资格。RULE_ONLY、B NO_PLAN 或 B 不足30mm不得成为 B 成功监督。",
        "T_local 与留一母场景 T_transfer 均在教师运行前冻结。D/S/N 共用统一 TRAIN 池；source1:1（同时存在时）及 mother→Task→偏好/family→唯一参考抽样。谱系/直接合格只表示有限共享搜索参与证据，不证明单一初值因果必要。", "",
        "|TRAIN Task|组合|入选监督组合|A合格/近质量/效果标签|B合格30mm/近质量/效果标签|A/B来源状态|", "|---|---|---|---|---|---|"]
    for r in teacher["pair_effects"]:
        a, b = r["endpoints"].get("A", {}), r["endpoints"].get("B", {})
        line = lambda e: "/".join(cell(e.get(k)) for k in ("qualified_endpoint", "near_historical_reference", "positive_supervision"))
        lines.append(f"|{cell(r['task_id'])}|{cell(r['combination'])}|{cell(r['selected_for_supervision'])}|{line(a)}|{line(b)}|{cell(a.get('label_status'))}/{cell(b.get('label_status'))}|")
    lines += ["", "## 两次真实训练与闭环选权重", "",
        "D 直接复用 C.2 的 cosine100、内部 v-prediction MSE、DDIM20；S 直接回归同归一化 z。两者使用相同条件、scaler、12维容器与最多4维非零 mask，均为两层128 SiLU。反归一化后无 clamp/project/补抽。",
        "每模型固定4000 optimizer updates、batch32、AdamW lr1e-4/weight_decay0.01、gradient norm clip1。共享预冻结参考抽样文件，各128000曝光；TRAIN-only scaler。训练不生成 DDIM、不根据 loss 选权重，仅保留250/4000。", "",
        "|模型|真实训练完成|更新|曝光|有效参数|初始化seed|训练秒|闭环选中|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for m in ("D", "S"):
        r = summary["training"][m]
        lines.append(f"|{m}|{cell(status['training_completed_' + m])}|{cell(r.get('optimizer_updates_total'))}|{cell(r.get('sample_exposures_total'))}|{cell(r.get('effective_parameter_count'))}|{cell(r.get('initialization_seed'))}|{cell(r.get('elapsed_s'))}|{cell(status['selected_checkpoint_' + m])}|")
    lines += ["", f"条件维数 {cell(summary['training_shared_condition_dim'])}；D/S 逐参考曝光身份必须相同。checkpoint、scaler 和 schema 身份见模型报告与 model_freeze.json。",
        "闭环 VAL 先完成全部生成/八槽搜索并封存选择，再从初态实际执行与原五门禁；两个 checkpoint 同任务同冻结噪声。R12 只作封存后的评分参考。顺序：完整实际端点→B30→近质量→raw非法较少→同参考集合首次命中（未命中右删失，编码9非真实命中）→物理/预演/几何/冷总时间→较早update。", "",
        "|VAL checkpoint|完整端点/4|B30/2|近质量端点|raw非法|首次近质量预算编码和|选中|", "|---|---:|---:|---:|---:|---:|---|"]
    for key in ("D250", "D4000", "S250", "S4000"):
        r = summary["model_selection"].get("scores", {}).get(key, {})
        chosen = key in (status["selected_checkpoint_D"], status["selected_checkpoint_S"])
        lines.append(f"|{key}|{cell(r.get('full_27s_five_gate_endpoints'))}|{cell(r.get('actual_B30_endpoints'))}|{cell(r.get('near_quality_endpoints'))}|{cell(r.get('raw_illegal_initializers'))}|{cell(r.get('first_near_budget_encoding_sum'))}|{cell(chosen)}|")
    lines += ["", "一个新 VAL 母场景只用于有限权重选择。TEST 在模型、N库、scaler、mask、seed、噪声规则和协议冻结后开始；不据 TEST 重训或改选250。", "",
        "## TEST 实际端点主表", "",
        f"所有分母固定4 Task；当前 TEST logical slots {summary['TEST_actual_logical_slots']}/40，unique actual attempts {summary['TEST_unique_actual_attempts']}，alias {summary['TEST_actual_aliases']}。Alias 是同 Task/模型/配置/历史/plan 的同一证据，不增加独立实际样本。",
        "完整指13500个2ms步、27s及原五门禁全过。N/A不算通过；NO_PLAN保留0步，不等于碰撞或任务不可行。raw拒绝是该方法A/B共享搜索计数，每偏好行重复展示，不能再次相加。", "",
        "|端点|偏好|完整27s+五门禁/4|B30/4|NO_PLAN/4|raw共享拒绝|actual前拒绝|actual失败|工具错误|缺测|unique|alias|近R12/合格参考|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary["endpoint_summary"]:
        lines.append("|" + "|".join(map(str, (r["endpoint"], r["preference"], f"{r['full_27s_five_gates']}/4", cell(r["actual_B30"]),
            f"{r['NO_PLAN']}/4", cell(r["raw_rejected_shared_stream"]), r["actual_pre_rejected"], r["actual_failed"], r["tool_error"], r["missing_slots"],
            r["unique_actual_attempts"], r["actual_aliases"], f"{r['near_R12_count']}/{r['R12_qualified_reference_count']} (N/A {r['near_R12_NA']})"))) + "|")
    lines += ["", "A近质量：I_support≤R12-A+0.001rad/s且L_full≤R12-A+0.005m。B近质量：完整五门禁、d_support≥0.030m且L_full≤R12-B+0.005m；R12无完整实际参考时记N/A。", "",
        "## 逐Task质量、成本和来源", "",
        "完整配对比较只使用双方实际完整端点。失败前缀的质量保留在CSV并标明scope，不能以较短路径或较低干预赢过完整任务。规划时间A/B共用；R8只有封存搜索器前缀计时，其请求冷总时间缺测，不能套用R12总时间。", "",
        "|Task|端点/偏好|状态|I_support rad/s|L_full m|d_support m|基座平移m/转角rad|冷规划s|源/谱系|", "|---|---|---|---:|---:|---:|---|---:|---|"]
    for r in per_task:
        lines.append(f"|{r['task_id']}|{r['endpoint']}/{r['preference']}|{r['status']}|{cell(r['actual_I_support'])}|{cell(r['actual_L_full'])}|{cell(r['actual_d_support'])}|{cell(r['actual_base_translation_peak_m'])}/{cell(r['actual_base_rotation_peak_rad'])}|{cell(r['end_to_end_cold_planning_s'])}|{cell(r['selected_origin_source'])}/{cell(r['source_attribution'])}|")
    lines += ["", "prediction→actual逐指标差值、五门禁、候选ID、plan SHA、源与完整谱系见 `result_tables/per_task_endpoints.csv`。谱系不等于因果必要；成功归因于初始化方式+同一搜索器+原控制器，不能称raw端到端策略成功。", "",
        "## 首次命中和预测预算曲线", "",
        "`first_hits.csv/json` 保留每Task每方法4/8/12前缀的首次完整、B30、A/B近R12命中。HIT为消耗评价槽位置；RIGHT_CENSORED不填0，sort encoding预算+1仅排序。NOT_RUN、TECHNICAL_INCOMPLETE及N/A分开。R8在R9前封存，4槽从未自动视作实际通过。", "",
        "|Task|方法/预算|首次完整|首次B30|首次近R12-A|首次近R12-B|", "|---|---|---|---|---|---|"]
    def h(value):
        return str(value["slot"]) if value["status"] == "HIT" else value["status"] + (f"@{value.get('right_censored_budget')}" if value["status"] == "RIGHT_CENSORED" else "")
    for r in hits:
        lines.append(f"|{r['task_id']}|{r['method']}/{r['budget']}|{h(r['first_complete'])}|{h(r['first_B30'])}|{h(r['first_near_R12_A'])}|{h(r['first_near_R12_B'])}|")
    lines += ["", "## 工作量与离线成本", "",
        "预留上限包括拒绝与NO_PLAN，预留不等于已消耗候选或真实rollout。三种计数各自保留：参数构造、消耗槽、真实预测；合法精确重复由原缓存处理。", "",
        "|阶段|完整搜索流/预计|参数构造|消耗槽|真实rollout|raw拒绝|合法cache hit|主预测步|预演步|几何查询|冷规划service秒|实际logical/unique/alias|主actual步|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for stage, r in accounting["stages"].items():
        lines.append(f"|{stage}|{r['streams_complete']}/{r['streams_expected']}|{cell(r['parameter_constructions'])}|{cell(r['consumed_candidate_slots'])}|{cell(r['prediction_rollouts_started'])}|{cell(r['raw_rejected'])}|{cell(r['legal_exact_cache_hits'])}|{cell(r['prediction_physics_steps'])}|{cell(r['private_preview_physics_steps'])}|{cell(r['native_geometry_query_calls'])}|{cell(r['summed_cold_planning_service_s'])}|{r['actual_logical_slots_present']}/{r['unique_actual_attempts']}/{r['actual_aliases']}|{r['main_actual_physics_steps']}|")
    lines += ["", f"预算预留 `{json.dumps(accounting['reservations'], ensure_ascii=False, sort_keys=True)}`；上限328候选槽、60最终actual逻辑槽、32 DDIM样本。主预测理论上限4,428,000步，主actual上限810,000步；预演、独立力矩重放和几何查询额外分账。",
        f"两个训练离线service合计 {cell(accounting['training']['summed_offline_training_service_s'])}秒。教师/VAL搜索与VAL实际成本均为离线；数据整理计时、外层进程墙钟和阶段makespan未测则null，不将累计worker service冒充makespan。仅测cold，不推算warm通过。",
        "完整分账及每流失败、原预算/缓存计数见 `cost_accounting.json` 与 `stream_accounting.csv`。非法提案或提前失败减少物理步不自动构成更优初值；实际能力和质量未保持时，不作成本收益或摊销肯定结论。", "",
        "逐流 nominal_status_counts、nominal_failed_rollouts 和 nominal_positive_step_incomplete_rollouts 单列名义预检查拒绝、执行拒绝、未完成预测和工具错误；缺失搜索流保留null。较少物理步可能来自提前拒绝或短失败轨迹，不能直接归因于更好的初值。", "",
        "## 两条比较线与默认决策", "",
        "同8槽比较R8/N8/S8/D8；缩预算比较N8/S8/D8对R12。8比12少33.3%只描述配额。下面先检查同Task能力、质量和B30，再比较全部四Task的实测总成本；不只挑成功/最快任务。额外raw拒绝造成的节省不满足学习收益。", "",
        "|比较|相同Task能力保持|相对比较器近质量|R12近质量保持|冷总成本减少比例|有效预测步减少比例|10%工程目标|", "|---|---|---|---|---:|---:|---|"]
    for r in summary["paired_comparisons"]:
        costs = r["cost_reductions_all_four_tasks_including_failed_requests"]
        lines.append(f"|{r['method']} vs {r['comparator']}|{cell(r['capability_preserved_same_tasks'])}|{cell(r['near_quality_preserved_against_comparator'])}|{cell(r['R12_near_quality_preserved'])}|{cell(costs['end_to_end_cold_planning_s']['reduction_fraction'])}|{cell(costs['prediction_physics_steps']['reduction_fraction'])}|{cell(r['engineering_10percent_goal_met'])}|")
    lines += ["", f"学习收益：对规则 `{status['learning_benefit_over_rules']}`；对检索 `{status['learning_benefit_over_retrieval']}`；对简单回归 `{status['learning_benefit_over_simple_regression']}`。下一轮D候选工程条件满足 `{status['D_next_round_candidate_engineering_criteria_met']}`。",
        "该判据要求D有实际成功端点的种子/后代参与证据；仅共同规则成功不称神经收益。若D只优于R8，未优于N/S，只能说明相对规则局部改善。D与便宜方法相等但更贵时保留简单方法。当前默认仍KEEP_C1_RULE，任何候选建议均不自动用于硬件。", "",
        "## 缺测、失败与完整状态", "",
        "工具/管线错误、真实NO_PLAN、预演或actual前拒绝、actual失败、未运行分列。未消费拒绝与非有限raw不修复；失败不追加候选、不换计划、不重抽、不拼接actual。当前安全几何范围不升级，缺测保持null/NOT_RUN。", "", "```json", json.dumps(status, indent=2, ensure_ascii=False), "```", "",
        "报告只读复核原actual及alias封存、VAL/TEST选择源、模型/数据冻结身份，以及当前候选/actual/预算计数与终态验证收据。已有封存哈希冲突会拒绝生成报告；尚未形成的封存或收据保持未完成。完整actual但缺少必要质量指标仍计入能力保持集合，近质量记N/A并阻止收益判定。", "",
        "```json", json.dumps(summary["evidence_verification"], indent=2, ensure_ascii=False), "```", "",
        "## 可核验产物", "",
        "`plan.json`、`source_identity.json`、`learning_split_manifest.json`、`teacher_search/`、`dataset/`、`models/`、`closed_loop_val/`、`model_selection.json`、`model_freeze.json`、`test_search/`、`frozen_test/actual/`、`validation/`。报告输入哈希、生成器源码哈希和派生表/图身份见 `result_tables/report_artifact_identity.json`。", ""]
    for figure in summary["core_figures"]:
        lines += [f"![C.3 core result figure]({figure})", ""]
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    args = parser.parse_args(); result = build_report(args.run)
    print(json.dumps({"status": result["status"], "run": result["run"], "core_figures": result["core_figures"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
