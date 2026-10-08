"""C.3 frozen closed-loop VAL and reusable final actual execution.

Search and generation belong to the caller. This module first seals *all*
selections, then executes the original feedback runner and its five independent
checks. The scorer only opens the two explicitly frozen VAL tasks. No physics
occurs on import, during sealing, or during checkpoint scoring.

Entry contract: task_id, endpoint, selection_path, candidate_registry_path,
proposals_path, planning_cost_path. The last three paths can be omitted for
mock execution; they are mandatory for formal VAL scoring. Costs must contain
end_to_end_cold_planning_s. Candidate registries are original consumed-slot
arrays, never proposal arrays. All source files and checkpoint bytes are sealed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

CHECKPOINTS = {"D250": ("D", 250), "D4000": ("D", 4000),
               "S250": ("S", 250), "S4000": ("S", 4000)}
VAL_ENDPOINTS = ("R12", "D250", "D4000", "S250", "S4000")
FIVE_GATES = ("task_requirements", "execution_contract", "independent_interval",
              "native_geometry", "reference_binding")
COST_ORDER = ("prediction_physics_steps", "private_preview_physics_steps",
              "native_geometry_query_calls", "end_to_end_cold_planning_s")
COST_ORDER_TEXT = "lexicographic: prediction physics steps, private preview physics steps, native geometry query calls, measured end-to-end cold planning seconds"
INITIAL_HISTORY_POLICY = "fresh runner/model/MjData/controller/reference provider/integrator/QP and ADMM cache; no previous candidate or stream state"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    # Match the original task_protocol.canonical_json byte-for-byte, including
    # ASCII escaping, when checking a frozen search registry content digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf8")).hexdigest()


def _seal(directory):
    directory = Path(directory)
    _write(directory / "manifest.json", {p.relative_to(directory).as_posix(): _sha(p)
        for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"})


def _verify_seal(directory):
    directory = Path(directory).resolve()
    for relative, expected in _read(directory / "manifest.json").items():
        path = (directory / relative).resolve()
        if directory not in path.parents or _sha(path) != expected:
            raise ValueError("actual evidence changed: " + str(path))


def verify_actual_slot(directory):
    """Verify both local evidence and a strict alias's original evidence."""
    directory = Path(directory).resolve(); _verify_seal(directory)
    slot = _read(directory / "slot.json")
    if slot.get("alias_of_slot"):
        source = Path(slot["alias_of_slot"])
        if _sha(source) != slot["alias_of_slot_sha256"]:
            raise ValueError("aliased original actual slot changed")
        _verify_seal(source.parent); original = _read(source)
        if original.get("alias_of_slot") or (original["task_sha256"], original["plan_sha256"], original["alias_identity"]) != (slot["task_sha256"], slot["plan_sha256"], slot["alias_identity"]):
            raise ValueError("actual alias Task/plan/config/history identity differs")
    return slot


def freeze_phase_selections(phase_root, entries, *, phase="VAL", bindings=None,
                            checkpoint_files=None, expected_endpoints=None):
    """Seal the whole phase before its first actual; never seal piecemeal.

    VAL has exactly two tasks and four checkpoint candidates. TEST callers must
    declare the expected endpoint set. Bindings should include the root protocol,
    source and (for TEST) frozen model-selection identities.
    """
    root = Path(phase_root).resolve(); bindings = bindings or {}
    target = root / "sealed_selections" / "all_selections.json"
    if target.exists():
        retained = verify_phase_selections(root, bindings=bindings)
        requested = sorted((e["task_id"], e["endpoint"], str(Path(e["selection_path"]).resolve())) for e in entries)
        saved = sorted((e["task_id"], e["endpoint"], e["selection_source_path"]) for e in retained["entries"])
        if requested != saved:
            raise ValueError("retained phase selection schedule differs")
        if checkpoint_files is not None and {k: str(Path(v).resolve()) for k, v in checkpoint_files.items()} != retained["checkpoint_files"]:
            raise ValueError("retained VAL checkpoint whitelist differs")
        return retained
    if phase not in ("VAL", "TEST"):
        raise ValueError("phase must be VAL or TEST")
    if (root / "actual").exists():
        raise ValueError("all phase selections must precede every actual")
    entries = list(entries); tasks = sorted({e["task_id"] for e in entries})
    endpoints = tuple(expected_endpoints or VAL_ENDPOINTS if phase == "VAL" else expected_endpoints or ())
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("explicit unique expected endpoints required")
    if phase == "VAL" and (len(tasks) != 2 or set(endpoints) != set(VAL_ENDPOINTS)):
        raise ValueError("formal VAL requires exactly two tasks and the four checkpoint whitelist plus R12")
    if {(e["task_id"], e["endpoint"]) for e in entries} != {(t, e) for t in tasks for e in endpoints} or len(entries) != len(tasks) * len(endpoints):
        raise ValueError("incomplete or duplicate phase selection schedule")
    checkpoints = {k: str(Path(v).resolve()) for k, v in (checkpoint_files or {}).items()}
    if phase == "VAL" and set(checkpoints) != set(CHECKPOINTS):
        raise ValueError("VAL accepts only D250 D4000 S250 S4000")
    files = {path: _sha(path) for path in checkpoints.values()}; frozen_entries = []
    for entry in entries:
        task_id, endpoint = entry["task_id"], entry["endpoint"]
        if any(Path(value).name != value or value in (".", "..") for value in (task_id, endpoint)):
            raise ValueError("task/endpoint must be a single path component")
        source = Path(entry["selection_path"]).resolve(); selection = _read(source)
        if selection.get("task_id") != task_id or set(selection.get("preferences", {})) != {"A", "B"}:
            raise ValueError("selection Task/preference binding differs")
        if selection.get("selection_reads_final_actual") is not False:
            raise ValueError("selection must declare it did not read final actual")
        copy = target.parent / task_id / (endpoint + ".json")
        if copy.exists():
            if _digest(_read(copy)) != _digest(selection):
                raise ValueError("partially sealed selection changed")
        else:
            _write(copy, selection)
        saved = {"task_id": task_id, "endpoint": endpoint,
                 "task_sha256": selection["task_sha256"],
                 "selection_source_path": str(source), "selection_path": str(copy),
                 "selection_sha256": _sha(copy)}
        for key in ("candidate_registry_path", "proposals_path", "planning_cost_path"):
            if entry.get(key):
                path = Path(entry[key]).resolve(); files[str(path)] = _sha(path); saved[key] = str(path)
        files[str(source)] = _sha(source); files[str(copy)] = _sha(copy)
        if endpoint in CHECKPOINTS:
            saved["checkpoint_sha256"] = _sha(checkpoints[endpoint])
        frozen_entries.append(saved)
    value = {"schema": "v64_c3_all_phase_selections_sealed_v1", "phase": phase,
             "sealed_utc": datetime.now(timezone.utc).isoformat(), "task_ids": tasks,
             "endpoints": list(endpoints), "entries": frozen_entries, "files": files,
             "checkpoint_files": checkpoints, "bindings": bindings,
             "logical_actual_slots": len(entries) * 2, "selection_reads_actual": False,
             "actual_started": False, "cost_order": list(COST_ORDER)}
    _write(target, value)
    return value


def verify_phase_selections(phase_root, *, bindings=None):
    value = _read(Path(phase_root) / "sealed_selections" / "all_selections.json")
    if bindings is not None and value["bindings"] != bindings:
        raise ValueError("phase protocol/model binding changed")
    for path, expected in value["files"].items():
        if _sha(path) != expected:
            raise ValueError("sealed phase file changed: " + path)
    return value


def cold_initial_history(task):
    """Explicit reset history identity, beyond the physical qpos/qvel alone."""
    values = {"policy": INITIAL_HISTORY_POLICY}
    for key in ("initial_qpos", "initial_qvel", "initial_planner_q", "initial_planner_dq"):
        value = getattr(task, key)
        values[key] = value.tolist() if hasattr(value, "tolist") else list(value)
    return values


def actual_alias_identity(task, plan, execution_run, *, initial_history_identity=None):
    run = Path(execution_run)
    return _digest({"task_sha256": task.sha256(), "model_contract_sha256": task.model_contract_sha256,
                    "plan_sha256": plan.sha256(),
                    "initial_history": initial_history_identity or cold_initial_history(task),
                    "source_identity_sha256": _sha(run / "source_identity.json"),
                    "config_sha256": _sha(run / "frozen_execution_config.json"),
                    "run_config_sha256": _sha(run / "frozen_run_config.json")})


def five_gates_passed(evaluation):
    return bool(evaluation.get("evidence_valid") is True and all(
        evaluation.get(key, {}).get("passed") is True for key in FIVE_GATES))


def _diagnostic_category(result):
    failure = result.get("execution_failure") or {}
    if result.get("pipeline_failure") or (failure and failure.get("type") != "UncertifiedExecutionError"):
        return "TOOL_ERROR"
    if result.get("status") == "REFERENCE_PRECHECK_REJECTED":
        return "ACTUAL_PRECHECK_REJECTED"
    if result.get("full_task_success") and result.get("actual_steps") == 13500 and five_gates_passed(result.get("evaluation") or {}):
        return "FULL_TASK_AND_FIVE_GATES_PASSED"
    if not result.get("actual_steps"):
        return "ACTUAL_ZERO_STEP_REFUSAL"
    return "ACTUAL_FAILURE"


def _run_original(task, plan, directory, *, execution_run, frozen, slot_id, quality_reader=None):
    """One new feedback run; all numerical/control/safety code stays original."""
    from .conditional_execution import ExecutionCostLedger
    from .residual_execution import execute_residual_attempt
    from .route_candidate_evaluator import fresh_quality
    run = Path(execution_run); ledger = ExecutionCostLedger(); before = time.perf_counter()
    result = {}; quality = None
    try:
        with ledger.installed():
            result = execute_residual_attempt(task, plan, directory / "attempt",
                qp_config_path=run / "frozen_execution_config.json", identity_path=run / "source_identity.json",
                slot_id=slot_id, execution_diagnostics=True, diagnostic_obstacle_name=frozen["obstacle_name"])
            replay = directory / "attempt" / "actual" / "evaluation" / "fresh_replay.npz"
            if quality_reader is not None:
                quality = quality_reader(task, plan, result, frozen, directory)
            elif result.get("trace_path") and result.get("actual_steps") and replay.exists():
                with ledger.scope("actual_quality"):
                    quality, _ = fresh_quality(task, plan, result["trace_path"], frozen,
                        directory / "quality", evidence_root=directory / "attempt" / "actual", replay_path=replay)
    except Exception as error:
        # Preserve already-consumed work if a later evidence/quality tool fails.
        # No replay, alternative candidate, or second feedback execution follows.
        steps = result.get("actual_steps", ledger.physics_steps("actual"))
        result.update(status="PIPELINE_FAILURE", actual_steps=steps, entered_actual=steps > 0,
            actual_runner_started=result.get("actual_runner_started", (directory / "attempt" / "actual").exists()),
            pipeline_failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    return {**result, "quality": quality, "costs": ledger.to_dict(),
            "elapsed_wall_s": time.perf_counter() - before}


def execute_frozen_task(phase_root, task, frozen, *, execution_run, endpoints=None,
                        initial_history_identity=None, executor=None, quality_reader=None, plan_loader=None):
    """Execute one task's frozen endpoints, with strict same-task actual aliases.

    Mock executor signature: (task, plan, directory, *, execution_run, frozen,
    slot_id, quality_reader). It returns the original attempt fields plus quality
    and costs. No incomplete attempt is retried; a saved TOOL_ERROR blocks resume.
    """
    root = Path(phase_root).resolve(); manifest = verify_phase_selections(root)
    expected = [e for e in manifest["entries"] if e["task_id"] == task.task_id]
    if not expected or any(e["task_sha256"] != task.sha256() for e in expected):
        raise ValueError("actual Task differs from frozen phase Task")
    selected_endpoints = list(endpoints or manifest["endpoints"])
    if len(set(selected_endpoints)) != len(selected_endpoints) or not set(selected_endpoints) <= {e["endpoint"] for e in expected}:
        raise ValueError("unknown or repeated endpoint")
    if plan_loader is None:
        from .task_anchored_reference import TaskAnchoredResidualPlan
        plan_loader = TaskAnchoredResidualPlan.from_dict
    runner = executor or _run_original; aliases = {}; slots = []
    history = initial_history_identity or cold_initial_history(task)
    for endpoint in selected_endpoints:
        entry = next(e for e in expected if e["endpoint"] == endpoint)
        selection = _read(entry["selection_path"])
        for preference in ("A", "B"):
            chosen = selection["preferences"][preference]; method = endpoint + "_" + preference
            directory = root / "actual" / task.task_id / method
            plan = plan_loader(chosen["selected_plan"]) if chosen.get("selected_plan") else None
            identity = actual_alias_identity(task, plan, execution_run, initial_history_identity=history) if plan else None
            binding = {"task_sha256": task.sha256(), "selection_sha256": entry["selection_sha256"],
                       "plan_sha256": plan.sha256() if plan else None, "initial_history_sha256": _digest(history),
                       "alias_identity": identity}
            if directory.exists():
                if not (directory / "slot.json").exists():
                    raise RuntimeError("unfinished consumed actual retained; never retry: " + str(directory))
                slot = verify_actual_slot(directory)
                if slot["actual_binding"] != binding:
                    raise ValueError("retained actual Task/plan/config/history/selection differs")
                if slot["diagnostic_category"] == "TOOL_ERROR":
                    raise RuntimeError("retained actual tool error; never retry: " + str(directory))
                if identity and not slot.get("alias_of_slot"):
                    aliases[identity] = slot
                slots.append(slot); continue
            directory.mkdir(parents=True)
            slot = {"schema": "v64_c3_actual_logical_slot_v1", "phase": manifest["phase"],
                    "task_id": task.task_id, "endpoint": endpoint, "preference": preference, "method": method,
                    "logical_slot": len(slots), "actual_binding": binding, "alias_identity": identity,
                    "selection_sha256": entry["selection_sha256"], "task_sha256": task.sha256(),
                    "plan_sha256": plan.sha256() if plan else None, "slot_path": str(directory / "slot.json"),
                    "source_candidate_id": chosen.get("source_candidate_id"),
                    "source_attribution": chosen.get("source_attribution"),
                    "selected_origin_source": chosen.get("selected_origin_source"),
                    "selected_lineage": chosen.get("selected_lineage"),
                    "unique_run": False, "entered_actual": False, "actual_steps": 0,
                    "status": "NO_PLAN", "diagnostic_category": "NO_PLAN", "full_task_success": False,
                    "full_27s_success": False, "original_independent_gates_passed": False,
                    "five_gates": {k: None for k in FIVE_GATES}, "clearance_30mm_met": None,
                    "quality": None, "actual_quality": None, "costs": {"no_actual_work": True},
                    "prediction_actual_difference": None, "prediction_actual_consistent": None,
                    "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED"}
            if plan is not None:
                _write(directory / "selected_plan.json", plan.to_dict())
                if identity in aliases:
                    source = aliases[identity]
                    for key in ("status", "diagnostic_category", "entered_actual", "actual_steps", "full_task_success",
                                "full_27s_success", "original_independent_gates_passed", "five_gates", "clearance_30mm_met",
                                "quality", "actual_quality", "independent_evaluation_path", "trace_path"):
                        slot[key] = source.get(key)
                    slot["alias_of_slot"] = source["slot_path"]; slot["alias_of_slot_sha256"] = _sha(source["slot_path"])
                    slot["costs"] = {"alias_zero_new_work": True}; slot["elapsed_wall_s"] = 0.
                else:
                    try:
                        result = runner(task, plan, directory, execution_run=Path(execution_run), frozen=frozen,
                                        slot_id=manifest["phase"] + "_" + task.task_id + "_" + method,
                                        quality_reader=quality_reader)
                        evaluation = result.get("evaluation") or {}; gates = five_gates_passed(evaluation)
                        category = _diagnostic_category(result)
                        full = bool(category == "FULL_TASK_AND_FIVE_GATES_PASSED")
                        quality = result.get("quality")
                        slot.update(status=result.get("status", "PIPELINE_FAILURE"), diagnostic_category=category,
                            unique_run=bool(result.get("actual_runner_started")), entered_actual=bool(result.get("entered_actual")),
                            actual_steps=int(result.get("actual_steps", 0)), full_task_success=full, full_27s_success=full,
                            original_independent_gates_passed=gates,
                            five_gates={k: evaluation.get(k, {}).get("passed") for k in FIVE_GATES},
                            quality=quality, actual_quality=quality, costs=result.get("costs"),
                            elapsed_wall_s=result.get("elapsed_wall_s"), trace_path=result.get("trace_path"),
                            independent_evaluation_path=result.get("evaluation_path"),
                            clearance_30mm_met=bool(quality.get("d_support") is not None and quality["d_support"] >= .030) if quality else None)
                        if category == "TOOL_ERROR":
                            slot["tool_error"] = result.get("pipeline_failure") or result.get("execution_failure") or {"message": "actual pipeline failed"}
                    except Exception as error:
                        slot.update(status="PIPELINE_FAILURE", diagnostic_category="TOOL_ERROR", tool_error={
                            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
                    aliases[identity] = slot
                predicted = chosen.get("prediction_metrics") or {}
                if predicted and slot["quality"]:
                    differences = {k: slot["quality"][k] - predicted[k] for k in ("I_support", "L_full", "d_support")
                                   if slot["quality"].get(k) is not None and predicted.get(k) is not None}
                    slot["prediction_actual_difference"] = differences
                    slot["prediction_actual_consistent"] = bool(len(differences) == 3 and slot["full_task_success"] and
                        slot["original_independent_gates_passed"] and all(abs(v) <= 1e-9 for v in differences.values()))
            _write(directory / "slot.json", slot); _seal(directory); slots.append(slot)
            print(json.dumps({"event": "C3_FINAL_ACTUAL", "phase": manifest["phase"], "task": task.task_id,
                "endpoint": endpoint, "preference": preference, "category": slot["diagnostic_category"],
                "steps": slot["actual_steps"], "unique": slot["unique_run"]}), flush=True)
            if slot.get("tool_error"):
                raise RuntimeError("consumed actual tool error; no retry: " + str(directory))
    return slots


def actual_near_quality(method, baseline, preference):
    """Missing R12 actual reference is N/A; method failure is false, never N/A."""
    if not _qualified_reference(baseline, preference):
        return None
    if not (method.get("full_task_success") and method.get("original_independent_gates_passed") and method.get("quality")):
        return False
    a, b = method["quality"], baseline["quality"]
    keys = ("L_full", "I_support") if preference == "A" else ("L_full", "d_support")
    if any(a.get(k) is None or not math.isfinite(a[k]) for k in keys) or b.get("L_full") is None:
        return False
    return bool(a["L_full"] <= b["L_full"] + .005 and
        (a["I_support"] <= b["I_support"] + .001 if preference == "A" else a["d_support"] >= .030))


def _qualified_reference(baseline, preference):
    quality = baseline.get("quality") or {}
    required = ("I_support", "L_full") if preference == "A" else ("L_full", "d_support")
    return bool(baseline.get("full_task_success") and baseline.get("original_independent_gates_passed") and
                all(quality.get(k) is not None and math.isfinite(quality[k]) for k in required))


def first_near_quality(rows, baseline, preference, *, budget=8):
    if not _qualified_reference(baseline, preference):
        return {"status": "N/A_R12_NO_QUALIFIED_ACTUAL", "slot": None, "sort_encoding": None}
    reference = baseline["quality"]
    for index, row in enumerate(rows[:budget], 1):
        m = row.get("prediction_metrics") or {}
        required = ("I_support", "L_full") if preference == "A" else ("d_support", "L_full")
        if row.get("prediction_admissible") is not True or any(m.get(k) is None or not math.isfinite(m[k]) for k in required):
            continue
        if m["L_full"] <= reference["L_full"] + .005 and (m["I_support"] <= reference["I_support"] + .001 if preference == "A" else m["d_support"] >= .030):
            return {"status": "HIT", "slot": index, "sort_encoding": index}
    return {"status": "RIGHT_CENSORED", "slot": None, "right_censored_budget": budget,
            "sort_encoding": budget + 1, "encoding_is_observed_hit": False}


def _finite_cost(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("missing/nonfinite/negative formal VAL cost: " + label)
    return value


def score_checkpoints(phase_root):
    """Apply the predeclared exact lexicographic order; never read TEST files."""
    root = Path(phase_root).resolve(); manifest = verify_phase_selections(root)
    if manifest["phase"] != "VAL" or len(manifest["task_ids"]) != 2 or set(manifest["endpoints"]) != set(VAL_ENDPOINTS):
        raise ValueError("checkpoint selection may read only the complete whitelist VAL phase")
    target = root / "model_selection.json"
    if target.exists():
        result = _read(target)
        if result["phase_manifest_sha256"] != _sha(root / "sealed_selections" / "all_selections.json"):
            raise ValueError("VAL selection manifest changed")
        for path, expected in result["actual_slot_files"].items():
            if _sha(path) != expected:
                raise ValueError("scored VAL actual evidence changed")
            verify_actual_slot(Path(path).parent)
        return result
    slots = {}; slot_files = {}
    for task in manifest["task_ids"]:
        for endpoint in VAL_ENDPOINTS:
            for preference in ("A", "B"):
                directory = root / "actual" / task / (endpoint + "_" + preference)
                path = directory / "slot.json"; slot = verify_actual_slot(directory)
                if (slot["task_id"], slot["endpoint"], slot["preference"]) != (task, endpoint, preference):
                    raise ValueError("VAL actual slot binding differs")
                if slot["diagnostic_category"] == "TOOL_ERROR":
                    raise ValueError("incomplete VAL due to tool error cannot select weights")
                slots[(task, endpoint, preference)] = slot; slot_files[str(path)] = _sha(path)
    qualified = [(task, preference) for task in manifest["task_ids"] for preference in ("A", "B")
                 if actual_near_quality(slots[(task, "R12", preference)], slots[(task, "R12", preference)], preference) is not None]
    scores = {}
    for endpoint, (model, update) in CHECKPOINTS.items():
        actual = [slots[(t, endpoint, p)] for t in manifest["task_ids"] for p in ("A", "B")]
        full = sum(s["full_task_success"] and s["original_independent_gates_passed"] and s["actual_steps"] == 13500 for s in actual)
        b30 = sum(s["preference"] == "B" and s["full_task_success"] and s["original_independent_gates_passed"] and s["clearance_30mm_met"] is True for s in actual)
        near = [{"task_id": t, "preference": p, "near_quality": actual_near_quality(slots[(t, endpoint, p)], slots[(t, "R12", p)], p)}
                for t in manifest["task_ids"] for p in ("A", "B")]
        raw_illegal = 0; first = []; workload = {k: 0 for k in COST_ORDER}
        for task in manifest["task_ids"]:
            entry = next(e for e in manifest["entries"] if e["task_id"] == task and e["endpoint"] == endpoint)
            if not all(entry.get(k) for k in ("candidate_registry_path", "proposals_path", "planning_cost_path")):
                raise ValueError("formal VAL score requires sealed registry/proposals/planning cost")
            selection = _read(entry["selection_path"]); rows = _read(entry["candidate_registry_path"])
            if selection["budget"].get("stop_reason") == "TOOL_ERROR" or selection["budget"]["candidate_budget"] != 8:
                raise ValueError("formal model VAL must be an unmodified eight-slot search without tool errors")
            consumed = selection["budget"]["slots_consumed"]
            rows = rows[:consumed]
            if consumed > 8 or len(rows) != consumed or _digest(rows) != selection.get("registry_content_sha256"):
                raise ValueError("VAL registry exceeds or lacks consumed slots")
            raw_illegal += sum(r.get("status") == "INITIALIZER_RAW_REJECTED" for r in rows)
            for preference in ("A", "B"):
                first.append({"task_id": task, "preference": preference,
                              **first_near_quality(rows, slots[(task, "R12", preference)], preference)})
            for key in COST_ORDER[:-1]:
                workload[key] += sum(_finite_cost((r.get("costs") or {}).get(key,
                    0 if r.get("status") == "INITIALIZER_RAW_REJECTED" else None), key) for r in rows)
            cost = _read(entry["planning_cost_path"])
            workload[COST_ORDER[-1]] += _finite_cost(cost.get(COST_ORDER[-1]), COST_ORDER[-1])
        first_sum = sum(row["sort_encoding"] for row in first if row["sort_encoding"] is not None)
        key = [-full, -b30, -sum(r["near_quality"] is True for r in near), raw_illegal, first_sum,
               *[workload[k] for k in COST_ORDER], update]
        scores[endpoint] = {"model": model, "update": update, "full_27s_five_gate_endpoints": full,
            "actual_B30_endpoints": b30, "near_quality_endpoints": sum(r["near_quality"] is True for r in near),
            "near_quality_by_endpoint": near, "raw_illegal_initializers": raw_illegal,
            "first_near_quality": first, "first_near_budget_encoding_sum": first_sum,
            "workload": workload, "lexicographic_min_key": key,
            "checkpoint_sha256": _sha(manifest["checkpoint_files"][endpoint])}
    selected = {model: min((e for e in CHECKPOINTS if CHECKPOINTS[e][0] == model),
                           key=lambda e: scores[e]["lexicographic_min_key"]) for model in ("D", "S")}
    result = {"schema": "v64_c3_closed_loop_checkpoint_selection_v1", "phase": "VAL",
              "phase_manifest_sha256": _sha(root / "sealed_selections" / "all_selections.json"),
              "actual_slot_files": slot_files, "task_ids": manifest["task_ids"],
              "checkpoint_whitelist": list(CHECKPOINTS), "scores": scores,
              "selected_checkpoint_D": selected["D"], "selected_checkpoint_S": selected["S"],
              "selected_checkpoint_files": {m: manifest["checkpoint_files"][e] for m, e in selected.items()},
              "selected_checkpoint_sha256": {m: scores[e]["checkpoint_sha256"] for m, e in selected.items()},
              "R12_qualified_reference_set": [{"task_id": t, "preference": p} for t, p in qualified],
              "first_hit_encoding": "1..8 observed HIT; 9 budget-without-hit encoding, right-censored, never observed slot9",
              "cost_order": list(COST_ORDER), "test_read": False, "closed_loop_val_completed": True,
              "deployment": "NOT_MET", "scope": "one VAL mother, finite model selection; no generalization claim"}
    _write(target, result)
    return result
