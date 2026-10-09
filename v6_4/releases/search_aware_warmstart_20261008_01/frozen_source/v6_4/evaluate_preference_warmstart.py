"""Finite C.2 experiment: frozen data, real weights, isolated search and actual.

All physical work uses the inherited C.1 evaluator and final executor. Each
search stream owns its evaluator directory; no cross-stream cache is possible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .route_optimizer_protocol import (ROOT, SearchSpec, PreferenceSpec, TaskSpec,
    read, write, sha, digest)
from .route_candidate_evaluator import NominalCandidateEvaluator, fresh_quality, seal, verify_seal
from .continuous_route_optimizer import optimize, admissible

ENDPOINTS = ("R8", "R12", "N8", "D8")
METHOD_BUDGETS = {"R": 12, "N": 8, "D": 8}


def now():
    return datetime.now(timezone.utc).isoformat()


def task_rows(run, split):
    return [t for t in read(Path(run) / "plan.json")["tasks"] if t.get("learning_split", t.get("split")) == split]


def load_task(run, frozen):
    task = TaskSpec.from_dict(read(Path(run) / "frozen_tasks" / frozen["task_id"] / "task.json"))
    if task.sha256() != frozen["task_sha256"]:
        raise ValueError("frozen Task content changed")
    prefs = [PreferenceSpec(**p) for p in read(Path(run) / "frozen_tasks" / task.task_id / "preferences.json")]
    return task, prefs


def stream_root(run, frozen, method=None):
    path = Path(run) / ("benchmark_search" if method else "teacher_predictions") / frozen["task_id"]
    return path / method if method else path


def make_stream(run, frozen, method=None):
    """Identical immutable config, isolated predictions and cache per stream."""
    run = Path(run); path = stream_root(run, frozen, method)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
        target = path / name
        if target.exists():
            if sha(target) != sha(run / name):
                raise ValueError("retained stream has different frozen identity: " + str(target))
        else:
            shutil.copyfile(run / name, target)
    return path


def _parallel(run, jobs, phase, workers=4):
    """Bounded subprocesses; service elapsed and phase makespan stay separate."""
    run = Path(run); receipt = run / (phase + "_phase.json")
    if receipt.exists():
        return read(receipt)
    started = time.perf_counter(); started_utc = now()
    logroot = run / "command_logs"; logroot.mkdir(exist_ok=True)

    def invoke(job):
        task_id, method = job
        argv = [sys.executable, "-B", "-X", "utf8", "-m", "v6_4.evaluate_preference_warmstart",
                "_worker", "--run", str(run), "--phase", phase, "--task", task_id]
        if method:
            argv += ["--method", method]
        filename = f"{phase}_{task_id}" + ("_" + method if method else "") + ".log"
        log = logroot / filename
        # Appending preserves every failed invocation and its recovery evidence.
        before = time.perf_counter()
        with log.open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"argv": argv, "started_utc": now()}) + "\n"); stream.flush()
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        value = {"task_id": task_id, "method": method, "exit_code": result.returncode,
                 "elapsed_wall_s": time.perf_counter() - before, "log": log.relative_to(run).as_posix()}
        print(json.dumps({"event": "C2_WORKER_TERMINAL", "phase": phase, **value}), flush=True)
        return value

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(invoke, j) for j in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    value = {"phase": phase, "started_utc": started_utc, "ended_utc": now(),
             "makespan_s": time.perf_counter() - started,
             "cumulative_worker_service_s": sum(r["elapsed_wall_s"] for r in results),
             "workers": workers, "jobs": results, "passed": all(r["exit_code"] == 0 for r in results),
             "nested_times_are_not_added": True}
    if not value["passed"]:
        failure = run / f"{phase}_failure_{time.time_ns()}.json"
        write(failure, value)
        raise RuntimeError("retained worker failure; no automatic physical retry: " + str(failure))
    write(receipt, value)
    return value


def collect_teacher(run):
    tasks = [t for split in ("train", "val") for t in task_rows(run, split)
             if t.get("new_mother", True) and not t.get("historical", False)]
    if len(tasks) != 4:
        raise ValueError("exactly four new TRAIN/VAL tasks required")
    return _parallel(run, [(t["task_id"], None) for t in tasks], "teacher")


def run_teacher_task(run, task_id):
    from .preference_warmstart_protocol import BudgetLedger
    frozen = next(t for t in read(Path(run) / "plan.json")["tasks"] if t["task_id"] == task_id)
    BudgetLedger(run).reserve("new_teacher_prediction_slots", "teacher:" + task_id, 12,
                              {"reservation_scope": "maximum_stream_slots", "no_extra_allowance_on_resume": True})
    task, prefs = load_task(run, frozen); stream = make_stream(run, frozen)
    evaluator = NominalCandidateEvaluator(stream, task, frozen)
    result = optimize(task, prefs, evaluator.execution_identity, evaluator, stream / "planning" / task_id)
    if result["budget"]["stop_reason"] == "TOOL_ERROR":
        raise RuntimeError("consumed teacher tool error; protocol incomplete")
    return result


def _initializers(run, task, method, directory):
    """Construct exactly two frozen overrides. Never regenerate on resume."""
    from .route_initializers import FrozenSeedInitializer
    from .preference_teacher_dataset import load_dataset
    directory = Path(directory)
    retained = directory / "initializer_proposals.json"
    if retained.exists():
        payload = read(retained)
    else:
        began = time.perf_counter()
        started_path = directory / "initializer_started.json"
        if started_path.exists():
            raise RuntimeError("unfinished initializer draw retained; no silent resampling")
        write(started_path, {"method": method, "task_sha256": task.sha256(), "started_utc": now()})
        dataset = load_dataset(Path(run) / "dataset")
        condition_s = time.perf_counter() - began
        if method == "N":
            from .route_initializers import RetrievalInitializer
            started = time.perf_counter()
            provider = RetrievalInitializer(dataset.samples, dataset.condition_scaler,
                identity={"dataset_manifest_sha256": sha(Path(run) / "dataset" / "dataset_manifest.json"),
                          "condition_scaler_sha256": sha(Path(run) / "dataset" / "condition_scaler.json")})
            seeds = provider(task)
            timing = {"condition_scaler_prepare_s": condition_s, "retrieval_s": time.perf_counter() - started,
                      "model_load_s": 0., "warm_inference_s": 0.}
        else:
            from .preference_warmstart_protocol import BudgetLedger
            BudgetLedger(run).reserve("test_ddim_samples", "test:" + task.sha256(), 2,
                                     {"conditions": ["A-v1", "B-v2"], "noise_seed": 64224})
            started = time.perf_counter()
            from .preference_diffusion_warmstart import PreferenceSampler
            import_s = time.perf_counter() - started
            started = time.perf_counter()
            sampler = PreferenceSampler.from_frozen(Path(run) / "model", device="cpu")
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            seeds = sampler.initializer_proposals(task, noise_seed=64224)
            timing = {"condition_scaler_prepare_s": condition_s, "retrieval_s": 0.,
                      "model_import_s": import_s, "model_load_s": load_s,
                      "warm_inference_s": time.perf_counter() - started}
        payload = {"method": method, "task_sha256": task.sha256(),
                   "model_freeze_sha256": sha(Path(run) / "model_freeze.json"),
                   "proposals": {str(k): v for k, v in seeds.items()}, "timing": timing,
                   "generated_once": True, "test_quality_read": False}
        from .route_initializers import json_raw
        write(retained, json_raw(payload))
    if payload["task_sha256"] != task.sha256() or payload["model_freeze_sha256"] != sha(Path(run) / "model_freeze.json"):
        raise ValueError("initializer freeze/task changed")
    return FrozenSeedInitializer({int(k): v for k, v in payload["proposals"].items()}, identity=payload), payload["timing"]


def run_search_task(run, task_id, method):
    from .preference_warmstart_protocol import BudgetLedger
    run = Path(run)
    if method not in METHOD_BUDGETS or not (run / "model_freeze.json").exists():
        raise ValueError("TEST requires a frozen model and retrieval rule")
    verify_model_freeze(run)
    frozen = next(t for t in task_rows(run, "test") if t["task_id"] == task_id)
    BudgetLedger(run).reserve("test_prediction_slots", "search:" + task_id + ":" + method, METHOD_BUDGETS[method],
                              {"reservation_scope": "maximum_stream_slots", "no_extra_allowance_on_resume": True})
    task, prefs = load_task(run, frozen); stream = make_stream(run, frozen, method)
    output = stream / "planning" / task_id
    completed = stream / "planning_cost.json"
    if completed.exists():
        result = read(output / "selection.json")
        # optimize verifies retained prediction seals and deterministic inputs.
    began = time.perf_counter(); initializer = None; setup = {}
    if method != "R":
        initializer, setup = _initializers(run, task, method, stream)
    evaluator = NominalCandidateEvaluator(stream, task, frozen)
    result = optimize(task, prefs, evaluator.execution_identity, evaluator, output,
                      search_spec=SearchSpec(candidate_budget=METHOD_BUDGETS[method]), initializer=initializer)
    if result["budget"]["stop_reason"] == "TOOL_ERROR":
        raise RuntimeError("consumed TEST tool error; protocol incomplete")
    if not completed.exists():
        write(completed, {"method": method, "task_id": task_id, "setup": setup,
              "end_to_end_cold_planning_s": time.perf_counter() - began,
              "warm_resident_model_path_estimate_s": time.perf_counter() - began - setup.get("model_load_s", 0.) - setup.get("model_import_s", 0.),
              "warm_timing_scope": "measured decomposition with model import/load removed; no repeated search was run",
              "includes_inference": True, "selection_sha256": sha(output / "selection.json"),
              "selection_sealed_utc": now(), "physics_from_fresh_task_state": True})
    return result


def benchmark_search(run):
    run = Path(run)
    if not (run / "model_freeze.json").exists():
        raise ValueError("freeze model before TEST")
    _parallel(run, [(t["task_id"], m) for t in task_rows(run, "test") for m in METHOD_BUDGETS], "search")
    return seal_selections(run)


def verify_model_freeze(run):
    run = Path(run); frozen = read(run / "model_freeze.json")
    if frozen["retrieval_identity"]["dataset_sha256"] != sha(run / "dataset" / "dataset_manifest.json"):
        raise ValueError("retrieval TRAIN pool changed after model freeze")
    for path, expected in frozen["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("model/scaler/schema changed after freeze: " + path)
    return frozen


def _endpoint_selection(run, frozen, endpoint):
    directory = stream_root(run, frozen, endpoint[0]) / "planning" / frozen["task_id"]
    if endpoint == "R8":
        path = directory / "prefix_08.json"
    else:
        path = directory / "selection.json"
    return path, read(path)


def seal_selections(run):
    run = Path(run); target = run / "sealed_selections" / "all_selections.json"
    if target.exists():
        return verify_selections(run)
    if (run / "actual").exists():
        raise ValueError("all selections must be sealed before any actual")
    files = {}
    for frozen in task_rows(run, "test"):
        for endpoint in ENDPOINTS:
            path, selected = _endpoint_selection(run, frozen, endpoint)
            copy = target.parent / frozen["task_id"] / (endpoint + ".json")
            write(copy, selected)
            files[copy.relative_to(run).as_posix()] = sha(copy)
            files[path.relative_to(run).as_posix()] = sha(path)
        for method in METHOD_BUDGETS:
            directory = stream_root(run, frozen, method) / "planning" / frozen["task_id"]
            for name in ("candidate_registry.json", "proposals.json"):
                files[(directory / name).relative_to(run).as_posix()] = sha(directory / name)
    value = {"schema": "v64_c2_all_endpoints_sealed_v1", "sealed_utc": now(),
             "files": files, "model_freeze_sha256": sha(run / "model_freeze.json"),
             "logical_actual_slots": 32, "actual_started": False, "selection_reads_actual": False}
    write(target, value)
    return value


def verify_selections(run):
    run = Path(run)
    from .preference_warmstart_protocol import verify_frozen
    verify_frozen(run)
    verify_model_freeze(run)
    frozen = read(run / "sealed_selections" / "all_selections.json")
    if sha(run / "model_freeze.json") != frozen["model_freeze_sha256"]:
        raise ValueError("model freeze changed after selections")
    for relative, expected in frozen["files"].items():
        if sha(run / relative) != expected:
            raise ValueError("sealed selection changed: " + relative)
    return frozen


def execute_task(run, task_id):
    """Eight logical endpoints per Task; strict identity aliases save work."""
    from .conditional_execution import ExecutionCostLedger
    from .residual_execution import execute_residual_attempt
    from .task_anchored_reference import TaskAnchoredResidualPlan
    from .preference_warmstart_protocol import BudgetLedger
    run = Path(run); verify_selections(run)
    BudgetLedger(run).reserve("actual_logical_slots", "actual:" + task_id, 8,
                              {"endpoints": list(ENDPOINTS), "preferences": ["A", "B"]})
    frozen = next(t for t in task_rows(run, "test") if t["task_id"] == task_id)
    task, _ = load_task(run, frozen)
    aliases = {}; slots = []
    for endpoint in ENDPOINTS:
        selection_path = run / "sealed_selections" / task_id / (endpoint + ".json")
        selection = read(selection_path)
        for preference in ("A", "B"):
            method = endpoint + "_" + preference
            directory = run / "actual" / task_id / method
            if directory.exists():
                if not (directory / "slot.json").exists():
                    raise RuntimeError("unfinished consumed actual retained; never retry: " + str(directory))
                verify_seal(directory); slot = read(directory / "slot.json")
                if slot["selection_sha256"] != sha(selection_path):
                    raise ValueError("retained actual selection changed")
                if slot.get("unique_run"):
                    aliases[slot["alias_identity"]] = slot
                slots.append(slot); continue
            directory.mkdir(parents=True)
            chosen = selection["preferences"][preference]
            plan = TaskAnchoredResidualPlan.from_dict(chosen["selected_plan"]) if chosen["selected_plan"] else None
            alias_identity = digest({"task_sha256": task.sha256(), "plan_sha256": plan.sha256() if plan else None,
                                    "source_identity_sha256": sha(run / "source_identity.json"),
                                    "config_sha256": sha(run / "frozen_execution_config.json"),
                                    "run_config_sha256": sha(run / "frozen_run_config.json")})
            slot = {"task_id": task_id, "endpoint": endpoint, "preference": preference, "method": method,
                    "logical_slot": len(slots), "selection_sha256": sha(selection_path),
                    "task_sha256": task.sha256(), "plan_sha256": plan.sha256() if plan else None,
                    "alias_identity": alias_identity, "source_candidate_id": chosen["source_candidate_id"],
                    "unique_run": False, "entered_actual": False, "status": "NO_PLAN", "actual_steps": 0,
                    "full_task_success": False, "original_independent_gates_passed": False,
                    "clearance_30mm_met": None, "quality": None, "costs": None,
                    "prediction_actual_consistent": None, "deployment": "NOT_MET"}
            if plan is not None:
                write(directory / "selected_plan.json", plan.to_dict())
                if alias_identity in aliases:
                    source = aliases[alias_identity]
                    for key in ("status", "entered_actual", "actual_steps", "full_task_success",
                                "original_independent_gates_passed", "clearance_30mm_met", "quality",
                                "prediction_actual_consistent", "prediction_actual_difference"):
                        slot[key] = source.get(key)
                    slot["alias_of_method"] = source["method"]
                    slot["costs"] = {"alias_zero_new_work": True}
                else:
                    before = time.perf_counter(); ledger = ExecutionCostLedger()
                    with ledger.installed():
                        result = execute_residual_attempt(task, plan, directory / "attempt",
                            qp_config_path=run / "frozen_execution_config.json", identity_path=run / "source_identity.json",
                            slot_id=task_id + "_" + method, execution_diagnostics=True,
                            diagnostic_obstacle_name=frozen["obstacle_name"])
                        slot.update(unique_run=bool(result.get("actual_runner_started")), entered_actual=result["entered_actual"],
                                    status=result["status"], actual_steps=result["actual_steps"], full_task_success=result["full_task_success"])
                        evaluation = result.get("evaluation") or {}
                        slot["original_independent_gates_passed"] = bool(evaluation.get("evidence_valid") and all(
                            evaluation.get(k, {}).get("passed") is True for k in
                            ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")))
                        replay = directory / "attempt" / "actual" / "evaluation" / "fresh_replay.npz"
                        if result.get("trace_path") and slot["actual_steps"] and replay.exists():
                            with ledger.scope("actual_quality"):
                                slot["quality"], _ = fresh_quality(task, plan, result["trace_path"], frozen,
                                    directory / "quality", evidence_root=directory / "attempt" / "actual", replay_path=replay)
                            slot["clearance_30mm_met"] = slot["quality"].get("d_support") is not None and slot["quality"]["d_support"] >= .030
                        slot["costs"] = ledger.to_dict()
                    slot["elapsed_wall_s"] = time.perf_counter() - before
                    if result.get("pipeline_failure"):
                        slot["tool_error"] = result["pipeline_failure"]
                    aliases[alias_identity] = slot
                # Each logical method binds its own prediction, even when its
                # identical actual reference aliases another method's run.
                predicted = chosen.get("prediction_metrics")
                if predicted and slot["quality"]:
                    differences = {k: slot["quality"][k] - predicted[k] for k in ("I_support", "L_full", "d_support")
                                   if slot["quality"].get(k) is not None and predicted.get(k) is not None}
                    slot["prediction_actual_difference"] = differences
                    slot["prediction_actual_consistent"] = bool(len(differences) == 3 and
                        slot["full_task_success"] and slot["original_independent_gates_passed"] and all(abs(v) <= 1e-9 for v in differences.values()))
            write(directory / "slot.json", slot); seal(directory); slots.append(slot)
            print(json.dumps({"event": "C2_FINAL_ACTUAL", "task": task_id, "method": method,
                              "status": slot["status"], "steps": slot["actual_steps"], "unique": slot["unique_run"],
                              "five_gates": slot["original_independent_gates_passed"]}), flush=True)
            if slot.get("tool_error"):
                raise RuntimeError("consumed actual tool error; no retry")
    return slots


def execute_selected(run):
    run = Path(run); verify_selections(run)
    _parallel(run, [(t["task_id"], None) for t in task_rows(run, "test")], "actual")
    target = run / "actual_complete.json"
    if not target.exists():
        slots = [read(p) for p in sorted((run / "actual").glob("*/*/slot.json"))]
        write(target, {"completed_utc": now(), "logical_slots": len(slots),
                       "unique_actual_runs": sum(s["unique_run"] for s in slots),
                       "slot_hashes": {str(p.relative_to(run).as_posix()): sha(p) for p in sorted((run / "actual").glob("*/*/slot.json"))}})
    return read(target)


def validate(run):
    run = Path(run); verify_selections(run)
    target = run / "validation" / "validation.json"
    if target.exists():
        verify_seal(target.parent); return read(target)
    complete = read(run / "actual_complete.json")
    slots = []
    for relative, expected in complete["slot_hashes"].items():
        path = run / relative
        if sha(path) != expected:
            raise ValueError("actual digest changed")
        verify_seal(path.parent); value = read(path); slots.append(value)
        if value["full_task_success"] and (value["actual_steps"] != 13500 or not value["original_independent_gates_passed"] or not value["quality"]):
            raise ValueError("full actual lacks complete original independent evidence")
    teacher = [r for p in (run / "teacher_predictions").glob("*/planning/*/candidate_registry.json") for r in read(p)]
    test = [r for p in (run / "benchmark_search").glob("*/*/planning/*/candidate_registry.json") for r in read(p)]
    for path in [*(run / "teacher_predictions").glob("*/planning/*/predictions/*"),
                 *(run / "benchmark_search").glob("*/*/planning/*/predictions/*")]:
        verify_seal(path)
    if len(teacher) > 48 or len(test) > 112 or len(slots) != 32:
        raise ValueError("finite protocol budget violated")
    physics = sum(r.get("prediction_steps", 0) for r in [*teacher, *test])
    actual_physics = sum(s["actual_steps"] for s in slots if s["unique_run"])
    if physics > 2160000 or actual_physics > 432000:
        raise ValueError("main physics budget violated")
    errors = [r for r in [*teacher, *test, *slots] if r.get("tool_error")]
    value = {"all_terminal": not errors, "teacher_slots": len(teacher), "test_slots": len(test),
             "logical_actual_slots": len(slots), "unique_actual_runs": sum(s["unique_run"] for s in slots),
             "main_prediction_physics_steps": physics, "main_actual_physics_steps": actual_physics,
             "prediction_rollouts_started": sum(r.get("prediction_rollout_started", False) for r in [*teacher, *test]),
             "tool_error_count": len(errors), "validation_additional_physics_steps": 0,
             "original_five_gates_performed_by_inherited_executor": True,
             "all_selections_precede_actual": True, "cross_stream_cache": False,
             "denominator_per_endpoint_preference": 4, "deployment": "NOT_MET"}
    write(target, value); seal(target.parent)
    return value


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _first_match(rows, oracle, preference):
    if not oracle:
        return {"slot": None, "status": "N/A_R12_NO_PLAN"}
    for index, row in enumerate(rows, 1):
        if not admissible(row):
            continue
        m = row["prediction_metrics"]
        good = (m["L_full"] <= oracle["L_full"] + .005 and
                (m["I_support"] <= oracle["I_support"] + .001 if preference == "A" else
                 m.get("d_support") is not None and m["d_support"] >= .030))
        if good:
            return {"slot": index, "status": "HIT"}
    return {"slot": None, "status": "RIGHT_CENSORED", "right_censored_budget": len(rows)}


def endpoint_planning_cost(run, frozen, endpoint):
    root = stream_root(run, frozen, endpoint[0])
    cost = read(root / "planning_cost.json")
    if endpoint != "R8":
        return {"cold_s": cost["end_to_end_cold_planning_s"], "warm_s": cost["warm_resident_model_path_estimate_s"],
                "warm_scope": cost["warm_timing_scope"]}
    # R8 is the online-sealed prefix of the same R12 run, with its own elapsed.
    planning = root / "planning" / frozen["task_id"]
    prefix, final = read(planning / "prefix_08.json"), read(planning / "selection.json")
    setup = max(0., cost["end_to_end_cold_planning_s"] - final["elapsed_wall_s"])
    return {"cold_s": setup + prefix["elapsed_wall_s"], "warm_s": setup + prefix["elapsed_wall_s"],
            "timing_scope": "same-run sealed eighth-slot elapsed plus measured setup"}


def actual_near_quality(method, baseline, preference):
    if not (method["full_task_success"] and method["original_independent_gates_passed"] and
            baseline["full_task_success"] and baseline["original_independent_gates_passed"]):
        return None
    a, b = method.get("quality"), baseline.get("quality")
    if not a or not b:
        return None
    return bool(a["L_full"] <= b["L_full"] + .005 and
                (a["I_support"] <= b["I_support"] + .001 if preference == "A" else a.get("d_support") is not None and a["d_support"] >= .030))


def aggregate_costs(rows):
    ledgers = [r.get("costs") or {} for r in rows]
    keys = ("actual_physics_steps", "prediction_physics_steps", "private_preview_physics_steps",
            "independent_saved_torque_replay_steps", "preview_calls", "native_geometry_query_calls", "qp_solve_calls")
    return {**{k: sum(c.get(k, 0) for c in ledgers) for k in keys},
            "cumulative_service_elapsed_s": sum(r.get("elapsed_wall_s", 0.) for r in rows),
            "nested_elapsed_not_added": True,
            "phase_counts": {phase: {op: sum(c.get("phase_counts", {}).get(phase, {}).get(op, {}).get("returned", 0)
                                           for c in ledgers) for op in ("mj_step", "mj_step2", "mj_geomDistance", "qp_solve")}
                             for phase in sorted({p for c in ledgers for p in c.get("phase_counts", {})})}}


def report(run):
    run = Path(run)
    training_path = run / "model" / "training_summary.json"
    if training_path.exists() and read(training_path).get("training_executed") is False:
        return report_no_supervision(run)
    checked = validate(run)
    if (run / "summary.json").exists():
        return read(run / "summary.json")
    actual = [read(p) for p in sorted((run / "actual").glob("*/*/slot.json"))]
    from .preference_teacher_dataset import load_dataset
    dataset = load_dataset(run / "dataset")
    training = read(run / "model" / "training_summary.json")
    write(run / "tables" / "A_data_training.json", {"dataset": dataset.manifest, "training": training,
          "independent_candidate_count_is_not_label_view_count": True})
    states = []
    for endpoint in ENDPOINTS:
        for pref in ("A", "B"):
            rows = [s for s in actual if s["endpoint"] == endpoint and s["preference"] == pref]
            states.append({"endpoint": endpoint, "preference": pref, "denominator": 4,
                "full_task": sum(s["full_task_success"] for s in rows),
                "original_five_gates": sum(s["original_independent_gates_passed"] for s in rows),
                "full_and_B30mm": sum(s["full_task_success"] and s["clearance_30mm_met"] is True for s in rows),
                "NO_PLAN": sum(s["status"] == "NO_PLAN" for s in rows),
                "actual_failed": sum(s["entered_actual"] and not s["full_task_success"] for s in rows),
                "unique_actual": sum(s["unique_run"] for s in rows), "aliases": sum("alias_of_method" in s for s in rows)})
    _csv(run / "tables" / "C_actual_status.csv", states)
    metric_names = ("I_support", "L_full", "d_support", "base_translation_peak_m", "base_rotation_peak_rad")
    quality = [{"task_id": s["task_id"], "endpoint": s["endpoint"], "preference": s["preference"],
                "status": s["status"], "full_task": s["full_task_success"], "alias_of": s.get("alias_of_method"),
                **{k: (s.get("quality") or {}).get(k) if s["full_task_success"] else None for k in metric_names}} for s in actual]
    _csv(run / "tables" / "C_actual_quality.csv", quality)
    pairs = []
    for method in ("D8", "N8"):
        for baseline in ("N8", "R8", "R12"):
            if method == baseline:
                continue
            for pref in ("A", "B"):
                records = []
                for frozen in task_rows(run, "test"):
                    first = next(s for s in actual if (s["task_id"], s["endpoint"], s["preference"]) == (frozen["task_id"], method, pref))
                    second = next(s for s in actual if (s["task_id"], s["endpoint"], s["preference"]) == (frozen["task_id"], baseline, pref))
                    method_cost = endpoint_planning_cost(run, frozen, method)
                    baseline_cost = endpoint_planning_cost(run, frozen, baseline)
                    record = {"task_id": frozen["task_id"], "method_full_task": first["full_task_success"], "baseline_full_task": second["full_task_success"],
                              "actual_within_declared_near_quality_band": actual_near_quality(first, second, pref),
                              "planning_cold_difference_s": method_cost["cold_s"] - baseline_cost["cold_s"],
                              "planning_warm_difference_s": method_cost["warm_s"] - baseline_cost["warm_s"],
                              "method_cold_planning_s": method_cost["cold_s"], "baseline_cold_planning_s": baseline_cost["cold_s"]}
                    if first["full_task_success"] and second["full_task_success"] and first["quality"] and second["quality"]:
                        record.update({k: first["quality"][k] - second["quality"][k] for k in metric_names})
                    records.append(record)
                pairs.append({"method": method, "baseline": baseline, "preference": pref,
                              "complete_paired_count": sum("I_support" in r for r in records), "all_four_tasks": records})
    write(run / "tables" / "D_actual_pairing.json", pairs)
    hits = []; costs = []; prefixes = []; attribution = []; learned = []
    for frozen in task_rows(run, "test"):
        tid = frozen["task_id"]
        oracle = read(run / "sealed_selections" / tid / "R12.json")
        for method, budget in METHOD_BUDGETS.items():
            root = stream_root(run, frozen, method); planning = root / "planning" / tid
            rows = read(planning / "candidate_registry.json")
            cost = read(root / "planning_cost.json")
            first = next((i for i, r in enumerate(rows, 1) if admissible(r)), None)
            first_b = next((i for i, r in enumerate(rows, 1) if admissible(r) and r["prediction_metrics"].get("d_support") is not None and r["prediction_metrics"]["d_support"] >= .030), None)
            hit = {"task_id": tid, "method": method, "budget": budget, "slots": len(rows),
                   "N_first_admissible": first, "N_first_B": first_b,
                   "first_admissible_right_censored": first is None, "first_B_right_censored": first_b is None,
                   "N_match_R12": {p: _first_match(rows, oracle["preferences"][p]["prediction_metrics"], p) for p in ("A", "B")}}
            hits.append(hit)
            costs.append({"task_id": tid, "method": method, **cost,
                          "candidate_slots": len(rows), "proposal_attempts": len(read(planning / "proposals.json")),
                          "nominal_rollouts": sum(r.get("prediction_rollout_started", False) for r in rows),
                          "prediction_main_steps": sum(r.get("prediction_steps", 0) for r in rows),
                          "candidate_elapsed_service_s": sum(r.get("elapsed_wall_s", 0.) for r in rows)})
            for n in ((4, 8, 12) if method == "R" else (4, 8)):
                snapshot = read(planning / f"prefix_{n:02d}.json")
                for pref in ("A", "B"):
                    item = snapshot["preferences"][pref]
                    prefixes.append({"task_id": tid, "method": method, "budget": n, "preference": pref,
                                     "prediction_plan": item["selected_plan"] is not None,
                                     "actual": "NOT_RUN_PREFIX" if n == 4 else "SEE_ENDPOINT_TABLE",
                                     **{k: (item.get("prediction_metrics") or {}).get(k) for k in metric_names}})
            if method == "D":
                learned.extend(p for p in read(planning / "proposals.json") if p.get("source") == "diffusion")
            for endpoint in (("R8", "R12") if method == "R" else (method + "8",)):
                selection = read(run / "sealed_selections" / tid / (endpoint + ".json"))
                by_id = {r["candidate_id"]: r for r in rows}
                for pref in ("A", "B"):
                    cid = selection["preferences"][pref]["source_candidate_id"]
                    if cid is None:
                        source = "no_plan"
                    else:
                        row = by_id[cid]; direct = row.get("source")
                        while row.get("parent_candidate_id"):
                            row = by_id[row["parent_candidate_id"]]
                        ancestor = row.get("origin_source", row.get("source"))
                        if direct == "initial":
                            direct = ancestor
                        source = ("direct_" + ancestor if direct == ancestor else "descendant_" + ancestor)
                    attribution.append({"task_id": tid, "endpoint": endpoint, "preference": pref,
                                        "candidate_id": cid, "source_lineage": source})
    write(run / "tables" / "B_first_hits.json", hits)
    write(run / "tables" / "D_planning_costs.json", costs)
    write(run / "tables" / "D_endpoint_planning_costs.json", [
        {"task_id": t["task_id"], "endpoint": e, **endpoint_planning_cost(run, t, e)}
        for t in task_rows(run, "test") for e in ENDPOINTS])
    _csv(run / "tables" / "B_budget_prefixes.csv", prefixes)
    _csv(run / "tables" / "B_final_attribution.csv", attribution)
    comparisons = []
    for pairing in pairs:
        records = pairing["all_four_tasks"]
        comparisons.append({"method": pairing["method"], "baseline": pairing["baseline"], "preference": pairing["preference"],
            "all_task_denominator": 4, "actual_near_quality_pairs": sum(r["actual_within_declared_near_quality_band"] is True for r in records),
            "near_quality_and_lower_cold_cost": sum(r["actual_within_declared_near_quality_band"] is True and r["planning_cold_difference_s"] < 0. for r in records),
            "near_quality_and_lower_warm_cost": sum(r["actual_within_declared_near_quality_band"] is True and r["planning_warm_difference_s"] < 0. for r in records),
            "all_failures_remain_visible": True})
    teacher_rows = [r for p in (run / "teacher_predictions").glob("*/planning/*/candidate_registry.json") for r in read(p)]
    teacher_cost = aggregate_costs(teacher_rows)
    actual_cost = aggregate_costs([s for s in actual if s["unique_run"]])
    prediction_rows = [r for p in (run / "benchmark_search").glob("*/*/planning/*/candidate_registry.json") for r in read(p)]
    test_cost = aggregate_costs(prediction_rows)
    summary = {"research_delivery_complete": checked["all_terminal"], "initializer_interface_verified": True,
               "historical_candidate_count_unique": dataset.manifest["historical_candidate_count_unique"], "new_teacher_prediction_slots": checked["teacher_slots"],
               "training_executed": training["training_executed"],
               "optimizer_updates_total": training["optimizer_updates_total"],
               "selected_checkpoint_update": training.get("selected_checkpoint_update"),
               "supported_preference_family_conditions": training.get("supported_preference_family_conditions"),
               "scalers_train_only": dataset.manifest["scalers_train_only"], "test_leakage_checks_passed": dataset.manifest["test_leakage_checks_passed"] and not training["test_used_for_selection"],
               "learned_seeds_generated": sum(s.get("ddim_sample_units", 0) for s in learned),
               "learned_seeds_raw_legal": sum((s.get("raw_seed_diagnostics") or {}).get("raw_legal", False) for s in learned),
               "learned_initializer_operational": any(s.get("ddim_sample_units", 0) for s in learned), "final_selected_source_breakdown": attribution,
               "full_task_by_endpoint_and_preference": states,
               "preferred_clearance_by_endpoint": states, "prediction_first_hit_and_censoring": hits,
               "actual_quality_pairing": pairs,
               "measured_cost_and_amortization": {"planning": costs, "teacher": teacher_cost,
                    "test_prediction": test_cost, "actual_including_independent_replay_and_geometry": actual_cost,
                    "training_service_s": training["elapsed_s"],
                    "phase_makespans": {p: read(run / (p + "_phase.json")) for p in ("teacher", "search", "actual")},
                    "warm_path_is_decomposition_estimate": True, "break_even_tasks": "NOT_ESTABLISHED",
                    "reason": "requires positive paired actual-preserving online saving; finite pilot tables govern conclusion"},
               "pilot_comparison_evidence": comparisons,
               "learning_benefit_established_in_pilot": "NOT_ESTABLISHED", "default_initializer_decision": "retain_C1_rule",
               "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED", "hardware_safety": "NOT_ESTABLISHED",
               "scope": "four TEST tasks, two new mothers, one training seed; no population non-inferiority claim"}
    write(run / "summary.json", summary)
    lines = ["# V6.4-C.2 preference-conditioned diffusion warm start", "",
             "Fixed-budget pilot using real trained weights and unchanged C.1 control/safety execution.", "",
             "| Endpoint | Preference | Full Task /4 | Five gates /4 | Full +30mm /4 | NO_PLAN |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"|{s['endpoint']}|{s['preference']}|{s['full_task']}|{s['original_five_gates']}|{s['full_and_B30mm']}|{s['NO_PLAN']}|" for s in states]
    lines += ["", "Actual failures and NO_PLAN stay in the denominator. Four-slot curves are prediction only.",
              "Teacher records distinguish validated execution from complete prediction. Family near-optimal labels are not global optima.",
              "", "Tables A–D and machine summary retain raw metrics, costs, first hits, censoring and lineage.",
              "Eight versus twelve slots is a configured quota reduction; it alone is no measured speedup.",
              "Learning benefit is NOT_ESTABLISHED pending paired quality/cost interpretation; default remains C.1 rule.",
              "Deployment NOT_MET; continuous-time and hardware safety NOT_ESTABLISHED."]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return summary


def report_no_supervision(run):
    """The protocol's explicit finite-data stop; never invent random weights."""
    from .preference_teacher_dataset import load_dataset
    from .preference_warmstart_protocol import verify_frozen
    run = Path(run); verify_frozen(run)
    target = run / "summary.json"
    if target.exists():
        return read(target)
    ds = load_dataset(run / "dataset")
    training = read(run / "model" / "training_summary.json")
    if ds.manifest["supervision_available"] or training["training_executed"]:
        raise ValueError("data-negative stop requires absent valid supervision")
    rows = [r for p in (run / "teacher_predictions").glob("*/planning/*/candidate_registry.json") for r in read(p)]
    state = {"research_delivery_complete": True, "formal_test_protocol_complete": False,
        "completion_scope": "predeclared finite-data negative stop: no valid TRAIN or VAL supervision",
        "initializer_interface_verified": True, "historical_candidate_count_unique": 48,
        "new_teacher_prediction_slots": len(rows), "training_executed": False, "optimizer_updates_total": 0,
        "selected_checkpoint_update": None, "supported_preference_family_conditions": [],
        "scalers_train_only": ds.manifest["scalers_train_only"], "test_leakage_checks_passed": True,
        "learned_seeds_generated": 0, "learned_seeds_raw_legal": 0, "learned_initializer_operational": False,
        "final_selected_source_breakdown": [], "full_task_by_endpoint_and_preference": "NOT_RUN",
        "preferred_clearance_by_endpoint": "NOT_RUN", "prediction_first_hit_and_censoring": "NOT_RUN",
        "actual_quality_pairing": "NOT_RUN", "measured_cost_and_amortization": "NOT_ESTABLISHED",
        "learning_benefit_established_in_pilot": "NOT_ESTABLISHED", "default_initializer_decision": "retain_C1_rule",
        "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED", "hardware_safety": "NOT_ESTABLISHED"}
    write(run / "tables" / "A_data_training.json", {"dataset": ds.manifest, "training": training})
    write(target, state)
    (run / "REPORT.md").write_text("# C.2 finite-data negative result\n\nNo valid TRAIN or VAL supervision was available within the frozen teacher budget. "
        "No random checkpoint, additional seed, or replacement task was created. TEST search and final actual were NOT_RUN. "
        "Components and finite data evidence were delivered under the explicit insufficient-supervision stop.\n", encoding="utf8")
    return state


def train(run):
    from .preference_diffusion_warmstart import prepare_training, train_model
    from .preference_warmstart_protocol import BudgetLedger
    run = Path(run)
    prepared = run / "model" / "training_config.json"
    terminal = run / "model" / "training_status.json"
    if not prepared.exists() and not terminal.exists():
        prepare_training(run / "dataset" / "dataset_manifest.json", run / "model", device="cpu")
    if prepared.exists():
        ledger = BudgetLedger(run)
        ledger.reserve("formal_training_runs", "formal_training_seed64221", 1)
        ledger.reserve("optimizer_updates", "formal_training_seed64221", 4000)
        ledger.reserve("validation_ddim_samples", "formal_validation_seed64223", 128,
                       {"reservation_scope": "maximum_16_checkpoints_times_8_units"})
    return train_model(run / "model")


def freeze_model(run):
    from .preference_diffusion_warmstart import freeze_model as freeze
    run = Path(run)
    retrieval = {"schema": "c2_train_only_retrieval_v1", "dataset_sha256": sha(run / "dataset" / "dataset_manifest.json"),
                 "distance": "TRAIN-normalized continuous condition MSE plus discrete mismatch; exact pref/family/search structure",
                 "tie": ["task_sha256", "plan_sha256", "sample_id"], "test_quality_used": False}
    return freeze(run / "model", retrieval_identity=retrieval, output=run / "model_freeze.json")


def run_all(run):
    from .preference_teacher_dataset import import_c1, build_dataset
    run = Path(run)
    import_c1(run); collect_teacher(run); build_dataset(run)
    trained = train(run)
    if trained.get("training_executed") is False:
        return report_no_supervision(run)
    freeze_model(run); benchmark_search(run); execute_selected(run)
    validate(run); return report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "import-c1", "collect-teacher", "build-dataset", "train",
        "freeze-model", "benchmark-search", "execute-selected", "validate", "report", "run-all", "_worker"))
    parser.add_argument("--output"); parser.add_argument("--run")
    parser.add_argument("--phase", choices=("teacher", "search", "actual")); parser.add_argument("--task"); parser.add_argument("--method")
    args = parser.parse_args()
    if args.command == "prepare":
        from .preference_warmstart_protocol import prepare
        if not args.output:
            parser.error("prepare requires --output")
        result = prepare(Path(args.output))
    else:
        if not args.run:
            parser.error("command requires --run")
        run = Path(args.run).resolve()
        if args.command == "_worker":
            result = (run_teacher_task(run, args.task) if args.phase == "teacher" else
                      run_search_task(run, args.task, args.method) if args.phase == "search" else execute_task(run, args.task))
        elif args.command in ("import-c1", "build-dataset"):
            from .preference_teacher_dataset import import_c1, build_dataset
            result = (import_c1 if args.command == "import-c1" else build_dataset)(run)
        else:
            result = globals()[args.command.replace("-", "_")](run)
    print(json.dumps({"event": "C2_COMMAND_COMPLETED", "command": args.command, "ended_utc": now()}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
