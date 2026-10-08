"""C.3 finite search-aware warmstarts, closed-loop VAL and independent TEST.

The inherited physical evaluator, search order and final independent gates are
unchanged. This module owns declarations, isolation, budgets and orchestration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .route_optimizer_protocol import (ROOT, SearchSpec, PreferenceSpec, VERSIONS,
    active_intervals, digest, git, read, related_pairs, sha, write)
from .task_protocol import TaskSpec, task_from_scenario
from .task_anchored_reference import build_reference_definition, _target

BASE = "1758e13b01735b80c5a512b81cbc1d5e47a07ec9"
BRANCH = "v6.4-c3-search-aware-closed-loop-val"
C2 = ROOT / "v6_4/releases/preference_warmstart_20261008_01/snapshot"
METHODS = ("R", "N", "S", "D")
ENDPOINTS = ("R8", "R12", "N8", "S8", "D8")
CHECKPOINTS = ("D250", "D4000", "S250", "S4000")
LIMITS = {"teacher_candidate_slots": 96, "val_candidate_slots": 88,
    "test_candidate_slots": 144, "val_actual_slots": 20, "test_actual_slots": 40,
    "val_ddim_samples": 8, "test_ddim_samples": 8, "smoke_ddim_samples": 8,
    "diagnostic_ddim_samples": 8, "D_training_runs": 1, "S_training_runs": 1}
SEEDS = {"D_init": 64321, "S_init": 64331, "training_draw": 64322,
    "D_training_noise": 64323, "test_noise": 64324, "val_noise": 64325}


def now():
    return datetime.now(timezone.utc).isoformat()


class BudgetLedger:
    """Exclusive immutable unit reservations, including rejected/NO_PLAN slots."""

    def __init__(self, run):
        self.root = Path(run) / "budget_ledger"
        self.root.mkdir(parents=True, exist_ok=True)

    def entries(self):
        return [read(p) for p in sorted(self.root.glob("*.json"))]

    def totals(self):
        totals = {k: 0 for k in LIMITS}
        for row in self.entries():
            totals[row["category"]] += row["count"]
        totals["candidate_slots_total"] = sum(totals[k] for k in
            ("teacher_candidate_slots", "val_candidate_slots", "test_candidate_slots"))
        totals["actual_slots_total"] = totals["val_actual_slots"] + totals["test_actual_slots"]
        totals["ddim_samples_total"] = sum(totals[k] for k in
            ("val_ddim_samples", "test_ddim_samples", "smoke_ddim_samples", "diagnostic_ddim_samples"))
        return totals

    def reserve(self, category, unit_id, count, metadata=None):
        if category not in LIMITS or type(count) is not int or count < 1:
            raise ValueError("invalid C.3 budget reservation")
        metadata = metadata or {}
        path = self.root / (digest([category, unit_id]) + ".json")
        if path.exists():
            row = read(path)
            if row["count"] != count or row["metadata"] != metadata:
                raise ValueError("reserved unit changed on recovery")
            return row
        totals = self.totals()
        if totals[category] + count > LIMITS[category]:
            raise ValueError("predeclared C.3 budget exhausted")
        row = {"category": category, "unit_id": unit_id, "count": count,
            "metadata": metadata, "utc": now(), "status": "CONSUMED_NO_SILENT_RETRY"}
        write(path, row)
        return row


def validate_splits(manifest):
    seen, mothers = {}, {}
    for row in manifest["tasks"]:
        key, split = row["task_sha256"], row["split"]
        if key in seen or split not in ("train", "val", "test"):
            raise ValueError("invalid external split")
        for mother in (row["mother_id"], row["mother_source_sha256"]):
            if mother in mothers and mothers[mother] != split:
                raise ValueError("mother/mirror split leakage")
            mothers[mother] = split
        if row.get("historical") and split != "train":
            raise ValueError("historical tasks are TRAIN only")
        seen[key] = row
    counts = {s: sum(r["split"] == s for r in seen.values()) for s in ("train", "val", "test")}
    groups = {s: len({r["mother_id"] for r in seen.values() if r["split"] == s}) for s in counts}
    if counts != {"train": 6, "val": 2, "test": 4} or groups != {"train": 3, "val": 1, "test": 2}:
        raise ValueError("C.3 exact 6/2/4 tasks and 3/1/2 mothers required")
    return seen


def _mask(definition):
    mask = [False] * 6
    for index in active_intervals(definition):
        mask[index] = True
    return mask


def _used_seeds():
    from .preference_warmstart_protocol import used_mother_seeds
    used = set(used_mother_seeds())
    inventory = []
    paths = set((ROOT / "v6_4/releases").glob("*/snapshot/learning_split_manifest.json"))
    paths.update((ROOT / "v6_4/releases").glob("*/snapshot/plan.json"))
    for local in (Path("E:/v64c2/v6_4/output"), ROOT / "v6_4/output"):
        paths.update(local.glob("*/plan.json"))
        paths.update(local.glob("*/learning_split_manifest.json"))
    for path in sorted(paths):
        value = read(path)
        for row in value.get("tasks", []) + value.get("all_learning_tasks", []):
            if row.get("seed") is not None:
                used.add(int(row["seed"]))
            used.update(int(s) for s in row.get("old_mother_seeds", []))
        inventory.append({"path": str(path.resolve()), "sha256": sha(path)})
    return used, inventory


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        verify_run(output)
        return read(output / "plan.json")
    if git("status", "--porcelain"):
        raise ValueError("commit implementation before freezing C.3 producer")
    if git("merge-base", BASE, "HEAD") != BASE or git("branch", "--show-current") != BRANCH:
        raise ValueError("C.3 fixed base/branch mismatch")
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec, build_scenarios, V6LiteRunConfig
    from .route_pair_protocol import _geometry_check, validate_pair_contract
    from .preference_warmstart_protocol import next_unused_seeds
    output.mkdir(parents=True)
    for name in ("frozen_execution_config.json", "frozen_run_config.json"):
        shutil.copyfile(C2 / name, output / name)
    old_manifest = read(C2 / "learning_split_manifest.json")
    rows = []
    for original in old_manifest["tasks"]:
        if original["split"] != "train":
            continue
        tid = original["task_id"]
        source = C2 / "frozen_tasks" / tid
        task = TaskSpec.from_dict(read(source / "task.json"))
        if task.sha256() != original["task_sha256"]:
            raise ValueError("C.2 TRAIN task SHA mismatch")
        directory = output / "frozen_tasks" / tid
        directory.mkdir(parents=True)
        for name in ("task.json", "preferences.json", "geometry_precheck.json"):
            if (source / name).exists():
                shutil.copyfile(source / name, directory / name)
        row = {**original, "split": "train", "historical": True,
            "role": "c2_original_train", "original_task_declared_split": task.split,
            "original_task_file_sha256": sha(source / "task.json"),
            "task_path": f"frozen_tasks/{tid}/task.json"}
        if "geometry_precheck_passed" not in row:
            row["geometry_precheck_passed"] = read(source / "geometry_precheck.json")["passed"]
        rows.append(row)
    used, inventory = _used_seeds()
    choices = next_unused_seeds(used, count=3)
    spec = default_v6_lite_robot_spec()
    for role, (source_index, seed) in zip(("val", "test0", "test1"), choices):
        split = "val" if role == "val" else "test"
        group = "c3_" + role
        mother = build_scenarios(spec, V6LiteRunConfig(scenario_count=1, seed=seed))[0]
        mother_sha = digest(mother.to_dict())
        layout = {"mother_seed": seed, "mother_source_sha256": mother_sha,
            "next_unused_source_index": source_index, "evaluation_role": role, "sphere_offset_m": .055}
        base = task_from_scenario(spec, mother, task_id=group + "_source", group_id=group,
            family="end_effector_detour", split=split, layout_diagnostics=layout)
        definition = build_reference_definition(base)
        active = active_intervals(definition)
        point = _target(base).sample(float(np.mean(definition["intervals_s"][2])))[0]
        basis = np.asarray(definition["transverse_bases"][2])
        pair = []
        for side, label in ((1, "plus"), (-1, "minus")):
            tid = group + "_" + label
            obstacles = list(mother.obstacles)
            if obstacles[1].radius != .025:
                raise ValueError("frozen sphere radius differs")
            obstacles[1] = replace(obstacles[1], center=point + side * .055 * basis[:, 0])
            scene = replace(mother, scenario_id=tid, obstacles=tuple(obstacles))
            task = task_from_scenario(spec, scene, task_id=tid, group_id=group,
                family="end_effector_detour", split=split, requirements=base.requirements,
                layout_diagnostics=layout)
            directory = output / "frozen_tasks" / tid
            write(directory / "task.json", task.to_dict())
            d = build_reference_definition(task)
            check = _geometry_check(spec, task, d)
            write(directory / "geometry_precheck.json", check)
            verifier, pairs = related_pairs(spec, task)
            pair_ids = tuple((p.geom_a_name, p.geom_b_name) for p in pairs)
            windows = [d["intervals_s"][i] for i in active]
            preferences = [PreferenceSpec(name, task.sha256(), (tuple(d["intervals_s"][2]),),
                tuple(tuple(w) for w in windows), task.scenario["workspace_obstacles"][1]["name"],
                pair_ids).to_dict() for name in ("A", "B")] if active else []
            write(directory / "preferences.json", preferences)
            rows.append({"task_id": tid, "group_id": group, "mother_id": group,
                "role": role, "split": split, "seed": seed,
                "task_sha256": task.sha256(), "mother_source_sha256": mother_sha,
                "next_unused_source_index": source_index, "old_mother_seeds": sorted(used),
                "mother_identity_new": True, "historical": False,
                "active_intervals": active, "active_dim": 2 * len(active),
                "reference_interval_mask": d["interval_mask"], "search_interval_mask": _mask(d),
                "W_key": [d["intervals_s"][2]], "W_support": windows, "W_full": [[0., 27.]],
                "obstacle_name": task.scenario["workspace_obstacles"][1]["name"],
                "related_pair_ids": pair_ids, "pair_policy_sha256": verifier._pair_policy_sha256,
                "geometry_precheck_passed": check["passed"],
                "initial_geometry_query_count": check["native_distance_queries"],
                "applicable": bool(active), "original_task_declared_split": task.split,
                "task_path": f"frozen_tasks/{tid}/task.json"})
            pair.append(task)
        validate_pair_contract(*pair)
    manifest = {"schema": "v64_c3_external_learning_split_v1", "split_unit": "mother_source_identity",
        "historical_TaskSpec_unchanged": True, "tasks": rows,
        "intended_mother_counts": {"train": 3, "val": 1, "test": 2},
        "intended_task_counts": {"train": 6, "val": 2, "test": 4},
        "seed_source_inventory": inventory, "DATA_LIMITED": True}
    validate_splits(manifest)
    write(output / "learning_split_manifest.json", manifest)
    write(output / "split_manifest.json", manifest)
    from .closed_loop_warmstart_validation import COST_ORDER_TEXT
    plan = {"schema": "v64_c3_search_aware_protocol_v1", "run_id": output.name,
        "base_publication_commit": BASE, "algorithm_producer_commit": git("rev-parse", "HEAD"),
        "tasks": rows, "all_learning_tasks": rows, "search": asdict(SearchSpec()),
        "methods": list(METHODS), "search_budgets": {"R": 12, "N": 8, "S": 8, "D": 8},
        "actual_endpoints": list(ENDPOINTS), "checkpoint_whitelist": list(CHECKPOINTS),
        "checkpoint_rule": ["full_27s_five_gates_desc", "actual_B30_desc", "near_R12_count_desc", "raw_illegal_asc", "first_near_position_encode9_asc", COST_ORDER_TEXT, "update_asc"], "budget_limits": LIMITS,
        "candidate_slots_max": 328, "actual_method_slots_max": 60, "seeds": SEEDS,
        "mother_seed_rule": "first three unused 2026100701+104729*i; role val,test0,test1; identity only, no outcome redraw",
        "method_order": "VAL per-task rotate [R12,D250,S250,D4000,S4000]; TEST per-task rotate [R,N,S,D]",
        "timing_load": "sequential formal search, no concurrent heavy work",
        "model": {"hidden": [128, 128], "activation": "SiLU", "schedule": "cosine100",
            "D_target": "v_prediction", "DDIM_steps": 20, "batch": 32, "lr": 1e-4,
            "weight_decay": .01, "gradient_clip": 1., "updates": 4000,
            "checkpoint_updates": [250, 4000], "paired_reference_draws": True},
        "near_quality": {"A_I_band": .001, "A_L_band": .005, "B_L_band": .005, "B_clearance_m": .030},
        "engineering_goal": "quality/capability preserved; >=10% planning cost or effective prediction work versus both N8 and S8",
        "created_utc": now(), "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED",
        "hardware_safety": "NOT_ESTABLISHED", "default_initializer": "C.1_rule"}
    write(output / "plan.json", plan)
    sources = [p for name in ("v6_4", "v6_lite", "model_test") for p in (ROOT / name).rglob("*.py")
        if not any(x in ("output", "releases", "visualization", "__pycache__") for x in p.relative_to(ROOT).parts)]
    sources.extend(spec._source_assets())
    identity = {"schema": "v64_c3_source_identity_v1", "git_head": git("rev-parse", "HEAD"),
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "base_publication_commit": BASE,
        "source_sha256": {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(sources))},
        "protected_artifacts": {str(p.resolve()): sha(p) for p in output.rglob("*") if p.is_file()},
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "argv": sys.argv, "cwd": str(ROOT), "frozen_utc": now(),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
    write(output / "source_identity.json", identity)
    write(output / "prepare_receipt.json", {"status": "COMPLETED", "physics_steps": 0,
        "new_tasks_frozen_before_teacher_and_training": True, "test_results_computed": False})
    return plan


def verify_run(run):
    from .residual_execution import source_guard
    run = Path(run)
    source_guard(run / "source_identity.json")
    validate_splits(read(run / "learning_split_manifest.json"))
    return read(run / "plan.json")


def task_rows(run, split):
    return [r for r in read(Path(run) / "plan.json")["tasks"] if r["split"] == split]


def load_task(run, frozen):
    run = Path(run)
    task = TaskSpec.from_dict(read(run / frozen["task_path"]))
    if task.sha256() != frozen["task_sha256"]:
        raise ValueError("frozen task changed")
    prefs = [PreferenceSpec(**p) for p in read(run / "frozen_tasks" / task.task_id / "preferences.json")]
    return task, prefs


def stream_path(run, frozen, stream_id, stage):
    return Path(run) / {"teacher": "teacher_search", "val": "closed_loop_val/search", "test": "test_search"}[stage] / frozen["task_id"] / stream_id


def run_stream(run, frozen, stream_id, budget, proposals=None, stage="teacher"):
    """A fresh stream owns every predicted state and exact reference cache."""
    started = time.perf_counter()
    from .route_candidate_evaluator import NominalCandidateEvaluator
    from .continuous_route_optimizer import optimize
    from .route_initializers import FrozenSeedInitializer
    run = Path(run); verify_run(run)
    path = stream_path(run, frozen, stream_id, stage)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
        target = path / name
        if target.exists():
            if sha(target) != sha(run / name):
                raise ValueError("stream identity mismatch")
        else:
            shutil.copyfile(run / name, target)
    task, prefs = load_task(run, frozen)
    BudgetLedger(run).reserve(stage + "_candidate_slots", frozen["task_id"] + ":" + stream_id, budget,
        {"maximum_stream_slots": True, "task_sha256": task.sha256()})
    complete = path / "planning_cost.json"
    if complete.exists():
        initializer = FrozenSeedInitializer(proposals) if proposals is not None else None
        evaluator = NominalCandidateEvaluator(path, task, frozen)
        result = optimize(task, prefs, evaluator.execution_identity, evaluator,
            path / "planning" / task.task_id, search_spec=SearchSpec(candidate_budget=budget), initializer=initializer)
        if sha(path / "planning" / task.task_id / "selection.json") != read(complete)["selection_sha256"]:
            raise ValueError("retained selection changed")
        if result["budget"]["stop_reason"] == "TOOL_ERROR":
            raise RuntimeError("retained technical search failure; no physical retry")
        return {"path": str(path), "selection": result}
    if (path / "planning" / task.task_id / "selection.json").exists():
        raise RuntimeError("selection exists without original planning timer; technical incomplete, no replacement timing")
    initializer = FrozenSeedInitializer(proposals) if proposals is not None else None
    evaluator = NominalCandidateEvaluator(path, task, frozen)
    result = optimize(task, prefs, evaluator.execution_identity, evaluator,
        path / "planning" / task.task_id, search_spec=SearchSpec(candidate_budget=budget), initializer=initializer)
    setup = read(path / "initializer_proposals.json").get("timing", {}) if (path / "initializer_proposals.json").exists() else {}
    elapsed = time.perf_counter() - started
    write(complete, {"task_id": task.task_id, "stream_id": stream_id, "stage": stage,
        "end_to_end_cold_planning_s": elapsed + setup.get("total_setup_s", 0.),
        "search_evaluator_selection_s": elapsed, "setup": setup, "measured_mode": "cold",
        "warm_latency": None, "selection_sha256": sha(path / "planning" / task.task_id / "selection.json"),
        "selection_sealed_utc": now(), "cross_stream_cache": False,
        "physics_from_fresh_task_state": True})
    if result["budget"]["stop_reason"] == "TOOL_ERROR":
        raise RuntimeError("consumed candidate tool failure retained; no automatic physical retry")
    return {"path": str(path), "selection": result}


def _initializers(run, frozen, stream_id, stage):
    from .route_initializers import RetrievalInitializer, json_raw
    from .search_effect_teacher import load_search_aware_dataset
    run = Path(run); path = stream_path(run, frozen, stream_id, stage)
    path.mkdir(parents=True, exist_ok=True)
    retained = path / "initializer_proposals.json"
    if retained.exists():
        value = read(retained)
        body = {k: v for k, v in value.items() if k != "content_sha256"}
        if digest(body) != value.get("content_sha256") or value["task_sha256"] != frozen["task_sha256"]:
            raise ValueError("retained initializer content/task changed")
        if value.get("dataset_manifest_sha256") != sha(run / "dataset" / "manifest.json"):
            raise ValueError("retained initializer TRAIN pool changed")
        if value.get("checkpoint") and sha(value["checkpoint"]) != value["checkpoint_sha256"]:
            raise ValueError("retained initializer checkpoint changed")
        if stage == "test" and stream_id in ("S", "D"):
            if value["checkpoint"] != verify_model_freeze(run)["selected_checkpoints"][stream_id]:
                raise ValueError("retained TEST initializer uses a different selected checkpoint")
        return {int(k): v for k, v in value["proposals"].items()}
    begun = time.perf_counter(); task, _ = load_task(run, frozen)
    if (path / "initializer_started.json").exists():
        raise RuntimeError("unfinished initializer consumed; no silent regeneration")
    write(path / "initializer_started.json", {"task_sha256": task.sha256(), "stream_id": stream_id,
        "stage": stage, "started_utc": now()})
    if stream_id == "N":
        ds = load_search_aware_dataset(run / "dataset")
        provider = RetrievalInitializer(ds.samples, ds.condition_scaler,
            identity={"dataset_manifest_sha256": sha(ds.manifest_path)})
        seeds = provider(task)
        checkpoint = None
    else:
        from .simple_warmstart_regression import SearchAwareSampler
        model = stream_id[0]
        if stage == "val":
            update = int(stream_id[1:])
            checkpoint = run / "models" / model / f"checkpoint_{update:04d}.pt"
        else:
            freeze = verify_model_freeze(run)
            checkpoint = Path(freeze["selected_checkpoints"][model])
        if model == "D":
            BudgetLedger(run).reserve(stage + "_ddim_samples", task.task_id + ":" + stream_id, 2,
                {"noise_seed": SEEDS["val_noise" if stage == "val" else "test_noise"], "DDIM_steps": 20})
        sampler = SearchAwareSampler(checkpoint, device="cpu")
        seeds = sampler.initializer_proposals(task, noise_seed=SEEDS["val_noise" if stage == "val" else "test_noise"])
    value = {"task_sha256": task.sha256(), "stream_id": stream_id, "stage": stage,
        "dataset_manifest_sha256": sha(run / "dataset" / "manifest.json"),
        "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
        "proposals": {str(k): v for k, v in seeds.items()},
        "timing": {"total_setup_s": time.perf_counter() - begun}, "generated_once": True,
        "candidate_quality_read": False}
    value = json_raw(value)
    value["content_sha256"] = digest(value)
    write(retained, value)
    return seeds


def _selection_entry(run, frozen, stream_id, endpoint, stage, prefix=None):
    path = stream_path(run, frozen, stream_id, stage)
    directory = path / "planning" / frozen["task_id"]
    return {"task_id": frozen["task_id"], "endpoint": endpoint,
        "selection_path": str(directory / (f"prefix_{prefix:02d}.json" if prefix else "selection.json")),
        "candidate_registry_path": str(directory / "candidate_registry.json"),
        "proposals_path": str(directory / "proposals.json"),
        "planning_cost_path": str(path / "planning_cost.json")}


def run_formal_stream(run, frozen, stream_id, stage):
    """Sequential fresh processes record startup and complete request timing."""
    run = Path(run).resolve(); path = stream_path(run, frozen, stream_id, stage)
    receipt = path / "outer_process.json"
    if receipt.exists():
        retained = read(receipt)
        if retained["exit_code"] != 0:
            raise RuntimeError("retained worker failure; no automatic physical retry")
        seeds = None if stream_id == "R" else _initializers(run, frozen, stream_id, stage)
        return run_stream(run, frozen, stream_id, 12 if stream_id == "R" else 8, seeds, stage)
    if (path / "planning_cost.json").exists():
        raise RuntimeError("completed request lacks outer timer; technical incomplete")
    path.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-X", "utf8", "-m", "v6_4.search_aware_warmstart_experiment", "_worker",
        "--run", str(run), "--stage", stage, "--task", frozen["task_id"], "--stream", stream_id]
    started = now(); before = time.perf_counter()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with (path / "command.log").open("a", encoding="utf8") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    write(receipt, {"argv": command, "started_utc": started, "ended_utc": now(),
        "elapsed_wall_s": time.perf_counter() - before, "exit_code": completed.returncode,
        "load": "sequential fresh interpreter, cold model weights; no concurrent heavy workloads"})
    if completed.returncode:
        raise RuntimeError("formal request tool failure retained: " + str(path / "command.log"))
    return {"path": str(path), "selection": read(path / "planning" / frozen["task_id"] / "selection.json")}


def closed_loop_val(run):
    from .closed_loop_warmstart_validation import freeze_phase_selections, execute_frozen_task, score_checkpoints
    run = Path(run); verify_run(run)
    phase = run / "closed_loop_val"
    if (phase / "model_selection.json").exists():
        return score_checkpoints(phase)
    entries = []
    order = ["R12", "D250", "S250", "D4000", "S4000"]
    for index, frozen in enumerate(task_rows(run, "val")):
        for endpoint in order[index:] + order[:index]:
            stream_id = "R" if endpoint == "R12" else endpoint
            run_formal_stream(run, frozen, stream_id, "val")
            entries.append(_selection_entry(run, frozen, stream_id, endpoint, "val"))
    files = {name: str(run / "models" / name[0] / f"checkpoint_{int(name[1:]):04d}.pt") for name in CHECKPOINTS}
    freeze_phase_selections(phase, entries, phase="VAL", checkpoint_files=files,
        bindings={"source_identity_sha256": sha(run / "source_identity.json")})
    for frozen in task_rows(run, "val"):
        task, _ = load_task(run, frozen)
        BudgetLedger(run).reserve("val_actual_slots", task.task_id, 10, {"endpoints": order})
        execute_frozen_task(phase, task, frozen, execution_run=run, endpoints=order)
    result = score_checkpoints(phase)
    if not (run / "model_selection.json").exists():
        write(run / "model_selection.json", result)
    return result


def freeze_models(run):
    from .closed_loop_warmstart_validation import score_checkpoints
    run = Path(run); verify_run(run)
    target = run / "model_freeze.json"
    if target.exists():
        return verify_model_freeze(run)
    selected = read(run / "model_selection.json")
    if selected != score_checkpoints(run / "closed_loop_val"):
        raise ValueError("root selection differs from sealed closed-loop VAL")
    # The closed-loop scorer is the sole authority for selecting updates.
    chosen = selected["selected_checkpoint_files"]
    artifacts = {}
    for directory in (run / "models", run / "dataset"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                artifacts[str(path.resolve())] = sha(path)
    for path in (run / "plan.json", run / "learning_split_manifest.json", run / "model_selection.json"):
        artifacts[str(path.resolve())] = sha(path)
    value = {"schema": "v64_c3_frozen_models_and_retrieval_v1", "selected_checkpoints": chosen,
        "artifacts": artifacts, "seeds": SEEDS, "frozen_utc": now(),
        "retrieval": "TRAIN-only complete standardized C.2 condition; exact family/preference/mask; original tie",
        "test_search_started": False, "deployment": "NOT_MET"}
    write(target, value)
    return value


def verify_model_freeze(run):
    frozen = read(Path(run) / "model_freeze.json")
    for path, expected in frozen["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("model/data/config changed after freeze: " + path)
    return frozen


def test_search(run):
    from .closed_loop_warmstart_validation import freeze_phase_selections
    run = Path(run); verify_run(run); verify_model_freeze(run)
    entries = []
    for index, frozen in enumerate(task_rows(run, "test")):
        order = list(METHODS[index:] + METHODS[:index])
        for method in order:
            run_formal_stream(run, frozen, method, "test")
        entries.append(_selection_entry(run, frozen, "R", "R8", "test", prefix=8))
        entries.append(_selection_entry(run, frozen, "R", "R12", "test"))
        entries.extend(_selection_entry(run, frozen, m, m + "8", "test") for m in ("N", "S", "D"))
    return freeze_phase_selections(run / "frozen_test", entries, phase="TEST", expected_endpoints=list(ENDPOINTS),
        bindings={"source_identity_sha256": sha(run / "source_identity.json"), "model_freeze_sha256": sha(run / "model_freeze.json")})


def execute_test(run):
    from .closed_loop_warmstart_validation import execute_frozen_task, verify_phase_selections
    run = Path(run); verify_run(run); verify_model_freeze(run)
    phase = run / "frozen_test"; verify_phase_selections(phase)
    slots = []
    for frozen in task_rows(run, "test"):
        task, _ = load_task(run, frozen)
        BudgetLedger(run).reserve("test_actual_slots", task.task_id, 10, {"endpoints": list(ENDPOINTS)})
        slots.extend(execute_frozen_task(phase, task, frozen, execution_run=run, endpoints=list(ENDPOINTS)))
    target = run / "actual_complete.json"
    if not target.exists():
        write(target, {"logical_slots": len(slots), "unique_actual": sum(s.get("unique_run", False) for s in slots),
            "slot_hashes": {p.relative_to(run).as_posix(): sha(p) for p in sorted((phase / "actual").glob("*/*/slot.json"))},
            "ended_utc": now()})
    return slots


def train(run):
    from .simple_warmstart_regression import prepare_training_pair, train_model_pair
    run = Path(run); verify_run(run)
    for model in ("D", "S"):
        BudgetLedger(run).reserve(model + "_training_runs", model, 1, {"updates": 4000, "batch": 32})
    if not (run / "models" / "training_config.json").exists():
        prepare_training_pair(run / "dataset", run / "models", device="cpu")
    return train_model_pair(run / "models")


def validate(run):
    from .closed_loop_warmstart_validation import verify_phase_selections, verify_actual_slot
    run = Path(run); verify_run(run); verify_model_freeze(run)
    for phase in (run / "closed_loop_val", run / "frozen_test"):
        verify_phase_selections(phase)
    registries = {stage: [row for p in (run / directory).glob("**/candidate_registry.json") for row in read(p)]
        for stage, directory in (("teacher", "teacher_search"), ("val", "closed_loop_val/search"), ("test", "test_search"))}
    slots = {stage: [read(p) for p in (run / directory).glob("actual/*/*/slot.json")]
        for stage, directory in (("val", "closed_loop_val"), ("test", "frozen_test"))}
    for directory in ("closed_loop_val", "frozen_test"):
        for path in (run / directory).glob("actual/*/*/slot.json"):
            verify_actual_slot(path.parent)
    errors = [r for rows in [*registries.values(), *slots.values()] for r in rows if r.get("tool_error")]
    totals = BudgetLedger(run).totals()
    physics = sum(r.get("prediction_steps", 0) for rows in registries.values() for r in rows)
    actual_physics = sum(r.get("actual_steps", 0) for rows in slots.values() for r in rows if r.get("unique_run"))
    if physics > 4428000 or actual_physics > 810000 or len(slots["val"]) != 20 or len(slots["test"]) != 40:
        raise ValueError("C.3 terminal counts/main physics budget violation")
    value = {"all_terminal": not errors, "candidate_slots": {k: len(v) for k, v in registries.items()},
        "actual_logical_slots": {k: len(v) for k, v in slots.items()}, "budget_reservations": totals,
        "main_prediction_physics_steps": physics, "main_actual_physics_steps": actual_physics,
        "tool_error_count": len(errors), "independent_acceptance": "inherited original five gates",
        "DATA_LIMITED": True, "deployment": "NOT_MET"}
    target = run / "validation" / "protocol_validation.json"
    if not target.exists():
        write(target, value)
    return value


def report(run):
    """A reporting-only helper is separate from the frozen execution source."""
    from .visualization.report_search_aware_warmstart import build_report
    return build_report(Path(run))


def run_all(run):
    from . import search_effect_teacher as teacher
    run = Path(run); verify_run(run)
    teacher.import_history(run)
    teacher.freeze_teacher_pairs(run)
    teacher.run_teacher_search(run)
    teacher.build_dataset(run)
    train(run)
    closed_loop_val(run)
    freeze_models(run)
    test_search(run)
    execute_test(run)
    validate(run)
    return report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "import-history", "teacher-search", "build-dataset", "train",
        "closed-loop-val", "freeze-models", "test-search", "execute-test", "validate", "report", "run-all", "_worker"))
    parser.add_argument("--output"); parser.add_argument("--run")
    parser.add_argument("--stage", choices=("val", "test")); parser.add_argument("--task"); parser.add_argument("--stream")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if args.command == "_worker":
        if not all((args.run, args.stage, args.task, args.stream)):
            parser.error("worker requires run/stage/task/stream")
        frozen = next(t for t in task_rows(args.run, args.stage) if t["task_id"] == args.task)
        if args.stage == "test":
            verify_model_freeze(args.run)
        proposals = None if args.stream == "R" else _initializers(args.run, frozen, args.stream, args.stage)
        value = run_stream(args.run, frozen, args.stream, 12 if args.stream == "R" else 8, proposals, args.stage)
    elif args.command == "prepare":
        if not args.output:
            parser.error("prepare requires --output")
        value = prepare(args.output)
    else:
        if not args.run:
            parser.error("command requires --run")
        from . import search_effect_teacher as teacher
        actions = {"import-history": teacher.import_history, "teacher-search": teacher.run_teacher_search,
            "build-dataset": teacher.build_dataset, "train": train, "closed-loop-val": closed_loop_val,
            "freeze-models": freeze_models, "test-search": test_search, "execute-test": execute_test,
            "validate": validate, "report": report, "run-all": run_all}
        value = actions[args.command](Path(args.run).resolve())
    print(json.dumps({"event": "C3_STAGE_COMPLETED", "stage": args.command, "run": args.run or args.output}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
