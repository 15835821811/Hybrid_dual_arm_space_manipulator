"""Frozen C.2 split, finite budgets and declaration-only preparation.

Historical TaskSpec bytes and SHA identities are never rewritten.  The external
learning manifest alone assigns their role in this new study.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from .route_optimizer_protocol import (ROOT, SearchSpec, PreferenceSpec, VERSIONS,
    active_intervals, digest, git, read, related_pairs, sha, write)
from .task_anchored_reference import build_reference_definition, _target
from .task_protocol import TaskSpec, task_from_scenario

BASE = "bad70fdec9cd97e004e02afbfb301e21c91c8335"
C1_PRODUCER = "6f603437b300de882d24234130b6669b00f62bd6"
BRANCH = "v6.4-c2-preference-diffusion-warmstart"
HISTORY = ROOT / "v6_4/releases/continuous_route_optimizer_20261007_01"
ROLES = ("new_train", "new_val", "test0", "test1")
ROLE_SPLITS = {"new_train": "train", "new_val": "val", "test0": "test", "test1": "test"}
SEEDS = {"train": 64221, "training_draw": 64222, "validation": 64223, "test_noise": 64224}
BUDGETS = {"new_teacher_prediction_slots": 48, "test_prediction_slots": 112,
    "total_new_prediction_slots": 160, "actual_logical_slots": 32,
    "formal_training_runs": 1, "optimizer_updates": 4000,
    "validation_ddim_samples": 128, "test_ddim_samples": 16,
    "smoke_ddim_samples": 8, "total_ddim_samples": 152}


def search_interval_mask(definition):
    result = np.zeros(6, dtype=bool)
    result[active_intervals(definition)] = True
    return result


def next_unused_seeds(used, count=4):
    """Choose solely by source identity, never by geometry or rollout outcome."""
    choices = []
    for index in range(100000):
        seed = 2026100701 + 104729 * index
        if seed not in set(used):
            choices.append((index, seed))
            if len(choices) == count:
                return choices
    raise ValueError("seed identity sequence exhausted")


def historical_tasks():
    snapshot = HISTORY / "snapshot"
    plan = read(snapshot / "plan.json")
    result = []
    for row in plan["tasks"]:
        path = snapshot / "frozen_tasks" / row["task_id"] / "task.json"
        task = TaskSpec.from_dict(read(path))
        if task.sha256() != row["task_sha256"]:
            raise ValueError("historical Task SHA mismatch")
        result.append((task, dict(row), path))
    return result


def used_mother_seeds():
    result = {t.seed for t, _, _ in historical_tasks()}
    # Every published B.2/B.3/B.3.1 declaration belongs to development history.
    releases = ROOT / "v6_4/releases"
    for path in releases.glob("*/snapshot/tasks.json"):
        value = read(path)
        result.update(TaskSpec.from_dict(t).seed for t in value.get("tasks", []))
    for path in releases.glob("*/snapshot/task_manifest.json"):
        for row in read(path).get("tasks", []):
            task_path = path.parent / "tasks" / row["task_id"] / "task.json"
            if task_path.exists():
                result.add(TaskSpec.from_dict(read(task_path)).seed)
    return result


def validate_learning_splits(manifest, tasks=None):
    """External Task SHA and mother identity are authoritative, not task names."""
    by_sha, mothers = {}, {}
    for row in manifest["tasks"]:
        key, group, split = row["task_sha256"], row["mother_id"], row["split"]
        if split not in ("train", "val", "test") or key in by_sha:
            raise ValueError("invalid or repeated external split identity")
        by_sha[key] = row
        identity = row["mother_source_sha256"]
        for group_key in (group, identity):
            if group_key in mothers and mothers[group_key] != split:
                raise ValueError("mother/mirror candidates cross external splits")
            mothers[group_key] = split
        if row.get("historical") and split != "train":
            raise ValueError("historical C.1 tasks are TRAIN/development only")
    if tasks is not None:
        for task in tasks:
            if task.sha256() not in by_sha:
                raise ValueError("task absent from external SHA split manifest")
    return by_sha


def freeze_tasks(output):
    """Reuse C.1 generator, sphere offset and analytic/static checks; no steps."""
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec, build_scenarios, V6LiteRunConfig
    from .route_pair_protocol import _geometry_check, validate_pair_contract
    output = Path(output)
    spec = default_v6_lite_robot_spec()
    used = used_mother_seeds()
    choices = next_unused_seeds(used)
    rows = []
    for role, (source_index, seed) in zip(ROLES, choices):
        mother = build_scenarios(spec, V6LiteRunConfig(scenario_count=1, seed=seed))[0]
        group = "c2_" + role
        mother_sha = digest(mother.to_dict())
        layout = {"mother_seed": seed, "mother_source_sha256": mother_sha,
            "next_unused_source_index": source_index, "evaluation_role": role, "sphere_offset_m": .055}
        base = task_from_scenario(spec, mother, task_id=group + "_source", group_id=group,
            family="end_effector_detour", split=ROLE_SPLITS[role], layout_diagnostics=layout)
        definition = build_reference_definition(base)
        active = active_intervals(definition)
        point = _target(base).sample(float(np.mean(definition["intervals_s"][2])))[0]
        basis = np.asarray(definition["transverse_bases"][2])
        pair = []
        for side, label in ((1, "plus"), (-1, "minus")):
            tid = group + "_" + label
            obstacles = list(mother.obstacles)
            if obstacles[1].radius != .025:
                raise ValueError("frozen route sphere radius differs")
            obstacles[1] = replace(obstacles[1], center=point + side * .055 * basis[:, 0])
            scene = replace(mother, scenario_id=tid, obstacles=tuple(obstacles))
            task = task_from_scenario(spec, scene, task_id=tid, group_id=group,
                family="end_effector_detour", split=ROLE_SPLITS[role], requirements=base.requirements,
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
                "role": role, "split": ROLE_SPLITS[role], "seed": seed,
                "task_sha256": task.sha256(), "mother_source_sha256": mother_sha,
                "next_unused_source_index": source_index, "old_mother_seeds": sorted(used),
                "mother_identity_new": True, "historical": False,
                "active_intervals": active, "active_dim": 2 * len(active),
                "reference_interval_mask": d["interval_mask"],
                "search_interval_mask": search_interval_mask(d).tolist(),
                "W_key": [d["intervals_s"][2]], "W_support": windows, "W_full": [[0., 27.]],
                "obstacle_name": task.scenario["workspace_obstacles"][1]["name"],
                "related_pair_ids": pair_ids, "pair_policy_sha256": verifier._pair_policy_sha256,
                "geometry_precheck_passed": check["passed"],
                "initial_geometry_query_count": check["native_distance_queries"], "applicable": bool(active)})
            pair.append(task)
        validate_pair_contract(*pair)
    return rows


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        return verify_frozen(output)
    if git("status", "--porcelain"):
        raise ValueError("commit source before freezing the actual C.2 producer")
    if git("merge-base", BASE, "HEAD") != BASE:
        raise ValueError("fixed C.2 development base is not an ancestor")
    output.mkdir(parents=True)
    snapshot = HISTORY / "snapshot"
    for name in ("frozen_execution_config.json", "frozen_run_config.json"):
        shutil.copyfile(snapshot / name, output / name)
    tasks = freeze_tasks(output)
    splits = []
    for task, row, path in historical_tasks():
        directory = output / "frozen_tasks" / task.task_id
        directory.mkdir(parents=True)
        shutil.copyfile(path, directory / "task.json")
        shutil.copyfile(path.parent / "preferences.json", directory / "preferences.json")
        d = build_reference_definition(task)
        splits.append({**row, "mother_id": row["group_id"], "split": "train", "role": "historical_train",
            "historical": True, "original_task_declared_split": task.split,
            "original_task_file_sha256": sha(path), "task_path": f"frozen_tasks/{task.task_id}/task.json",
            "reference_interval_mask": d["interval_mask"],
            "search_interval_mask": search_interval_mask(d).tolist()})
    splits.extend({**t, "original_task_declared_split": t["split"],
        "task_path": f"frozen_tasks/{t['task_id']}/task.json"} for t in tasks)
    split_manifest = {"schema": "v64_c2_external_learning_split_v1", "split_unit": "mother_source_identity",
        "historical_TaskSpec_unchanged": True, "tasks": splits,
        "intended_mother_counts": {"train": 3, "val": 1, "test": 2},
        "intended_task_counts": {"train": 6, "val": 2, "test": 4}}
    validate_learning_splits(split_manifest)
    write(output / "learning_split_manifest.json", split_manifest)
    plan = {"schema": "v64_c2_preference_warmstart_protocol_v1", "run_id": output.name,
        "base_publication_commit": BASE, "historical_actual_producer": C1_PRODUCER,
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "tasks": tasks,
        "all_learning_tasks": splits, "methods": ["R", "N", "D"],
        "search_budgets": {"R": 12, "N": 8, "D": 8}, "search": asdict(SearchSpec()),
        "prefixes": [4, 8, 12], "actual_endpoints": ["R8", "R12", "N8", "D8"],
        "candidate_slots_max": 160, "actual_method_slots_max": 32,
        "budget_limits": BUDGETS, "seeds": SEEDS,
        "mother_seed_rule": "2026100701 + 104729 * source_index; first four unused source identities; role order new_train,new_val,test0,test1; no outcome replacement",
        "label_rules": {"A_family_I_band_rad_s": .001, "B_clearance_m": .030, "B_family_L_band_m": .005},
        "model": {"hidden": [128, 128], "activation": "SiLU", "schedule": "cosine100",
            "target": "v_prediction", "DDIM_steps": 20, "batch": 32, "lr": 1e-4,
            "weight_decay": .01, "gradient_clip": 1., "optimizer_updates_max": 4000},
        "checkpoint_rule": ["VAL_analytic_legal_count_desc", "Task_equal_nearest_reference_RMS_asc",
            "VAL_v_MSE_asc", "update_asc"], "test_results_unavailable_before_model_freeze": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "deployment": "NOT_MET",
        "continuous_time_safety": "NOT_ESTABLISHED", "hardware_safety": "NOT_ESTABLISHED"}
    write(output / "plan.json", plan)
    names = [p for directory in ("v6_4", "v6_lite", "model_test") for p in (ROOT / directory).rglob("*.py")
        if not any(x in ("output", "releases", "visualization", "__pycache__") for x in p.relative_to(ROOT).parts)]
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    names.extend(default_v6_lite_robot_spec()._source_assets())
    identity = {"schema": "v64_c2_source_identity_v1", "git_head": git("rev-parse", "HEAD"),
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "base_publication_commit": BASE,
        "historical_actual_producer": C1_PRODUCER,
        "source_sha256": {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(names))},
        "protected_artifacts": {str(p.resolve()): sha(p) for p in output.rglob("*") if p.is_file()},
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "argv": sys.argv, "cwd": str(ROOT),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTHONPATH")},
        "frozen_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "source_identity.json", identity)
    write(output / "prepare_receipt.json", {"status": "COMPLETED", "physics_steps": 0,
        "tasks_defined_before_training": True, "test_results_computed": False})
    return plan


def verify_frozen(run):
    from .residual_execution import source_guard
    run = Path(run)
    source_guard(run / "source_identity.json")
    validate_learning_splits(read(run / "learning_split_manifest.json"))
    return read(run / "plan.json")


class BudgetLedger:
    """Exclusive per-unit reservations survive crashes and forbid silent retries.

    A batch DDIM call reserves its actual sample count. Reserving the same unit
    again returns the immutable receipt and never grants a second allowance.
    """
    def __init__(self, run):
        self.root = Path(run) / "budget_ledger"
        self.root.mkdir(parents=True, exist_ok=True)

    def entries(self):
        return [read(p) for p in sorted(self.root.glob("*.json"))]

    def totals(self):
        result = {key: 0 for key in BUDGETS}
        for row in self.entries():
            result[row["category"]] += row["count"]
        result["total_new_prediction_slots"] = result["new_teacher_prediction_slots"] + result["test_prediction_slots"]
        result["total_ddim_samples"] = sum(result[k] for k in ("validation_ddim_samples", "test_ddim_samples", "smoke_ddim_samples"))
        return result

    def reserve(self, category, unit_id, count=1, metadata=None):
        if category not in BUDGETS or category.startswith("total_") or type(count) is not int or count < 1:
            raise ValueError("invalid finite budget reservation")
        key = digest({"category": category, "unit_id": unit_id})
        path = self.root / (key + ".json")
        if path.exists():
            row = read(path)
            if row["count"] != count or row["metadata"] != (metadata or {}):
                raise ValueError("retained budget unit identity differs")
            return row
        # Atomic lock coordinates independent teacher/test workers on Windows.
        lock = self.root / ".reserve.lock"
        deadline = time.monotonic() + 30.
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError("budget reservation busy; retry reservation without running physics") from error
                time.sleep(.02)
        try:
            if path.exists():
                row = read(path)
                if row["count"] != count or row["metadata"] != (metadata or {}):
                    raise ValueError("retained budget unit identity differs")
                return row
            totals = self.totals()
            totals[category] += count
            totals["total_new_prediction_slots"] = totals["new_teacher_prediction_slots"] + totals["test_prediction_slots"]
            totals["total_ddim_samples"] = sum(totals[k] for k in ("validation_ddim_samples", "test_ddim_samples", "smoke_ddim_samples"))
            if any(totals[k] > limit for k, limit in BUDGETS.items()):
                raise ValueError("predeclared finite experiment budget exhausted")
            row = {"category": category, "unit_id": unit_id, "count": count, "metadata": metadata or {},
                "status": "RESERVED_CONSUMED_NO_SILENT_RETRY", "utc": datetime.now(timezone.utc).isoformat()}
            write(path, row)
            return row
        finally:
            os.close(descriptor)
            lock.unlink()
