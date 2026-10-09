"""Frozen C.1 task, preference, search and identity contracts (no rollouts)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

from .task_protocol import TaskSpec, canonical_json, task_from_scenario
from .task_anchored_reference import (TaskAnchoredResidualPlan, build_reference_definition,
    REPRESENTATION_VERSION, REPRESENTATION_VERSION_V2, _target)

ROOT = Path(__file__).resolve().parents[1]
BASE = "cd288d7c2db74dc938903332400c7e97dd56bca8"
BRANCH = "v6.4-c1-continuous-route-optimizer"
HISTORY = ROOT / "v6_4/releases/execution_aware_route_teacher_20261007_01"
KEY_SLOT = 2
METHODS = ("Z0", "G0", "OI", "OC")
VERSIONS = {"v1": REPRESENTATION_VERSION, "v2": REPRESENTATION_VERSION_V2}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf8").strip()


@dataclass(frozen=True)
class SearchSpec:
    candidate_budget: int = 12
    proposal_limit: int = 128
    step_ladder_m: tuple = (.005, .0025, .00125)
    tie_band_rad_s: float = .001
    # One shared cursor visits all coordinates before a discrete family poll.
    coordinate_sign_order: tuple = ((0, 1), (0, -1), (1, 1), (1, -1),
                                    (2, 1), (2, -1), (3, 1), (3, -1))

    def __post_init__(self):
        if not 4 <= self.candidate_budget <= 12 or not 1 <= self.proposal_limit <= 128:
            raise ValueError("C.1 shared budget must be 4..12 and proposal limit 1..128")
        if self.step_ladder_m != (.005, .0025, .00125) or self.tie_band_rad_s != .001:
            raise ValueError("frozen step ladder/tie band differs")
        if self.coordinate_sign_order != SearchSpec.__dataclass_fields__["coordinate_sign_order"].default:
            raise ValueError("frozen coordinate order differs")


@dataclass(frozen=True)
class PreferenceSpec:
    name: str
    task_sha256: str
    W_key: tuple
    W_support: tuple
    route_obstacle_id: str
    related_pair_ids: tuple
    clearance_m: float = .030

    def __post_init__(self):
        if self.name not in ("A", "B") or self.clearance_m != .030:
            raise ValueError("only frozen A/B preferences with 30mm quality threshold are supported")
        if not self.W_key or not self.W_support or not self.related_pair_ids:
            raise ValueError("explicit frozen windows and original related pair IDs required")

    def to_dict(self):
        return asdict(self)


def active_intervals(definition):
    mask = definition["interval_mask"]
    if not mask[KEY_SLOT]:
        return []
    previous = [i for i in range(KEY_SLOT) if mask[i]]
    return ([previous[-1]] if previous else []) + [KEY_SLOT]


def parameter_plan(task, family, x):
    definition = build_reference_definition(task, version=VERSIONS[family])
    active = active_intervals(definition)
    if not active:
        raise ValueError("NOT_APPLICABLE: key slot 2 is disabled; no replacement")
    x = np.asarray(x, dtype=float)
    if x.shape != (2 * len(active),) or not np.isfinite(x).all():
        raise ValueError("wrong active parameter dimension/finiteness")
    z = np.zeros((6, 2))
    z[active] = x.reshape(-1, 2)
    z[z == 0.] = 0.  # Canonicalize signed zero before exact content hashing.
    # The only zero representation is v1, even after a discrete family poll.
    if not np.any(z) and family != "v1":
        definition = build_reference_definition(task, version=VERSIONS["v1"])
    return TaskAnchoredResidualPlan.from_definition(definition, z)


def project_disks(x):
    values = np.asarray(x, dtype=float).reshape(-1, 2).copy()
    for row in values:
        norm = np.linalg.norm(row)
        if norm > .020:
            row *= np.nextafter(.020, 0.) / norm
    return values.reshape(-1)


def geometry_direction(task):
    d = build_reference_definition(task)
    point = _target(task).sample(float(np.mean(d["intervals_s"][KEY_SLOT])))[0]
    center = np.asarray(task.scenario["workspace_obstacles"][1]["center_w"])
    direction = np.asarray(d["transverse_bases"][KEY_SLOT]).T @ (point - center)
    norm = float(np.linalg.norm(direction))
    degenerate = norm <= 1e-12
    return (np.array([1., 0.]) if degenerate else direction / norm), degenerate


def initial_candidates(task):
    d = build_reference_definition(task)
    active = active_intervals(d)
    if not active:
        return []
    away, degenerate = geometry_direction(task)
    result = []
    for family, amplitude in (("v1", 0.), ("v1", .012), ("v2", .012), ("v2", .020)):
        x = np.zeros(2 * len(active)); x[-2:] = away * amplitude
        # A unit direction times 20 mm can exceed the disk by one float ULP.
        # Use the existing inward proposal projection; keep the hard bound exact.
        x = project_disks(x)
        result.append({"family": family, "x_m": x.tolist(), "source": "initial",
                       "geometric_direction_degenerate": degenerate})
    return result


def related_pairs(spec, task):
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
    from .reference_adapter import scenario_from_task
    verifier = WholeBodyCollisionVerifier(spec, scenario_from_task(task).obstacles,
        WholeBodyVerificationConfig(minimum_clearance=.005, query_distance_max=2.5,
            adaptive_subdivisions=4, self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True))
    bodies = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
    obstacle = task.scenario["workspace_obstacles"][1]["name"]
    name = f"v5_workspace_sphere_001_{obstacle}"
    pairs = [p for p in verifier.pairs if p.pair_class == "arm_obstacle"
             and p.geom_b_name == name and int(verifier.model.geom_bodyid[p.geom_a]) in bodies]
    if not pairs:
        raise ValueError("frozen related pair scope is empty")
    return verifier, pairs


def next_unused_mother_seeds(used_seeds):
    result = []; index = 0
    while len(result) < 2:
        seed = 2026100701 + 104729 * index
        if seed not in used_seeds:
            result.append((index, seed))
        index += 1
    return result


def freeze_tasks(output):
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec, build_scenarios, V6LiteRunConfig
    from .route_pair_protocol import _geometry_check, validate_pair_contract
    spec = default_v6_lite_robot_spec()
    # Source generator identity is checked against the old development seed and
    # canonical source scene, before any outcome is read. No outcome seed search.
    old = read(HISTORY / "snapshot/task_manifest.json")
    old_seeds = {TaskSpec.from_dict(read(HISTORY / "snapshot/tasks" / row["task_id"] / "task.json")).seed
                 for row in old["tasks"]}
    b2 = ROOT / "v6_4/releases/task_anchored_residual_20261007_01/snapshot/tasks.json"
    old_seeds.update(TaskSpec.from_dict(t).seed for t in read(b2)["tasks"])
    seed_choices = next_unused_mother_seeds(old_seeds)
    mothers = [build_scenarios(spec, V6LiteRunConfig(scenario_count=1, seed=seed))[0]
               for _, seed in seed_choices]
    rows = []
    for index, mother in enumerate(mothers):
        if mother.seed in old_seeds:
            raise ValueError("source identity repeats old development; protocol must be frozen again before any rollout")
        group = f"c1_mother_{index:02d}"
        layout = {"mother_seed": mother.seed, "mother_source_sha256": digest(mother.to_dict()),
                  "next_unused_source_index": seed_choices[index][0],
                  "evaluation_role": "new_frozen_planning_evaluation", "sphere_offset_m": .055}
        base = task_from_scenario(spec, mother, task_id=group + "_source", group_id=group,
                                 family="end_effector_detour", split="test", layout_diagnostics=layout)
        d = build_reference_definition(base)
        # A disabled key is retained. The original base chord still declares a
        # canonical e1 only when legal; no alternate task is generated.
        active = active_intervals(d)
        midpoint = float(np.mean(d["intervals_s"][KEY_SLOT]))
        point = _target(base).sample(midpoint)[0]
        basis = np.asarray(d["transverse_bases"][KEY_SLOT])
        pair = []
        for side, label in ((1, "plus"), (-1, "minus")):
            task_id = group + "_" + label
            obstacles = list(mother.obstacles)
            if obstacles[1].radius != .025:
                raise ValueError("route sphere radius changed")
            obstacles[1] = replace(obstacles[1], center=point + side * .055 * basis[:, 0])
            scene = replace(mother, scenario_id=task_id, obstacles=tuple(obstacles))
            task = task_from_scenario(spec, scene, task_id=task_id, group_id=group,
                family="end_effector_detour", split="test", requirements=base.requirements,
                layout_diagnostics=layout)
            directory = output / "frozen_tasks" / task_id
            write(directory / "task.json", task.to_dict())
            definition = build_reference_definition(task)
            check = _geometry_check(spec, task, definition)
            write(directory / "geometry_precheck.json", check)
            verifier, pairs = related_pairs(spec, task)
            pair_ids = tuple((p.geom_a_name, p.geom_b_name) for p in pairs)
            windows = [definition["intervals_s"][i] for i in active]
            preferences = [PreferenceSpec(name, task.sha256(), (tuple(definition["intervals_s"][2]),),
                tuple(tuple(w) for w in windows), task.scenario["workspace_obstacles"][1]["name"],
                pair_ids).to_dict() for name in ("A", "B")] if active else []
            write(directory / "preferences.json", preferences)
            rows.append({"task_id": task_id, "group_id": group, "seed": mother.seed,
                "task_sha256": task.sha256(), "mother_source_sha256": digest(mother.to_dict()),
                "old_mother_seeds": sorted(old_seeds), "mother_identity_new": True,
                "next_unused_source_index": seed_choices[index][0],
                "active_intervals": active, "active_dim": len(active) * 2,
                "W_key": [definition["intervals_s"][2]], "W_support": windows, "W_full": [[0., 27.]],
                "obstacle_name": task.scenario["workspace_obstacles"][1]["name"],
                "related_pair_ids": pair_ids, "pair_policy_sha256": verifier._pair_policy_sha256,
                "geometry_precheck_passed": check["passed"],
                "initial_geometry_query_count": check["native_distance_queries"],
                "applicable": bool(active)})
            pair.append(task)
        validate_pair_contract(*pair)
    return rows


def prepare(output):
    started = datetime.now(timezone.utc).isoformat()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("exclusive run root required")
    if git("status", "--porcelain"):
        raise ValueError("commit source before freezing the actual producer")
    if git("merge-base", BASE, "HEAD") != BASE:
        raise ValueError("fixed development base is not an ancestor")
    output.mkdir(parents=True)
    shutil.copyfile(HISTORY / "snapshot/frozen_execution_config.json", output / "frozen_execution_config.json")
    inherited_metadata = HISTORY / "snapshot/attempts/EA_03/actual/run_metadata.json"
    write(output / "frozen_run_config.json", read(inherited_metadata)["run_config"])
    tasks = freeze_tasks(output)
    plan = {"schema": "v64_c1_frozen_protocol_v1", "run_id": output.name,
        "base_publication_commit": BASE, "historical_actual_producer": "d4464c8ae2aa7913a730ebbd775917e7a3b1af71",
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks, "search": asdict(SearchSpec()), "methods": METHODS,
        "candidate_slots_max": 48, "actual_method_slots_max": 16,
        "mother_seed_rule": "2026100701 + 104729 * source_index; take first two unused against ALL published B2/B3/B31 seeds before any outcome; no outcome replacement",
        "projection_rule": "separate 20mm closed disks; inward floating roundoff for projected points",
        "poll_rule": "shared coordinate +/- cursor then family switch; alternate A/B center; shrink only after full no-improvement sweep",
        "window_sampling": "same frozen closed union as inherited route metrics; protected gaps are excluded",
        "training_runs": 0, "neural_sampling_calls": 0, "optimizer_updates": 0,
        "deployment": "NOT_MET", "continuous_time_safety": "NOT_ESTABLISHED"}
    write(output / "plan.json", plan)
    write(output / "prepare_receipt.json", {"argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
        "status": "COMPLETED", "physics_steps": 0})
    names = [p for directory in ("v6_4", "v6_lite", "model_test")
             for p in (ROOT / directory).rglob("*.py")
             if not any(x in ("output", "releases", "visualization", "__pycache__") for x in p.relative_to(ROOT).parts)]
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    names.extend(default_v6_lite_robot_spec()._source_assets())
    names = sorted({p.resolve() for p in names})
    protected = {str(p.resolve()): sha(p) for p in output.rglob("*") if p.is_file()}
    identity = {"schema": "v64_c1_source_identity_v1", "git_head": git("rev-parse", "HEAD"),
        "algorithm_producer_commit": git("rev-parse", "HEAD"), "base_publication_commit": BASE,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in names},
        "protected_artifacts": protected, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "argv": sys.argv, "cwd": str(ROOT),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTHONPATH")},
        "run_config_source": {"relative_path": inherited_metadata.relative_to(ROOT).as_posix(), "sha256": sha(inherited_metadata)},
        "frozen_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "source_identity.json", identity)
    return plan


def verify_frozen(run):
    from .residual_execution import source_guard
    run = Path(run)
    source_guard(run / "source_identity.json")
    return read(run / "plan.json")
