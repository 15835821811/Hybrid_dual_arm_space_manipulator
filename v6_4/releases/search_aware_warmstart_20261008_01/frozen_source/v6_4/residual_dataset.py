"""Frozen successful Cartesian residual labels and task-only conditioning.

This module never simulates, repairs a proposal, fits actual q, or changes z.
All distances and scalers are fitted from successful TRAIN declarations only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import canonical, object_sha, sha256
from .task_protocol import TaskSpec, validate_task_splits

DATASET_SCHEMA = "v6_4_successful_task_anchored_residual_dataset_v1"
CONDITION_SCHEMA = "v6_4_compact_declared_world_residual_condition_v1"
MAX_REQUIREMENTS, MAX_OBSTACLES, MAX_PROTECTED, MAX_PATH_POINTS = 16, 8, 16, 16
RESIDUAL_STD_FLOOR_M = .001


def _read(value):
    return json.loads(Path(value).read_text(encoding="utf-8-sig")) if isinstance(value, (str, Path)) else value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _task(value):
    return value if isinstance(value, TaskSpec) else TaskSpec.from_dict(value)


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


def validate_z(z_m, mask):
    z = _array(z_m, (6, 2), "z_m")
    mask = np.asarray(mask)
    if mask.shape != (6,) or mask.dtype != np.bool_:
        raise ValueError("interval_mask must be six booleans")
    if np.any(z[~mask] != 0):
        raise ValueError("inactive z must be exactly zero; no masking repairs input")
    if not mask.any():
        raise ValueError("no active interval")
    return z, mask


@dataclass
class ResidualNormalizer:
    mean_m: np.ndarray
    std_m: np.ndarray
    count_active_intervals: int

    @classmethod
    def fit(cls, z_m, masks):
        if len(z_m) == 0 or len(z_m) != len(masks):
            raise ValueError("successful TRAIN residuals are required")
        values = np.concatenate([validate_z(z, np.asarray(m, dtype=bool))[0][m] for z, m in zip(z_m, masks)])
        return cls(values.mean(0), np.maximum(values.std(0), RESIDUAL_STD_FLOOR_M), len(values))

    def normalize(self, z_m, mask):
        z, mask = validate_z(z_m, mask)
        result = np.zeros((6, 2), dtype=np.float64)
        result[mask] = (z[mask] - self.mean_m) / self.std_m
        return result

    def inverse(self, latent, mask):
        x, mask = validate_z(latent, mask)
        result = np.zeros((6, 2), dtype=np.float64)
        result[mask] = x[mask] * self.std_m + self.mean_m
        return result

    denormalize = inverse

    def to_dict(self):
        return {"schema": "v6_4_shared_transverse_axis_train_scaler_v1", "fit_split": "train",
                "axis_pooling": "all active intervals of successful TRAIN references",
                "mean_m": self.mean_m.tolist(), "std_m": self.std_m.tolist(),
                "std_floor_m": RESIDUAL_STD_FLOOR_M, "count_active_intervals": self.count_active_intervals}

    @classmethod
    def from_dict(cls, d):
        if d["schema"] != "v6_4_shared_transverse_axis_train_scaler_v1" or d["std_floor_m"] != RESIDUAL_STD_FLOOR_M or d["fit_split"] != "train":
            raise ValueError("unsupported residual scaler")
        mean, std = _array(d["mean_m"], (2,), "mean"), _array(d["std_m"], (2,), "std")
        if np.any(std < RESIDUAL_STD_FLOOR_M):
            raise ValueError("invalid residual scale")
        return cls(mean, std, int(d["count_active_intervals"]))


def encode_residual_condition(task, definition):
    """Pure declarations, world frame; identifiers and outcomes are excluded.

    Features carry an explicit valid mask and per-unit std floor. Fixed caps
    reject overflow. Types/presence flags remain literal, never standardized.
    """
    t = _task(task).to_dict()
    d = definition
    if d["task_id"] != t["task_id"] or d["task_sha256"] != object_sha(t):
        raise ValueError("definition/task binding mismatch")
    vals, valid, floors, categorical, names = [], [], [], [], []

    def add(name, value, unit="dimensionless", active=True, literal=False):
        a = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.isfinite(a).all():
            raise ValueError("nonfinite condition")
        floor = {"m": .01, "m/s": .01, "rad": .01, "rad/s": .01,
                 "s": 1., "dimensionless": .01}[unit]
        vals.extend(a if active else np.zeros_like(a))
        valid.extend([bool(active)] * len(a))
        floors.extend([floor] * len(a))
        categorical.extend([literal] * len(a))
        names.extend([f"{name}.{i}:{unit}" for i in range(len(a))])

    add("initial_planner_q", t["initial_planner_q"], "rad")
    add("initial_planner_dq", t["initial_planner_dq"], "rad/s")
    for key in ("base_pose", "target_pose"):
        add(key + ".position", t[key][:3], "m")
        add(key + ".wxyz", t[key][3:])
    for key in ("base_twist", "target_twist"):
        add(key + ".linear", t[key][:3], "m/s")
        add(key + ".angular", t[key][3:], "rad/s")
    s = t["scenario"]
    add("grasp_point_target_frame", s["grasp_point_target_frame_m"], "m")
    add("grasp_rotation_target_frame", s["grasp_rotation_target_frame"])
    add("continuum_target_rotation_world", s["continuum_target_rotation_world"])
    source = s["continuum_target"]
    if source.get("mode") != "irregular_waypoints" or source.get("reference_profile") != "minimum_jerk_c2":
        raise ValueError("unsupported declared base reference")
    add("base_path.initial_position", source["initial_position_w"], "m")
    add("base_path.target_rotation_world", source["target_rotation_world"])
    add("base_path.transition_and_duration", [source["transition_duration_s"], source["path_duration_s"]], "s")
    points, durations = source["waypoint_points_m"], source["segment_durations_s"]
    if len(points) > MAX_PATH_POINTS or len(durations) != len(points)-1:
        raise ValueError("base path capacity or segment count mismatch")
    for i in range(MAX_PATH_POINTS):
        active = i < len(points)
        add(f"base_path.{i}.present", [active], literal=True)
        add(f"base_path.{i}.position", points[i] if active else [0]*3, "m", active)
        segment = i < len(durations)
        add(f"base_path.{i}.segment_present", [segment], literal=True)
        add(f"base_path.{i}.segment_duration", [durations[i] if segment else 0], "s", segment)
    add("duration_periods", [t["duration_s"], t["task_period_s"], t["physics_period_s"]], "s")
    add("family", [t["family"] == x for x in ("end_effector_detour", "mid_arm_detour", "multiple_routes")], literal=True)
    add("free_path", [t["path_freedom"] == "free_intermediate_path_between_fixed_requirements"], literal=True)
    reqs = sorted(t["requirements"], key=lambda p: (p["time_s"], p["arm"], p["kind"], canonical({k:v for k,v in p.items() if k != "point_id"})))
    if len(reqs) > MAX_REQUIREMENTS:
        raise ValueError("too many requirements")
    for i in range(MAX_REQUIREMENTS):
        p = reqs[i] if i < len(reqs) else None
        active = p is not None
        p = p or dict(arm="", frame="", kind="", position_m=[0]*3, rotation=[0]*9,
                      time_s=0, time_window_s=[0]*2, position_tolerance_m=0, orientation_tolerance_rad=0)
        prefix = f"requirement.{i}"
        add(prefix+".present", [active], literal=True)
        add(prefix+".types", [p["arm"] == x for x in ("rigid", "continuum")] + [p["frame"] == x for x in ("world", "target")] + [p["kind"] == x for x in ("waypoint", "terminal")], active=active, literal=True)
        add(prefix+".position", p["position_m"], "m", active)
        add(prefix+".rotation", p["rotation"], active=active)
        add(prefix+".times", [p["time_s"], *p["time_window_s"]], "s", active)
        add(prefix+".position_tolerance", [p["position_tolerance_m"]], "m", active)
        add(prefix+".orientation_tolerance", [p["orientation_tolerance_rad"]], "rad", active)
    obstacles = []
    for o in s["workspace_obstacles"]:
        kind = o.get("type", "sphere")
        if kind not in ("sphere", "box"):
            raise ValueError("unsupported declared obstacle")
        size = [o["radius_m"]]*3 if kind == "sphere" else o["size_m"]
        if np.any(_array(size, (3,), "obstacle size") <= 0):
            raise ValueError("nonpositive obstacle size")
        row = [*o["center_w"], *(np.eye(3).flat if kind == "sphere" else o["rotation_world"]), *size,
               *o.get("linear_velocity_w_m_s", [0]*3), *o.get("angular_velocity_w_rad_s", [0]*3)]
        obstacles.append((kind, row))
    obstacles.sort(key=lambda x: (x[0], *x[1]))
    if len(obstacles) > MAX_OBSTACLES:
        raise ValueError("too many declared obstacles")
    for i in range(MAX_OBSTACLES):
        active = i < len(obstacles)
        kind, row = obstacles[i] if active else ("", [0]*21)
        prefix = f"obstacle.{i}"
        add(prefix+".present", [active], literal=True)
        add(prefix+".types", [kind == x for x in ("sphere", "box")], active=active, literal=True)
        for name, values, unit in (("center", row[:3], "m"), ("rotation", row[3:12], "dimensionless"),
                                   ("size", row[12:15], "m"), ("linear_velocity", row[15:18], "m/s"),
                                   ("angular_velocity", row[18:21], "rad/s")):
            add(prefix+"."+name, values, unit, active)
    mask = np.asarray(d["interval_mask"], dtype=bool)
    intervals = _array(d["intervals_s"], (6, 2), "intervals")
    bases = _array(d["transverse_bases"], (6, 3, 2), "bases")
    for i in range(6):
        add(f"interval.{i}.mask", [mask[i]], literal=True)
        add(f"interval.{i}.times", intervals[i], "s", mask[i])
        add(f"interval.{i}.world_basis", bases[i], active=mask[i])
    protected = d["protected_time_intervals_s"]
    if len(protected) > MAX_PROTECTED:
        raise ValueError("too many protected intervals")
    for i in range(MAX_PROTECTED):
        active = i < len(protected)
        add(f"protected.{i}.present", [active], literal=True)
        add(f"protected.{i}.times", protected[i] if active else [0, 0], "s", active)
    add("support_cutoff", [d["support_cutoff_s"]], "s")
    add("coefficient_bound", [d["coefficient_norm_bound_m"]], "m")
    return {"schema": CONDITION_SCHEMA, "values": np.asarray(vals), "valid": np.asarray(valid, dtype=bool),
            "literal": np.asarray(categorical, dtype=bool), "std_floors": np.asarray(floors), "names": names}


@dataclass
class ConditionNormalizer:
    mean: np.ndarray
    std: np.ndarray
    literal: np.ndarray
    floors: np.ndarray
    names: list
    fit_task_ids: list

    @classmethod
    def fit(cls, task_definition_pairs):
        pairs = sorted(task_definition_pairs, key=lambda x: _task(x[0]).task_id)
        ids = [_task(t).task_id for t, _ in pairs]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("condition scaler requires unique successful TRAIN tasks")
        if any(_task(t).split != "train" for t, _ in pairs):
            raise ValueError("condition scaler may only fit TRAIN")
        enc = [encode_residual_condition(t, d) for t, d in pairs]
        e = enc[0]
        if any(x["names"] != e["names"] for x in enc):
            raise ValueError("condition schema mismatch")
        values, valid = np.stack([x["values"] for x in enc]), np.stack([x["valid"] for x in enc])
        count = valid.sum(0)
        mean = np.divide((values*valid).sum(0), count, out=np.zeros(values.shape[1]), where=count>0)
        var = np.divide(((values-mean)**2*valid).sum(0), count, out=np.zeros_like(mean), where=count>0)
        std = np.maximum(np.sqrt(var), e["std_floors"])
        mean[e["literal"]], std[e["literal"]] = 0., 1.
        return cls(mean, std, e["literal"], e["std_floors"], e["names"], ids)

    def transform(self, task, definition):
        e = encode_residual_condition(task, definition)
        if e["names"] != self.names or not np.array_equal(e["literal"], self.literal):
            raise ValueError("condition schema mismatch")
        return np.where(e["valid"], (e["values"] - self.mean)/self.std, 0.).astype(np.float32)

    def to_dict(self):
        return {"schema": CONDITION_SCHEMA, "fit_split": "train", "fit_unique_task_ids": self.fit_task_ids,
                "frame": "world", "invalid_features": "exact zero, excluded from fit", "literal_types_and_masks": True,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "literal": self.literal.tolist(),
                "std_floors": self.floors.tolist(), "feature_names_with_units": self.names}

    @classmethod
    def from_dict(cls, d):
        if d["schema"] != CONDITION_SCHEMA or d["fit_split"] != "train":
            raise ValueError("unsupported condition scaler")
        n = len(d["feature_names_with_units"])
        mean, std, floors = [_array(d[k], (n,), k) for k in ("mean", "std", "std_floors")]
        literal = np.asarray(d["literal"], dtype=bool)
        if literal.shape != (n,) or np.any(std < floors) or np.any(mean[literal] != 0) or np.any(std[literal] != 1):
            raise ValueError("invalid condition scaler")
        return cls(mean, std, literal, floors, d["feature_names_with_units"], d["fit_unique_task_ids"])


@dataclass
class ResidualDataset:
    manifest_path: Path
    manifest: dict
    samples: list
    tasks: dict
    definitions: dict
    z_m: np.ndarray
    masks: np.ndarray

    def indices(self, split):
        return np.asarray([i for i, s in enumerate(self.samples) if s["split"] == split], dtype=np.int64)


def _source_path(parent, name):
    p = Path(name)
    return p if p.is_absolute() else parent / p


def freeze_residual_dataset(teacher_manifest, task_suite, definition_manifest, output_dir):
    """Copy small immutable labels, with strict independent full-success gates."""
    from .task_anchored_reference import TaskAnchoredResidualPlan
    output = Path(output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError("residual dataset already frozen")
    teacher, suite, definitions = _read(teacher_manifest), _read(task_suite), _read(definition_manifest)
    tasks = [_task(x) for x in suite["tasks"]]
    validate_task_splits(tasks)
    by_id = {t.task_id: t for t in tasks}
    if "task_hashes" in suite and suite["task_hashes"] != {t.task_id:t.sha256() for t in tasks}:
        raise ValueError("task suite hashes differ")
    definitions = definitions.get("definitions", definitions)
    if isinstance(definitions, list):
        definitions = {d["task_id"]: d for d in definitions}
    records = teacher["records"]
    if len(records) > 24:
        raise ValueError("teacher budget exceeded")
    parent = Path(teacher_manifest).parent if isinstance(teacher_manifest, (str, Path)) else Path.cwd()
    seen, eligible, excluded, sources = set(), [], [], []
    for record in records:
        r = dict(record)
        tid, ci = r["task_id"], r["candidate_index"]
        if tid not in by_id or by_id[tid].split not in ("train", "val") or r["split"] != by_id[tid].split:
            raise ValueError("teacher task/split binding mismatch or TEST leakage")
        if type(ci) is not int or ci not in range(3) or (tid, ci) in seen:
            raise ValueError("teacher slot duplicated or invalid")
        seen.add((tid, ci))
        evidence = {}
        for key in ("plan", "attempt", "evaluation"):
            path = _source_path(parent, r[key+"_path"])
            if sha256(path) != r[key+"_sha256"]:
                raise ValueError(f"{key} SHA mismatch")
            evidence[key] = _read(path)
            sources.append({"path": str(path.resolve()), "sha256": r[key+"_sha256"]})
        plan = TaskAnchoredResidualPlan.from_dict(evidence["plan"])
        definition = definitions[tid]
        if plan.definition != definition or definition["task_sha256"] != by_id[tid].sha256():
            raise ValueError("original plan/definition/task mismatch")
        z, mask = validate_z(plan.z_m, plan.interval_mask)
        a, e = evidence["attempt"], evidence["evaluation"]
        if a.get("task_id", tid) != tid or e.get("task_id", tid) != tid or e.get("task_sha256", by_id[tid].sha256()) != by_id[tid].sha256():
            raise ValueError("success evidence task mismatch")
        metrics = e.get("metrics") or {}
        binding = e.get("reference_binding") or {}
        horizon = metrics.get("actual_saved_horizon_s")
        full_horizon = type(horizon) in (float, int) and np.isfinite(horizon) and abs(horizon-27.) <= 1e-9
        gates = {"record_full_task_success": r.get("full_task_success") is True,
                 "record_nonzero_reference": r.get("nonzero_reference") is True and bool(np.any(z != 0)),
                 "record_binding": r.get("consumed_reference_binding_passed") is True,
                 "attempt_full": a.get("full_task_success") is True and a.get("full_27s_success") is True,
                 "attempt_plan_binding": a.get("plan_content_sha256") == plan.sha256(),
                 "attempt_task_binding": a.get("task_sha256") == by_id[tid].sha256(),
                 "full_13500_physics_steps": a.get("actual_steps") == 13500 and metrics.get("physics_steps") == 13500,
                 "full_27s_horizon": bool(full_horizon),
                 "nonzero_consumed": binding.get("nonzero_reference_consumed") is True,
                 "evaluation_full": e.get("task_success") is True and e.get("full_task_success") is True,
                 "complete": e.get("complete") is True, "evidence_valid": e.get("evidence_valid") is True}
        for key in ("reference_binding", "execution_contract", "independent_interval", "native_geometry", "task_requirements"):
            gates[key] = (e.get(key) or {}).get("passed") is True
        if not all(gates.values()):
            excluded.append({**r, "eligibility_gates": gates})
            continue
        sample_id = f"{tid}__candidate_{ci}"
        base = output / "samples" / sample_id
        for name, value in (("task", by_id[tid].to_dict()), ("definition", definition), ("plan", evidence["plan"])):
            _write(base / (name+".json"), value)
        eligible.append({**r, "sample_id": sample_id, "eligibility_gates": gates,
                         "task_sha256": by_id[tid].sha256(),
                         "frozen_task": str((base/"task.json").relative_to(output)),
                         "frozen_definition": str((base/"definition.json").relative_to(output)),
                         "frozen_plan": str((base/"plan.json").relative_to(output)),
                         "frozen_file_shas": {name: sha256(base/(name+".json")) for name in ("task", "definition", "plan")}})
    counts = {split: {"references": sum(x["split"] == split for x in eligible),
                      "tasks": len({x["task_id"] for x in eligible if x["split"] == split})} for split in ("train", "val")}
    enough = counts["train"]["references"] >= 8 and counts["train"]["tasks"] >= 4 and counts["val"]["references"] >= 2 and counts["val"]["tasks"] >= 2
    runnable = counts["train"]["references"] > 0 and counts["val"]["references"] > 0
    manifest = {"schema": DATASET_SCHEMA, "samples": eligible, "excluded": excluded, "counts": counts,
                "teacher_records": len(records), "data_status": "SUFFICIENT_ADVISORY_DATA" if enough else "DATA_LIMITED",
                "training_status": "READY" if runnable else "TRAINING_NOT_RUN",
                "reason": None if runnable else "successful nonzero full-task TRAIN and VAL labels are required",
                "source_files": sources, "task_suite_sha256": object_sha(suite),
                "definition_manifest_sha256": object_sha(definitions), "teacher_manifest_sha256": object_sha(teacher),
                "test_labels_used": 0, "representation": "task_anchored_cartesian_residual_v1"}
    _write(output/"manifest.json", manifest)
    return manifest


def load_residual_dataset(manifest_path):
    path = Path(manifest_path)
    m = _read(path)
    if m["schema"] != DATASET_SCHEMA:
        raise ValueError("unsupported residual dataset")
    for source in m["source_files"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("original residual evidence changed")
    tasks, defs, zs, masks = {}, {}, [], []
    for s in m["samples"]:
        values = {}
        for name in ("task", "definition", "plan"):
            p = path.parent / s["frozen_"+name]
            if sha256(p) != s["frozen_file_shas"][name]:
                raise ValueError("frozen label changed")
            values[name] = _read(p)
        tid = s["task_id"]
        task = _task(values["task"])
        if task.sha256() != s["task_sha256"] or task.split != s["split"] or task.split not in ("train", "val"):
            raise ValueError("frozen task binding changed")
        z, mask = validate_z(values["plan"]["z_m"], np.asarray(values["definition"]["interval_mask"], dtype=bool))
        tasks[tid], defs[tid] = task, values["definition"]
        zs.append(z); masks.append(mask)
    return ResidualDataset(path, m, m["samples"], tasks, defs,
                           np.asarray(zs, dtype=np.float64).reshape(-1, 6, 2), np.asarray(masks, dtype=bool).reshape(-1, 6))


def compatibility_key(definition):
    d = definition
    version = d.get("base_reference_version")
    if not version:
        raise ValueError("explicit base_reference_version required for retrieval")
    return (d["representation_version"], version, d["frame"], d["time_mapping"],
            float(d["coefficient_norm_bound_m"]), tuple(d["interval_mask"]))


def retrieve_train_residual(dataset, task, definition, normalizer):
    query = normalizer.transform(task, definition).astype(np.float64)
    candidates = []
    for i in dataset.indices("train"):
        s = dataset.samples[i]; tid = s["task_id"]
        if compatibility_key(dataset.definitions[tid]) != compatibility_key(definition):
            continue
        diff = normalizer.transform(dataset.tasks[tid], dataset.definitions[tid]).astype(np.float64)-query
        candidates.append((float(np.dot(diff, diff)), tid, s["candidate_index"], s["plan_sha256"], int(i)))
    if not candidates:
        raise ValueError("RETRIEVAL_NO_COMPATIBLE_REFERENCE")
    distance, tid, ci, plan_sha, i = min(candidates)
    return dataset.z_m[i].copy(), {"method": "E1", "source_task_id": tid, "candidate_index": ci,
                                   "source_plan_sha256": plan_sha, "squared_condition_distance": distance,
                                   "tie_rule": "distance,task_id,candidate_index,plan_sha256", "z_modified": False}
