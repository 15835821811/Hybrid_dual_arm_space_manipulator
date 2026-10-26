"""Successful nominal teachers, immutable initial conditions, and group splits.

No video, static geometric case, perturbed open-loop replay, or failed controller
trace is eligible.  Conditioning reads TaskSpec declarations only, never future
actual trace states.  NumPy-only utilities remain usable without learning deps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DATASET_SCHEMA = "v6_4_successful_teacher_dataset_v1"
CONDITION_SCHEMA = "v6_4_initial_base_condition_v1"
REPRESENTATION = "v6_4_clamped_cubic_32x17_start_eliminated_v1"
MAX_POINTS = 16
MAX_OBSTACLES = 8
SPLITS = ("train", "val", "test")


def canonical(value: Any) -> str:
    # Matches the unique TaskSpec.sha256 canonical encoding.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def _rotation(quaternion) -> np.ndarray:
    q = _vector(quaternion, 4, "wxyz quaternion")
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 1e-6:
        raise ValueError("condition quaternion is not normalized")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 9 or not np.all(np.isfinite(array)):
        raise ValueError(name + " must contain nine finite entries")
    array = array.reshape(3, 3)
    if not np.allclose(array.T @ array, np.eye(3), rtol=0, atol=1e-6) or np.linalg.det(array) <= 0:
        raise ValueError(name + " is not a rotation")
    return array


def task_dict(task) -> dict:
    value = task.to_dict() if hasattr(task, "to_dict") else task
    if not isinstance(value, dict):
        raise ValueError("condition requires a TaskSpec dictionary")
    return json.loads(canonical(value))


def task_sha(task) -> str:
    if hasattr(task, "sha256"):
        value = task.sha256
        return value() if callable(value) else value
    return object_sha(task_dict(task))


def encode_condition(task, *, return_feature_names: bool = False):
    """Fixed vector in B0: only declared initial state, motion and task geometry.

    Target-local requirements are expressed using its initial pose and retain
    their frame indicator.  Future target behavior comes from declared twist,
    never observed future qpos.  Unknown frames and over-capacity tasks reject.
    """
    task = task_dict(task)
    if task.get("schema") != "v6_4_task_protocol_v1":
        raise ValueError("condition requires the frozen V6.4 TaskSpec schema")
    if task.get("twist_convention") != "world_linear_then_angular":
        raise ValueError("TaskSpec must explicitly supply world linear/angular twists")
    if float(task["duration_s"]) != 27. or float(task["task_period_s"]) != .02 or float(task["physics_period_s"]) != .002:
        raise ValueError("V6.4-A conditioning requires declared 27s / 20ms / 2ms")
    if task.get("path_freedom") != "free_intermediate_path_between_fixed_requirements":
        raise ValueError("unknown task path-freedom protocol")
    families = ("end_effector_detour", "mid_arm_detour", "multiple_routes")
    if task.get("family") not in families:
        raise ValueError("unknown task family")
    base = _vector(task["base_pose"], 7, "initial base pose")
    target = _vector(task["target_pose"], 7, "initial target pose")
    rb, rt = _rotation(base[3:]), _rotation(target[3:])
    inverse = rb.T
    values, names = [], []

    def add(name, entries):
        entries = np.asarray(entries, dtype=np.float64).ravel()
        if not np.all(np.isfinite(entries)):
            raise ValueError("nonfinite declared condition " + name)
        values.extend(entries.tolist())
        names.extend([f"{name}[{i}]" for i in range(len(entries))])

    def position(point):
        return inverse @ (_vector(point, 3, "world position") - base[:3])

    add("planner_q", _vector(task["initial_planner_q"], 17, "initial planner q"))
    add("planner_dq", _vector(task["initial_planner_dq"], 17, "initial planner dq"))
    add("base_pose_in_B0", [0., 0., 0., 1., 0., 0., 0.])
    for body in ("base", "target"):
        twist = _vector(task[body + "_twist"], 6, body + " world twist")
        add(body + "_twist_in_B0", np.r_[inverse @ twist[:3], inverse @ twist[3:]])
    add("target_position_in_B0", position(target[:3]))
    add("target_rotation_in_B0", inverse @ rt)
    scenario = task["scenario"]
    # Fixed nominal geometry is identified by TaskSpec.model_contract_sha256;
    # do not invent dimensions that are absent from the declared scenario.
    add("task_family", [float(task["family"] == name) for name in families])
    add("free_intermediate_path", [1.])
    add("grasp_point_target_frame_m", _vector(scenario["grasp_point_target_frame_m"], 3, "grasp point"))
    add("grasp_rotation_target_frame", _matrix(scenario["grasp_rotation_target_frame"], "grasp rotation"))
    add("continuum_orientation_in_B0", inverse @ _matrix(scenario["continuum_target_rotation_world"], "continuum orientation"))
    add("time_configuration_s", [task["duration_s"], task["task_period_s"], task["physics_period_s"]])
    requirements = task["requirements"]
    if len(requirements) > MAX_POINTS:
        raise ValueError("more than 16 task requirements; refusing truncation")
    for index in range(MAX_POINTS):
        if index >= len(requirements):
            add(f"requirement_{index}", np.zeros(24))
            continue
        point = requirements[index]
        arm, frame, kind = point["arm"], point["frame"], point["kind"]
        if arm not in ("rigid", "continuum") or frame not in ("world", "target") or kind not in ("waypoint", "terminal"):
            raise ValueError("unknown task requirement arm/frame/kind")
        p = _vector(point["position_m"], 3, "task point")
        rotation = _matrix(point["rotation"], "task requirement rotation")
        if frame == "target":
            p, rotation = target[:3] + rt @ p, rt @ rotation
        window = _vector(point["time_window_s"], 2, "requirement window")
        nominal_time = float(point["time_s"])
        if not 0 <= window[0] <= nominal_time <= window[1] <= 27:
            raise ValueError("invalid declared requirement time window")
        if float(point["position_tolerance_m"]) <= 0 or not 0 < float(point["orientation_tolerance_rad"]) <= np.pi:
            raise ValueError("invalid declared requirement tolerances")
        add(f"requirement_{index}", np.r_[1., float(arm == "rigid"), float(arm == "continuum"),
                    float(frame == "world"), float(frame == "target"), position(p), (inverse @ rotation).ravel(),
                    nominal_time, window, point["position_tolerance_m"], point["orientation_tolerance_rad"],
                    float(kind == "waypoint"), float(kind == "terminal")])
    obstacles = scenario.get("workspace_obstacles", [])
    if len(obstacles) > MAX_OBSTACLES:
        raise ValueError("more than eight obstacles; refusing truncation")
    for index in range(MAX_OBSTACLES):
        if index < len(obstacles):
            obstacle = obstacles[index]
            radius = float(obstacle["radius_m"])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("invalid declared obstacle radius")
            add(f"obstacle_{index}", np.r_[1., position(obstacle["center_w"]), radius])
        else:
            add(f"obstacle_{index}", np.zeros(5))
    result = np.asarray(values, dtype=np.float64)
    return (result, tuple(names)) if return_feature_names else result


def _file_record(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _verified_file(record: dict, base: Path) -> Path:
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError("teacher source bytes differ from recorded identity: " + str(path))
    return path


def _successful_physics(report: dict, trace_path: Path, task: dict, controls: np.ndarray) -> Path:
    if not (report.get("complete") is True and report.get("evidence_valid") is True and report.get("task_success") is True):
        raise ValueError("teacher requires complete, independently evaluated successful nominal closed loop")
    if report.get("schema") != "v6_4_planning_task_evaluation_v1":
        raise ValueError("teacher success must come from the fresh V6.4 task evaluator")
    for key, expected in (("task_id", task["task_id"]), ("task_sha256", object_sha(task)),
                          ("model_contract_sha256", task["model_contract_sha256"]),
                          ("trace_sha256", sha256(trace_path))):
        if report.get(key) != expected:
            raise ValueError("teacher evaluation differs from frozen task/physical source: " + key)
    reference_path = Path(report["reference_path"]).resolve()
    if not reference_path.is_file() or sha256(reference_path) != report["reference_sha256"]:
        raise ValueError("executed teacher reference differs from fresh evaluation identity")
    with np.load(reference_path, allow_pickle=False) as reference:
        full = np.asarray(reference["control_points"])
        if full.shape != (32, 17) or not np.all(np.isfinite(full)) or not np.array_equal(full[2:], controls):
            raise ValueError("training controls differ from the executed successful teacher reference")
        from .trajectory_codec import CubicBSplineCodec
        codec = CubicBSplineCodec(task["initial_planner_q"], task["initial_planner_dq"])
        codec.encode_free(full)  # Independent contract check, including exact C0/C1.
    with np.load(trace_path, allow_pickle=False) as trace:
        needed = ("time", "torque", "planner_q", "initial_qpos", "initial_qvel")
        if not all(k in trace for k in needed):
            raise ValueError("teacher source is not a physical controller trace")
        if trace["time"].shape != (13500,) or trace["torque"].shape != (13500, 67) or trace["planner_q"].shape != (13500, 17):
            raise ValueError("teacher physical horizon/interface is incomplete")
        if trace["initial_qpos"].shape != (81,) or trace["initial_qvel"].shape != (79,):
            raise ValueError("teacher physical initial-state layout differs from model")
        if not np.array_equal(trace["initial_qpos"], task["initial_qpos"]) or not np.array_equal(trace["initial_qvel"], task["initial_qvel"]):
            raise ValueError("teacher initial state differs from declared condition")
        if not np.allclose(trace["time"], np.arange(1, 13501)*.002, rtol=0, atol=1e-9):
            raise ValueError("teacher physics clock is not the full declared 2ms grid")
        if not all(np.all(np.isfinite(trace[k])) for k in needed):
            raise ValueError("teacher physical trace contains nonfinite values")
    return reference_path


def register_teacher_sample(*, sample_id: str, task, controls_free, success_report: Path,
                            trace_path: Path, output_dir: Path, source_files=()) -> dict:
    """Freeze one genuinely successful teacher; return a manifest sample record.

    The caller retains all failed attempts outside this successful-only dataset.
    Source grouping is inherited from TaskSpec and cannot be reassigned here.
    """
    task_value = task_dict(task)
    controls = np.asarray(controls_free, dtype=np.float64)
    if controls.shape != (30, 17) or not np.all(np.isfinite(controls)):
        raise ValueError("successful teacher controls must be finite 30x17")
    encode_condition(task_value)
    report = json.loads(Path(success_report).read_text(encoding="utf-8"))
    reference_path = _successful_physics(report, Path(trace_path), task_value, controls)
    if task_value["split"] not in SPLITS:
        raise ValueError("TaskSpec must freeze train/val/test before teacher collection")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    controls_path = output_dir / "controls_free.npy"
    np.save(controls_path, controls, allow_pickle=False)
    return {"sample_id": sample_id, "task_id": task_value["task_id"], "source_group_id": task_value["group_id"],
            "split": task_value["split"], "kind": "nominal_closed_loop_teacher", "task": task_value,
            "task_sha256": task_sha(task), "controls": _file_record(controls_path),
            "success_report": _file_record(success_report), "physical_trace": _file_record(trace_path),
            "executed_reference": _file_record(reference_path),
            "sources": [_file_record(p) for p in source_files], "representation": REPRESENTATION}


def write_teacher_manifest(samples: list[dict], path: Path) -> dict:
    manifest = {"schema": DATASET_SCHEMA, "representation": REPRESENTATION, "condition_schema": CONDITION_SCHEMA,
                "samples": samples, "success_only": True, "videos_or_static_or_perturbed_samples_eligible": False}
    path = Path(path)
    if path.exists():
        raise FileExistsError("refuse to replace a frozen teacher manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


def grouped_split(samples: list[dict]) -> dict:
    """Enforce connected task/source groups, including overlapping augmentations.

    TaskSpec predeclares the split.  Sharing either a task ID or a complete
    source-group ID links records; any conflicting split within a component fails.
    """
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError("teacher sample IDs must be unique and nonempty")
    parents = list(range(len(samples)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for index, sample in enumerate(samples):
        if sample["split"] not in SPLITS or not sample["task_id"] or not sample["source_group_id"]:
            raise ValueError("split/task/complete-source-group identity required")
        identities = [("task", sample["task_id"]), ("source", sample["source_group_id"])]
        if "task" in sample:
            task = sample["task"]
            identities.append(("initial_source", object_sha({"scenario": task["scenario"],
                "qpos": task["initial_qpos"], "qvel": task["initial_qvel"]})))
        for identity in identities:
            if identity in owners:
                parents[root(index)] = root(owners[identity])
            else:
                owners[identity] = index
    groups = {}
    for index in range(len(samples)):
        groups.setdefault(root(index), []).append(samples[index])
    result = {split: [] for split in SPLITS}
    components = []
    for group in groups.values():
        splits = {s["split"] for s in group}
        if len(splits) != 1:
            raise ValueError("same task or complete source group leaked across data splits")
        split = next(iter(splits))
        result[split].extend(s["sample_id"] for s in group)
        components.append({"split": split, "sample_ids": sorted(s["sample_id"] for s in group),
                           "task_ids": sorted({s["task_id"] for s in group}),
                           "source_group_ids": sorted({s["source_group_id"] for s in group})})
    for split in SPLITS:
        result[split].sort()
    return {"schema": "v6_4_task_source_group_split_v1", "sample_ids": result,
            "components": sorted(components, key=lambda c: c["sample_ids"]),
            "all_same_source_versions_and_augmentations_share_split": True}


@dataclass
class TeacherDataset:
    manifest_path: Path
    manifest_sha256: str
    samples: list[dict]
    controls: np.ndarray
    conditions: np.ndarray
    feature_names: tuple[str, ...]
    split: dict
    source_files: list[Path]

    def indices(self, split: str) -> np.ndarray:
        allowed = set(self.split["sample_ids"][split])
        return np.array([i for i, s in enumerate(self.samples) if s["sample_id"] in allowed], dtype=int)


def load_teacher_dataset(manifest_path: Path) -> TeacherDataset:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != DATASET_SCHEMA or manifest.get("representation") != REPRESENTATION
            or manifest.get("condition_schema") != CONDITION_SCHEMA or manifest.get("success_only") is not True):
        raise ValueError("not a V6.4-A successful teacher dataset")
    samples = manifest["samples"]
    if not samples:
        raise ValueError("no successful teachers; training must not fabricate samples")
    split = grouped_split(samples)
    if len({sample["task"]["model_contract_sha256"] for sample in samples}) != 1:
        raise ValueError("training must use one frozen nominal model contract")
    controls, conditions, refs, feature_names = [], [], {manifest_path}, None
    for sample in samples:
        if sample.get("kind") != "nominal_closed_loop_teacher" or sample.get("representation") != REPRESENTATION:
            raise ValueError("videos, static geometry, perturbation shadows and failed trials cannot be teachers")
        task = sample["task"]
        if task["task_id"] != sample["task_id"] or task["group_id"] != sample["source_group_id"] or task["split"] != sample["split"]:
            raise ValueError("sample reassigns its TaskSpec task/source/split")
        # TaskSpec SHA is canonical serialized contract, not a future trace hash.
        if object_sha(task) != sample["task_sha256"]:
            raise ValueError("frozen task declaration changed")
        files = {}
        for key in ("controls", "success_report", "physical_trace", "executed_reference"):
            files[key] = _verified_file(sample[key], manifest_path.parent)
            refs.add(files[key])
        for record in sample.get("sources", []):
            refs.add(_verified_file(record, manifest_path.parent))
        values = np.load(files["controls"], allow_pickle=False)
        if values.shape != (30, 17) or not np.all(np.isfinite(values)):
            raise ValueError("teacher free controls invalid")
        reference = _successful_physics(json.loads(files["success_report"].read_text(encoding="utf-8")), files["physical_trace"], task, values)
        if reference != files["executed_reference"]:
            raise ValueError("manifest teacher reference differs from fresh evaluated reference")
        condition, names = encode_condition(task, return_feature_names=True)
        if feature_names is not None and names != feature_names:
            raise ValueError("condition schema changed between successful teachers")
        feature_names = names
        controls.append(values)
        conditions.append(condition)
    return TeacherDataset(manifest_path, sha256(manifest_path), samples, np.stack(controls), np.stack(conditions), feature_names, split, sorted(refs))


@dataclass(frozen=True)
class TrainingNormalizer:
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    control_mean: np.ndarray
    control_scale: np.ndarray
    training_sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, dataset: TeacherDataset):
        indices = dataset.indices("train")
        if not len(indices):
            raise ValueError("normalizer needs genuine successful train teachers")
        conditions, controls = dataset.conditions[indices], dataset.controls[indices]
        return cls(conditions.mean(axis=0), np.maximum(conditions.std(axis=0), 1e-6),
                   controls.mean(axis=(0, 1)), np.maximum(controls.std(axis=(0, 1)), 1e-6),
                   tuple(sorted(dataset.samples[i]["sample_id"] for i in indices)), dataset.feature_names)

    def normalize_conditions(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.condition_mean) or not np.all(np.isfinite(array)):
            raise ValueError("invalid condition for frozen training normalizer")
        return ((array - self.condition_mean) / self.condition_scale).astype(np.float32)

    def normalize_controls(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-2:] != (30, 17) or not np.all(np.isfinite(array)):
            raise ValueError("invalid free controls for frozen normalizer")
        return ((array - self.control_mean) / self.control_scale).astype(np.float32)

    def denormalize_controls(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-2:] != (30, 17) or not np.all(np.isfinite(array)):
            raise ValueError("nonfinite/raw model proposal")
        return array * self.control_scale + self.control_mean

    def to_dict(self):
        return {"schema": "v6_4_train_only_affine_normalizer_v1", "condition_schema": CONDITION_SCHEMA,
                "condition_mean": self.condition_mean.tolist(), "condition_scale": self.condition_scale.tolist(),
                "control_mean": self.control_mean.tolist(), "control_scale": self.control_scale.tolist(),
                "training_sample_ids": list(self.training_sample_ids), "feature_names": list(self.feature_names),
                "fit_validation_or_test": False, "proposal_clipping_or_projection": False}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "v6_4_train_only_affine_normalizer_v1" or value.get("fit_validation_or_test") is not False:
            raise ValueError("normalizer provenance is invalid")
        arrays = [np.asarray(value[k], dtype=np.float64) for k in ("condition_mean", "condition_scale", "control_mean", "control_scale")]
        if arrays[0].ndim != 1 or arrays[1].shape != arrays[0].shape or arrays[2].shape != (17,) or arrays[3].shape != (17,):
            raise ValueError("normalizer dimensions differ from proposal contract")
        if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
            raise ValueError("normalizer has invalid numeric values")
        if (len(value["feature_names"]) != len(arrays[0]) or len(set(value["feature_names"])) != len(value["feature_names"])
                or not value["training_sample_ids"] or len(set(value["training_sample_ids"])) != len(value["training_sample_ids"])):
            raise ValueError("normalizer feature/train identities are invalid")
        return cls(*arrays, tuple(value["training_sample_ids"]), tuple(value["feature_names"]))
