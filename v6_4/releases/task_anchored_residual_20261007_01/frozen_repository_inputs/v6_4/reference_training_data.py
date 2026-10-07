"""Freeze controls that were consumed as successful references, with group holdout.

The source TaskSpec and historical split never change.  B.1's allocation is a
separate field; actual executed states are provenance, never learning targets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.dataset import (DATASET_SCHEMA, REPRESENTATION, canonical, grouped_split,
                         object_sha, sha256, _verified_file)
from v6_4.task_protocol import TaskSpec
from v6_4.trajectory_codec import CubicBSplineCodec

SCHEMA = "v6_4_b1_consumed_reference_dataset_v1"
NORMALIZER_SCHEMA = "v6_4_b1_train_only_reference_control_normalizer_v1"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _allocation(samples, group_splits):
    if set(group_splits) != {s["source_group_id"] for s in samples}:
        raise ValueError("allocation must name every complete source group exactly once")
    if any(split not in ("train", "val") for split in group_splits.values()):
        raise ValueError("reference holdout allocation is train/val only")
    # Reuse the established task/source/initial-state connected-group check, but
    # in copies only: the original TaskSpec and source split remain immutable.
    allocated = []
    for source in samples:
        item = json.loads(canonical(source))
        split = group_splits[item["source_group_id"]]
        item["split"] = split
        if "task" in item:
            item["task"]["split"] = split
        allocated.append(item)
    return grouped_split(allocated)


def freeze_reference_dataset(source_manifest, output_dir, group_splits, *,
                             available_proposal_only_count=0):
    """Copy small original controls/tasks after verifying their consumed lineage.

    Historical force trace bytes are hash checked, not copied or replayed.  No
    success, TaskSpec, geometry, or dynamics re-evaluation is performed here.
    """
    source_manifest = Path(source_manifest).resolve()
    original = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (original.get("schema") != DATASET_SCHEMA or original.get("representation") != REPRESENTATION
            or original.get("success_only") is not True or not original.get("samples")):
        raise ValueError("source must contain established successful reference samples")
    source_samples = original["samples"]
    grouped_split(source_samples)
    split = _allocation(source_samples, group_splits)
    validated = []
    for source in source_samples:
        if source.get("kind") != "nominal_closed_loop_teacher" or source.get("representation") != REPRESENTATION:
            raise ValueError("actual/derived/proposal-only labels cannot become successful references")
        task = source["task"]
        if (task["split"] not in ("train", "val") or source["split"] != task["split"]
                or task["task_id"] != source["task_id"] or task["group_id"] != source["source_group_id"]
                or object_sha(task) != source["task_sha256"]):
            raise ValueError("source TaskSpec identity/split invalid or development TEST excluded")
        paths = {key: _verified_file(source[key], source_manifest.parent)
                 for key in ("controls", "executed_reference", "success_report", "physical_trace")}
        sources = [_verified_file(record, source_manifest.parent) for record in source.get("sources", [])]
        free = np.load(paths["controls"], allow_pickle=False)
        codec = CubicBSplineCodec(task["initial_planner_q"], task["initial_planner_dq"])
        full = codec.decode_free(free)
        with np.load(paths["executed_reference"], allow_pickle=False) as reference:
            if not np.array_equal(reference["control_points"], full):
                raise ValueError("learning target differs from consumed selected controls")
        report = json.loads(paths["success_report"].read_text(encoding="utf-8"))
        if (report.get("schema") != "v6_4_planning_task_evaluation_v1"
                or not all(report.get(key) is True for key in ("complete", "evidence_valid", "task_success"))
                or report.get("task_sha256") != source["task_sha256"]
                or report.get("task_id") != source["task_id"]
                or report.get("model_contract_sha256") != task["model_contract_sha256"]
                or report.get("trace_sha256") != source["physical_trace"]["sha256"]
                or report.get("reference_sha256") != source["executed_reference"]["sha256"]
                or Path(report["reference_path"]).resolve() != paths["executed_reference"]):
            raise ValueError("established successful report is not bound to the consumed reference")
        proposal_paths = [p for p in sources if p.name == "raw_proposal.json"]
        if len(proposal_paths) != 1:
            raise ValueError("one explicit original selected proposal identity is required")
        proposal = TrajectoryProposal.from_dict(json.loads(proposal_paths[0].read_text(encoding="utf-8")))
        if (proposal.task_sha256 != source["task_sha256"] or proposal.task_id != source["task_id"]
                or not np.array_equal(proposal.free_controls, free)):
            raise ValueError("selected proposal differs from successful consumed reference controls")
        validated.append((source, free, full, proposal))
    if len({s[0]["task"]["model_contract_sha256"] for s in validated}) != 1:
        raise ValueError("all references must use one nominal model identity")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index, (source, free, full, proposal) in enumerate(validated):
        directory = output_dir / "samples" / f"sample_{index:04d}"
        directory.mkdir(parents=True)
        task_path = directory / "task.json"
        _write(task_path, source["task"])
        free_path, full_path = directory / "controls_free.npy", directory / "controls_full.npy"
        np.save(free_path, free, allow_pickle=False)
        np.save(full_path, full, allow_pickle=False)
        provenance = {"schema": "v6_4_b1_consumed_reference_label_v1",
            "label_semantics": "repaired_reference" if proposal.postprocessing else "original_reference",
            "data_level": "successful_consumed_reference", "label_is_actual_executed_state": False,
            "label_is_derived_actual_q_fit": False, "teacher_proposal_origin": proposal.origin,
            "reference_construction": proposal.metadata.get("intrinsic_algorithm", "explicit_selected_proposal"),
            "planning_prior_actual_fit_ancestry": "teacher planning prior may have actual-q-fit ancestry; selected controls were independently executed successfully",
            "postprocessing": list(proposal.postprocessing), "source_sample": source,
            "source_manifest": _record(source_manifest), "source_split": source["split"],
            "experiment_split": group_splits[source["source_group_id"]],
            "decoder": "CubicBSplineCodec; original physical time t in [0,27]; 32 uniform clamped cubic controls",
            "coordinate_frame": "17 planner coordinates; Cartesian/private zero-momentum poses in world; conditioning in frozen B0",
            "start_boundary": codec.to_dict()["start_boundary"],
            "source_force_trace_is_target": False, "new_physics_steps": 0}
        provenance_path = directory / "reference_provenance.json"
        _write(provenance_path, provenance)
        samples.append({"sample_id": source["sample_id"], "task_id": source["task_id"],
            "source_group_id": source["source_group_id"], "family": source["task"]["family"],
            "source_split": source["split"], "experiment_split": group_splits[source["source_group_id"]],
            "task_sha256": source["task_sha256"], "task_file": _record(task_path),
            "controls_free_file": _record(free_path), "controls_full_file": _record(full_path),
            "provenance_file": _record(provenance_path), "data_level": "successful_consumed_reference",
            "label_semantics": provenance["label_semantics"], "source_reference": source["executed_reference"]})
    counts = {}
    for name in ("train", "val"):
        subset = [s for s in samples if s["experiment_split"] == name]
        counts[name] = {"reference_count": len(subset), "successful_reference_count": len(subset),
            "independent_task_count": len({s["task_id"] for s in subset}),
            "independent_source_group_count": len({s["source_group_id"] for s in subset}),
            "proposal_only_count_used": 0}
    manifest = {"schema": SCHEMA, "representation": REPRESENTATION, "samples": samples,
        "source_manifest": _record(source_manifest), "group_allocation": group_splits,
        "split": split, "counts": counts, "source_tasks_or_splits_modified": False,
        "success_only": True, "development_D0_D3_used": False,
        "available_proposal_only_reference_count": int(available_proposal_only_count),
        "proposal_only_references_used": 0,
        "proposal_only_exclusion_reason": "failed or unexecuted proposals do not select checkpoints in this pilot",
        "data_scope": "small independent task-group holdout; single-seed pilot, broad generalization DATA_LIMITED",
        "actual_trace_copied": False, "new_physics_steps": 0, "native_distance_queries": 0}
    _write(output_dir / "manifest.json", manifest)
    _write(output_dir / "split.json", {"group_allocation": group_splits, "counts": counts,
        "split": split, "preserve_original_task_split": True,
        "allocation_frozen_before_training": True, "random_window_split": False})
    return load_reference_dataset(output_dir / "manifest.json")


@dataclass
class ReferenceDataset:
    manifest_path: Path
    manifest_sha256: str
    samples: list[dict]
    tasks: list[dict]
    controls_free: np.ndarray
    controls: np.ndarray
    split: dict

    @property
    def full_controls(self):
        return self.controls

    def indices(self, split):
        if split not in ("train", "val"):
            raise ValueError("unknown reference allocation")
        return np.asarray([i for i, s in enumerate(self.samples) if s["experiment_split"] == split], dtype=int)


def load_reference_dataset(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("representation") != REPRESENTATION or manifest.get("success_only") is not True:
        raise ValueError("unknown consumed-reference dataset")
    tasks, frees, fulls = [], [], []
    allocation_check = []
    for sample in manifest["samples"]:
        if sample.get("data_level") != "successful_consumed_reference" or sample.get("label_semantics") not in ("original_reference", "repaired_reference"):
            raise ValueError("only explicit successful consumed controls are learning targets")
        paths = {key: _verified_file(sample[key], manifest_path.parent)
                 for key in ("task_file", "controls_free_file", "controls_full_file", "provenance_file")}
        task = json.loads(paths["task_file"].read_text(encoding="utf-8"))
        TaskSpec.from_dict(task)
        provenance = json.loads(paths["provenance_file"].read_text(encoding="utf-8"))
        if (object_sha(task) != sample["task_sha256"] or task["task_id"] != sample["task_id"]
                or task["group_id"] != sample["source_group_id"] or task["split"] != sample["source_split"]
                or task["split"] not in ("train", "val") or provenance.get("label_is_actual_executed_state") is not False
                or provenance.get("label_is_derived_actual_q_fit") is not False
                or provenance.get("experiment_split") != sample["experiment_split"]
                or sample["experiment_split"] != manifest["group_allocation"].get(sample["source_group_id"])):
            raise ValueError("reference allocation, provenance or original task identity changed")
        free, full = np.load(paths["controls_free_file"], allow_pickle=False), np.load(paths["controls_full_file"], allow_pickle=False)
        codec = CubicBSplineCodec(task["initial_planner_q"], task["initial_planner_dq"])
        if not np.array_equal(codec.decode_free(free), full):
            raise ValueError("saved full controls violate exact C0/C1 or differ from free controls")
        tasks.append(task); frees.append(free); fulls.append(full)
        allocation_check.append({"sample_id": sample["sample_id"], "task_id": sample["task_id"],
            "source_group_id": sample["source_group_id"], "split": task["split"], "task": task})
    expected = _allocation(allocation_check, manifest["group_allocation"])
    if expected != manifest["split"]:
        raise ValueError("frozen connected-group split differs from sample allocation")
    if len({t["model_contract_sha256"] for t in tasks}) != 1:
        raise ValueError("dataset nominal model identity changed")
    return ReferenceDataset(manifest_path, sha256(manifest_path), manifest["samples"], tasks,
                            np.stack(frees), np.stack(fulls), expected)


@dataclass(frozen=True)
class ControlNormalizer:
    control_mean: np.ndarray
    control_scale: np.ndarray
    training_sample_ids: tuple[str, ...]

    @classmethod
    def fit(cls, dataset):
        indices = dataset.indices("train")
        if not len(indices):
            raise ValueError("control normalizer requires training references")
        values = dataset.controls_free[indices]
        return cls(values.mean(axis=(0, 1)), np.maximum(values.std(axis=(0, 1)), 1e-6),
                   tuple(sorted(dataset.samples[i]["sample_id"] for i in indices)))

    def normalize_controls(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape[-2:] != (30, 17) or not np.all(np.isfinite(value)):
            raise ValueError("finite free controls [*,30,17] required")
        return ((value - self.control_mean) / self.control_scale).astype(np.float32)

    def denormalize_controls(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape[-2:] != (30, 17) or not np.all(np.isfinite(value)):
            raise ValueError("finite normalized controls [*,30,17] required")
        return value * self.control_scale + self.control_mean

    normalize = normalize_controls
    denormalize = denormalize_controls

    def to_dict(self):
        return {"schema": NORMALIZER_SCHEMA, "control_mean": self.control_mean.tolist(),
            "control_scale": self.control_scale.tolist(), "training_sample_ids": list(self.training_sample_ids),
            "fit_validation_or_test": False, "free_control_shape": [30, 17],
            "reduction": "all TRAIN references and their free control positions, per planner coordinate",
            "proposal_clipping_or_projection": False}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != NORMALIZER_SCHEMA or value.get("fit_validation_or_test") is not False:
            raise ValueError("invalid train-only normalizer provenance")
        mean, scale = (np.asarray(value[name], dtype=np.float64) for name in ("control_mean", "control_scale"))
        ids = tuple(value["training_sample_ids"])
        if (mean.shape != (17,) or scale.shape != (17,) or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale)) or np.any(scale <= 0) or not ids or len(set(ids)) != len(ids)):
            raise ValueError("invalid reference control normalizer values")
        return cls(mean, scale, ids)


def check_reference_representation(dataset, normalizer, output_path, *, maximum_samples=8):
    """Compare native reference controls; never fit or compare with actual q."""
    if not 1 <= maximum_samples <= 8:
        raise ValueError("representation pilot checks at most eight existing references")
    # First cover source groups, then add existing alternate routes in source
    # order. Selection is independent of model outputs and never exceeds eight.
    indices, seen_groups = [], set()
    for index, sample in enumerate(dataset.samples):
        if sample["source_group_id"] not in seen_groups:
            indices.append(index); seen_groups.add(sample["source_group_id"])
    indices += [index for index in range(len(dataset.samples)) if index not in indices]
    indices = indices[:maximum_samples]
    from unittest.mock import patch
    import mujoco
    from v6_4.run_planning import prepare_provider
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    spec = default_v6_lite_robot_spec()
    rows, started = [], time.perf_counter()
    for index in indices:
        sample, task = dataset.samples[index], TaskSpec.from_dict(dataset.tasks[index])
        reference_path = _verified_file(sample["source_reference"], dataset.manifest_path.parent)
        with np.load(reference_path, allow_pickle=False) as original:
            saved = {k: np.asarray(original[k]).copy() for k in original.files}
        controls = dataset.controls[index]
        roundtrip = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq).decode_free(
            normalizer.denormalize(normalizer.normalize(dataset.controls_free[index])))
        codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
        analytic = codec.sample(controls, saved["time"])
        decoded = codec.sample(roundtrip, saved["time"])
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("representation check cannot execute physics")), \
             patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("representation check cannot query collision")):
            provider, _ = prepare_provider(task, controls, spec)
        if not np.array_equal(provider.prediction["time"], saved["time"]):
            raise ValueError("original reference and current private reconstruction clocks differ")
        rows.append({"sample_id": sample["sample_id"], "experiment_split": sample["experiment_split"],
            "reference_source": sample["source_reference"],
            "control_normalization_roundtrip_max_abs_rad": float(np.max(np.abs(roundtrip-controls))),
            "roundtrip_decoded_max_abs": {k: float(np.max(np.abs(decoded[k]-analytic[k]))) for k in ("q", "dq", "ddq")},
            "original_saved_grid_vs_same_controls_analytic_max_abs": {k: float(np.max(np.abs(saved[k]-analytic[k]))) for k in ("q", "dq", "ddq")},
            "current_private_free_base_reconstruction_vs_saved_max_abs": {
                k: float(np.max(np.abs(provider.prediction[k]-saved[k])))
                for k in ("full_qpos", "full_qvel", "base_pose", "base_twist", "target_position", "target_rotation",
                          "rigid_position", "rigid_rotation", "continuum_position", "continuum_rotation")},
            "trajectory_fit_performed": False, "fit_error": "NOT_APPLICABLE_already_native_reference_controls",
            "actual_q_used_as_reference": False, "mj_step_calls": 0, "native_distance_queries": 0})
    report = {"schema": "v6_4_b1_reference_representation_check_v1", "dataset_sha256": dataset.manifest_sha256,
        "normalizer_sha256": object_sha(normalizer.to_dict()), "sample_count": len(rows), "rows": rows,
        "label_and_prediction_time": "original physical time t in [0,27]; no terminal progress applied",
        "normalization_precision": "float64 statistics -> float32 latent -> float64 inverse",
        "unavoidable_32_control_fitting_bias": "NOT_MEASURED_native_control_reference_requires_no_fitting",
        "actual_soft_reference_tracking_error_is_codec_check": False,
        "physics_steps": 0, "native_distance_queries": 0,
        "private_kinematic_prediction": "mj_forward/mj_integratePos only, original zero-momentum decoder",
        "elapsed_wall_s": time.perf_counter()-started}
    _write(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available-proposal-only-count", type=int, default=0)
    parser.add_argument("--check-representation", action="store_true")
    args = parser.parse_args()
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    dataset = freeze_reference_dataset(args.source_manifest, args.output, allocation,
        available_proposal_only_count=args.available_proposal_only_count)
    normalizer = ControlNormalizer.fit(dataset)
    _write(args.output / "normalizer.json", normalizer.to_dict())
    if args.check_representation:
        check_reference_representation(dataset, normalizer, args.output / "representation_check.json")
    print(json.dumps({"manifest": str(dataset.manifest_path), "sha256": dataset.manifest_sha256,
        "train_references": len(dataset.indices("train")), "val_references": len(dataset.indices("val")),
        "physics_steps": 0, "native_distance_queries": 0}), flush=True)


if __name__ == "__main__":
    main()
