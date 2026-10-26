"""Synthetic file fixtures check eligibility/splitting; no physics is executed."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from v6_4.dataset import (TeacherDataset, TrainingNormalizer, encode_condition,
    grouped_split, load_teacher_dataset, object_sha, register_teacher_sample,
    write_teacher_manifest, sha256)
from v6_4.task_protocol import TaskPoint, TaskSpec, canonical_json


def fixture_task(split="train", task_id="fixture", group_id="fixture", shift=0.):
    """Only a unit-test contract, never a physical-success dataset claim."""
    identity = tuple(np.eye(3).flat)
    points = (
        TaskPoint("continuum_terminal", "continuum", "world", (shift + .4, .2, .3),
                  identity, 27., (26., 27.), .001, .01, "terminal"),
        TaskPoint("rigid_terminal", "rigid", "target", (.1, 0., 0.), identity,
                  27., (26., 27.), .001, .01, "terminal"))
    scenario = {"workspace_obstacles": [{"center_w": [shift + .1, .3, .5], "radius_m": .04}],
                "grasp_point_target_frame_m": [.1, 0., 0.], "grasp_rotation_target_frame": list(identity),
                "continuum_target_rotation_world": list(identity)}
    return TaskSpec(task_id, group_id, split, "multiple_routes", 64, canonical_json(scenario),
        tuple(np.zeros(81)), tuple(np.zeros(79)), tuple(np.zeros(17)), tuple(np.zeros(17)),
        (0., 0., 0., 1., 0., 0., 0.), (.1, .2, .3, .2, .1, .4),
        (shift + .5, .1, .2, 1., 0., 0., 0.), (.01, .02, .03, .02, .01, .03),
        points, "a" * 64)


def synthetic_teacher(parent: Path, task, sample_id="sample", free_value=0.):
    """13500-row shape fixture; does not simulate or certify a robot."""
    trace_path, report_path = parent / (sample_id + ".npz"), parent / (sample_id + ".json")
    reference_path = parent / (sample_id + "_reference.npz")
    free = np.full((30, 17), free_value)
    initial = np.asarray(task.initial_planner_q)
    second = initial + (27./29.) / 3. * np.asarray(task.initial_planner_dq)
    np.savez(reference_path, control_points=np.vstack([initial, second, free]))
    np.savez(trace_path, time=np.arange(1, 13501) * .002, torque=np.zeros((13500, 67)),
             planner_q=np.zeros((13500, 17)), initial_qpos=task.initial_qpos, initial_qvel=task.initial_qvel)
    report_path.write_text(json.dumps({"schema": "v6_4_planning_task_evaluation_v1", "complete": True,
        "evidence_valid": True, "task_success": True, "task_id": task.task_id, "task_sha256": task.sha256(),
        "model_contract_sha256": task.model_contract_sha256, "trace_sha256": sha256(trace_path),
        "reference_path": str(reference_path.resolve()), "reference_sha256": sha256(reference_path),
        "fixture_only": True}), encoding="utf-8")
    return register_teacher_sample(sample_id=sample_id, task=task, controls_free=free,
        success_report=report_path, trace_path=trace_path, output_dir=parent / (sample_id + "_record"))


class LearningDatasetTests(unittest.TestCase):
    def test_condition_task_hash_and_no_future_actual_input(self):
        task = fixture_task()
        self.assertEqual(task.sha256(), object_sha(task.to_dict()))
        values, names = encode_condition(task, return_feature_names=True)
        self.assertEqual(values.shape, (517,))
        self.assertEqual(len(set(names)), len(names))
        enriched = task.to_dict()
        enriched["future_actual_trace"] = {"base_qpos": [999.], "target_qpos": [-999.]}
        np.testing.assert_array_equal(values, encode_condition(enriched))
        changed = task.to_dict()
        changed["requirements"][0]["time_s"] = 26.5
        self.assertFalse(np.array_equal(values, encode_condition(changed)))

    def test_condition_global_rigid_transform_invariance(self):
        original = fixture_task().to_dict()
        moved = deepcopy(original)
        angle, translation = .7, np.array([3., -2., .5])
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        quaternion = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        for field in ("base_pose", "target_pose"):
            moved[field] = list(rotation @ np.array(original[field][:3]) + translation) + quaternion
        for field in ("base_twist", "target_twist"):
            moved[field] = list(rotation @ np.array(original[field][:3])) + list(rotation @ np.array(original[field][3:]))
        moved["scenario"]["continuum_target_rotation_world"] = list(rotation.flat)
        for obstacle in moved["scenario"]["workspace_obstacles"]:
            obstacle["center_w"] = list(rotation @ np.array(obstacle["center_w"]) + translation)
        for point in moved["requirements"]:
            if point["frame"] == "world":
                point["position_m"] = list(rotation @ np.array(point["position_m"]) + translation)
                point["rotation"] = list((rotation @ np.array(point["rotation"]).reshape(3, 3)).flat)
        np.testing.assert_allclose(encode_condition(original), encode_condition(moved), rtol=0, atol=1e-14)

    def test_unknown_frames_and_capacity_reject_instead_of_truncate(self):
        task = fixture_task().to_dict()
        task["requirements"][0]["frame"] = "future_actual_base"
        with self.assertRaisesRegex(ValueError, "arm/frame/kind"):
            encode_condition(task)
        task = fixture_task().to_dict()
        task["requirements"] = task["requirements"] * 9
        with self.assertRaisesRegex(ValueError, "16 task"):
            encode_condition(task)
        task = fixture_task().to_dict()
        task["scenario"]["workspace_obstacles"] *= 9
        with self.assertRaisesRegex(ValueError, "eight obstacles"):
            encode_condition(task)

    def test_split_transitive_task_source_group_leak_rejects(self):
        samples = [{"sample_id": "a", "task_id": "task_a", "source_group_id": "origin_a", "split": "train"},
                   {"sample_id": "b", "task_id": "task_b", "source_group_id": "origin_a", "split": "train"},
                   {"sample_id": "c", "task_id": "task_b", "source_group_id": "origin_b", "split": "test"}]
        with self.assertRaisesRegex(ValueError, "leaked"):
            grouped_split(samples)
        samples[-1]["split"] = "train"
        self.assertEqual(grouped_split(samples)["sample_ids"]["train"], ["a", "b", "c"])

    def test_renaming_same_source_does_not_evade_split(self):
        a, b = fixture_task("train", "a", "a"), fixture_task("test", "b", "b")
        samples = [{"sample_id": t.task_id, "task_id": t.task_id, "source_group_id": t.group_id,
                    "split": t.split, "task": t.to_dict()} for t in (a, b)]
        with self.assertRaisesRegex(ValueError, "leaked"):
            grouped_split(samples)

    def test_only_train_teachers_fit_normalizer(self):
        samples = [{"sample_id": name, "split": split} for name, split in (("a", "train"), ("b", "train"), ("v", "val"), ("t", "test"))]
        split = {"sample_ids": {"train": ["a", "b"], "val": ["v"], "test": ["t"]}}
        dataset = TeacherDataset(Path("fixture"), "x", samples,
            np.stack([np.full((30, 17), value) for value in (0., 2., 1000., -9999.)]),
            np.array([[0., 1.], [2., 3.], [1000., 2000.], [-9999., -9999.]]), ("a", "b"), split, [])
        normalizer = TrainingNormalizer.fit(dataset)
        np.testing.assert_array_equal(normalizer.condition_mean, [1., 2.])
        np.testing.assert_array_equal(normalizer.control_mean, np.ones(17))
        self.assertEqual(normalizer.training_sample_ids, ("a", "b"))
        restored = TrainingNormalizer.from_dict(normalizer.to_dict())
        free = np.full((2, 30, 17), 700.)
        np.testing.assert_allclose(restored.denormalize_controls(restored.normalize_controls(free)), free)
        self.assertGreater(restored.denormalize_controls(free).max(), 700.)

    def test_frozen_files_and_physical_initial_state_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = synthetic_teacher(root, fixture_task())
            manifest_path = root / "manifest.json"
            write_teacher_manifest([sample], manifest_path)
            dataset = load_teacher_dataset(manifest_path)
            self.assertEqual(dataset.controls.shape, (1, 30, 17))
            trace = Path(sample["physical_trace"]["path"])
            with np.load(trace) as values:
                copied = {key: values[key].copy() for key in values.files}
            copied["initial_qpos"][0] = 1.
            np.savez(trace, **copied)
            with self.assertRaisesRegex(ValueError, "source bytes differ"):
                load_teacher_dataset(manifest_path)
            report = Path(sample["success_report"]["path"])
            report_value = json.loads(report.read_text(encoding="utf-8"))
            report_value["trace_sha256"] = sha256(trace)
            report.write_text(json.dumps(report_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial state"):
                register_teacher_sample(sample_id="wrong", task=fixture_task(), controls_free=np.zeros((30, 17)),
                    success_report=report, trace_path=trace, output_dir=root / "wrong")
            self.assertFalse((root / "wrong").exists())

    def test_failed_or_shadow_trials_are_not_successful_training_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = synthetic_teacher(root, fixture_task())
            sample["kind"] = "inertia_open_loop_shadow"
            path = root / "manifest.json"
            write_teacher_manifest([sample], path)
            with self.assertRaisesRegex(ValueError, "cannot be teachers"):
                load_teacher_dataset(path)
            report = Path(sample["success_report"]["path"])
            report.write_text(json.dumps({"complete": False, "evidence_valid": True, "task_success": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete"):
                register_teacher_sample(sample_id="partial", task=fixture_task(), controls_free=np.zeros((30, 17)),
                    success_report=report, trace_path=Path(sample["physical_trace"]["path"]), output_dir=root / "partial")


if __name__ == "__main__":
    unittest.main()
