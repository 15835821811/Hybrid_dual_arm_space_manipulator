"""Synthetic lineage/split regressions, not new robot success evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.dataset import sha256, write_teacher_manifest
from v6_4.reference_training_data import (ControlNormalizer, ReferenceDataset,
    _allocation, _record, freeze_reference_dataset, load_reference_dataset)
from v6_4.tests.test_learning_dataset import fixture_task, synthetic_teacher


def source_fixture(parent):
    records = []
    for index in range(2):
        task = fixture_task(task_id=f"task_{index}", group_id=f"group_{index}", shift=.2*index)
        sample = synthetic_teacher(parent, task, sample_id=f"sample_{index}", free_value=float(index))
        proposal_path = parent / f"proposal_{index}" / "raw_proposal.json"
        proposal_path.parent.mkdir()
        proposal_path.write_text(json.dumps(TrajectoryProposal.from_controls(task,
            np.full((30, 17), float(index)), origin="teacher").to_dict()), encoding="utf-8")
        sample["sources"] = [_record(proposal_path)]
        records.append(sample)
    manifest = parent / "source_manifest.json"
    write_teacher_manifest(records, manifest)
    return manifest, {"group_0": "train", "group_1": "val"}


class ReferenceTrainingDataTests(unittest.TestCase):
    def test_consumed_controls_preserve_source_task_and_new_group_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, allocation = source_fixture(parent)
            data = freeze_reference_dataset(source, parent / "frozen", allocation,
                available_proposal_only_count=9)
            self.assertEqual(data.controls.shape, (2, 32, 17))
            self.assertEqual(data.controls_free.shape, (2, 30, 17))
            self.assertEqual([t["split"] for t in data.tasks], ["train", "train"])
            np.testing.assert_array_equal(data.indices("train"), [0])
            np.testing.assert_array_equal(data.indices("val"), [1])
            manifest = json.loads(data.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["available_proposal_only_reference_count"], 9)
            self.assertEqual(manifest["proposal_only_references_used"], 0)
            self.assertFalse(manifest["actual_trace_copied"])
            self.assertEqual(manifest["counts"]["val"]["successful_reference_count"], 1)
            provenance = json.loads(Path(data.samples[0]["provenance_file"]["path"]).read_text(encoding="utf-8"))
            self.assertFalse(provenance["label_is_actual_executed_state"])
            self.assertEqual(provenance["label_semantics"], "original_reference")

    def test_same_source_task_cannot_cross_new_allocation(self):
        a = {"sample_id": "a", "task_id": "same_task", "source_group_id": "one", "split": "train"}
        b = {"sample_id": "b", "task_id": "same_task", "source_group_id": "two", "split": "train"}
        with self.assertRaisesRegex(ValueError, "leaked"):
            _allocation([a, b], {"one": "train", "two": "val"})
        with self.assertRaisesRegex(ValueError, "every complete source group"):
            _allocation([a, b], {"one": "train"})

    def test_normalizer_does_not_fit_validation_values(self):
        values = np.stack([np.arange(30*17).reshape(30, 17)*.001,
                           np.full((30, 17), 1000.)])
        data = ReferenceDataset(Path("unused"), "a"*64,
            [{"sample_id": "train", "experiment_split": "train"},
             {"sample_id": "val", "experiment_split": "val"}], [], values,
             np.zeros((2, 32, 17)), {})
        scaler = ControlNormalizer.fit(data)
        np.testing.assert_array_equal(scaler.control_mean, values[0].mean(axis=0))
        self.assertEqual(scaler.training_sample_ids, ("train",))
        restored = ControlNormalizer.from_dict(scaler.to_dict())
        np.testing.assert_allclose(restored.denormalize(restored.normalize(values[0])), values[0], atol=2e-8, rtol=0)
        invalid = scaler.to_dict(); invalid["fit_validation_or_test"] = True
        with self.assertRaisesRegex(ValueError, "train-only"):
            ControlNormalizer.from_dict(invalid)

    def test_actual_derived_labels_and_consumed_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, allocation = source_fixture(parent)
            original = json.loads(source.read_text(encoding="utf-8"))
            altered = deepcopy(original)
            altered["samples"][0]["kind"] = "derived_actual_reference"
            source.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actual/derived/proposal-only"):
                freeze_reference_dataset(source, parent / "bad_kind", allocation)
            self.assertFalse((parent / "bad_kind").exists())
            altered = deepcopy(original)
            ref = Path(altered["samples"][0]["executed_reference"]["path"])
            with np.load(ref, allow_pickle=False) as prior:
                controls = prior["control_points"].copy()
            controls[2, 0] += .1
            np.savez(ref, control_points=controls)
            altered["samples"][0]["executed_reference"] = _record(ref)
            source.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from consumed"):
                freeze_reference_dataset(source, parent / "bad_controls", allocation)

    def test_loader_rejects_changed_exact_start_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, allocation = source_fixture(parent)
            data = freeze_reference_dataset(source, parent / "frozen", allocation)
            manifest = json.loads(data.manifest_path.read_text(encoding="utf-8"))
            full_path = Path(manifest["samples"][0]["controls_full_file"]["path"])
            values = np.load(full_path, allow_pickle=False); values[0, 0] = .01
            np.save(full_path, values, allow_pickle=False)
            manifest["samples"][0]["controls_full_file"] = _record(full_path)
            data.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact C0/C1"):
                load_reference_dataset(data.manifest_path)


if __name__ == "__main__":
    unittest.main()
