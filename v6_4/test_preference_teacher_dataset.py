"""C.2 label/split evidence checks; no simulation or training."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .preference_teacher_dataset import (ConditionNormalizer, PortableEvidence,
    classify_evidence, deduplicate_candidates, encode_condition, import_c1, preference_labels)
from .preference_warmstart_protocol import (BudgetLedger, historical_tasks,
    next_unused_seeds, search_interval_mask, validate_learning_splits)
from .residual_dataset import ResidualNormalizer
from .route_optimizer_protocol import VERSIONS, read, write
from .task_anchored_reference import TaskAnchoredResidualPlan, build_reference_definition


class PreferenceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temporary.name)
        cls.old = historical_tasks()
        cls.rows = [dict(row, mother_id=row["group_id"], split="train", historical=True,
            task_path=f"frozen_tasks/{task.task_id}/task.json") for task, row, path in cls.old]
        cls.manifest = {"tasks": cls.rows}
        write(cls.run_dir / "learning_split_manifest.json", cls.manifest)
        cls.imported = import_c1(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_released_rows_collapse_without_losing_zero(self):
        result = self.imported
        self.assertEqual(result["historical_preference_export_rows"], 96)
        self.assertEqual(result["historical_candidate_count_unique"], 48)
        self.assertEqual(result["evidence_tier_counts"]["VALIDATED_EXECUTION"], 10)
        self.assertEqual(result["evidence_tier_counts"]["PREDICTED_COMPLETE"], 27)
        facts = result["candidates"]
        self.assertEqual(len(deduplicate_candidates(facts + copy.deepcopy(facts))), 48)
        labels, buckets = preference_labels(facts, self.rows)
        self.assertTrue(any(not np.any(s["z_m"]) for s in labels))
        self.assertTrue(any(b["status"] == "NO_PREFERENCE_LABEL_WITHIN_TEACHER_BUDGET" for b in buckets))

    def test_validated_actual_does_not_imply_B_and_prediction_not_actual(self):
        facts = self.imported["candidates"]
        validated_below_B = [f for f in facts if f["evidence_tier"] == "VALIDATED_EXECUTION" and f["prediction_metrics"]["d_support"] < .030]
        self.assertTrue(validated_below_B)
        labels, _ = preference_labels(facts, self.rows)
        B_ids = {s["physical_candidate_id"] for s in labels if s["preference"] == "B"}
        self.assertTrue(all(f["physical_candidate_id"] not in B_ids for f in validated_below_B))
        f = next(f for f in facts if f["evidence_tier"] == "PREDICTED_COMPLETE")
        task = next(t for t, _, _ in self.old if t.sha256() == f["task_sha256"])
        plan = TaskAnchoredResidualPlan.from_dict(f["plan"])
        self.assertEqual(classify_evidence(f["prediction"], task, plan), "PREDICTED_COMPLETE")
        corrupt = copy.deepcopy(f["prediction"])
        corrupt["prediction_metrics"]["consumed_reference_binding"]["plan_sha256"] = "wrong"
        self.assertEqual(classify_evidence(corrupt, task, plan), "MISSING_OR_UNBOUND")

    def test_family_near_optimum_bands_and_no_missing_metric_zero_fill(self):
        source = next(f for f in self.imported["candidates"] if f["evidence_tier"] == "VALIDATED_EXECUTION")
        rows = []
        for index, (family, I, L, d) in enumerate((
                ("v1", .1, 1., .029), ("v1", .1009, 1.004, .031),
                ("v1", .1011, 1.006, .032), ("v2", .3, 1.5, .031),
                ("v2", .3005, 1.504, .032), ("v2", .3001, None, None))):
            fact = copy.deepcopy(source)
            fact.update(physical_candidate_id=str(index), family=family, reference_family=family)
            fact["prediction_metrics"].update(I_support=I, L_full=L, d_support=d)
            rows.append(fact)
        labels, _ = preference_labels(rows, [next(r for r in self.rows if r["task_sha256"] == source["task_sha256"])])
        self.assertEqual({s["physical_candidate_id"] for s in labels if s["preference"] == "A" and s["family"] == "v1"}, {"0", "1"})
        self.assertEqual({s["physical_candidate_id"] for s in labels if s["preference"] == "B" and s["family"] == "v1"}, {"1", "2"})
        self.assertEqual({s["physical_candidate_id"] for s in labels if s["preference"] == "B" and s["family"] == "v2"}, {"3", "4"})

    def test_external_SHA_split_and_train_only_discrete_scaler(self):
        task, _, path = self.old[0]
        original = path.read_bytes()
        self.assertEqual(task.split, "test")
        d1 = build_reference_definition(task, version=VERSIONS["v1"])
        d2 = build_reference_definition(task, version=VERSIONS["v2"])
        conditions = [(task, d1, "A", "v1"), (task, d2, "B", "v2")]
        scaler = ConditionNormalizer.fit(conditions, self.manifest)
        encoded = encode_condition(task, d2, "B", "v2")
        transformed = scaler.transform_condition(encoded)
        np.testing.assert_array_equal(transformed[encoded["literal"]], encoded["values"][encoded["literal"]])
        restored = ConditionNormalizer.from_dict(scaler.to_dict())
        np.testing.assert_array_equal(restored.transform(task, d2, "B", "v2"), transformed)
        self.assertGreater(sum(d1["interval_mask"]), sum(search_interval_mask(d1)))
        forbidden = copy.deepcopy(self.manifest)
        for row in forbidden["tasks"]:
            if row["mother_id"] == self.rows[0]["mother_id"]:
                row.update(split="val", historical=False)
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            ConditionNormalizer.fit(conditions, forbidden)
        leaking = copy.deepcopy(self.manifest)
        leaking["tasks"][0].update(split="val", historical=False)
        with self.assertRaises(ValueError):
            validate_learning_splits(leaking)
        self.assertEqual(original, path.read_bytes())

    def test_residual_zero_roundtrip_and_finite_budget(self):
        mask = np.array([False, True, True, False, False, False])
        z0 = np.zeros((6, 2)); z1 = z0.copy(); z1[2] = [.004, -.008]
        scaler = ResidualNormalizer.fit([z0, z1], [mask, mask])
        for z in (z0, z1):
            np.testing.assert_allclose(scaler.inverse(scaler.normalize(z, mask), mask), z, atol=1e-17)
            self.assertTrue(np.all(scaler.normalize(z, mask)[~mask] == 0.))
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(directory)
            receipt = ledger.reserve("test_ddim_samples", "batch", 16)
            self.assertEqual(ledger.reserve("test_ddim_samples", "batch", 16), receipt)
            self.assertEqual(ledger.totals()["test_ddim_samples"], 16)
            with self.assertRaises(ValueError):
                ledger.reserve("test_ddim_samples", "extra")
        self.assertEqual(next_unused_seeds({2026100701}, 2), [(1, 2026205430), (2, 2026310159)])


if __name__ == "__main__":
    unittest.main()
