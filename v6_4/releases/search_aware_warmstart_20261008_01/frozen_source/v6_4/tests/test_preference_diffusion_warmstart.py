"""Pure C.2 conditioning/training protocol tests; no formal training or physics."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v6_4.dataset import sha256
from v6_4.residual_dataset import ResidualNormalizer, _write
from v6_4.preference_teacher_dataset import ConditionNormalizer
from v6_4.preference_diffusion_warmstart import (TRAINING_DEFAULTS, balanced_index,
    condition_key, draw_training_indices, inverse_raw, make_validation_draws, prepare_training,
    make_tensors, raw_reference_assessment, search_mask, selection_key,
    support_sample_times, supported_conditions, train_model, validate_checkpoint,
    PreferenceSampler)
from v6_4.route_optimizer_protocol import HISTORY, VERSIONS, read
from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import build_reference_definition


def fixture():
    base = TaskSpec.from_dict(read(HISTORY / "snapshot/tasks/b3_mother_00_c_plus/task.json"))
    tasks, splits, mothers, rows, definitions = {}, {}, {}, [], {}
    for tid, split, mother in (("tr0", "train", "m0"), ("tr1", "train", "m1"),
                                ("v0", "val", "mv"), ("v1", "val", "mv")):
        task = replace(base, task_id=tid, group_id=mother, split="test")
        tasks[tid], splits[tid], mothers[tid] = task, split, mother
        for pref, fam in (("A", "v1"), ("B", "v2")):
            definition = build_reference_definition(task, version=VERSIONS[fam])
            definitions[(tid, fam)] = definition
            mask = search_mask(definition)
            z = np.zeros((6, 2)); z[mask, 0] = .01
            rows.append({"sample_id": tid + pref + fam, "task_id": tid,
                "mother_id": mother, "preference": pref, "reference_family": fam,
                "split": split, "z_m": z.tolist(), "search_mask": mask})
    split_manifest = {"tasks": [{"task_sha256": task.sha256(), "mother_id": mothers[tid],
        "mother_source_sha256": mothers[tid] + "_sha", "split": splits[tid]}
        for tid, task in tasks.items()]}
    train = [s for s in rows if s["split"] == "train"]
    cn = ConditionNormalizer.fit([(tasks[s["task_id"]], definitions[(s["task_id"], s["reference_family"])],
        s["preference"], s["reference_family"]) for s in train], split_manifest)
    zn = ResidualNormalizer.fit([s["z_m"] for s in train], [s["search_mask"] for s in train])
    ds = SimpleNamespace(manifest={"data_status": "DATA_LIMITED"}, samples=rows,
        z_m=np.asarray([s["z_m"] for s in rows]), search_masks=np.stack([s["search_mask"] for s in rows]),
        tasks=tasks, definitions=definitions, condition_scaler=cn, residual_scaler=zn,
        task_splits=splits, task_mothers=mothers)
    ds.indices = lambda split: np.asarray([i for i, s in enumerate(rows) if s["split"] == split], dtype=np.int64)
    return ds


class PreferenceTrainingTests(unittest.TestCase):
    def test_fixed_budget_and_schema_uses_search_mask(self):
        ds = fixture()
        self.assertEqual([TRAINING_DEFAULTS[k] for k in ("seed", "draw_seed", "validation_seed", "test_noise_seed")],
                         [64221, 64222, 64223, 64224])
        self.assertEqual(TRAINING_DEFAULTS["optimizer_updates"], 4000)
        self.assertEqual(TRAINING_DEFAULTS["validate_every"], 250)
        clean, condition, masks = make_tensors(ds)
        self.assertEqual(tuple(clean.shape), (8, 12))
        self.assertEqual(condition.shape[1], len(ds.condition_scaler.names))
        self.assertNotEqual(condition.shape[1], 878)
        self.assertTrue(torch.all(clean[~masks.repeat_interleave(2, -1)] == 0))
        self.assertEqual(set(supported_conditions(ds)), {condition_key(p, f, ds.search_masks[0])
                                                      for p, f in (("A", "v1"), ("B", "v2"))})

    def test_balanced_sampler_draws_train_only_reproducibly(self):
        ds = fixture()
        index = balanced_index(ds)
        self.assertEqual(sorted(index), ["m0", "m1"])
        a = draw_training_indices(index, torch.Generator().manual_seed(64222), 128)
        b = draw_training_indices(index, torch.Generator().manual_seed(64222), 128)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(all(ds.samples[int(i)]["split"] == "train" for i in a))
        self.assertEqual({ds.samples[int(i)]["mother_id"] for i in a}, {"m0", "m1"})

    def test_fixed_eight_val_units_and_missing_bucket_is_not_pass(self):
        ds = fixture()
        a, b = make_validation_draws(ds), make_validation_draws(ds)
        self.assertEqual(a, b)
        self.assertEqual(len(a["units"]), 8)
        self.assertEqual({u["task_id"] for u in a["units"]}, {"v0", "v1"})
        self.assertTrue(all(ds.samples[i]["split"] == "val" for i in a["reference_indices"]))
        ds.samples[7]["preference"] = "A"
        draws = make_validation_draws(ds)
        self.assertEqual(sum(u["status"] == "MISSING_VAL_LABEL" for u in draws["units"]), 2)

    def test_checkpoint_lexicographic_rule_and_zero_legal_distance(self):
        def v(legal, distance, loss):
            return {"legal_count": legal, "decoded_position_rms_m": distance, "fixed_v_mse": loss}
        self.assertLess(selection_key(v(2, .1, 10.), 500), selection_key(v(1, 0., 0.), 250))
        self.assertLess(selection_key(v(2, .01, 10.), 500), selection_key(v(2, .1, 0.), 250))
        self.assertLess(selection_key(v(0, None, 1.), 500), selection_key(v(0, None, 2.), 250))
        self.assertLess(selection_key(v(0, None, 1.), 250), selection_key(v(0, None, 1.), 500))

    def test_no_clipping_amplitude_and_inactive_reject_before_reference(self):
        ds = fixture(); task = ds.tasks["tr0"]; mask = ds.search_masks[0]
        z = np.zeros((6, 2)); z[mask, 0] = .021
        with patch("v6_4.preference_diffusion_warmstart.reference_precheck", side_effect=AssertionError("must reject raw")):
            assessment, plan = raw_reference_assessment(task, "v1", z, mask)
            self.assertEqual(assessment["reason"], "RAW_AMPLITUDE_EXCEEDS_20MM")
            self.assertIsNone(plan)
            z[:] = 0.; z[np.flatnonzero(~mask)[0], 0] = 1e-30
            assessment, _ = raw_reference_assessment(task, "v1", z, mask)
            self.assertEqual(assessment["reason"], "INACTIVE_SEARCH_DIMENSION_NONZERO")

    def test_val_uses_decoded_same_task_family_and_no_physics(self):
        ds = fixture()
        tensors = make_tensors(ds)
        class MockModel:
            def loss(self, clean, *args, **kwargs):
                return torch.arange(len(clean), dtype=torch.float32) + 1
            def sample_ddim(self, condition, mask, initial_noise):
                return torch.zeros((1, 12))
        # All mean-scaled outputs equal the 10mm labels. No real DDIM is run.
        with patch("v6_4.preference_diffusion_warmstart.reference_precheck",
                   return_value={"passed": True, "status": "REFERENCE_ACCEPTED", "checks": {}}):
            result = validate_checkpoint(MockModel(), ds, tensors, make_validation_draws(ds))
        self.assertEqual(result["legal_count"], 8)
        self.assertEqual(result["ddim_sample_units"], 8)
        self.assertAlmostEqual(result["decoded_position_rms_m"], 0., places=12)
        self.assertEqual(result["fixed_v_mse"], 2.5)
        self.assertEqual(result["physics_steps"], 0)
        definition = ds.definitions[("v0", "v1")]
        times = support_sample_times(definition)
        for slot in np.flatnonzero(ds.search_masks[0]):
            lower, upper = definition["intervals_s"][slot]
            self.assertLessEqual(np.diff(times[(times >= lower) & (times <= upper)]).max(), .02000000000001)

    def test_unsupported_sampler_does_not_generate_or_fake_raw(self):
        ds = fixture()
        sampler = object.__new__(PreferenceSampler)
        sampler.checkpoint = {"supported_conditions": {}}
        sampler.identity = {}
        sampler.sample_units = 0
        raw, meta = sampler.sample(ds.tasks["v0"], "B", "v2", np.zeros(12))
        self.assertIsNone(raw)
        self.assertEqual(meta["initializer_rejection"], "UNSUPPORTED_TRAINING_CONDITION")
        self.assertEqual(meta["ddim_sample_units"], 0)

    def test_raw_decoder_preserves_invalid_output_for_qualification(self):
        ds = fixture(); mask = ds.search_masks[0]
        latent = np.zeros((6, 2))
        valid = inverse_raw(ds.residual_scaler, latent, mask)
        np.testing.assert_array_equal(valid, ds.residual_scaler.inverse(latent, mask))
        latent[np.flatnonzero(mask)[0], 0] = np.inf
        latent[np.flatnonzero(~mask)[0], 0] = 1.
        raw = inverse_raw(ds.residual_scaler, latent, mask)
        self.assertTrue(np.isinf(raw[np.flatnonzero(mask)[0], 0]))
        self.assertNotEqual(raw[np.flatnonzero(~mask)[0], 0], 0.)

    def test_noise_exactly_two_fixed_task_bound_draws(self):
        ds = fixture(); sampler = object.__new__(PreferenceSampler)
        calls = []
        def sample(task, preference, family, noise):
            calls.append(noise.copy())
            return np.zeros((6, 2)), {"source": "diffusion", "preference": preference, "family": family}
        sampler.sample = sample
        first = sampler.initializer_proposals(ds.tasks["v0"])
        sampler.initializer_proposals(ds.tasks["v0"])
        sampler.initializer_proposals(ds.tasks["v1"])
        self.assertEqual(list(first), [1, 3])
        self.assertEqual(len(calls), 6)
        np.testing.assert_array_equal(calls[0], calls[2])
        self.assertFalse(np.array_equal(calls[0], calls[4]))
        self.assertEqual(len(first[1]["raw_z_m"]), 12)

    def test_prepare_never_trains_and_preserves_external_split(self):
        ds = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / "dataset.json"; _write(manifest, {})
            with patch("v6_4.preference_diffusion_warmstart._load_dataset", return_value=ds),\
                 patch.object(torch.optim, "AdamW", side_effect=AssertionError("prepare cannot train")):
                result = prepare_training(manifest, root / "training")
            self.assertEqual(result["status"], "READY")
            self.assertTrue(all(task.split == "test" for task in ds.tasks.values()))
            self.assertFalse((root / "training" / "model").exists())

    def test_completed_training_reuse_verifies_artifacts_never_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); model = output / "model"; model.mkdir()
            for name in ("selected", "last"):
                (model / (name + ".pt")).write_bytes(name.encode())
            report = {"optimizer_updates_total": 4000, "validation_checkpoint_count": 16,
                **{n + "_checkpoint_sha256": sha256(model / (n + ".pt")) for n in ("selected", "last")}}
            _write(model / "training_report.json", report)
            with patch("v6_4.preference_diffusion_warmstart.load_training_bundle", return_value={}),\
                 patch.object(torch.optim, "AdamW", side_effect=AssertionError("reuse cannot train")):
                self.assertEqual(train_model(output), report)
                (model / "selected.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    train_model(output)


if __name__ == "__main__":
    unittest.main()
