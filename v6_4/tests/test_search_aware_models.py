"""C.3 model interfaces with small arrays/mocks; no optimizer or real DDIM."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v6_4.preference_diffusion_warmstart import inverse_raw
from v6_4.residual_diffusion import ResidualDDPM, ResidualDiffusionConfig
from v6_4.route_initializers import raw_seed_plan
from v6_4.simple_warmstart_regression import (CHECKPOINT_UPDATES, TRAINING_DEFAULTS,
    SimpleWarmstartRegression, SearchAwareSampler, draw_source_balanced_indices,
    frozen_noise, make_train_tensors, prepare_training_pair, source_balanced_index,
    train_model_pair)
from v6_4.tests.test_preference_diffusion_warmstart import fixture


def dataset():
    ds = fixture()
    ds.manifest["scalers_train_only"] = True
    for s in ds.samples:
        s["source_types"] = ["route_quality_example"]
    return ds


class SearchAwareModelTests(unittest.TestCase):
    def test_same_conditions_scalers_mask_and_honest_parameter_counts(self):
        ds = dataset()
        clean, conditions, masks = make_train_tensors(ds)
        dim = conditions.shape[-1]
        reg = SimpleWarmstartRegression(dim)
        diffusion = ResidualDDPM(ResidualDiffusionConfig(dim))
        expected_s = 128 * dim + 19724
        self.assertEqual(sum(p.numel() for p in reg.parameters()), expected_s)
        self.assertEqual(sum(p.numel() for p in diffusion.parameters()), expected_s + 17920)
        self.assertEqual([x.out_features for x in reg.network if isinstance(x, torch.nn.Linear)], [128, 128, 12])
        self.assertEqual([x.out_features for x in diffusion.denoiser if isinstance(x, torch.nn.Linear)], [128, 128, 12])
        # Only TRAIN tensors are constructed. Poisoned historical VAL values
        # cannot enter either model's reference tensor or condition encoding.
        ds.z_m[ds.indices("val")] = np.nan
        clean2, conditions2, masks2 = make_train_tensors(ds)
        self.assertTrue(torch.equal(clean, clean2))
        self.assertTrue(torch.equal(conditions, conditions2))
        self.assertTrue(torch.equal(masks, masks2))
        self.assertTrue(torch.all(clean[ds.indices("val")] == 0))
        output = reg(conditions[:4], masks[:4])
        self.assertEqual(tuple(output.shape), (4, 12))
        self.assertTrue(torch.all(output[~masks[:4].repeat_interleave(2, -1)] == 0))

    def test_regression_mse_is_per_original_reference_in_legal_dimensions(self):
        ds = dataset()
        clean, conditions, masks = make_train_tensors(ds)
        reg = SimpleWarmstartRegression(conditions.shape[1])
        target = clean[:4].clone()
        target[0, masks[0].repeat_interleave(2)] += 2.
        with patch.object(reg, "forward", return_value=clean[:4]):
            losses = reg.loss(target, conditions[:4], masks[:4], reduction="none")
        np.testing.assert_allclose(losses.numpy(), [4., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "legal-space"):
            target[0, torch.where(~masks[0].repeat_interleave(2))[0][0]] = 1e-30
            reg.loss(target, conditions[:4], masks[:4])

    def test_source_balancing_and_paired_draws_are_train_only(self):
        ds = dataset()
        extra = copy.deepcopy(ds.samples[0]); extra["sample_id"] += "effect"
        extra["source_types"] = ["initializer_effect_example"]
        extra["z_m"] = (np.asarray(extra["z_m"]) * .5).tolist()
        ds.samples.append(extra)
        ds.z_m = np.concatenate([ds.z_m, [extra["z_m"]]])
        ds.search_masks = np.concatenate([ds.search_masks, [ds.search_masks[0]]])
        index = source_balanced_index(ds)
        self.assertEqual(set(index), {"m0", "m1"})
        a = draw_source_balanced_indices(index, torch.Generator().manual_seed(64322), 2048)
        b = draw_source_balanced_indices(index, torch.Generator().manual_seed(64322), 2048)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(all(ds.samples[int(i)]["split"] == "train" for i in a))
        bucket = a[np.isin(a, [0, len(ds.samples) - 1])]
        fraction = np.mean(bucket == 0)
        self.assertGreater(fraction, .35); self.assertLess(fraction, .65)
        ds.samples[-1]["z_m"] = copy.deepcopy(ds.samples[0]["z_m"])
        ds.z_m[-1] = ds.z_m[0]
        ds.samples[-1]["source_types"] = ["route_quality_example"]
        with self.assertRaisesRegex(ValueError, "duplicate parameter"):
            source_balanced_index(ds)

    def test_fixed_seeds_and_only_two_predeclared_checkpoints(self):
        self.assertEqual(CHECKPOINT_UPDATES, (250, 4000))
        self.assertEqual(TRAINING_DEFAULTS["optimizer_updates"], 4000)
        self.assertEqual(TRAINING_DEFAULTS["batch_size"], 32)
        self.assertEqual(TRAINING_DEFAULTS["optimizer"], {"name": "AdamW", "lr": 1e-4,
            "weight_decay": .01, "gradient_norm_clip": 1.})
        task = dataset().tasks["v0"]
        a, b, test = frozen_noise(task, 64325), frozen_noise(task, 64325), frozen_noise(task, 64324)
        for slot in (1, 3):
            np.testing.assert_array_equal(a[slot], b[slot])
            self.assertFalse(np.array_equal(a[slot], test[slot]))
        with self.assertRaisesRegex(ValueError, "predeclared"):
            frozen_noise(task, 999)

    def test_prepare_has_no_optimizer_no_sampling_and_seals_paired_references(self):
        ds = dataset()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "dataset.json"; manifest.write_text("{}")
            with patch("v6_4.simple_warmstart_regression._load_dataset", return_value=ds), \
                 patch("v6_4.simple_warmstart_regression._source_identity", return_value={}), \
                 patch.object(ResidualDDPM, "sample_ddim", side_effect=AssertionError("no real DDIM")), \
                 patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no optimizer updates")):
                result = prepare_training_pair(manifest, root / "models")
                self.assertEqual(result["status"], "READY")
                with np.load(root / "models" / "paired_reference_draws.npz") as data:
                    refs = data["reference_indices"]
                self.assertEqual(refs.shape, (4000, 32))
                self.assertTrue(set(refs.reshape(-1)).issubset(set(ds.indices("train"))))
                self.assertFalse(list((root / "models").rglob("*.pt")))
                self.assertEqual(result["config"]["ddim_sample_units"], 0)
                self.assertEqual(result["config"]["sample_exposures_per_model"], 128000)
                with self.assertRaises(FileExistsError):
                    prepare_training_pair(manifest, root / "models")

    def test_sampler_preserves_invalid_raw_never_resamples_or_projects(self):
        ds = dataset()
        task, mask = ds.tasks["v0"], ds.search_masks[0]
        latent = torch.zeros((1, 12))
        latent[0, np.flatnonzero(~mask)[0] * 2] = 1.
        latent[0, np.flatnonzero(mask)[0] * 2] = 100.
        for name in ("D", "S"):
            sampler = object.__new__(SearchAwareSampler)
            sampler.device = torch.device("cpu"); sampler.model_name = name
            sampler.condition_normalizer = ds.condition_scaler
            sampler.residual_normalizer = ds.residual_scaler
            sampler.identity = {}; sampler.sample_units = 0
            from v6_4.preference_diffusion_warmstart import supported_conditions
            sampler.checkpoint = {"supported_conditions": supported_conditions(ds)}
            from unittest.mock import Mock
            sampler.model = Mock(return_value=latent)
            sampler.model.sample_ddim = Mock(return_value=latent)
            raw, meta = sampler.sample(task, "A", "v1", np.zeros(12) if name == "D" else None)
            np.testing.assert_array_equal(raw, inverse_raw(ds.residual_scaler, latent.numpy(), mask))
            self.assertNotEqual(raw[~mask][0, 0], 0.)
            plan, diagnostics = raw_seed_plan(task, {**meta, "raw_z_m": raw}, 1)
            self.assertIsNone(plan); self.assertFalse(diagnostics["raw_repaired"])
            self.assertFalse(diagnostics["resampled"])
            self.assertEqual(sampler.sample_units, 1)
            self.assertEqual(sampler.model.sample_ddim.call_count, int(name == "D"))
            self.assertEqual(sampler.model.call_count, int(name == "S"))
            self.assertEqual(meta["source"], "diffusion" if name == "D" else "regression")
            sampler.checkpoint["supported_conditions"] = {}
            raw, meta = sampler.sample(task, "A", "v1", np.zeros(12) if name == "D" else None)
            self.assertIsNone(raw); self.assertEqual(meta["initializer_rejection"], "UNSUPPORTED_TRAINING_CONDITION")
            self.assertEqual(sampler.sample_units, 1)

    def test_regression_same_raw_qualifier_rejects_all_illegalities(self):
        ds = dataset(); task = ds.tasks["v0"]; mask = ds.search_masks[0]
        seeds = []
        inactive = np.zeros((6, 2)); inactive[~mask, 0] = 1e-30; seeds.append(inactive)
        over = np.zeros((6, 2)); over[mask] = [.019, .019]; seeds.append(over)
        nonfinite = np.zeros((6, 2)); nonfinite[mask, 0] = np.nan; seeds.append(nonfinite)
        for z in seeds:
            with patch("v6_4.task_anchored_reference.reference_precheck", side_effect=AssertionError("raw rejects first")):
                plan, diag = raw_seed_plan(task, {"source": "regression", "preference": "A", "family": "v1", "raw_z_m": z}, 1)
            self.assertIsNone(plan); self.assertFalse(diag["raw_repaired"])
        with patch("v6_4.task_anchored_reference.reference_precheck", return_value={"passed": False}):
            plan, diag = raw_seed_plan(task, {"source": "regression", "preference": "A", "family": "v1", "raw_z_m": np.zeros(12)}, 1)
        self.assertIsNone(plan); self.assertIn("analytic reference precheck", diag["rejection_reason"])

    def test_pair_wrapper_requires_identical_exposure_reports(self):
        a = {"sample_exposures": {"x": 128000}, "paired_reference_indices_sha256": "same"}
        b = {"sample_exposures": {"x": 128000}, "paired_reference_indices_sha256": "different"}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("v6_4.simple_warmstart_regression.load_training_pair", return_value={}), \
             patch("v6_4.simple_warmstart_regression._train_one", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "exposures differ"):
                train_model_pair(tmp)

    def test_regression_source_lineage_uses_original_shared_search(self):
        from v6_4.tests.test_route_initializers import InitializerTests
        from v6_4.route_initializers import FrozenSeedInitializer
        from v6_4.route_optimizer_protocol import read
        InitializerTests.setUpClass()
        helper = InitializerTests()
        raw = np.zeros((6, 2))
        raw[helper.active[0]] = [.003, .004]
        raw[helper.active[-1]] = [.007, -.002]
        def evaluator(plan, cid):
            row = helper.evidence(plan, cid)
            row["prediction_metrics"].update(I_support=.01 if cid == "C01" else 0. if cid == "C04" else .2,
                L_full=.5 if cid == "C04" else 1., d_support=.04)
            return row
        with tempfile.TemporaryDirectory() as tmp:
            selection = helper.run_search(tmp, initializer=FrozenSeedInitializer(helper.seeds(raw, source="regression")), evaluator=evaluator)
            rows = read(Path(tmp) / "candidate_registry.json")
        self.assertEqual(rows[4]["parent_candidate_id"], "C01")
        self.assertEqual(rows[4]["origin_source"], "regression")
        self.assertEqual(selection["preferences"]["A"]["source_attribution"], "learning_seed_descendant")
        self.assertEqual(selection["preferences"]["A"]["selected_origin_source"], "regression")


if __name__ == "__main__":
    unittest.main()
