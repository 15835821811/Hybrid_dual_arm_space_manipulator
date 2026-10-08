"""C.2 raw slots, immutable prefixes and frozen C.1 numerical regression.

These tests use only mock nominal evidence and block physical integration.
"""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_4.continuous_route_optimizer import optimize
from v6_4.route_initializers import FrozenSeedInitializer, RetrievalInitializer, raw_seed_plan, search_mask
from v6_4.route_optimizer_protocol import (HISTORY, ROOT, PreferenceSpec, SearchSpec, TaskSpec,
    active_intervals, build_reference_definition, digest, read)


class InitializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = TaskSpec.from_dict(read(HISTORY / "snapshot/tasks/b3_mother_00_c_plus/task.json"))
        cls.definition = build_reference_definition(cls.task)
        cls.active = active_intervals(cls.definition)
        cls.identity = {"source_identity_sha256": "source", "config_sha256": "qp",
            "task_sha256": cls.task.sha256(), "model_contract_sha256": cls.task.model_contract_sha256}
        cls.preferences = [PreferenceSpec(name, cls.task.sha256(), (tuple(cls.definition["intervals_s"][2]),),
            tuple(tuple(cls.definition["intervals_s"][i]) for i in cls.active), "sphere", (("body", "sphere"),))
            for name in ("A", "B")]

    @staticmethod
    def evidence(plan, cid):
        # Deterministic nonflat quality makes centers and rankings meaningful.
        x = plan.z_m.reshape(-1)
        return {"status": "PREDICTION_ADMISSIBLE", "prediction_rollout_started": True,
            "prediction_admissible": True, "prediction_task_passed": True, "online_guards_passed": True,
            "prediction_metrics": {"I_support": .1 + float(np.sum(x)),
                "L_full": 1. + float(np.sum(x ** 2)), "d_support": .03 + float(np.sum(x))},
            "costs": {"prediction_physics_steps": 13500}}

    def seeds(self, raw=None, **fields):
        raw = np.zeros(12).tolist() if raw is None else raw
        return {slot: {"source": "diffusion", "family": family, "preference": preference,
            "raw_z_m": copy.deepcopy(raw), **fields}
            for slot, preference, family in ((1, "A", "v1"), (3, "B", "v2"))}

    @contextlib.contextmanager
    def no_physics(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(mujoco, "mj_step", side_effect=AssertionError("no physical integration allowed")):
            yield

    def run_search(self, tmp, budget=8, initializer=None, evaluator=None, function=optimize):
        with self.no_physics():
            return function(self.task, self.preferences, self.identity, evaluator or self.evidence,
                tmp, search_spec=SearchSpec(candidate_budget=budget), **({"initializer": initializer} if initializer is not None else {}))

    def test_default_numerics_match_actual_frozen_c1_source(self):
        frozen = ROOT / "v6_4/releases/continuous_route_optimizer_20261007_01/frozen_source/v6_4/continuous_route_optimizer.py"
        spec = importlib.util.spec_from_file_location("v6_4._frozen_c1_optimizer_regression", frozen)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as before, tempfile.TemporaryDirectory() as after:
            old_selection = self.run_search(before, budget=12, function=old.optimize)
            new_selection = self.run_search(after, budget=12)
            old_rows, new_rows = read(Path(before) / "candidate_registry.json"), read(Path(after) / "candidate_registry.json")
            self.assertEqual(len(old_rows), len(new_rows))
            for original, current in zip(old_rows, new_rows):
                self.assertEqual(original, {k: current[k] for k in original})
            old_proposals, new_proposals = read(Path(before) / "proposals.json"), read(Path(after) / "proposals.json")
            self.assertEqual(len(old_proposals), len(new_proposals))
            for original, current in zip(old_proposals, new_proposals):
                self.assertEqual(original, {k: current[k] for k in original})
            for name in ("A", "B"):
                original = old_selection["preferences"][name]
                self.assertEqual(original, {k: new_selection["preferences"][name][k] for k in original})
            self.assertEqual(old_selection["budget"], new_selection["budget"])

    def test_r8_is_r12_prefix_and_sealed_before_ninth_evaluation(self):
        captured = []
        with tempfile.TemporaryDirectory() as eight, tempfile.TemporaryDirectory() as twelve:
            r8 = self.run_search(eight)
            def evaluator(plan, cid):
                if int(cid[1:]) >= 8:
                    path = Path(twelve) / "prefix_08.json"
                    self.assertTrue(path.is_file())
                    captured.append(path.read_bytes())
                    row = self.evidence(plan, cid)
                    row["prediction_metrics"].update(I_support=0., L_full=.01, d_support=.1)
                    return row
                return self.evidence(plan, cid)
            r12 = self.run_search(twelve, budget=12, evaluator=evaluator)
            a = read(Path(eight) / "candidate_registry.json")
            b = read(Path(twelve) / "candidate_registry.json")
            self.assertEqual(a, b[:8])
            prefix = read(Path(twelve) / "prefix_08.json")
            self.assertEqual(prefix["preferences"], r8["preferences"])
            self.assertEqual(prefix["budget"]["slots_consumed"], 8)
            self.assertEqual(prefix["budget"]["prediction_rollouts_started"], 8)
            self.assertTrue(all(value == captured[0] for value in captured))
            self.assertEqual((Path(twelve) / "prefix_08.json").read_bytes(), captured[0])
            self.assertNotEqual(prefix["tie_selected_winner"], r12["tie_selected_winner"])
            seal = prefix.pop("snapshot_content_sha256")
            self.assertEqual(seal, digest(prefix))
            self.assertFalse(prefix["later_slots_read"])

    def test_illegal_raw_seeds_consume_slots_without_nominal_calls_or_repair(self):
        inactive = next(i for i in range(6) if i not in self.active)
        bad_inactive = np.zeros((6, 2)); bad_inactive[inactive, 0] = 1e-20
        over = np.zeros((6, 2)); over[self.active[-1], 0] = .020000000001
        nan = np.zeros(12); nan[2 * self.active[-1]] = np.nan
        inf = np.zeros(12); inf[2 * self.active[-1]] = np.inf
        for raw in (bad_inactive, over, nan, inf, [0.] * 4):
            with self.subTest(raw=repr(raw)), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def evaluator(plan, cid):
                    calls.append(cid); return self.evidence(plan, cid)
                result = self.run_search(tmp, initializer=FrozenSeedInitializer(self.seeds(raw)), evaluator=evaluator)
                rows = read(Path(tmp) / "candidate_registry.json")
                rejected = [r for r in rows if r["status"] == "INITIALIZER_RAW_REJECTED"]
                self.assertEqual(len(rejected), 2)
                self.assertEqual(len(calls), 6)
                self.assertEqual(result["budget"]["slots_consumed"], 8)
                self.assertEqual(result["budget"]["prediction_rollouts_started"], 6)
                for row in rejected:
                    self.assertIsNone(row["x_m"]); self.assertIsNone(row["plan"])
                    self.assertFalse(row["raw_seed_diagnostics"]["raw_repaired"])
                    self.assertFalse(row["raw_seed_diagnostics"]["resampled"])
                    self.assertFalse(row["prediction_rollout_started"])
                    self.assertEqual(row["prediction_steps"], 0)
                self.assertTrue(any(r["source"] == "adaptive_poll" for r in rows))

    def test_missing_diagnostics_and_unsupported_are_failed_slots(self):
        seeds = {1: {"raw_z_m": None}, 3: {"source": "diffusion", "family": "v2", "preference": "B",
            "initializer_rejection": "UNSUPPORTED_TRAINING_CONDITION", "raw_z_m": None}}
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_search(tmp, initializer=FrozenSeedInitializer(seeds))
            rows = read(Path(tmp) / "candidate_registry.json")
            self.assertEqual(result["budget"]["slots_consumed"], 8)
            self.assertEqual(rows[1]["status"], "INITIALIZER_RAW_REJECTED")
            self.assertEqual(rows[3]["raw_seed_diagnostics"]["rejection_reason"], "UNSUPPORTED_TRAINING_CONDITION")

    def test_legal_zero_duplicates_cache_without_regeneration(self):
        provider = FrozenSeedInitializer(self.seeds())
        calls = []
        class Counted:
            identity = provider.identity
            def __call__(self, task):
                calls.append(task.sha256()); return provider(task)
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_search(tmp, initializer=Counted())
            proposals = read(Path(tmp) / "proposals.json")
            rows = read(Path(tmp) / "candidate_registry.json")
            self.assertEqual(len(calls), 1)
            self.assertEqual(sum(p.get("source") == "diffusion" for p in proposals), 2)
            for p in (proposals[1], proposals[3]):
                self.assertEqual(p["family"], "v1")
                self.assertEqual(p["cache_hit_candidate_id"], "C00")
                self.assertTrue(p["raw_seed_diagnostics"]["raw_legal"])
            self.assertGreaterEqual(result["budget"]["cache_hits"], 2)
            self.assertEqual(result["budget"]["prediction_rollouts_started"], 8)
            self.assertEqual(rows[0]["origin_source"], "zero")
            again = self.run_search(tmp, initializer=Counted(), evaluator=lambda *args: self.fail("resume must not reevaluate"))
            self.assertEqual(result, again); self.assertEqual(len(calls), 1)

    def test_analytic_reference_rejection_never_reaches_evaluator(self):
        raw = np.zeros((6, 2)); raw[self.active[-1], 0] = .01
        with patch("v6_4.task_anchored_reference.reference_precheck", return_value={"passed": False, "errors": ["analytic bound"]}):
            plan, diagnostics = raw_seed_plan(self.task, self.seeds(raw)[1], 1)
        self.assertIsNone(plan)
        self.assertFalse(diagnostics["raw_legal"])
        self.assertIn("reference precheck", diagnostics["rejection_reason"])
        self.assertEqual(diagnostics["raw_z_m"], raw.tolist())

    def test_learning_seed_lineage_survives_original_adaptive_poll(self):
        raw = np.zeros((6, 2)); raw[self.active[0]] = [.003, .004]; raw[self.active[-1]] = [.007, -.002]
        with tempfile.TemporaryDirectory() as tmp:
            def evaluator(plan, cid):
                row = self.evidence(plan, cid)
                row["prediction_metrics"].update(I_support=.01 if cid == "C01" else 0. if cid == "C04" else .2,
                    L_full=.5 if cid == "C04" else 1., d_support=.04)
                return row
            result = self.run_search(tmp, initializer=FrozenSeedInitializer(self.seeds(raw)), evaluator=evaluator)
            rows = read(Path(tmp) / "candidate_registry.json")
            winner = rows[4]
            self.assertEqual(winner["parent_candidate_id"], "C01")
            self.assertEqual(winner["origin_source"], "diffusion")
            self.assertEqual(winner["proposal_lineage"][0], {"initial_position": 1, "source": "diffusion"})
            self.assertEqual(winner["projection_scope"], "original_nonlearning_adaptive_poll_only")
            self.assertEqual(result["preferences"]["A"]["source_attribution"], "learning_seed_descendant")
            self.assertEqual(result["preferences"]["A"]["selected_origin_source"], "diffusion")

    def test_retrieval_uses_train_same_mask_and_stable_ties_zero_allowed(self):
        mask = search_mask(self.task)
        labels = []
        for pref, family in (("A", "v1"), ("B", "v2")):
            for sid, split, task_id in (("b", "train", "b"), ("a", "train", "a"), ("forbidden", "test", "0")):
                labels.append({"sample_id": sid, "task_sha256": task_id, "plan_sha256": "plan",
                    "preference": pref, "reference_family": family, "search_interval_mask": mask,
                    "condition_normalized": [1., 2.], "z_m": np.zeros((6, 2)).tolist(), "split": split})
            labels.append({**labels[-1], "sample_id": "wrong_mask", "split": "train", "search_interval_mask": [False] * 6})
        class Scaler:
            def transform_condition(self, encoded): return np.array([1., 2.])
        provider = RetrievalInitializer(labels, Scaler(), identity={"scaler_sha256": "frozen"})
        with patch("v6_4.preference_teacher_dataset.encode_condition", return_value={}):
            proposals = provider(self.task)
        for proposal in proposals.values():
            self.assertEqual(proposal["retrieval_source"]["sample_id"], "a")
            self.assertEqual(proposal["retrieval_distance"], 0.)
            self.assertFalse(np.any(proposal["raw_z_m"]))

    def test_normal_early_stop_and_tool_errors_have_distinct_prefix_receipts(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_physics():
            optimize(self.task, self.preferences, self.identity, self.evidence, tmp,
                search_spec=SearchSpec(candidate_budget=12, proposal_limit=4))
            for budget in (8, 12):
                prefix = read(Path(tmp) / f"prefix_{budget:02d}.json")
                self.assertTrue(prefix["protocol_completed"])
                self.assertEqual(prefix["budget"]["slots_consumed"], 4)
                self.assertEqual(prefix["budget"]["stop_reason"], "PROPOSAL_LIMIT")
        def broken(plan, cid):
            return {"status": "TOOL_ERROR", "prediction_admissible": False, "prediction_rollout_started": False,
                "prediction_metrics": None, "tool_error": {"type": "IOError"}}
        with tempfile.TemporaryDirectory() as tmp:
            self.run_search(tmp, budget=12, evaluator=broken)
            for budget in (4, 8, 12):
                prefix = read(Path(tmp) / f"prefix_{budget:02d}.json")
                self.assertFalse(prefix["protocol_completed"])
                self.assertEqual(prefix["budget"]["slots_consumed"], 1)
                self.assertEqual(prefix["budget"]["stop_reason"], "TOOL_ERROR")

    def test_resume_detects_sealed_prefix_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_search(tmp)
            path = Path(tmp) / "prefix_08.json"
            snapshot = read(path); snapshot["preferences"]["A"]["prediction_metrics"]["I_support"] = 0.
            import json
            path.write_text(json.dumps(snapshot), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "prefix seal"):
                self.run_search(tmp)


if __name__ == "__main__":
    unittest.main()
