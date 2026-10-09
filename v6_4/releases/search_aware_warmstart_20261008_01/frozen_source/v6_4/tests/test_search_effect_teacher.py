"""C.3 teacher evidence, fixed pairs and TRAIN isolation; no physics."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_4.continuous_route_optimizer import optimize
from v6_4.preference_teacher_dataset import preference_labels
from v6_4.route_initializers import FrozenSeedInitializer, search_mask
from v6_4.route_optimizer_protocol import (PreferenceSpec, SearchSpec, VERSIONS,
    active_intervals, build_reference_definition, digest, parameter_plan, read, sha, write)
from v6_4.search_effect_teacher import (BoundArchive, C2_RELEASE, compare_teacher_pairs,
    build_dataset, deduplicate_physical_facts, deduplicate_supervision,
    freeze_teacher_pairs, preference_qualified, run_teacher_search, summarize_teacher_pair)
from v6_4.task_protocol import TaskSpec


def complete_row(cid="C00", d=.031, I=.1, L=1., initial_position=0):
    return {"candidate_id": cid, "prediction_admissible": True, "prediction_task_passed": True,
        "online_guards_passed": True, "prediction_steps": 13500,
        "prediction_metrics": {"I_support": I, "L_full": L, "d_support": d,
            "clearance_status": "MEASURED", "native_state_count": 13501, "saved_horizon_s": 27.},
        "proposal_lineage": [{"initial_position": initial_position}]}


class SearchEffectTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_manifest = read(C2_RELEASE / "snapshot/learning_split_manifest.json")
        cls.facts = [f for f in read(C2_RELEASE / "snapshot/dataset/candidate_facts.json") if f["split"] == "train"]
        cls.task = TaskSpec.from_dict(read(C2_RELEASE / "snapshot/frozen_tasks/c1_mother_00_plus/task.json"))

    def pair(self):
        z = np.zeros((6, 2)); z[2, 0] = .001
        return {"seed_pair_id": "pair", "task_id": self.task.task_id, "task_sha256": self.task.sha256(),
            "mother_id": "mother", "combination": "T_local",
            "proposals": {str(slot): {"raw_z_m": z.tolist(), "family": family,
                "preference": pref, "source": "retrieval", "teacher_construction": {"construction_status": "LOCAL_QUALIFIED_ROUTE_TEACHER"}}
                for slot, pref, family in ((1, "A", "v1"), (3, "B", "v2"))}}

    @staticmethod
    def selection(a="C00", b="C00"):
        return {"budget": {"candidate_budget": 8, "slots_consumed": 4, "stop_reason": "CANDIDATE_BUDGET_EXHAUSTED"},
            "preferences": {"A": {"source_candidate_id": a}, "B": {"source_candidate_id": b}}}

    def test_missing_absolute_path_is_not_physical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "portable_paths.json", {"E:/missing/v6_4/output/run/item.npz": {
                "available": False, "path": None, "sha256": "hash"}})
            archive = BoundArchive(root)
            self.assertIsNone(archive.path("item.npz", required=False))
            self.assertEqual(archive.inventory[0]["availability"], "MISSING")
            with self.assertRaises(FileNotFoundError):
                archive.path("item.npz")

    def test_archive_bytes_must_match_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "snapshot/item.json"; write(p, {"identity": 1})
            write(root / "portable_paths.json", {str(p): {"available": True, "path": "snapshot/item.json", "sha256": sha(p)}})
            archive = BoundArchive(root)
            p.write_text('{"identity": 2}', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                archive.path("item.json")

    def test_physical_dedup_merges_preference_views_and_actual_alias(self):
        fact = copy.deepcopy(self.facts[0]); alias = copy.deepcopy(fact)
        alias["original_candidate_ids"].append("preference_B_view")
        alias["actual_bindings"].append({"alias_of": "R8_A"})
        dedup = deduplicate_physical_facts([fact, alias])
        self.assertEqual(len(dedup), 1)
        self.assertIn("preference_B_view", dedup[0]["original_candidate_ids"])
        self.assertIn({"alias_of": "R8_A"}, dedup[0]["actual_bindings"])

    def test_B_never_uses_A_success_or_below_30mm(self):
        rows = [complete_row(d=.029999)]
        self.assertTrue(preference_qualified(rows[0], "A"))
        self.assertFalse(preference_qualified(rows[0], "B"))
        summary = summarize_teacher_pair(self.pair(), self.selection(), rows, {"A": None, "B": None})
        self.assertTrue(summary["endpoints"]["A"]["qualified_endpoint"])
        self.assertFalse(summary["endpoints"]["B"]["qualified_endpoint"])
        self.assertFalse(any(s["preference"] == "B" for s in summary["initializer_effect_labels"]))

    def test_rule_only_and_common_cache_alias_are_not_effect_labels(self):
        rows = [complete_row()]
        seed = {"initializer_slot": 1, "raw_seed_diagnostics": {"raw_legal": True}, "cache_hit_candidate_id": "C00"}
        result = summarize_teacher_pair(self.pair(), self.selection(), rows, {"A": None, "B": None}, [seed])
        self.assertEqual(result["initializer_effect_labels"], [])
        self.assertEqual(result["endpoints"]["A"]["label_status"], "RULE_ONLY")

    def test_descendant_label_is_original_seed_not_final_z(self):
        rows = [complete_row("C00"), complete_row("C01", initial_position=1)]
        rows[1]["parent_candidate_id"] = "C00"
        seed = {"initializer_slot": 1, "raw_seed_diagnostics": {"raw_legal": True}, "candidate_id": "C00"}
        pair = self.pair()
        result = summarize_teacher_pair(pair, self.selection(a="C01", b=None), rows, {"A": None, "B": None}, [seed])
        self.assertEqual(len(result["initializer_effect_labels"]), 1)
        label = result["initializer_effect_labels"][0]
        self.assertEqual(label["z_m"], pair["proposals"]["1"]["raw_z_m"])
        self.assertEqual(label["source_refs"][0]["fixed_partner"], pair["proposals"]["3"])
        self.assertTrue(label["source_refs"][0]["shared_search_is_not_independent_counterfactuals"])

    def test_raw_rejection_cannot_receive_descendant_supervision(self):
        row = complete_row(initial_position=1)
        seed = {"initializer_slot": 1, "raw_seed_diagnostics": {"raw_legal": False}, "candidate_id": "C00"}
        result = summarize_teacher_pair(self.pair(), self.selection(), [row], {"A": None, "B": None}, [seed])
        self.assertEqual(result["initializer_effect_labels"], [])

    def test_no_plan_preserves_right_censor_and_no_B_label(self):
        result = summarize_teacher_pair(self.pair(), self.selection(a=None, b=None), [complete_row()], {
            "A": {"prediction_metrics": {"I_support": .01, "L_full": .5}}, "B": None})
        self.assertEqual(result["endpoints"]["A"]["first_near_quality_status"], "RIGHT_CENSORED")
        self.assertIsNone(result["endpoints"]["A"]["first_near_quality_slot"])
        self.assertEqual(result["endpoints"]["A"]["first_near_quality_right_censored_budget"], 1)
        self.assertEqual(result["endpoints"]["B"]["first_near_quality_status"], "N/A_NO_HISTORICAL_REFERENCE")

    def test_finite_teacher_selection_retains_complete_ties(self):
        one = summarize_teacher_pair(self.pair(), self.selection(), [complete_row()], {"A": None, "B": None})
        two = copy.deepcopy(one); two["seed_pair_id"] = "two"
        self.assertEqual(compare_teacher_pairs([one, two]), ["pair", "two"])
        two["endpoints"]["B"]["qualified_endpoint"] = False
        self.assertEqual(compare_teacher_pairs([one, two]), ["pair"])

    def test_supervision_dedup_retains_route_and_pair_sources(self):
        pair = self.pair(); z = pair["proposals"]["1"]["raw_z_m"]
        base = {"task_id": self.task.task_id, "task_sha256": self.task.sha256(), "mother_id": "mother", "split": "train",
            "family": "v1", "preference": "A", "z_m": z, "source_types": ["route_quality_example"],
            "source_refs": [{"source_type": "route_quality_example", "physical_candidate_id": "fact"}]}
        effect = {**base, "source_types": ["initializer_effect_example"], "source_refs": [{"seed_pair_id": "pair"}]}
        labels = deduplicate_supervision([base, effect], {self.task.task_id: self.task})
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels[0]["source_types"]), 2)
        self.assertEqual(len(labels[0]["source_refs"]), 2)
        effect["split"] = "test"
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            deduplicate_supervision([effect], {self.task.task_id: self.task})

    def test_pairs_prefreeze_and_transfer_excludes_whole_mother(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(mujoco, "mj_step", side_effect=AssertionError("no physics")):
            run = Path(tmp)
            write(run / "learning_split_manifest.json", self.old_manifest)
            for row in self.old_manifest["tasks"]:
                write(run / row["task_path"], read(C2_RELEASE / "snapshot" / row["task_path"]))
            train = [r for r in self.old_manifest["tasks"] if r["split"] == "train"]
            labels, buckets = preference_labels(self.facts, train)
            history = {"original_train_tasks": train, "candidates": self.facts, "route_quality_labels": labels, "buckets": buckets}
            write(run / "historical_import/history.json", {"mock": True})
            result = freeze_teacher_pairs(run, history)
            self.assertEqual(len(result["pairs"]), 12)
            self.assertEqual(result["maximum_new_candidate_slots"], 96)
            for pair in result["pairs"]:
                for proposal in pair["proposals"].values():
                    if pair["combination"] == "T_transfer":
                        self.assertNotIn(pair["mother_id"], proposal["teacher_construction"]["eligible_mothers"])
                        self.assertNotEqual(pair["mother_id"], (proposal.get("retrieval_source") or {}).get("mother_id"))
            second = freeze_teacher_pairs(run)
            self.assertEqual(result["content_sha256"], second["content_sha256"])

    def test_original_optimizer_eight_slots_and_shared_pool(self):
        task = self.task; definition = build_reference_definition(task); active = active_intervals(definition)
        identity = {"source_identity_sha256": "source", "config_sha256": "config", "task_sha256": task.sha256(),
            "model_contract_sha256": task.model_contract_sha256}
        preferences = [PreferenceSpec(p, task.sha256(), (tuple(definition["intervals_s"][2]),),
            tuple(tuple(definition["intervals_s"][i]) for i in active), "sphere", (("body", "sphere"),)) for p in ("A", "B")]
        proposals = {int(k): v for k, v in self.pair()["proposals"].items()}
        def evaluator(plan, cid):
            return {**complete_row(cid, I=.1 + float(np.sum(plan.z_m))), "prediction_rollout_started": True,
                "costs": {"prediction_physics_steps": 13500}, "status": "PREDICTION_ADMISSIBLE"}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(mujoco, "mj_step", side_effect=AssertionError("no physics")):
            selection = optimize(task, preferences, identity, evaluator, tmp, search_spec=SearchSpec(candidate_budget=8),
                initializer=FrozenSeedInitializer(proposals))
            self.assertLessEqual(selection["budget"]["slots_consumed"], 8)
            self.assertTrue(selection["budget"]["shared_A_B_pool"])
            self.assertEqual(selection["formal_actual_validation"], "NOT_RUN")

    def test_mock_teacher_dataset_and_resume_do_not_need_VAL_labels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(mujoco, "mj_step", side_effect=AssertionError("no physics")):
            run = Path(tmp)
            write(run / "learning_split_manifest.json", self.old_manifest)
            write(run / "plan.json", {"tasks": self.old_manifest["tasks"]})
            for row in self.old_manifest["tasks"]:
                write(run / row["task_path"], read(C2_RELEASE / "snapshot" / row["task_path"]))
            train = [r for r in self.old_manifest["tasks"] if r["split"] == "train"]
            labels, buckets = preference_labels(self.facts, train)
            for s in labels:
                s.update(source_types=["route_quality_example"], source_refs=[{"source_type": "route_quality_example"}])
            history = {"original_train_tasks": train, "candidates": self.facts, "route_quality_labels": labels,
                "buckets": buckets, "files": {}}
            write(run / "historical_import/history.json", history)
            freeze_teacher_pairs(run, history)
            calls = []
            def stream_runner(run, frozen, name, budget, proposals, stage):
                calls.append((frozen["task_id"], name, budget))
                task = TaskSpec.from_dict(read(run / frozen["task_path"]))
                definition = build_reference_definition(task); active = active_intervals(definition)
                prefs = [PreferenceSpec(p, task.sha256(), (tuple(definition["intervals_s"][2]),),
                    tuple(tuple(definition["intervals_s"][i]) for i in active), "sphere", (("body", "sphere"),)) for p in ("A", "B")]
                identity = {"source_identity_sha256": "source", "config_sha256": "config", "task_sha256": task.sha256(),
                    "model_contract_sha256": task.model_contract_sha256}
                root = run / "teacher_search" / frozen["task_id"] / name
                def evidence(plan, cid):
                    return {**complete_row(cid, I=.1 + float(np.sum(plan.z_m))), "prediction_rollout_started": True,
                        "costs": {"prediction_physics_steps": 13500}, "status": "PREDICTION_ADMISSIBLE"}
                selection = optimize(task, prefs, identity, evidence, root / "planning" / task.task_id,
                    search_spec=SearchSpec(candidate_budget=budget), initializer=FrozenSeedInitializer(proposals))
                return {"path": str(root), "selection": selection}
            result = run_teacher_search(run, stream_runner)
            self.assertEqual(len(calls), 12)
            self.assertLessEqual(result["slots_consumed"], 96)
            run_teacher_search(run, lambda *a, **k: self.fail("completed search reran"))
            dataset = build_dataset(run)
            self.assertTrue(all(s["split"] == "train" for s in dataset.samples))
            self.assertEqual(len(dataset.indices("val")), 0)
            self.assertEqual(dataset.z_m.shape[1:], (6, 2))
            self.assertEqual(dataset.search_masks.shape[1:], (6,))
            self.assertEqual(set(dataset.condition_scaler.fit_task_sha256), {r["task_sha256"] for r in train})
            self.assertTrue(dataset.manifest["D_S_N_identical_pool"])
            self.assertEqual(build_dataset(run).manifest, dataset.manifest)


if __name__ == "__main__":
    unittest.main()
