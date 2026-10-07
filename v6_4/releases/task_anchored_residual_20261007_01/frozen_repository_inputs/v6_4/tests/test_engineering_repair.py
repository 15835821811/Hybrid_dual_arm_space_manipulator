import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v6_4.contracts import TrajectoryProposal
from v6_4.engineering_repair import (
    BUDGET, BUDGET_V2, REPAIR_TAG, REPAIR_TAG_V2, ROOT, _ProjectionPlanner, _domain_checks,
    evaluate_engineering_candidate_set, fixed_joint_prior_seed, load_selected_proposal, repair_proposal,
)
from v6_4.trajectory_codec import CubicBSplineCodec


class EngineeringRepairTests(unittest.TestCase):
    def setUp(self):
        self.task = SimpleNamespace(task_id="engineering_unit", initial_planner_q=np.zeros(17),
            initial_planner_dq=np.zeros(17), initial_qpos=np.zeros(81), initial_qvel=np.zeros(79),
            base_pose=(0., 0., 0., 1., 0., 0., 0.), requirements=(),
            sha256=lambda: "a" * 64, to_dict=lambda: {"task_id": "engineering_unit"})
        self.spec = SimpleNamespace(planner_lower=np.full(17, -2.), planner_upper=np.full(17, 2.),
            planner_velocity_limits=np.ones(17), planner_acceleration_limits=np.ones(17))
        self.codec = CubicBSplineCodec(self.task.initial_planner_q, self.task.initial_planner_dq)
        self.raw = TrajectoryProposal.from_controls(self.task, np.zeros((30, 17)), origin="diffusion",
            seed=64, metadata={"checkpoint_sha256": "b" * 64})
        source = ROOT / "v6_4/engineering_repair.py"
        self.sources = {"v6_4/engineering_repair.py": {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}}
        self.domain = SimpleNamespace(work_domain_lower_rad=np.full(10, -1.),
                                      work_domain_upper_rad=np.full(10, 1.))

    def prediction(self, controls):
        samples = self.codec.sample(controls, np.arange(1351) * .02)
        u = np.linspace(0., 1., 1351)
        base = np.zeros((1351, 7)); base[:, 3] = 1.
        return {"time": np.arange(1351) * .02, **samples, "base_pose": base,
                "rigid_position": np.c_[.4 * u, np.zeros((1351, 2))],
                "continuum_position": np.c_[np.zeros(1351), .1 * u, np.zeros(1351)]}

    def provider(self, task, controls, spec):
        provider = SimpleNamespace(prediction=self.prediction(controls), controls=controls, metadata={})
        return provider, SimpleNamespace(model=object())

    def gate(self, task, proposal, spec, *, provider, output_path):
        value = {"passed": bool(proposal.postprocessing), "raw_passed": False,
                 "elapsed_wall_s": .001, "nominal_screen": {"geometry": {"query_count": 10}}}
        Path(output_path).write_text(json.dumps(value), encoding="utf-8")
        return value

    def optimize(self, planner, task, codec, controls, *args):
        self.assertEqual(planner.optimization_rounds, 8)
        self.assertEqual(planner.ik_max_evaluations, 35)
        self.assertEqual(planner.geometry_optimization_rounds, 0)
        controls = controls.copy(); controls[-1, 0] = .1
        return controls, [{"round": i, "ik": []} for i in range(8)]

    def mocks(self, *, optimize=None, gate=None, allow_geometry=False):
        stack = ExitStack()
        for name, kwargs in [
            ("_source_records", {"return_value": self.sources}),
            ("_asset_records", {"return_value": {}}),
            ("default_continuum_model_spec", {"return_value": self.domain}),
            ("prepare_provider", {"side_effect": self.provider}),
            ("scenario_from_task", {"return_value": object()}),
            ("gate_proposal", {"side_effect": gate or self.gate}),
            ("mujoco.MjData", {"side_effect": lambda _: SimpleNamespace(
                qpos=np.zeros(81), qvel=np.zeros(79), ctrl=np.zeros(67))}),
            ("mujoco.mj_forward", {"return_value": None}),
            ("mujoco.mj_step", {"side_effect": AssertionError("physics forbidden")}),
        ]:
            stack.enter_context(patch("v6_4.engineering_repair." + name, **kwargs))
        stack.enter_context(patch("v6_4.engineering_repair.TeacherPlanner._optimize", autospec=True,
                                  side_effect=optimize or self.optimize))
        if not allow_geometry:
            stack.enter_context(patch("v6_4.engineering_repair.TeacherPlanner._optimize_geometry",
                                      side_effect=AssertionError("geometry optimization forbidden")))
        stack.enter_context(patch("v6_4.engineering_repair.TeacherPlanner.propose",
                                  side_effect=AssertionError("intrinsic teacher generation forbidden")))
        return stack

    def test_tagged_repair_preserves_raw_controls_sha_and_fixed_start(self):
        before = self.raw.to_dict(); sha = self.raw.sha256()
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            output = Path(temporary) / "repair"
            report = repair_proposal(self.task, self.raw, output, spec=self.spec)
            proposal = load_selected_proposal(report, output, K=1)
            self.assertEqual(proposal.postprocessing, (REPAIR_TAG,))
            self.assertEqual(proposal.metadata["input_proposal_sha256"], sha)
            self.assertEqual(proposal.metadata["input_raw_metadata"]["checkpoint_sha256"], "b" * 64)
            self.assertEqual(proposal.origin, "diffusion"); self.assertEqual(proposal.seed, 64)
            row = report["records"][0]
            self.assertFalse(row["raw_proposal_passed"]); self.assertFalse(row["repaired_raw_gate_passed"])
            self.assertTrue(row["repaired_full_gate_passed"])
            self.assertEqual(row["modification"]["fixed_C0_C1_change_max_rad"], 0.)
            self.assertEqual(row["physics_steps"], 0)
            self.assertAlmostEqual(row["score"], .5)
            self.assertTrue((output / "source_snapshot/v6_4/engineering_repair.py").is_file())
            np.testing.assert_array_equal(self.codec.decode_free(proposal.free_controls)[:2], self.codec.fixed_controls)
        self.assertEqual(self.raw.to_dict(), before); self.assertEqual(self.raw.sha256(), sha)

    def test_all_generation_failures_keep_eight_slots_and_k1_prefix(self):
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            report = evaluate_engineering_candidate_set(self.task, [None] * 8,
                Path(temporary) / "repair", spec=self.spec)
            self.assertEqual(len(report["records"]), 8)
            self.assertIsNone(report["selectedIndex"]); self.assertIsNone(report["selectedIndexK1"])
            self.assertTrue(report["evidence_valid"])
            self.assertTrue(all(r["physics_steps"] == 0 for r in report["records"]))

    def test_k1_candidate_zero_never_becomes_first_survivor_and_ties_keep_index(self):
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            report = evaluate_engineering_candidate_set(self.task, [None, self.raw, self.raw],
                Path(temporary) / "repair", spec=self.spec)
            self.assertIsNone(report["selectedIndexK1"])
            self.assertEqual(report["selectedIndex"], 1)

    def test_original_acceleration_limit_rejects_even_if_full_gate_passes(self):
        self.spec.planner_acceleration_limits[:] = 0.
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            report = repair_proposal(self.task, self.raw, Path(temporary) / "repair", spec=self.spec)
            row = report["records"][0]
            self.assertTrue(row["repaired_full_gate_passed"])
            self.assertFalse(row["domain_checks"]["original_reference_acceleration_limits"])
            self.assertFalse(row["eligible_for_selection"]); self.assertIsNone(report["selectedIndex"])

    def test_no_geometry_optimization_and_no_bypass_when_final_gate_rejects(self):
        def reject(*args, **kwargs):
            value = self.gate(*args, **kwargs)
            value["passed"] = False
            Path(kwargs["output_path"]).write_text(json.dumps(value), encoding="utf-8")
            return value
        with tempfile.TemporaryDirectory() as temporary, self.mocks(gate=reject):
            report = repair_proposal(self.task, self.raw, Path(temporary) / "repair", spec=self.spec)
            self.assertEqual(report["records"][0]["status"], "REPAIR_REJECTED")
            self.assertIsNone(report["selectedIndex"])

    def test_projection_exception_retains_original_and_other_slots(self):
        def fail(*args):
            raise RuntimeError("finite IK failed")
        with tempfile.TemporaryDirectory() as temporary, self.mocks(optimize=fail):
            output = Path(temporary) / "repair"
            report = evaluate_engineering_candidate_set(self.task, [self.raw, None], output, spec=self.spec)
            self.assertEqual(len(report["records"]), 2)
            self.assertEqual(report["records"][0]["failure"]["phase"], "task_projection")
            original = TrajectoryProposal.from_dict(json.loads((output / "candidate_000/input_raw_proposal.json").read_text()))
            self.assertEqual(original.sha256(), self.raw.sha256())
            self.assertTrue((output / "candidate_000/input_raw_gate.json").is_file())
            self.assertFalse((output / "candidate_000/repaired_proposal.json").exists())
            self.assertGreaterEqual(report["records"][0]["cost"]["task_projection_wall_s"], 0.)

    def test_input_fallback_or_already_repaired_is_saved_and_rejected(self):
        repaired = TrajectoryProposal.from_controls(self.task, self.raw.free_controls, origin="diffusion", postprocessing=("prior_repair",))
        fallback = TrajectoryProposal.from_controls(self.task, self.raw.free_controls, origin="fallback")
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            report = evaluate_engineering_candidate_set(self.task, [repaired, fallback], Path(temporary) / "repair", spec=self.spec)
            self.assertIsNone(report["selectedIndex"])
            self.assertTrue(all("non-fallback" in r["failure"]["message"] for r in report["records"]))

    def test_source_mutation_invalidates_selection_without_erasing_completed_records(self):
        altered = {key: {**value, "sha256": "f" * 64} for key, value in self.sources.items()}
        with tempfile.TemporaryDirectory() as temporary, self.mocks():
            with patch("v6_4.engineering_repair._source_records", side_effect=[self.sources, altered]):
                report = repair_proposal(self.task, self.raw, Path(temporary) / "repair", spec=self.spec)
            self.assertFalse(report["sources_unchanged"])
            self.assertIsNone(report["selectedIndex"])
            self.assertTrue(report["records"][0]["repaired_full_gate_passed"])

    def test_tampered_saved_proposal_or_gate_cannot_be_loaded(self):
        for artifact in ("repaired_proposal.json", "repaired_full_gate.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary, self.mocks():
                output = Path(temporary) / "repair"
                report = repair_proposal(self.task, self.raw, output, spec=self.spec)
                with (output / "candidate_000" / artifact).open("a") as stream:
                    stream.write(" ")
                with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                    load_selected_proposal(report, output)

    def test_partial_ik_journal_preserves_success_and_failure(self):
        point = SimpleNamespace(point_id="terminal", time_s=27.)
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "ik.jsonl"
            planner = _ProjectionPlanner(self.spec, 64, journal)
            with patch("v6_4.engineering_repair.TeacherPlanner._anchor_configuration",
                       return_value=(np.zeros(17), {"nfev": 4, "status": 1})):
                planner._anchor_configuration(None, None, None, [point])
            with patch("v6_4.engineering_repair.TeacherPlanner._anchor_configuration", side_effect=ValueError("bad IK")):
                with self.assertRaises(ValueError):
                    planner._anchor_configuration(None, None, None, [point])
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([r["call_status"] for r in records], ["COMPLETED", "FAILED"])
            self.assertEqual(records[0]["nfev"], 4)

    def test_budget_not_configurable_and_original_pcc_domain_rejects(self):
        self.assertEqual(BUDGET["task_projection_outer_rounds"], 8)
        self.assertEqual(BUDGET["ik_max_nfev_per_requirement_time"], 35)
        self.assertEqual(BUDGET["geometry_optimization_rounds"], 0)
        controls = self.codec.decode_free(np.tile(np.r_[1.1, np.zeros(16)], (30, 1)))
        with patch("v6_4.engineering_repair.default_continuum_model_spec", return_value=self.domain):
            checks = _domain_checks(self.task, self.codec, controls, self.spec)
        self.assertTrue(checks["control_point_position_domain"])
        self.assertFalse(checks["original_pcc_work_domain"])

    def test_v2_stages_fixed_budget_then_full_gate_and_v1_does_not_run_geometry(self):
        calls = []
        def anchors(planner, task, codec, controls, *args):
            calls.append(planner.projection_stage)
            self.assertEqual(planner.optimization_rounds, 8)
            self.assertEqual(planner.ik_max_evaluations, 35)
            controls = controls.copy(); controls[-1, 0] += .02
            return controls, [{"round": i, "ik": []} for i in range(8)]
        def geometry(planner, task, codec, controls, *args):
            calls.append("geometry")
            self.assertEqual(planner.geometry_optimization_rounds, 8)
            self.assertEqual(planner.geometry_line_search_steps, 4)
            self.assertEqual(planner.geometry_stride, 10)
            controls = controls.copy(); controls[-2, 0] += .01
            return controls, {"rounds": [], "private_prediction_count": 1,
                "native_distance_queries": 10, "private_configuration_perturbations": 0}
        # The fake verifier retains the native pair argument passed to the
        # frozen optimizer; no alternative policy is manufactured by repair.
        original_provider = self.provider
        def provider(*args):
            result, verifier = original_provider(*args); verifier.pairs = ("original_pair",)
            return result, verifier
        with tempfile.TemporaryDirectory() as temporary, self.mocks(optimize=anchors, allow_geometry=True):
            with patch("v6_4.engineering_repair.prepare_provider", side_effect=provider), \
                 patch("v6_4.engineering_repair.TeacherPlanner._optimize_geometry", autospec=True, side_effect=geometry):
                output = Path(temporary) / "v2"
                report = repair_proposal(self.task, self.raw, output, spec=self.spec, version="v2")
                proposal = load_selected_proposal(report, output, K=1)
                self.assertEqual(calls, ["pre_geometry_task_projection", "geometry", "post_geometry_task_projection"])
                self.assertEqual(proposal.postprocessing, (REPAIR_TAG_V2,))
                self.assertFalse(report["records"][0]["repaired_raw_gate_passed"])
                self.assertEqual(report["budget"], BUDGET_V2)
                self.assertTrue((output / "candidate_000/geometry_optimization.json").is_file())
                self.assertTrue((output / "candidate_000/post_geometry_task_projection_rounds.json").is_file())
            calls.clear()
            report = repair_proposal(self.task, self.raw, Path(temporary) / "v1", spec=self.spec)
            self.assertEqual(calls, ["task_projection"])
            self.assertEqual(report["repair_tag"], REPAIR_TAG)
            self.assertEqual(report["budget"], BUDGET)

    def test_v2_geometry_failure_keeps_preprojection_and_raw_gate(self):
        def provider(*args):
            result, verifier = self.provider(*args); verifier.pairs = ("original_pair",)
            return result, verifier
        def anchors(planner, task, codec, controls, *args):
            return controls.copy(), [{"round": i, "ik": []} for i in range(8)]
        with tempfile.TemporaryDirectory() as temporary, self.mocks(optimize=anchors):
            with patch("v6_4.engineering_repair.prepare_provider", side_effect=provider), \
                 patch("v6_4.engineering_repair.TeacherPlanner._optimize_geometry", side_effect=ValueError("geometry failed")):
                output = Path(temporary) / "v2"
                report = repair_proposal(self.task, self.raw, output, spec=self.spec, version="v2")
            row = report["records"][0]
            self.assertEqual(row["failure"]["phase"], "native_geometry_optimization")
            self.assertIsNone(report["selectedIndex"])
            self.assertTrue((output / "candidate_000/pre_geometry_control_points.npz").is_file())
            self.assertTrue((output / "candidate_000/input_raw_gate.json").is_file())
            self.assertGreaterEqual(row["cost"]["native_geometry_optimization_wall_s"], 0.)

    def test_partial_native_call_unknown_query_count_is_not_fabricated_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            planner = _ProjectionPlanner(self.spec, 64, Path(temporary) / "ik.jsonl", geometry_rounds=8)
            with patch("v6_4.engineering_repair.TeacherPlanner._native_geometry", side_effect=ValueError("nonfinite native")):
                with self.assertRaises(ValueError):
                    planner._native_geometry(None, None, None, None)
            row = json.loads(planner.geometry_journal.read_text())
            self.assertEqual(row["status"], "FAILED")
            self.assertIsNone(row["native_distance_queries"])

    def test_v2_postprojection_failure_retains_geometry_output_and_failed_slot(self):
        def provider(*args):
            result, verifier = self.provider(*args); verifier.pairs = ("original_pair",)
            return result, verifier
        def anchors(planner, task, codec, controls, *args):
            if planner.projection_stage == "post_geometry_task_projection":
                raise RuntimeError("post task projection failed")
            return controls.copy(), [{"round": i, "ik": []} for i in range(8)]
        geometry = {"rounds": [], "private_prediction_count": 1,
                    "native_distance_queries": 10, "private_configuration_perturbations": 0}
        with tempfile.TemporaryDirectory() as temporary, self.mocks(optimize=anchors, allow_geometry=True):
            with patch("v6_4.engineering_repair.prepare_provider", side_effect=provider), \
                 patch("v6_4.engineering_repair.TeacherPlanner._optimize_geometry",
                       side_effect=lambda task, codec, controls, *args: (controls.copy(), geometry)):
                output = Path(temporary) / "v2"
                report = repair_proposal(self.task, self.raw, output, spec=self.spec, version="v2")
            row = report["records"][0]
            self.assertEqual(row["failure"]["phase"], "post_geometry_task_projection")
            self.assertIsNone(report["selectedIndex"])
            self.assertTrue((output / "candidate_000/post_geometry_control_points.npz").is_file())
            self.assertTrue((output / "candidate_000/geometry_optimization.json").is_file())
            self.assertGreaterEqual(row["cost"]["post_geometry_task_projection_wall_s"], 0.)

    def test_fixed_joint_prior_is_named_seed_from_only_declared_old_scene04(self):
        times = np.arange(1, 13501) * .002
        q = np.zeros((13500, 17)); q[:, 0] = .01 * times / 27.
        lineage = {"source_split": "bootstrap", "source_group_id": "original_scene04", "sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            with patch("v6_4.engineering_repair.load_bootstrap", return_value=(times, q, lineage)) as load:
                output = Path(temporary) / "seed"
                proposal = fixed_joint_prior_seed(self.task, output)
            self.assertEqual(load.call_args.args[0].name, "v6_lite_scenario_04.npz")
            self.assertEqual(proposal.origin, "fixed_joint_prior")
            self.assertFalse(proposal.metadata["original_fixed_cartesian_baseline"])
            self.assertFalse(proposal.metadata["gate_or_actual_success_established"])
            np.testing.assert_array_equal(self.codec.decode_free(proposal.free_controls)[:2], self.codec.fixed_controls)

    def test_unknown_version_rejected_before_output_and_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v3"
            with self.assertRaises(ValueError):
                repair_proposal(self.task, self.raw, output, spec=self.spec, version="v3")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
