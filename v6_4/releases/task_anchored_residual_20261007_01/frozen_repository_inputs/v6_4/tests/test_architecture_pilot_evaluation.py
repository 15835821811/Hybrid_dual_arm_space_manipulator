"""Finite protocol/evidence tests with synthetic data; no model training or physics."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import architecture_pilot_evaluation as pilot
from v6_4.contracts import TrajectoryProposal
from v6_4.tests.test_learning_dataset import fixture_task
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_lite.hierarchical_qp import HierarchicalQPConfig


def six_tasks():
    families = ("end_effector_detour", "mid_arm_detour", "multiple_routes")
    return tuple(replace(fixture_task("test", f"task_{i}", f"group_{i}", shift=i*.02), family=families[i//2])
                 for i in range(6))


def write_suite(path, tasks):
    path.write_text(json.dumps({"tasks": [t.to_dict() for t in tasks],
        "task_hashes": {t.task_id: t.sha256() for t in tasks},
        "splits": {"train": [], "val": [], "test": [t.task_id for t in tasks]}}), encoding="utf-8")


def fake_prediction(task, controls):
    sample = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq).sample(controls, np.arange(1351)*.02)
    identity = np.tile(np.eye(3), (1351, 1, 1))
    result = {**sample, "time": np.arange(1351)*.02,
        "target_position": np.tile(task.target_pose[:3], (1351, 1)), "target_rotation": identity,
        "continuum_position": np.tile(task.requirements[0].position_m, (1351, 1)),
        "continuum_rotation": identity, "rigid_rotation": identity,
        "rigid_position": np.tile(np.asarray(task.target_pose[:3])+task.requirements[1].position_m, (1351, 1)),
        "full_qpos": np.tile(task.initial_qpos, (1351, 1))}
    return result


class PilotEvaluationTests(unittest.TestCase):
    def test_freeze_and_validate_exact_six_tasks_and_four_paired_latents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); tasks = six_tasks()
            write_suite(root / "tasks.json", tasks)
            config = pilot.freeze_candidate_config(root / "tasks.json", root / "candidate_config.json")
            self.assertEqual(config["candidate_slots_total"], 48)
            self.assertEqual(config["maximum_actual_attempts"], 12)
            self.assertEqual(config["candidate_seeds"], list(pilot.CANDIDATE_SEEDS))
            self.assertEqual(config["K1_candidate_index"], 0)
            self.assertEqual(config["diagnostic_shadow_native_evaluations"], 3)
            loaded, actual_tasks = pilot.load_candidate_config(root / "candidate_config.json")
            self.assertEqual(loaded, config); self.assertEqual(len(actual_tasks), 6)
            with self.assertRaises(FileExistsError):
                pilot.freeze_candidate_config(root / "tasks.json", root / "candidate_config.json")
            changed = deepcopy(config); changed["physics_period_s"] = .004
            (root / "invalid.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen pilot"):
                pilot.load_candidate_config(root / "invalid.json")
            write_suite(root / "two.json", tasks[:2])
            with self.assertRaisesRegex(ValueError, "six independent"):
                pilot.freeze_candidate_config(root / "two.json", root / "bad_config.json")

    def test_sample_preserves_all_48_slots_paired_latent_bytes_and_no_clipping(self):
        class FakeSampler:
            def __init__(self, model_name):
                self.model = SimpleNamespace(architecture_identity=lambda: {"model": model_name, "parameters": 1})
            def sample(self, task, latent):
                return np.full((30, 17), 19.) + latent.astype(float)*.01, {"synthetic_unit_fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); tasks = six_tasks()
            write_suite(root / "tasks.json", tasks)
            pilot.freeze_candidate_config(root / "tasks.json", root / "candidate_config.json")
            for model_name in pilot.MODELS:
                directory = root / "training" / "models" / model_name
                directory.mkdir(parents=True); (directory / "checkpoint.pt").write_bytes(b"fixture")
            with patch.object(pilot, "_load_sampler", side_effect=lambda path, device: FakeSampler(path.parent.name)), \
                    patch.object(pilot, "_sources", return_value={"synthetic": "unit_fixture"}):
                report = pilot.sample_candidates(root / "candidate_config.json", root / "training", root / "sample")
            self.assertEqual(report["slot_count"], 48); self.assertEqual(report["generated_count"], 48)
            for task in tasks:
                for index in range(4):
                    left, left_row = pilot._load_candidate(root / "sample", "M0", task, index)
                    right, right_row = pilot._load_candidate(root / "sample", "M1", task, index)
                    self.assertEqual(left_row["latent_sha256"], right_row["latent_sha256"])
                    np.testing.assert_array_equal(left.free_controls, right.free_controls)
                    self.assertGreater(left.free_controls.min(), np.pi)
                    self.assertTrue(left_row["fixed_C0_C1_exact"])
                    self.assertFalse(left.postprocessing)
            altered = root / "sample" / "M0" / tasks[0].task_id / "candidate_000" / "latent.npy"
            np.save(altered, np.zeros((30, 17), dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "latent bytes"):
                pilot._load_candidate(root / "sample", "M0", tasks[0], 0)

    def test_raw_short_circuit_retains_complete_prediction_and_NOT_RUN_geometry(self):
        task = fixture_task()
        free = np.full((30, 17), 2.)
        proposal = TrajectoryProposal.from_controls(task, free, origin="diffusion", seed=pilot.CANDIDATE_SEEDS[0])
        codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
        full = codec.decode_free(free)
        provider = SimpleNamespace(prediction=fake_prediction(task, full), metadata={"fixture": True})
        spec = SimpleNamespace(planner_lower=np.full(17, -np.pi), planner_upper=np.full(17, np.pi),
                               planner_velocity_limits=np.ones(17))
        domain = SimpleNamespace(work_domain_lower_rad=np.full(10, -1.), work_domain_upper_rad=np.ones(10))
        gate = {"raw_passed": False, "errors": [], "nominal_screen": {"geometry": None, "errors": []}, "elapsed_wall_s": .01}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(pilot, "prepare_provider", return_value=(provider, None)), \
                patch.object(pilot, "save_reference") as saved, \
                patch.object(pilot, "gate_proposal", return_value=gate), \
                patch("v6_lite.continuum_model_spec.default_continuum_model_spec", return_value=domain):
            row = pilot.evaluate_raw_candidate(task, proposal, Path(temporary)/"raw", spec=spec)
        self.assertEqual(saved.call_count, 1)
        self.assertEqual(len(provider.prediction["time"]), 1351)
        self.assertEqual(row["geometry_status"], "NOT_RUN"); self.assertIsNone(row["geometry_passed"])
        self.assertFalse(row["raw_passed"])
        self.assertTrue(row["metrics"]["initial_boundary_passed"])
        self.assertGreater(row["metrics"]["velocity_violation_max_rad_s"], 0.)
        self.assertIn("pcc_work_domain", row["rejection_reasons"])
        self.assertEqual(row["raw_closed_loop"], "NOT_RUN")

    def test_raw_summary_keeps_K1_slot_zero_and_full_denominators(self):
        metric = {name: True for name in ("initial_boundary_passed", "task_passed", "control_point_range_passed",
            "reconstructed_range_passed", "pcc_work_domain_passed", "velocity_passed")}
        rows = [{"model": model, "task_id": f"task_{task}", "candidate_index": index,
                 "controls_sha256": f"hash{task}", "raw_passed": index == 2, "metrics": metric,
                 "geometry_status": "NOT_RUN", "geometry_passed": None, "rejection_reasons": []}
                for model in pilot.MODELS for task in range(6) for index in range(4)]
        report = pilot.summarize_raw(rows)
        self.assertEqual(report["M0"]["slot_denominator"], 24)
        self.assertEqual(report["M0"]["task_denominator"], 6)
        self.assertEqual(report["M0"]["raw_comprehensive_passed"], 6)
        self.assertEqual(report["M0"]["duplicate_slots"], 18)
        self.assertFalse(report["M0"]["tasks"]["task_0"]["K1_raw_passed"])
        self.assertTrue(report["M0"]["tasks"]["task_0"]["any_of_K4_raw_passed"])

    def test_repair_rejection_counts_and_finished_attempt_cannot_execute_again(self):
        task = fixture_task()
        proposal = TrajectoryProposal.from_controls(task, np.zeros((30, 17)), origin="diffusion",
                                                     seed=pilot.CANDIDATE_SEEDS[0])
        repair_report = {"records": [{"status": "REPAIR_REJECTED", "modification": None}], "total_invocation_wall_s": .1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "qp.json"
            config.write_text(json.dumps(asdict(HierarchicalQPConfig(enable_pcc_braking_guard=True))), encoding="utf-8")
            with patch.object(pilot, "A1_EXECUTION_CONFIG_SHA256", pilot._sha(config)), \
                    patch.object(pilot, "_load_candidate", return_value=(proposal, {"status": "RAW_GENERATED"})), \
                    patch.object(pilot, "_sources", return_value={"synthetic": "unit_fixture"}), \
                    patch("v6_4.engineering_repair.repair_proposal", return_value=repair_report) as repair, \
                    patch("v6_4.engineering_repair.load_selected_proposal", return_value=None) as select, \
                    patch("v6_4.reference_execution_repair.execute") as execute:
                row = pilot.execute_k1_attempt(task, "M0", root/"sampling", root/"attempt", config)
                repeat = pilot.execute_k1_attempt(task, "M0", root/"sampling", root/"attempt", config)
            self.assertEqual(repair.call_count, 1); self.assertEqual(select.call_args.kwargs["K"], 1)
            self.assertEqual(repair.call_args.kwargs["version"], "v2")
            execute.assert_not_called(); self.assertEqual(row, repeat)
            self.assertEqual(row["status"], "REPAIR_REJECTED"); self.assertEqual(row["actual_steps"], 0)
            self.assertEqual(row["task_denominator_per_model"], 6)
            self.assertEqual(pilot.summarize_actual([row])["M0"]["full_27s_success"], 0)

    def test_unfinished_attempt_blocks_reexecution_and_partial_step_count_is_retained(self):
        task = fixture_task()
        proposal = TrajectoryProposal.from_controls(task, np.zeros((30, 17)), origin="diffusion")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/"attempt").mkdir()
            config = root/"qp.json"; config.write_text("{}", encoding="utf-8")
            with patch.object(pilot, "A1_EXECUTION_CONFIG_SHA256", pilot._sha(config)), \
                    patch.object(pilot, "_load_candidate", return_value=(proposal, {"status": "RAW_GENERATED"})), \
                    patch("v6_4.engineering_repair.repair_proposal") as repair, \
                    patch("v6_4.reference_execution_repair.execute") as execute:
                with self.assertRaisesRegex(FileExistsError, "never automatically restart"):
                    pilot.execute_k1_attempt(task, "M0", root/"sampling", root/"attempt", config)
                repair.assert_not_called(); execute.assert_not_called()
            (root/"actual"/"failures").mkdir(parents=True)
            np.savez(root/"actual"/"failures"/"scene_interval_partial_trace.npz", torque=np.zeros((30, 67)))
            row = pilot._actual_result_row(task, "M0", proposal, {"records": []},
                {"status": "EXECUTION_OR_PIPELINE_FAILURE", "task_success": False, "complete": False}, root/"actual", .2)
            self.assertEqual(row["actual_steps"], 30); self.assertTrue(row["entered_actual"])
            self.assertFalse(row["full_27s_success"])

    def test_full_success_requires_independent_execution_interval_and_native_evidence(self):
        task = fixture_task()
        evaluation = {"task_success": True, "complete": True, "evidence_valid": True,
            "execution_contract": {"passed": True}, "independent_interval": {"passed": True},
            "native_geometry": {"passed": False}}
        result = {"status": "TASK_COMPLETED", "task_success": True, "complete": True, "evaluation": evaluation}
        with patch.object(pilot, "_actual_steps", return_value=(13500, [])):
            row = pilot._actual_result_row(task, "M1", None, {"records": []}, result, Path("not_real"), .1)
            self.assertFalse(row["full_27s_success"])
            result["evaluation"]["native_geometry"]["passed"] = True
            row = pilot._actual_result_row(task, "M1", None, {"records": []}, result, Path("not_real"), .1)
            self.assertTrue(row["full_27s_success"])

    def test_unscored_native_geometry_is_separate_from_main_raw_acceptance(self):
        task = fixture_task()
        proposal = TrajectoryProposal.from_controls(task, np.zeros((30, 17)), origin="diffusion")
        prediction = {"time": np.arange(1351)*.02, "full_qpos": np.zeros((1351, 81))}
        verifier = SimpleNamespace(verify_qpos_sequence=lambda qpos: SimpleNamespace(to_dict=lambda: {
            "feasible": True, "minimum_clearance": .006, "distance_query_count": 3954377}))
        with tempfile.TemporaryDirectory() as temporary, patch.object(pilot, "prepare_provider", return_value=(
                SimpleNamespace(prediction=prediction), verifier)):
            report = pilot.diagnostic_shadow_geometry(task, proposal, Path(temporary)/"diagnostic", spec=object())
        self.assertEqual(report["state_count"], 1351)
        self.assertEqual(report["minimum_clearance_threshold_m"], .005)
        self.assertFalse(report["changes_raw_acceptance"])
        self.assertFalse(report["query_pass_establishes_task_or_closed_loop_success"])
        self.assertEqual(report["new_actual_steps"], 0)


if __name__ == "__main__":
    unittest.main()
