"""Read-only shadow partition and frozen-reach screening checks."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import mujoco
import numpy as np

from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator, IntervalPartition
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shadow_b2_interval_cbf import (
    partition_from_bounded_result, select_intervals_by_frozen_reach,
)
from v6_lite.shape_clearance import target_box_from_mujoco


class B2ShadowTests(unittest.TestCase):
    def test_b1_leaf_recovery_and_reach_screen_include_all_possible_actives(self) -> None:
        robot = default_v6_lite_robot_spec()
        model = robot.compile_dynamic_model()
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        bounded = PCCBoundedClearanceEvaluator(evaluator.shape_model)
        low_level = data.qpos[evaluator.qpos_ids[:60]]
        q = evaluator.shape_spec.project_actual_configuration(low_level).planner_configuration
        from v6_lite.continuum_shape_model import transform_from_free_qpos
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        box = target_box_from_mujoco(model, data, evaluator.target_geom_id)
        result = bounded.evaluate(q, base, box, max_evaluations=31)
        partition = partition_from_bounded_result(
            result, evaluator.shape_spec.segment_lengths_m
        )
        self.assertTrue(partition.coverage(evaluator.shape_spec.segment_lengths_m).coverage_complete)
        self.assertEqual(len(partition.leaves), result.interval_count)
        generalized_map = evaluator.reaction_map(data)
        cfg = HierarchicalQPConfig()
        selected, reach = select_intervals_by_frozen_reach(
            result, partition, evaluator, data, cfg, generalized_map,
        )
        planner_q = robot.low_level_to_planner @ data.qpos[evaluator.qpos_ids]
        lower, upper, ramp_speed, box_valid = ramp_velocity_abs_bound(
            robot, cfg, planner_q, np.zeros(17),
        )
        self.assertTrue(box_valid)
        self.assertTrue(np.all(lower <= upper))
        self.assertTrue(np.all(ramp_speed <= cfg.velocity_limit_scale
                               * robot.planner_velocity_limits + 1e-12))
        tight_selected, tight_reach = select_intervals_by_frozen_reach(
            result, partition, evaluator, data, cfg, generalized_map,
            velocity_abs_bound=ramp_speed,
        )
        self.assertLessEqual(len(tight_selected), len(selected))
        self.assertTrue(tight_selected <= selected)
        self.assertTrue(all(tight_reach[key] <= reach[key] + 1e-12
                            for key in reach))
        self.assertEqual(set(reach), {x.interval_id for x in partition.leaves})
        batch = evaluator.evaluate_state(
            data, partition, generalized_map=generalized_map,
            derivative_interval_ids=selected,
        )
        self.assertEqual(batch.point_evaluation_count, len(partition.leaves))
        for row in batch.rows:
            if row.distance_lower_bound_m <= 0.100:
                self.assertIn(row.interval_id, selected)
            if row.interval_id not in selected:
                self.assertGreater(row.distance_lower_bound_m,
                                   0.100 + reach[row.interval_id] - 1e-8)
            if row.distance_lower_bound_m <= cfg.pcc_clearance_activation_m:
                self.assertIn(row.interval_id, tight_selected)
            if row.interval_id not in tight_selected:
                self.assertGreater(row.distance_lower_bound_m,
                                   cfg.pcc_clearance_activation_m
                                   + tight_reach[row.interval_id] - 1e-8)

    def test_ramp_rate_bound_covers_start_and_every_box_endpoint(self) -> None:
        robot = default_v6_lite_robot_spec()
        cfg = HierarchicalQPConfig()
        q = 0.5 * (robot.planner_lower + robot.planner_upper)
        start = 0.2 * cfg.velocity_limit_scale * robot.planner_velocity_limits
        lower, upper, bound, valid = ramp_velocity_abs_bound(
            robot, cfg, q, start,
        )
        self.assertTrue(valid)
        self.assertTrue(np.all(bound >= np.abs(start)))
        self.assertTrue(np.all(bound >= np.abs(lower)))
        self.assertTrue(np.all(bound >= np.abs(upper)))
        for weight in np.linspace(0.0, 1.0, 11):
            for endpoint in (lower, upper):
                self.assertTrue(np.all(np.abs((1.0 - weight) * start
                                               + weight * endpoint) <= bound + 1e-12))
        empty_lower, empty_upper, fallback, empty_valid = ramp_velocity_abs_bound(
            robot, cfg, robot.planner_upper + 10.0, start,
        )
        self.assertFalse(empty_valid)
        self.assertTrue(np.any(empty_lower > empty_upper))
        self.assertTrue(np.all(fallback >= cfg.velocity_limit_scale
                               * robot.planner_velocity_limits))
        self.assertTrue(np.all(fallback >= np.abs(start)))
        model = robot.compile_dynamic_model()
        with self.assertRaises(ValueError):
            select_intervals_by_frozen_reach(
                {}, IntervalPartition.uniform(),
                FixedIntervalCBFEvaluator(robot, model),
                mujoco.MjData(model), cfg,
                np.zeros((6, 17)), velocity_abs_bound=np.full(17, np.nan),
            )

    def test_formal_holdout_artifacts_have_real_comparisons(self) -> None:
        base = Path(__file__).parent / "output" / "v6_2_b2"
        near = json.loads((base / "near_gate_audit" / "near_gate_audit.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(near["counts"]["checked_count"], 128)
        self.assertEqual(near["counts"]["PROXY_CLEARANCE_BELOW_GATE"], 64)
        self.assertEqual(near["counts"]["PROXY_CLEARANCE_AT_LEAST_GATE"], 64)
        heldout = json.loads((base / "heldout_geometry" / "heldout_geometry_report.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(heldout["counts"]["checked_count"], 1024)
        self.assertTrue(heldout["passed"])

    def test_refined_start_terms_reconstruct_saved_pcc_violation(self) -> None:
        path = (Path(__file__).parent / "output" / "v6_2_b2"
                / "refined_start" / "refined_start_report.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        for record in report["records"]:
            terms = record["worst_interval_start_terms"]
            if terms is None:
                continue
            self.assertEqual(terms["interval_id"],
                             record["feasibility"]["worst_start_source"])
            self.assertAlmostEqual(
                terms["point_signed_distance_m"] - terms["tube_radius_m"]
                - terms["coverage_term_m"] - terms["numerical_pad_m"]
                - terms["safe_distance_m"], terms["h_m"], places=10,
            )
            self.assertAlmostEqual(
                terms["shape_rate_m_s"] + terms["base_reaction_rate_m_s"]
                + terms["target_drift_m_s"] + terms["barrier_rate_m_s"],
                terms["start_residual_m_s"], places=10,
            )
            self.assertAlmostEqual(
                terms["start_residual_m_s"],
                record["feasibility"]["start_clearance_min_slack_m_s"],
                places=9,
            )
        for mode in ("baseline", "enabled"):
            bad = [record for record in report["records"]
                   if record["mode"] == mode
                   and record["feasibility"]["status"] == "START_CLEARANCE_VIOLATION"]
            self.assertEqual(
                sum(record["worst_interval_start_terms"]["h_m"] >= 0
                    for record in bad),
                report["modes"][mode]["start_violations_with_positive_interval_h"],
            )
            self.assertEqual(
                sum(record["worst_interval_start_terms"]["h_m"] < 0
                    for record in bad),
                report["modes"][mode]["start_violations_with_negative_interval_h"],
            )

    def test_stage2_manifest_preserves_failed_admission_gate(self) -> None:
        base = Path(__file__).parent / "output" / "v6_2_b2"
        manifest = json.loads((base / "stage2_summary" / "stage2_manifest.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(manifest["status"], "GATE_NOT_MET")
        self.assertIn("budget_frontier", manifest["sources"])
        self.assertIn("refined_start", manifest["sources"])
        for item in [*manifest["sources"].values(), manifest["generated_document"]]:
            path = Path(item["path"])
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(len(data), item["bytes"])
        warm = json.loads((base / "shadow_warm" / "shadow_report.json").read_text(
            encoding="utf-8"
        ))
        cold = json.loads((base / "shadow_a1" / "shadow_report.json").read_text(
            encoding="utf-8"
        ))
        for report in (cold, warm):
            for source, digest in report["source_sha256"].items():
                self.assertEqual(hashlib.sha256(Path(source).read_bytes()).hexdigest(), digest)
            self.assertEqual(report["task_spatial_refresh"],
                             "mj_forward_before_each_50Hz_geometry_query")
        self.assertEqual(warm["online_admission_gate"]["status"], "NOT_MET")
        self.assertFalse(warm["online_admission_gate"]["shadow_conditions"][
            "no_frozen_action_infeasibility"])
        for mode in ("baseline", "enabled"):
            item = warm["modes"][mode]
            self.assertEqual(len(item["frozen_feasibility_records"]), item["sample_count"])
            self.assertGreater(item["counts"].get("frozen_executable_false", 0), 0)
        self.assertGreater(warm["modes"]["enabled"]["counts"][
            "warm_all_task_unknown"], 0)
        frontier_path = base / "budget_frontier" / "budget_frontier.json"
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        self.assertEqual(frontier["budgets"], [31, 63, 127, 255, 511])
        self.assertEqual(len(frontier["entries"]), 270)
        self.assertEqual(frontier["input_shadow"]["sha256"], hashlib.sha256(
            (base / "shadow_warm" / "shadow_report.json").read_bytes()
        ).hexdigest())
        refined = json.loads((base / "refined_start" / "refined_start_report.json").read_text(
            encoding="utf-8"
        ))
        for source, digest in refined["source_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(source).read_bytes()).hexdigest(), digest)
        self.assertEqual(refined["input_frontier"]["sha256"], hashlib.sha256(
            frontier_path.read_bytes()
        ).hexdigest())
        self.assertEqual(len(refined["native_replay_checks"]), 10)
        self.assertEqual(len(refined["records"]), 270)
        self.assertFalse(refined["online_admission_claim"])
        for mode in ("baseline", "enabled"):
            self.assertGreater(refined["modes"][mode][
                "old_bad_remains_unexecutable"], 0)


if __name__ == "__main__":
    unittest.main()
