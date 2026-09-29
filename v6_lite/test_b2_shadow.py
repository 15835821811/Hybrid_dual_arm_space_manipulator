"""Read-only shadow partition and frozen-reach screening checks."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import mujoco
import numpy as np

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
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
        selected, reach = select_intervals_by_frozen_reach(
            result, partition, evaluator, data, HierarchicalQPConfig(), generalized_map,
        )
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

    def test_stage2_manifest_preserves_failed_admission_gate(self) -> None:
        base = Path(__file__).parent / "output" / "v6_2_b2"
        manifest = json.loads((base / "stage2_summary" / "stage2_manifest.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(manifest["status"], "GATE_NOT_MET")
        for item in [*manifest["sources"].values(), manifest["generated_document"]]:
            path = Path(item["path"])
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(len(data), item["bytes"])
        warm = json.loads((base / "shadow_warm" / "shadow_report.json").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(warm["online_admission_gate"]["status"], "NOT_MET")
        self.assertGreater(warm["modes"]["enabled"]["counts"][
            "warm_all_task_unknown"], 0)


if __name__ == "__main__":
    unittest.main()
