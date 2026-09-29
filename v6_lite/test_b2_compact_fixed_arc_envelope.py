"""Regression for the private envelope summary's rejection semantics."""

from __future__ import annotations

import unittest

import mujoco

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.b2_compact_fixed_arc_envelope import (
    CompactFixedArcStateLocalPCCEnvelopeAudit,
)
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import joint_addresses
from v6_lite.pcc_fixed_arc_state_envelope import FixedArcStateLocalPCCEnvelopeAudit
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, default_v6_lite_robot_spec, free_joint_slices,
)


class CompactFixedArcEnvelopeTest(unittest.TestCase):
    def test_full_and_summary_agree_when_margin_is_forced_negative(self) -> None:
        robot = default_v6_lite_robot_spec()
        run_config = V6LiteRunConfig()
        verifier = WholeBodyCollisionVerifier(
            robot, (), WholeBodyVerificationConfig(
                minimum_clearance=run_config.whole_body_minimum_clearance_m,
                query_distance_max=2.5,
                adaptive_subdivisions=run_config.verification_subdivisions,
                self_collision_ancestor_exclusion_depth=3,
                include_target_satellite_pairs=True,
            ),
        )
        model = verifier.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        shape_spec = default_continuum_model_spec(robot)
        qpos_ids, _ = joint_addresses(model, robot)
        projection = shape_spec.project_actual_configuration(
            data.qpos[qpos_ids[:60]])
        base_qpos_slice, _ = free_joint_slices(model, robot.base_joint_name)
        base = transform_from_free_qpos(data.qpos[base_qpos_slice])
        full = FixedArcStateLocalPCCEnvelopeAudit(model, shape_spec)
        compact = CompactFixedArcStateLocalPCCEnvelopeAudit(model, shape_spec)

        def check() -> None:
            full_result = full.evaluate(
                data, projection.planner_configuration, base)
            compact_result = compact.evaluate(
                data, projection.planner_configuration, base)
            self.assertEqual(compact_result.status, full_result.status)
            self.assertEqual(compact_result.min_margin_m,
                             full_result.min_margin_m)
            self.assertEqual(compact_result.checked_capsule_count,
                             len(full_result.capsules))
            self.assertGreater(compact_result.checked_capsule_count, 0)

        check()
        full._declared_m[0] = -1.0
        compact._declared_m[0] = -1.0
        check()
        self.assertEqual(compact.evaluate(
            data, projection.planner_configuration, base).status,
            "NOT_COVERED_AT_THIS_STATE")


if __name__ == "__main__":
    unittest.main()
