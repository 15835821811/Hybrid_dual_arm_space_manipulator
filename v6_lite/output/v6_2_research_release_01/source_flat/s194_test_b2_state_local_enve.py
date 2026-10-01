"""Independent dense-axis check of the state-local PCC tube containment bound."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class StateLocalEnvelopeTests(unittest.TestCase):
    def test_actual_geom_capsules_have_full_axis_coverage_at_one_off_subspace_state(self) -> None:
        robot = default_v6_lite_robot_spec()
        verifier = WholeBodyCollisionVerifier(robot)
        model, data = verifier.model, verifier.data
        evaluator = FixedIntervalCBFEvaluator(robot, model)
        spec = default_continuum_model_spec(robot)
        audit = StateLocalPCCEnvelopeAudit(model, spec)
        data.qpos[evaluator.qpos_ids[0]] = 1e-6
        mujoco.mj_forward(model, data)
        projection = spec.project_actual_configuration(
            data.qpos[evaluator.qpos_ids[:60]])
        self.assertGreater(projection.residual_linf_rad, 1e-10)
        base = transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
        result = audit.evaluate(data, projection.planner_configuration, base)
        self.assertEqual(result.status, "COVERED_AT_THIS_STATE")
        self.assertEqual(result.fallback_geom_names, ("collision_0003",))
        self.assertEqual(len(result.capsules), len(audit.envelopes.capsules))
        self.assertGreater(result.min_margin_m, 0.0)

        pcc_points = audit.shape.batch_query(
            projection.planner_configuration, base, audit.arclengths,
            with_jacobians=False,
        )
        pcc_by_segment = np.asarray([point.position for point in pcc_points]).reshape(5, 17, 3)
        for capsule, row in zip(audit.envelopes.capsules, result.capsules):
            self.assertEqual(capsule.geom_name, row.geom_name)
            rotation = np.asarray(data.xmat[capsule.body_id]).reshape(3, 3)
            parameters = np.linspace(0.0, 1.0, 101)
            local = ((1 - parameters[:, None]) * capsule.local_start
                     + parameters[:, None] * capsule.local_end)
            axis = np.asarray(data.xpos[capsule.body_id]) + local @ rotation.T
            distances = np.linalg.norm(
                axis[:, None, :] - pcc_by_segment[row.segment_id][None, :, :],
                axis=2,
            )
            dense_max = float(np.max(np.min(distances, axis=1)))
            self.assertLessEqual(
                dense_max, row.sampled_axis_to_pcc_max_m
                + row.axis_sampling_allowance_m + 1e-12,
            )
            self.assertLessEqual(row.required_tube_radius_upper_m,
                                 row.declared_tube_radius_m)

        mismatched = audit.evaluate(data, np.ones(10), base)
        self.assertEqual(mismatched.status, "NOT_COVERED_AT_THIS_STATE")
        self.assertLess(mismatched.min_margin_m, 0.0)


if __name__ == "__main__":
    unittest.main()
