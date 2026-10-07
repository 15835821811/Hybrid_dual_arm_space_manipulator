"""The optional posture input changes only the existing soft objective."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M, HierarchicalQPConfig, HierarchicalVelocityQP, joint_addresses,
)
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class PostureReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = default_v6_lite_robot_spec()
        cls.verifier = WholeBodyCollisionVerifier(
            cls.spec, (), WholeBodyVerificationConfig(include_target_satellite_pairs=True))

    def capture(self, **kwargs):
        model = self.verifier.model
        data = mujoco.MjData(model)
        qp_ids, _ = joint_addresses(model, self.spec)
        data.qpos[qp_ids] = self.spec.encode_position(self.spec.planner_zero)
        mujoco.mj_forward(model, data)
        cfg = HierarchicalQPConfig(enable_pcc_cbf=False, enable_capsule_cbf=False)
        qp = HierarchicalVelocityQP(self.spec, model, self.verifier.pairs, cfg)
        r = np.asarray(data.xmat[qp.rigid_body_id]).reshape(3, 3).copy()
        c = np.asarray(data.xmat[qp.continuum_body_id]).reshape(3, 3).copy()
        targets = dict(
            rigid_target_position=np.asarray(data.xpos[qp.rigid_body_id]).copy(),
            rigid_target_velocity=np.zeros(3), rigid_target_rotation=r,
            rigid_target_angular_velocity=np.zeros(3),
            continuum_target_position=np.asarray(data.xpos[qp.continuum_body_id]).copy()
                + c @ CONTINUUM_EE_OFFSET_M,
            continuum_target_velocity=np.zeros(3), continuum_target_rotation=c,
            continuum_target_angular_velocity=np.zeros(3),
        )
        captured = {}
        original = qp._solve_qp_admm

        def solver(h, g, matrix, lower, upper, initial, dual):
            captured.update(h=h.copy(), g=g.copy(), matrix=matrix.copy(),
                            lower=lower.copy(), upper=upper.copy())
            return original(h, g, matrix, lower, upper, initial, dual)

        with patch.object(qp, "_solve_qp_admm", side_effect=solver):
            result = qp.solve(data, **targets, **kwargs)
        return captured, result

    def test_none_preserves_default_numerics(self):
        a, ar = self.capture()
        b, br = self.capture(posture_reference_q=None, posture_reference_dq=None)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(ar.solver_candidate, br.solver_candidate)

    def test_reference_changes_soft_objective_without_changing_hard_rows(self):
        a, _ = self.capture()
        q = self.spec.planner_zero + np.linspace(-.01, .01, 17)
        dq = np.linspace(-.015, .015, 17)
        b, _ = self.capture(posture_reference_q=q, posture_reference_dq=dq)
        for key in ("h", "matrix", "lower", "upper"):
            np.testing.assert_array_equal(a[key], b[key])
        cfg = HierarchicalQPConfig()
        np.testing.assert_allclose(
            b["g"] - a["g"],
            -cfg.posture_weight * (dq + cfg.posture_gain * (q - self.spec.planner_zero)),
            atol=1e-16, rtol=0)
        self.assertGreater(np.linalg.norm(b["g"] - a["g"]), 1e-6)

    def test_invalid_or_incomplete_posture_inputs_fail(self):
        for kwargs in (
            {"posture_reference_q": np.zeros(17)},
            {"posture_reference_q": np.zeros(16), "posture_reference_dq": np.zeros(17)},
            {"posture_reference_q": np.full(17, np.nan), "posture_reference_dq": np.zeros(17)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.capture(**kwargs)


if __name__ == "__main__":
    unittest.main()
