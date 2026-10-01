"""Workspace reuse must preserve branch arithmetic and all source state."""

import unittest

import mujoco
import numpy as np

from v6_lite.b2_interval_runtime import RuntimeWorkspace, preview_ramp
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = default_v6_lite_robot_spec()
        cls.model = cls.robot.compile_dynamic_model()
        cls.model.geom_contype[:] = 0
        cls.model.geom_conaffinity[:] = 0
        cls.evaluator = FixedIntervalCBFEvaluator(cls.robot, cls.model)
        cls.envelope = StateLocalPCCEnvelopeAudit(cls.model, cls.evaluator.shape_spec)

    def preview(self, source, workspace):
        return preview_ramp(
            self.model, self.robot, self.evaluator, self.envelope,
            source.qpos.copy(), source.qvel.copy(), float(source.time),
            self.robot.planner_zero.copy(), np.zeros(17), np.zeros(17),
            physics_period_s=.002, task_period_s=.02, prepared_step=True,
            workspace=workspace, source_data=source)[0]

    def test_zero_force_baseline_is_bitwise_equal(self):
        source = mujoco.MjData(self.model)
        workspace = RuntimeWorkspace(self.model, self.robot)
        old = self.preview(source, None)
        new = self.preview(source, workspace)
        np.testing.assert_array_equal(old["torques"], new["torques"])
        np.testing.assert_array_equal(old["qpos_states"], new["qpos_states"])
        self.assertEqual(old["coverage"], new["coverage"])

    def test_reused_workspace_copies_forces_and_does_not_leak_old_state(self):
        source = mujoco.MjData(self.model)
        reused = RuntimeWorkspace(self.model, self.robot)
        for force in (.15, -.12, 0.):
            source.qfrc_applied[0] = force
            source.xfrc_applied[1, 1] = force
            source.qacc_warmstart[:] = force
            source.ctrl[:] = force
            source.time += .02
            before = np.empty_like(reused.integration_state)
            mujoco.mj_getState(self.model, source, before, reused.state_spec)
            old = self.preview(source, RuntimeWorkspace(self.model, self.robot))
            new = self.preview(source, reused)
            np.testing.assert_array_equal(old["torques"], new["torques"])
            np.testing.assert_array_equal(old["qpos_states"], new["qpos_states"])
            after = np.empty_like(before)
            mujoco.mj_getState(self.model, source, after, reused.state_spec)
            np.testing.assert_array_equal(before, after)
