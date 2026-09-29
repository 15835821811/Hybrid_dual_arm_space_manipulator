"""Verify the saved compensated servo evidence and its admission boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v6_lite.finalize_b2_discrete_private import run


class B2DiscretePrivateEvidenceTests(unittest.TestCase):
    def test_five_scene_replay_is_bound_to_sources_and_still_fails_timing_gate(self) -> None:
        base = Path(__file__).parent / "output" / "v6_2_b2"
        with tempfile.TemporaryDirectory() as temporary:
            fresh = run(
                base / "discrete_private_rollout_400",
                base / "discrete_private_recompute_400",
                base / "servo_subspace_origin",
                base / "discrete_servo_first_tick",
                Path(temporary) / "result",
            )
        saved = json.loads((base / "discrete_private_400_summary"
                            / "discrete_private_five_scene_summary.json").read_text(
            encoding="utf-8"))
        for actual, expected in zip(fresh["scenes"], saved["scenes"]):
            self.assertTrue(Path(actual["folder"]).as_posix().endswith(
                expected["folder"]))
            actual["folder"] = expected["folder"]
        self.assertEqual(fresh, saved)
        self.assertEqual(fresh["complete_400_tick_scene_count"], 5)
        self.assertEqual(fresh["checked_500hz_states"], 20005)
        self.assertLessEqual(
            fresh["maximum_500hz_subspace_residual_linf_rad"], 1e-10)
        self.assertEqual(fresh["independent_interval_row_count"], 5257)
        self.assertEqual(fresh["independent_recompute_failure_count"], 0)
        self.assertEqual(fresh["torque_limit_violation_count"], 0)
        self.assertTrue(all(scene["executed_ticks"] == 400
                            for scene in fresh["scenes"]))
        self.assertTrue(all(scene["preflight_plus_qp_timing"]["p95_ms"] > 20.0
                            for scene in fresh["scenes"]))
        self.assertFalse(fresh["full_cycle_20ms_acceptance"])
        self.assertFalse(fresh["stage3_admission"])

        old = json.loads((base / "private_recompute_400"
                          / "private_recompute_summary.json").read_text(
            encoding="utf-8"))
        self.assertFalse(old[
            "independent_strict_online_domain_all_executed_ticks"])
        self.assertFalse(old["all_500hz_states_on_declared_shape_subspace"])


if __name__ == "__main__":
    unittest.main()
