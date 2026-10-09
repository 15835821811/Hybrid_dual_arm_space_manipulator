"""Verify the saved compensated servo evidence and its admission boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v6_lite.finalize_b2_discrete_private import run
from v6_lite.audit_b2_discrete_screened_parity import run as check_screened


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

    def test_pair_screen_keeps_executed_trace_and_misses_timing_gate(self) -> None:
        base = Path(__file__).parent / "output" / "v6_2_b2"
        with tempfile.TemporaryDirectory() as temporary:
            fresh = check_screened(
                base / "discrete_private_rollout_400" / "scene_00",
                base / "discrete_screened_private_rollout_400" / "scene_00",
                base / "discrete_private_full_qp_sphere",
                Path(temporary) / "result",
            )
        saved = json.loads((base / "discrete_screened_parity"
                            / "discrete_screened_parity_summary.json").read_text(
            encoding="utf-8"))
        self.assertEqual(fresh, saved)
        self.assertEqual(fresh["maximum_qpos_difference_m_or_rad"], 0.0)
        self.assertEqual(fresh["maximum_torque_difference_nm"], 0.0)
        self.assertEqual(fresh["paired_record_count"], 4000)
        self.assertEqual(fresh["paired_failure_count"], 0)
        self.assertGreater(fresh["paired_five_scene_preflight_plus_qp"]
                           ["sphere_screen"]["p95"], 20.0)
        self.assertGreater(fresh["screened_scene_00_preflight_plus_qp"]
                           ["p95_ms"], 20.0)
        self.assertFalse(fresh["stage3_admission"])


if __name__ == "__main__":
    unittest.main()
