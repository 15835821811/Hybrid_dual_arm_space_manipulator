"""Late arrivals, replayed packets and identity changes never pass the gate."""

from dataclasses import replace
import unittest

import numpy as np
import mujoco

from v6_lite.runtime_command import CommandCertificate, DispatchGate, model_id


class RuntimeCommandTests(unittest.TestCase):
    def setUp(self):
        self.command = np.zeros(17)
        self.certificate = CommandCertificate.issue(
            command_id=4, source_state_id="state", source_model_hash="model",
            source_partition_id="partition", command=self.command,
            acquired=100., source_simulation_s=.4, solve_finished=100.008,
            validated=100.012)
        self.gate = DispatchGate()

    def check(self, certificate=None, **overrides):
        arguments = dict(now=100.015, model_hash="model", partition_hash="partition",
                         command=self.command, observed_state_id="state",
                         observed_simulation_s=.4)
        arguments.update(overrides)
        return self.gate.check(certificate or self.certificate, **arguments)

    def test_deadline_does_not_refresh_at_solve_or_validation(self):
        self.assertEqual(self.certificate.valid_until, 100.02)
        self.assertFalse(self.check(now=100.021)["accepted"])
        self.assertTrue(self.check(now=100.02)["accepted"])

    def test_changed_identity_start_and_payload_are_rejected(self):
        for argument, value in (("model_hash", "other"), ("partition_hash", "other"),
                                ("observed_state_id", "other"), ("command", np.ones(17)),
                                ("observed_simulation_s", .402), ("microstate_matches", False),
                                ("now", float("nan"))):
            with self.subTest(argument=argument):
                self.assertFalse(self.check(**{argument: value})["accepted"])

    def test_packet_order_and_ramp_exhaustion(self):
        self.assertTrue(self.check()["accepted"])
        self.assertFalse(self.check()["accepted"])
        self.assertFalse(self.check(substep=2, observed_simulation_s=.404)["accepted"])
        for step in range(1, 10):
            self.assertTrue(self.check(substep=step, observed_simulation_s=.4 + step * .002)["accepted"])
        self.assertFalse(self.check(substep=10, observed_simulation_s=.42)["accepted"])

    def test_stale_target_is_separate_from_robot_age(self):
        old_target = replace(self.certificate, target_acquisition_time=99.97)
        self.assertEqual(self.check(old_target)["reason"], "STALE_TARGET_WALL_CLOCK")

    def test_bad_certificate_times_and_expiry_cannot_be_renewed(self):
        for bad in (replace(self.certificate, valid_until=100.04),
                    replace(self.certificate, planned_execution_start=100.01),
                    replace(self.certificate, validation_finished_time=100.03)):
            self.assertFalse(self.check(bad)["accepted"])

    def test_offline_replay_is_explicit_and_still_checks_state(self):
        result = self.check(now=101., policy="offline_replay")
        self.assertTrue(result["accepted"])
        self.assertFalse(result["wall_clock_current"])
        self.gate = DispatchGate()
        self.assertFalse(self.check(observed_state_id="changed", now=101.,
                                    policy="offline_replay")["accepted"])

    def test_compiled_model_mutation_changes_identity(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        before = model_id(model, "source")
        model.body_mass[1] *= 1.01
        self.assertNotEqual(before, model_id(model, "source"))
