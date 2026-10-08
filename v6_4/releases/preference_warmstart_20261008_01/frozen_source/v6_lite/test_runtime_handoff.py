"""Virtual monotonic clock logic tests; these are not wall timing evidence."""
from dataclasses import replace
import unittest
import numpy as np
from v6_lite.runtime_handoff import CommandPacket, HandoffCertificate, HandoffBuffer, array_id, payload_id


def packet(i=0, start=1., source="snapshot", predecessor=None):
    states = np.linspace(i, i + 1, 11)[:, None] * np.ones((11, 8))
    c = HandoffCertificate(i, i - 1 if predecessor is None else predecessor,
        source, start - .019, start - .019, start - .018, start - .010,
        start - .004, start, start, start + .020, i * .020,
        "model", "config", "partition", array_id(states[0]),
        array_id(states[0]) if i else "startup",
        payload_id(states, np.ones((10, 67)), np.ones(17), np.zeros(17)))
    return CommandPacket.create(c, states, np.ones((10, 67)), np.ones(17), np.zeros(17))


class HandoffTests(unittest.TestCase):
    def test_normal_start_and_one_hundred_continuous_handoffs_with_jitter(self):
        gate = HandoffBuffer("model", "config")
        for i in range(100):
            p = packet(i, 1. + i * .020)
            self.assertTrue(gate.publish(p, p.certificate.execution_start - .003,
                source_state_id="snapshot", partition_id="partition")["accepted"])
            start = p.certificate.execution_start
            self.assertTrue(gate.handoff(start + .0001, p.integration_states[0], i * .020)["accepted"])
            for step in range(10):
                self.assertTrue(gate.servo(start + step * .002 + .0002,
                    p.integration_states[step], i * .020 + step * .002, step)["accepted"])
        self.assertEqual(gate.last_id, 99)
        self.assertIsNone(gate.pending)

    def test_late_duplicate_order_predecessor_model_config_partition_and_source_reject(self):
        cases = [("late", {}, 1.001), ("order", {"command_id": 2}, .999),
            ("predecessor", {"predecessor_command_id": 3}, .999),
            ("model", {"model_hash": "wrong"}, .999),
            ("config", {"controller_config_hash": "wrong"}, .999),
            ("partition", {"partition_id": "wrong"}, .999),
            ("source", {"source_state_id": "wrong"}, .999)]
        for name, change, now in cases:
            with self.subTest(name=name):
                gate = HandoffBuffer("model", "config")
                p = packet()
                p = replace(p, certificate=replace(p.certificate, **change))
                self.assertFalse(gate.publish(p, now, source_state_id="snapshot", partition_id="partition")["accepted"])
                self.assertIsNone(gate.pending)
        gate = HandoffBuffer("model", "config")
        p = packet()
        self.assertTrue(gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")["accepted"])
        self.assertFalse(gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")["accepted"])

    def test_target_state_and_future_prediction_cannot_be_substituted(self):
        gate = HandoffBuffer("model", "config")
        p = packet()
        gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")
        observed = p.integration_states[0].copy()
        observed[-1] += 1e-6
        self.assertEqual(gate.handoff(1., observed, 0.)["reason"], "HANDOFF_STATE_MISMATCH")
        self.assertIsNone(gate.active)

    def test_exhausted_packet_has_no_zero_or_endpoint_fallback(self):
        gate = HandoffBuffer("model", "config")
        p = packet()
        gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")
        gate.handoff(1., p.integration_states[0], 0.)
        for step in range(10):
            gate.servo(1. + .002 * step, p.integration_states[step], .002 * step, step)
        self.assertFalse(gate.servo(1.020, p.integration_states[-1], .020, 10)["accepted"])
        self.assertEqual(gate.handoff(1.020, p.integration_states[-1], .020)["reason"], "NO_VALID_CONTINUATION")

    def test_published_arrays_cannot_be_made_writeable(self):
        p = packet()
        with self.assertRaises(ValueError):
            p.torques.setflags(write=True)
        with self.assertRaises(ValueError):
            p.integration_states[0, 0] = 5

    def test_modified_torque_and_nonfinite_observed_state_are_rejected(self):
        p = packet()
        changed = replace(p, torques=np.zeros((10, 67)))
        gate = HandoffBuffer("model", "config")
        self.assertEqual(gate.publish(changed, .999, source_state_id="snapshot",
            partition_id="partition")["reason"], "COMMAND_PAYLOAD_MISMATCH")
        gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")
        observed = p.integration_states[0].copy()
        observed[0] = np.nan
        self.assertFalse(gate.handoff(1., observed, 0.)["accepted"])

    def test_every_nonfinite_certificate_time_is_rejected(self):
        for field in ("source_acquisition_time", "planning_release", "solve_started",
                "solve_finished", "validation_finished", "publish_deadline",
                "execution_start", "execution_end", "execution_start_simulation_s"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    gate = HandoffBuffer("model", "config")
                    p = packet()
                    p = replace(p, certificate=replace(p.certificate, **{field: value}))
                    result = gate.publish(p, .999, source_state_id="snapshot", partition_id="partition")
                    self.assertEqual(result["reason"], "NONFINITE_TIME")
                    self.assertIsNone(gate.pending)


if __name__ == "__main__":
    unittest.main()
