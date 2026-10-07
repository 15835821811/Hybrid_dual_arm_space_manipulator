"""Exercise the native actor with real MuJoCo physics and declared clocks."""
from dataclasses import replace
import multiprocessing as mp
import unittest

import mujoco
import numpy as np

from v6_lite.runtime_command import model_id
from v6_lite.runtime_handoff import (
    CommandPacket, HandoffCertificate, array_id, payload_id,
)
from v6_lite.runtime_native_loop import NativeBuffers, NativeHashContext, atomic_read
from v6_lite.runtime_native_executor import _rejected_native_attempt


class NativeLoopTests(unittest.TestCase):
    def setUp(self):
        bodies = "".join(
            f'<body pos="{i*.1} 0 0"><joint name="j{i}"/>'
            '<geom type="sphere" size=".01"/></body>' for i in range(67))
        actuators = "".join(f'<motor joint="j{i}"/>' for i in range(67))
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 0" timestep=".002"/>'
            f'<worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>')
        self.data = mujoco.MjData(self.model)
        self.crypto = NativeHashContext(self.model, "0"*64, model_id(self.model, "0"*64))
        self.addCleanup(self.crypto.close)
        self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.size = mujoco.mj_stateSize(self.model, self.state_spec)
        self.buffers = NativeBuffers(mp.get_context("spawn"), self.size, 20)
        self.times = np.column_stack([100 + np.arange(20)*.002 + .00001]*2)
        preview = mujoco.MjData(self.model)
        self.packets = []
        for command in range(2):
            states = [self.state(preview)]
            torques = np.full((10, 67), .00001 * (command+1))
            for torque in torques:
                preview.ctrl[:] = torque
                mujoco.mj_step(self.model, preview)
                states.append(self.state(preview))
            states = np.asarray(states)
            tail = np.zeros(17)
            certificate = HandoffCertificate(
                command, command-1, array_id(states[0]), 99.9, 99.9, 99.9,
                99.91, 99.92, 100+command*.020, 100+command*.020,
                100+(command+1)*.020, command*.020,
                model_id(self.model, "0"*64), "config", "partition",
                array_id(states[0]), "pred", payload_id(states, torques, tail, tail))
            self.packets.append(CommandPacket.create(certificate, states, torques, tail, tail))

    def state(self, data):
        result = np.empty(self.size)
        mujoco.mj_getState(self.model, data, result, self.state_spec)
        return result

    def run_actor(self, packets=None, count=20):
        for packet in self.packets if packets is None else packets:
            self.buffers.publish(packet, 99.99)
        count = self.buffers.execute(
            self.model, self.data, self.crypto, 100., count, self.times)
        self.assertEqual(count, atomic_read(self.buffers.view("signals"), 1))
        self.assertAlmostEqual(self.data.time, count*.002)
        return count, atomic_read(self.buffers.view("signals"), 2)

    def test_twenty_native_steps_match_private_preview(self):
        self.assertEqual(self.run_actor(), (20, 2))
        np.testing.assert_array_equal(self.buffers.view("states")[20], self.packets[1].integration_states[-1])
        self.assertTrue(np.all(self.buffers.view("timings")[:, 2] < self.times[:, 0]+.002))

    def test_unarmed_actor_has_no_fabricated_attempt_or_timestamps(self):
        attempt = _rejected_native_attempt(self.buffers, 100., 20, 101.)
        self.assertFalse(attempt["attempted"])
        self.assertIsNone(attempt["physics_step"])
        self.assertIsNone(attempt["actual_start"])
        self.assertFalse(attempt["torque_consumed"])
        self.assertEqual(attempt["operation_phase"], "not_attempted")

    def test_partial_last_segment_has_a_source_slot_and_realizes_eleven_steps(self):
        self.buffers = NativeBuffers(mp.get_context("spawn"), self.size, 11)
        self.assertEqual(len(self.buffers.view("source_times")), 2)
        self.assertEqual(self.run_actor(count=11), (11, 2))
        np.testing.assert_array_equal(self.buffers.view("states")[11],
                                      self.packets[1].integration_states[1])
        self.assertEqual(self.buffers.view("source_times")[1], self.times[10, 1])

    def test_nonfinite_native_metadata_never_advances_physics(self):
        for column in range(4):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(column=column, value=value):
                    self.buffers = NativeBuffers(mp.get_context("spawn"), self.size, 20)
                    self.buffers.publish(self.packets[0], 99.99)
                    self.buffers.view("slot_meta")[0, column] = value
                    self.assertEqual(self.run_actor([]), (0, -4))

    def test_nonfinite_certificate_times_are_rejected_before_staging(self):
        for field in ("source_acquisition_time", "planning_release", "solve_started",
                "solve_finished", "validation_finished", "publish_deadline",
                "execution_start", "execution_end", "execution_start_simulation_s"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    changed = replace(self.packets[0], certificate=replace(
                        self.packets[0].certificate, **{field: value}))
                    with self.assertRaisesRegex(ValueError, "NONFINITE_NATIVE_CLOCK_CONTRACT"):
                        self.buffers.stage(changed)
                    self.assertEqual(atomic_read(self.buffers.view("ready"), 0), -1)

    def test_exhausted_packet_does_not_execute_eleventh_step(self):
        self.assertEqual(self.run_actor([self.packets[0]]), (10, -1))

    def test_late_publication_is_rejected_before_release(self):
        with self.assertRaisesRegex(ValueError, "LATE_OR_INVALID_PUBLICATION"):
            self.buffers.publish(self.packets[0], 100.0001)
        self.assertEqual(atomic_read(self.buffers.view("ready"), 0), -1)
        self.assertEqual(self.run_actor([]), (0, -1))

    def test_owned_slot_cannot_be_overwritten(self):
        self.buffers.publish(self.packets[0], 99.99)
        with self.assertRaisesRegex(ValueError, "STILL_OWNED"):
            self.buffers.stage(self.packets[0])
        self.assertEqual(self.run_actor([self.packets[1]]), (20, 2))

    def test_wrong_predecessor_is_rejected(self):
        wrong = replace(self.packets[0], certificate=replace(
            self.packets[0].certificate, predecessor_command_id=77))
        self.assertEqual(self.run_actor([wrong]), (0, -3))

    def test_payload_tampering_is_rejected(self):
        self.buffers.publish(self.packets[0], 99.99)
        self.buffers.view("slot_torques")[0, 0, 0] += .001
        self.assertEqual(self.run_actor([]), (0, -8))

    def test_live_model_mutation_is_rejected(self):
        self.model.opt.tolerance *= 2
        self.assertEqual(self.run_actor(), (0, -7))

    def test_wrong_start_state_is_rejected(self):
        self.data.qvel[1] = .1
        self.assertEqual(self.run_actor(), (0, -6))

    def test_servo_late_wakeup_stops_without_physics(self):
        self.times[0] = 100.0021
        self.assertEqual(self.run_actor(), (0, -5))

    def test_late_consumption_stops_without_physics(self):
        self.times[0, 1] = 100.0021
        self.assertEqual(self.run_actor(), (0, -12))
        attempt = _rejected_native_attempt(self.buffers, 100., 20, 101.)
        self.assertEqual(attempt["physics_step"], 0)
        self.assertFalse(attempt["torque_consumed"])
        self.assertEqual(attempt["consumption_check_time"], 100.0021)

    def test_poststep_mismatch_records_the_step_that_really_ran(self):
        packet = self.packets[0]
        states = packet.integration_states.copy()
        states[1, 1] += .001
        altered = CommandPacket.create(replace(packet.certificate,
            payload_sha256=payload_id(states, packet.torques,
                packet.endpoint_velocity, packet.reference_end)),
            states, packet.torques, packet.endpoint_velocity, packet.reference_end)
        self.assertEqual(self.run_actor([altered]), (1, -9))
        np.testing.assert_array_equal(self.buffers.view("states")[1], self.state(self.data))
        attempt = _rejected_native_attempt(self.buffers, 100., 20, 101.)
        self.assertEqual(attempt["physics_step"], 0)
        self.assertEqual(attempt["operation_phase"], "poststep_realization_check")
        self.assertTrue(attempt["torque_consumed"])
        self.assertTrue(attempt["physical_step_executed"])
        self.assertEqual(attempt["actual_start"], self.times[0, 0])


if __name__ == "__main__":
    unittest.main()
