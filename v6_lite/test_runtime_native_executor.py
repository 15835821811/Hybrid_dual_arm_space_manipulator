"""Faulted postrun transfers remain bounded and cannot skip resource cleanup."""
import threading
import unittest
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from v6_lite.runtime_native_executor import (
    _cleanup_native_runtime, _receive_planner_records, run_native_scenario,
)
from v6_lite.runtime_native_loop import atomic_write
from v6_lite.run_v6_lite import UncertifiedExecutionError, V6LiteRunConfig


class ScriptedConnection:
    def __init__(self, messages, stall=False):
        self.messages = list(messages)
        self.stall = stall
        self.closed = threading.Event()

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        if self.stall:
            self.closed.wait()
        raise EOFError("sender ended midway through transfer")

    def close(self):
        self.closed.set()

    def send(self, value):
        pass


class NativeRecordCleanupTests(unittest.TestCase):
    def test_eof_keeps_already_received_records(self):
        record = {"payload": "validated first packet"}
        connection = ScriptedConnection([{"record_count": 2}, record])
        records = []
        header, problem = _receive_planner_records(connection, records, 2, timeout_s=.05)
        self.assertEqual(header["record_count"], 2)
        self.assertEqual(records, [record])
        self.assertIn("EOFError", problem)

    def test_stalled_complete_receive_times_out_and_keeps_records(self):
        record = {"payload": "validated first packet"}
        connection = ScriptedConnection([{"record_count": 2}, record], stall=True)
        records = []
        header, problem = _receive_planner_records(connection, records, 2, timeout_s=.02)
        self.assertEqual(header["record_count"], 2)
        self.assertEqual(records, [record])
        self.assertEqual(problem, "PLANNER_RECORD_TRANSFER_TIMEOUT")
        self.assertTrue(connection.closed.is_set())

    def test_invalid_transfer_count_cannot_allocate_an_unbounded_queue(self):
        for value in (3, -1, True, "2"):
            with self.subTest(value=value):
                records = []
                connection = ScriptedConnection([{"record_count": value}])
                _, problem = _receive_planner_records(connection, records, 2, timeout_s=.05)
                self.assertIn("invalid planner transfer record count", problem)
                self.assertEqual(records, [])

    def test_eof_transfer_still_restores_all_resources(self):
        records = []
        connection = ScriptedConnection([{"record_count": 1}])
        _, problem = _receive_planner_records(connection, records, 1, timeout_s=.05)
        self.assertIn("EOFError", problem)
        worker, scheduling, crypto = Mock(), Mock(), Mock()
        worker.is_alive.return_value = False
        with patch("v6_lite.runtime_native_executor.gc.enable") as enable_gc:
            self.assertEqual(_cleanup_native_runtime(worker, connection, scheduling,
                                                    crypto, True), [])
        self.assertTrue(connection.closed.is_set())
        scheduling.restore.assert_called_once()
        crypto.close.assert_called_once()
        enable_gc.assert_called_once()

    def test_one_cleanup_exception_cannot_skip_other_cleanup(self):
        worker, connection, scheduling, crypto = Mock(), Mock(), Mock(), Mock()
        worker.join.side_effect = RuntimeError("join failed")
        worker.is_alive.return_value = False
        connection.close.side_effect = OSError("close failed")
        with patch("v6_lite.runtime_native_executor.gc.enable") as enable_gc:
            problems = _cleanup_native_runtime(worker, connection, scheduling, crypto, True)
        self.assertEqual(len(problems), 2)
        scheduling.restore.assert_called_once()
        crypto.close.assert_called_once()
        enable_gc.assert_called_once()

    def test_live_actor_hash_is_not_freed_during_failed_cancellation(self):
        worker, connection, scheduling, crypto = Mock(), Mock(), Mock(), Mock()
        worker.is_alive.return_value = False
        self.assertEqual(_cleanup_native_runtime(worker, connection, scheduling,
                                                crypto, False, actor_stopped=False), [])
        scheduling.restore.assert_called_once()
        crypto.close.assert_not_called()

    def test_eof_during_scenario_cleanup_still_saves_failure_and_native_trace(self):
        # Only simulate IPC/actor failure: no wall loop or process priority runs.
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        data = mujoco.MjData(model)
        context = SimpleNamespace(model=model, data=data, physics_steps=20,
            initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy(), base_body_id=0,
            source_contract_hash="0"*64, source_model_hash="0"*64,
            controller_config_hash="config", controller_source_hashes={}, numerical_thread_pools=[])
        scheduling_record = {"applied": True, "process_priority_set": True,
                             "actual_process_priority_class": 0x80}
        worker_ready = {"ready": True, "private_initialization_solves": 1,
                        "thread_scheduling": scheduling_record}
        connection = ScriptedConnection([worker_ready, {"record_count": 1}])
        worker = Mock()
        worker.is_alive.return_value = False
        real_context = mp.get_context("spawn")
        fake_context = SimpleNamespace(RawArray=real_context.RawArray,
            Pipe=lambda: (connection, Mock()), Process=lambda **kwargs: worker)
        crypto = Mock(counter=np.zeros(1, dtype=np.int64), frequency=1.)
        scheduling = Mock(record=scheduling_record)
        error_slot = Mock()
        error_slot.take.return_value = None
        def fake_execute(buffers, _model, _data, _crypto, epoch, count, *_virtual):
            if count:
                buffers.view("timings")[0, :2] = epoch
                atomic_write(buffers.view("signals"), 0, 4)
                atomic_write(buffers.view("signals"), 2, 5)
                atomic_write(buffers.view("signals"), -1, 2)
            return 0
        with tempfile.TemporaryDirectory(prefix="c11_cleanup_test_") as temporary:
            trace_dir = Path(temporary)/"traces"
            trace_dir.mkdir()
            with patch("v6_lite.runtime_native_executor._context", return_value=context), \
                    patch("v6_lite.runtime_native_executor.mp.get_context", return_value=fake_context), \
                    patch("v6_lite.runtime_native_executor.NativeHashContext", return_value=crypto), \
                    patch("v6_lite.runtime_native_executor.NativeBuffers.execute", fake_execute), \
                    patch("v6_lite.runtime_native_executor.SharedResultSlot", return_value=error_slot), \
                    patch("v6_lite.runtime_native_executor.ThreadScheduling", return_value=scheduling), \
                    patch("v6_lite.runtime_native_executor.now", return_value=100.), \
                    patch("v6_lite.runtime_native_executor.time.perf_counter", return_value=100.), \
                    patch("v6_lite.runtime_native_executor._robot_momentum", return_value=np.zeros(6)):
                with self.assertRaisesRegex(UncertifiedExecutionError, "NO_VALID_CONTINUATION"):
                    run_native_scenario(SimpleNamespace(planner_zero=np.zeros(17)),
                        V6LiteRunConfig(), None, SimpleNamespace(scenario_id="fixture"), trace_dir)
            failure_path = Path(temporary)/"failures/fixture_interval_failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertIn("EOFError", failure["record_transfer_error"])
            self.assertEqual(failure["physics_steps_executed"], 0)
            self.assertEqual(failure["physics_step"], 0)
            self.assertFalse(failure["rejected_step_executed"])
            self.assertTrue((Path(temporary)/"native/fixture.npz").exists())
            crypto.close.assert_called_once()
            self.assertEqual(scheduling.restore.call_count, 2)


if __name__ == "__main__":
    unittest.main()
