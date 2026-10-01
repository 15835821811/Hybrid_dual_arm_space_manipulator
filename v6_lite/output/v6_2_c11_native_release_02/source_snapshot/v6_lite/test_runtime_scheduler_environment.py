"""Own-process scheduling conditions and restoration on the declared platform."""
import ctypes
from ctypes import wintypes
import unittest

from v6_lite.runtime_scheduler_environment import ThreadScheduling


class SchedulerEnvironmentTests(unittest.TestCase):
    def test_declared_high_policy_restores_process_and_thread(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        kernel.GetPriorityClass.argtypes = (wintypes.HANDLE,)
        kernel.GetThreadPriority.argtypes = (wintypes.HANDLE,)
        process, thread = kernel.GetCurrentProcess(), kernel.GetCurrentThread()
        old_class = kernel.GetPriorityClass(process)
        old_priority = kernel.GetThreadPriority(thread)
        scheduling = ThreadScheduling("supervisor", "high")
        try:
            self.assertTrue(scheduling.record["applied"])
            self.assertEqual(scheduling.record["actual_process_priority_class"], 0x80)
            self.assertFalse(scheduling.record["process_realtime_priority_used"])
        finally:
            scheduling.restore()
        self.assertEqual(kernel.GetPriorityClass(process), old_class)
        self.assertEqual(kernel.GetThreadPriority(thread), old_priority)

    def test_realtime_request_records_actual_class_and_restores(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetPriorityClass.argtypes = (wintypes.HANDLE,)
        old_class = kernel.GetPriorityClass(kernel.GetCurrentProcess())
        scheduling = ThreadScheduling("supervisor", "realtime")
        try:
            record = scheduling.record
            self.assertEqual(record["process_realtime_priority_used"],
                record["actual_process_priority_class"] == 0x100)
            if record["existing_priority_privilege_enabled"]:
                self.assertEqual(record["actual_process_priority_class"], 0x100)
            else:
                self.assertNotEqual(record["actual_process_priority_class"], 0x100)
        finally:
            scheduling.restore()
        self.assertEqual(kernel.GetPriorityClass(kernel.GetCurrentProcess()), old_class)


if __name__ == "__main__":
    unittest.main()
