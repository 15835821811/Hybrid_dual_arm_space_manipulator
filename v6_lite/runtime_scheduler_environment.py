"""Recorded, reversible per-thread Windows scheduling experiment settings."""
import ctypes
from ctypes import wintypes
import os


class ThreadScheduling:
    def __init__(self, role):
        self.record = {"role": role, "platform": os.name, "applied": False}
        self.old_affinity = None
        if os.name != "nt":
            return
        k = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentThread.restype = wintypes.HANDLE
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        k.SetThreadAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        k.SetThreadAffinityMask.restype = ctypes.c_size_t
        k.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
        k.GetThreadPriority.argtypes = (wintypes.HANDLE,)
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.GetPriorityClass.argtypes = (wintypes.HANDLE,)
        self.old_process_priority = k.GetPriorityClass(k.GetCurrentProcess())
        process_priority_set = k.SetPriorityClass(k.GetCurrentProcess(), 0x80)
        thread = k.GetCurrentThread()
        self.old_priority = k.GetThreadPriority(thread)
        available, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(available), ctypes.byref(system)):
            self.record["error"] = ctypes.get_last_error()
            return
        cpus = [i for i in range(64) if available.value & (1 << i)]
        index = min(1 if role == "executor" else 3, len(cpus) - 1)
        mask = 1 << cpus[index]
        self.old_affinity = k.SetThreadAffinityMask(thread, mask)
        priority = 2 if role == "executor" else 1
        success = k.SetThreadPriority(thread, priority)
        self.record.update({"applied": bool(self.old_affinity and success),
            "processor": cpus[index], "affinity_mask": mask,
            "thread_priority": priority, "process_realtime_priority_used": False,
            "process_priority_class": "HIGH_PRIORITY_CLASS",
            "process_priority_set": bool(process_priority_set),
            "error": ctypes.get_last_error() if not self.old_affinity or not success else None})

    def restore(self):
        if os.name == "nt" and self.old_affinity:
            thread = self.kernel.GetCurrentThread()
            self.kernel.SetThreadAffinityMask(thread, self.old_affinity)
            self.kernel.SetThreadPriority(thread, self.old_priority)
            self.kernel.SetPriorityClass(self.kernel.GetCurrentProcess(), self.old_process_priority)
