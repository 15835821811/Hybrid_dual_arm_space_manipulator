"""Recorded, reversible per-thread Windows scheduling experiment settings."""
import ctypes
from ctypes import wintypes
import os
import struct


def processor_cores(kernel, allowed_mask):
    """Current-group core topology, queried only before task acquisition."""
    query = kernel.GetLogicalProcessorInformationEx
    query.argtypes = (wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
    length = wintypes.DWORD()
    query(0, None, ctypes.byref(length))  # RelationProcessorCore
    if not length.value:
        raise OSError(ctypes.get_last_error(), "processor topology query")
    buffer = ctypes.create_string_buffer(length.value)
    if not query(0, buffer, ctypes.byref(length)):
        raise OSError(ctypes.get_last_error(), "processor topology query")
    raw, offset, cores = buffer.raw, 0, []
    while offset < length.value:
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size < 32 or offset + size > length.value:
            raise ValueError("malformed Windows processor topology")
        if relationship == 0:
            efficiency = raw[offset + 9]
            count = struct.unpack_from("<H", raw, offset + 30)[0]
            for i in range(count):
                base = offset + 32 + i * 16
                mask, group = struct.unpack_from("<QH", raw, base)
                mask &= allowed_mask
                if mask and group == 0:
                    cores.append({"mask": mask, "efficiency_class": efficiency})
        offset += size
    return sorted(cores, key=lambda item: (item["mask"] & -item["mask"]).bit_length())


class ThreadScheduling:
    def __init__(self, role, policy="high"):
        if policy not in ("high", "realtime"):
            raise ValueError("unsupported per-process scheduler policy")
        self.record = {"role": role, "platform": os.name, "applied": False, "policy": policy}
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
        thread = k.GetCurrentThread()
        self.old_priority = k.GetThreadPriority(thread)
        # Lower the coordinator before raising its own process class. Native
        # executor and planner retain distinct physical cores; Windows and
        # interactive applications retain the other cores.
        priority = (-7 if role == "supervisor" else
                    (-1 if role == "planner" else 0)) if policy == "realtime" else 2
        k.SetThreadPriority(thread, priority)
        process_priority_set = k.SetPriorityClass(k.GetCurrentProcess(), 0x100 if policy == "realtime" else 0x80)
        available, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(available), ctypes.byref(system)):
            self.record["error"] = ctypes.get_last_error()
            return
        cpus = [i for i in range(64) if available.value & (1 << i)]
        cores = processor_cores(k, available.value)
        fastest = max((core["efficiency_class"] for core in cores), default=0)
        preferred = [core for core in cores if core["efficiency_class"] == fastest]
        # On the declared hybrid CPU use distinct physical cores, leaving the
        # first core outside both timed roles. The executor may migrate among
        # its disjoint set rather than being tied to one interrupted sibling.
        if role == "supervisor":
            lower = [core for core in cores if core["efficiency_class"] != fastest]
            chosen = lower if lower else preferred[:1]
            processors = [(core["mask"] & -core["mask"]).bit_length() - 1 for core in chosen]
        elif len(preferred) >= 5:
            chosen = preferred[4:] if role == "executor" else preferred[2:4]
            processors = [(core["mask"] & -core["mask"]).bit_length() - 1 for core in chosen]
        elif len(preferred) >= 3:
            chosen = preferred[2:] if role == "executor" else preferred[1:2]
            processors = [(core["mask"] & -core["mask"]).bit_length() - 1 for core in chosen]
        else:
            processors = [cpus[min(1 if role == "executor" else 3, len(cpus) - 1)]]
        mask = sum(1 << processor for processor in processors)
        self.old_affinity = k.SetThreadAffinityMask(thread, mask)
        success = k.SetThreadPriority(thread, priority)
        self.record.update({"applied": bool(self.old_affinity and success),
            "processors": processors, "affinity_mask": mask,
            "physical_core_topology": cores,
            "affinity_policy": "disjoint physical core sets of highest reported efficiency class; first two cores excluded when available",
            "thread_priority": priority, "process_realtime_priority_used": policy == "realtime",
            "process_priority_class": "REALTIME_PRIORITY_CLASS" if policy == "realtime" else "HIGH_PRIORITY_CLASS",
            "actual_process_priority_class": k.GetPriorityClass(k.GetCurrentProcess()),
            "process_priority_set": bool(process_priority_set),
            "error": ctypes.get_last_error() if not self.old_affinity or not success else None})

    def restore(self):
        if os.name == "nt" and hasattr(self, "kernel"):
            thread = self.kernel.GetCurrentThread()
            if self.old_affinity:
                self.kernel.SetThreadAffinityMask(thread, self.old_affinity)
            self.kernel.SetThreadPriority(thread, self.old_priority)
            self.kernel.SetPriorityClass(self.kernel.GetCurrentProcess(), self.old_process_priority)
