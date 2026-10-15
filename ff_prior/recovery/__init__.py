"""Explicit dead-owner recovery. Never removes a live or ambiguous run lock."""
from pathlib import Path
import ctypes
import os
import time
from ..storage import read_json,atomic_json


def process_is_alive(pid):
    if os.name=="nt":
        from ctypes import wintypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            error=ctypes.get_last_error()
            if error==87: return False
            raise RuntimeError(f"process state ambiguous, Win32 error {error}")
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise RuntimeError("cannot inspect lock owner")
            return code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid),0)
        return True
    except ProcessLookupError:
        return False


def recover_dead_lock(directory):
    directory=Path(directory).resolve()
    lock=directory/"run.lock"
    if not lock.exists(): return {"status":"NO_LOCK"}
    owner=read_json(lock)
    if process_is_alive(owner["pid"]):
        raise RuntimeError("run owner is alive; refusing lock recovery")
    report={"status":"DEAD_OWNER_LOCK_RECOVERED","owner":owner,"verified_at":time.time(),
            "state_files_preserved":True,"unknown_inflight_calls":"NOT_ASSUMED_ZERO",
            "next_step":"resume; interrupted proposal remains charged and is not regenerated"}
    atomic_json(directory/"lock_recovery.json",report)
    lock.unlink()
    return report
