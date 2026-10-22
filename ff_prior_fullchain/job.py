"""Status/next/report only: this module never launches or resumes an experiment."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
TERMINAL = {"FINISHED", "FINISHED_WITH_BLOCKED_ITEMS", "FAILED", "BLOCKED"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def process_probe(pid, started_at):
    """Read Windows process identity, including creation time to detect PID reuse."""
    if os.name != "nt":
        return {"alive": None, "identity_matches": None, "reason": "WINDOWS_PROBE_REQUIRED"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        return {"alive": False if error == 87 else None, "identity_matches": None, "winerror": error}
    try:
        exit_code = wintypes.DWORD()
        creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return {"alive": None, "identity_matches": None, "reason": "EXIT_CODE_UNAVAILABLE"}
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(x) for x in (creation, exit_time, kernel_time, user_time)]):
            return {"alive": exit_code.value == 259, "identity_matches": None, "reason": "CREATION_TIME_UNAVAILABLE"}
        created = ((creation.dwHighDateTime << 32) | creation.dwLowDateTime) / 1e7 - 11644473600
        matches = started_at is not None and abs(created - float(started_at)) <= 120
        return {"alive": exit_code.value == 259, "identity_matches": matches, "created_at": created}
    finally:
        kernel.CloseHandle(handle)


def snapshot():
    config = read(PACKAGE / "control.json")
    path = ROOT / config["active_run"] / "status.json"
    answer = {"goal_id": config["goal_id"], "goal_status": config["goal_status"],
              "phase": config["current_phase"], "active_run": config["active_run"],
              "run_role": config.get("active_run_role", "current_worker"),
              "allow_dataset_expansion": config["allow_dataset_expansion"],
              "allow_final_TEST": config["allow_final_TEST"], "physics_relaxed": False,
              "status_source": str(path)}
    try:
        status = read(path)
    except (FileNotFoundError, ValueError) as error:
        return dict(answer, action="INSPECT_MISSING_OR_INVALID_STATUS", error=str(error))
    probe = process_probe(status.get("pid", 0), status.get("started_at"))
    heartbeat = status.get("heartbeat")
    age = time.time() - heartbeat if isinstance(heartbeat, (int, float)) else None
    answer.update(worker=status, process=probe, heartbeat_age_seconds=age)
    if status.get("status") == "RUNNING":
        healthy = (probe["alive"] is True and probe["identity_matches"] is True
                   and age is not None and -10 <= age <= config["heartbeat_stale_seconds"])
        answer["action"] = "WAIT_CURRENT_WORKER" if healthy else "DIAGNOSE_CURRENT_WORKER_NO_AUTOMATIC_RESTART"
    elif status.get("status") in TERMINAL:
        # A status string is not scientific success and is not report completion.
        receipt_path = ROOT / config["active_closeout_receipt"]
        if not receipt_path.exists():
            answer["action"] = "CLOSEOUT_CURRENT_FROZEN_RUN"
        else:
            receipt = read(receipt_path)
            closed = (receipt.get("run") == config["active_run"]
                      and receipt.get("closed") is True and bool(receipt.get("commit"))
                      and all((ROOT / receipt.get(k, "__missing__")).is_file()
                              for k in ("analysis", "independent_review", "completed_resume")))
            answer["action"] = "IMPLEMENT_AND_FREEZE_NEXT_PHASE" if closed else "INSPECT_INCOMPLETE_CLOSEOUT_RECEIPT"
            if closed:
                answer["next_phase"] = config["next_phase"]
    else:
        answer["action"] = "INSPECT_UNRECOGNIZED_STATUS"
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "next", "report"), default="status", nargs="?")
    args = parser.parse_args()
    if args.action == "report":
        print((PACKAGE / "docs/ROADMAP.md").read_text(encoding="utf-8"))
        config = read(PACKAGE / "control.json")
        report = ROOT / config["active_report"]
        print("\n当前阶段报告：" + (str(report) if report.exists() else "尚未完成；不可据目标卡或运行状态写成通过。"))
    else:
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
