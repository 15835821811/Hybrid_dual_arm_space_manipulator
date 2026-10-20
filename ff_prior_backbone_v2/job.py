"""Detached bounded worker; never replay an uncertain optimizer/physics unit."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor, status
from ff_prior.recovery import process_is_alive, recover_dead_lock
from .protocol import ROOT, DEFAULT_OUT


def completed_no_work(out):
    from .inputs import identity
    if read_json(out / "run.json")["identity"] != identity():
        raise ValueError("completed identity changed")
    if read_json(out / "summary.json")["status"] not in ("FINISHED", "FINISHED_WITH_BLOCKED_ITEMS"):
        raise ValueError("not complete")
    return dict(status(out), resume="NO_WORK_ALREADY_COMPLETE")


def worker(out, resume=False):
    previous = read_json(out / "status.json").get("elapsed_s", 0.) if resume else 0.
    with Monitor(out, previous) as monitor:
        try:
            from .experiment import run
            result = run(out, monitor.update, previous)
            monitor.finish(result)
            return 0 if result["status"] == "FINISHED" else 2
        except Exception as error:
            traceback.print_exc()
            failure = {"status": "FAILED", "reason": repr(error), "traceback": traceback.format_exc()}
            atomic_json(out / "failure.json", failure)
            if not (out / "REPORT.md").exists():
                (out / "REPORT.md").write_text("# 本轮尚未完成\n\n"+repr(error)+"\n旧证据保留；未完成不计通过，不自动重试。\n", encoding="utf-8")
            monitor.finish(failure)
            return 1


def launch(out, resume=False):
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    else:
        current = status(out)
        if (out / "status.json").exists() and process_is_alive(read_json(out / "status.json")["pid"]):
            return dict(current, resume="NO_WORK_OWNER_ALIVE")
        if current["status"] in ("FINISHED", "FINISHED_WITH_BLOCKED_ITEMS"):
            return completed_no_work(out)
        from .experiment import assert_resume_safe
        from .inputs import identity
        if read_json(out / "run.json")["identity"] != identity():
            raise ValueError("resume source changed")
        assert_resume_safe(out)
    guard = out / "launch.guard"
    fd = os.open(guard, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        if (out / "run.lock").exists():
            recover_dead_lock(out)
        argv = [sys.executable, "-u", "-m", "ff_prior_backbone_v2.job", "worker"]
        if resume:
            argv.append("--resume")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with (out / "worker.stdout.log").open("ab") as stdout, (out / "worker.stderr.log").open("ab") as stderr:
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout,
                                    stderr=stderr, env=env, **flags)
        atomic_json(out / "launcher.json", {"pid": proc.pid, "argv": argv, "started": time.time(), "detached": True})
        for _ in range(100):
            if (out / "status.json").exists() or proc.poll() is not None:
                break
            time.sleep(.1)
        return {"status": "LAUNCHED", "launcher_pid": proc.pid, "worker": status(out), "output": str(out)}
    finally:
        os.close(fd)
        guard.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status", "report", "resume"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = DEFAULT_OUT
    if args.action == "worker":
        return worker(out, args.resume)
    if args.action in ("start", "resume"):
        result = launch(out, args.action == "resume")
    elif args.action == "status":
        result = status(out)
    elif (out / "REPORT.md").exists():
        print((out / "REPORT.md").read_text(encoding="utf-8"))
        return 0
    else:
        result = {"status": "REPORT_NOT_READY", "progress": status(out)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
