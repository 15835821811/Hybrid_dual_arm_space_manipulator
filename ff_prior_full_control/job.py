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
from .protocol import ROOT, OUT, PACKAGE

def launch(resume=False):
    if not resume:
        OUT.mkdir(parents=True, exist_ok=False)
    else:
        s = status(OUT)
        if (OUT/"status.json").exists() and process_is_alive(read_json(OUT/"status.json")["pid"]):
            return dict(s, resume="NO_WORK_OWNER_ALIVE")
        from .inputs import identity
        if read_json(OUT/"run.json")["identity"] != identity():
            raise ValueError("frozen identity changed")
        if s["status"] in ("FINISHED", "FINISHED_WITH_BLOCKED_ITEMS"):
            return dict(s, resume="NO_WORK_ALREADY_COMPLETE")
        from .experiment import assert_resume_safe
        assert_resume_safe(OUT)
    guard = OUT/"launch.guard"
    fd = os.open(guard, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        if (OUT/"run.lock").exists(): recover_dead_lock(OUT)
        argv = [sys.executable, "-u", "-m", "ff_prior_full_control.job", "worker"]
        if resume: argv.append("--resume")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with (OUT/"worker.stdout.log").open("ab") as stdout, (OUT/"worker.stderr.log").open("ab") as stderr:
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=env, creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        atomic_json(OUT/"launcher.json", {"pid": proc.pid, "argv": argv, "started": time.time(), "detached": True})
        for _ in range(100):
            if (OUT/"status.json").exists() or proc.poll() is not None: break
            time.sleep(.1)
        return {"status": "LAUNCHED", "launcher_pid": proc.pid, "worker": status(OUT)}
    finally:
        os.close(fd); guard.unlink()

def worker(resume=False):
    previous = read_json(OUT/"status.json").get("elapsed_s", 0.) if resume else 0.
    with Monitor(OUT, previous) as monitor:
        try:
            from .experiment import run
            result = run(OUT, monitor.update, previous)
            monitor.finish(result)
            return 0 if result["status"] == "FINISHED" else 2
        except Exception as e:
            traceback.print_exc()
            result = {"status": "FAILED", "reason": repr(e), "traceback": traceback.format_exc()}
            atomic_json(OUT/"failure.json", result)
            (OUT/"REPORT.md").write_text("# 本轮未完成\n\n"+repr(e)+"\n\n已有结果保留，未完成阶段不通过，不重放未知工作。\n", encoding="utf-8")
            monitor.finish(result)
            return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "worker", "resume", "status", "report"])
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    os.chdir(ROOT)
    if a.action == "worker": return worker(a.resume)
    if a.action in ("start", "resume"): answer = launch(a.action == "resume")
    elif a.action == "status": answer = status(OUT)
    else:
        path = PACKAGE/"docs/ANALYSIS.md"
        if not path.exists(): path = OUT/"REPORT.md"
        if path.exists():
            print(path.read_text(encoding="utf-8")); return 0
        answer = {"status": "REPORT_NOT_READY", "progress": status(OUT)}
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__": sys.exit(main())
