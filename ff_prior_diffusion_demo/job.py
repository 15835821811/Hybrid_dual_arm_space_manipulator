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

def assert_single_worker():
    from ff_prior_fullchain.job import process_probe
    for path in (ROOT/'artifacts/v7_1').glob('*/status.json'):
        s=read_json(path)
        if s.get('status')!='RUNNING': continue
        probe=process_probe(s.get('pid',0),s.get('started_at'))
        if probe['alive'] is True and probe['identity_matches'] is True:
            raise RuntimeError('ANOTHER_ACTIVE_WORKER: '+str(path))
        if probe['alive'] is None:
            raise RuntimeError('UNRESOLVED_WORKER_IDENTITY: '+str(path))

def launch(resume=False):
    # Preflight checks happen before creating a run directory or starting physics.
    from .inputs import identity
    ident=identity()
    assert read_json(PACKAGE/'evidence/software_tests.json')['source_hashes']==ident['source_hashes']
    if not resume:
        assert_single_worker()
        OUT.mkdir(parents=True, exist_ok=False)
    else:
        s = status(OUT)
        from ff_prior_fullchain.job import process_probe
        state=read_json(OUT/"status.json")
        probe=process_probe(state['pid'],state.get('started_at'))
        if probe['alive'] is True and probe['identity_matches'] is True:
            return dict(s, resume="NO_WORK_OWNER_ALIVE")
        if probe['alive'] is None:
            raise RuntimeError('UNKNOWN_PROCESS_IDENTITY')
        from .inputs import identity
        if read_json(OUT/"run.json")["identity"] != identity():
            raise ValueError("frozen identity changed")
        if s["status"] in ("FINISHED", "FINISHED_WITH_BLOCKED_ITEMS"):
            return dict(s, resume="NO_WORK_ALREADY_COMPLETE")
        raise RuntimeError("Unknown work is never automatically replayed; diagnose and freeze a separate identity")
    assert_single_worker()
    guard = OUT/"launch.guard"
    fd = os.open(guard, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        if (OUT/"run.lock").exists(): recover_dead_lock(OUT)
        argv = [sys.executable, "-u", "-m", "ff_prior_diffusion_demo.job", "worker"]
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
            result = run(OUT, monitor.update)
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
