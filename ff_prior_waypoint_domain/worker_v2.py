"""Set numeric thread environment BEFORE importing NumPy or any storage module."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
import time,traceback
from ff_prior.storage import atomic_json,file_digest
from .experiment_v2 import main,OUT,ROOT


if __name__=="__main__":
    failed=ROOT/"artifacts/v7_1/waypoint_domain_001"
    old={str(p.relative_to(failed)):file_digest(p) for p in failed.rglob("*") if p.is_file()}
    try:
        main()
        atomic_json(OUT/"launcher_repair.json",dict(
            cause="v1 worker imported NumPy through storage before thread environment was configured",
            evidence="One-thread initialization reproduced original task/base/Z exactly in isolated diagnostic",
            previous_run=str(failed),previous_files=old,completed_baseline_reused=True,
            executions_across_attempts=4,replays_across_attempts=4,
            frozen_diffusion_proposals_reused=True,new_network_calls_in_final_run=0))
        atomic_json(OUT/"status.json",dict(status="RUNNING",pid=os.getpid(),heartbeat=time.time(),stage="visualization"))
        from . import visualize
        visualize.OUT=OUT
        visualize.main()
        if old!={str(p.relative_to(failed)):file_digest(p) for p in failed.rglob("*") if p.is_file()}:
            raise RuntimeError("Failed attempt changed")
        atomic_json(OUT/"status.json",dict(status="FINISHED",pid=os.getpid(),heartbeat=time.time(),
            stage="simulation_and_visualization_complete_review_pending"))
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        atomic_json(OUT/"status.json",dict(status="FAILED",pid=os.getpid(),heartbeat=time.time()))
        raise
