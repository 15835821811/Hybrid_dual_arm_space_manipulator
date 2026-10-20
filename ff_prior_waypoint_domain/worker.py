import os,time,traceback
from ff_prior.storage import atomic_json,read_json
from .experiment import main,OUT


if __name__=="__main__":
    try:
        main()
        atomic_json(OUT/"status.json",dict(status="RUNNING",pid=os.getpid(),heartbeat=time.time(),stage="visualization"))
        from .visualize import main as visualize
        visualize()
        atomic_json(OUT/"status.json",dict(status="FINISHED",pid=os.getpid(),heartbeat=time.time(),
            stage="simulation_and_visualization_complete_review_pending"))
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        atomic_json(OUT/"status.json",dict(status="FAILED",pid=os.getpid(),heartbeat=time.time()))
        raise
