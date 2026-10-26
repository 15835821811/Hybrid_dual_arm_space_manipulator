"""One evidence-triggered prefix-derivative refinement, still using validation only."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
import subprocess
import sys
import time
from .round2_data import BASE,read
from .round2_train_all import process_alive
from .round2_runs import run
from .pipeline import write_json


def main():
    pid=int((BASE/"screen.pid").read_text().strip())
    while True:
        p=BASE/"screen_stage.json"
        if p.exists() and read(p).get("status")=="complete": break
        if not process_alive(pid): raise RuntimeError("Validation screen stopped before completion")
        time.sleep(5)
    configurations=read(BASE/"screen_results.json")["configurations"]
    best=max(configurations,key=lambda c:(c["passed"],c["strict"]))
    if best["passed"]<29 or best["strict"]<27:
        write_json(BASE/"refine_stage.json",{"status":"training","reason":"T1/T2 validation below targets", "previous_best":best})
        if not (BASE/"train_t3/model.pt").exists():
            with (BASE/"train_t3.stdout.log").open("a",encoding="utf-8") as log:
                subprocess.run([sys.executable,"-m","tracking_diffusion.paper_model","--dataset",str(BASE/"data_500/dataset.npz"),
                    "--output",str(BASE/"train_t3"),"--steps","30000","--startup-fraction","0.2", "--prefix-velocity-loss","0.05","--device","cuda"],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        for scoring in ("legacy","prefix"):
            name=f"screen_t3_{scoring}"
            options=dict(best["options"],scoring=scoring,continuity=.01 if scoring=="prefix" else 0.)
            rows=run(BASE,name,"diffusion",BASE/"train_t3/model.pt",[f"old_task_{i:03d}" for i in range(80,90)],None,[73,137,211],4,options)
            configurations.append({"name":name,"checkpoint":str(BASE/"train_t3/model.pt"),"options":options,
                "finished":len(rows),"completed":sum(r["completed"] for r in rows),"passed":sum(r["passed"] for r in rows),"strict":sum(r["strict"] for r in rows)})
            write_json(BASE/"refine_results.json",{"configurations":configurations,"test_used":False})
    best=max(configurations,key=lambda c:(c["passed"],c["strict"]))
    write_json(BASE/"refine_stage.json",{"status":"complete","best":best,"test_used":False})


if __name__=="__main__": main()
