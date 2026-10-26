"""Pre-registered T1/T2 closed-loop validation after full-data training."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
import time
from .round2_data import BASE,read
from .round2_train_all import process_alive
from .round2_runs import run
from .pipeline import write_json


def main():
    pid=int((BASE/"train_all.pid").read_text().strip())
    while True:
        p=BASE/"training_stage.json"
        if p.exists() and read(p).get("status")=="complete": break
        if not process_alive(pid): raise RuntimeError("Training chain stopped; inspect logs before continuing")
        time.sleep(5)
    names=[f"old_task_{i:03d}" for i in range(80,90)]
    configurations=[]
    for model in ("t1","t2"):
        for scoring in ("legacy","prefix"):
            name=f"screen_{model}_{scoring}"
            options={"device":"cuda","steps":60,"candidates":5,"guidance":True,"projection":True,
                "scoring":scoring,"continuity":.01 if scoring=="prefix" else 0.,"diagnostic":False,
                "duration":None,"cuda_graph":True,"isolated_timing":False}
            write_json(BASE/"screen_stage.json",{"status":"running","name":name})
            rows=run(BASE,name,"diffusion",BASE/f"train_{model}/model.pt",names,None,[73,137,211],4,options)
            configs={"name":name,"checkpoint":str(BASE/f"train_{model}/model.pt"),"options":options,
                "finished":len(rows),"completed":sum(r["completed"] for r in rows),
                "passed":sum(r["passed"] for r in rows),"strict":sum(r["strict"] for r in rows)}
            configurations.append(configs)
            write_json(BASE/"screen_results.json",{"configurations":configurations,"test_used":False})
    best=max(configurations,key=lambda c:(c["passed"],c["strict"]))
    write_json(BASE/"screen_stage.json",{"status":"complete","best":best,"test_used":False,
        "interpretation":"Validation screening only; not final performance or isolated timing"})


if __name__=="__main__": main()
