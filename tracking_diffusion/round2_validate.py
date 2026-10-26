"""Training-seed checks, isolated step-budget timing, and full validation."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
import shutil
import subprocess
import sys
import time
import numpy as np
from .round2_data import BASE,read
from .round2_train_all import process_alive
from .round2_runs import run
from .pipeline import write_json


def summarize(name,checkpoint,options,rows):
    return {"name":name,"checkpoint":str(checkpoint),"options":options,"finished":len(rows),
        "completed":sum(r["completed"] for r in rows),"passed":sum(r["passed"] for r in rows),"strict":sum(r["strict"] for r in rows)}


def completed_selection_inputs():
    expected={"seedcheck_43":30,"seedcheck_44":30,"ablation_gradient_off_projection_on":30,
        "speed_40":30,"speed_30":30,"timing_60":3,"timing_40":3,"timing_30":3}
    return all((BASE/name/"summary.json").exists() and read(BASE/name/"summary.json")["finished"]==count
        for name,count in expected.items())


def resume_selection():
    """Reuse fully recorded earlier runs after an orchestration-only failure."""
    from pathlib import Path
    best=read(BASE/"refine_stage.json")["best"]
    checkpoint=Path(best["checkpoint"])
    variants=[dict(best)]
    for steps in (40,30):
        s=read(BASE/f"speed_{steps}/summary.json")
        variants.append(summarize(f"speed_{steps}",checkpoint,s["protocol"]["options"],s["episodes"]))
    for v in variants:
        s=read(BASE/f"timing_{v['options']['steps']}/summary.json")
        durations=[]
        for row in s["episodes"]:
            with np.load(Path(row["result_path"]).parent/"trace.npz") as f:durations.extend(f["planning_seconds"])
        v["isolated_planning_p95_ms"]=float(np.percentile(durations,95)*1000)
        v["isolated_planning_100ms_misses"]=int(sum(t>.1 for t in durations))
    eligible=[v for v in variants if v["passed"]>=29 and v["strict"]>=27 and v["isolated_planning_p95_ms"]<=100.]
    selected=min(eligible,key=lambda v:v["isolated_planning_p95_ms"]) if eligible else max(variants,key=lambda v:(v["passed"],v["strict"],-v["isolated_planning_p95_ms"]))
    write_json(BASE/"selected_validation_config.json",{"selected":selected,"variants":variants,"test_used":False,
        "screen_targets_met":bool(eligible),"note":"Resumed from complete saved inputs after NumPy integer JSON failure; no completed physical run repeated or dropped"})
    write_json(BASE/"validation_stage.json",{"status":"full_validation","selected":selected,"resumed_selection":True})
    complete=BASE/"validation_full/summary.json"
    if complete.exists() and read(complete)["finished"]==250:
        rows=read(complete)["episodes"]
    else:
        rows=run(BASE,"validation_full","diffusion",checkpoint,None,"validation",[73,137,211,307,401],4,selected["options"])
    write_json(BASE/"validation_stage.json",{"status":"complete","selected":selected,
        "full_validation":summarize("validation_full",checkpoint,selected["options"],rows),"test_used":False})


def main():
    if completed_selection_inputs():
        return resume_selection()
    pid=int((BASE/"refine.pid").read_text().strip())
    while True:
        p=BASE/"refine_stage.json"
        if p.exists() and read(p).get("status")=="complete": break
        if not process_alive(pid): raise RuntimeError("Refinement chain stopped before completion")
        time.sleep(5)
    best=read(BASE/"refine_stage.json")["best"]
    checkpoint=BASE.parent.parent/best["checkpoint"]
    # Stored checkpoint strings are project-relative; keep them so provenance
    # and replay use the same path from the documented project root.
    from pathlib import Path
    checkpoint=Path(best["checkpoint"])
    recipe=read(checkpoint.parent/"training_config.json")
    names=[f"old_task_{i:03d}" for i in range(80,90)]
    seed_results=[]
    for seed in (43,44):
        write_json(BASE/"validation_stage.json",{"status":"training_seed_check","seed":seed,"best":best})
        output=BASE/f"train_selected_seed{seed}"
        if not (output/"model.pt").exists():
            with (BASE/f"train_seed{seed}.stdout.log").open("a",encoding="utf-8") as log:
                subprocess.run([sys.executable,"-m","tracking_diffusion.paper_model","--dataset",str(BASE/"data_500/dataset.npz"),
                    "--output",str(output),"--steps","30000","--seed",str(seed),"--startup-fraction",str(recipe.get("startup_fraction",0.)),
                    "--prefix-velocity-loss",str(recipe.get("prefix_velocity_loss_weight",0.)),"--device","cuda"],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=run(BASE,f"seedcheck_{seed}","diffusion",output/"model.pt",names,None,[73,137,211],4,best["options"])
        seed_results.append(summarize(f"seedcheck_{seed}",output/"model.pt",best["options"],rows))
        write_json(BASE/"training_seed_results.json",{"primary_training_seed":42,"supplementary":seed_results})
    ablation_options=dict(best["options"],guidance=False,projection=True)
    run(BASE,"ablation_gradient_off_projection_on","diffusion",checkpoint,names,None,[73,137,211],4,ablation_options)
    variants=[best]
    for steps in (40,30):
        options=dict(best["options"],steps=steps)
        name=f"speed_{steps}"
        write_json(BASE/"validation_stage.json",{"status":"step_budget_screen","steps":steps})
        rows=run(BASE,name,"diffusion",checkpoint,names,None,[73,137,211],4,options)
        variants.append(summarize(name,checkpoint,options,rows))
    # Throughput screens above do not establish deployment latency.
    gpu=shutil.which("nvidia-smi")
    if gpu:
        p=subprocess.run([gpu,"--query-compute-apps=pid,process_name,used_memory","--format=csv"],capture_output=True,text=True)
        (BASE/"isolated_timing_gpu_processes.txt").write_text(p.stdout+p.stderr,encoding="utf-8")
    for variant in variants:
        steps=variant["options"]["steps"]
        options=dict(variant["options"],isolated_timing=True)
        write_json(BASE/"validation_stage.json",{"status":"isolated_timing","steps":steps})
        rows=run(BASE,f"timing_{steps}","diffusion",checkpoint,[names[0]],None,[73,137,211],1,options)
        durations=[]
        for row in rows:
            with np.load(Path(row["result_path"]).parent/"trace.npz") as f: durations.extend(f["planning_seconds"])
        variant["isolated_planning_p95_ms"]=float(np.percentile(durations,95)*1000)
        variant["isolated_planning_100ms_misses"]=int(sum(t>.1 for t in durations))
    eligible=[v for v in variants if v["passed"]>=29 and v["strict"]>=27 and v["isolated_planning_p95_ms"]<=100.]
    selected=min(eligible,key=lambda v:v["isolated_planning_p95_ms"]) if eligible else max(variants,
        key=lambda v:(v["passed"],v["strict"],-v["isolated_planning_p95_ms"]))
    write_json(BASE/"selected_validation_config.json",{"selected":selected,"variants":variants,"test_used":False,
        "screen_targets_met":bool(eligible),"note":"Training seed 42 remains primary; no favorable seed replacement"})
    write_json(BASE/"validation_stage.json",{"status":"full_validation","selected":selected})
    rows=run(BASE,"validation_full","diffusion",checkpoint,None,"validation",[73,137,211,307,401],4,selected["options"])
    result=summarize("validation_full",checkpoint,selected["options"],rows)
    write_json(BASE/"validation_stage.json",{"status":"complete","selected":selected,"full_validation":result,"test_used":False})


if __name__=="__main__": main()
