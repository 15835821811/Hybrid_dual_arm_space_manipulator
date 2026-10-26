"""Frozen second-round tasks, bounded teacher production and immutable datasets."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .environment import Task
from .pipeline import run_episode, write_json
from .task_domain import DOMAIN, profile_hash, sample_task

BASE = Path("tracking_output/paper500_v2")
OLD = Path("tracking_output/paper100_v1")


def read(path):
    # Legacy in-flight producers may still use their original non-atomic writer.
    for attempt in range(10):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt==9: raise
            time.sleep(.02)


def freeze(folder=BASE):
    path = folder/"manifest.json"
    if path.exists():
        value = read(path)
        if value["profile_sha256"] != profile_hash():
            raise ValueError("Frozen profile changed")
        return value
    old = read(OLD/"manifest.json")
    entries = []
    for e in old["tasks"]:
        item = dict(e, id="old_"+e["id"], origin="paper100_v1",
                    teacher_path=str(OLD/"teachers"/e["id"]))
        if e["split"] == "test":
            item["split"] = "diagnostic"
        entries.append(item)
    for split, start, count in (("train",200000000,521),("validation",300000000,40),("test",400000000,20)):
        for i in range(count):
            task = Task(start+104729*i,27.,2,DOMAIN,False)
            entries.append({"id":f"{split}_{i:03d}","task":asdict(task),"split":split,
                "role":"reserve" if split=="train" and i>=421 else "primary",
                "origin":"paper500_v2","parameters":sample_task(task.seed)})
    seeds = [e["task"]["seed"] for e in entries]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Task seed overlap")
    value = {"domain":DOMAIN,"profile_sha256":profile_hash(),"tasks":entries,
        "sampling_seeds":[73,137,211,307,401],"training_seeds":[42,43,44],
        "training_target":500,"validation_tasks":50,"test_tasks":20,
        "purpose":"Task-level freeze before production; old test is diagnostic only; all failures retained"}
    write_json(path,value)
    write_json(folder/"protocol.json",{"created_before_production":True,
        "plan":"paper/EXPERIMENT_PLAN_20261013_000010.md","train_target":500,
        "manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "final_test_runs":100,"basic_target":95,"strict_target":90,"planning_p95_ms":100})
    return value


def teacher_path(entry,folder=BASE):
    return Path(entry.get("teacher_path",folder/"teachers"/entry["id"]))


def status(entry,folder=BASE):
    path = teacher_path(entry,folder)/"result.json"
    return read(path) if path.exists() else None


def teacher(entry,folder):
    output = teacher_path(entry,Path(folder))
    output.mkdir(parents=True,exist_ok=True)
    if (output/"result.json").exists():
        return entry["id"]
    from contextlib import redirect_stdout,redirect_stderr
    with (output/"run.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
        run_episode(Task(**entry["task"]),output,mode="teacher",progress=True)
    return entry["id"]


def production_summary(manifest,folder):
    rows=[]
    for e in manifest["tasks"]:
        if e["split"] not in ("train","validation"):
            continue
        r=status(e,folder)
        if r:
            rows.append({"id":e["id"],"split":e["split"],"origin":e["origin"],
                "completed":r["completed"],"passed":r["accepted_for_teacher_dataset"],
                "strict":bool(r["meets_stage_tracking"] and r["meets_original_tracking_precision"]),
                "failure":r["failure"],"wall_seconds":r["wall_seconds"]})
    return {"total_primary_tasks":551,"finished":len(rows),"completed":sum(x["completed"] for x in rows),
        "passed":sum(x["passed"] for x in rows),"strict":sum(x["strict"] for x in rows),
        "train_admitted":sum(x["passed"] and x["split"]=="train" for x in rows),
        "validation_admitted":sum(x["passed"] and x["split"]=="validation" for x in rows),
        "training_target":500,"episodes":rows}


def produce(folder=BASE,workers=4):
    manifest=freeze(folder)
    # Preserve the actual importable code used by all teacher workers.
    from .evaluation import snapshot
    session=folder/"production_sessions"/datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot(session)
    write_json(session/"resume.json",{"teacher_results_reused":True,"physical_model_unchanged":True})
    mid=None
    errors=[]
    def progress():
        nonlocal mid
        value=production_summary(manifest,folder)
        value["worker_errors"]=errors
        write_json(folder/"teachers/summary.json",value)
        print(json.dumps({k:v for k,v in value.items() if k not in ("episodes","worker_errors")}),flush=True)
        if value["train_admitted"]>=150 and mid is None and not (folder/"train_mid/model.pt").exists():
            log=(folder/"mid.stdout.log").open("a",encoding="utf-8")
            mid=subprocess.Popen([sys.executable,"-m","tracking_diffusion.round2_data","midtrain","--folder",str(folder)],
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            log.close()
            write_json(folder/"mid_process.json",{"pid":mid.pid,"started_at_train_admitted":value["train_admitted"]})
        return value
    primary=[e for e in manifest["tasks"] if e["origin"]=="paper500_v2" and e["split"] in ("train","validation") and e["role"]=="primary"]
    # Validation comes first so both snapshot sizes have usable independent validation.
    primary.sort(key=lambda e:(e["split"]!="validation",e["id"]))
    reserve=[e for e in manifest["tasks"] if e.get("role")=="reserve"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group in [primary]+[reserve[i:i+workers] for i in range(0,len(reserve),workers)]:
            if group is not primary and progress()["train_admitted"]>=500:
                break
            jobs=iter([e for e in group if status(e,folder) is None])
            pending={}
            def submit():
                e=next(jobs,None)
                if e is not None:
                    pending[pool.submit(teacher,e,str(folder))]=e
            for _ in range(workers): submit()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for f in done:
                    e=pending.pop(f)
                    try: f.result()
                    except Exception as exc: errors.append({"id":e["id"],"error":repr(exc)})
                    progress()
                    submit()
    value=progress()
    value["production_complete"]=True
    write_json(folder/"teachers/summary.json",value)
    if mid:
        mid.wait()
        write_json(folder/"mid_exit.json",{"returncode":mid.returncode})


def build(folder=BASE,target=500,name=None):
    from model_test.robot_model_spec_v5 import RobotModelSpecV5
    from .spline_data import build_episode,metadata
    output=folder/(name or f"data_{target}")
    if (output/"dataset.npz").exists():
        return output/"dataset.npz"
    output.mkdir(parents=True,exist_ok=True)
    manifest=freeze(folder)
    spec=RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
    records,rows=[],[]
    train_count=0
    cache=folder/"label_cache"
    cache.mkdir(exist_ok=True)
    for e in manifest["tasks"]:
        if e["split"] not in ("train","validation") or (e["split"]=="train" and train_count>=target): continue
        result=status(e,folder)
        if not result: continue
        row={"id":e["id"],"split":e["split"],"admitted":False,"teacher_path":str(teacher_path(e,folder))}
        if not result["accepted_for_teacher_dataset"]:
            row["reason"]="Teacher admission failed; attempt retained"
        else:
            cp=cache/(e["id"]+".npz")
            if cp.exists():
                with np.load(cp) as f: arrays={k:f[k] for k in f.files}
                rejected=[]
            else:
                arrays,rejected=build_episode(teacher_path(e,folder),spec)
                if not rejected: np.savez_compressed(cp,**arrays)
            if rejected:
                row.update(reason="Incomplete fitting; whole task excluded",rejected_windows=rejected)
            else:
                arrays["split"]=np.full(len(arrays["time"]),int(e["split"]=="validation"))
                arrays["seed"]=np.full(len(arrays["time"]),e["task"]["seed"])
                records.append(arrays)
                row.update(admitted=True,windows=len(arrays["time"]),max_fit_error_rad=float(arrays["fit_error"].max()))
                train_count+=int(e["split"]=="train")
        rows.append(row)
        write_json(output/"coverage.json",{"episodes":rows,"train_admitted":train_count,"test_never_loaded":True})
    if not records: raise ValueError("No supervision")
    merged={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    merged["metadata"]=np.array(json.dumps(dict(metadata(),task_manifest=str((folder/"manifest.json").resolve()),round=2)))
    np.savez_compressed(output/"dataset.npz",**merged)
    write_json(output/"summary.json",{"train_tasks":train_count,"validation_tasks":sum(r["admitted"] and r["split"]=="validation" for r in rows),
        "train_windows":int(np.sum(merged["split"]==0)),"validation_windows":int(np.sum(merged["split"]==1)),
        "max_fit_error_rad":float(merged["fit_error"].max()),"test_never_loaded":True,
        "dataset_sha256":hashlib.sha256((output/"dataset.npz").read_bytes()).hexdigest()})
    return output/"dataset.npz"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","produce","build","midtrain"))
    p.add_argument("--folder",type=Path,default=BASE)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--target",type=int,default=500)
    a=p.parse_args()
    if a.stage=="freeze": freeze(a.folder)
    elif a.stage=="produce": produce(a.folder,a.workers)
    elif a.stage=="build": build(a.folder,a.target)
    else:
        data=build(a.folder,150)
        subprocess.run([sys.executable,"-m","tracking_diffusion.paper_model","--dataset",str(data),
            "--output",str(a.folder/"train_mid"),"--steps","12000","--device","cuda"],check=True)


if __name__=="__main__": main()
