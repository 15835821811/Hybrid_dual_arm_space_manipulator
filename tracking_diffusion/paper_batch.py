"""Task-first data production and evaluation, without serial pilot pass gates."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .environment import Task
from .evaluation import snapshot
from .pipeline import run_episode,write_json
from .task_domain import DOMAIN,profile_hash,sample_task


def freeze(folder):
    manifest=folder/"manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    tasks=[]
    for i in range(100):
        task=Task(91000000+i*104729,27.,2,DOMAIN,False)
        tasks.append({"id":f"task_{i:03d}","task":asdict(task),
                      "split":"train" if i<80 else "validation" if i<90 else "test",
                      "parameters":sample_task(task.seed)})
    value={"domain":DOMAIN,"profile_sha256":profile_hash(),"sampling_seeds":[73,137,211],
           "tasks":tasks,"purpose":"100 new tasks, split before production, no pilot tasks in final test"}
    write_json(manifest,value)
    return value


def worker(entry,folder,mode,checkpoint=None,sampling_seed=73,guidance=True):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"result.json").exists():
        result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
    else:
        with (folder/"run.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            policy=None
            if mode=="diffusion":
                from .paper_model import PaperPolicy
                policy=PaperPolicy(checkpoint,"cuda",sampling_seed,5,60,guidance)
                if entry["split"]=="test" and entry["task"]["seed"] in policy.train_seeds+policy.val_seeds:
                    raise ValueError("Test task found in checkpoint's training/validation set")
            result,_,_=run_episode(Task(**entry["task"]),folder,policy,mode,progress=True)
    return {"id":entry["id"],"split":entry["split"],"sampling_seed":sampling_seed if mode=="diffusion" else None,
            "passed":result["meets_stage_tracking"],"strict":bool(result["meets_stage_tracking"] and result["meets_original_tracking_precision"]),
            "completed":result["completed"],"result_path":str(folder/"result.json"),"failure":result["failure"],
            "metrics":result["metrics"],"execution":result["execution"]}


def run_batch(folder,name,mode,workers=4,checkpoint=None,splits=None,sampling_seeds=(73,137,211)):
    manifest=freeze(folder)
    if manifest["profile_sha256"]!=profile_hash():
        raise ValueError("Domain changed")
    output=folder/name
    snapshot(output,checkpoint)
    jobs=[]
    for entry in manifest["tasks"]:
        if splits and entry["split"] not in splits:
            continue
        for seed in sampling_seeds if mode=="diffusion" else [73]:
            path=output/entry["id"]/(f"sample_{seed}" if mode=="diffusion" else "")
            jobs.append((entry,path,mode,checkpoint,seed))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(worker,*job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                result=future.result()
            except Exception as exc:
                result={"id":job[0]["id"],"split":job[0]["split"],"sampling_seed":job[-1],
                        "completed":False,"passed":False,"strict":False,"failure":repr(exc),"result_path":str(job[1])}
            results.append(result)
            write_json(output/"summary.json",{"mode":mode,"total":len(jobs),"finished":len(results),
                "completed":sum(r["completed"] for r in results),"passed":sum(r["passed"] for r in results),
                "strict":sum(r["strict"] for r in results),"episodes":sorted(results,key=lambda r:(r["id"],r.get("sampling_seed") or 0)),
                "timing_note":"Concurrent throughput run; do not interpret timings as isolated deployment latency"})
            print(json.dumps({"mode":mode,"finished":len(results),"total":len(jobs),"id":result["id"],
                              "passed":result["passed"],"strict":result["strict"],"failure":result["failure"]}),flush=True)


def build(folder,teacher_name,output):
    from model_test.robot_model_spec_v5 import RobotModelSpecV5
    from .spline_data import build_episode,metadata
    if (output/"dataset.npz").exists():
        raise FileExistsError("Dataset already exists")
    manifest=freeze(folder)
    spec=RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
    records,report=[],[]
    for entry in manifest["tasks"]:
        if entry["split"]=="test":
            continue
        path=folder/teacher_name/entry["id"]
        result=json.loads((path/"result.json").read_text(encoding="utf-8")) if (path/"result.json").exists() else {}
        record={"id":entry["id"],"split":entry["split"],"admitted":False}
        if not result.get("accepted_for_teacher_dataset",False):
            record["reason"]="teacher execution did not pass; task retained in denominator"
        else:
            arrays,rejected=build_episode(path,spec)
            if rejected:
                record["reason"]="incomplete window coverage; whole episode excluded from supervision"
                record["rejected_windows"]=rejected
            else:
                arrays["split"]=np.full(len(arrays["time"]),0 if entry["split"]=="train" else 1)
                arrays["seed"]=np.full(len(arrays["time"]),entry["task"]["seed"])
                records.append(arrays)
                record.update(admitted=True,windows=len(arrays["time"]),max_fit_error_rad=float(arrays["fit_error"].max()))
        report.append(record)
        write_json(output/"coverage.json",{"episodes":report,"metadata":metadata(),"test_never_loaded":True})
    if not records:
        raise ValueError("No valid supervision")
    merged={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    merged["metadata"]=np.array(json.dumps(dict(metadata(),task_manifest=str((folder/"manifest.json").resolve()))))
    np.savez_compressed(output/"dataset.npz",**merged)
    print(json.dumps({"admitted_tasks":sum(r["admitted"] for r in report),"attempted_train_val_tasks":len(report),"windows":len(merged["time"])}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","run","build"))
    p.add_argument("--folder",type=Path,default=Path("tracking_output/paper100_v1"))
    p.add_argument("--name",default="teachers")
    p.add_argument("--mode",default="teacher",choices=("teacher","diffusion","feedback-only"))
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--splits",nargs="+")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.stage=="freeze":
        freeze(args.folder)
    elif args.stage=="run":
        run_batch(args.folder,args.name,args.mode,args.workers,args.checkpoint,args.splits)
    else:
        build(args.folder,args.name,args.output or args.folder/"data")


if __name__=="__main__":
    main()
