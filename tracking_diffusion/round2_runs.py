"""Explicit second-round run protocols; diagnostics, ablations and fixed test."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .environment import Task
from .evaluation import snapshot
from .pipeline import run_episode,write_json
from .round2_data import BASE,freeze,read


def worker(entry,output,mode,checkpoint,seed,options):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/"result.json"
    if result_path.exists():
        result=read(result_path)
    else:
        with (output/"run.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            policy=None
            if mode=="diffusion":
                from .paper_model import PaperPolicy
                policy=PaperPolicy(Path(checkpoint),options["device"],seed,options["candidates"],options["steps"],
                    options["guidance"],options["projection"],options["scoring"],options["continuity"],options["diagnostic"],options.get("cuda_graph",False))
                if entry["split"]=="test" and entry["task"]["seed"] in policy.train_seeds+policy.val_seeds:
                    raise ValueError("Test task leaked into learning")
            task=Task(**entry["task"])
            if options["duration"]: task=replace(task,duration=options["duration"])
            result,_,_=run_episode(task,output,policy,mode,progress=False)
    return {"id":entry["id"],"sampling_seed":seed if mode=="diffusion" else None,
        "completed":result["completed"],"passed":result["meets_stage_tracking"],
        "strict":bool(result["meets_stage_tracking"] and result["meets_original_tracking_precision"]),
        "result_path":str(result_path),"failure":result["failure"],"metrics":result["metrics"],
        "planning_p95_ms":result["sampling_and_selection_p95_ms"],"wall_seconds":result["wall_seconds"]}


def run(folder,name,mode,checkpoint,names,split,seeds,workers,options):
    manifest=freeze(folder)
    entries=[e for e in manifest["tasks"] if (not names or e["id"] in names) and (not split or e["split"]==split)]
    if names and len(entries)!=len(set(names)): raise ValueError("Unknown or duplicate task")
    if not entries: raise ValueError("Empty evaluation")
    output=folder/name
    output.mkdir(parents=True,exist_ok=True)
    snapshot(output,Path(checkpoint) if checkpoint else None)
    protocol={"names":[e["id"] for e in entries],"mode":mode,"checkpoint":str(checkpoint),
        "seeds":seeds if mode=="diffusion" else [],"options":options,"workers":workers}
    if (output/"protocol.json").exists() and read(output/"protocol.json")!=protocol:
        raise ValueError("Evaluation protocol changed")
    write_json(output/"protocol.json",protocol)
    jobs=[]
    for e in entries:
        for seed in seeds if mode=="diffusion" else [73]:
            jobs.append((e,str(output/e["id"]/(f"sample_{seed}" if mode=="diffusion" else "teacher")),mode,str(checkpoint),seed,options))
    rows=[]
    def record(row):
        rows.append(row)
        value={"total":len(jobs),"finished":len(rows),"completed":sum(x["completed"] for x in rows),
            "passed":sum(x["passed"] for x in rows),"strict":sum(x["strict"] for x in rows),
            "episodes":sorted(rows,key=lambda x:(x["id"],x["sampling_seed"] or 0)),"mode":mode,
            "single_worker":workers==1,"timing_isolated":workers==1 and options.get("isolated_timing",False),"protocol":protocol}
        write_json(output/"summary.json",value)
        print(json.dumps({k:v for k,v in value.items() if k not in ("episodes","protocol")},ensure_ascii=False),flush=True)
    def failed(job,exc):
        return {"id":job[0]["id"],"sampling_seed":job[4],"completed":False,"passed":False,"strict":False,
            "failure":repr(exc),"result_path":str(Path(job[1])/"result.json")}
    if workers==1:
        for job in jobs:
            try: row=worker(*job)
            except Exception as exc: row=failed(job,exc)
            record(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(worker,*job):job for job in jobs}
            for future in as_completed(pending):
                try: row=future.result()
                except Exception as exc: row=failed(pending[future],exc)
                record(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=BASE)
    p.add_argument("--name",required=True)
    p.add_argument("--mode",default="diffusion",choices=("teacher","diffusion"))
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--names",nargs="+")
    p.add_argument("--split",choices=("validation","test","diagnostic"))
    p.add_argument("--seeds",nargs="+",type=int,default=[73,137,211])
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--device",default="cuda")
    p.add_argument("--steps",type=int,default=60)
    p.add_argument("--candidates",type=int,default=5)
    p.add_argument("--no-guidance",action="store_true")
    p.add_argument("--projection",choices=("auto","on","off"),default="auto")
    p.add_argument("--scoring",choices=("legacy","prefix","nonlinear"),default="legacy")
    p.add_argument("--continuity",type=float,default=0.)
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--duration",type=float)
    p.add_argument("--cuda-graph",action="store_true")
    p.add_argument("--isolated-timing",action="store_true")
    a=p.parse_args()
    if a.mode=="diffusion" and not a.checkpoint: p.error("Diffusion requires checkpoint")
    if a.split=="test" and a.duration: p.error("Final test requires full duration")
    options={"device":a.device,"steps":a.steps,"candidates":a.candidates,"guidance":not a.no_guidance,
        "projection":None if a.projection=="auto" else a.projection=="on","scoring":a.scoring,
        "continuity":a.continuity,"diagnostic":a.diagnostic,"duration":a.duration,"cuda_graph":a.cuda_graph,"isolated_timing":a.isolated_timing}
    run(a.folder,a.name,a.mode,a.checkpoint,a.names,a.split,a.seeds,a.workers,options)


if __name__=="__main__": main()
