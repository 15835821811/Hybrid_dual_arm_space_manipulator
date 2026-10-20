import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="2"
import sys,json,hashlib
from pathlib import Path
DEST=Path(__file__).resolve().parent;ROOT=DEST.parents[3]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True;os.chdir(ROOT)
import numpy as np
import torch
torch.set_num_threads(2)
from ff_prior.storage import read_json,digest
OUT=ROOT/"artifacts/v7_1/backbone_pair_smoke_002"
CONTROL=ROOT/"artifacts/v7_1/expand256_train_dev_001"
entries=read_json(OUT/"frozen_tasks.json")["entries"]
assert entries==read_json(CONTROL/"frozen_tasks.json")["entries"]
def metric(r):
    return {"p":1000*max(r["final_position_error_m"]),"r":float(np.rad2deg(max(r["final_rotation_error_rad"])))}
res={"frozen_DEV_exact":True,"selected_batch_pair":[],"strata":{},"source_summary_consistency":True}
for step in (1,3000,6000):
    a,b=[torch.load(OUT/"learning"/m/"updates"/f"{step:04d}"/"before.pt",map_location="cpu",weights_only=False) for m in ("mlp","transformer")]
    assert np.array_equal(a["indices"],b["indices"]) and torch.equal(a["t"],b["t"]) and torch.equal(a["noise"],b["noise"])
    res["selected_batch_pair"].append(step)
for label,js,seeds in [("DEV4",range(4),range(3)),("used8",range(4,12),range(3))]+[(f"seed{s}",range(12),[s]) for s in range(3)]:
    means={};rs={}
    for kind in ("mlp","transformer","structured","learned_mean"):
        root=OUT if kind in ("mlp","transformer") else CONTROL
        rs[kind]=[metric(read_json(root/"executions"/str(s)/entries[j]["sample_id"]/kind/"report.json")) for s in seeds for j in js]
        means[kind]={k:float(np.mean([r[k] for r in rs[kind]])) for k in ("p","r")}
    res["strata"][label]={"n":len(rs["mlp"]),"means":means,"T_vs_MLP_position_better":sum(y["p"]<x["p"] for x,y in zip(rs["mlp"],rs["transformer"]))}
summary=read_json(OUT/"paired_statistics.json")
own=read_json(DEST/"reviewer_raw_audit.json")
for kind in own["methods"]:
    for stage in ("candidate","execution"):
        a,b=own["methods"][kind][stage],summary["methods"][kind][stage]
        for key in ("denominator","missing_endpoint","qualified","pose_pass"): assert a[key]==b[key]
        for key in ("position_mm","rotation_deg","Q"):
            for stat in ("mean","median","p95","max"): assert abs(a[key][stat]-b[key][stat])<1e-12
for kind in ("mlp","transformer"):
    rows=[read_json(p) for p in OUT.glob(f"unguided/*/{kind}/*.json")]
    res[kind+"_unguided"]={"position_pass":sum(max(r["validation"]["frozen_goal_position_error_m"])<=.005 for r in rows),
        "rotation_pass":sum(max(r["validation"]["frozen_goal_rotation_error_rad"])<=np.pi/180 for r in rows),
        "MODEL_VALID":sum(r["validation"]["validation_level"]=="MODEL_VALID" for r in rows),
        "geometry_fail":sum(not r["validation"]["gates"]["geometry"] for r in rows)}
ledger=read_json(OUT/"ledger.json")
actual=sum(read_json(p)["model_calls"]["coupled"] for p in OUT.glob("candidates/*/*/*/*.json"))
un=sum(read_json(p)["calls"]["coupled"] for p in OUT.glob("unguided/*/*/*.json"))
assert actual==ledger["calls"]["candidate.coupled"] and un==ledger["calls"]["unguided.coupled"]
res["actual_coupled"]=actual+un
res["completed_statuses"]=sorted(set(v["status"] for v in ledger["completed"].values()))
res["journals"]={m:read_json(OUT/"learning"/m/"journal.json") for m in ("mlp","transformer")}
res["resume_handoff"]=read_json(ROOT/"ff_prior_backbone_v2/evidence/completed/resume_verification.json")
before=read_json(ROOT/"ff_prior_backbone_v2/evidence/completed/core_manifest_before_resume.json")
after=read_json(ROOT/"ff_prior_backbone_v2/evidence/completed/core_manifest_after_resume.json")
assert before==after
assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in after.items())
res["current_core_matches_handoff_manifest"]=len(after)
res["failed_001"]=read_json(ROOT/"ff_prior_backbone_v2/evidence/failed_launch_001.json")
(DEST/"reviewer_supplemental.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in res.items() if k!="failed_001"},ensure_ascii=False,indent=2))
