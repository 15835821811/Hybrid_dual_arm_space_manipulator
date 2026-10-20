"""Fresh reviewer: reads frozen results, performs algebra only, writes beside this file."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
from pathlib import Path
DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.chdir(ROOT)
import json, hashlib, zipfile, collections, time
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from ff_prior.storage import read_json, digest
from ff_prior_backbone_v2.protocol import CONTROL
from ff_prior_latent.learning import Codec
from ff_prior_endpoint_jacobian.data import load
from ff_prior_backbone_v2.audit_results import verify

OUT = ROOT / "artifacts/v7_1/backbone_pair_smoke_002"
run = read_json(OUT / "run.json")
ledger = read_json(OUT / "ledger.json")
result = {"scope": "fresh independent read-only review; no optimizer.step, propagation, replay or resume"}
def save():
    (DEST / "reviewer_raw_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
def dist(a):
    a = np.asarray(a, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a,95)), max=float(a.max()), min=float(a.min()))
def metrics(r, execution=False):
    v = r if execution else r["validation"]
    prefix = "final_" if execution else "frozen_goal_"
    p = max(v[prefix + "position_error_m"])*1000
    a = float(np.rad2deg(max(v[prefix + "rotation_error_rad"])))
    return dict(position_mm=p, rotation_deg=a, Q=max(p/5,a))
def aggregate(rs, execution=False):
    ms = [metrics(r, execution) for r in rs]
    return {"denominator": len(rs), "missing_endpoint": 0,
            "qualified": sum(r.get("eligible", r.get("level") == "EXECUTION_VERIFIED") for r in rs),
            "pose_pass": sum(m["Q"]<=1 for m in ms),
            **{k:dist([m[k] for m in ms]) for k in ms[0]}}
def pairs(a,b,execution=False):
    result = {}
    for k in ("position_mm","rotation_deg","Q"):
        delta = np.array([metrics(y,execution)[k]-metrics(x,execution)[k] for x,y in zip(a,b)])
        result[k] = {**dist(delta), "T_better":int((delta<0).sum()),"T_worse":int((delta>0).sum()),"equal":int((delta==0).sum())}
    return result

print("load exact data", flush=True)
codec = Codec(read_json(CONTROL / "learning/goal_loss/codec.json"))
arrays, rows, G, data = load(codec)
result["data"] = {"TRAIN":len(data["tr"]), "VAL":len(data["va"]), "x_shape":list(data["x"].shape),
                  "c_shape":list(data["c"].shape),"G_shape":list(G.shape)}
print("fresh raw helper, identity hashes, six gradient/AdamW reconstructions, 192 neural samples, 72 saved trajectories", flush=True)
result["raw_helper"] = (read_json(DEST / "reviewer_raw_audit.json")["raw_helper"]
                        if "--continue-own-audit" in sys.argv else verify(OUT, run["identity"], arrays, rows, G, data))
save()
assert result["raw_helper"]["status"] == "PASS", result["raw_helper"]
result["binding"] = {"source_files":len(run["identity"]["source_hashes"]), "input_files":len(run["identity"]["binding"]["files"]),
                     "producer_head":run["producer_head"]}
result["training"] = {}
for kind in ("mlp","transformer"):
    d = OUT / "learning" / kind
    t = read_json(d/"training.json")
    p = torch.load(d/"progress.pt",map_location="cpu",weights_only=False)
    counts = np.asarray(p["counts"])
    assert np.array_equal(counts, np.asarray([t["train_draw_counts"][arrays[3][i]] for i in data["tr"]]))
    assert len(counts)==14222 and counts.min()>0 and counts.sum()==384000
    assert len(p["segments"])==300 and p["segments"]==t["segments"]
    assert torch.isfinite(torch.cat([x.flatten() for x in p["weights"].values()])).all()
    assert all(torch.isfinite(v["exp_avg"]).all() and torch.isfinite(v["exp_avg_sq"]).all() and v["exp_avg_sq"].abs().sum()>0 for v in p["optimizer"]["state"].values())
    proofs = []
    for step in (1,3000,6000):
        pre=torch.load(d/"updates"/f"{step:04d}"/"before.pt",map_location="cpu",weights_only=False)
        post=torch.load(d/"updates"/f"{step:04d}"/"after.pt",map_location="cpu",weights_only=False)
        assert all(float(s["step"])==step for s in post["optimizer"]["state"].values())
        assert all(float(s["step"])==step-1 for s in pre["optimizer"]["state"].values())
        assert any(not torch.equal(pre["weights"][k],post["weights"][k]) for k in pre["weights"])
        proofs.append({"step":step,"before_states":len(pre["optimizer"]["state"]),"after_states":len(post["optimizer"]["state"])})
    result["training"][kind] = {k:t[k] for k in ("updates","actual_TRAIN_draws","TRAIN_rows_seen","wall_s","world_events","VAL_gradient_updates","VAL_checkpoint_selection","initial_weight_hash","trained_weight_hash")}
    result["training"][kind].update(parameter_count=t["metadata"]["parameter_count"], minimum_row_draws=int(counts.min()), maximum_row_draws=int(counts.max()), optimizer_states=len(p["optimizer"]["state"]), segments=len(p["segments"]), proof_states=proofs)
result["unguided"] = {}
ug={}
for kind in ("mlp","transformer"):
    ps=sorted(OUT.glob(f"unguided/*/{kind}/*.json"))
    assert len(ps)==96
    ug[kind]=[read_json(p) for p in ps]
    result["unguided"][kind]=aggregate(ug[kind])
    result["unguided"][kind]["by_seed"]={str(s):aggregate([read_json(p) for p in sorted(OUT.glob(f"unguided/{s}/{kind}/*.json"))]) for s in range(2)}
    result["unguided"][kind]["model_gate_failures"]=dict(collections.Counter(k for r in ug[kind] for k,v in r["validation"]["gates"].items() if not v))
    result["unguided"][kind]["shrink"] = dist([r["preparation"]["coefficient_shrink"] for r in ug[kind]])
result["unguided"]["paired"]=pairs(ug["mlp"],ug["transformer"])
result["methods"]={}
raw={}
for kind in ("mlp","transformer","structured","learned_mean"):
    root=OUT if kind in ("mlp","transformer") else CONTROL
    cs=[read_json(p) for p in sorted(root.glob(f"candidates/*/*/{kind}/*.json"))]
    es=[read_json(p) for p in sorted(root.glob(f"executions/*/*/{kind}/report.json"))]
    os_=[read_json(p) for p in sorted(root.glob(f"executions/*/*/{kind}/outcome.json"))]
    assert len(cs)==180 and len(es)==36 and len(os_)==36
    result["methods"][kind] = {"candidate":aggregate(cs),"execution":aggregate(es,True),
       "candidate_status":dict(collections.Counter(r["status"] for r in cs)),
       "execution_status":dict(collections.Counter(r["status"] for r in os_)),
       "new_physics":root==OUT,
       "candidate_wall_s":dist([r["wall_s"] for r in cs]),
       "coupled":dist([r["model_calls"]["coupled"] for r in cs]),
       "hold_position_mm":dist([1000*max(r["hold_max_position_error_m"]) for r in es]),
       "hold_rotation_deg":dist([float(np.rad2deg(max(r["hold_max_rotation_error_rad"]))) for r in es])}
    raw[kind]=(cs,es)
result["paired"]={f"T_vs_{other}_{stage}":pairs(raw[other][i],raw["transformer"][i],i==1) for other in ("mlp","structured","learned_mean") for i,stage in enumerate(("candidate","execution"))}
result["ledger"]={"attempts":ledger["attempts"],"completed":len(ledger["completed"]),"calls":ledger["calls"],
                  "inflight":ledger["inflight"],"unresolved":ledger.get("unresolved_call_accounting")}
result["budget"]={"elapsed_s":read_json(OUT/"status.json")["elapsed_s"],"output_bytes":sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),"query":read_json(OUT/"query_budget.json")}
result["paper_claim_audit"] = read_json(ROOT/"paper/PAPER_CLAIM_AUDIT.json") if (ROOT/"paper/PAPER_CLAIM_AUDIT.json").exists() else "unavailable"
save()
print(json.dumps({"raw_helper":result["raw_helper"],"unguided":result["unguided"],"methods":result["methods"],"paired":result["paired"],"budget":result["budget"]},indent=2),flush=True)
