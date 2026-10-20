"""Reviewer-only saved-array checks. No project imports, optimizer or physics."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/v7_1/capture_feasibility_001"
DEST = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)
def digest(x):
    return hashlib.sha256(json.dumps(x, default=plain, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def filehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
run = read(OUT / "run.json")
ident = run["identity"]
selection = ident["selection"]
task = selection["entry"]["task"]
assert selection == read(ROOT / "ff_prior_capture_feasibility/evidence/selection.json")
assert task == read(OUT / "frozen_task.json")
assert digest({k:v for k,v in task.items() if k != "task_hash"}) == task["task_hash"]
assert len(selection["entry"]["scene"]["objects"]) == 5
assert all(x["size"][0] == .025 for x in selection["entry"]["scene"]["objects"])
physical = read(OUT / "physical_contract.json")
old_physical = read(ROOT / "artifacts/v7_1/sample_alignment_001/physical_contract.json")
assert physical["acceptance"] == old_physical["acceptance"] == task["contract"]["acceptance"]
assert physical["identity"] == old_physical["identity"]
assert physical["initial_state_id"] == old_physical["initial_state_id"]
lo, hi = [np.array(physical["identity"][k]) for k in ("planner_lower", "planner_upper")]
for p,h in ident["source_hashes"].items():
    assert filehash(ROOT/p) == h, p
with zipfile.ZipFile(OUT/"executed_source.zip") as z:
    for p,h in ident["source_hashes"].items():
        assert hashlib.sha256(z.read(p)).hexdigest() == h, p
binding = dict(ident["binding"]["files"], **ident["binding"]["previous_source_hashes"])
for p,h in binding.items():
    assert filehash(ROOT/p) == h, p
old = ROOT/"artifacts/v7_1/sample_alignment_001"
original = next(x for x in read(old/"frozen_tasks.json")["application"]["entries"] if x["sample_id"]=="capture_5_obstacles")
assert original == selection["entry"]
z0 = np.asarray(read(old/"application/capture_5_obstacles/aligned/candidates/00.json")["Z"])
assert np.array_equal(z0, selection["seeds"][0]["Z"])
for i,m in enumerate(("aligned","plain","structured"),1):
    selected = read(old/f"application/capture_3_obstacles/{m}/execution/selection.json")["selected_id"]
    candidate = read(old/f"application/capture_3_obstacles/{m}/candidates/{selected:02d}.json")
    assert candidate["eligible"]
    assert np.array_equal(candidate["Z"], selection["seeds"][i]["Z"])
phase = np.r_[np.linspace(0,1,20)[1:-1],1.]
bow = np.sin(np.pi*phase)**2
bow[-1] = 0.
for seed in selection["seeds"][4:]:
    z = z0.copy()
    z[:,seed["joint"]] += seed["sign"]*.45*bow
    assert np.array_equal(z,seed["Z"])
goals = np.array(task["condition"]["goals"])
def pose(actual):
    actual = np.asarray(actual)
    return np.r_[((actual[:,:3,3]-goals[:,:3,3])/.005).ravel(), (Rotation.from_matrix(np.swapaxes(goals[:,:3,:3],-1,-2)@actual[:,:3,:3]).as_rotvec()/np.deg2rad(1)).ravel()]
def norms(e): return np.linalg.norm(e.reshape(4,3),axis=1)
def reference(z):
    q0=np.asarray(task["condition"]["initial_features"])
    control=np.vstack([np.repeat(q0[None],3,axis=0),z[:18],np.repeat(z[-1:],3,axis=0)])
    return digest({"C":control,"knots":np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)],"degree":3,"T":task["condition"]["duration"],"hold":.5})
def metrics(a):
    return {"Q":float(max(norms(a["goal"]))),"collision_violation":float(np.mean(np.minimum((a["current_minima"]-.0002)/.003,0)**2)),"minimum_clearance_m":float(min(a["current_minima"]))}
counts={"iterations":0,"derivative_columns":0,"blocked_columns":0,"accepted_updates":0,"solver_success":0,"solver_pose_feasible":0,"zero_step_pose_feasible":0,"evaluated_trials":0,"quick_rejected_trials":0}
rows=[]
for seed in selection["seeds"]:
    d=OUT/"teachers"/seed["id"]
    start=np.clip(np.array(seed["Z"]),lo,hi)
    sr=read(d/"seed.json")
    assert sr["frozen_seed"]==seed and np.array_equal(sr["actual_start_Z"],start)
    initial=read(d/"initial.json")
    assert np.array_equal(initial["Z"],start)
    previous=start
    for p in sorted(d.glob("iterations/*/record.json")):
        r=read(p)
        with np.load(p.with_name("linearization.npz")) as a:
            assert np.array_equal(previous,a["Z"])
            assert digest(a["Z"])==r["before_Z_hash"] and digest(a["next_Z"])==r["after_Z_hash"]
            assert np.max(abs(pose(a["current_ee"])-a["goal"]))<1e-12
            for key,val in metrics(a).items(): assert abs(val-r["before"][key])<1e-12
            zero=bool(max(norms(a["goal"]))<=.65 and np.all(a["Z"]>=lo) and np.all(a["Z"]<=hi))
            counts["zero_step_pose_feasible"]+=zero
            for j,epsilon in enumerate(a["steps"]):
                if epsilon==0:
                    assert np.all(a["jg"][:,j]==0) and np.all(a["jd"][:,j]==0)
                    counts["blocked_columns"]+=1
                    continue
                assert np.allclose((pose(a["perturb_ee"][j])-a["goal"])/epsilon,a["jg"][:,j],atol=1e-10,rtol=1e-10)
                assert np.allclose((a["perturb_distances"][j]-a["active_distances"])/epsilon,a["jd"][:,j],atol=1e-10,rtol=1e-10)
                delta=np.zeros(323); delta[j]=epsilon
                perturbed=a["Z"]+delta.reshape(19,17)
                assert np.all(perturbed>=lo-1e-15) and np.all(perturbed<=hi+1e-15)
            accepted=[t for t in r["trials"] if t.get("accepted")]
            for t in r["trials"]:
                if t["status"]=="QUICK_REJECTED":
                    counts["quick_rejected_trials"]+=1
                    continue
                counts["evaluated_trials"]+=1
                b=r["before"]
                expected=(t["Q"]<=.8 and t["collision_violation"]<b["collision_violation"]-1e-10) if b["Q"]<=.8 else (sum(x*x for x in t["goal_groups"])+t["collision_violation"]<sum(x*x for x in b["goal_groups"])+b["collision_violation"]-1e-9)
                assert expected==t["accepted"]
            assert len(accepted)==int(r["accepted"])
            nextz=a["Z"]+accepted[0]["scale"]*a["step"].reshape(19,17) if accepted else a["Z"]
            assert np.array_equal(nextz,a["next_Z"])
            previous=a["next_Z"].copy()
            counts["iterations"]+=1
            counts["derivative_columns"]+=len(a["steps"])
            counts["accepted_updates"]+=r["accepted"]
            counts["solver_success"]+=r["solver"]["success"]
            counts["solver_pose_feasible"]+=r["solver"]["linear_pose_feasible"]
    final=read(d/"result.json")
    with np.load(d/"final_path.npz") as a:
        assert np.array_equal(previous,a["Z"]) and np.array_equal(previous,final["Z"])
        assert abs(max(norms(pose(a["ee"])))-final["Q"])<1e-12
        assert abs(min(a["minima"])-final["minimum_clearance_m"])<1e-12
    validation=read(d/"validation.json")
    assert np.array_equal(previous,validation["Z"])
    rows.append({"seed":seed["id"],"iterations":final["iterations"],"initial_Q":initial["Q"],"final_Q":final["Q"],"coarse_clearance_m":final["minimum_clearance_m"],"goal_pass":validation["frozen_goal_pass"],"fine_clearance_m":validation["validation"]["geometry"]["minimum_clearance_m"],"geometry_step_m":validation["validation"]["geometry"]["maximum_geometry_step_m"],"eligible":validation["eligible"]})
validation_count=0
for p in OUT.glob("**/validation.json"):
    r=read(p); v=r["validation"]
    assert reference(np.array(r["Z"]))==r["reference_hash"]
    assert r["task_hash"]==task["task_hash"] and not r["goal_relabel"]
    with np.load(p.with_name("prediction.npz")) as a:
        ns=norms(pose(a["ee"][-1]))
        assert np.allclose(ns[:2]*.005,v["frozen_goal_position_error_m"],rtol=0,atol=1e-12)
        assert np.allclose(ns[2:]*np.deg2rad(1),v["frozen_goal_rotation_error_rad"],rtol=0,atol=1e-12)
        assert r["frozen_goal_pass"]==bool(max(ns)<=1.)
        assert len(a["phase"])==r["validation_grids"][-1]["intervals"]+1
    assert r["eligible"]==(r["frozen_goal_pass"] and all(v["gates"][x] for x in v["required"]))
    assert v["gates"]["geometry"]==all(g["gates"]["geometry"] for g in r["validation_grids"])
    validation_count+=1
b=np.asarray(read(old/"learning/aligned/codec.json")["basis"])
maps={"full323":np.eye(323)}
maps["codec102"]=np.einsum("tk,kjv->jtv",b,np.eye(102).reshape(102,6,17).transpose(1,0,2)).reshape(102,323).T
h=phase*phase*(3-2*phase); bow=np.sin(np.pi*phase); bow[-1]=0
dirs=np.concatenate([np.eye(17)[:,None,:]*h[None,:,None],np.eye(17)[10:,None,:]*bow[None,:,None]])
maps["legacy24"]=np.einsum("tk,sk,nsj->ntj",b,b,dirs).reshape(24,323).T
local=read(OUT/"local_subspaces/result.json")
local_check={}
with np.load(OUT/"teachers/00_original_failed/iterations/00/linearization.npz") as a:
    for kind,m in maps.items():
        with np.load(OUT/f"local_subspaces/{kind}.npz") as d:
            assert np.array_equal(m,d["direction_map"])
            assert np.allclose(a["jg"]@m,d["jg"],atol=1e-12,rtol=1e-12)
            assert np.allclose(a["jd"]@m,d["jd"],atol=1e-12,rtol=1e-12)
            assert np.allclose(np.linalg.svd(d["jg"],compute_uv=False),local["variants"][kind]["goal_singular_values"])
            local_check[kind]={"direction_rank":int(np.linalg.matrix_rank(m)),"goal_rank":int(np.linalg.matrix_rank(d["jg"])),"zero_step_pose_slack":float(min(.65**2-norms(a["goal"])**2)),"coefficient_step_max":float(abs(d["step"]).max()),"control_delta_max":float(abs(m@d["step"]).max()),"linear_returned_step_Q":float(max(norms(a["goal"]+d["jg"]@d["step"]))),"step_matches_teacher":bool(kind=="full323" and np.array_equal(a["step"],d["step"]))}
repair=read(OUT/"legacy_local/repair.json")
previous=z0
for stage in repair["stages"]:
    assert np.array_equal(previous,stage["raw_Z"])
    assert digest(stage["output_Z"])==stage["output_hash"]
    previous=np.array(stage["output_Z"])
assert np.array_equal(previous,repair["Z"])
assert np.array_equal(previous,read(OUT/"legacy_local/validation.json")["Z"])
assert sum(s["coupled_calls"] for s in repair["stages"])==repair["used"]==109
selected=read(OUT/"teacher_execution_selection.json")
choices=[(s["id"],read(OUT/"teachers"/s["id"]/"validation.json")) for s in selection["seeds"]]
assert selected["candidate_digest"]==digest(choices) and selected["ids"]==[]
assert not list(OUT.glob("**/outcome.json"))
ledger=read(OUT/"ledger.json")
assert ledger["inflight"] is None and not ledger.get("unresolved_call_accounting")
assert len(ledger["completed"])==14
assert sum(x for k,x in ledger["calls"].items() if k.endswith(".physics_steps"))==0
report={"status":"PASS_SAVED_ALGEBRA_ONLY","source_hashes_checked":len(ident["source_hashes"]),"binding_files_checked":len(binding),"producer_head":run["producer_head"],"task_hash":task["task_hash"],"counts":counts,"teacher_rows":rows,"validation_records":validation_count,"local_algebra":local_check,"legacy_used":repair["used"],"selected_teacher_ids":selected["ids"],"executions":0,"coupled_ledger":sum(x for k,x in ledger["calls"].items() if k.endswith(".coupled")),"paper_claim_audit":"unavailable" if not (ROOT/"paper/PAPER_CLAIM_AUDIT.json").exists() else "available","scope_limit":"No new MuJoCo/FK/propagation/optimizer. Saved geometry distances and rejected trial metrics are read, not independently re-simulated; solver internal candidate is not saved."}
(DEST/"algebra_evidence.json").write_text(json.dumps(report,default=plain,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="teacher_rows"},default=plain,indent=2,ensure_ascii=False))
