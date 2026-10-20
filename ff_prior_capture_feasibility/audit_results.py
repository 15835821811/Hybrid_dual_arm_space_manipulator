"""Read-only evidence checks; no new propagation, optimization or execution."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_fixed.correction import pose_vector
from .inputs import identity
from .protocol import P
from .solver import basis, metric
from ff_prior_latent.learning import Codec
from .protocol import SOURCE

def verify(out, ident):
    errors = []; derivatives = validations = executions = steps = 0
    try:
        assert ident == identity()
        task = ident["selection"]["entry"]["task"]
        assert read_json(out/"frozen_task.json") == task
        with zipfile.ZipFile(out/"executed_source.zip") as z:
            assert all(hashlib.sha256(z.read(p)).hexdigest() == h for p, h in ident["source_hashes"].items())
        codec = Codec(read_json(SOURCE/"learning/aligned/codec.json"))
        for p in sorted(out.glob("**/iterations/*/record.json")):
            r = read_json(p)
            with np.load(p.with_name("linearization.npz")) as a:
                z, next_z = a["Z"], a["next_Z"]
                e = pose_vector(a["current_ee"], task["condition"]["goals"])
                assert np.allclose(e, a["goal"], rtol=0, atol=1e-12)
                for j, epsilon in enumerate(a["steps"]):
                    if epsilon == 0:
                        assert np.all(a["jg"][:, j] == 0) and np.all(a["jd"][:, j] == 0)
                        continue
                    eg = (pose_vector(a["perturb_ee"][j], task["condition"]["goals"])-e)/epsilon
                    dg = (a["perturb_distances"][j]-a["active_distances"])/epsilon
                    assert np.allclose(eg, a["jg"][:, j], rtol=1e-10, atol=1e-10)
                    assert np.allclose(dg, a["jd"][:, j], rtol=1e-10, atol=1e-10)
                assert digest(z) == r["before_Z_hash"] and digest(next_z) == r["after_Z_hash"]
                accepted = [v for v in r["trials"] if v.get("accepted")]
                if accepted:
                    assert len(accepted) == 1
                    delta = np.einsum("i,ijk->jk", a["step"], basis(r["direction_basis"], codec))
                    assert np.array_equal(next_z, z+accepted[0]["scale"]*delta)
                    b = r["before"]; n = accepted[0]
                    if b["Q"] <= P["accept_goal_fraction"]:
                        assert n["Q"] <= P["accept_goal_fraction"] and n["collision_violation"] < b["collision_violation"]-1e-10
                    else:
                        assert (sum(v*v for v in n["goal_groups"])+n["collision_violation"]
                            < sum(v*v for v in b["goal_groups"])+b["collision_violation"]-1e-9)
                else:
                    assert np.array_equal(z, next_z)
                derivatives += len(a["steps"])
        records = {}
        for p in out.glob("**/validation.json"):
            r = read_json(p); check_candidate_validation(r, task)
            assert r["task_hash"] == task["task_hash"] and not r["goal_relabel"]
            if r["validation_grids"]:
                assert all(g["intervals"] in P["validation_intervals"] for g in r["validation_grids"])
                if r["eligible"]:
                    assert r["validation"]["geometry"]["maximum_geometry_step_m"] <= .002
            records[r["reference_hash"]] = r
            validations += 1
        selected = read_json(out/"teacher_execution_selection.json")
        choices = [(s["id"], read_json(out/"teachers"/s["id"]/"validation.json")) for s in ident["selection"]["seeds"]]
        expected = [sid for sid, r in sorted((v for v in choices if v[1]["eligible"]),
                                              key=lambda v: (v[1]["score"], v[0]))[:2]]
        assert expected == selected["ids"] and selected["before_execution"]
        assert selected["candidate_digest"] == digest(choices)
        for p in out.glob("**/outcome.json"):
            selection = read_json(p.with_name("selection.json"))
            assert selection["task_hash"] == task["task_hash"] and selection["before_execution"]
            result = read_json(p)
            if result["status"] == "NO_CANDIDATE":
                assert not selection["eligible"]
                continue
            assert records[selection["reference_hash"]]["eligible"]
            r, count = audit_physics(p.parent, task, selection["reference_hash"])
            assert r["level"] == result["status"]
            assert r["goal_source"] == "frozen_user_static_application_world_goal"
            executions += 1; steps += count
        ledger = read_json(out/"ledger.json")
        assert ledger["inflight"] is None and not ledger.get("unresolved_call_accounting")
        assert len(choices) == 12 and executions <= P["execution_limit"]
        assert sum(v for k, v in ledger["calls"].items() if k.endswith(".physics_steps")) == steps
        assert sum(v for k, v in ledger["calls"].items() if k.endswith(".coupled")) <= P["coupled_cap"]
    except Exception as e:
        import traceback
        errors.append({"error": repr(e), "traceback": traceback.format_exc()})
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "derivative_columns_checked": derivatives,
            "validation_records": validations, "actual_execution_records": executions,
            "native_plus_replay_steps": steps, "no_new_physics_or_optimization": True}

def report(out, audit):
    rows = [(p.parent.name, read_json(p)) for p in sorted((out/"teachers").glob("*/validation.json"))]
    execution = {p.parent.name: read_json(p) for p in (out/"executions").glob("*/outcome.json")}
    representation = {p.parent.name: read_json(p) for p in (out/"representation").glob("*/validation.json")}
    successes = sum(r["status"] == "EXECUTION_VERIFIED" for r in execution.values())
    calls = read_json(out/"ledger.json")["calls"]
    summary = {"status": "FINISHED" if audit["status"] == "PASS" else "FINISHED_WITH_BLOCKED_ITEMS",
        "raw_audit": audit["status"], "teacher_slots": len(rows),
        "model_qualified_teachers": sum(r["eligible"] for _, r in rows),
        "teacher_executions": execution, "execution_verified_teacher_witnesses": successes,
        "representation": {k: {"eligible": r["eligible"], "validation": r["validation"]} for k, r in representation.items()},
        "local_diagnosis": read_json(out/"local_subspaces/result.json"),
        "actual_coupled": sum(v for k, v in calls.items() if k.endswith(".coupled")),
        "TEST": False, "TRAIN_added": 0, "physics_relaxed": False,
        "infeasibility_proven": False, "independent_claim_review": "PENDING"}
    atomic_json(out/"summary.json", summary)
    text = ["# 固定五障碍可行路径诊断：后台结果", "",
            f"状态 {summary['status']}；原始审计 {audit['status']}；独立主张审查待收尾。",
            f"12槽中完整模型合格 {summary['model_qualified_teachers']}；真实执行重放见证 {successes}。",
            "未找到见证不等于任务不可达。未扩库、训练或运行TEST；目标、障碍与物理门未改变。",
            "", "|槽|完整合格|模型等级|终点位置mm|最小间距mm|执行|", "|---|---|---|---:|---:|---|"]
    for sid, r in rows:
        v = r["validation"]
        pos = max(v.get("frozen_goal_position_error_m", [float("nan")]))*1000
        clearance = v.get("geometry", {}).get("minimum_clearance_m", float("nan"))*1000
        text.append(f"|{sid}|{r['eligible']}|{v['validation_level']}|{pos:.4f}|{clearance:.4f}|{execution.get(sid,{}).get('status','未预选')}|")
    text += ["", "表示诊断：" + ("已运行，见summary.json及representation目录。" if representation else
                "BLOCKED：没有完整模型合格见证，不能据此判定102维表示不可能。"),
             "局部子空间与旧引导诊断已独立记录；不是同预算Diffusion收益实验。",
             "最终中文分析将在独立审查后写入本任务docs/ANALYSIS.md。"]
    (out/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary
