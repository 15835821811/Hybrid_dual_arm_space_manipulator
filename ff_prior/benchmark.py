"""Frozen DEV tasks, matched candidates, model selection before any execution."""
from pathlib import Path
import time
import numpy as np
from .contracts import assert_frozen
from .storage import read_json,atomic_json,atomic_npz,digest
from .spline import Reference
from .coupled_predictor import pose_errors
from .model_gates import evaluate_reference
from .diffusion import load_checkpoint,ddim
from .guidance import FullPathCost,SubspaceGuidance,public_goal_initializer
from .physical_audit import audit_execution

METHODS = ("structured_prior_same_guidance","diffusion_in_denoising_guidance","diffusion_no_guidance")


def plan(public_task,checkpoint,budget,robot,initial,collision,profile,method,seed):
    assert_frozen(public_task)
    if method not in METHODS:
        raise ValueError("unknown planner")
    wall,cpu = time.perf_counter(),time.process_time()
    before = robot.calls.copy()
    cost = FullPathCost(public_task,robot,initial,collision,profile,budget.get("deadline"))
    base = public_goal_initializer(public_task,robot,initial) if method == METHODS[0] else None
    if method != METHODS[0]:
        model,scaler = load_checkpoint(checkpoint,initial.model_hash,initial.state_id)
    candidates = []
    for k in range(budget["candidates"]):
        candidate_seed = seed+k*997
        guide = SubspaceGuidance(cost,seed=candidate_seed)
        try:
            if base is not None:
                rng = np.random.default_rng(candidate_seed)
                z = base + .015*np.sin(np.linspace(0,np.pi,19))[:,None]*rng.normal(size=(1,17))
                logs = []
                for i in range(2):
                    z,info = guide(z,i)
                    logs.append(info)
                sampling = {"actual_network_calls":0,"guidance":logs,"initialization":"public_goal_generalized_jacobian"}
            else:
                z,sampling = ddim(model,scaler,public_task,budget["ddim_steps"],candidate_seed,
                                   guide if method==METHODS[1] else None)
            ref = Reference(z,np.asarray(public_task["condition"]["initial_features"]),public_task["condition"]["duration"])
            prediction,validation = evaluate_reference(robot,initial,ref,collision,profile,budget.get("deadline"))
            goal_valid = False
            score = None
            if prediction is not None:
                p,r = pose_errors(prediction["ee"][-1],public_task["condition"]["goals"])
                score = float(np.sum((p/profile.position_error_m)**2)+np.sum((r/profile.rotation_error_rad)**2))
                goal_valid = bool(np.all(p<=profile.position_error_m) and np.all(r<=profile.rotation_error_rad))
                validation["frozen_goal_position_error_m"] = p
                validation["frozen_goal_rotation_error_rad"] = r
            eligible = validation["validation_level"]=="MODEL_VALID" and goal_valid
            candidates.append({"candidate_id":k,"Z":z,"reference_hash":ref.reference_hash,"sampling":sampling,
                "validation":validation,"frozen_goal_pass":goal_valid,"eligible":eligible,"score":score,"seed":candidate_seed})
        except Exception as e:
            candidates.append({"candidate_id":k,"eligible":False,"error":repr(e),"seed":candidate_seed,"score":None})
    eligible = [c for c in candidates if c["eligible"]]
    selected = min(eligible,key=lambda c:(c["score"],c["candidate_id"]))["candidate_id"] if eligible else None
    assert_frozen(public_task)
    return {"method":method,"task_hash":public_task["task_hash"],"candidates":candidates,"selected_id":selected,
        "selection_rule":"minimum_normalized_dual_pose_error_among_model_and_frozen_goal_valid",
        "selection_before_execution":True,"wall_s":time.perf_counter()-wall,"cpu_s":time.process_time()-cpu,
        "calls":{k:robot.calls[k]-before[k] for k in before},"full_path_cost_evaluations":cost.calls,
        "public_conditions_hash":digest(public_task["condition"]),"private_teacher_fields_accessed":False}


def benchmark(run_dir,rows,budget,config,robot,initial,collision,ledger):
    from .pipeline import acceptance
    run_dir = Path(run_dir)
    taskfile = run_dir/"benchmark_tasks.json"
    if taskfile.exists():
        tasks = read_json(taskfile)
    else:
        # Freeze before invoking any tested method, independent of successes.
        audit_ids = set(sum([read_json(run_dir/"audit_selection.json")[g] for g in ("random","risk")],[]))
        audit_families = set(read_json(run_dir/"audit_selection.json")["family_ids"])
        pool = [r for r in rows if r["source_partition"]=="dev" and r["validation_level"]=="MODEL_VALID" and r["family_id"] not in audit_families]
        rng = np.random.default_rng(20261016)
        families = set()
        tasks = []
        for index in rng.permutation(len(pool)):
            row = pool[index]
            if row["family_id"] in families:
                continue
            families.add(row["family_id"])
            tasks.append({"task":row["task"],"source_family":row["family_id"],"source_kind":"MODEL_DERIVED"})
            if len(tasks)>=budget["evaluation_tasks_max"]:
                break
        atomic_json(taskfile,tasks)
    results = []
    profile = acceptance(config)
    for task_index,entry in enumerate(tasks):
        task = entry["task"]
        for method in METHODS:
            identity = f"benchmark_{task_index:03d}_{method}"
            out = run_dir/"benchmark"/identity
            if identity in ledger.data["completed"]:
                if (out/"planning.json").exists():
                    results.append(read_json(out/"planning.json"))
                continue
            ledger.begin(identity,"benchmark",86400,budget["benchmark_selected_execution_max"])
            before = robot.calls.copy()
            try:
                result = plan(task,run_dir/"learning",{"candidates":5,"ddim_steps":16},robot,initial,collision,profile,method,20261017+task_index*101)
                atomic_json(out/"planning.json",result) # Frozen selection, before physical use.
                selected = result["selected_id"]
                if selected is not None:
                    ref = Reference(np.asarray(result["candidates"][selected]["Z"]),np.asarray(task["condition"]["initial_features"]),task["condition"]["duration"])
                    execution = audit_execution(task,profile,robot,initial,ref,out/"execution")
                    status = execution["level"]
                else:
                    status = "NO_MODEL_VALID_FROZEN_GOAL_CANDIDATE"
                ledger.finish({"status":status,"method":method,"task":task_index},{k:robot.calls[k]-before[k] for k in before})
                results.append(result)
            except Exception as e:
                ledger.finish({"status":"ERROR","reason":repr(e),"method":method,"task":task_index},{k:robot.calls[k]-before[k] for k in before})
            print(f"benchmark {task_index} {method}: {ledger.data['completed'][identity]['status']}",flush=True)
    summary = {"status":"FINISHED" if len(tasks)==budget["evaluation_tasks_max"] else "INSUFFICIENT_DEV_TASKS",
        "predefined_task_denominator":budget["evaluation_tasks_max"],"frozen_tasks":len(tasks),"methods":{},
        "benefit":"NOT_ESTABLISHED","guidance_backend":"SUBSPACE_FD_DEV_ONLY","final_TEST":"NOT_RUN"}
    for method in METHODS:
        terminal = [v for k,v in ledger.data["completed"].items() if k.startswith("benchmark_") and v.get("method")==method]
        summary["methods"][method] = {"denominator":budget["evaluation_tasks_max"],"completed":len(terminal),
            "execution_verified":sum(v["status"]=="EXECUTION_VERIFIED" for v in terminal),
            "no_candidate":sum(v["status"]=="NO_MODEL_VALID_FROZEN_GOAL_CANDIDATE" for v in terminal),
            "failed_or_error":sum(v["status"] in ("ERROR","EXECUTION_FAILED") for v in terminal)}
    atomic_json(run_dir/"benchmark.json",summary)
    return summary
