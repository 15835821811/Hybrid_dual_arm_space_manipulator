"""Explicit, gated held-out study and strict-pool verification; never automatic."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .storage import read_json,atomic_json,digest,RunLock,Ledger,file_digest
from .contracts import assert_frozen,strict_pool
from .spline import from_arrays
from .model_gates import evaluate_reference
from .coupled_predictor import pose_errors
from .physical_audit import audit_execution


def verify_strict_pool(pool,archive_root,output,robot,initial,collision,profile,remaining_steps,deadline=None):
    """Validate every member of an externally frozen root-attempt pool.

    A pool is an explicit inverse-planning input, not five forward-generated
    goals. At most 40 root attempts are accepted; retimings share a root.
    """
    task=pool["task"]
    assert_frozen(task)
    candidates=pool["candidates"]
    if len({c["root_id"] for c in candidates})>40 or len(candidates)>120 or len({c["sample_id"] for c in candidates})!=len(candidates) or pool.get("frozen_before_verification") is not True:
        raise ValueError("strict root-attempt/frozen-pool contract")
    validated, references = [],{}
    for c in candidates:
        if deadline is not None and time.time()>=deadline:
            raise RuntimeError("BUDGET_EXHAUSTED")
        with np.load(Path(archive_root)/c["reference_npz"]) as data:
            reference=from_arrays(data)
        if c.get("reference_hash")!=reference.reference_hash:
            raise ValueError("strict frozen reference identity mismatch")
        prediction,report=evaluate_reference(robot,initial,reference,collision,profile,deadline)
        p,r=(pose_errors(prediction["ee"][-1],task["condition"]["goals"]) if prediction is not None else (np.array([1e30]),np.array([1e30])))
        eligible=report["validation_level"]=="MODEL_VALID" and np.all(p<=profile.position_error_m) and np.all(r<=profile.rotation_error_rad)
        validated.append({"sample_id":c["sample_id"],"root_id":c["root_id"],"task_hash":task["task_hash"],
            "validation_level":"MODEL_VALID" if eligible else "REJECTED","path":reference.evaluate(np.linspace(0,reference.duration,128)),
            "report":report,"frozen_goal_position_error_m":p,"frozen_goal_rotation_error_rad":r})
        references[c["sample_id"]]=reference
    terminal={}
    quota=strict_pool(validated,terminal)
    steps=0
    if quota=="INCOMPLETE_POOL_VERIFICATION":
        for c in validated:
            if c["validation_level"]!="MODEL_VALID":
                terminal[c["sample_id"]]="EXECUTION_FAILED"
                continue
            ref=references[c["sample_id"]]
            required=2*int(round((ref.duration+ref.hold)/robot.model.opt.timestep))
            if steps+required>remaining_steps:
                terminal[c["sample_id"]]="BUDGET_EXHAUSTED"
                continue
            if deadline is not None and time.time()>=deadline:
                terminal[c["sample_id"]]="BUDGET_EXHAUSTED"
                continue
            report=audit_execution(task,profile,robot,initial,ref,Path(output)/c["sample_id"],deadline)
            steps+=required
            terminal[c["sample_id"]]=report["level"]
    result={"status":strict_pool(validated,terminal),"quota_before_execution":quota,"candidates":validated,
        "terminal_results":terminal,"physics_steps_including_replay":steps,"five_physical_successes_required":False,
        "all_frozen_candidates_accounted":len(validated)==len(candidates),"task_hash":task["task_hash"]}
    atomic_json(Path(output)/"strict_pool.json",result)
    return result


def run_study(config,archive,budget_path,checkpoint,output,dry_run=False,resume=False):
    from .pipeline import audit,acceptance,source_manifest
    from .benchmark import plan,METHODS
    archive=Path(archive)
    bank=read_json(archive)
    budget=read_json(budget_path)
    required={"wall_seconds","physics_steps_including_replay","task_count","strict_task_count","candidate_count","ddim_steps"}
    if set(budget)!=required or budget["task_count"]!=200 or budget["strict_task_count"]!=20 or budget["candidate_count"]!=5:
        raise ValueError("study requires explicit frozen 200 TEST / 20 strict / five-candidate budget")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in budget.values()):
        raise ValueError("invalid study resource budget")
    if len(bank.get("tasks",[]))!=200 or len(bank.get("strict_pools",[]))!=20 or bank.get("split")!="TEST":
        raise ValueError("frozen study archive must contain 200 TEST tasks and 20 strict pools")
    if bank.get("selection_independent_of_tested_methods") is not True or bank.get("frozen_before_planning") is not True:
        raise ValueError("unfrozen or method-selected test archive")
    gate=config.get("explicit_scaleup_evidence")
    if not gate or read_json(gate).get("STUDY_ELIGIBLE") is not True:
        raise ValueError("study requires completed explicit pilot evidence")
    for entry in bank["tasks"]:
        assert_frozen(entry)
    if dry_run:
        return {"status":"DRY_RUN","archive_hash":digest(bank),"budget":budget}
    output=Path(output)
    if (output/"run.json").exists() and not resume:
        raise ValueError("study output exists; use resume")
    with RunLock(output):
        from .scenes import SceneSpec
        baseline,robot,initial,collision=audit(config,SceneSpec.from_dict(bank.get("scene",{})))
        profile=acceptance(config)
        inputs={str((archive.parent/c["reference_npz"]).resolve()):file_digest(archive.parent/c["reference_npz"])
                for pool in bank["strict_pools"] for c in pool["candidates"]}
        identity={"profile":"study","archive_hash":digest(bank),"budget":budget,"robot_hash":initial.model_hash,
            "initial_state_id":initial.state_id,"source_hash":digest(source_manifest()),"config_hash":digest(config),
            "checkpoint_hash":file_digest(Path(checkpoint)/"model.pt"),"strict_inputs":inputs}
        ledger=Ledger(output,identity)
        atomic_json(output/"run.json",{"kind":"study","identity":identity,"config":config,
            "archive":str(archive.resolve()),"budget_path":str(Path(budget_path).resolve()),"checkpoint":str(Path(checkpoint).resolve())})
        atomic_json(output/"frozen_inputs.json",{"archive":bank,"budget":budget,"baseline":baseline,"reference_hashes":inputs})
        deadline=time.time()+max(0,budget["wall_seconds"]-ledger.data["wall_s"].get("study",0))
        max_attempts=200*len(METHODS)+20
        def reserve(key,steps):
            reservations=ledger.data.setdefault("physics_reservations",{})
            if sum(reservations.values())+steps>budget["physics_steps_including_replay"]:
                return False
            reservations[key]=steps
            ledger.save()
            return True
        for index,task in enumerate(bank["tasks"]):
            for method in METHODS:
                identity=f"task_{index:03d}_{method}"
                if identity in ledger.data["completed"] or time.time()>=deadline:
                    continue
                ledger.begin(identity,"study",budget["wall_seconds"],max_attempts)
                before=robot.calls.copy()
                try:
                    result=plan(task,checkpoint,{"candidates":5,"ddim_steps":budget["ddim_steps"],"deadline":deadline},robot,initial,collision,profile,method,20261015+index)
                    atomic_json(output/identity/"planning.json",result)
                    selected=result["selected_id"]
                    status="NO_MODEL_VALID_FROZEN_GOAL_CANDIDATE"
                    if selected is not None:
                        from .spline import Reference
                        ref=Reference(result["candidates"][selected]["Z"],np.asarray(task["condition"]["initial_features"]),task["condition"]["duration"])
                        required_steps=2*int(round((ref.duration+ref.hold)/robot.model.opt.timestep))
                        if reserve(identity,required_steps) and time.time()<deadline:
                            execution=audit_execution(task,profile,robot,initial,ref,output/identity/"execution",deadline)
                            status=execution["level"]
                        else: status="BUDGET_EXHAUSTED"
                    ledger.finish({"status":status,"method":method,"task":index},{k:robot.calls[k]-before[k] for k in before})
                except Exception as e:
                    ledger.finish({"status":"ERROR","reason":repr(e),"method":method,"task":index},{k:robot.calls[k]-before[k] for k in before})
        for index,pool in enumerate(bank["strict_pools"]):
            key=f"strict_{index:03d}"
            if key in ledger.data["completed"] or time.time()>=deadline: continue
            ledger.begin(key,"study",budget["wall_seconds"],max_attempts)
            before=robot.calls.copy()
            try:
                remaining=budget["physics_steps_including_replay"]-sum(ledger.data.get("physics_reservations",{}).values())
                # A killed pool consumes its frozen worst-case reservation.
                reserved=min(remaining,len(pool["candidates"])*30000)
                reserve(key,reserved)
                result=verify_strict_pool(pool,archive.parent,output/key,robot,initial,collision,profile,reserved,deadline)
                ledger.finish({"status":result["status"]},{k:robot.calls[k]-before[k] for k in before})
            except Exception as e:
                ledger.finish({"status":"ERROR","reason":repr(e)},{k:robot.calls[k]-before[k] for k in before})
        results=[dict(id=f"task_{i:03d}_{m}",**ledger.data["completed"].get(f"task_{i:03d}_{m}",{"status":"BUDGET_EXHAUSTED"})) for i in range(200) for m in METHODS]
        strict=[ledger.data["completed"].get(f"strict_{i:03d}",{"status":"BUDGET_EXHAUSTED"}) for i in range(20)]
        result={"status":"ACCOUNTED","test_task_denominator":200,"strict_task_denominator":20,
            "results":results,"strict":strict,"physics_steps_including_replay":ledger.data["calls"].get("study.physics_steps",0),
            "reserved_physics_steps":sum(ledger.data.get("physics_reservations",{}).values()),
            "wall_s":ledger.data["wall_s"].get("study",0),"benefit":"NOT_ESTABLISHED"}
        atomic_json(output/"study.json",result)
        return result
