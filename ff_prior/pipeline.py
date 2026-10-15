"""Bounded smoke/pilot orchestration with separate generation and audit budgets."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import subprocess
import time
import sys
import numpy as np
import yaml
from .storage import atomic_json,atomic_npz,digest,read_json,Ledger,RunLock
from .robot_adapter import RobotAdapter,ROOT
from .initial_state import InitialState
from .geometry import CollisionBackend
from .contracts import AcceptanceConfig,frozen_task
from .proposal_generator import propose
from .model_gates import evaluate_reference
from .dataset import split_manifest,export_dataset
from .spline import from_arrays


def config_from(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if config["reference"]["control_points"] != 24 or config["reference"]["latent_tokens"] != 19:
        raise ValueError("representation mismatch")
    if config["robot"]["planner_dof"] != 17 or config["robot"]["robot_actuator_dof"] != 67:
        raise ValueError("robot representation mismatch")
    if config["scope"]["initial_momentum_support"] != "audited_zero_momentum_only":
        raise ValueError("unsupported momentum profile")
    for section,key,value in (("reference","duration_candidates_s",[10.,15.,20.,25.,29.5]),
        ("reference","training_view_points",128),("reference","hold_duration_s",.5),
        ("reference","deadline_s",30.),("physical_audit","physics_dt_s",.002),
        ("training","hidden_dim",128),("training","layers",4),("training","attention_heads",4),
        ("sampling","candidate_count",5),("prior_generation","proposals_per_family",5)):
        if config[section][key]!=value:
            raise ValueError(f"unsupported frozen implementation setting {section}.{key}; create an audited version first")
    if config["prior_generation"]["external_obstacles"] or config["split"]["historical_import_enabled"]:
        raise ValueError("default generator supports static empty environment and no historical import")
    if config["validation"].get("reference_torque_required",False):
        raise ValueError("BLOCKED: required reference torque feasibility is NOT_IMPLEMENTED")
    return config


def acceptance(config):
    v = config["validation"]["inherited_uncalibrated_proposal"]
    return AcceptanceConfig(position_error_m=v["terminal_position_tolerance_m_each_arm"],
        rotation_error_rad=np.deg2rad(v["terminal_rotation_tolerance_deg_each_arm"]),
        planning_clearance_m=v["planning_clearance_m"],actual_clearance_m=v["actual_clearance_m"])


def source_manifest():
    import hashlib
    paths = [p for p in (ROOT/"ff_prior").rglob("*.py") if not {"tests","evidence"}&set(p.relative_to(ROOT/"ff_prior").parts)] + [ROOT/"model_test/robot_model_spec_v5.py",
        ROOT/"model_test/angle_convention.py",ROOT/"v6_lite/continuum_model_spec.py"]
    return {str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def audit(config,scene=None):
    from .scenes import SceneSpec
    scene=SceneSpec() if scene is None else scene
    robot = RobotAdapter(scene=scene)
    initial = InitialState.capture(robot,robot.new_data())
    collision = CollisionBackend(robot)
    d = initial.reset(robot)
    result = {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
        "python":sys.executable,"versions":{p:importlib.metadata.version(p) for p in ("numpy","scipy","mujoco","torch","pytest","pyyaml")},
        "identity":robot.identity(),"initial_state_id":initial.state_id,"initial_velocity_max":float(np.max(abs(d.qvel))),
        "initial_momentum_max":float(np.max(abs((robot.mass(d)@d.qvel)[robot.base_dofs]))),
        "mapping_left_inverse_error":float(np.max(abs(robot.Bplus@robot.B-np.eye(17)))),
        "initial_geometry":collision.scan(d,acceptance(config).planning_clearance_m),
        "pair_count":len(collision.pairs),"exclusion_count":len(collision.exclusions),
        "acceptance":asdict(acceptance(config)),"scene_spec":scene.to_dict(),"source_hashes":source_manifest(),
        "scope":"DEVELOPMENT_ONLY","initial_state_mask":int(__import__('mujoco').mjtState.mjSTATE_INTEGRATION)}
    return result,robot,initial,collision


def generation_partition(config,budget,run_dir,partition,robot,initial,collision,ledger):
    count = budget["main_proposals"] if partition == "main" else budget["independent_dev_proposals_max"]
    wall_budget = budget["main_generation_wall_budget_s"] if partition == "main" else budget["dev_generation_wall_budget_s"]
    profile = acceptance(config)
    for index in range(count):
        sample_id = f"{partition}_{index:06d}"
        try:
            if not ledger.begin(sample_id,partition,wall_budget,count):
                continue
        except RuntimeError as e:
            if str(e) == "BUDGET_EXHAUSTED":
                break
            raise
        before = robot.calls.copy()
        start,cpu = time.perf_counter(),time.process_time()
        family,variant = index//5,index%5
        family_id = f"{partition}_family_{family:06d}"
        seed = config["prior_generation"]["master_seed"]+(0 if partition == "main" else 10000019)
        ref,provenance = propose(robot,family,variant,seed)
        record = {"sample_id":sample_id,"family_id":family_id,"root_id":sample_id,"parent_id":None,
            "source_partition":partition,"scene_family_id":None,"provenance":provenance,
            "producer_commit":ledger.identity["head"],"source_hash":ledger.identity["source_hash"],
            "initial_state_id":initial.state_id,"robot_hash":initial.model_hash,"mapping_hash":digest(robot.B),
            "scene_hash":initial.model_hash,"pair_policy_hash":collision.policy_hash,"config_hash":digest(config),
            "goal_source":"coupled_forward_generated","execution_status":"NOT_RUN","publication":"DRAFT"}
        try:
            if ref is None:
                prediction,report = None,{"validation_level":"REJECTED","reason":"DURATION_SET_EXHAUSTED"}
            else:
                remaining = wall_budget-ledger.data["wall_s"].get(partition,0)
                prediction,report = evaluate_reference(robot,initial,ref,collision,profile,time.time()+remaining)
            record["validation"] = report
            record["validation_level"] = report["validation_level"]
            valid = report["validation_level"] == "MODEL_VALID"
            record.update(data_role="FF_MODEL_PRIOR",
                validation_profile_id=profile.profile_id,training_eligible=valid and partition=="main")
            if valid:
                condition = {"initial_features":robot.spec.planner_zero,"goals":prediction["ee"][-1],
                             "duration":ref.duration,"pose_mask":np.ones((2,6))}
                record["condition"] = condition
                record["path_descriptor"] = ref.evaluate(np.linspace(0,ref.duration,16))
                record["reference_hash"] = ref.reference_hash
                record["arrays"] = f"arrays/{sample_id}.npz"
                record["task"] = frozen_task(condition,{"initial_state_id":initial.state_id,"robot_hash":initial.model_hash,
                    "pair_policy_hash":collision.policy_hash,"acceptance":asdict(profile)})
                arrays = ref.arrays()
                arrays.update({"prediction_"+k:v for k,v in prediction.items() if isinstance(v,np.ndarray)})
                atomic_npz(run_dir/record["arrays"],**arrays)
                record["storage_bytes"] = (run_dir/record["arrays"]).stat().st_size
            record["resources"] = {"wall_s":time.perf_counter()-start,"cpu_s":time.process_time()-cpu,
                "calls":{k:robot.calls[k]-before[k] for k in before}}
            if any(record["resources"]["calls"][k] for k in ("rrt","sqp","torque_rollout","physics_steps")):
                raise RuntimeError("forbidden generator execution/search call")
            atomic_json(run_dir/"records"/(sample_id+".json"),record)
            ledger.finish({"status":record["validation_level"],"reason":report.get("reason")},record["resources"]["calls"])
        except Exception as e:
            ledger.finish({"status":"ERROR","reason":f"{type(e).__name__}: {e}"},
                          {k:robot.calls[k]-before[k] for k in before})
            if str(e) == "BUDGET_EXHAUSTED":
                break
        if (index+1)%25 == 0:
            print(f"{partition}: {index+1}/{count} proposals accounted",flush=True)


def freeze_dataset(run_dir,ledger):
    from .storage import file_digest
    publication=run_dir/"publication.json"
    if publication.exists():
        for name,expected in read_json(publication)["files"].items():
            if file_digest(run_dir/name)!=expected:
                raise ValueError("immutable published record/array changed: "+name)
    rows = ledger.index()
    valid = [r for r in rows if r["validation_level"] == "MODEL_VALID"]
    split = split_manifest(valid)
    path = run_dir/"splits.json"
    if path.exists() and digest(read_json(path)) != digest(split):
        raise ValueError("split manifest already frozen with different data")
    atomic_json(path,split)
    for row in rows:
        assigned = split["records"].get(row["sample_id"],{"split":"REJECTED"})["split"]
        if row["publication"] == "DRAFT":
            row.update(publication="IMMUTABLE",split=assigned,
                training_eligible=row["validation_level"]=="MODEL_VALID" and assigned=="TRAIN")
            atomic_json(run_dir/"records"/(row["sample_id"]+".json"),row)
        elif row["split"] != assigned:
            raise ValueError("published split mutation")
    dataset = export_dataset(rows,split)
    atomic_json(run_dir/"dataset.json",dataset)
    files={str(p.relative_to(run_dir)).replace("\\","/"):file_digest(p)
           for p in list((run_dir/"records").glob("*.json"))+list((run_dir/"arrays").glob("*.npz"))}
    atomic_json(publication,{"files":files,"split_hash":digest(split),"immutable_after_publication":True})
    ledger.index()
    return rows,dataset


def select_audits(run_dir,rows,budget,seed):
    path = run_dir/"audit_selection.json"
    if path.exists():
        return read_json(path)
    dev = [r for r in rows if r["source_partition"]=="dev" and r["validation_level"]=="MODEL_VALID"]
    rng = np.random.default_rng(seed)
    families = sorted({r["family_id"] for r in dev})
    chosen_families = set(np.asarray(families)[rng.permutation(len(families))[:budget["random_audit_samples_max"]]])
    random = []
    for family in sorted(chosen_families):
        members = [r for r in dev if r["family_id"]==family]
        random.append(members[int(rng.integers(len(members)))]["sample_id"])
    remaining = sorted((r for r in dev if r["family_id"] not in chosen_families),
        key=lambda r:(r["validation"]["geometry"]["minimum_clearance_m"],r["sample_id"]))
    risk = []
    for row in remaining:
        if row["family_id"] not in chosen_families and len(risk)<budget["risk_audit_samples_max"]:
            risk.append(row["sample_id"])
            chosen_families.add(row["family_id"])
    selection = {"random":random,"risk":risk,"seed":seed,"frozen_before_execution":True,
                 "replacement_after_failure":False,"source":"independent_DEV_generation",
                 "family_ids":sorted(chosen_families),"one_sample_per_family":True}
    if len(random)+len(risk)>budget["actual_audit_samples_max"]:
        raise ValueError("audit selection exceeds frozen budget")
    atomic_json(path,selection)
    return selection


def generate_prior(config,budget,output,profile_name="smoke",stages="all",resume=False):
    integer_fields=("main_proposals","main_families","independent_dev_proposals_max","actual_audit_samples_max",
        "random_audit_samples_max","risk_audit_samples_max","training_updates_max","evaluation_tasks_max","benchmark_selected_execution_max")
    for name in integer_fields:
        if isinstance(budget[name],bool) or not isinstance(budget[name],int) or budget[name]<0:
            raise ValueError("invalid budget "+name)
    if budget["main_proposals"]!=budget["main_families"]*5 or budget["random_audit_samples_max"]+budget["risk_audit_samples_max"]>budget["actual_audit_samples_max"]:
        raise ValueError("inconsistent family/audit budgets")
    if budget["evaluation_tasks_max"]*3>budget["benchmark_selected_execution_max"]:
        raise ValueError("benchmark budget cannot account for all predefined methods/tasks")
    run_dir = Path(output)
    if run_dir.exists() and (run_dir/"run.json").exists() and not resume:
        raise ValueError("output exists: use resume")
    if profile_name != "smoke":
        gate = config.get("explicit_scaleup_evidence")
        if not gate or not Path(gate).is_file() or read_json(gate).get("PILOT_ELIGIBLE") is not True:
            raise ValueError("BLOCKED: explicit prior-profile evidence gate missing")
    with RunLock(run_dir):
        report,robot,initial,collision = audit(config)
        identity = {"head":report["head"],"robot_hash":initial.model_hash,"initial_state_id":initial.state_id,
            "source_hash":digest(report["source_hashes"]),"config_hash":digest(config),"budget":budget,"profile":profile_name,
            "environment":report["versions"]}
        if resume and (run_dir/"run.json").exists():
            # A commit of the same source bytes is not a new experiment.
            identity["head"] = read_json(run_dir/"run.json")["identity"]["head"]
        ledger = Ledger(run_dir,identity)
        if resume and (run_dir/"initial_state.npz").exists():
            with np.load(run_dir/"initial_state.npz") as state:
                if not np.array_equal(state["values"],initial.values):
                    raise ValueError("frozen initial-state file changed")
        if resume and (run_dir/"pair_policy.json").exists() and digest(read_json(run_dir/"pair_policy.json"))!=collision.policy_hash:
            raise ValueError("frozen pair policy file changed")
        run = {"identity":identity,"config":config,"scope":"DEVELOPMENT_ONLY","source_hashes":report["source_hashes"]}
        atomic_json(run_dir/"run.json",run)
        if not (run_dir/"baseline_audit.json").exists():
            atomic_json(run_dir/"baseline_audit.json",report)
        atomic_json(run_dir/"pair_policy.json",collision.policy)
        atomic_json(run_dir/"scene.json",robot.scene.to_dict())
        atomic_npz(run_dir/"initial_state.npz",values=initial.values)
        for partition in ("main","dev"):
            generation_partition(config,budget,run_dir,partition,robot,initial,collision,ledger)
        if ledger.begin("dataset_freeze","dataset",86400,1):
            rows,dataset = freeze_dataset(run_dir,ledger)
            ledger.finish({"status":dataset["status"]})
        else:
            rows,dataset = freeze_dataset(run_dir,ledger)
        if stages == "generate":
            return {"status":"GENERATION_COMPLETE","dataset":dataset}
        selection = select_audits(run_dir,rows,budget,config["prior_generation"]["master_seed"])
        from .physical_audit import audit_execution
        by_id = {r["sample_id"]:r for r in rows}
        for group in ("random","risk"):
            for sample in selection[group]:
                work_id = "audit_"+sample
                if not ledger.begin(work_id,"physical_audit",86400,budget["actual_audit_samples_max"]):
                    continue
                before = robot.calls.copy()
                try:
                    row = by_id[sample]
                    with np.load(run_dir/row["arrays"]) as a:
                        ref = from_arrays(a)
                    result = audit_execution(row["task"],acceptance(config),robot,initial,ref,run_dir/"audits"/sample)
                    ledger.finish({"status":result["level"],"group":group}, {k:robot.calls[k]-before[k] for k in before})
                except Exception as e:
                    ledger.finish({"status":"ERROR","group":group,"reason":repr(e)}, {k:robot.calls[k]-before[k] for k in before})
                print(f"physical audit {group}: {sample} {ledger.data['completed'][work_id]['status']}",flush=True)
        if stages == "audit":
            return {"status":"PHYSICAL_AUDIT_COMPLETE"}
        from .diffusion import train
        if budget["training_updates_max"]:
            training = train(run_dir,rows,dataset,initial.model_hash,initial.state_id,budget["training_updates_max"])
        else:
            training={"status":"NOT_RUN","reason":"zero update budget"}
            atomic_json(run_dir/"learning/training.json",training)
        if training["status"] == "PASS":
            from .benchmark import benchmark
            benchmark(run_dir,rows,budget,config,robot,initial,collision,ledger)
        else:
            atomic_json(run_dir/"benchmark.json",{"status":"BLOCKED","reason":"training unavailable","benefit":"NOT_RUN"})
        ledger.index()
        return {"status":"RUN_FINISHED","training":training["status"]}
