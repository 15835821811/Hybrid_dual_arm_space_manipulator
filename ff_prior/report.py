"""Evidence-derived status and costs, never a blanket SUCCESS flag."""
from pathlib import Path
import math
from .storage import read_json,atomic_json


def wilson(success,count):
    if not count:
        return None
    z = 1.959963984540054
    p = success/count
    mid = (p+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.,mid-half),min(1.,mid+half)]


def status(run_dir):
    run_dir = Path(run_dir)
    ledger = read_json(run_dir/"ledger.json")
    summary = {"scope":"DEVELOPMENT_ONLY","partitions":{},"physical_audits":{},
        "resource_wall_s":ledger["wall_s"],"resource_cpu_s":ledger["cpu_s"],"calls":ledger["calls"],
        "active_work":ledger["inflight"],"training":{},"benchmark":{},"SCALEUP_ELIGIBLE":False}
    summary["profile"]=ledger["identity"]["profile"]
    summary["unresolved_call_accounting"]=ledger.get("unresolved_call_accounting",[])
    summary["counter_definitions"]={"fk":"explicit scratch mj_kinematics calls",
        "forward":"explicit mj_forward calls, including tracker/replay/native verification",
        "mass_matrix":"explicit full mass-matrix reads","physics_steps":"mj_step calls, execution plus independent replay",
        "internal_engine_calls":"not double-counted as Python API calls"}
    for stage in ("main","dev"):
        values = [r for r in ledger["completed"].values() if r["stage"]==stage]
        valid = sum(v["status"]=="MODEL_VALID" for v in values)
        attempted = ledger["attempts"].get(stage,0)
        summary["partitions"][stage] = {"attempted":attempted,"model_valid":valid,"rejected_or_error":len(values)-valid,
            "wall_s_per_model_valid":ledger["wall_s"].get(stage,0)/valid if valid else None}
    for group in ("random","risk"):
        values = [r for r in ledger["completed"].values() if r["stage"]=="physical_audit" and r.get("group")==group]
        verified = sum(v["status"]=="EXECUTION_VERIFIED" for v in values)
        reports = [read_json(p) for p in (run_dir/"audits").glob("*/report.json")]
        selection = read_json(run_dir/"audit_selection.json") if (run_dir/"audit_selection.json").exists() else {group:[]}
        count = len(selection[group])
        summary["physical_audits"][group] = {"frozen_count":count,"finished":len(values),"execution_verified":verified,
            "failed_or_error":sum(v["status"] in ("ERROR","EXECUTION_FAILED") for v in values),
            "conversion":verified/count if count else None,"wilson95":wilson(verified,count),
            "unbiased_population_estimate":group=="random"}
    reports = [read_json(p) for p in (run_dir/"audits").glob("*/report.json")]
    summary["physical_execution_count"] = len(reports)
    summary["independent_replay_count"] = sum("replay" in r for r in reports)
    summary["independent_replay_pass"] = sum(r.get("replay",{}).get("passed") is True for r in reports)
    experts = sum(r["level"]=="EXECUTION_VERIFIED" for r in reports)
    summary["wall_s_per_execution_verified"] = ledger["wall_s"].get("physical_audit",0)/experts if experts else None
    if (run_dir/"learning/training.json").exists():
        training = read_json(run_dir/"learning/training.json")
        summary["training"] = {k:v for k,v in training.items() if k not in ("history","metadata")}
    if (run_dir/"benchmark.json").exists():
        summary["benchmark"] = read_json(run_dir/"benchmark.json")
    summary["OFFLINE_PIPELINE_STATUS"] = "GENERATION_EVIDENCE_AVAILABLE" if summary["partitions"]["main"]["attempted"] else "NOT_RUN"
    summary["MODEL_PRIOR_STATUS"] = "MODEL_VALID_SAMPLES_AVAILABLE" if summary["partitions"]["main"]["model_valid"] else "BLOCKED"
    summary["PHYSICAL_AUDIT_STATUS"] = "EXECUTION_EVIDENCE_AVAILABLE" if reports else "NOT_RUN"
    summary["TRAINING_STATUS"] = summary["training"].get("status","NOT_RUN")
    summary["GUIDED_SAMPLING_STATUS"] = "DEV_EVIDENCE_AVAILABLE" if summary["benchmark"] else "NOT_RUN"
    summary["BENCHMARK_STATUS"] = summary["benchmark"].get("benefit","NOT_RUN")
    summary["scaleup_reasons"] = ["pilot explicitly NOT_RUN","SUBSPACE_FD_DEV_ONLY is not production eligible",
        "inherited physical tolerance profile is developmental"]
    summary["storage_bytes"] = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    return summary


def report(run_dir):
    run_dir = Path(run_dir)
    result = status(run_dir)
    atomic_json(run_dir/"summary.json",result)
    lines = ["# V7.1 measured smoke status", "", "Scope: DEVELOPMENT_ONLY. Clearance: user-authorized nonpenetration (>= 0 m).", ""]
    for key in ("OFFLINE_PIPELINE_STATUS","MODEL_PRIOR_STATUS","PHYSICAL_AUDIT_STATUS","TRAINING_STATUS","GUIDED_SAMPLING_STATUS","BENCHMARK_STATUS","SCALEUP_ELIGIBLE"):
        lines.append(f"- {key}: {result[key]}")
    lines += ["", "## Actual counts", "", "| Partition | Attempted | MODEL_VALID | Rejected/error |", "|---|---:|---:|---:|"]
    for k,v in result["partitions"].items():
        lines.append(f"| {k} | {v['attempted']} | {v['model_valid']} | {v['rejected_or_error']} |")
    lines += ["",f"Physical executions: {result['physical_execution_count']}; independent replays: {result['independent_replay_count']}; replay passes: {result['independent_replay_pass']}.",
        "", "Random and risk-directed evidence is reported separately in summary.json. Unexecuted references remain MODEL_VALID.",
        "", "Pilot, study, strict multimodal and final TEST: NOT_RUN. Dynamic obstacles, arbitrary initial states, continuous-time certification and hardware: NOT_VALIDATED.",
        "", "No equal-quality expert speedup is claimed. MODEL_VALID and EXECUTION_VERIFIED costs are separate."]
    (run_dir/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
