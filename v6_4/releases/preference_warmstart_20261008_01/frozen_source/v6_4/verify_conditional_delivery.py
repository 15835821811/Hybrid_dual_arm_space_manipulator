"""Read-only, zero-physics completion audit of the finite B.3 negative pilot.

Requires terminal six-slot evidence plus the collected report and both figures.
It never imports a runner, MuJoCo or torch, and never runs a model or geometry.
Only JSON/CSV/NPZ reads, SHA256 checks and saved-array arithmetic are used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SCHEMA="v64_b3_negative_delivery_checks_v1"
NEGATIVE="ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION"
CHECKPOINT="9ace53f482253216cea7baabed7479d84f37a9e80e71231219c7e2f9d3c94ca1"
GATES=("task_requirements","execution_contract","independent_interval","native_geometry","reference_binding")
STATE_FIELDS=("research_delivery_complete","route_value_identifiable","training_executed",
              "condition_response_observed","conditional_value_supported_in_pilot","advantage_over_retrieval",
              "independent_test_task_success_by_method","deployment")


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def close(a,b,message,tolerance=1e-12):
    require(a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
            and abs(float(a)-float(b))<=tolerance,message)


class Audit:
    def __init__(self,source,repository):
        self.source=Path(source).resolve();self.repository=Path(repository).resolve()
        self.bindings={};self._cache={};self.checks={}

    def bind(self,path,expected=None):
        p=Path(path)
        if not p.is_absolute():p=self.source/p
        p=p.resolve()
        require(p.is_file(),"required evidence is absent: "+str(p))
        stamp=(p.stat().st_size,p.stat().st_mtime_ns)
        cached=self._cache.get(str(p))
        digest=cached[1] if cached and cached[0]==stamp else sha(p)
        self._cache[str(p)]=(stamp,digest)
        require(expected is None or digest==expected,"evidence hash mismatch: "+str(p))
        record={"path":str(p),"sha256":digest,"bytes":stamp[0]}
        require(str(p) not in self.bindings or self.bindings[str(p)]==record,"evidence changed during audit: "+str(p))
        self.bindings[str(p)]=record
        return p

    def read(self,path,expected=None):
        return json.loads(self.bind(path,expected).read_text(encoding="utf-8-sig"))

    def local_manifest(self,directory):
        directory=Path(directory).resolve()
        manifest=self.read(directory/"manifest.json")
        require(isinstance(manifest,dict),"evidence manifest must be a digest map")
        for name,digest in manifest.items():
            p=(directory/name).resolve()
            require(directory in p.parents,"manifest member escapes evidence directory")
            self.bind(p,digest)
        return manifest

    def source_guard(self,plan,identity):
        for relative,digest in identity["source_sha256"].items():self.bind(self.repository/relative,digest)
        for path,digest in identity["protected_artifacts"].items():self.bind(path,digest)
        require(len(plan["frozen_task_artifacts"])==126,"expected 126 predeclared task/candidate artifacts")
        inputs=self.read("route_pair_inputs.json",plan["route_pair_inputs_sha256"])
        require(inputs["artifacts"]==plan["frozen_task_artifacts"],"plan/input artifact ledger differs")
        for r in plan["frozen_task_artifacts"]:self.bind(r["path"],r["sha256"])
        require(len(inputs["pair_records"])==7 and len(inputs["pilot_task_ids"])==2,"frozen mother/pilot count differs")
        require({k:len(v) for k,v in inputs["splits"].items()}=={"train":6,"val":2,"test":4},"formal frozen split counts differ")
        require(len(plan["pilot_slots"])==6 and len({r["slot_id"] for r in plan["pilot_slots"]})==6,"six unique pilot slots required")
        require(plan["invariants"]["coefficient_bound_m"]==.020 and plan["invariants"]["deployment"]=="NOT_MET","frozen bound/deployment changed")
        self.checks["frozen_source_guard"]={"passed":True,"source_files":len(identity["source_sha256"]),
            "protected_artifacts":len(identity["protected_artifacts"]),"fixed_task_candidate_inputs":126}
        return inputs

    def probe(self,plan):
        directory=self.source/"old_checkpoint_probe"
        require(not list(directory.rglob("attempt_result.json")) and not (directory/"actual").exists(),"P0 contains forbidden actual execution artifacts")
        manifest=self.read(directory/"manifest.json")
        for record in manifest["files"]:self.bind(record["path"],record["sha256"])
        self.bind(manifest["plan"]["path"],manifest["plan"]["sha256"])
        report=self.read(directory/"report.json")
        require(report["status"]=="COMPLETED" and report["ddim_calls"]==32,"P0 is not terminal at exactly 32 calls")
        require(report["checkpoint_sha256"]==CHECKPOINT and report["selected_update"]==250 and report["selected_exposures"]==8000,"P0 selected weight identity differs")
        for k in ("physics_steps","actual_attempts","private_preview_steps","independent_physics_replay_steps","optimizer_updates"):
            require(report[k]==0,"P0 nonzero forbidden work: "+k)
        require(report["correct_adaptation_claim"]=="NOT_EVALUATED_NO_ROUTE_PREFERENCE_LABELS" and report["conditional_value_supported"]=="NOT_ESTABLISHED_BY_P0","P0 overclaims conditional value")
        config=self.read(directory/"probe_config.json")
        require(config["protocol"]==plan["old_checkpoint_probe"],"P0 configuration differs from frozen plan")
        identity=self.read(directory/"source_identity.json")
        self.bind(identity["checkpoint"]["path"],CHECKPOINT)
        for record in identity["sources"].values():self.bind(record["path"],record["sha256"])
        for entry in identity["producer_source_bindings"]:
            old=self.bind(entry["old"]["path"],entry["old"]["sha256"])
            imported=self.bind(entry["imported"]["path"],entry["imported"]["sha256"])
            require(old.read_bytes().replace(b"\r\n",b"\n")==imported.read_bytes().replace(b"\r\n",b"\n"),"P0 imported network/scaler algorithm differs from old producer")
        normalizer=self.read(identity["sources"]["training/residual_normalizer.json"]["path"])
        old_definitions=self.read(identity["sources"]["definitions.json"]["path"])
        with np.load(self.bind(directory/"raw_outputs.npz"),allow_pickle=False) as saved:
            arrays={k:saved[k] for k in saved.files}
        noise,z,latent=arrays["initial_noise"],arrays["raw_z_m"],arrays["normalized_latents"]
        require(noise.shape==(4,4,12) and z.shape==latent.shape==(4,4,2,6,2),"P0 raw array dimensions differ")
        require(np.isfinite(z).all() and np.isfinite(latent).all(),"P0 raw output is nonfinite")
        task_order=config["task_order"]
        require(task_order==[t for pair in plan["old_checkpoint_probe"]["task_pairs"] for t in pair],"P0 task order differs")
        for ti,tid in enumerate(task_order):
            mask=np.asarray(old_definitions[tid]["interval_mask"],dtype=bool)
            for ni,seed in enumerate(plan["old_checkpoint_probe"]["noise_seeds"]):
                expected=np.random.default_rng(seed).standard_normal(12).astype(np.float32)
                expected[~np.repeat(mask,2)]=0.
                require(np.array_equal(noise[ti,ni],expected),"P0 initial noise differs from declared old seed")
            expected=np.zeros_like(z[ti]);expected[:,:,mask]=latent[ti][:,:,mask]*np.asarray(normalizer["std_m"])+np.asarray(normalizer["mean_m"])
            require(np.array_equal(expected,z[ti]),"P0 raw output differs from inverse TRAIN scaler")
        journal=[json.loads(line) for line in self.bind(directory/"ddim_call_ledger.jsonl").read_text().splitlines() if line.strip()]
        require(len(journal)==64,"P0 journal must contain exactly 32 starts and 32 completions")
        records=report["records"]
        require(len(records)==32,"P0 report raw count differs")
        legal=0
        for index,row in enumerate(records):
            ti,remaining=divmod(index,8);ni,ci=divmod(remaining,2);condition=("correct","swap")[ci]
            started,completed=journal[2*index:2*index+2]
            for entry,status in ((started,"STARTED"),(completed,"COMPLETED"),(row,"COMPLETED")):
                require((entry["call_index"],entry["task_id"],entry["noise_index"],entry["condition"],entry["status"])==
                        (index+1,task_order[ti],ni,condition,status),"P0 journal/report slot order differs")
                require(entry["initial_noise_sha256"]==hashlib.sha256(noise[ti,ni].tobytes()).hexdigest(),"P0 paired noise hash differs")
            require(np.array_equal(np.asarray(row["raw_z_m"]),z[ti,ni,ci]),"P0 JSON raw z differs from retained NPZ")
            norms=np.linalg.norm(z[ti,ni,ci],axis=1)
            is_legal=bool(np.all(norms<=.02));legal+=is_legal
            require(row["within_original_20mm_amplitude"]==is_legal and row["output_modified"] is False,"P0 amplitude or repair declaration differs")
            require(row["actual_or_reference_precheck_executed"] is False,"P0 added an actual/precheck")
        comparisons=report["same_noise_between_obstacle_conditions"]
        require(len(comparisons)==16 and len(report["different_noise_within_obstacle_condition"])==48,"P0 comparison counts differ")
        observed=0
        for index,row in enumerate(comparisons):
            ti,ni=divmod(index,4)
            delta=arrays["raw_reference_world_offsets_m"][ti,ni,1]-arrays["raw_reference_world_offsets_m"][ti,ni,0]
            peak=float(np.linalg.norm(delta,axis=1).max());flag=peak>plan["old_checkpoint_probe"]["observed_response_threshold_m"]
            close(peak,row["reference_world_offset_peak_delta_m"],"P0 response peak differs")
            require(row["condition_response_observed"]==flag,"P0 response label differs from frozen numerical threshold")
            observed+=flag
        require(report["same_noise_pairs_observed"]==observed and report["amplitude_legal_raw_outputs"]==legal,"P0 aggregate response/amplitude differs")
        raw_conditions=arrays["raw_conditions"]
        require(raw_conditions.shape==(4,2,878),"P0 condition shape differs")
        wiring=self.read(directory/"condition_wiring.json")
        for ti,entry in enumerate(wiring["tasks"]):
            changed=np.flatnonzero(raw_conditions[ti,0]!=raw_conditions[ti,1]).tolist()
            require(changed==entry["changed_feature_indices"] and entry["changed_feature_names"]==["obstacle.1.center.1:m"],"P0 changed a nondeclared condition field")
            require(entry["original_definition_unchanged"] is True and entry["all_non_obstacle_features_bitwise_equal"] is True,"P0 task/definition invariants differ")
        self.checks["P0_bounded_diagnostic"]={"passed":True,"started":32,"completed":32,"actual":0,"physics_steps":0,"same_noise_responses":observed,"amplitude_legal_raw_outputs":legal}
        return report

    def pilot(self,plan,inputs):
        qualities=self.read("pilot/quality_records.json")
        decision=self.read("pilot/decision.json")
        require(len(qualities)==6 and len({q["slot_id"] for q in qualities})==6,"all six terminal quality records required")
        require({p.name for p in (self.source/"pilot/attempts").iterdir() if p.is_dir()}=={d["slot_id"] for d in plan["pilot_slots"]},"extra/missing pilot attempt directory")
        require({p.parent.resolve() for p in self.source.rglob("attempt_result.json")}==
            {(self.source/"pilot/attempts"/d["slot_id"]).resolve() for d in plan["pilot_slots"]},"unbudgeted extra or missing new actual slot evidence")
        spec=importlib.util.spec_from_file_location("verified_b3_quality",self.repository/"v6_4/route_quality_dataset.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        by_slot={q["slot_id"]:q for q in qualities};rows=[];costs=[];attempts=[];fresh_states={}
        for declared in plan["pilot_slots"]:
            slot=declared["slot_id"];directory=self.source/"pilot/attempts"/slot;quality_dir=self.source/"pilot/quality"/slot
            self.local_manifest(quality_dir)
            quality=self.read(quality_dir/"route_quality.json")
            require(quality==by_slot[slot],"aggregate quality differs from per-slot frozen record: "+slot)
            for path,digest in quality["sources"].items():self.bind(path,digest)
            attempt=self.read(directory/"attempt_result.json");cost=self.read(directory/"cost_ledger.json")
            task=self.read(directory/"task.json");reference=self.read(directory/"plan.json")
            fixed=self.read(declared["plan_path"])
            require(reference==fixed,"executed proposal differs from frozen fixed candidate")
            require(attempt["plan_file_sha256"]==sha(directory/"plan.json") and attempt["plan_content_sha256"]==object_sha(reference),"executed plan binding differs")
            require(task==self.read(f"tasks/{declared['task_id']}/task.json"),"actual task differs from frozen pair task")
            require(attempt["task_sha256"]==quality["task_sha256"]==object_sha(task)==reference["definition"]["task_sha256"],"actual task/reference/quality binding differs")
            require((attempt["slot_id"],attempt["task_id"],quality["candidate_name"],quality["route_intervals_s"])==
                    (slot,declared["task_id"],declared["candidate_name"],[declared["T_route_s"]]),"pilot slot/candidate/window binding differs")
            require(attempt["status"] in ("TASK_COMPLETED","EXECUTION_REFUSED","TASK_OR_EVIDENCE_FAILED","REFERENCE_PRECHECK_REJECTED",
                "ZERO_STEP_EXECUTION_REFUSAL","EXECUTION_FAILURE_NO_TRACE","PIPELINE_FAILURE") and attempt["sources_unchanged"] is True,"pilot attempt is incomplete or changed source")
            require(attempt["source_sha256_before"]==attempt["source_sha256_after"]==self.read("source_identity.json")["source_sha256"],"actual before/after source hashes differ from frozen execution sources")
            require(attempt["source_identity_sha256"]==sha(self.source/"source_identity.json") and attempt["fallback_used"] is False,"pilot source/fallback invariant differs")
            n=attempt["actual_steps"]
            require(type(n) is int and 0<=n<=13500 and n%10==0,"invalid consumed full-ramp actual step count")
            require(cost["slot_id"]==slot and cost["actual_physics_steps"]==cost["saved_actual_trace_steps"]==n,"cost ledger differs from consumed trace steps")
            for phase,operations in cost["phase_counts"].items():
                for op,counts in operations.items():require(counts["started"]==counts["returned"]+counts["raised"],"cost operation count does not close")
            for phase,key in (("actual","actual_physics_steps"),("private_preview","private_preview_physics_steps"),("independent_torque_replay","independent_saved_torque_replay_steps")):
                values=cost["phase_counts"].get(phase,{})
                require(cost[key]==sum(values.get(op,{}).get("returned",0) for op in ("mj_step","mj_step2")),"cost phase integration total differs")
            require(cost["preview_calls"]==len(cost["preview_records"]) and cost["private_preview_physics_steps"]==sum(r["completed_physics_steps"] for r in cost["preview_records"]),"private preview accounting differs")
            require(cost["native_geometry_query_calls"]==sum(p["mj_geomDistance"]["returned"] for p in cost["phase_counts"].values()),"native geometry phase sum differs")
            require(cost["control_behavior_modified"] is False and cost["wall_20ms_is_gate"] is False,"instrumentation changed execution policy")
            evaluation=self.read(attempt["evaluation_path"],attempt["evaluation_sha256"]) if attempt.get("evaluation_path") else {}
            if attempt.get("evaluation_path"):self.local_manifest(Path(attempt["evaluation_path"]).parent)
            safety=module.safety_eligibility(attempt,evaluation)
            require(safety==quality["safety"],"independently recomputed quality safety eligibility differs")
            eligible=safety["path_quality_comparison_eligible"]
            require(quality["quality_label_eligible"]==eligible and (quality["full_metrics"] is not None)==eligible,"full quality eligibility differs from original gates")
            metrics=quality["full_metrics"] if eligible else None
            if n:
                trace_path=self.bind(attempt["trace_path"],attempt["trace_sha256"])
                with np.load(trace_path,allow_pickle=False) as trace:
                    require(trace["torque"].shape==(n,67),"saved torque count/interface differs from cost ledger")
                    if eligible:
                        intervention=module.command_intervention_metrics(trace,n//10,[declared["T_route_s"]])
                        for k in ("I_route_rad_s","I_full_rad_s"):close(intervention[k],metrics[k],"saved trace intervention differs: "+slot+"/"+k)
                        require(intervention["route_planning_sample_count"]==metrics["route_planning_sample_count"],"route time-window sample count differs")
            fresh=Path(attempt["evaluation_path"]).parent/"fresh_replay.npz" if attempt.get("evaluation_path") else None
            if fresh is not None and fresh.exists():
                self.bind(fresh)
                with np.load(fresh,allow_pickle=False) as data:state={k:data[k] for k in ("time","continuum_position","base_pose","qpos","qvel")}
                require(len(state["time"])==n+1 and abs(state["time"][-1]-n*.002)<1e-8,"independent saved state horizon differs from actual prefix")
                fresh_states[slot]=state
                if eligible:
                    path_metrics=module.path_and_base_metrics(state,[declared["T_route_s"]])
                    for k,v in path_metrics.items():close(v,metrics[k],"fresh path/base metric differs: "+slot+"/"+k)
            else:require(not eligible,"eligible full quality lacks independent fresh states")
            metric=metrics or quality.get("failed_prefix_metrics")
            clearance_path=quality_dir/"route_clearance.npz"
            if metric:
                require(clearance_path.exists() and slot in fresh_states,"quality numbers lack saved fresh local geometry arrays")
                with np.load(self.bind(clearance_path),allow_pickle=False) as saved:
                    times=saved["time"];minimum=saved["continuum_obstacle_minimum_m"]
                require(np.array_equal(times,fresh_states[slot]["time"]) and minimum.shape==times.shape and np.isfinite(minimum).all(),"saved local clearance shape/time differs")
                geometry=metric["continuum_route_obstacle_clearance"]
                selected=module.window_mask(times,[declared["T_route_s"]])
                close(float(minimum.min()),geometry["full_saved_horizon_minimum_m"],"saved local full clearance minimum differs")
                if selected.any():close(float(minimum[selected].min()),geometry["route_window_minimum_m"],"saved local T_route clearance minimum differs")
                else:require(geometry["route_window_minimum_m"] is None,"prefix ending before T_route has a route minimum")
                require(geometry["query_count"]==len(times)*geometry["pair_count"]==quality["costs"]["additional_route_quality_geometry_queries"],"quality geometry query count differs")
            else:require(not clearance_path.exists(),"unavailable quality has unexplained clearance artifact")
            full=metrics or {};clearance=full.get("continuum_route_obstacle_clearance") or {}
            rows.append({"slot_id":slot,"task_id":attempt["task_id"],"method":declared["candidate_name"],"status":attempt["status"],
                "actual_steps":n,"saved_horizon_s":n*.002,"full_task_success":attempt["full_task_success"],
                "I_route_rad_s":full.get("I_route_rad_s"),"I_full_rad_s":full.get("I_full_rad_s"),
                "continuum_path_length_m":full.get("continuum_path_length_m"),"route_clearance_m":clearance.get("route_window_minimum_m"),
                "full_clearance_m":clearance.get("full_saved_horizon_minimum_m"),"base_translation_peak_m":full.get("base_translation_peak_m"),
                "base_rotation_peak_rad":full.get("base_rotation_peak_rad"),"failed_prefix_metrics":quality.get("failed_prefix_metrics") or None,
                "metric_unavailable":quality.get("metric_unavailable"),"execution_failure":attempt.get("execution_failure"),
                "evaluation_errors":(attempt.get("evaluation") or {}).get("errors"),"safety":safety})
            costs.append(cost);attempts.append(attempt)
        recomputed=module.pilot_discriminability(qualities,inputs["pilot_task_ids"])
        require(recomputed==decision,"recomputed pilot decision differs from stored decision")
        require(decision["status"]==NEGATIVE and decision["route_value_identifiable"] is False and decision["training_authorized_by_pilot"] is False,"negative-study verifier requires terminal predeclared negative stop")
        self.checks["pilot_terminal_recomputed"]={"passed":True,"terminal_slots":6,"complete_safe_slots":sum(q["quality_label_eligible"] for q in qualities),
            "recomputed_decision":decision["status"],"fresh_state_slots":len(fresh_states),"I_route_recomputed_from_saved_trace":True,
            "local_clearance_recomputed_from_saved_query_array":True,"additional_geometry_queries":0}
        return qualities,decision,rows,costs,attempts,fresh_states

    def collected(self,plan,identity,probe,qualities,decision,rows,costs,attempts,allow_pending):
        report=self.read("report.json");paired=self.read("paired_metrics.json")
        for path,digest in report["sources"].items():self.bind(path,digest)
        require(report["plan_sha256"]==paired["plan_sha256"]==sha(self.source/"plan.json") and report["source_identity_sha256"]==sha(self.source/"source_identity.json"),"collected report input identity differs")
        require(report["source_producer_commit"]==identity["algorithm_producer_commit"],"report actual producer differs")
        require(report["status"]==NEGATIVE and report["decision"]==paired["pilot"]["decision"]==decision,"collected terminal decision differs")
        require(paired["pilot"]["slots"]==rows,"paired_metrics slot table differs from six authoritative evidence chains")
        require(paired["pilot"]["task_count"]==2 and paired["pilot"]["mother_scene_count"]==1,"pilot is counted as incorrect independent tasks")
        require(all(k in report for k in STATE_FIELDS),"eight required outcome states are not separately reported")
        require(report["research_execution_complete"] is True,"research execution incomplete")
        require(report["research_delivery_complete"] is True or allow_pending and report["research_delivery_complete"] is False,"delivery is not terminal; use --allow-pending-delivery only for preliminary audit")
        require(report["route_value_identifiable"] is False and report["training_executed"] is False
            and report["conditional_value_supported_in_pilot"] is False and report["advantage_over_retrieval"]=="NOT_EVALUATED"
            and report["deployment"]=="NOT_MET","negative study state fields overclaim outcome")
        require(report["condition_response_observed"]==(probe["same_noise_pairs_observed"]>0),"P0 response state differs")
        methods=plan["TEST"]["methods"]
        require(report["independent_test_task_success_by_method"]=={m:"NOT_RUN_PILOT_STOP" for m in methods}
            and paired["new_independent_TEST"]["status"]=="NOT_RUN_PILOT_STOP"
            and paired["new_independent_TEST"]["success_by_method"]=={m:None for m in methods},"unexecuted formal TEST is credited with outcomes")
        require(report["not_run"]=={"teacher":"PILOT_NEGATIVE_STOP","training":"PILOT_NEGATIVE_STOP",
            "new_model_sampling":"PILOT_NEGATIVE_STOP","new_TEST":"PILOT_NEGATIVE_STOP","K4_actual":"NOT_RUN"},"P2/P3/P4 stop states differ")
        require(not list(self.source.rglob("*.pt")) and not list(self.source.rglob("*.pth")),"negative pilot unexpectedly has new model weights")
        for folder in ("teacher","training","model","test","dataset"):
            p=self.source/folder
            if p.exists():
                require(not list(p.rglob("attempt_result.json")) and not list(p.rglob("*.npz"))
                    and not list(p.rglob("training_config.json")),"negative-stop phase contains execution/training/dataset artifacts: "+folder)
        questions=report["five_questions"]
        require(len(questions)==5 and all(isinstance(q.get("question"),str) and isinstance(q.get("answer"),str)
            and q["question"].strip() and q["answer"].strip() for q in questions),"five direct questions/answers incomplete")
        require(len({q["question"] for q in questions})==5,"duplicate direct question")
        report_md=self.bind("REPORT.md").read_text(encoding="utf-8")
        for q in questions:require(q["question"] in report_md and q["answer"] in report_md,"Markdown report omitted a direct answer")
        require(report["pilot_full_task_success"]=={"numerator":sum(a["full_task_success"] is True for a in attempts),"denominator":6},"reported full-task success count differs")
        core={name:sum(c[name] for c in costs) for name in ("actual_physics_steps","private_preview_physics_steps",
            "independent_saved_torque_replay_steps","native_geometry_query_calls","preview_calls")}
        core.update(additional_route_quality_geometry_queries=sum(q["costs"]["additional_route_quality_geometry_queries"] for q in qualities),
            input_precheck_geometry_queries=self.read("route_pair_inputs.json")["geometry_query_count"],actual_slots_consumed=6,
            actual_runner_entries=sum(a.get("actual_runner_started") is True for a in attempts),slots_with_physics=sum(a["actual_steps"]>0 for a in attempts),
            old_checkpoint_ddim_calls=32,teacher_actual_slots=0,new_model_ddim_calls=0,new_training_runs=0,optimizer_updates=0,new_TEST_actual_slots=0)
        for name,value in core.items():require(report["costs"][name]==paired["costs"][name]==value,"collected cost differs: "+name)
        for item in paired["pilot"]["complete_safe_nonzero_vs_zero"]:
            local={r["method"]:r for r in rows if r["task_id"]==item["task_id"]};nonzero=[r for m,r in local.items() if m!="z0" and r["I_route_rad_s"] is not None]
            best=min(nonzero,key=lambda r:r["I_route_rad_s"]) if nonzero else None;zero=local["z0"]
            require(item["best_complete_safe_nonzero"]==(best["method"] if best else None),"paired best complete-safe nonzero differs")
            change=100*(best["I_route_rad_s"]-zero["I_route_rad_s"])/zero["I_route_rad_s"] if best and zero["I_route_rad_s"] else None
            if change is None:require(item["best_nonzero_vs_zero_I_change_percent"] is None,"unavailable comparison was filled")
            else:close(change,item["best_nonzero_vs_zero_I_change_percent"],"paired I_route change differs")
        with self.bind("method_table.csv").open(encoding="utf-8-sig",newline="") as f:table=list(csv.DictReader(f))
        require(len(table)==6,"CSV must contain all six pilot slots")
        for actual,expected in zip(table,rows):
            for key,value in actual.items():
                require(key in expected,"CSV adds an unbound metric")
                expected_value=expected[key]
                if expected_value is None:require(value=="","CSV filled missing full metric")
                elif isinstance(expected_value,bool):require(value==str(expected_value),"CSV boolean differs")
                elif isinstance(expected_value,(float,int)):close(float(value),expected_value,"CSV numeric differs: "+key)
                else:require(value==expected_value,"CSV identity/status differs")
        self.checks["collected_report_tables_stop"]={"passed":True,"states_checked":list(STATE_FIELDS),"five_questions":5,
            "all_six_table_rows_checked":True,"P2_P3_P4":"NOT_RUN_PILOT_STOP","new_weights":0,
            "research_delivery_complete":report["research_delivery_complete"],"preliminary_pending_allowed":allow_pending}
        return report,paired

    def visualization(self,directory,rows,fresh_states):
        directory=Path(directory).resolve();manifest=self.read(directory/"manifest.json")
        for record in manifest["inputs"]:self.bind(record["path"],record["sha256"])
        for record in manifest["artifacts"]:self.bind(directory/record["path"],record["sha256"])
        plot=self.read(directory/"plot_data.json")
        require(len(plot["scenes"])==2 and len(plot["panels"])==2,"exactly two paired route figures required")
        require(plot["paired_scene_mirroring_used"] is False and plot["actual_trace_substitution"] is False,"figure substituted or mirrored an actual trace")
        seen=[]
        from PIL import Image
        for panel in plot["panels"]:
            for key in ("png","pdf"):self.bind(directory/panel[key])
            with Image.open(directory/panel["png"]) as im:require(im.width>=2000 and im.height>=1400 and np.asarray(im.convert("RGB")).std()>5,"figure is blank or undersized")
            require((directory/panel["pdf"]).read_bytes().startswith(b"%PDF"),"figure PDF header invalid")
        by_slot={r["slot_id"]:r for r in rows}
        for scene in plot["scenes"]:
            task=self.read(f"tasks/{scene['task_id']}/task.json")
            definition=self.read(f"tasks/{scene['task_id']}/definition.json")
            require(scene["task_sha256"]==object_sha(task),"plot frozen task identity differs")
            times=np.asarray(scene["reference_times_s"],dtype=float)
            # Independently evaluate the declared minimum-jerk path for the
            # displayed reference samples; no actual trajectory interpolation.
            s=task["scenario"]["continuum_target"];points=np.asarray(s["waypoint_points_m"]);durations=np.asarray(s["segment_durations_s"])
            cumulative=np.r_[0.,np.cumsum(durations)];base=np.empty((len(times),3))
            for i,t in enumerate(times):
                if t<s["transition_duration_s"]:lower=np.asarray(s["initial_position_w"]);upper=points[0];u=t/s["transition_duration_s"]
                elif t-s["transition_duration_s"]>=s["path_duration_s"]:base[i]=points[-1];continue
                else:
                    elapsed=t-s["transition_duration_s"];j=min(int(np.searchsorted(cumulative,elapsed,side="right")-1),len(durations)-1)
                    lower=points[j];upper=points[j+1];u=(elapsed-cumulative[j])/durations[j]
                base[i]=lower+(10*u**3-15*u**4+6*u**5)*(upper-lower)
            require(np.max(np.abs(base-np.asarray(scene["base_reference_position_world_m"])))<1e-14,"plot base reference differs from frozen declaration")
            require(scene["T_route_s"]==definition["intervals_s"][scene["key_interval_slot"]],"plot route window differs from frozen definition")
            sphere=task["scenario"]["workspace_obstacles"][1]
            require(scene["moved_sphere"]=={"name":sphere["name"],"center_world_m":sphere["center_w"],"radius_m":sphere["radius_m"]},"plot sphere differs from the real scene")
            require({c["mode"] for c in scene["reference_candidates"]}=={"z0","z+","z-"},"plot omitted a fixed analytic reference")
            for candidate in scene["reference_candidates"]:
                expected_z=np.zeros((6,2));key=scene["key_interval_slot"]
                if candidate["mode"]!="z0":expected_z[key,0]=.012 if candidate["mode"]=="z+" else -.012
                require(np.array_equal(np.asarray(candidate["z_m"]),expected_z),"plot analytic candidate differs from fixed input")
                offset=np.zeros_like(base)
                for interval,basis,z,enabled in zip(definition["intervals_s"],definition["transverse_bases"],expected_z,definition["interval_mask"]):
                    if enabled:
                        inside=(times>interval[0])&(times<interval[1]);u=np.where(inside,(times-interval[0])/(interval[1]-interval[0]),0.)
                        offset+=np.where(inside,64*u**3*(1-u)**3,0.)[:,None]*(np.asarray(basis)@z)
                require(np.max(np.abs(np.asarray(candidate["reference_position_world_m"])-(base+offset)))<1e-14,"plot fixed reference route differs from analytic residual")
            require(len(scene["actual_records"])==3,"side plot omitted a fixed candidate")
            for actual in scene["actual_records"]:
                slot=actual["slot_id"];seen.append(slot);row=by_slot[slot]
                require(actual["task_id"]==scene["task_id"]==row["task_id"] and actual["candidate_name"]==row["method"]
                    and actual["status"]==row["status"] and actual["actual_physics_steps"]==row["actual_steps"],"plot metadata differs from actual slot")
                require(actual["complete_quality_eligible"]==row["safety"]["path_quality_comparison_eligible"],"plot complete eligibility differs")
                if actual["fresh_states_available"]:
                    require(slot in fresh_states,"plot invents independent fresh states")
                    state=fresh_states[slot];indices=np.asarray(actual["selected_source_state_indices"])
                    require(indices.dtype.kind in "iu" and len(indices)>0 and indices[0]==0 and indices[-1]==len(state["time"])-1
                        and np.all(np.diff(indices)>0),"plot state selection omits endpoint or invents samples")
                    require(np.array_equal(np.asarray(actual["time_s"]),state["time"][indices])
                        and np.array_equal(np.asarray(actual["continuum_position_world_m"]),state["continuum_position"][indices]),"plot actual curve is not its own independent saved states")
                    require(slot in Path(actual["state_source"]).parts,"plot state source belongs to another slot")
                    self.bind(actual["state_source"],actual["source_fresh_sha256"])
                    close(actual["saved_horizon_s"],row["saved_horizon_s"],"plot failure/full horizon differs")
                else:require(slot not in fresh_states and actual["time_s"]==[] and actual["continuum_position_world_m"]==[],"missing fresh plot uses a substitute trajectory")
                if not actual["complete_quality_eligible"]:require(actual["full_metrics"] is None,"failed/ineligible plot credits complete quality")
        require(set(seen)==set(by_slot) and len(seen)==6,"plot omitted/duplicated a pilot slot")
        self.checks["two_route_figures_and_hashes"]={"passed":True,"panels":2,"PNG":2,"PDF":2,
            "figure_manifest_inputs":len(manifest["inputs"]),"figure_manifest_artifacts":len(manifest["artifacts"]),
            "actual_curves_match_own_saved_states":True,"extra_dashboard_files_scope":"root separate visualization_validation manifest"}

    def verify_final_unchanged(self):
        for path,record in self.bindings.items():require(sha(path)==record["sha256"],"source changed before audit finished: "+path)


def verify(source,repository,visualization,allow_pending=False):
    audit=Audit(source,repository);plan=audit.read("plan.json");identity=audit.read("source_identity.json")
    inputs=audit.source_guard(plan,identity);probe=audit.probe(plan)
    qualities,decision,rows,costs,attempts,fresh=audit.pilot(plan,inputs)
    report,paired=audit.collected(plan,identity,probe,qualities,decision,rows,costs,attempts,allow_pending)
    audit.visualization(visualization,rows,fresh);audit.verify_final_unchanged()
    return {"schema":SCHEMA,"passed":True,"checked_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"local original B.3 negative-pilot evidence audit; no scientific rerun; final seal/publication handled separately",
        "plan_sha256":sha(Path(source)/"plan.json"),"source_identity_sha256":sha(Path(source)/"source_identity.json"),
        "report_sha256":sha(Path(source)/"report.json"),"paired_metrics_sha256":sha(Path(source)/"paired_metrics.json"),
        "checks":audit.checks,"bound_file_count":len(audit.bindings),
        "input_bindings":list(audit.bindings.values()),"additional_physics_steps":0,"additional_DDIM_calls":0,
        "additional_geometry_queries":0,"additional_optimizer_updates":0,
        "research_delivery_complete":report["research_delivery_complete"],"allow_pending_delivery":allow_pending}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--repository",required=True,type=Path)
    parser.add_argument("--visualization",required=True,type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--verify-only",action="store_true")
    group.add_argument("--output",type=Path)
    parser.add_argument("--allow-pending-delivery",action="store_true")
    args=parser.parse_args()
    try:
        if args.output:require(not args.output.exists(),"exclusive checks output already exists")
        result=verify(args.source,args.repository,args.visualization,args.allow_pending_delivery)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open("x",encoding="utf-8") as stream:
                json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        print(json.dumps({k:v for k,v in result.items() if k!="input_bindings"},sort_keys=True,allow_nan=False))
    except Exception as error:
        print(json.dumps({"schema":SCHEMA,"passed":False,"error_type":type(error).__name__,"error":str(error),
            "additional_physics_steps":0,"additional_DDIM_calls":0,"additional_geometry_queries":0},sort_keys=True))
        return 1
    return 0


if __name__=="__main__":sys.exit(main())
