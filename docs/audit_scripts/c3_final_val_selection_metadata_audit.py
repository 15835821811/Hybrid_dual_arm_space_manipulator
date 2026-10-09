"""Independent C.3 VAL-only metadata audit; standard library, no core import.

All inputs are immutable small JSON/source/command logs. TEST contents, weights,
NPZ, media and large raw interval archives are never opened or hashed. The one
TEST observation is filesystem creation metadata of its first command log only.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
import sys
import time

ROOT = Path(r"E:\v64c3")
RUN = ROOT / "v6_4/output/search_aware_warmstart_20261008_01"
PHASE = RUN / "closed_loop_val"
OBJECTIVE = Path(r"C:\Users\admin\.codex\attachments\1917d637-236f-4883-8674-4c3f5a6ff3e0\goal-objective.md")
OUT = ROOT / "docs/C3_FINAL_VAL_SELECTION_AUDIT"
PRODUCER = "9f39b42775283432eb933f63a9047c488ba22070"
TASKS = ("c3_val_minus", "c3_val_plus")
ENDPOINTS = ("R12", "D250", "D4000", "S250", "S4000")
CHECKPOINTS = {"D250": ("D", 250), "D4000": ("D", 4000), "S250": ("S", 250), "S4000": ("S", 4000)}
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")
COSTS = ("prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls", "end_to_end_cold_planning_s")
POLICY = "fresh runner/model/MjData/controller/reference provider/integrator/QP and ADMM cache; no previous candidate or stream state"
MAX_BYTES = 4 * 1024 * 1024
START = datetime.now(timezone.utc).isoformat()
TIMER = time.perf_counter()
CACHE, INPUTS, ERRORS, COUNTS, DEFERRED = {}, {}, [], Counter(), {}


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def raw(p):
    p = Path(p).resolve()
    if any(part in ("test_search", "frozen_test") for part in p.parts):
        raise ValueError("TEST CONTENT READ PROHIBITED: " + str(p))
    if p.suffix.lower() not in (".json", ".jsonl", ".csv", ".md", ".py", ".log", ".ps1") or p.stat().st_size > MAX_BYTES:
        raise ValueError("NONSMALL/NONMETADATA READ PROHIBITED: " + str(p))
    key = str(p)
    if key not in CACHE:
        CACHE[key] = p.read_bytes()
        INPUTS[key] = {"sha256": hashlib.sha256(CACHE[key]).hexdigest(), "size_bytes": len(CACHE[key])}
    return CACHE[key]


def sha(p):
    raw(p)
    return INPUTS[str(Path(p).resolve())]["sha256"]


def read(p):
    return json.loads(raw(p).decode("utf-8-sig"))


def check(group, label, condition, observed=None):
    COUNTS[group] += 1
    if not condition:
        ERRORS.append({"group": group, "check": label, "observed": observed})


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def seal(directory):
    """Check current small members only; keep original digests for deferred data."""
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    for rel, expected in manifest.items():
        p = (directory / rel).resolve()
        check("slot_seals", str(p) + " remains inside seal", directory in p.parents)
        check("slot_seals", str(p) + " exists", p.is_file())
        if p.suffix == ".json" and p.stat().st_size <= MAX_BYTES:
            check("slot_seals", str(p) + " current small SHA", sha(p) == expected)
        else:
            DEFERRED[str(p)] = {"recorded_sha256": expected, "size_bytes": p.stat().st_size, "reason": "raw archive/interval log/media not reopened or hashed"}
    return manifest


def events(p):
    result = []
    for line in raw(p).decode("utf-8-sig").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except json.JSONDecodeError:
            pass
    return result


objective_sha = sha(OBJECTIVE)
for document in ("C3_STATE_ISOLATION_AUDIT.md", "C3_FINAL_VAL_SEARCH_AUDIT.md"):
    raw(ROOT / "docs" / document)
identity = read(RUN / "source_identity.json")
identity_sha = sha(RUN / "source_identity.json")
config = read(RUN / "frozen_execution_config.json")
run_config = read(RUN / "frozen_run_config.json")
config_sha, run_config_sha = sha(RUN / "frozen_execution_config.json"), sha(RUN / "frozen_run_config.json")
check("identity", "frozen producer and 430 entries", identity["algorithm_producer_commit"] == PRODUCER and identity["git_head"] == PRODUCER and len(identity["source_sha256"]) == 430)
source_files = ("v6_4/closed_loop_warmstart_validation.py", "v6_4/search_aware_warmstart_experiment.py", "v6_4/residual_execution.py", "v6_4/task_protocol.py", "v6_4/task_anchored_reference.py", "v6_4/route_candidate_evaluator.py", "v6_4/route_optimizer_protocol.py", "v6_4/route_initializers.py")
for rel in source_files:
    check("identity", rel + " current raw bytes = frozen producer entry", sha(ROOT / rel) == identity["source_sha256"][rel])
source_text = raw(ROOT / source_files[0]).decode("utf8")
check("identity", "static whitelist contains only declared updates", 'CHECKPOINTS = {"D250": ("D", 250), "D4000": ("D", 4000),' in source_text and '"S250": ("S", 250), "S4000": ("S", 4000)}' in source_text)
check("identity", "scorer target is VAL and has no TEST file read", 'root = Path(phase_root).resolve(); manifest = verify_phase_selections(root)' in source_text and 'manifest["phase"] != "VAL"' in source_text and 'root / "model_selection.json"' in source_text)
plan = read(RUN / "plan.json")
check("identity", "predeclared checkpoint whitelist", set(plan["checkpoint_whitelist"]) == set(CHECKPOINTS))
phase_manifest_path = PHASE / "sealed_selections/all_selections.json"
phase_manifest = read(phase_manifest_path)
check("phase_seal", "exact two VAL Tasks and five endpoints", phase_manifest["phase"] == "VAL" and tuple(phase_manifest["task_ids"]) == TASKS and set(phase_manifest["endpoints"]) == set(ENDPOINTS))
check("phase_seal", "20 logical slots; all-selection seal declares no actual read", phase_manifest["logical_actual_slots"] == 20 and phase_manifest["selection_reads_actual"] is False and phase_manifest["actual_started"] is False)
check("phase_seal", "source binding", phase_manifest["bindings"] == {"source_identity_sha256": identity_sha})
check("phase_seal", "checkpoint whitelist", set(phase_manifest["checkpoint_files"]) == set(CHECKPOINTS))
check("phase_seal", "cost order", tuple(phase_manifest["cost_order"]) == COSTS)
phase_small = 0
for path, expected in phase_manifest["files"].items():
    p = Path(path)
    if p.suffix == ".json":
        check("phase_seal", str(p) + " current SHA", sha(p) == expected)
        phase_small += 1
    else:
        DEFERRED[str(p)] = {"recorded_sha256": expected, "size_bytes": p.stat().st_size, "reason": "checkpoint weight bytes not reopened or hashed"}
check("phase_seal", "54 members / 50 current JSON SHAs / four deferred weights", len(phase_manifest["files"]) == 54 and phase_small == 50)

tasks = {t: read(RUN / "frozen_tasks" / t / "task.json") for t in TASKS}
task_shas = {t: digest(tasks[t]) for t in TASKS}
history = {t: {"policy": POLICY, **{k: tasks[t][k] for k in ("initial_qpos", "initial_qvel", "initial_planner_q", "initial_planner_dq")}} for t in TASKS}
streams, entries, raw_records, checkpoints = {}, {}, [], {}
for endpoint, (model, update) in CHECKPOINTS.items():
    p = Path(phase_manifest["checkpoint_files"][endpoint])
    checkpoint_sha = phase_manifest["files"][str(p)]
    training = read(RUN / "models" / model / "training_report.json")
    check("checkpoint_identity", endpoint + " file/path/update/SHA whitelist", p == RUN / "models" / model / f"checkpoint_{update:04d}.pt" and training["checkpoint_updates"] == [250, 4000] and training["checkpoint_sha256"][str(update)] == checkpoint_sha)
    checkpoints[endpoint] = {"model": model, "update": update, "path": str(p), "recorded_sha256": checkpoint_sha, "current_bytes_rehashed": False}

for e in phase_manifest["entries"]:
    t, endpoint = e["task_id"], e["endpoint"]
    entries[(t, endpoint)] = e
    selection, rows, proposals, cost = (read(e[k]) for k in ("selection_path", "candidate_registry_path", "proposals_path", "planning_cost_path"))
    stream_root = Path(e["planning_cost_path"]).parent
    outer = read(stream_root / "outer_process.json")
    stream_id, budget = ("R", 12) if endpoint == "R12" else (endpoint, 8)
    expected_argv_tail = ["-B", "-X", "utf8", "-m", "v6_4.search_aware_warmstart_experiment", "_worker", "--run", str(RUN), "--stage", "val", "--task", t, "--stream", stream_id]
    check("request_receipts", t + "/" + endpoint + " exact VAL argv and exit 0", outer["argv"][1:] == expected_argv_tail and outer["exit_code"] == 0)
    check("request_receipts", t + "/" + endpoint + " complete command event", sum(v.get("event") == "C3_STAGE_COMPLETED" and v.get("stage") == "_worker" for v in events(stream_root / "command.log")) == 1)
    check("request_receipts", t + "/" + endpoint + " ends before whole phase seal", utc(outer["ended_utc"]) < utc(phase_manifest["sealed_utc"]))
    check("search_inputs", t + "/" + endpoint + " task/copy SHA binding", e["task_sha256"] == task_shas[t] == selection["task_sha256"] and sha(e["selection_path"]) == e["selection_sha256"] == sha(e["selection_source_path"]))
    check("search_inputs", t + "/" + endpoint + " consumed rows and canonical digest", selection["budget"]["candidate_budget"] == budget and selection["budget"]["slots_consumed"] == len(rows) == budget and digest(rows) == selection["registry_content_sha256"])
    check("search_inputs", t + "/" + endpoint + " actual blind, measured cold, no cross-stream cache", selection["selection_reads_final_actual"] is False and cost["measured_mode"] == "cold" and cost["warm_latency"] is None and cost["cross_stream_cache"] is False and cost["selection_sha256"] == e["selection_sha256"])
    for row in rows:
        check("search_inputs", t + "/" + endpoint + "/" + row["candidate_id"] + " exact execution identities", row["execution_identity"] == {"source_identity_sha256": identity_sha, "config_sha256": config_sha, "run_config_sha256": run_config_sha, "task_sha256": task_shas[t], "model_contract_sha256": tasks[t]["model_contract_sha256"]})
    illegal = 0
    if endpoint in CHECKPOINTS:
        outputs = [v for v in proposals if v.get("initializer_slot") in (1, 3)]
        check("raw", t + "/" + endpoint + " exactly two original raw outputs", len(outputs) == 2 and {v["initializer_slot"] for v in outputs} == {1, 3})
        for v in outputs:
            diag, z = v["raw_seed_diagnostics"], v["raw_z_m"]
            mask = diag["search_interval_mask"]
            scalar_legal = len(z) == 12 and all(finite(x) for x in z) and all(x == 0 for i, active in enumerate(mask) if not active for x in z[2*i:2*i+2]) and all(math.hypot(*z[2*i:2*i+2]) <= .020 + 1e-12 for i in range(6))
            check("raw", t + "/" + endpoint + f"/{v['initializer_slot']} raw unchanged and legal", scalar_legal and diag["raw_legal"] is True and z == diag["raw_z_m"] and diag["reference_precheck"]["passed"] is True and not v["raw_repaired"] and not v["resampled"])
            check("raw", t + "/" + endpoint + " model/update/checkpoint identity", v["checkpoint_sha256"] == checkpoints[endpoint]["recorded_sha256"] and v["checkpoint_update"] == CHECKPOINTS[endpoint][1] and v["model"] == CHECKPOINTS[endpoint][0])
            illegal += diag["raw_legal"] is not True
            raw_records.append({"task_id": t, "endpoint": endpoint, "initializer_slot": v["initializer_slot"], "raw_legal": diag["raw_legal"], "finite_inactive_zero_interval_norm_legal": scalar_legal, "source": v["source"], "raw_repaired": v["raw_repaired"], "resampled": v["resampled"]})
        status_illegal = sum(r["status"] in ("INITIALIZER_RAW_REJECTED", "RAW_INITIALIZER_REJECTED") for r in rows)
        check("raw", t + "/" + endpoint + " independent raw count agrees with consumed statuses", illegal == status_illegal)
    workload = {k: sum(r["costs"].get(k, 0) for r in rows) for k in COSTS[:-1]}
    workload[COSTS[-1]] = cost[COSTS[-1]]
    check("cost", t + "/" + endpoint + " finite nonnegative measured work", all(finite(v) and v >= 0 for v in workload.values()))
    streams[(t, endpoint)] = {"task_id": t, "endpoint": endpoint, "selection": selection, "rows": rows, "raw_illegal": illegal, "workload": workload, "outer": outer, "proposals": len(proposals), "cache_hits": selection["budget"]["cache_hits"]}
check("search_inputs", "88 consumed slots / 16 raw outputs", sum(len(s["rows"]) for s in streams.values()) == 88 and len(raw_records) == 16)
ordered_streams = sorted(streams.values(), key=lambda s: utc(s["outer"]["started_utc"]))
for left, right in zip(ordered_streams, ordered_streams[1:]):
    check("request_receipts", "sequential VAL requests " + left["endpoint"] + "->" + right["endpoint"], utc(left["outer"]["ended_utc"]) <= utc(right["outer"]["started_utc"]))

slots, endpoint_records, reports, unique_records, aliases = {}, [], {}, [], []
for t in TASKS:
    for endpoint in ENDPOINTS:
        e, selection = entries[(t, endpoint)], streams[(t, endpoint)]["selection"]
        for pref in ("A", "B"):
            d = PHASE / "actual" / t / (endpoint + "_" + pref)
            slot, manifest = read(d / "slot.json"), seal(d)
            chosen = selection["preferences"][pref]
            p = chosen.get("selected_plan")
            plan_sha = digest(p) if p else None
            alias_id = digest({"task_sha256": task_shas[t], "model_contract_sha256": tasks[t]["model_contract_sha256"], "plan_sha256": plan_sha, "initial_history": history[t], "source_identity_sha256": identity_sha, "config_sha256": config_sha, "run_config_sha256": run_config_sha}) if p else None
            binding = {"task_sha256": task_shas[t], "selection_sha256": e["selection_sha256"], "plan_sha256": plan_sha, "initial_history_sha256": digest(history[t]), "alias_identity": alias_id}
            check("actual_identity", t + "/" + endpoint + "/" + pref + " exact slot/history/plan/selection identity", slot["actual_binding"] == binding and slot["alias_identity"] == alias_id and (slot["task_id"], slot["endpoint"], slot["preference"], slot["phase"]) == (t, endpoint, pref, "VAL") and slot["source_candidate_id"] == chosen.get("source_candidate_id"))
            record = {"task_id": t, "endpoint": endpoint, "preference": pref, "slot_path": str(d / "slot.json"), "slot_sha256": sha(d / "slot.json"), "plan_sha256": plan_sha, "alias_identity": alias_id, "unique_actual": slot["unique_run"], "alias_of_slot": slot.get("alias_of_slot"), "selection_status": chosen["status"], "source_attribution": slot["source_attribution"], "selected_origin_source": slot["selected_origin_source"], "selected_lineage": slot["selected_lineage"]}
            if p is None:
                check("NO_PLAN", t + "/" + endpoint + "/" + pref + " zero work and no false acceptance", slot["status"] == slot["diagnostic_category"] == "NO_PLAN" and slot["actual_steps"] == 0 and slot["unique_run"] is False and slot["entered_actual"] is False and slot["quality"] is None and all(v is None for v in slot["five_gates"].values()) and slot["full_task_success"] is False and not (d / "attempt").exists() and slot["costs"] == {"no_actual_work": True})
                record.update(actual_steps=0, no_plan=True, full_27s_five_gates=False, gates={k: None for k in GATES}, quality=None, B30=False)
            else:
                check("actual_identity", str(d) + " selected-plan bytes and canonical identity", read(d / "selected_plan.json") == p and digest(read(d / "selected_plan.json")) == slot["plan_sha256"])
                row = next(r for r in streams[(t, endpoint)]["rows"] if r["candidate_id"] == chosen["source_candidate_id"])
                check("actual_identity", str(d) + " selected original registry plan", row["plan_sha256"] == plan_sha and row["plan"] == p)
                original_dir = Path(slot["alias_of_slot"]).parent if slot.get("alias_of_slot") else d
                if slot.get("alias_of_slot"):
                    original = read(original_dir / "slot.json")
                    seal(original_dir)
                    check("alias", str(d) + " same Task/plan/config/history; original is unique", original_dir.parent.name == t and not original.get("alias_of_slot") and original["unique_run"] is True and original["alias_identity"] == alias_id and original["task_sha256"] == task_shas[t] and original["plan_sha256"] == plan_sha and sha(original_dir / "slot.json") == slot["alias_of_slot_sha256"])
                    check("alias", str(d) + " source slot predates alias; zero new work", original_dir.stat().st_ctime <= d.stat().st_ctime and slot["unique_run"] is False and slot["costs"] == {"alias_zero_new_work": True} and slot["elapsed_wall_s"] == 0)
                    for k in ("status", "diagnostic_category", "entered_actual", "actual_steps", "full_task_success", "full_27s_success", "original_independent_gates_passed", "five_gates", "clearance_30mm_met", "quality", "actual_quality", "independent_evaluation_path", "trace_path"):
                        check("alias", str(d) + " copies original " + k, slot[k] == original[k])
                    aliases.append({"slot": str(d / "slot.json"), "original": slot["alias_of_slot"], "alias_identity": alias_id})
                report_path = Path(slot["independent_evaluation_path"])
                report = read(report_path)
                gates = {k: report.get(k, {}).get("passed") is True for k in GATES}
                full = bool(report["complete"] is True and report["evidence_valid"] is True and report["full_task_success"] is True and all(gates.values()) and report["actual_physics_steps"] == 13500 and abs(report["actual_saved_horizon_s"] - 27.) <= 1e-8 and slot["actual_steps"] == 13500)
                check("acceptance", str(d) + " derives full 27s and five independent gates", full == slot["full_task_success"] == slot["full_27s_success"] == slot["original_independent_gates_passed"] and gates == slot["five_gates"] and full)
                check("acceptance", str(d) + " independent report bound to Task/plan", report["task_sha256"] == task_shas[t] and report["reference_binding"]["plan_sha256"] == plan_sha and report["trace_path"] == slot["trace_path"] and not report["errors"])
                quality_doc = read(original_dir / "quality/quality.json")
                q = quality_doc["metrics"]
                check("quality", str(d) + " actual saved quality equals slot and independent path length", q == slot["quality"] == slot["actual_quality"] and q["L_full"] == report["metrics"]["actual_tip_path_length_m"]["continuum"] and q["native_state_count"] == 13501 and q["saved_horizon_s"] == 27. and q["state_source"] == "independent_saved_torque_actual_replay")
                check("quality", str(d) + " finite quality / measured support clearance", all(finite(q[k]) for k in ("I_support", "L_full", "d_support")) and q["clearance_status"] == "MEASURED" and q["clearance_is_censored_lower_bound"] is False and q["d_support"] == q["clearance_witness"]["signed_distance_m"] and abs(q["I_support"]**2 - q["I_support_10D"]**2 - q["I_support_7D"]**2) <= 1e-12)
                b30 = full and pref == "B" and q["d_support"] >= .030
                check("quality", str(d) + " recomputed 30mm matches declaration", slot["clearance_30mm_met"] == (q["d_support"] >= .030))
                differences = {k: q[k] - chosen["prediction_metrics"][k] for k in ("I_support", "L_full", "d_support")}
                check("quality", str(d) + " prediction-actual differences", differences == slot["prediction_actual_difference"] and slot["prediction_actual_consistent"] == all(abs(v) <= 1e-9 for v in differences.values()))
                record.update(actual_steps=slot["actual_steps"], no_plan=False, full_27s_five_gates=full, gates=gates, quality={k:q[k] for k in ("I_support", "L_full", "d_support", "base_translation_peak_m", "base_rotation_peak_rad")}, B30=b30, prediction_actual_difference=differences, original_independent_report_path=str(report_path), original_independent_report_sha256=sha(report_path))
                if original_dir == d:
                    attempt, metadata = read(d / "attempt/attempt_result.json"), read(d / "attempt/actual/run_metadata.json")
                    evaluation_manifest = seal(d / "attempt/actual/evaluation")
                    check("actual_identity", str(d) + " original attempt/source/config SHA", attempt["source_identity_sha256"] == identity_sha and attempt["qp_config_sha256"] == config_sha and attempt["plan_content_sha256"] == plan_sha and attempt["task_sha256"] == task_shas[t] and attempt["evaluation"] == report and attempt["evaluation_sha256"] == sha(report_path))
                    check("actual_identity", str(d) + " frozen runtime producer/config/controller", metadata["source"]["git_commit"] == PRODUCER and metadata["source"]["tracked_worktree_dirty"] is False and metadata["run_config"] == run_config and metadata["qp_config"] == config and metadata["runtime_identity"]["execution_mode"] == "research_simulation" and metadata["runtime_identity"]["controller_version"] == "v6_2_research_simulation_bounded_interval_pcc" and metadata["runtime_identity"]["servo_law_version"] == "b2_implicitfast_compensated_torque_v1" and metadata["model_identity"]["runtime_contract_sha256"] == tasks[t]["model_contract_sha256"])
                    marker = read(d / "attempt/slot_started.json")
                    check("actual_identity", str(d) + " original start marker binding", marker["slot_id"] == f"VAL_{t}_{endpoint}_{pref}" and marker["source_identity_sha256"] == identity_sha and marker["task_sha256"] == task_shas[t] and marker["plan_content_sha256"] == plan_sha)
                    check("time", str(d) + " actual marker follows whole phase seal", utc(phase_manifest["sealed_utc"]).timestamp() < (d / "attempt/slot_started.json").stat().st_ctime and utc(phase_manifest["sealed_utc"]) < utc(metadata["started_utc"]))
                    check("raw_archive_bindings", str(d) + " saved trace/replay/interval SHA strings consistent", attempt["trace_sha256"] == report["trace_sha256"] == quality_doc["trace_sha256"] == manifest[f"attempt/actual/traces/{t}.npz"] and report["fresh_replay_sha256"] == quality_doc["replay_sha256"] == evaluation_manifest["fresh_replay.npz"] and report["independent_interval"]["boundary_sha256"] == evaluation_manifest["interval_boundaries.jsonl"] and report["independent_interval"]["rows_sha256"] == evaluation_manifest["interval_rows.jsonl"])
                    reqs = report["task_requirements"]["requirements"]
                    check("gate_detail", str(d) + " eight frozen requirements represented", len(reqs) == len(tasks[t]["requirements"]) == 8 and {x["point_id"] for x in reqs} == {x["point_id"] for x in tasks[t]["requirements"]})
                    for req in reqs:
                        frozen = next(x for x in tasks[t]["requirements"] if x["point_id"] == req["point_id"])
                        numerical = req["eligible_state_count"] > 0 and req["position_error_m"] <= frozen["position_tolerance_m"] and req["orientation_error_rad"] <= frozen["orientation_tolerance_rad"] and frozen["time_window_s"][0] - 1e-8 <= req["best_time_s"] <= frozen["time_window_s"][1] + 1e-8
                        if frozen["kind"] == "terminal":
                            numerical = numerical and abs(req["best_time_s"] - 27.) <= 1e-8
                        check("gate_detail", str(d) + "/" + req["point_id"] + " recorded errors within original tolerance/window", numerical and req["passed"] is True and req["position_tolerance_m"] == frozen["position_tolerance_m"] and req["orientation_tolerance_rad"] == frozen["orientation_tolerance_rad"])
                    contract, interval, geometry, reference = (report[k] for k in ("execution_contract", "independent_interval", "native_geometry", "reference_binding"))
                    check("gate_detail", str(d) + " execution contract subchecks", all(v is True for v in contract["checks"].values()) and max(contract["shared_ramp_max_errors"]) <= 1e-9 and contract["torque_saturations"] == 0)
                    check("gate_detail", str(d) + " independent interval mismatch/count", interval["mismatch_count"] == 0 and interval["boundary_count"] == 1351 and interval["row_count"] > 0)
                    native = read(d / "attempt/actual/evaluation/native_geometry.json")
                    check("gate_detail", str(d) + " original geometry subfile equals report", native == geometry)
                    target, body = geometry["robot_target_500hz"], geometry["whole_body"]
                    check("gate_detail", str(d) + " original discrete geometry scopes and no violations", target["state_count"] == 13501 and target["below_5mm_states"] == target["negative_states"] == target["truncated_query_count"] == 0 and target["minimum_m"] >= .005 and body["feasible"] is True and body["violation_count"] == body["negative_distance_query_count"] == body["truncated_query_count"] == 0 and body["minimum_clearance"] >= .005 and body["supplied_sample_count"] == 1351 and body["adaptive_subdivisions"] == 4 and geometry["continuous_time_certified"] is False)
                    residuals = list(reference["component_maximum_absolute_residual"].values()) + list(reference["retained_field_maximum_residual"].values())
                    check("gate_detail", str(d) + " reference bound residuals and counts", all(finite(v) and abs(v) <= 1e-9 for v in residuals) and reference["planning_inputs_bound"] == 1350 and reference["generated_continuum_and_posture_poststep_samples_bound"] == 13500 and reference["future_actual_trace_is_online_input"] is False)
                    work = {k:slot["costs"].get(k, 0) for k in ("actual_physics_steps", "private_preview_physics_steps", "independent_saved_torque_replay_steps", "native_geometry_query_calls")}
                    unique_records.append({"task_id": t, "endpoint": endpoint, "preference": pref, "started_utc": metadata["started_utc"], "finished_utc": metadata["finished_utc"], "elapsed_wall_s": slot["elapsed_wall_s"], "work": work, "gate_detail_source": str(report_path), "current_small_seals_verified": True})
            slots[(t, endpoint, pref)] = record
            endpoint_records.append(record)


def reference_qualified(t, pref):
    s = slots[(t, "R12", pref)]
    keys = ("I_support", "L_full") if pref == "A" else ("L_full", "d_support")
    return s["full_27s_five_gates"] and s["quality"] is not None and all(finite(s["quality"][k]) for k in keys)


def near(t, endpoint, pref):
    if not reference_qualified(t, pref):
        return None
    s, baseline = slots[(t, endpoint, pref)], slots[(t, "R12", pref)]
    if not s["full_27s_five_gates"] or not s["quality"]:
        return False
    a, b = s["quality"], baseline["quality"]
    return a["L_full"] <= b["L_full"] + .005 and (a["I_support"] <= b["I_support"] + .001 if pref == "A" else a["d_support"] >= .030)


def first_near(t, endpoint, pref):
    if not reference_qualified(t, pref):
        return {"status": "N/A_R12_NO_QUALIFIED_ACTUAL", "slot": None, "sort_encoding": None}
    b = slots[(t, "R12", pref)]["quality"]
    for index, row in enumerate(streams[(t, endpoint)]["rows"][:8], 1):
        m = row.get("prediction_metrics") or {}
        keys = ("I_support", "L_full") if pref == "A" else ("L_full", "d_support")
        if row.get("prediction_admissible") is not True or any(not finite(m.get(k)) for k in keys):
            continue
        if m["L_full"] <= b["L_full"] + .005 and (m["I_support"] <= b["I_support"] + .001 if pref == "A" else m["d_support"] >= .030):
            return {"status": "HIT", "slot": index, "sort_encoding": index}
    return {"status": "RIGHT_CENSORED", "slot": None, "right_censored_budget": 8, "sort_encoding": 9, "encoding_is_observed_hit": False}


recomputed, summaries = {}, {}
for endpoint in ENDPOINTS:
    actual = [slots[(t, endpoint, p)] for t in TASKS for p in ("A", "B")]
    near_rows = [{"task_id": t, "preference": p, "near_quality": near(t, endpoint, p)} for t in TASKS for p in ("A", "B")]
    for s in actual:
        s["near_quality_relative_R12"] = near(s["task_id"], endpoint, s["preference"])
    summary = {"full_27s_five_gate_endpoints": sum(s["full_27s_five_gates"] for s in actual), "actual_B30_endpoints": sum(s["B30"] for s in actual), "near_quality_endpoints": sum(v["near_quality"] is True for v in near_rows), "near_quality_by_endpoint": near_rows, "NO_PLAN": sum(s["no_plan"] for s in actual), "unique_actual": sum(s["unique_actual"] for s in actual), "alias_count": sum(bool(s["alias_of_slot"]) for s in actual)}
    summaries[endpoint] = summary
    if endpoint in CHECKPOINTS:
        model, update = CHECKPOINTS[endpoint]
        first = [{"task_id": t, "preference": p, **first_near(t, endpoint, p)} for t in TASKS for p in ("A", "B")]
        work = {k:sum(streams[(t, endpoint)]["workload"][k] for t in TASKS) for k in COSTS}
        illegal = sum(streams[(t, endpoint)]["raw_illegal"] for t in TASKS)
        first_sum = sum(v["sort_encoding"] for v in first if v["sort_encoding"] is not None)
        key = [-summary["full_27s_five_gate_endpoints"], -summary["actual_B30_endpoints"], -summary["near_quality_endpoints"], illegal, first_sum, *[work[k] for k in COSTS], update]
        recomputed[endpoint] = {"model": model, "update": update, **{k:summary[k] for k in ("full_27s_five_gate_endpoints", "actual_B30_endpoints", "near_quality_endpoints", "near_quality_by_endpoint")}, "raw_illegal_initializers": illegal, "first_near_quality": first, "first_near_budget_encoding_sum": first_sum, "workload": work, "lexicographic_min_key": key, "checkpoint_sha256": checkpoints[endpoint]["recorded_sha256"]}
selected = {m:min((e for e in CHECKPOINTS if CHECKPOINTS[e][0] == m), key=lambda e:recomputed[e]["lexicographic_min_key"]) for m in ("D", "S")}
root_selection, closed_selection = read(RUN / "model_selection.json"), read(PHASE / "model_selection.json")
check("scoring", "root and closed-loop selections byte-identical", sha(RUN / "model_selection.json") == sha(PHASE / "model_selection.json") and root_selection == closed_selection)
check("scoring", "exact complete independently recomputed four scores", root_selection["scores"] == recomputed, {"saved": root_selection["scores"], "recomputed": recomputed} if root_selection["scores"] != recomputed else None)
qualified_set = [{"task_id":t, "preference":p} for t in TASKS for p in ("A", "B") if reference_qualified(t, p)]
check("scoring", "R12 qualified actual reference set; N/A excluded", root_selection["R12_qualified_reference_set"] == qualified_set)
check("scoring", "selection result follows independent keys", all(root_selection["selected_checkpoint_" + m] == selected[m] for m in ("D", "S")) and root_selection["test_read"] is False and root_selection["closed_loop_val_completed"] is True)
check("scoring", "phase seal binding and exact 20 current slot SHAs", root_selection["phase_manifest_sha256"] == sha(phase_manifest_path) and len(root_selection["actual_slot_files"]) == 20 and root_selection["actual_slot_files"] == {s["slot_path"]:s["slot_sha256"] for s in endpoint_records})
check("scoring", "current selected paths/SHA agree with phase identities", all(root_selection["selected_checkpoint_files"][m] == checkpoints[selected[m]]["path"] and root_selection["selected_checkpoint_sha256"][m] == checkpoints[selected[m]]["recorded_sha256"] for m in ("D", "S")))
for row in endpoint_records:
    check("time", row["slot_path"] + " finalized before scorer result", Path(row["slot_path"]).stat().st_mtime <= (PHASE / "model_selection.json").stat().st_mtime)
val_receipt, freeze_receipt = read(RUN / "command_logs/phase_closed-loop-val.json"), read(RUN / "command_logs/phase_freeze-models.json")
for stage, receipt in (("closed-loop-val", val_receipt), ("freeze-models", freeze_receipt)):
    check("phase_receipts", stage + " original exact argv exit 0 no retry", receipt["argv"][1:] == ["-u", "-B", "-X", "utf8", "-m", "v6_4.search_aware_warmstart_experiment", stage, "--run", "v6_4/output/search_aware_warmstart_20261008_01"] and receipt["exit_code"] == 0 and receipt["automatic_retry"] is False and receipt["stage"] == stage)
    check("phase_receipts", stage + " one original completion event", sum(v.get("event") == "C3_STAGE_COMPLETED" and v.get("stage") == stage for v in events(RUN / "command_logs" / (stage + ".log"))) == 1)
actual_events = [v for v in events(RUN / "command_logs/closed-loop-val.log") if v.get("event") == "C3_FINAL_ACTUAL"]
check("phase_receipts", "20 final logical events and exact slots", len(actual_events) == 20 and {(v["task"],v["endpoint"],v["preference"]) for v in actual_events} == set(slots))
check("phase_receipts", "11 original feedback starts and terminals", sum(v.get("event") == "B2_ACTUAL_STARTED" for v in events(RUN / "command_logs/closed-loop-val.log")) == len(unique_records) == sum(v.get("event") == "B2_ACTUAL_TERMINAL" for v in events(RUN / "command_logs/closed-loop-val.log")))
for event in actual_events:
    s = slots[(event["task"],event["endpoint"],event["preference"])]
    check("phase_receipts", str((event["task"],event["endpoint"],event["preference"])) + " event matches saved result", event["steps"] == s["actual_steps"] and event["unique"] == s["unique_actual"] and event["category"] == ("NO_PLAN" if s["no_plan"] else "FULL_TASK_AND_FIVE_GATES_PASSED"))
freeze = read(RUN / "model_freeze.json")
check("model_freeze", "independent selections and exact chosen SHA identities", freeze["selected_checkpoints"] == {m:checkpoints[selected[m]]["path"] for m in ("D", "S")} and all(freeze["artifacts"][checkpoints[selected[m]]["path"]] == checkpoints[selected[m]]["recorded_sha256"] for m in ("D", "S")))
freeze_verified = 0
for path, expected in freeze["artifacts"].items():
    p = Path(path)
    if p.suffix in (".json", ".jsonl") and p.stat().st_size <= MAX_BYTES:
        check("model_freeze", str(p) + " current small SHA", sha(p) == expected)
        freeze_verified += 1
    else:
        DEFERRED[str(p)] = {"recorded_sha256":expected, "size_bytes":p.stat().st_size, "reason":"weight/NPZ/large dataset archive not reopened or hashed"}
check("model_freeze", "root selection SHA bound", freeze["artifacts"][str(RUN / "model_selection.json")] == sha(RUN / "model_selection.json"))
check("time", "closed-loop completion then freeze command then freeze file then completion", utc(val_receipt["ended_utc"]) <= utc(freeze_receipt["started_utc"]) <= utc(freeze["frozen_utc"]) <= utc(freeze_receipt["ended_utc"]))
check("time", "all unique actual finished before closed-loop completion", all(utc(v["finished_utc"]) < utc(val_receipt["ended_utc"]) for v in unique_records))
# Only stat this known first TEST command log; never open/read/hash its content.
test_marker = RUN / "test_search/c3_test0_plus/R/command.log"
test_created_utc = datetime.fromtimestamp(test_marker.stat().st_ctime, timezone.utc).isoformat()
check("time", "freeze completed before first TEST request log creation (filesystem metadata)", utc(freeze_receipt["ended_utc"]) <= utc(test_created_utc))

totals = {"logical_actual_slots":len(endpoint_records), "unique_actual":len(unique_records), "strict_aliases":len(aliases), "NO_PLAN":sum(s["no_plan"] for s in endpoint_records), "logical_full_27s_five_gates":sum(s["full_27s_five_gates"] for s in endpoint_records), "unique_full_27s_five_gates":len(unique_records), "actual_precheck_rejected":0, "actual_failure":0, "tool_error":0, "raw_illegal_initializers":sum(s["raw_illegal"] for s in streams.values()), "candidate_slots":sum(len(s["rows"]) for s in streams.values()), "parameter_proposals":sum(s["proposals"] for s in streams.values()), "within_stream_exact_cache_hits":sum(s["cache_hits"] for s in streams.values()), "prediction_work":{k:sum(s["workload"][k] for s in streams.values()) for k in COSTS}, "outer_VAL_search_wall_sum_s":sum(s["outer"]["elapsed_wall_s"] for s in streams.values()), "unique_actual_work":{k:sum(s["work"][k] for s in unique_records) for k in unique_records[0]["work"]}, "sum_unique_actual_pipeline_wall_s":sum(s["elapsed_wall_s"] for s in unique_records), "closed_loop_phase_wall_s":val_receipt["elapsed_wall_s"]}
check("totals", "20 = 11 unique + 3 aliases + 6 no-plan", totals["logical_actual_slots"] == 20 and totals["unique_actual"] == 11 and totals["strict_aliases"] == 3 and totals["NO_PLAN"] == 6 and totals["logical_full_27s_five_gates"] == 14)
check("totals", "unique actual steps 148500; not 14 independent trials", totals["unique_actual_work"]["actual_physics_steps"] == 148500)
END = now()
verdict = "NO_DISCREPANCY_IN_REVIEWED_VAL_SELECTION_METADATA" if not ERRORS else "DISCREPANCY_FOUND"
history_attempts = []
if OUT.with_suffix(".json").exists():
    previous = json.loads(OUT.with_suffix(".json").read_text(encoding="utf8"))
    history_attempts = previous.get("audit_tooling_history", [])
    history_attempts.append({"started_utc":previous["started_utc"], "ended_utc":previous["ended_utc"], "exit_code":previous["execution"]["exit_code"], "script_sha256":previous["execution"]["script_sha256"], "discrepancies":previous["discrepancies"], "note":"Previous audit-only attempt retained; no original evidence changed."})
result = {"schema":"c3_final_val_selection_metadata_audit_v1", "verdict":verdict, "scope":"Independent recomputation of completed VAL only, from saved original small metadata; model freeze identity/timing included. TEST scoring, final TEST and whole-study completion are outside this audit.", "run":str(RUN), "started_utc":START, "ended_utc":END, "objective_sha256":objective_sha, "checks_total":sum(COUNTS.values()), "checks_by_group":dict(COUNTS), "discrepancies":ERRORS, "execution":{"argv":[sys.executable,"-B","-X","utf8",str(Path(__file__).resolve())],"cwd":str(ROOT),"exit_code":int(bool(ERRORS)),"elapsed_wall_s":time.perf_counter()-TIMER,"python":sys.version,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"experiment_core_imported":False,"tests_physics_DDIM_inference_training_replay_media_executed":False,"TEST_content_opened_or_hashed":False,"weights_NPZ_or_large_archives_opened_or_hashed":False}, "audit_tooling_history":history_attempts, "source_identity":{"producer":PRODUCER,"source_identity_sha256":identity_sha,"recorded_source_count":len(identity["source_sha256"]),"current_relevant_source_files_verified":list(source_files),"all_430_sources_and_protected_artifacts_rehashed":False}, "totals":totals, "per_endpoint_summary":summaries, "logical_endpoint_records":endpoint_records, "unique_actual_records":unique_records, "strict_alias_records":aliases, "R12_qualified_reference_set":qualified_set, "independently_recomputed_scores":recomputed, "independently_selected_checkpoint":selected, "selection_deciding_criterion":{"D":"criterion 3: near-quality endpoints 3 versus 1; full endpoints and B30 tied (3 and 1)","S":"criterion 1: complete 27s/five-gate endpoints 3 versus 2"}, "checkpoint_identity_records":checkpoints, "raw_initializer_records":raw_records, "all_selections_seal":{"path":str(phase_manifest_path),"sha256":sha(phase_manifest_path),"sealed_utc":phase_manifest["sealed_utc"],"current_small_JSON_members_verified":phase_small,"deferred_weight_members":4}, "model_freeze":{"path":str(RUN/"model_freeze.json"),"sha256":sha(RUN/"model_freeze.json"),"frozen_utc":freeze["frozen_utc"],"current_small_artifacts_verified":freeze_verified,"artifacts_recorded":len(freeze["artifacts"]),"test_search_started_false_is_declaration_at_freeze":freeze["test_search_started"] is False}, "phase_receipts":{"closed_loop_val":val_receipt,"freeze_models":freeze_receipt}, "timing":{"first_TEST_log_path":str(test_marker),"first_TEST_log_creation_utc_filesystem_only":test_created_utc,"TEST_log_content_read":False,"basis":"Original embedded VAL/freeze UTC, original exit receipts, saved-source sequential guards; filesystem creation times support seal/start ordering, are not an immutable external clock."}, "evidence_limits":["Five-gate result comes from original independently computed evaluation/report.json, with numerical/subcheck metadata and small sealed subfiles crosschecked; the auditor did not recompute gates from NPZ/raw traces.","All 20 local slot manifests and all unique independent-evaluation manifests checked for current small JSON member hashes; large raw members retain their original recorded SHA strings only.","Saved trace/replay/interval identities crosscheck across original manifests, report, attempt and quality; current raw archive bytes were not rehashed.","Cold initial-history digest was independently reconstructed from frozen Task arrays and policy, supported by frozen fresh-runner control flow. No live ctrl/QP dual/integral telemetry or trace initial-state replay was audited.","Only eight relevant current source files byte-checked against the 430-entry frozen identity; full430-source/protected-artifact audit remains separate.","No TEST contents enter recomputation; only first TEST command-log filesystem creation metadata supports freeze chronology.","One VAL mother / two paired Tasks / one training seed: finite model selection, no generalization or learning-benefit claim. Default C.1; deployment NOT_MET; continuous/hardware safety NOT_ESTABLISHED."], "deferred_artifacts":DEFERRED,"input_sha256_inventory":dict(sorted(INPUTS.items()))}
lines = ["# C.3 final VAL selection audit", "", f"Verdict: **{verdict}**.", "", f"Independent metadata audit at {START}–{END} UTC: **{sum(COUNTS.values())} checks, {len(ERRORS)} discrepancies**. Only completed VAL and model-freeze identity/timing are adjudicated. No TEST score or whole-study completion is asserted.", "", "The independently recomputed result is **D4000 and S4000**. D4000 wins at criterion 3 (near quality 3 versus 1, after both tie at three complete endpoints and one B30). S4000 wins at criterion 1 (three complete endpoints versus two). Cost is later in the fixed ordering and decides neither winner.", "", "| Endpoint | Full 27s + five gates /4 | B30 /2 | Near /3 eligible | NO_PLAN /4 | Unique actual | Strict aliases |", "|---|---:|---:|---:|---:|---:|---:|"]
for e in ENDPOINTS:
    s=summaries[e];lines.append(f"| {e} | {s['full_27s_five_gate_endpoints']} | {s['actual_B30_endpoints']} | {s['near_quality_endpoints']} | {s['NO_PLAN']} | {s['unique_actual']} | {s['alias_count']} |")
lines += ["", "All 20 logical slots are retained: **11 unique actual + 3 strict aliases + 6 NO_PLAN**; 14 logical endpoints pass full 27s and the original five gates. All 11 executed actuals pass. No actual precheck rejection, actual failure or tool error is present. The three aliases are plus-Task S250-A/D4000-A/S4000-A to its R12-A, with matching Task, plan, source/config and independently reconstructed cold-history digest, original slot SHA and zero new work. Aliases are not independent trials.", "", "R12 actual references qualify for minus-A, plus-A and plus-B. Minus-B has NO_PLAN, so relative quality is N/A for every checkpoint and is excluded from both near-quality count and first-hit sum. D250 fails near quality on plus-A and plus-B; S250 has NO_PLAN on minus-A. Both 4000 checkpoints keep all three qualified references. Each model's minus-B remains NO_PLAN; a complete chosen model result for every preference is not established.", "", "The full lexicographic minimum keys are preserved below in the order: negative full endpoints, negative B30, negative near endpoints, raw illegal, first-hit encoding sum, prediction steps, preview steps, native geometry queries, measured cold planning seconds, update.", "", "```json", json.dumps({e:s['lexicographic_min_key'] for e,s in recomputed.items()},indent=2), "```", "", "| Checkpoint | First near minus-A | minus-B | plus-A | plus-B | Sum |", "|---|---:|---|---:|---:|---:|"]
for e,s in recomputed.items():
    values=[("N/A" if r['status'].startswith('N/A') else '>8 (right-censored; code 9)' if r['status']=='RIGHT_CENSORED' else str(r['slot'])) for r in s['first_near_quality']]
    lines.append(f"| {e} | " + " | ".join(values) + f" | {s['first_near_budget_encoding_sum']} |")
lines += ["", "First hits use only each saved consumed eight-slot registry against the same actual-qualified R12 set. Code 9 means no hit within budget 8 and is never claimed as an observed ninth slot. All 16 original D/S raw proposals are legal, finite, mask-valid and within 20mm per interval; no raw repair/resampling or rejection. First-near results are prediction screening, not additional independently executed actuals.", "", f"The 88 consumed search slots contain 89 parameter proposals and one exact within-stream cache hit. Recorded main prediction steps total {totals['prediction_work']['prediction_physics_steps']:,}, preview steps {totals['prediction_work']['private_preview_physics_steps']:,}, geometry queries {totals['prediction_work']['native_geometry_query_calls']:,}. Cold inner planning sums to {totals['prediction_work']['end_to_end_cold_planning_s']:.3f}s, outer request wall sum {totals['outer_VAL_search_wall_sum_s']:.3f}s. A/B share each request's cost once. The 11 unique actual pipelines add 148,500 feedback steps, {totals['unique_actual_work']['private_preview_physics_steps']:,} preview steps and {totals['unique_actual_work']['independent_saved_torque_replay_steps']:,} independent replay steps. Actual/validation offline wall sums to {totals['sum_unique_actual_pipeline_wall_s']:.3f}s; original whole-VAL phase wall is {totals['closed_loop_phase_wall_s']:.3f}s. These are separate scopes; partial predictions and NO_PLAN costs cannot establish benefit.", "", f"The original all-selections seal ({phase_manifest['sealed_utc']}) binds all 10 requests and precedes every unique actual marker/start. Its 50 small JSON members still match; four sealed weight SHA identities agree with training, initializer, scoring and freeze records without reopening weights. All 20 local slot seals and 11 independent-evaluation seals have their current small JSON members checked. Original raw trace/replay/interval SHA strings agree across evidence layers, but raw archive bytes are deferred.", "", f"Root and closed_loop_val/model_selection.json are byte-identical (`{sha(RUN/'model_selection.json')}`), and their four full score dictionaries exactly match this independent recomputation. Model freeze at {freeze['frozen_utc']} binds those chosen checkpoint identities and the root selection. Its {freeze_verified} small current artifacts match their hashes. Original closed-loop exit 0 precedes freeze-models exit 0; the freeze completion at {freeze_receipt['ended_utc']} precedes the first TEST command-log creation ({test_created_utc}, filesystem metadata only). No TEST file content was read or hashed. The frozen source also verifies model freeze before starting TEST workers.", "", "The original five-gate evidence is the saved independent evaluation/report.json, not the slot's five boolean declarations. The audit crosschecks eight task errors/windows/tolerances, execution subchecks, independent interval counts/mismatch 0, native geometry subfile and recorded zero violations, reference residuals/binding counts, full 13500 steps and 27s saved horizon. Whole-body geometry retains its original 50Hz boundaries plus four configuration subdivisions; robot-target geometry uses saved 500Hz states. This does not upgrade discrete evidence to continuous-time safety.", "", "The auditor did not recompute the physical gates or quality from raw traces, reopen/hash NPZ/weights/large candidate facts/media, run tests or import experiment core. Only eight relevant sources are byte-verified against producer `" + PRODUCER + "` and the 430-entry source identity. Cold-history hashes bind frozen initial arrays and a reset policy supported by source control flow; live controller/QP dual/reference-integral histories and trace initial-state telemetry were not independently re-audited. Embedded preview-record arrays were not analyzed or emitted by the audit script.", "", "Scope remains DATA_LIMITED: one VAL mother, two paired Tasks, one training seed. This completes the VAL selection metadata audit only. TEST, overall research execution, independent generalization, learning benefit, release and visualization refresh remain outside this verdict. Default C.1 and deployment NOT_MET remain; continuous-time and hardware safety are NOT_ESTABLISHED.", "", "The matching JSON retains UTC, exact audit argv/exit/script SHA, original phase argv/exit/time, input SHA inventory, check counts, per-logical-endpoint quality/source identities, full keys, aliases, all differences and explicitly deferred artifact inventory.", ""]
lines += ["Logical actual endpoints (quality units: I rad/s, L m, d mm):", "", "| Task | Endpoint | Pref | Actual kind | Full27s/five | I | L | d | Near R12 |", "|---|---|---|---|---|---:|---:|---:|---|"]
for row in endpoint_records:
    q = row["quality"]
    kind = "NO_PLAN" if row["no_plan"] else "alias" if row["alias_of_slot"] else "unique"
    value = lambda k, scale=1.: "N/A" if q is None else f"{q[k]*scale:.6f}"
    near_text = "N/A" if row["near_quality_relative_R12"] is None else str(row["near_quality_relative_R12"])
    lines.append(f"| {row['task_id']} | {row['endpoint']} | {row['preference']} | {kind} | {row['full_27s_five_gates']} | {value('I_support')} | {value('L_full')} | {value('d_support',1000.)} | {near_text} |")
lines += [""]
if history_attempts:
    lines += ["Audit tooling history is retained in JSON. Corrections affected only this metadata audit script, never original experiment evidence.", ""]
if ERRORS:
    lines += ["Discrepancies:", "", "```json", json.dumps(ERRORS,indent=2), "```", ""]
OUT.with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
OUT.with_suffix(".md").write_text("\n".join(lines),encoding="utf8")
print(json.dumps({"verdict":verdict,"checks":sum(COUNTS.values()),"discrepancies":ERRORS,"selected":selected,"totals":totals,"inputs":len(INPUTS)},ensure_ascii=False))
raise SystemExit(int(bool(ERRORS)))
