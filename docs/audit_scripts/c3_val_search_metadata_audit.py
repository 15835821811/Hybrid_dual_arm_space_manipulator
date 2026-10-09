"""Read-only C.3 VAL search evidence audit. No experiment module imports.

Only original JSON, command logs and source text are opened. Embedded preview
records are preserved for canonical seal checking but are not analyzed, emitted
or treated as independent evidence. No result.json, model weights or NPZ is read.
Writes only the two new documentation artifacts named below.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(r"E:\v64c3")
RUN = ROOT / "v6_4/output/search_aware_warmstart_20261008_01"
OBJECTIVE = Path(r"C:\Users\admin\.codex\attachments\1917d637-236f-4883-8674-4c3f5a6ff3e0\goal-objective.md")
OUTPUT = ROOT / "docs/C3_FINAL_VAL_SEARCH_AUDIT"
PRODUCER = "9f39b42775283432eb933f63a9047c488ba22070"
START = datetime.now(timezone.utc).isoformat()
TIMER = time.perf_counter()
INPUTS = {}
CHECKS = Counter()
ERRORS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


def blob(path):
    path = Path(path).resolve()
    if path.suffix not in (".json", ".py", ".log", ".md") or path.name == "result.json":
        raise ValueError("audit refuses non-metadata artifact: " + str(path))
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("audit refuses metadata above 4 MiB: " + str(path))
    data = path.read_bytes()
    key = path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)
    INPUTS[key] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    return data


def sha(path):
    return hashlib.sha256(blob(path)).hexdigest()


def read(path):
    return json.loads(blob(path).decode("utf-8-sig"))


def check(group, label, condition, details=None):
    CHECKS[group] += 1
    if not condition:
        ERRORS.append({"group": group, "check": label, "details": details})


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near_equal(a, b, atol=1e-6):
    return finite(a) and finite(b) and abs(a - b) <= atol


def flatten(value):
    return [x for row in value for x in row] if value and isinstance(value[0], list) else value


def source_anchors(path, terms):
    lines = blob(path).decode("utf-8-sig").splitlines()
    return [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines)
            if any(term in line for term in terms)]


objective_sha = sha(OBJECTIVE)
plan = read(RUN / "plan.json")
identity = read(RUN / "source_identity.json")
identity_sha = sha(RUN / "source_identity.json")
dataset_sha = sha(RUN / "dataset/manifest.json")
config_sha = sha(RUN / "frozen_execution_config.json")
run_config_sha = sha(RUN / "frozen_run_config.json")
prior = read(ROOT / "docs/C3_FINAL_TEACHER_DATASET_AUDIT.json")
training = {m: read(RUN / f"models/{m}/training_report.json") for m in ("D", "S")}
phase = RUN / "closed_loop_val"
manifest_path = phase / "sealed_selections/all_selections.json"
manifest = read(manifest_path)
manifest_sha = sha(manifest_path)
tasks = [t for t in plan["tasks"] if t["split"] == "val"]
task_map = {t["task_id"]: t for t in tasks}
endpoints = ["R12", "D250", "S250", "D4000", "S4000"]
schedule = [(t["task_id"], e) for i, t in enumerate(tasks)
            for e in endpoints[i:] + endpoints[:i]]
check("protocol", "frozen producer", plan["algorithm_producer_commit"] == identity["algorithm_producer_commit"] == PRODUCER)
check("protocol", "two VAL Tasks, one new mother", len(tasks) == 2 and {t["mother_id"] for t in tasks} == {"c3_val"} and all(not t.get("historical") for t in tasks))
check("protocol", "exact checkpoint whitelist", set(plan["checkpoint_whitelist"]) == {"D250", "D4000", "S250", "S4000"})
check("protocol", "fixed candidate and DDIM maxima", plan["budget_limits"]["val_candidate_slots"] == 88 and plan["budget_limits"]["val_ddim_samples"] == 8)
check("protocol", "protocol predates search", utc(plan["created_utc"]) < utc(identity["frozen_utc"]))
check("protocol", "prior finalized teacher/dataset audit reviewed", "NO_DISCREPANCY" in prior["verdict"])
check("phase_seal", "original VAL seal schema", manifest["schema"] == "v64_c3_all_phase_selections_sealed_v1" and manifest["phase"] == "VAL")
check("phase_seal", "exact 10 entry schedule", [(e["task_id"], e["endpoint"]) for e in manifest["entries"]] == schedule)
check("phase_seal", "two Tasks and five endpoints", set(manifest["task_ids"]) == set(task_map) and set(manifest["endpoints"]) == set(endpoints))
check("phase_seal", "20 future logical actual slots, no actual read", manifest["logical_actual_slots"] == 20 and manifest["selection_reads_actual"] is False and manifest["actual_started"] is False)
check("phase_seal", "phase source binding", manifest["bindings"]["source_identity_sha256"] == identity_sha)
check("phase_seal", "four checkpoint whitelist", set(manifest["checkpoint_files"]) == {"D250", "D4000", "S250", "S4000"})
check("phase_seal", "predeclared physical/preview/geometry/cold cost order", manifest["cost_order"] == ["prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls", "end_to_end_cold_planning_s"])
expected_phase_files = set(manifest["checkpoint_files"].values())

# All JSON seal members are checked against current bytes. Checkpoint bytes are
# deliberately not reopened: compare original seal/initializer/training identities.
checkpoint_identities = {}
for name, value in manifest["checkpoint_files"].items():
    expected_path = RUN / f"models/{name[0]}/checkpoint_{int(name[1:]):04d}.pt"
    check("checkpoint_identity", name + " exact path", Path(value) == expected_path)
    declared = manifest["files"][value]
    reported = training[name[0]]["checkpoint_sha256"]
    checkpoint_identities[name] = {"path": value, "sha256": declared,
        "current_size_bytes_only": expected_path.stat().st_size,
        "current_weight_bytes_rehashed": False}
    check("checkpoint_identity", name + " training report hash", declared == reported[str(int(name[1:]))])

seal_json_members = 0
for path, expected in manifest["files"].items():
    member = Path(path)
    if member.suffix == ".pt":
        continue
    check("phase_seal", "current sealed bytes " + str(member.relative_to(RUN)), sha(member) == expected)
    seal_json_members += 1

source_files = ["search_aware_warmstart_experiment.py", "closed_loop_warmstart_validation.py",
                "simple_warmstart_regression.py", "continuous_route_optimizer.py",
                "route_initializers.py", "route_candidate_evaluator.py", "route_optimizer_protocol.py",
                "preference_diffusion_warmstart.py"]
anchors = {}
terms = {
    "search_aware_warmstart_experiment.py": ["order = [", "for endpoint in order", "freeze_phase_selections(phase", "execute_frozen_task(phase", "subprocess.run", "elapsed_wall_s", "total_setup_s", "reserve(stage + \"_ddim", "initializer_started", "unfinished initializer"],
    "closed_loop_warmstart_validation.py": ["(root / \"actual\").exists()", "manifest = verify_phase_selections(root)", "actual_started", "selected_endpoints =", "files = {"],
    "simple_warmstart_regression.py": ["def frozen_noise", "SeedSequence", "rng.standard_normal", "raw_repaired", "resampled", "initial_noise_np", "ddim_sample_units", "regression_forward_units", "raw_postprocessing"],
    "continuous_route_optimizer.py": ["exactly slots 1 and 3", "snapshot_prefix", "raw_legal", "cache_hits", "cache_hit", "RAW_INITIALIZER_REJECTED", "source in", "raw_seed_diagnostics"],
    "route_initializers.py": ["def check_raw", "def inverse_raw", "raw_repaired", "resampled", "source", "project"],
    "route_candidate_evaluator.py": ["fresh", "NominalCandidateEvaluator", "execution_identity", "reset", "source_identity_sha256"],
    "route_optimizer_protocol.py": ["def initial_candidates", "0.012", "0.020", "def admissible", "def rank_candidates"],
    "preference_diffusion_warmstart.py": ["def inverse_raw", "raw = x * normalizer.std_m", "raw[m] += normalizer.mean_m", "return raw"],
}
for name in source_files:
    path = ROOT / "v6_4" / name
    current_sha = sha(path)
    key = "v6_4/" + name
    check("source_binding", key + " frozen source identity", identity["source_sha256"].get(key) == current_sha)
    anchors[key] = source_anchors(path, terms[name])

reservations = []
for path in sorted((RUN / "budget_ledger").glob("*.json")):
    # Immutable reservation records are small. Only VAL candidate/generation
    # records and the first VAL actual reservation enter this search audit.
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw["category"] in ("val_candidate_slots", "val_ddim_samples", "val_actual_slots"):
        reservations.append(read(path))
candidate_res = {r["unit_id"]: r for r in reservations if r["category"] == "val_candidate_slots"}
ddim_res = {r["unit_id"]: r for r in reservations if r["category"] == "val_ddim_samples"}
check("reservations", "exact candidate requests", set(candidate_res) == {t + ":" + ("R" if e == "R12" else e) for t, e in schedule})
check("reservations", "88 candidate slots reserved", sum(r["count"] for r in candidate_res.values()) == 88)
check("reservations", "exact four diffusion requests", set(ddim_res) == {t + ":" + e for t, e in schedule if e.startswith("D")})
check("reservations", "8 DDIM samples reserved", sum(r["count"] for r in ddim_res.values()) == 8)

streams = []
noise_by_task = {}
common = {}
generation = Counter()
previous_end = None
total_status = Counter()
selected_status = Counter()
shared_generation_ids = {}
raw_records = []
prefix_records = []
for task_id, endpoint in schedule:
    stream_id = "R" if endpoint == "R12" else endpoint
    budget = 12 if endpoint == "R12" else 8
    directory = phase / "search" / task_id / stream_id
    planning = directory / "planning" / task_id
    selection_path = planning / "selection.json"
    selection = read(selection_path)
    registry = read(planning / "candidate_registry.json")
    proposals = read(planning / "proposals.json")
    cost = read(directory / "planning_cost.json")
    outer = read(directory / "outer_process.json")
    entry = next(e for e in manifest["entries"] if (e["task_id"], e["endpoint"]) == (task_id, endpoint))
    sealed_copy = phase / "sealed_selections" / task_id / (endpoint + ".json")
    expected_paths = {"selection_source_path": selection_path,
        "selection_path": sealed_copy, "candidate_registry_path": planning / "candidate_registry.json",
        "proposals_path": planning / "proposals.json", "planning_cost_path": directory / "planning_cost.json"}
    expected_phase_files.update(str(path) for path in expected_paths.values())
    check("phase_seal", task_id + ":" + stream_id + " exact original/copy paths", all(Path(entry[k]) == path for k, path in expected_paths.items()))
    check("phase_seal", task_id + ":" + stream_id + " entry Task binding", entry["task_sha256"] == task_map[task_id]["task_sha256"])
    check("phase_seal", task_id + ":" + stream_id + " copied selection SHA binding", entry["selection_sha256"] == sha(selection_path) == sha(sealed_copy))
    check("phase_seal", task_id + ":" + stream_id + " copied selection exact content", read(sealed_copy) == selection)
    logs = [json.loads(line) for line in blob(directory / "command.log").decode("utf8").splitlines()
            if line.startswith("{")]
    terminals = [event for event in logs if event.get("event") == "C1_CANDIDATE_TERMINAL"]
    requested_sha = task_map[task_id]["task_sha256"]
    identity_expected = {"source_identity_sha256": identity_sha, "config_sha256": config_sha,
                         "run_config_sha256": run_config_sha, "task_sha256": requested_sha}
    unit = task_id + ":" + stream_id
    check("stream_identity", unit + " copied identities", all(sha(directory / name) == sha(RUN / name) for name in ("plan.json", "source_identity.json", "frozen_execution_config.json", "frozen_run_config.json")))
    check("stream_identity", unit + " selection Task binding", selection["task_id"] == task_id and selection["task_sha256"] == requested_sha)
    check("stream_identity", unit + " selection runtime binding", all(selection["execution_identity"].get(k) == v for k, v in identity_expected.items()))
    check("candidate_accounting", unit + " exact consumed rows", len(registry) == selection["budget"]["slots_consumed"] == selection["budget"]["candidate_budget"] == budget)
    check("candidate_accounting", unit + " proposal/cache equation", len(proposals) == selection["budget"]["proposal_attempts"] == budget + selection["budget"]["cache_hits"])
    check("candidate_accounting", unit + " original registry digest", digest(registry) == selection["registry_content_sha256"])
    check("candidate_accounting", unit + " normal terminal reason", selection["budget"]["stop_reason"] == "CANDIDATE_BUDGET_EXHAUSTED")
    check("candidate_accounting", unit + " A/B pool shared", selection["budget"]["shared_A_B_pool"] is True)
    check("candidate_accounting", unit + " rollout count", sum(bool(r.get("prediction_rollout_started")) for r in registry) == selection["budget"]["prediction_rollouts_started"])
    check("candidate_accounting", unit + " reservation count and Task", candidate_res[unit]["count"] == budget and candidate_res[unit]["metadata"]["task_sha256"] == requested_sha)
    check("command_receipt", unit + " exact argv", outer["argv"] == [identity["python_executable"], "-B", "-X", "utf8", "-m", "v6_4.search_aware_warmstart_experiment", "_worker", "--run", str(RUN), "--stage", "val", "--task", task_id, "--stream", stream_id])
    check("command_receipt", unit + " exit 0 and finite measured outer", outer["exit_code"] == 0 and finite(outer["elapsed_wall_s"]) and outer["elapsed_wall_s"] > 0)
    check("command_receipt", unit + " UTC/timer agreement", abs((utc(outer["ended_utc"]) - utc(outer["started_utc"])).total_seconds() - outer["elapsed_wall_s"]) < .1)
    check("command_receipt", unit + " sequential order", previous_end is None or previous_end <= utc(outer["started_utc"]))
    previous_end = utc(outer["ended_utc"])
    check("command_receipt", unit + " one worker completion", len([e for e in logs if e.get("event") == "C3_STAGE_COMPLETED" and e.get("stage") == "_worker"]) == 1)
    check("command_receipt", unit + " all consumed slots terminal", len(terminals) == budget)
    for row, terminal in zip(registry, terminals):
        tag = unit + "/" + row["candidate_id"]
        check("row_binding", tag + " compact log identity/status", terminal["candidate"] == row["candidate_id"] and terminal["task"] == task_id and terminal["status"] == row["status"])
        check("row_binding", tag + " fresh identity", all(row["execution_identity"].get(k) == v for k, v in identity_expected.items()))
        check("row_binding", tag + " no actual claimed", row["formal_actual_validation"] == "NOT_RUN")
        costs = row.get("costs") or {}
        for key in ("prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls"):
            check("cost_accounting", tag + " finite nonnegative " + key, finite(costs.get(key)) and costs[key] >= 0)
        check("cost_accounting", tag + " steps match prediction integration", costs.get("prediction_physics_steps") == row["prediction_steps"])
        check("cost_accounting", tag + " no new actual in search", costs.get("actual_physics_steps") == 0)
    check("cost_accounting", unit + " cost Task/stage", (cost["task_id"], cost["stream_id"], cost["stage"]) == (task_id, stream_id, "val"))
    check("cost_accounting", unit + " selection SHA", cost["selection_sha256"] == sha(selection_path))
    check("cost_accounting", unit + " measured cold and no inferred warm", cost["measured_mode"] == "cold" and cost["warm_latency"] is None)
    check("cost_accounting", unit + " fresh state no cross stream cache", cost["cross_stream_cache"] is False and cost["physics_from_fresh_task_state"] is True)
    check("cost_accounting", unit + " setup included", near_equal(cost["end_to_end_cold_planning_s"], cost["search_evaluator_selection_s"] + cost["setup"].get("total_setup_s", 0.)))
    check("cost_accounting", unit + " outer covers inner", outer["elapsed_wall_s"] >= cost["end_to_end_cold_planning_s"] > 0)
    check("phase_seal", unit + " timer/seal order", utc(outer["started_utc"]) < utc(cost["selection_sealed_utc"]) <= utc(outer["ended_utc"]) <= utc(manifest["sealed_utc"]))
    check("selection", unit + " no actual input", selection["formal_actual_validation"] == "NOT_RUN" and selection["selection_reads_final_actual"] is False)
    picks = {}
    for pref, chosen in selection["preferences"].items():
        cid = chosen.get("source_candidate_id")
        selected_status[chosen["status"]] += 1
        if cid is None:
            # B may have admissible nominal rows but no row meeting its 30mm
            # preference. Preserve this distinct status rather than relabeling
            # it as an absence of all admissible plans.
            allowed = ("NO_ADMISSIBLE_PLAN_WITHIN_BUDGET", "PREFERENCE_UNMET_WITHIN_BUDGET")
            check("selection", unit + "/" + pref + " absent selected plan explicit", chosen["selected_plan"] is None and chosen["status"] in allowed)
            check("selection", unit + "/" + pref + " preference-unmet only B", chosen["status"] != "PREFERENCE_UNMET_WITHIN_BUDGET" or pref == "B")
        else:
            row = next(r for r in registry if r["candidate_id"] == cid)
            check("selection", unit + "/" + pref + " qualified original row", row["prediction_admissible"] is True and row["prediction_task_passed"] is True and row["online_guards_passed"] is True and row["prediction_steps"] == 13500)
            check("selection", unit + "/" + pref + " exact original plan", chosen["selected_plan"] == row["plan"])
        picks[pref] = {"status": chosen["status"], "candidate_id": cid,
                       "source_attribution": chosen.get("source_attribution"),
                       "selected_origin_source": chosen.get("selected_origin_source"),
                       "preference_met": chosen.get("preference_met")}
    for size in (4, 8, 12) if budget == 12 else (4, 8):
        path = planning / f"prefix_{size:02d}.json"
        prefix = read(path)
        body = {k: v for k, v in prefix.items() if k != "snapshot_content_sha256"}
        pcount = prefix["budget"]["proposal_attempts"]
        check("prefix", unit + f"/{size} self seal", digest(body) == prefix["snapshot_content_sha256"])
        check("prefix", unit + f"/{size} exact consumed prefix", prefix["prefix_budget"] == prefix["budget"]["slots_consumed"] == size and prefix["candidate_ids"] == [r["candidate_id"] for r in registry[:size]])
        check("prefix", unit + f"/{size} registry digest", digest(registry[:size]) == prefix["registry_content_sha256"])
        check("prefix", unit + f"/{size} original proposal prefix digest", digest(proposals[:pcount]) == prefix["proposals_content_sha256"])
        check("prefix", unit + f"/{size} no later/actual read", prefix["later_slots_read"] is False and prefix["selection_reads_final_actual"] is False and prefix["formal_actual_validation"] == "NOT_RUN" and prefix["protocol_completed"] is True)
        check("prefix", unit + f"/{size} cost binding", prefix["candidate_costs"] == {r["candidate_id"]: r.get("costs") for r in registry[:size]})
        check("prefix", unit + f"/{size} measured time", finite(prefix["elapsed_wall_s"]) and 0 < prefix["elapsed_wall_s"] <= selection["elapsed_wall_s"])
        if size == budget:
            check("prefix", unit + " terminal selection equals final prefix", prefix["preferences"] == selection["preferences"])
        prefix_records.append({"task_id": task_id, "endpoint": endpoint, "prefix_budget": size,
            "proposal_attempts": pcount, "sha256": sha(path), "snapshot_content_sha256": prefix["snapshot_content_sha256"]})
    for slot in (0, 2):
        seed = {k: proposals[slot].get(k) for k in ("family", "x_m", "source", "origin_source")}
        expected = common.setdefault((task_id, slot), seed)
        check("initializers", unit + f" common slot {slot}", seed == expected and seed["source"] == "initial")
    check("initializers", unit + " common zero slot", all(x == 0 for x in proposals[0]["x_m"]) and proposals[0]["family"] == "v1")
    check("initializers", unit + " common 12mm slot", near_equal(math.sqrt(sum(x*x for x in proposals[2]["x_m"])), .012) and proposals[2]["family"] == "v2")
    if stream_id != "R":
        init = read(directory / "initializer_proposals.json")
        started = read(directory / "initializer_started.json")
        check("initializers", unit + " content seal", init["content_sha256"] == digest({k: v for k, v in init.items() if k != "content_sha256"}))
        check("initializers", unit + " exact generation source", init["task_sha256"] == requested_sha and init["stream_id"] == stream_id and init["stage"] == "val" and init["dataset_manifest_sha256"] == dataset_sha)
        check("initializers", unit + " two proposals once no quality read", set(init["proposals"]) == {"1", "3"} and init["generated_once"] is True and init["candidate_quality_read"] is False)
        check("initializers", unit + " checkpoint identity", init["checkpoint"] == manifest["checkpoint_files"][endpoint] and init["checkpoint_sha256"] == checkpoint_identities[endpoint]["sha256"])
        check("initializers", unit + " original setup cost", cost["setup"] == init["timing"])
        check("initializers", unit + " generation before selection", utc(outer["started_utc"]) <= utc(started["started_utc"]) < utc(cost["selection_sealed_utc"]))
        for slot, pref, fam in ((1, "A", "v1"), (3, "B", "v2")):
            saved = init["proposals"][str(slot)]
            proposal = proposals[slot]
            diag = proposal["raw_seed_diagnostics"]
            model = stream_id[0]
            raw = saved["raw_z_m"]
            check("raw_proposal", unit + f"/{slot} unchanged metadata", all(proposal.get(k) == v for k, v in saved.items()))
            check("raw_proposal", unit + f"/{slot} same raw gate", diag["requested_preference"] == pref and diag["requested_family"] == fam and diag["raw_z_m"] == raw)
            check("raw_proposal", unit + f"/{slot} honest source", saved["source"] == ("diffusion" if model == "D" else "regression") and saved["model"] == model and saved["checkpoint_update"] == int(endpoint[1:]))
            shared_ids = {k: saved[k] for k in ("training_config_sha256", "condition_scaler_sha256", "residual_scaler_sha256", "condition_schema_sha256")}
            check("raw_proposal", unit + f"/{slot} identical D/S TRAIN information", all(shared_generation_ids.setdefault(k, v) == v for k, v in shared_ids.items()))
            check("raw_proposal", unit + f"/{slot} training config binding", saved["training_config_sha256"] == training[model]["training_config_sha256"])
            check("raw_proposal", unit + f"/{slot} no repair/resample", saved["raw_repaired"] is False and saved["resampled"] is False and diag["raw_repaired"] is False and diag["resampled"] is False)
            check("raw_proposal", unit + f"/{slot} no generation physics", saved["physics_steps"] == 0)
            check("raw_proposal", unit + f"/{slot} exact active mask", diag["search_interval_mask"] == task_map[task_id]["search_interval_mask"])
            check("raw_proposal", unit + f"/{slot} finite 12D native output", raw is not None and len(raw) == 12 and all(finite(x) for x in raw))
            active = [x for interval, flag in enumerate(task_map[task_id]["search_interval_mask"]) if flag for x in raw[2*interval:2*interval+2]]
            inactive = [x for interval, flag in enumerate(task_map[task_id]["search_interval_mask"]) if not flag for x in raw[2*interval:2*interval+2]]
            check("raw_proposal", unit + f"/{slot} inactive zero", all(x == 0. for x in inactive))
            check("raw_proposal", unit + f"/{slot} no transformed search parameter", diag["raw_legal"] is not True or proposal["x_m"] == active)
            check("raw_proposal", unit + f"/{slot} per-interval 20mm", diag["raw_legal"] is not True or all(math.hypot(raw[i],raw[i+1]) <= .020+1e-12 for i in range(0,12,2)))
            check("raw_proposal", unit + f"/{slot} units", saved["ddim_sample_units"] == int(model == "D") and saved["regression_forward_units"] == int(model == "S"))
            generation["ddim_sample_units"] += saved["ddim_sample_units"]
            generation["regression_forward_units"] += saved["regression_forward_units"]
            if model == "D":
                key = (task_id, slot)
                noises = {k: saved[k] for k in ("noise_seed", "noise_id", "noise_derivation", "noise_sha256", "masked_initial_noise_sha256")}
                check("noise", unit + f"/{slot} fixed VAL noise", saved["noise_seed"] == 64325 and saved["noise_id"] == f"{requested_sha}/{pref}/{fam}/64325")
                check("noise", unit + f"/{slot} checkpoint independent bytes", noises == noise_by_task.setdefault(key, noises))
            else:
                check("noise", unit + f"/{slot} deterministic S", saved["noise_seed"] is None and saved["noise_id"] is None and saved["noise_derivation"] is None and "noise_sha256" not in saved)
            raw_records.append({"task_id": task_id, "endpoint": endpoint, "initializer_slot": slot,
                "source": saved["source"], "raw_legal": diag["raw_legal"], "raw_repaired": False, "resampled": False,
                "raw_z_m": raw, "proposal_index": proposal["proposal_index"],
                "cache_hit_candidate_id": proposal.get("cache_hit_candidate_id"),
                "rejection_reason": diag.get("rejection_reason")})
    statuses = Counter(r["status"] for r in registry)
    total_status.update(statuses)
    scalar_cost = {key: sum((r.get("costs") or {}).get(key, 0) for r in registry)
                   for key in ("prediction_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls")}
    streams.append({"task_id": task_id, "endpoint": endpoint, "stream_id": stream_id,
        "candidate_slots": len(registry), "proposal_attempts": len(proposals),
        "cache_hits": selection["budget"]["cache_hits"],
        "prediction_rollouts_started": selection["budget"]["prediction_rollouts_started"],
        "nominal_status_counts": dict(statuses),
        "incomplete_positive_step_rollouts": sum(0 < r["prediction_steps"] < 13500 for r in registry),
        "selected_predictions": picks, "costs": scalar_cost,
        "end_to_end_cold_planning_s": cost["end_to_end_cold_planning_s"],
        "outer_process_elapsed_wall_s": outer["elapsed_wall_s"],
        "outer_minus_recorded_inner_s": outer["elapsed_wall_s"] - cost["end_to_end_cold_planning_s"],
        "outer_started_utc": outer["started_utc"], "outer_ended_utc": outer["ended_utc"],
        "outer_exit_code": outer["exit_code"], "selection_sha256": sha(selection_path)})

check("totals", "88 consumed slots", sum(s["candidate_slots"] for s in streams) == 88)
check("totals", "8 DDIM and 8 direct S outputs", generation == {"ddim_sample_units": 8, "regression_forward_units": 8})
check("totals", "all 16 raw outputs legal", len(raw_records) == 16 and all(r["raw_legal"] for r in raw_records))
check("totals", "all 22 declared search prefixes", len(prefix_records) == 22)
check("phase_seal", "exact 54-member complete seal inventory", set(manifest["files"]) == expected_phase_files and len(expected_phase_files) == 54)
check("noise", "four distinct saved Task/preference noises", len(noise_by_task) == 4 and len({v["noise_sha256"] for v in noise_by_task.values()}) == 4)

live_path = RUN / "validation/val_R8_live_prefix_observation.json"
live = read(live_path)
retained_r8 = phase / "search/c3_val_plus/R/planning/c3_val_plus/prefix_08.json"
live_r8_sha = sha(retained_r8)
check("live_R8", "same original path and current hash", Path(live["path"]) == retained_r8 and live["sha256"] == live_r8_sha)
check("live_R8", "live eight terminals and no actual", live["prefix_budget"] == 8 and live["later_slots_read"] is False and live["val_snapshot"][0]["terminal_events"] == 8 and live["val_snapshot"][0]["actual_slot_files"] == 0 and live["val_snapshot"][0]["completed_requests"] == [])
first_outer = streams[0]
check("live_R8", "observation before R12 completion", utc(first_outer["outer_started_utc"]) < utc(live["observed_utc"]) < utc(first_outer["outer_ended_utc"]))
check("live_R8", "prefix written before observation", utc(live["prefix_file_write_utc"]) < utc(live["observed_utc"]))
first_actual = phase / "actual/c3_val_plus/R12_A/attempt/slot_started.json"
actual_started = read(first_actual)
actual_creation = datetime.fromtimestamp(first_actual.stat().st_ctime, timezone.utc).isoformat()
check("seal_before_actual", "first actual start marker Task and source", actual_started["slot_id"] == "VAL_c3_val_plus_R12_A" and actual_started["source_identity_sha256"] == identity_sha and actual_started["task_sha256"] == tasks[0]["task_sha256"])
check("seal_before_actual", "first actual marker filesystem creation follows phase seal", utc(manifest["sealed_utc"]) < utc(actual_creation))
actual_res = [r for r in reservations if r["category"] == "val_actual_slots"]
check("seal_before_actual", "actual reservations follow whole-phase seal", bool(actual_res) and all(utc(r["utc"]) >= utc(manifest["sealed_utc"]) for r in actual_res))

# Explicitly retain method/source limits: source identity is inspected and the
# eight relevant source files are byte-checked; the other frozen sources and
# large protected artifacts are not revalidated during the active benchmark.
END = now()
verdict = "NO_DISCREPANCY_IN_REVIEWED_VAL_SEARCH_EVIDENCE" if not ERRORS else "DISCREPANCY_FOUND"
totals = {"streams": len(streams), "candidate_slots": sum(s["candidate_slots"] for s in streams),
          "proposal_attempts": sum(s["proposal_attempts"] for s in streams),
          "cache_hits": sum(s["cache_hits"] for s in streams),
          "prediction_rollouts_started": sum(s["prediction_rollouts_started"] for s in streams),
          "nominal_status_counts": {key: total_status.get(key, 0) for key in ("RAW_INITIALIZER_REJECTED", "REFERENCE_PRECHECK_REJECTED", "TASK_INITIAL_OR_ANCHOR_PRECHECK_REJECTED", "EXECUTION_REFUSED", "PREDICTION_TASK_UNMET", "TOOL_ERROR", "PREDICTION_ADMISSIBLE")}, "selected_prediction_status_counts": dict(selected_status),
          "raw_outputs": len(raw_records), "raw_rejected": sum(not r["raw_legal"] for r in raw_records),
          "generation_units": dict(generation),
          "search_prefixes": len(prefix_records),
          "prediction_physics_steps": sum(s["costs"]["prediction_physics_steps"] for s in streams),
          "private_preview_physics_steps": sum(s["costs"]["private_preview_physics_steps"] for s in streams),
          "native_geometry_query_calls": sum(s["costs"]["native_geometry_query_calls"] for s in streams),
          "sum_inner_cold_planning_s": sum(s["end_to_end_cold_planning_s"] for s in streams),
          "sum_outer_request_wall_s": sum(s["outer_process_elapsed_wall_s"] for s in streams)}
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result = {"schema": "c3_final_val_search_metadata_audit_v1", "verdict": verdict,
    "scope": "All ten finalized VAL SEARCH requests only; actual and checkpoint scoring remain pending and are not adjudicated.",
    "run": str(RUN), "objective_sha256": objective_sha, "started_utc": START, "ended_utc": END,
    "checks_total": sum(CHECKS.values()), "checks_by_group": dict(CHECKS), "discrepancies": ERRORS,
    "source_identity": {"algorithm_producer_commit": PRODUCER, "source_identity_sha256": identity_sha,
        "frozen_utc": identity["frozen_utc"], "reviewed_source_files_byte_verified": source_files,
        "other_frozen_source_and_protected_artifacts_rehashed": False},
    "execution": {"argv": [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve())],
        "cwd": str(ROOT), "exit_code": int(bool(ERRORS)), "elapsed_wall_s": time.perf_counter() - TIMER,
        "python": sys.version, "retained_script_relative_path": Path(__file__).relative_to(ROOT).as_posix(),
        "retained_script_sha256": script_sha, "experiment_modules_imported": False,
        "physics_training_inference_DDIM_or_encoding_executed": False,
        "weights_or_NPZ_opened_or_hashed": False,
        "standalone_result_json_opened": False,
        "embedded_preview_records": "not analyzed or emitted by this audit; retained in original small JSON for canonical seals"},
    "audit_tooling_history": [{"attempt": 1, "exit_code": 1,
        "script_sha256": "932991d4b1e21b77ad42f79f27af33c03b61126fffb903a1887656dfce4fc52b",
        "reason": "Audit-only assertion initially required all absent selected plans to use NO_ADMISSIBLE_PLAN; four B endpoints correctly use PREFERENCE_UNMET_WITHIN_BUDGET. Corrected the audit assertion to preserve the original distinct B status. No experiment evidence or cost/terminal records changed."}],
    "totals": totals, "predeclared_schedule": schedule, "streams": streams,
    "checkpoint_identity_records": checkpoint_identities,
    "checkpoint_limit": "Checkpoint seal/initializer/training SHA strings crosschecked; current weight bytes intentionally not rehashed. Prior completed teacher/dataset audit records independent small-weight hashes.",
    "phase_seal": {"path": str(manifest_path), "sha256": manifest_sha, "sealed_utc": manifest["sealed_utc"],
        "current_JSON_members_verified": seal_json_members, "all_selections_before_any_actual": True if not ERRORS else None,
        "first_actual_marker_path": str(first_actual), "first_actual_marker_filesystem_creation_utc": actual_creation,
        "first_actual_marker_has_embedded_UTC": False,
        "ordering_basis": "exclusive whole-phase seal guard + sequential caller + first original actual marker filesystem creation + actual budget reservation UTC; no new timing probe"},
    "live_R8": {"observation_path": str(live_path), "observed_utc": live["observed_utc"],
        "original_prefix_sha256": live["sha256"], "current_prefix_sha256": live_r8_sha,
        "R12_outer_ended_utc": first_outer["outer_ended_utc"], "hash_unchanged": live["sha256"] == live_r8_sha},
    "raw_initializer_records": raw_records, "prefix_records": prefix_records,
    "checkpoint_shared_noise_records": [{"task_id": task, "initializer_slot": slot, **record}
        for (task, slot), record in sorted(noise_by_task.items())],
    "shared_D_S_generation_identities": shared_generation_ids,
    "source_anchors": anchors,
    "cost_interpretation": "Recorded inner cold planning includes initializer setup plus input checking/search/evaluator/selection; original outer request wall time separately includes interpreter startup/import. Warm mode was not measured. Candidate refusal/prefix failure costs remain visible and cannot establish benefit.",
    "pending": ["Complete and verify all 20 logical actual slots, strict aliases and five independent gates", "Compute actual A/B quality and NO_PLAN denominators", "Run original closed-loop checkpoint scorer and verify exact lexicographic selection", "Freeze chosen model/data/N/protocol identities before independent TEST", "No whole-VAL completion, chosen checkpoint, learning benefit or TEST verdict is supplied here"],
    "input_sha256_inventory": dict(sorted(INPUTS.items()))}
lines = ["# C.3 final VAL search metadata audit", "", f"Verdict: **{verdict}**.", "",
    f"Audited at {START} to {END} (UTC): {sum(CHECKS.values())} checks, {len(ERRORS)} discrepancies. Scope is the completed search stage of `{RUN.name}`. Actual execution, independent gates and checkpoint scoring remain pending; this document supplies no checkpoint or whole-VAL completion verdict.", "",
    "All 10 requests match the predeclared two new paired VAL Tasks and R12/D250/S250/D4000/S4000 schedule. The first Task ran R12 → D250 → S250 → D4000 → S4000; the second ran D250 → S250 → D4000 → S4000 → R12. Every fresh worker has the exact VAL/Task/stream argv, exit 0, one completion event and no overlap with the preceding request.", "",
    f"The registries contain {totals['candidate_slots']} consumed slots, {totals['proposal_attempts']} parameter proposals, {totals['cache_hits']} exact within-stream cache hit and {totals['prediction_rollouts_started']} nominal rollouts. The immutable reservations agree: 24 rule slots and 64 model slots, totaling 88; four D requests reserve and record exactly eight DDIM outputs. Four S requests record eight direct outputs. No new experiment computation was run for this audit.", "",
    "| Task | Endpoint | Slots / proposals / cache | Nominal statuses | Inner cold seconds | Outer seconds |", "|---|---|---:|---|---:|---:|"]
for s in streams:
    status = "; ".join(f"{k}={v}" for k, v in s["nominal_status_counts"].items())
    lines.append(f"| {s['task_id']} | {s['endpoint']} | {s['candidate_slots']} / {s['proposal_attempts']} / {s['cache_hits']} | {status} | {s['end_to_end_cold_planning_s']:.3f} | {s['outer_process_elapsed_wall_s']:.3f} |")
lines += ["", "Both checkpoint updates use the same per-Task A/v1 and B/v2 noise identifiers, full-noise SHA and masked-noise SHA, with fixed VAL seed 64325. The four distinct per-Task/preference noise identities were compared as saved metadata; this audit did not derive noise or run DDIM. Checkpoint paths and SHA strings agree across the original phase seal, initializer metadata and training reports. Current weight bytes were deliberately not reopened.", "",
    "All 16 D/S outputs are finite 12-dimensional raw proposals at declared initial positions 1 and 3. Their saved source is diffusion or regression respectively, with unchanged raw numbers entering the common raw legality diagnostics, inactive coordinates zero, original masks, and per-interval norms at most 20 mm. All 16 are legal; raw_repaired and resampled are false. All D/S proposals declare identical TRAIN configuration, condition/residual scalers and condition-schema identities. Common initial positions 0 and 2 are identical across methods within each Task. Position means initializer proposal position: exact cache aliases may shift consumed candidate row IDs.", "",
    f"The {seal_json_members} current JSON members of the original phase seal match their saved SHA values. All 10 selection copies and sources, registries, proposals and planning-cost files are bound before the phase seal at {manifest['sealed_utc']}. Every outer completion precedes that seal. The original first actual marker is created afterward ({actual_creation}); the first actual budget reservation also follows it. The marker has no embedded UTC, so its creation time is supporting filesystem metadata, combined with the source guard that refuses sealing after any actual directory and the caller that seals all entries before execution.", "",
    f"All 22 original 4/8/12 prefix files pass self seals, consumed-registry and original-proposal prefix digests, exact candidate IDs and costs, and no-later-slot/no-actual-read flags. The contemporaneous root observation at {live['observed_utc']} saw R8 with eight terminal slots, no completed R12 request and no actual slots. Its R8 SHA remains `{live_r8_sha}` after R12 ended at {first_outer['outer_ended_utc']}. Source control flow writes the prefix before the next candidate is constructed/evaluated.", "",
    f"Nominal status totals: {json.dumps(totals['nominal_status_counts'], ensure_ascii=False)}. Selected prediction status totals: {json.dumps(dict(selected_status), ensure_ascii=False)}. Four B endpoints have PREFERENCE_UNMET_WITHIN_BUDGET and no selected plan; this differs from the two endpoints with no admissible plan. Raw rejection is zero; nominal execution refusal, task-unmet, preference-unmet, NO_PLAN and TOOL_ERROR remain distinct. A nominal admissible prediction is not actual acceptance, and a NO_PLAN result does not establish collision or task infeasibility. Short failed prefixes cannot establish a cost benefit.", "",
    f"Recorded work totals are {totals['prediction_physics_steps']:,} main prediction steps, {totals['private_preview_physics_steps']:,} private preview steps, and {totals['native_geometry_query_calls']:,} native geometry queries. Inner cold planning sums to {totals['sum_inner_cold_planning_s']:.3f} seconds; outer request wall time sums to {totals['sum_outer_request_wall_s']:.3f} seconds. The inner timer includes initializer setup and search/selection work; the outer receipt separately includes fresh-interpreter startup/import. These are measured cold requests. Warm latency remains null. A/B share each search cost once; no cost or quality ranking is adjudicated here.", "",
    "The audit uses Python standard-library metadata reads and eight frozen source-file SHA checks, including the original inverse_raw implementation. It imports no experiment core, opens no standalone result.json, weight or NPZ, and runs no tests, inference, physics, training, rendering, encoding or export. Original run evidence and frozen core remain untouched. Embedded preview record arrays are not analyzed or emitted. The matching JSON retains input SHA/size inventory, UTC, check counts, command argv/exit, source identity, per-stream failures/costs, and retained audit-script SHA.", "",
    "The first metadata audit attempt returned exit 1 because its own assertion did not allow the original B-specific PREFERENCE_UNMET_WITHIN_BUDGET status for absent selected plans. That assertion was corrected; the attempt is retained as audit tooling history in JSON. It is outside the experiment and changes no experiment terminal or cost record.", "",
    "Pending:", "", "- Finish all 20 logical actual slots and verify strict aliases and all five independent gates.", "- Evaluate actual A/B quality, NO_PLAN denominators and the original lexicographic checkpoint scorer.", "- Freeze the selected models, shared TRAIN/N data and protocol before independent TEST.", "- Keep DATA_LIMITED, default C.1 and deployment NOT_MET; this search audit establishes no learning benefit.", ""]
if ERRORS:
    lines += ["Discrepancies:", "", "```json", json.dumps(ERRORS, indent=2), "```", ""]
OUTPUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
OUTPUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf8")
print(json.dumps({"verdict": verdict, "checks": sum(CHECKS.values()), "discrepancies": ERRORS,
                  "totals": totals, "inputs": len(INPUTS), "script_sha256": script_sha}, ensure_ascii=False))
raise SystemExit(int(bool(ERRORS)))
