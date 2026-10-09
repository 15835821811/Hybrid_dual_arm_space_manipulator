"""Audit two already observed TEST S raw-rejection events, using stdlib only.

Read only the selected C01/C03 JSONL rows, their compact command events, small
initializer/freeze/identity metadata and frozen source. Do not read model bytes,
NPZ, result.json, or analyze other candidate rows. Live files get row/event SHA
records with byte ranges, never an overall final-seal claim.
"""
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
STREAM = RUN / "test_search/c3_test0_plus/S"
PLANNING = STREAM / "planning/c3_test0_plus"
OUTPUT = ROOT / "docs/C3_INTERIM_RAW_REJECTION_AUDIT"
OBJECTIVE = Path(r"C:\Users\admin\.codex\attachments\1917d637-236f-4883-8674-4c3f5a6ff3e0\goal-objective.md")
PRODUCER = "9f39b42775283432eb933f63a9047c488ba22070"
START = datetime.now(timezone.utc).isoformat()
TIMER = time.perf_counter()
INPUTS = {}
CHECKS = Counter()
ERRORS = []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


def metadata_bytes(path):
    path = Path(path).resolve()
    if path.suffix not in (".json", ".py", ".md") or path.name == "result.json" or path.stat().st_size > 3 * 1024 * 1024:
        raise ValueError("outside metadata audit scope: " + str(path))
    value = path.read_bytes()
    key = path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)
    INPUTS[key] = {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value), "scope": "whole stable small metadata/source file"}
    return value


def sha(path):
    return hashlib.sha256(metadata_bytes(path)).hexdigest()


def read(path):
    return json.loads(metadata_bytes(path).decode("utf-8-sig"))


def check(group, label, condition):
    CHECKS[group] += 1
    if not condition:
        ERRORS.append({"group": group, "check": label})


def stamp(stat):
    return {"size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def selected_live_lines(path, kind):
    """Skip other raw lines without parsing or emitting their large arrays."""
    path = Path(path)
    before = path.stat()
    found = {}
    with path.open("rb") as stream:
        line_number = 0
        while len(found) < 2:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_number += 1
            for candidate in ("C01", "C03"):
                field = "candidate_id" if kind == "registry" else "candidate"
                token = ('"' + field + '": "' + candidate + '"').encode()
                compact = ('"' + field + '":"' + candidate + '"').encode()
                if token not in line and compact not in line:
                    continue
                row = json.loads(line)
                if kind == "command" and row.get("event") != "C1_CANDIDATE_TERMINAL":
                    continue
                if row.get(field) != candidate or candidate in found:
                    raise ValueError("unexpected selected live-line identity")
                record = {"path": str(path), "line_number": line_number, "byte_offset": offset,
                    "byte_length_with_newline": len(line), "terminated_by_newline": line.endswith(b"\n"),
                    "line_sha256_with_original_newline": hashlib.sha256(line).hexdigest(),
                    "canonical_content_sha256": digest(row), "scope": "only this observed complete line; not a final file seal"}
                INPUTS[path.relative_to(ROOT).as_posix() + f"#line={line_number};candidate={candidate}"] = record
                found[candidate] = {"value": row, "observation": record}
                break
    return found, {"path": str(path), "file_stat_before": stamp(before),
        "file_stat_after": stamp(path.stat()), "whole_file_hashed": False,
        "other_candidate_rows_parsed_or_emitted": False,
        "observed_candidates_only": list(found), "is_final_seal": False}


objective_sha = sha(OBJECTIVE)
plan = read(RUN / "plan.json")
identity = read(RUN / "source_identity.json")
identity_sha = sha(RUN / "source_identity.json")
frozen = read(RUN / "model_freeze.json")
selection = read(RUN / "model_selection.json")
training = read(RUN / "models/S/training_report.json")
initializer = read(STREAM / "initializer_proposals.json")
started = read(STREAM / "initializer_started.json")
inputs = read(PLANNING / "optimizer_inputs.json")
task_row = next(t for t in plan["tasks"] if t["task_id"] == "c3_test0_plus")
selected_rows, registry_observation = selected_live_lines(PLANNING / "candidate_registry.jsonl", "registry")
terminals, command_observation = selected_live_lines(STREAM / "command.log", "command")
check("observation", "both requested observed registry rows present", set(selected_rows) == {"C01", "C03"})
check("observation", "both requested compact terminal events present", set(terminals) == {"C01", "C03"})
check("identity", "frozen producer", plan["algorithm_producer_commit"] == identity["algorithm_producer_commit"] == PRODUCER)
check("identity", "original TEST Task and mask", task_row["split"] == "test" and task_row["search_interval_mask"] == [False, True, True, False, False, False])
check("identity", "initializer exact Task/stage/method", (initializer["task_sha256"], initializer["stage"], initializer["stream_id"]) == (task_row["task_sha256"], "test", "S"))
check("identity", "initializer small-content seal", initializer["content_sha256"] == digest({k:v for k,v in initializer.items() if k != "content_sha256"}))
check("identity", "original two outputs generated once without quality read", set(initializer["proposals"]) == {"1", "3"} and initializer["generated_once"] is True and initializer["candidate_quality_read"] is False)
check("identity", "initializer start Task/stage/method", (started["task_sha256"], started["stage"], started["stream_id"]) == (task_row["task_sha256"], "test", "S"))
check("identity", "generation after frozen models", datetime.fromisoformat(started["started_utc"]) > datetime.fromisoformat(frozen["frozen_utc"]))
check("identity", "frozen schema", frozen["schema"] == "v64_c3_frozen_models_and_retrieval_v1")
check("identity", "selected S4000 path", initializer["checkpoint"] == frozen["selected_checkpoints"]["S"] == selection["selected_checkpoint_files"]["S"] == str(RUN / "models/S/checkpoint_4000.pt"))
checkpoint_sha = initializer["checkpoint_sha256"]
check("identity", "S4000 metadata SHA across frozen/selected/training", checkpoint_sha == frozen["artifacts"][initializer["checkpoint"]] == selection["selected_checkpoint_sha256"]["S"] == training["checkpoint_sha256"]["4000"])
check("identity", "frozen selection metadata unchanged", frozen["artifacts"][str(RUN / "model_selection.json")] == sha(RUN / "model_selection.json"))
check("identity", "TRAIN dataset identity unchanged", initializer["dataset_manifest_sha256"] == sha(RUN / "dataset/manifest.json") == frozen["artifacts"][str(RUN / "dataset/manifest.json")])
check("identity", "optimizer Task/source binding", inputs["task_sha256"] == task_row["task_sha256"] and inputs["execution_identity"]["source_identity_sha256"] == identity_sha)
check("identity", "original eight-slot request", inputs["search_spec"]["candidate_budget"] == 8)
check("identity", "stream source copy unchanged", sha(STREAM / "source_identity.json") == identity_sha)
reservation_path = RUN / "budget_ledger" / (digest(["test_candidate_slots", "c3_test0_plus:S"]) + ".json")
reservation = read(reservation_path)
check("accounting", "eight-slot immutable request reservation", reservation["category"] == "test_candidate_slots" and reservation["unit_id"] == "c3_test0_plus:S" and reservation["count"] == 8 and reservation["metadata"]["task_sha256"] == task_row["task_sha256"])

source_names = ["route_initializers.py", "continuous_route_optimizer.py", "simple_warmstart_regression.py",
                "search_aware_warmstart_experiment.py", "route_candidate_evaluator.py", "preference_diffusion_warmstart.py"]
source_anchors = {}
terms = ["def raw_seed_plan", "np.linalg.norm(z, axis=1) > .020", "return None, diagnostics",
         "if plan is None:", "rows.append(row)", "INITIALIZER_RAW_REJECTED", "retain(row)",
         "return row", "evaluator(plan, cid)", "cache[key] = row", "predictions\" / candidate_id",
         "directory.mkdir", "raw_repaired", "resampled", "def inverse_raw", "raw = x * normalizer.std_m",
         "raw[m] += normalizer.mean_m", "if retained.exists():", "unfinished initializer",
         "return {int(k)", "self.model(condition, mask_tensor)", "for slot, (preference, family)"]
for name in source_names:
    path = ROOT / "v6_4" / name
    check("source", name + " matches frozen source identity", sha(path) == identity["source_sha256"]["v6_4/" + name])
    source_anchors["v6_4/" + name] = [{"line": index + 1, "text": line.strip()}
        for index, line in enumerate(metadata_bytes(path).decode("utf8").splitlines()) if any(term in line for term in terms)]

events = []
for candidate, slot, preference, family in (("C01", 1, "A", "v1"), ("C03", 3, "B", "v2")):
    saved = initializer["proposals"][str(slot)]
    row = selected_rows[candidate]["value"]
    terminal = terminals[candidate]["value"]
    diagnostics = row["raw_seed_diagnostics"]
    raw = saved["raw_z_m"]
    finite = len(raw) == 12 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in raw)
    norms = [math.hypot(raw[i], raw[i+1]) for i in range(0, 12, 2)]
    mask = task_row["search_interval_mask"]
    inactive = [value for interval, active in enumerate(mask) if not active for value in raw[2*interval:2*interval+2]]
    offending = [interval for interval, norm in enumerate(norms) if norm > .020]
    check("raw", candidate + " finite 12 dimensions", finite)
    check("raw", candidate + " exact inactive zero", all(value == 0. for value in inactive))
    check("raw", candidate + " exact declared mask", diagnostics["search_interval_mask"] == mask and row["active_intervals"] == [1,2])
    check("raw", candidate + " independently reproduced >20mm cause", offending == [2])
    check("raw", candidate + " original diagnostic matches norm failure", diagnostics["raw_legal"] is False and diagnostics["rejection_reason"] == "raw seed exceeds the original 20mm interval disk" and diagnostics["error_type"] == "ValueError")
    check("raw", candidate + " diagnostic raw unchanged", diagnostics["raw_z_m"] == row["raw_z_m"] == raw)
    check("raw", candidate + " complete original sampler metadata retained", all(row.get(k) == v for k,v in saved.items()))
    check("raw", candidate + " correct S condition and source", (saved["source"], saved["model"], saved["checkpoint_update"], saved["preference"], saved["family"]) == ("regression", "S", 4000, preference, family))
    check("raw", candidate + " per-row checkpoint frozen identity", saved["checkpoint_sha256"] == checkpoint_sha)
    check("raw", candidate + " no repair/no resample", saved["raw_repaired"] is False and saved["resampled"] is False and diagnostics["raw_repaired"] is False and diagnostics["resampled"] is False)
    check("raw", candidate + " one deterministic forward no DDIM", saved["regression_forward_units"] == 1 and saved["ddim_sample_units"] == 0 and saved["physics_steps"] == 0 and saved["noise_seed"] is None and saved["noise_id"] is None)
    check("raw", candidate + " inverse TRAIN scaler only", saved["raw_postprocessing"] == "inverse TRAIN scaler only")
    check("raw", candidate + " no nominal precheck fabricated", "reference_precheck" not in diagnostics)
    check("accounting", candidate + " explicit consumed rejection", row["candidate_id"] == candidate and row["initializer_slot"] == row["proposal_index"] == slot and row["status"] == "INITIALIZER_RAW_REJECTED")
    check("accounting", candidate + " terminal consumed count", terminal["candidate"] == candidate and terminal["task"] == "c3_test0_plus" and terminal["source"] == "regression" and terminal["status"] == row["status"] and terminal["slots"] == slot + 1)
    check("accounting", candidate + " zero prediction", row["prediction_rollout_started"] is False and row["prediction_steps"] == 0 and row["costs"] == {"prediction_physics_steps":0})
    check("accounting", candidate + " no plan or transformed parameter", row["plan"] is None and row["plan_sha256"] is None and row["x_m"] is None and row["prediction_metrics"] is None)
    check("accounting", candidate + " no admissibility or guards claim", row["prediction_admissible"] is False and row["prediction_task_passed"] is False and row["online_guards_passed"] is False)
    check("accounting", candidate + " no exact cache participation", "content_key" not in row and "cache_hit_candidate_id" not in row and "search_content_key" not in row)
    prediction_dir = PLANNING / "predictions" / candidate
    check("accounting", candidate + " no prediction directory observed", not prediction_dir.exists())
    check("accounting", candidate + " no actual claim", row["formal_actual_validation"] == "NOT_RUN")
    check("observation", candidate + " original complete line", selected_rows[candidate]["observation"]["terminated_by_newline"])
    events.append({"candidate_id": candidate, "initializer_slot": slot, "preference":preference,
        "family":family, "raw_z_m":raw, "search_interval_mask":mask,
        "interval_norms_m_zero_based":norms, "offending_intervals_zero_based":offending,
        "largest_norm_m": max(norms), "excess_over_original_limit_m": max(norms)-.020,
        "raw_seed_diagnostics":diagnostics, "consumed_slot_ordinal_one_based":terminal["slots"],
        "recorded_prediction_rollout_started":row["prediction_rollout_started"], "recorded_prediction_steps":row["prediction_steps"],
        "recorded_plan":row["plan"], "recorded_costs":row["costs"],
        "prediction_directory_path":str(prediction_dir), "prediction_directory_exists_at_observation":prediction_dir.exists(),
        "row_observation":selected_rows[candidate]["observation"], "compact_terminal_event":terminal,
        "terminal_event_observation":terminals[candidate]["observation"], "original_selected_registry_row":row})

verdict = "CONSISTENT_WITH_ORIGINAL_RAW_REJECTION_PROTOCOL" if not ERRORS else "DISCREPANCY_FOUND"
end = datetime.now(timezone.utc).isoformat()
script = Path(__file__).resolve()
result = {"schema":"c3_interim_raw_rejection_metadata_audit_v1", "verdict":verdict,
    "scope":"Only two already observed S4000 raw-rejection events C01/A-v1 and C03/B-v2 in live TEST c3_test0_plus/S; not final S8 or TEST validation.",
    "started_utc":START, "ended_utc":end, "checks_total":sum(CHECKS.values()),
    "checks_by_group":dict(CHECKS), "discrepancies":ERRORS, "run":str(RUN), "objective_sha256":objective_sha,
    "source_identity":{"algorithm_producer_commit":PRODUCER,"source_identity_sha256":identity_sha,
        "reviewed_source_files_byte_checked":source_names,"other_frozen_sources_and_protected_artifacts_rehashed":False},
    "frozen_checkpoint_identity":{"model":"S","update":4000,"path":initializer["checkpoint"],
        "sha256_in_original_freeze_selection_training_and_initializer":checkpoint_sha,
        "checkpoint_bytes_opened_or_hashed":False,"model_freeze_utc":frozen["frozen_utc"]},
    "execution":{"argv":[sys.executable,"-B","-X","utf8",str(script)],"cwd":str(ROOT),
        "exit_code":int(bool(ERRORS)),"elapsed_wall_s":time.perf_counter()-TIMER,
        "retained_script_relative_path":script.relative_to(ROOT).as_posix(),
        "retained_script_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),
        "core_imports_tests_inference_physics_training_DDIM_encoding_or_export":False,
        "model_weight_NPZ_or_large_archive_read_or_hash":False},
    "live_registry_observation":registry_observation,"live_command_log_observation":command_observation,
    "events":events,"source_anchors":source_anchors,
    "interpretation":"Each illegal raw consumes a candidate slot and one already recorded S forward, with zero nominal rollout and null plan. Reduced physical work from rejection is a failure-accounting fact, never evidence of learning benefit.",
    "pending":["The S8 request terminal selection/cost/outer receipt is outside this interim event audit", "All other TEST requests and failures remain unaudited here", "Whole TEST selection seal, actual execution, strict aliases and five independent gates remain pending", "No checkpoint score, overall research-completion or benefit conclusion is supplied"],
    "input_sha256_inventory":dict(sorted(INPUTS.items()))}
lines = ["# C.3 interim raw rejection audit", "", f"Verdict: **{verdict}**.", "",
    f"Observed {START} to {end} UTC: {sum(CHECKS.values())} checks, {len(ERRORS)} discrepancies. This audits only two existing raw-rejection events in live TEST `c3_test0_plus/S`: C01 (slot 1, A/v1) and the subsequently observed C03 (slot 3, B/v2). It supplies no final S8, TEST or research-completion verdict.", "",
    "Both saved proposals declare `source=regression`, model S, frozen update 4000, and exactly one deterministic regression forward per condition, zero DDIM units and zero generation physics. Initializer metadata is self-sealed, generated once and records no candidate-quality read. Its checkpoint path/SHA agree with the original frozen model, VAL selection and S training report. Checkpoint bytes were not opened or rehashed.", "",
    "| Event | Preference/family | Active interval 1 norm | Active interval 2 norm | Excess over 20 mm | Consumed slot ordinal |", "|---|---|---:|---:|---:|---:|"]
for event in events:
    n=event["interval_norms_m_zero_based"]
    lines.append(f"| {event['candidate_id']} | {event['preference']}/{event['family']} | {n[1]*1000:.9f} mm | {n[2]*1000:.9f} mm | {event['excess_over_original_limit_m']*1000:.9f} mm | {event['consumed_slot_ordinal_one_based']} |")
lines += ["", "The independent scalar calculation uses the original two-dimensional norm of each of the six residual interval pairs. Both outputs have exactly 12 finite coordinates. Mask `[false,true,true,false,false,false]` leaves four active coordinates; all inactive values, including signed zero, equal zero. Only interval index 2 (the third pair, using zero-based source indices) exceeds 0.020 m. The original `raw_seed_diagnostics` preserve the exact initializer raw numbers, report `raw_legal=false`, `raw_repaired=false`, `resampled=false`, and reject with `raw seed exceeds the original 20mm interval disk`. No reference-precheck result is invented after this earlier norm failure.", "",
    "Both registry rows independently record `INITIALIZER_RAW_REJECTED`, `prediction_rollout_started=false`, `prediction_steps=0`, null x/plan/plan-SHA/metrics, and `costs={prediction_physics_steps:0}`. Their compact original terminal events show consumed counts 2 and 4, so the rejected proposals occupy their slots. Neither row participates in the exact plan cache. Neither C01 nor C03 has a prediction directory at observation time.", "",
    "The frozen source confirms the same path: `route_initializers.raw_seed_plan` checks the original 20 mm disk before plan construction and returns null plan on rejection; the optimizer retains a consumed rejection row and returns before cache lookup or nominal evaluator invocation. Prediction directories are created inside that evaluator. The S sampler makes one direct forward per declared condition, applies only the original inverse TRAIN scaler, and has no refusal-driven resampling loop. Retained initializers return their original proposals; an unfinished initializer cannot be silently regenerated.", "",
    "The matching JSON retains exact raw values and norms, both original selected rejection rows, compact terminal events, source anchors, UTC, check counts, argv/exit and input identities. Each live registry/log input is bound only by its selected line SHA (including its original newline), byte offset/length and snapshot file size/mtime. Other candidate rows and their preview arrays are neither parsed nor emitted. No whole live-file SHA is presented as a final seal.", "",
    "The audit reads small metadata and six frozen source files with Python standard-library code. It runs no core imports, tests, model inference, physics, training, DDIM, rendering, encoding or export; it opens no weight, NPZ or large archive. Frozen source and run evidence remain unchanged.", "",
    "Pending:", "", "- Final S8 selection, measured planning cost and outer process completion are outside this event audit.", "- Complete the remaining TEST searches, whole-phase selection seal, actual execution, strict aliases and independent gates.", "- Keep failure accounting separate from quality or benefit: fewer integrations caused by illegal raw proposals cannot establish learning value.", "- TEST and research remain incomplete for this interim audit; default C.1 and deployment NOT_MET are preserved.", ""]
if ERRORS:
    lines += ["Discrepancies:", "", "```json", json.dumps(ERRORS,indent=2), "```", ""]
OUTPUT.with_suffix(".json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
OUTPUT.with_suffix(".md").write_text("\n".join(lines),encoding="utf8")
print(json.dumps({"verdict":verdict,"checks":sum(CHECKS.values()),"discrepancies":ERRORS,
    "events":[{"candidate":e["candidate_id"],"norms_m":e["interval_norms_m_zero_based"],"line_sha256":e["row_observation"]["line_sha256_with_original_newline"]} for e in events],
    "inputs":len(INPUTS),"script_sha256":result["execution"]["retained_script_sha256"]},ensure_ascii=False))
raise SystemExit(int(bool(ERRORS)))
