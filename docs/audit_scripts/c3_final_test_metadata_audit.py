"""Prepared, unexecuted C.3 final TEST metadata audit (stdlib only).

No experiment imports, model loading, NumPy, physics, subprocess, or archive
rehashing. The entry point performs a completeness preflight before opening any
candidate registry. Missing evidence gives PARTIAL/exit 2, a contradiction gives
FAIL/exit 1, and complete consistent metadata gives exit 0 with binary/source
verification explicitly deferred. Output is restricted to documentation paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(r"E:\v64c3")
DEFAULT_RUN = ROOT / "v6_4/output/search_aware_warmstart_20261008_01"
PRODUCER = "9f39b42775283432eb933f63a9047c488ba22070"
MAX_BYTES = 4 * 1024 * 1024
METHODS = ("R", "N", "S", "D")
ENDPOINTS = ("R8", "R12", "N8", "S8", "D8")
PREFS = ("A", "B")
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")
QUALITY = ("I_support", "L_full", "d_support", "base_translation_peak_m", "base_rotation_peak_rad")
WORK = ("prediction_physics_steps", "actual_physics_steps", "private_preview_physics_steps",
        "independent_saved_torque_replay_steps", "native_geometry_query_calls", "qp_solve_calls")
POLICY = "fresh runner/model/MjData/controller/reference provider/integrator/QP and ADMM cache; no previous candidate or stream state"
RAW_REJECTED = "INITIALIZER_RAW_REJECTED"
LIMITS = {"teacher_candidate_slots": 96, "val_candidate_slots": 88, "test_candidate_slots": 144,
          "val_actual_slots": 20, "test_actual_slots": 40, "val_ddim_samples": 8, "test_ddim_samples": 8,
          "smoke_ddim_samples": 8, "diagnostic_ddim_samples": 8, "D_training_runs": 1, "S_training_runs": 1}
SOURCES = ("search_aware_warmstart_experiment.py", "closed_loop_warmstart_validation.py",
           "continuous_route_optimizer.py", "route_initializers.py", "simple_warmstart_regression.py",
           "preference_diffusion_warmstart.py", "route_candidate_evaluator.py", "conditional_execution.py",
           "residual_execution.py", "route_optimizer_protocol.py", "task_protocol.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b, tolerance=1e-8):
    return finite(a) and finite(b) and abs(a-b) <= tolerance


def flat(raw):
    if isinstance(raw, list) and len(raw) == 6 and all(isinstance(pair, list) and len(pair) == 2 for pair in raw):
        return [value for pair in raw for value in pair]
    return raw if isinstance(raw, list) else None


def scalar_raw(raw, mask):
    z = flat(raw)
    shape = z is not None and len(z) == 12
    valid = shape and all(finite(value) for value in z)
    inactive_zero = bool(valid and all(z[2*i+j] == 0. for i, active in enumerate(mask) if not active for j in (0, 1)))
    norms = [math.hypot(z[2*i], z[2*i+1]) for i in range(6)] if valid else None
    return {"container_12D": shape, "finite": valid, "inactive_zero": inactive_zero,
            "interval_norms_m": norms, "within_original_disks": bool(norms is not None and all(n <= .020 for n in norms))}


def history(task):
    return {"policy": POLICY, **{key: task[key] for key in
            ("initial_qpos", "initial_qvel", "initial_planner_q", "initial_planner_dq")}}


def qualified(slot, preference):
    required = ("I_support", "L_full") if preference == "A" else ("L_full", "d_support")
    return bool(slot and slot["full_27s_five_gates"] and all(finite((slot.get("quality") or {}).get(k)) for k in required))


def near(slot, reference, preference):
    if not qualified(reference, preference):
        return None
    if not slot or not slot["full_27s_five_gates"]:
        return False
    if not qualified(slot, preference):
        return None
    a, b = slot["quality"], reference["quality"]
    return a["L_full"] <= b["L_full"]+.005 and (a["I_support"] <= b["I_support"]+.001 if preference == "A" else a["d_support"] >= .030)


def predicted_full(row):
    quality = (row or {}).get("prediction_metrics") or {}
    return bool(row and row.get("prediction_admissible") is True and row.get("prediction_task_passed") is True
        and row.get("online_guards_passed") is True and row.get("prediction_steps") == 13500
        and quality.get("native_state_count") == 13501 and close(quality.get("saved_horizon_s"), 27.)
        and all(finite(quality.get(k)) for k in ("I_support", "L_full")))


class Audit:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.started = now()
        self.timer = time.perf_counter()
        self.cache, self.inputs, self.deferred = {}, {}, {}
        self.errors, self.pending = [], []
        self.counts = Counter()
        self.streams, self.entries, self.slots, self.tasks = {}, {}, {}, {}
        self.endpoint_costs, self.prefixes, self.raw_outputs = {}, [], []
        self.generation = Counter()
        self.all_rows, self.all_slots = [], []
        self.observations = {}
        self.results = {}

    def check(self, group, label, condition, observed=None):
        self.counts[group] += 1
        if not condition:
            self.errors.append({"group": group, "check": label, "observed": observed})

    def require(self, paths):
        missing = [str(Path(p)) for p in paths if not Path(p).is_file()]
        self.pending.extend({"path": path, "reason": "required terminal evidence absent; never treated as a pass"} for path in missing)
        return not missing

    def raw(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in (".json", ".py", ".md", ".log") or path.name == "result.json":
            raise ValueError("nonmetadata read refused: " + str(path))
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("metadata above 4 MiB refused: " + str(path))
        if str(path) not in self.cache:
            before = path.stat()
            value = path.read_bytes()
            after = path.stat()
            self.check("snapshot", str(path) + " stable read", (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
            self.cache[str(path)] = value
            self.inputs[str(path)] = {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
        return self.cache[str(path)]

    def sha(self, path):
        self.raw(path)
        return self.inputs[str(Path(path).resolve())]["sha256"]

    def read(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"))

    def events(self, path):
        records = []
        for line in self.raw(path).decode("utf-8-sig").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def seal(self, directory):
        """Verify only permitted current small members; defer binary/log bytes."""
        directory = Path(directory).resolve()
        if not self.require([directory / "manifest.json"]):
            return {}
        manifest = self.read(directory / "manifest.json")
        for relative, expected in manifest.items():
            path = (directory / relative).resolve()
            self.check("small_seals", str(path) + " contained", directory in path.parents)
            if directory not in path.parents or not self.require([path]):
                continue
            if path.suffix == ".json" and path.name != "result.json" and path.stat().st_size <= MAX_BYTES:
                self.check("small_seals", str(path) + " current SHA", self.sha(path) == expected)
            else:
                self.deferred[str(path)] = {"recorded_sha256": expected, "size_bytes": path.stat().st_size,
                    "reason": "binary, standalone result, interval JSONL or large member; existence only, no current-byte verification"}
        return manifest

    def phase_manifest(self, phase_name, stage, tasks, endpoints, bindings):
        path = self.run / phase_name / "sealed_selections/all_selections.json"
        manifest = self.read(path)
        pairs = {(t, e) for t in tasks for e in endpoints}
        self.check("phase_seal", phase_name + " exact schema/schedule", manifest.get("schema") == "v64_c3_all_phase_selections_sealed_v1"
            and manifest.get("phase") == stage and set(manifest.get("task_ids", [])) == set(tasks)
            and set(manifest.get("endpoints", [])) == set(endpoints)
            and {(e["task_id"], e["endpoint"]) for e in manifest["entries"]} == pairs and len(manifest["entries"]) == len(pairs))
        self.check("phase_seal", phase_name + " actual-blind logical slots/bindings", manifest.get("logical_actual_slots") == 2*len(pairs)
            and manifest.get("selection_reads_actual") is False and manifest.get("actual_started") is False and manifest.get("bindings") == bindings)
        expected_files = set(manifest.get("checkpoint_files", {}).values())
        for entry in manifest["entries"]:
            task, endpoint = entry["task_id"], entry["endpoint"]
            stream_id = "R" if endpoint in ("R8", "R12") else endpoint[0] if stage == "TEST" else endpoint
            stream_root = self.run / ("test_search" if stage == "TEST" else "closed_loop_val/search") / task / stream_id
            planning = stream_root / "planning" / task
            original = planning / ("prefix_08.json" if endpoint == "R8" else "selection.json")
            copied = self.run / phase_name / "sealed_selections" / task / (endpoint + ".json")
            paths = {"selection_source_path": original, "selection_path": copied,
                "candidate_registry_path": planning / "candidate_registry.json", "proposals_path": planning / "proposals.json",
                "planning_cost_path": stream_root / "planning_cost.json"}
            expected_files.update(str(p.resolve()) for p in paths.values())
            self.check("phase_seal", task+"/"+endpoint+" original paths", all(Path(entry.get(k, "")) == p for k, p in paths.items()))
            self.check("phase_seal", task+"/"+endpoint+" exact copied selection", self.read(original) == self.read(copied)
                and self.sha(original) == self.sha(copied) == entry.get("selection_sha256")
                and entry.get("task_sha256") == digest(self.tasks[task]))
            self.entries[(stage.lower(), task, endpoint)] = entry
        self.check("phase_seal", phase_name + " complete member inventory", set(manifest["files"]) == expected_files)
        for name, expected in manifest["files"].items():
            path = Path(name)
            if not self.require([path]):
                continue
            if path.suffix == ".json" and path.stat().st_size <= MAX_BYTES:
                self.check("phase_seal", name + " current SHA", self.sha(path) == expected)
            else:
                self.deferred[name] = {"recorded_sha256": expected, "size_bytes": path.stat().st_size,
                                       "reason": "phase checkpoint bytes deferred"}
        return manifest

    def work(self, costs, label, raw_rejected=False):
        if raw_rejected:
            self.check("cost", label + " raw-reject exact zero main work", costs == {"prediction_physics_steps": 0})
            return {key: 0 for key in WORK}
        self.check("cost", label + " original ledger schema", costs.get("schema") == "v64_b3_execution_cost_ledger_v1")
        phases = costs.get("phase_counts", {})
        def returned(phase, operation):
            return phases.get(phase, {}).get(operation, {}).get("returned", 0)
        def physics(phase):
            return returned(phase, "mj_step")+returned(phase, "mj_step2")
        calculated = {"prediction_physics_steps": physics("prediction"), "actual_physics_steps": physics("actual"),
            "private_preview_physics_steps": physics("private_preview"),
            "independent_saved_torque_replay_steps": physics("independent_torque_replay"),
            "native_geometry_query_calls": sum(returned(phase, "mj_geomDistance") for phase in phases),
            "qp_solve_calls": sum(returned(phase, "qp_solve") for phase in phases)}
        for key, value in calculated.items():
            # Actual ledgers do not add the separate prediction convenience key.
            required = key != "prediction_physics_steps" or "prediction_physics_steps" in costs
            self.check("cost", label + " scalar " + key, finite(value) and value >= 0
                and (not required or costs.get(key) == value))
        self.check("cost", label + " unchanged controller and research wall scope", costs.get("control_behavior_modified") is False and costs.get("wall_20ms_is_gate") is False)
        return calculated

    def stream(self, stage, task_id, method, budget):
        root = self.run / {"teacher": "teacher_search", "val": "closed_loop_val/search", "test": "test_search"}[stage] / task_id / method
        planning = root / "planning" / task_id
        selection = self.read(planning / "selection.json")
        rows, proposals = self.read(planning / "candidate_registry.json"), self.read(planning / "proposals.json")
        cost = self.read(root / "planning_cost.json")
        task = self.tasks[task_id]
        key = (stage, task_id, method)
        label = "/".join(key)
        expected_identity = {"source_identity_sha256": self.identity_sha, "config_sha256": self.config_sha,
            "run_config_sha256": self.run_config_sha, "task_sha256": digest(task), "model_contract_sha256": task["model_contract_sha256"]}
        self.check("streams", label + " full consumed budget and digest", selection["budget"]["slots_consumed"] == len(rows) == budget
            and selection["budget"]["candidate_budget"] == budget and selection["registry_content_sha256"] == digest(rows))
        self.check("streams", label + " complete proposal/cache equation", selection["budget"]["proposal_attempts"] == len(proposals)
            and len(proposals) == len(rows)+selection["budget"]["cache_hits"])
        self.check("streams", label + " no technical or alternate stop", selection["budget"]["stop_reason"] == "CANDIDATE_BUDGET_EXHAUSTED")
        self.check("streams", label + " Task/runtime identity and actual blindness", selection["task_id"] == task_id
            and selection["task_sha256"] == digest(task) and selection["execution_identity"] == expected_identity
            and selection["selection_reads_final_actual"] is False and selection["formal_actual_validation"] == "NOT_RUN")
        self.check("streams", label + " measured cold, setup included, no shared cross-stream cache", cost["selection_sha256"] == self.sha(planning / "selection.json")
            and cost["measured_mode"] == "cold" and cost["warm_latency"] is None and cost["cross_stream_cache"] is False
            and cost["physics_from_fresh_task_state"] is True
            and close(cost["end_to_end_cold_planning_s"], cost["search_evaluator_selection_s"]+cost.get("setup", {}).get("total_setup_s", 0.), 1e-6))
        workload = Counter()
        for index, row in enumerate(rows):
            row_label = label+"/"+row["candidate_id"]
            rejected = row.get("status") == RAW_REJECTED
            self.check("rows", row_label + " contiguous consumed identity", row["candidate_id"] == f"C{index:02d}" and row["formal_actual_validation"] == "NOT_RUN")
            if rejected:
                self.check("rows", row_label + " raw rejection consumes slot without nominal work", row.get("prediction_rollout_started") is False
                    and row["prediction_steps"] == 0 and row.get("plan") is None and row.get("plan_sha256") is None and row.get("x_m") is None
                    and not (planning / "predictions" / row["candidate_id"]).exists()
                    and "content_key" not in row and "cache_hit_candidate_id" not in row)
            else:
                self.check("rows", row_label + " frozen runtime and original plan", row.get("execution_identity") == expected_identity
                    and row.get("plan_sha256") == digest(row["plan"]))
            values = self.work(row.get("costs") or {}, row_label, rejected)
            self.check("rows", row_label + " main prediction steps", row.get("prediction_steps") == values["prediction_physics_steps"] and 0 <= row["prediction_steps"] <= 13500)
            self.check("rows", row_label + " no tool error", not row.get("tool_error") and row.get("status") != "TOOL_ERROR")
            workload.update(values)
        self.all_rows.extend(rows)
        record = {"stage": stage, "task_id": task_id, "method": method, "budget": budget,
            "rows": rows, "proposals": proposals, "selection": selection, "planning_cost": cost,
            "nominal_status_counts": dict(Counter(row["status"] for row in rows)), "work": dict(workload),
            "raw_rejected": sum(row["status"] == RAW_REJECTED for row in rows),
            "prediction_rollouts_started": sum(bool(row.get("prediction_rollout_started")) for row in rows),
            "positive_step_incomplete_rollouts": sum(0 < row["prediction_steps"] < 13500 for row in rows),
            "cache_hits": selection["budget"]["cache_hits"], "parameter_proposals": len(proposals),
            "cold_request_s": cost["end_to_end_cold_planning_s"]}
        self.check("streams", label + " original rollout count", record["prediction_rollouts_started"] == selection["budget"]["prediction_rollouts_started"])
        if stage != "teacher":
            outer = self.read(root / "outer_process.json")
            events = self.events(root / "command.log")
            terminal = [event for event in events if event.get("event") == "C1_CANDIDATE_TERMINAL"]
            expected_argv = [self.identity["python_executable"], "-B", "-X", "utf8", "-m", "v6_4.search_aware_warmstart_experiment", "_worker", "--run", str(self.run), "--stage", stage, "--task", task_id, "--stream", method]
            self.check("outer_receipts", label + " argv/exit/completion", outer["argv"] == expected_argv and outer["exit_code"] == 0
                and sum(event.get("event") == "C3_STAGE_COMPLETED" and event.get("stage") == "_worker" for event in events) == 1)
            self.check("outer_receipts", label + " measured timer boundaries", finite(outer["elapsed_wall_s"])
                and outer["elapsed_wall_s"] >= record["cold_request_s"] > 0
                and abs((utc(outer["ended_utc"])-utc(outer["started_utc"])).total_seconds()-outer["elapsed_wall_s"]) < .1
                and utc(outer["started_utc"]) <= utc(cost["selection_sealed_utc"]) <= utc(outer["ended_utc"]))
            self.check("outer_receipts", label + " terminal log count and identities", len(terminal) == len(rows)
                and all((event.get("candidate"), event.get("status"), event.get("task"), event.get("slots")) == (row["candidate_id"], row["status"], task_id, i+1) for i,(event,row) in enumerate(zip(terminal,rows))))
            record["outer"] = outer
        self.streams[key] = record
        return record

    def test_initializers(self, task_id, method, stream):
        root = self.run / "test_search" / task_id / method
        task, proposals = self.tasks[task_id], stream["proposals"]
        if method == "R":
            self.check("raw", task_id+"/R no learned provider", not (root / "initializer_proposals.json").exists())
            return
        init = self.read(root / "initializer_proposals.json")
        began = self.read(root / "initializer_started.json")
        self.check("raw", task_id+"/"+method+" once before evaluation", init["generated_once"] is True and init["candidate_quality_read"] is False
            and set(init["proposals"]) == {"1", "3"} and init["content_sha256"] == digest({k:v for k,v in init.items() if k != "content_sha256"})
            and init["task_sha256"] == digest(task) and init["stream_id"] == method and init["stage"] == "test"
            and utc(stream["outer"]["started_utc"]) <= utc(began["started_utc"]) < utc(stream["planning_cost"]["selection_sealed_utc"]))
        self.check("raw", task_id+"/"+method+" frozen TRAIN pool and setup", init["dataset_manifest_sha256"] == self.dataset_sha and stream["planning_cost"]["setup"] == init["timing"])
        if method in ("D", "S"):
            self.check("raw", task_id+"/"+method+" selected checkpoint only", init["checkpoint"] == self.freeze["selected_checkpoints"][method]
                and init["checkpoint_sha256"] == self.freeze["artifacts"][init["checkpoint"]] == self.model_selection["selected_checkpoint_sha256"][method])
        actual_outputs = [p for p in proposals if p.get("initializer_slot") in (1,3)]
        self.check("raw", task_id+"/"+method+" exactly two original proposals", len(actual_outputs) == 2 and {p["initializer_slot"] for p in actual_outputs} == {1,3})
        for slot, preference, family in ((1,"A","v1"),(3,"B","v2")):
            saved = init["proposals"][str(slot)]
            proposal = next(p for p in actual_outputs if p["initializer_slot"] == slot)
            diag, raw = proposal["raw_seed_diagnostics"], saved.get("raw_z_m")
            label = task_id+"/"+method+"/slot"+str(slot)
            mask = self.task_rows[task_id]["search_interval_mask"]
            scalar = scalar_raw(raw, mask)
            # The original exact-cache path canonicalizes a zero residual to
            # family v1. This is inherited plan identity, not raw repair.
            canonical_family = diag.get("canonical_family", saved["family"])
            self.check("raw", label+" raw and source unchanged", all(proposal.get(k) == value for k,value in saved.items() if k != "family")
                and proposal["family"] == canonical_family
                and raw == diag.get("raw_z_m") and diag["requested_preference"] == preference
                and diag["requested_family"] == family and diag["search_interval_mask"] == mask
                and saved["source"] == {"N":"retrieval","D":"diffusion","S":"regression"}[method])
            self.check("raw", label+" no repair/no resample", diag["raw_repaired"] is False and diag["resampled"] is False
                and (method == "N" or (saved["raw_repaired"] is False and saved["resampled"] is False)))
            if diag["raw_legal"] is True:
                self.check("raw", label+" independent scalar legality and original reference gate", scalar["container_12D"] and scalar["finite"]
                    and scalar["inactive_zero"] and scalar["within_original_disks"] and diag["reference_precheck"]["passed"] is True)
                expected_x = [value for i, active in enumerate(mask) if active for value in flat(raw)[2*i:2*i+2]]
                self.check("raw", label+" native active coefficients", proposal["x_m"] == expected_x)
            else:
                rejected = [row for row in stream["rows"] if row.get("initializer_slot") == slot and row["status"] == RAW_REJECTED]
                self.check("raw", label+" illegal occupies exactly one consumed row", len(rejected) == 1 and rejected[0]["raw_z_m"] == raw
                    and rejected[0]["prediction_rollout_started"] is False and rejected[0]["prediction_steps"] == 0 and bool(diag.get("rejection_reason")))
                if not scalar["within_original_disks"] and scalar["finite"]:
                    self.check("raw", label+" scalar disk violation is retained", any(n > .020 for n in scalar["interval_norms_m"]))
            if method in ("D", "S"):
                self.check("raw", label+" fixed model/update/train identities", saved["model"] == method
                    and saved["checkpoint_sha256"] == init["checkpoint_sha256"] and saved["physics_steps"] == 0
                    and saved["training_config_sha256"] == self.training[method]["training_config_sha256"]
                    and saved["checkpoint_update"] == int(Path(init["checkpoint"]).stem.split("_")[-1]))
                self.generation["D_DDIM"] += saved["ddim_sample_units"]
                self.generation["S_forward"] += saved["regression_forward_units"]
                self.check("raw", label+" one supported condition output", saved["ddim_sample_units"] == int(method == "D") and saved["regression_forward_units"] == int(method == "S"))
                if method == "D":
                    self.check("raw", label+" frozen TEST noise metadata", saved["noise_seed"] == self.freeze["seeds"]["test_noise"] == 64324
                        and saved["noise_id"] == f"{digest(task)}/{preference}/{family}/64324" and bool(saved["noise_sha256"]) and bool(saved["masked_initial_noise_sha256"]))
                else:
                    self.check("raw", label+" deterministic S", saved["noise_seed"] is None and saved["noise_id"] is None and "noise_sha256" not in saved)
            elif saved.get("retrieval_source"):
                source = saved["retrieval_source"]
                matches = [label for label in self.labels if label["sample_id"] == source["sample_id"]]
                self.check("retrieval", label+" exact TRAIN label source", len(matches) == 1 and matches[0]["split"] == "train"
                    and matches[0]["task_sha256"] in self.train_task_shas and matches[0]["preference"] == preference
                    and matches[0]["reference_family"] == family and matches[0]["search_interval_mask"] == mask
                    and flat(matches[0]["z_m"]) == flat(raw)
                    and saved["retrieval_rules"]["train_labels_sha256"] == digest(self.labels)
                    and saved["retrieval_rules"]["dataset_manifest_sha256"] == self.dataset_sha)
            self.raw_outputs.append({"task_id":task_id,"method":method,"slot":slot,"source":saved["source"],
                "raw_legal":diag["raw_legal"],"raw_repaired":False,"resampled":False,"scalar_check":scalar,
                "rejection_reason":diag.get("rejection_reason"),"checkpoint_sha256":saved.get("checkpoint_sha256"),
                "noise_sha256":saved.get("noise_sha256"),"masked_initial_noise_sha256":saved.get("masked_initial_noise_sha256")})
        self.check("raw", task_id+"/"+method+" rejected-row total", sum(p["raw_seed_diagnostics"]["raw_legal"] is not True for p in actual_outputs) == stream["raw_rejected"])

    def test_prefixes(self, task_id, method, stream):
        planning = self.run / "test_search" / task_id / method / "planning" / task_id
        for budget in (4,8,12) if method == "R" else (4,8):
            prefix = self.read(planning / f"prefix_{budget:02d}.json")
            body = {key:value for key,value in prefix.items() if key != "snapshot_content_sha256"}
            rows = stream["rows"][:budget]
            count = prefix["budget"]["proposal_attempts"]
            self.check("prefix", task_id+"/"+method+str(budget)+" self seal/exact causal prefix", digest(body) == prefix["snapshot_content_sha256"]
                and prefix["prefix_budget"] == prefix["budget"]["slots_consumed"] == budget
                and prefix["candidate_ids"] == [row["candidate_id"] for row in rows]
                and prefix["registry_content_sha256"] == digest(rows)
                and prefix["proposals_content_sha256"] == digest(stream["proposals"][:count]))
            self.check("prefix", task_id+"/"+method+str(budget)+" costs/time/actual-blind flags", prefix["candidate_costs"] == {r["candidate_id"]:r["costs"] for r in rows}
                and 0 < prefix["elapsed_wall_s"] <= stream["selection"]["elapsed_wall_s"]
                and prefix["protocol_completed"] is True and prefix["later_slots_read"] is False and prefix["selection_reads_final_actual"] is False)
            if budget == stream["budget"]:
                self.check("prefix", task_id+"/"+method+" final selection matches terminal prefix", prefix["preferences"] == stream["selection"]["preferences"])
            if budget < stream["budget"]:
                next_marker = planning / "predictions" / f"C{budget:02d}" / "started.json"
                if self.require([next_marker]):
                    started = self.read(next_marker)
                    self.check("prefix", task_id+"/"+method+str(budget)+" precedes next candidate start",
                        (planning / f"prefix_{budget:02d}.json").stat().st_ctime <= utc(started["started_utc"]).timestamp())
            selected = {}
            for preference in PREFS:
                cid = prefix["preferences"][preference].get("source_candidate_id")
                selected[preference] = next((r for r in rows if r["candidate_id"] == cid), None)
            self.prefixes.append({"task_id":task_id,"method":method,"budget":budget,
                "selected_A_candidate":(selected["A"] or {}).get("candidate_id"),
                "selected_B_candidate":(selected["B"] or {}).get("candidate_id"),
                "predicted_full_A":predicted_full(selected["A"]),
                "predicted_B30":predicted_full(selected["B"]) and finite(((selected["B"] or {}).get("prediction_metrics") or {}).get("d_support")) and selected["B"]["prediction_metrics"]["d_support"] >= .030,
                "prefix_sha256":self.sha(planning / f"prefix_{budget:02d}.json"),"actual_executed":False,
                "scope":"nominal prefix; actual only for separately sealed R8/R12/N8/S8/D8 slots"})

    def actual_slot(self, phase_name, stage, task_id, endpoint, preference, manifest):
        directory = self.run / phase_name / "actual" / task_id / (endpoint+"_"+preference)
        slot_path = directory / "slot.json"
        slot = self.read(slot_path)
        seal = self.seal(directory)
        entry = self.entries[(stage,task_id,endpoint)]
        selection = self.read(entry["selection_path"])
        chosen = selection["preferences"][preference]
        task = self.tasks[task_id]
        plan = chosen.get("selected_plan")
        plan_sha = digest(plan) if plan else None
        alias_identity = digest({"task_sha256":digest(task),"model_contract_sha256":task["model_contract_sha256"],
            "plan_sha256":plan_sha,"initial_history":history(task),"source_identity_sha256":self.identity_sha,
            "config_sha256":self.config_sha,"run_config_sha256":self.run_config_sha}) if plan else None
        binding = {"task_sha256":digest(task),"selection_sha256":entry["selection_sha256"],"plan_sha256":plan_sha,
                   "initial_history_sha256":digest(history(task)),"alias_identity":alias_identity}
        label = stage+"/"+task_id+"/"+endpoint+"/"+preference
        self.check("actual_identity", label+" complete original binding", slot.get("schema") == "v64_c3_actual_logical_slot_v1"
            and (slot["phase"],slot["task_id"],slot["endpoint"],slot["preference"]) == (stage.upper(),task_id,endpoint,preference)
            and slot["actual_binding"] == binding and slot["alias_identity"] == alias_identity
            and slot["selection_sha256"] == entry["selection_sha256"] and slot["task_sha256"] == digest(task)
            and slot["plan_sha256"] == plan_sha and Path(slot["slot_path"]) == slot_path
            and slot["source_candidate_id"] == chosen.get("source_candidate_id"))
        self.check("actual_time", label+" logical slot directory follows whole-phase seal",
            directory.stat().st_ctime > utc(manifest["sealed_utc"]).timestamp())
        for key in ("source_attribution","selected_origin_source","selected_lineage"):
            self.check("actual_identity", label+" selection provenance "+key, slot.get(key) == chosen.get(key))
        alias = slot.get("alias_of_slot")
        record = {"phase":stage,"task_id":task_id,"endpoint":endpoint,"preference":preference,
            "slot_path":str(slot_path),"slot_sha256":self.sha(slot_path),"plan_sha256":plan_sha,
            "source_attribution":slot.get("source_attribution"),"selected_origin_source":slot.get("selected_origin_source"),
            "unique_actual":slot.get("unique_run") is True,"alias_of_slot":alias,"actual_steps":slot["actual_steps"],
            "diagnostic_category":slot["diagnostic_category"],"full_27s_five_gates":False,"quality":None,
            "B30":False,"NO_PLAN":plan is None,"new_work":{key:0 for key in WORK}}
        if not plan:
            self.check("NO_PLAN", label+" exact absent-plan zero-work state", slot["status"] == slot["diagnostic_category"] == "NO_PLAN"
                and slot["actual_steps"] == 0 and slot["unique_run"] is False and slot["entered_actual"] is False
                and slot["quality"] is None and slot["actual_quality"] is None and all(v is None for v in slot["five_gates"].values())
                and slot["full_task_success"] is False and slot["full_27s_success"] is False
                and slot["original_independent_gates_passed"] is False and slot["costs"] == {"no_actual_work":True}
                and alias is None and not (directory / "attempt").exists())
        else:
            self.check("actual_identity", label+" exact selected-plan copy", self.read(directory / "selected_plan.json") == plan)
            method = "R" if endpoint in ("R8","R12") else endpoint[0] if stage == "test" else endpoint
            source_rows = self.streams[(stage,task_id,method)]["rows"][:8] if endpoint == "R8" else self.streams[(stage,task_id,method)]["rows"]
            source_row = next((row for row in source_rows if row["candidate_id"] == chosen.get("source_candidate_id")), None)
            self.check("actual_identity", label+" original selected candidate plan", source_row is not None and source_row.get("plan") == plan and source_row.get("plan_sha256") == plan_sha)
            original_dir = Path(alias).parent if alias else directory
            original = self.read(Path(alias)) if alias else slot
            original_seal = seal
            if alias:
                original_seal = self.seal(original_dir)
                self.check("alias", label+" same phase/Task/plan/config/cold history/source", original_dir.parent == directory.parent
                    and not original.get("alias_of_slot") and original["task_sha256"] == digest(task)
                    and original["plan_sha256"] == plan_sha and original["alias_identity"] == alias_identity
                    and original["actual_binding"]["initial_history_sha256"] == binding["initial_history_sha256"]
                    and self.sha(alias) == slot["alias_of_slot_sha256"])
                self.check("alias", label+" zero new actual work and source predates alias", slot["unique_run"] is False
                    and slot["costs"] == {"alias_zero_new_work":True} and slot["elapsed_wall_s"] == 0.
                    and original_dir.stat().st_ctime <= directory.stat().st_ctime)
                copied = ("status","diagnostic_category","entered_actual","actual_steps","full_task_success","full_27s_success",
                    "original_independent_gates_passed","five_gates","clearance_30mm_met","quality","actual_quality","independent_evaluation_path","trace_path")
                for name in copied:
                    self.check("alias", label+" copied original "+name, slot.get(name) == original.get(name))
            attempt_path = original_dir / "attempt/attempt_result.json"
            if not self.require([attempt_path]):
                return record
            attempt = self.read(attempt_path)
            self.check("actual_identity", label+" original attempt Task/plan/config/source", attempt["task_sha256"] == digest(task)
                and attempt["plan_content_sha256"] == plan_sha and attempt["source_identity_sha256"] == self.identity_sha
                and attempt["qp_config_sha256"] == self.config_sha and attempt["sources_unchanged"] is True
                and attempt["actual_steps"] == slot["actual_steps"]
                and attempt["source_sha256_before"] == attempt["source_sha256_after"] == self.identity["source_sha256"])
            marker = self.read(original_dir / "attempt/slot_started.json")
            self.check("actual_time", label+" all selections sealed before original actual marker", marker["task_sha256"] == digest(task)
                and marker["plan_content_sha256"] == plan_sha and marker["source_identity_sha256"] == self.identity_sha
                and utc(manifest["sealed_utc"]).timestamp() < (original_dir / "attempt/slot_started.json").stat().st_ctime)
            report = self.read(slot["independent_evaluation_path"]) if slot.get("independent_evaluation_path") else {}
            if report:
                self.check("actual_identity", label+" original independent evaluation binding", report["task_sha256"] == digest(task)
                    and report["reference_binding"].get("plan_sha256") == plan_sha and report.get("trace_path") == slot.get("trace_path")
                    and attempt.get("evaluation_sha256") == self.sha(slot["independent_evaluation_path"]) and attempt.get("evaluation") == report)
                evaluation_seal = self.seal(Path(slot["independent_evaluation_path"]).parent)
                for field,name in (("fresh_replay_sha256","fresh_replay.npz"),):
                    if report.get(field):
                        self.check("archive_strings", label+" replay recorded SHA agrees", report[field] == evaluation_seal.get(name))
                for field,name in (("boundary_sha256","interval_boundaries.jsonl"),("rows_sha256","interval_rows.jsonl")):
                    if report.get("independent_interval",{}).get(field):
                        self.check("archive_strings", label+" interval recorded SHA "+field, report["independent_interval"][field] == evaluation_seal.get(name))
            gates = {name:report.get(name,{}).get("passed") for name in GATES}
            gates_passed = report.get("evidence_valid") is True and all(value is True for value in gates.values())
            full = bool(attempt.get("full_task_success") and attempt.get("actual_steps") == 13500 and gates_passed)
            if full:
                self.check("acceptance", label+" complete original 27s report", report.get("complete") is True and report.get("full_task_success") is True
                    and report.get("actual_physics_steps") == 13500 and close(report.get("actual_saved_horizon_s"),27.) and not report.get("errors"))
                requirements = report["task_requirements"]["requirements"]
                declared = {req["point_id"]:req for req in task["requirements"]}
                self.check("gate_details", label+" original requirement set", len(requirements)==len(declared)
                    and {req["point_id"] for req in requirements}==set(declared))
                for requirement in requirements:
                    frozen = declared[requirement["point_id"]]
                    numerical = requirement["eligible_state_count"]>0 and requirement["position_error_m"]<=frozen["position_tolerance_m"]
                    numerical = numerical and requirement["orientation_error_rad"]<=frozen["orientation_tolerance_rad"]
                    numerical = numerical and frozen["time_window_s"][0]-1e-8<=requirement["best_time_s"]<=frozen["time_window_s"][1]+1e-8
                    if frozen["kind"]=="terminal":
                        numerical = numerical and close(requirement["best_time_s"],27.)
                    self.check("gate_details", label+" requirement "+requirement["point_id"], numerical and requirement["passed"] is True
                        and requirement["position_tolerance_m"]==frozen["position_tolerance_m"]
                        and requirement["orientation_tolerance_rad"]==frozen["orientation_tolerance_rad"])
                contract, interval = report["execution_contract"], report["independent_interval"]
                self.check("gate_details", label+" execution contract scalars", all(value is True for value in contract["checks"].values())
                    and max(contract["shared_ramp_max_errors"])<=1e-9 and contract["torque_saturations"]==0)
                self.check("gate_details", label+" interval counts/mismatch", interval["mismatch_count"]==0
                    and interval["boundary_count"]==1351 and interval["row_count"]>0)
                geometry = report["native_geometry"]
                target, body = geometry["robot_target_500hz"], geometry["whole_body"]
                self.check("gate_details", label+" saved discrete native geometry scope", target["state_count"]==13501
                    and target["below_5mm_states"]==target["negative_states"]==target["truncated_query_count"]==0
                    and target["minimum_m"]>=.005 and body["feasible"] is True
                    and body["violation_count"]==body["negative_distance_query_count"]==body["truncated_query_count"]==0
                    and body["minimum_clearance"]>=.005 and body["supplied_sample_count"]==1351
                    and body["adaptive_subdivisions"]==4 and geometry["continuous_time_certified"] is False)
                reference = report["reference_binding"]
                residuals = list(reference["component_maximum_absolute_residual"].values())+list(reference["retained_field_maximum_residual"].values())
                self.check("gate_details", label+" original reference residuals/counts", all(finite(value) and abs(value)<=1e-9 for value in residuals)
                    and reference["planning_inputs_bound"]==1350 and reference["generated_continuum_and_posture_poststep_samples_bound"]==13500
                    and reference["future_actual_trace_is_online_input"] is False)
            failure = attempt.get("execution_failure") or {}
            category = ("TOOL_ERROR" if attempt.get("pipeline_failure") or (failure and failure.get("type") != "UncertifiedExecutionError")
                else "ACTUAL_PRECHECK_REJECTED" if attempt.get("status") == "REFERENCE_PRECHECK_REJECTED"
                else "FULL_TASK_AND_FIVE_GATES_PASSED" if full else "ACTUAL_ZERO_STEP_REFUSAL" if not attempt.get("actual_steps") else "ACTUAL_FAILURE")
            self.check("acceptance", label+" independently derived outcome and five gates", slot["diagnostic_category"] == category
                and slot["five_gates"] == gates and slot["original_independent_gates_passed"] == gates_passed
                and slot["full_task_success"] == slot["full_27s_success"] == full)
            self.check("acceptance", label+" no hidden tool failure", category != "TOOL_ERROR" and not slot.get("tool_error"))
            quality_doc = self.read(original_dir / "quality/quality.json") if (original_dir / "quality/quality.json").is_file() else None
            quality = quality_doc.get("metrics") if quality_doc else None
            self.check("quality", label+" quality equals original saved metadata", quality == slot.get("quality") == slot.get("actual_quality"))
            if full:
                self.check("quality", label+" full quality evidence finite and independently bound", quality is not None
                    and all(finite(quality.get(name)) for name in ("I_support","L_full","d_support"))
                    and quality.get("native_state_count") == 13501 and close(quality.get("saved_horizon_s"),27.)
                    and quality.get("state_source") == "independent_saved_torque_actual_replay"
                    and quality.get("clearance_status") == "MEASURED" and quality.get("clearance_is_censored_lower_bound") is False
                    and quality["L_full"] == report["metrics"]["actual_tip_path_length_m"]["continuum"])
            if quality:
                self.check("quality", label+" recorded trace/replay SHA identities", quality_doc["trace_sha256"] == attempt.get("trace_sha256") == report.get("trace_sha256")
                    and quality_doc.get("replay_sha256") == report.get("fresh_replay_sha256"))
                trace = Path(attempt["trace_path"]).resolve()
                self.check("archive_strings", label+" trace remains inside original evidence", original_dir.resolve() in trace.parents)
                if original_dir.resolve() in trace.parents:
                    self.check("archive_strings", label+" trace declared SHA agrees with original slot seal",
                        original_seal.get(trace.relative_to(original_dir.resolve()).as_posix()) == attempt["trace_sha256"])
                self.check("quality", label+" B30 declaration", slot.get("clearance_30mm_met") == (finite(quality.get("d_support")) and quality["d_support"] >= .030))
                predicted = chosen.get("prediction_metrics") or {}
                difference = {name:quality[name]-predicted[name] for name in ("I_support","L_full","d_support") if finite(quality.get(name)) and finite(predicted.get(name))}
                self.check("quality", label+" exact prediction/actual deltas", slot.get("prediction_actual_difference") == difference
                    and slot.get("prediction_actual_consistent") == bool(len(difference)==3 and full and gates_passed and all(abs(v)<=1e-9 for v in difference.values())))
            if not alias:
                values = self.work(slot.get("costs") or {}, label)
                # Pre-actual refusal can have zero unique_run but retain measured
                # bookkeeping. Count only unique runs toward main actual work.
                record["new_work"] = values
                self.check("actual_identity", label+" original runner start flag", slot["unique_run"] == bool(attempt.get("actual_runner_started")))
                self.check("cost", label+" actual integration recorded once", values["actual_physics_steps"] == slot["actual_steps"])
                metadata_path = original_dir / "attempt/actual/run_metadata.json"
                if attempt.get("actual_runner_started"):
                    metadata = self.read(metadata_path)
                    self.check("actual_identity", label+" frozen runtime configuration and model", metadata["source"]["git_commit"] == PRODUCER
                        and metadata["run_config"] == self.run_config and metadata["qp_config"] == self.config
                        and metadata["model_identity"]["runtime_contract_sha256"] == task["model_contract_sha256"]
                        and metadata["runtime_identity"]["execution_mode"] == "research_simulation"
                        and metadata["runtime_identity"]["controller_version"] == "v6_2_research_simulation_bounded_interval_pcc"
                        and metadata["runtime_identity"]["servo_law_version"] == "b2_implicitfast_compensated_torque_v1")
                    self.check("actual_time", label+" original start after all-selection seal", utc(metadata["started_utc"]) > utc(manifest["sealed_utc"]))
                    record["started_utc"], record["finished_utc"] = metadata["started_utc"], metadata.get("finished_utc")
            record.update(full_27s_five_gates=full,quality={name:quality.get(name) for name in QUALITY} if quality else None,
                B30=bool(full and preference=="B" and quality and quality.get("d_support") is not None and quality["d_support"]>=.030))
        self.slots[(stage,task_id,endpoint,preference)] = record
        self.all_slots.append(slot)
        return record

    def phase_receipt(self, name, required=True):
        path = self.run / "command_logs" / ("phase_"+name+".json")
        if not path.exists() and not required:
            return None
        receipt = self.read(path)
        argv = receipt["argv"]
        run_argument = argv[argv.index("--run")+1]
        resolved = (ROOT / run_argument).resolve() if not Path(run_argument).is_absolute() else Path(run_argument).resolve()
        self.check("phase_receipts", name+" original successful no-retry command", receipt["stage"] == name and receipt["exit_code"] == 0
            and receipt.get("automatic_retry") is False and name in argv and resolved == self.run
            and argv[0] == self.identity["python_executable"])
        events = self.events(self.run / "command_logs" / (name+".log"))
        self.check("phase_receipts", name+" original completion event", sum(e.get("event")=="C3_STAGE_COMPLETED" and e.get("stage")==name for e in events)==1)
        return receipt

    def audit(self, report_summary=None):
        base = [self.run/name for name in ("plan.json","source_identity.json","frozen_execution_config.json","frozen_run_config.json",
            "model_selection.json","model_freeze.json","actual_complete.json","validation/protocol_validation.json",
            "frozen_test/sealed_selections/all_selections.json","closed_loop_val/sealed_selections/all_selections.json")]
        base += [self.run/"command_logs"/("phase_"+stage+".json") for stage in ("closed-loop-val","freeze-models","test-search","execute-test","validate")]
        if not self.require(base):
            return
        self.plan, self.identity = self.read(self.run/"plan.json"), self.read(self.run/"source_identity.json")
        self.identity_sha = self.sha(self.run/"source_identity.json")
        self.config, self.run_config = self.read(self.run/"frozen_execution_config.json"), self.read(self.run/"frozen_run_config.json")
        self.config_sha, self.run_config_sha = self.sha(self.run/"frozen_execution_config.json"), self.sha(self.run/"frozen_run_config.json")
        self.task_rows = {row["task_id"]:row for row in self.plan["tasks"]}
        stage_tasks = {stage:[row["task_id"] for row in self.plan["tasks"] if row["split"]==stage] for stage in ("train","val","test")}
        schedule = [(task,method) for i,task in enumerate(stage_tasks["test"]) for method in METHODS[i:]+METHODS[:i]]
        required = []
        for task,method in schedule:
            root = self.run/"test_search"/task/method
            required += [root/name for name in ("outer_process.json","planning_cost.json","command.log")]
            required += [root/"planning"/task/name for name in ("selection.json","candidate_registry.json","proposals.json","optimizer_inputs.json")]
            required += [root/"planning"/task/f"prefix_{budget:02d}.json" for budget in ((4,8,12) if method=="R" else (4,8))]
            if method != "R":
                required += [root/name for name in ("initializer_proposals.json","initializer_started.json")]
        for stage,phase,endpoints in (("val","closed_loop_val",("R12","D250","S250","D4000","S4000")),("test","frozen_test",ENDPOINTS)):
            for task in stage_tasks[stage]:
                required += [self.run/phase/"actual"/task/(endpoint+"_"+preference)/name
                    for endpoint in endpoints for preference in PREFS for name in ("slot.json","manifest.json")]
        if not self.require(required):
            return
        self.check("identity", "original producer/schema and 430 declared sources", self.identity["algorithm_producer_commit"] == self.identity["git_head"] == PRODUCER
            and self.plan["algorithm_producer_commit"] == PRODUCER and len(self.identity["source_sha256"]) == 430)
        for name in SOURCES:
            self.check("identity", name+" current reviewed source SHA", self.sha(ROOT/"v6_4"/name)==self.identity["source_sha256"]["v6_4/"+name])
        self.check("protocol", "exact external split and mother counts", {k:len(v) for k,v in stage_tasks.items()}=={"train":6,"val":2,"test":4}
            and {stage:len({self.task_rows[t]["mother_id"] for t in ids}) for stage,ids in stage_tasks.items()}=={"train":3,"val":1,"test":2})
        self.check("protocol", "original budget maxima", self.plan["budget_limits"] == LIMITS)
        self.tasks = {task:self.read(self.run/"frozen_tasks"/task/"task.json") for task in self.task_rows}
        self.check("identity", "Task canonical bindings", all(digest(task)==self.task_rows[name]["task_sha256"] for name,task in self.tasks.items()))
        self.train_task_shas = {digest(self.tasks[t]) for t in stage_tasks["train"]}
        self.labels = self.read(self.run/"dataset/labels.json")
        self.dataset_sha = self.sha(self.run/"dataset/manifest.json")
        self.check("retrieval", "shared final pool is TRAIN-only", bool(self.labels) and all(label["split"]=="train" and label["task_sha256"] in self.train_task_shas for label in self.labels))
        self.training = {model:self.read(self.run/f"models/{model}/training_report.json") for model in ("D","S")}
        self.check("training", "D/S once4000 paired128000 exposures", all(report["optimizer_updates_total"]==4000 and report["sample_exposures_total"]==128000 and report["checkpoint_updates"]==[250,4000] for report in self.training.values())
            and self.training["D"]["paired_reference_indices_sha256"]==self.training["S"]["paired_reference_indices_sha256"])
        self.freeze, self.model_selection = self.read(self.run/"model_freeze.json"), self.read(self.run/"model_selection.json")
        freeze_receipt = self.phase_receipt("freeze-models")
        val_receipt = self.phase_receipt("closed-loop-val")
        self.check("model_freeze", "completed VAL then frozen model command/file", utc(val_receipt["ended_utc"])<=utc(freeze_receipt["started_utc"])
            <=utc(self.freeze["frozen_utc"])<=utc(freeze_receipt["ended_utc"]))
        self.check("model_freeze", "only frozen selected D/S checkpoints", self.freeze["selected_checkpoints"]==self.model_selection["selected_checkpoint_files"]
            and self.model_selection["test_read"] is False and self.model_selection["closed_loop_val_completed"] is True)
        for path,expected in self.freeze["artifacts"].items():
            p=Path(path)
            if not self.require([p]):
                continue
            if p.suffix==".json" and p.stat().st_size<=MAX_BYTES:
                self.check("model_freeze", path+" current small bytes", self.sha(p)==expected)
            else:
                self.deferred[path]={"recorded_sha256":expected,"size_bytes":p.stat().st_size,"reason":"frozen weight/NPZ/large data current-byte verification deferred"}
        for model in ("D","S"):
            selected=self.freeze["selected_checkpoints"][model]
            self.check("model_freeze", model+" selected checkpoint SHA binding", self.freeze["artifacts"][selected]==self.model_selection["selected_checkpoint_sha256"][model]
                ==self.training[model]["checkpoint_sha256"][str(int(Path(selected).stem.split("_")[-1]))])
        val_manifest=self.phase_manifest("closed_loop_val","VAL",stage_tasks["val"],("R12","D250","S250","D4000","S4000"),{"source_identity_sha256":self.identity_sha})
        test_manifest=self.phase_manifest("frozen_test","TEST",stage_tasks["test"],ENDPOINTS,{"source_identity_sha256":self.identity_sha,"model_freeze_sha256":self.sha(self.run/"model_freeze.json")})
        for task in stage_tasks["train"]:
            for method in ("T_local","T_transfer"):
                self.stream("teacher",task,method,8)
        for task in stage_tasks["val"]:
            for method in ("R","D250","S250","D4000","S4000"):
                self.stream("val",task,method,12 if method=="R" else 8)
        previous_end=None
        for task,method in schedule:
            stream=self.stream("test",task,method,12 if method=="R" else 8)
            self.check("TEST_order", task+"/"+method+" rotated sequential order and model freeze before search", utc(self.freeze["frozen_utc"])<utc(stream["outer"]["started_utc"])
                and utc(freeze_receipt["ended_utc"])<=utc(stream["outer"]["started_utc"])
                and (previous_end is None or previous_end<=utc(stream["outer"]["started_utc"])))
            previous_end=utc(stream["outer"]["ended_utc"])
            self.check("TEST_order", task+"/"+method+" outer ends before all-selection seal", previous_end<utc(test_manifest["sealed_utc"]))
            self.test_initializers(task,method,stream)
            self.test_prefixes(task,method,stream)
            proposals=stream["proposals"]
            for slot in (0,2):
                content={key:proposals[slot].get(key) for key in ("family","x_m","source","origin_source")}
                common=self.observations.setdefault((task,slot),content)
                self.check("common_seeds", task+"/"+method+str(slot)+" same common seed", content==common and content["source"]=="initial")
            self.check("common_seeds", task+"/"+method+" zero and12mm", proposals[0]["family"]=="v1" and all(v==0 for v in proposals[0]["x_m"])
                and proposals[2]["family"]=="v2" and close(math.sqrt(sum(v*v for v in proposals[2]["x_m"])),.012))
        self.check("TEST_totals", "16 requests,144 rows,36 prefixes and8D/8S", sum(key[0]=="test" for key in self.streams)==16
            and sum(len(s["rows"]) for key,s in self.streams.items() if key[0]=="test")==144 and len(self.prefixes)==36
            and self.generation=={"D_DDIM":8,"S_forward":8})
        d_noise=[record["noise_sha256"] for record in self.raw_outputs if record["method"]=="D"]
        self.check("TEST_totals", "eight distinct saved D condition-noise identities", len(d_noise)==8 and len(set(d_noise))==8)
        for stage,phase,manifest,endpoints in (("val","closed_loop_val",val_manifest,("R12","D250","S250","D4000","S4000")),("test","frozen_test",test_manifest,ENDPOINTS)):
            for task in stage_tasks[stage]:
                for index,endpoint in enumerate(endpoints):
                    for pref_index,preference in enumerate(PREFS):
                        self.actual_slot(phase,stage,task,endpoint,preference,manifest)
                        slot=self.read(self.run/phase/"actual"/task/(endpoint+"_"+preference)/"slot.json")
                        self.check("actual_identity", stage+"/"+task+" local logical ordinal", slot["logical_slot"]==2*index+pref_index)
        if self.pending:
            return
        # Model-selection evidence cannot silently drift after TEST starts.
        val_hashes={str(self.run/"closed_loop_val/actual"/t/(e+"_"+p)/"slot.json"):self.sha(self.run/"closed_loop_val/actual"/t/(e+"_"+p)/"slot.json")
            for t in stage_tasks["val"] for e in ("R12","D250","S250","D4000","S4000") for p in PREFS}
        self.check("model_freeze", "scored VAL current slots bound exactly", self.model_selection["actual_slot_files"]==val_hashes
            and self.model_selection["phase_manifest_sha256"]==self.sha(self.run/"closed_loop_val/sealed_selections/all_selections.json"))
        self.summarize(stage_tasks,schedule,test_manifest)
        if report_summary:
            self.compare_report(Path(report_summary))

    def summarize(self, stage_tasks, schedule, test_manifest):
        test_slots=[s for key,s in self.slots.items() if key[0]=="test"]
        endpoint_rows=[]
        for endpoint in ENDPOINTS:
            for preference in PREFS:
                group=[self.slots[("test",task,endpoint,preference)] for task in stage_tasks["test"]]
                refs=[self.slots[("test",task,"R12",preference)] for task in stage_tasks["test"]]
                near_values=[near(slot,reference,preference) for slot,reference in zip(group,refs)]
                raw_count=sum(self.streams[("test",task,endpoint[0])]["raw_rejected"] for task in stage_tasks["test"])
                endpoint_rows.append({"endpoint":endpoint,"preference":preference,"denominator_tasks":4,"logical_slots_present":len(group),"missing_slots":4-len(group),
                    "full_27s_five_gates":sum(s["full_27s_five_gates"] for s in group),"actual_B30":sum(s["B30"] for s in group) if preference=="B" else None,
                    "NO_PLAN":sum(s["NO_PLAN"] for s in group),"raw_rejected_shared_stream":raw_count,
                    "actual_pre_rejected":sum(s["diagnostic_category"] in ("ACTUAL_PRECHECK_REJECTED","ACTUAL_ZERO_STEP_REFUSAL") for s in group),
                    "actual_failed":sum(s["diagnostic_category"]=="ACTUAL_FAILURE" for s in group),"tool_error":sum(s["diagnostic_category"]=="TOOL_ERROR" for s in group),
                    "unique_actual_attempts":sum(s["unique_actual"] for s in group),"unique_positive_step_actual":sum(s["unique_actual"] and s["actual_steps"]>0 for s in group),
                    "actual_aliases":sum(bool(s["alias_of_slot"]) for s in group),"near_R12_count":sum(v is True for v in near_values),
                    "R12_qualified_reference_count":sum(qualified(s,preference) for s in refs),"near_R12_NA":sum(not qualified(s,preference) for s in refs),
                    "near_R12_method_quality_NA":sum(qualified(r,preference) and s["full_27s_five_gates"] and not qualified(s,preference) for s,r in zip(group,refs)),
                    **{name:sum(s["source_attribution"]==name for s in group) for name in ("direct_learning_seed","learning_seed_descendant","rule_seed_or_descendant")}})
        partition=Counter()
        for slot in test_slots:
            category=("alias" if slot["alias_of_slot"] else "NO_PLAN" if slot["NO_PLAN"] else
                "unique_actual_success" if slot["unique_actual"] and slot["full_27s_five_gates"] else
                "unique_actual_failure" if slot["unique_actual"] else "pre_actual_failure_or_refusal")
            partition[category]+=1
            slot["disjoint_logical_category"]=category
        self.check("actual_totals", "40 disjoint logical slots; failed unique runs not double-counted", len(test_slots)==40 and sum(partition.values())==40)
        receipt=self.read(self.run/"actual_complete.json")
        test_hashes={Path(s["slot_path"]).relative_to(self.run).as_posix():s["slot_sha256"] for s in test_slots}
        self.check("actual_totals", "current exact actual-complete receipt", receipt["logical_slots"]==40 and receipt["unique_actual"]==sum(s["unique_actual"] for s in test_slots) and receipt["slot_hashes"]==test_hashes)
        reservations=Counter()
        ledger=[]
        for path in sorted((self.run/"budget_ledger").glob("*.json")):
            row=self.read(path)
            self.check("budget", str(path)+" immutable unit identity", row["category"] in LIMITS and type(row["count"]) is int and row["count"]>0
                and row["status"]=="CONSUMED_NO_SILENT_RETRY" and path.stem==digest([row["category"],row["unit_id"]]))
            reservations[row["category"]]+=row["count"]
            ledger.append(row)
        for category,limit in LIMITS.items():
            self.check("budget", category+" maximum", reservations[category]<=limit)
        totals={key:reservations[key] for key in LIMITS}
        totals.update(candidate_slots_total=sum(totals[k] for k in ("teacher_candidate_slots","val_candidate_slots","test_candidate_slots")),
            actual_slots_total=totals["val_actual_slots"]+totals["test_actual_slots"],
            ddim_samples_total=sum(totals[k] for k in ("val_ddim_samples","test_ddim_samples","smoke_ddim_samples","diagnostic_ddim_samples")))
        self.check("budget", "exact required stage reservations", all(totals[key]==LIMITS[key] for key in ("teacher_candidate_slots","val_candidate_slots","test_candidate_slots","val_actual_slots","test_actual_slots","val_ddim_samples","test_ddim_samples","D_training_runs","S_training_runs"))
            and totals["candidate_slots_total"]==328 and totals["actual_slots_total"]==60 and totals["ddim_samples_total"]<=32)
        candidate_counts={stage:sum(len(s["rows"]) for key,s in self.streams.items() if key[0]==stage) for stage in ("teacher","val","test")}
        stream_counts={stage:sum(key[0]==stage for key in self.streams) for stage in ("teacher","val","test")}
        main_prediction=sum(row.get("prediction_steps",0) for row in self.all_rows)
        main_actual=sum(slot.get("actual_steps",0) for slot in self.all_slots if slot.get("unique_run"))
        tool_errors=sum(bool(row.get("tool_error")) for row in [*self.all_rows,*self.all_slots])
        recomputed_validation={"all_terminal":tool_errors==0,"candidate_slots":candidate_counts,
            "actual_logical_slots":{"val":sum(key[0]=="val" for key in self.slots),"test":len(test_slots)},
            "budget_reservations":totals,"main_prediction_physics_steps":main_prediction,"main_actual_physics_steps":main_actual,"tool_error_count":tool_errors}
        saved=self.read(self.run/"validation/protocol_validation.json")
        self.check("terminal", "independent12/10/16 streams and96/88/144 rows", stream_counts=={"teacher":12,"val":10,"test":16} and candidate_counts=={"teacher":96,"val":88,"test":144})
        self.check("terminal", "main physical budgets and60 logical actual", main_prediction<=4428000 and main_actual<=810000 and len(self.slots)==60)
        for key,value in recomputed_validation.items():
            self.check("terminal", "saved validation independently reconstructed "+key, saved.get(key)==value)
        for task in stage_tasks["test"]:
            for method in METHODS:
                unit=task+":"+method
                found=[r for r in ledger if r["category"]=="test_candidate_slots" and r["unit_id"]==unit]
                self.check("budget", unit+" exact candidate reservation", len(found)==1 and found[0]["count"]==(12 if method=="R" else 8)
                    and found[0]["metadata"]["task_sha256"]==digest(self.tasks[task]))
            found=[r for r in ledger if r["category"]=="test_actual_slots" and r["unit_id"]==task]
            self.check("budget", task+" ten actual slots after phase seal", len(found)==1 and found[0]["count"]==10 and found[0]["metadata"]["endpoints"]==list(ENDPOINTS)
                and utc(found[0]["utc"])>=utc(test_manifest["sealed_utc"]))
            found=[r for r in ledger if r["category"]=="test_ddim_samples" and r["unit_id"]==task+":D"]
            self.check("budget", task+" two formal D samples", len(found)==1 and found[0]["count"]==2 and found[0]["metadata"]=={"noise_seed":64324,"DDIM_steps":20})
        phase_receipts={name:self.phase_receipt(name) for name in ("test-search","execute-test","validate")}
        self.check("phase_receipts", "TEST search completion before execution before validation", utc(phase_receipts["test-search"]["ended_utc"])<=utc(phase_receipts["execute-test"]["started_utc"])
            and utc(phase_receipts["execute-test"]["ended_utc"])<=utc(phase_receipts["validate"]["started_utc"]))
        execution_events=self.events(self.run/"command_logs/execute-test.log")
        actual_events=[e for e in execution_events if e.get("event")=="C3_FINAL_ACTUAL"]
        expected_events={(t,e,p) for t in stage_tasks["test"] for e in ENDPOINTS for p in PREFS}
        self.check("phase_receipts", "40 exact logical terminal events", len(actual_events)==40 and {(e["task"],e["endpoint"],e["preference"]) for e in actual_events}==expected_events)
        for event in actual_events:
            slot=self.slots[("test",event["task"],event["endpoint"],event["preference"])]
            self.check("phase_receipts", "event matches original saved logical result", (event["steps"],event["unique"],event["category"])==(slot["actual_steps"],slot["unique_actual"],slot["diagnostic_category"]))
        starts=[e for e in execution_events if e.get("event")=="B2_ACTUAL_STARTED"]
        terminals=[e for e in execution_events if e.get("event")=="B2_ACTUAL_TERMINAL"]
        started_ids={f"TEST_{s['task_id']}_{s['endpoint']}_{s['preference']}" for s in test_slots if s["unique_actual"]}
        attempt_ids={f"TEST_{s['task_id']}_{s['endpoint']}_{s['preference']}" for s in test_slots if not s["alias_of_slot"] and not s["NO_PLAN"]}
        self.check("phase_receipts", "exact unique feedback start events; no hidden retry", len(starts)==len(started_ids)
            and {e["slot_id"] for e in starts}==started_ids)
        self.check("phase_receipts", "one original terminal per nonalias planned attempt", len(terminals)==len(attempt_ids)
            and {e["slot_id"] for e in terminals}==attempt_ids)
        stage_work={stage:{name:sum(stream["work"][name] for key,stream in self.streams.items() if key[0]==stage) for name in WORK} for stage in ("teacher","val","test")}
        actual_work={stage:{name:sum(slot["new_work"][name] for key,slot in self.slots.items() if key[0]==stage and not slot["alias_of_slot"] and not slot["NO_PLAN"]) for name in WORK} for stage in ("val","test")}
        stage_accounting={}
        for stage in ("teacher","val","test"):
            streams=[stream for key,stream in self.streams.items() if key[0]==stage]
            actual=[slot for key,slot in self.slots.items() if key[0]==stage]
            counts=Counter()
            for stream in streams:
                counts.update(stream["nominal_status_counts"])
            stage_accounting[stage]={"streams_expected":len(streams),"streams_complete":len(streams),
                "parameter_constructions":sum(s["parameter_proposals"] for s in streams),
                "consumed_candidate_slots":sum(len(s["rows"]) for s in streams),
                "prediction_rollouts_started":sum(s["prediction_rollouts_started"] for s in streams),
                "legal_exact_cache_hits":sum(s["cache_hits"] for s in streams),"raw_rejected":sum(s["raw_rejected"] for s in streams),
                "nominal_failed_rollouts":sum(row.get("prediction_rollout_started") is True and not predicted_full(row) for s in streams for row in s["rows"]),
                "nominal_positive_step_incomplete_rollouts":sum(s["positive_step_incomplete_rollouts"] for s in streams),
                "nominal_status_counts":dict(counts),"summed_cold_planning_service_s":sum(s["cold_request_s"] for s in streams),
                "summed_outer_process_wall_s":sum(s["outer"]["elapsed_wall_s"] for s in streams) if all("outer" in s for s in streams) else None,
                "actual_logical_slots_present":len(actual),"unique_actual_attempts":sum(s["unique_actual"] for s in actual),
                "actual_aliases":sum(bool(s["alias_of_slot"]) for s in actual),"actual_NO_PLAN":sum(s["NO_PLAN"] for s in actual),
                "main_actual_physics_steps":sum(s["actual_steps"] for s in actual if s["unique_actual"]),
                **{name:stage_work[stage][name] for name in WORK if name!="actual_physics_steps"}}
        planning_rows=[]
        first_hits=[]
        for task in stage_tasks["test"]:
            for endpoint in ENDPOINTS:
                stream=self.streams[("test",task,endpoint[0])]
                budget=8 if endpoint!="R12" else 12
                rows=stream["rows"][:budget]
                prefix=self.read(self.run/"test_search"/task/endpoint[0]/"planning"/task/f"prefix_{budget:02d}.json")
                work={name:sum((row.get("costs") or {}).get(name,0) for row in rows) for name in WORK}
                planning_rows.append({"task_id":task,"endpoint":endpoint,**work,"consumed_candidate_slots":len(rows),
                    "raw_rejected":sum(row["status"]==RAW_REJECTED for row in rows),
                    "end_to_end_cold_planning_s":None if endpoint=="R8" else stream["cold_request_s"],
                    "outer_process_wall_s":None if endpoint=="R8" else stream["outer"]["elapsed_wall_s"],
                    "optimizer_elapsed_s":prefix["elapsed_wall_s"] if endpoint=="R8" else stream["selection"]["elapsed_wall_s"],
                    "planning_time_scope":"measured_optimizer_prefix_only; cold_request_total_not_measured" if endpoint=="R8" else "measured_cold_request_total"})
                for preference in PREFS:
                    reference=self.slots[("test",task,"R12",preference)]
                    applicable=qualified(reference,preference)
                    hit=None
                    if applicable:
                        q=reference["quality"]
                        for index,row in enumerate(rows,1):
                            predicted=row.get("prediction_metrics") or {}
                            if not predicted_full(row) or not all(finite(predicted.get(k)) for k in (("I_support","L_full") if preference=="A" else ("d_support","L_full"))):
                                continue
                            if predicted["L_full"]<=q["L_full"]+.005 and (predicted["I_support"]<=q["I_support"]+.001 if preference=="A" else predicted.get("clearance_status")=="MEASURED" and predicted["d_support"]>=.030):
                                hit=index
                                break
                    first_hits.append({"task_id":task,"endpoint":endpoint,"preference":preference,
                        "status":"N/A_R12_NO_QUALIFIED_ACTUAL" if not applicable else "HIT" if hit else "RIGHT_CENSORED",
                        "slot":hit,"right_censored_budget":budget if applicable and hit is None else None,
                        "sort_encoding":hit if hit else budget+1 if applicable else None,"encoding_is_observed_hit":hit is not None})
        comparisons=[]
        for method,comparator in (("D8","R8"),("D8","R12"),("D8","N8"),("D8","S8"),("N8","R12"),("S8","R12")):
            lost,quality_missing,quality_lost=[],[],[]
            for task in stage_tasks["test"]:
                for pref in PREFS:
                    a,b=self.slots[("test",task,method,pref)],self.slots[("test",task,comparator,pref)]
                    if b["full_27s_five_gates"] and not a["full_27s_five_gates"]:
                        lost.append({"task_id":task,"preference":pref})
                    if b["full_27s_five_gates"] and not qualified(b,pref):
                        quality_missing.append({"task_id":task,"preference":pref})
                    if qualified(b,pref) and near(a,b,pref) is not True:
                        quality_lost.append({"task_id":task,"preference":pref,"near":near(a,b,pref)})
            comparisons.append({"method":method,"comparator":comparator,"capability_lost":lost,"missing_comparator_quality":quality_missing,"near_quality_lost_or_missing":quality_lost,
                "cost_reductions_all_tasks_including_failed_requests":{name:self.cost_reduction(planning_rows,method,comparator,name) for name in ("end_to_end_cold_planning_s","prediction_physics_steps","private_preview_physics_steps","native_geometry_query_calls")},
                "raw_illegal_method":sum(row["raw_rejected"] for row in planning_rows if row["endpoint"]==method),
                "raw_illegal_comparator":sum(row["raw_rejected"] for row in planning_rows if row["endpoint"]==comparator),
                "benefit_not_adjudicated":True,"reason":"Raw rejections, nominal early failure, quality preservation and actual coverage must be considered before any savings claim"})
        self.results={"predeclared_TEST_schedule":schedule,"stream_counts":stream_counts,"candidate_counts":candidate_counts,
            "TEST_raw_outputs":self.raw_outputs,"TEST_generation":dict(self.generation),"TEST_prefixes":self.prefixes,
            "TEST_logical_partition_disjoint":dict(partition),"TEST_unique_actual_total":partition["unique_actual_success"]+partition["unique_actual_failure"],
            "endpoint_summary":endpoint_rows,"actual_quality_records":test_slots,"paired_capability_quality_checks":comparisons,
            "main_prediction_physics_steps":main_prediction,"main_actual_physics_steps":main_actual,
            "candidate_work_by_stage":stage_work,"actual_new_work_by_stage":actual_work,
            "independently_recomputed_stage_accounting":stage_accounting,
            "planning_costs_by_task_and_endpoint":planning_rows,"first_near_R12_nominal_hits":first_hits,
            "budget_totals":totals,"independently_recomputed_protocol_validation":recomputed_validation,
            "streams":[{k:v for k,v in stream.items() if k not in ("rows","proposals","selection","planning_cost")} for stream in self.streams.values()],
            "R8_cost_scope":"Original optimizer prefix time only; no separately measured cold request total. R12 cold cost is counted once per R request, never repeated for A/B or R8.",
            "quality_scope":"Only full27s/five-gate actual endpoints earn success/near-quality/B30 credit. Partial quality is retained as diagnostic and cannot win over full execution."}

    @staticmethod
    def cost_reduction(planning,method,comparator,key):
        a=[row[key] for row in planning if row["endpoint"]==method]
        b=[row[key] for row in planning if row["endpoint"]==comparator]
        total_a=sum(a) if len(a)==4 and all(finite(value) for value in a) else None
        total_b=sum(b) if len(b)==4 and all(finite(value) for value in b) else None
        return {"method_total":total_a,"comparator_total":total_b,
            "reduction_fraction":(total_b-total_a)/total_b if total_a is not None and total_b is not None and total_b>0 else None}

    def compare_report(self, path):
        if not self.require([path]):
            return
        summary=self.read(path)
        indexed={(r["endpoint"],r["preference"]):r for r in summary.get("endpoint_summary",[])}
        self.check("report_numbers", "exact ten method/preference rows", set(indexed)=={(e,p) for e in ENDPOINTS for p in PREFS})
        for row in self.results["endpoint_summary"]:
            reported=indexed.get((row["endpoint"],row["preference"]),{})
            for key,value in row.items():
                self.check("report_numbers", row["endpoint"]+"/"+row["preference"]+" independently recomputed "+key, reported.get(key)==value)
        for field,expected in (("TEST_actual_logical_slots",40),("TEST_unique_actual_attempts",self.results["TEST_unique_actual_total"]),
                               ("TEST_actual_aliases",self.results["TEST_logical_partition_disjoint"].get("alias",0))):
            self.check("report_numbers", field+" independently recomputed", summary.get(field)==expected)
        reported_stages=summary.get("cost_accounting",{}).get("stages",{})
        for stage,recomputed in self.results["independently_recomputed_stage_accounting"].items():
            for key,value in recomputed.items():
                reported=reported_stages.get(stage,{}).get(key)
                equal=close(reported,value,1e-6) if isinstance(value,float) else reported==value
                self.check("report_numbers", stage+" independently recomputed cost/accounting "+key,equal)
        self.results["report_comparison"]={"path":str(path),"sha256":self.sha(path),"completion_self_claims_used_as_evidence":False,
            "scope":"independently recomputed endpoint/count fields only; this script does not adopt report benefit or completion declarations"}

    def finish(self, output):
        output=Path(output).resolve()
        docs=(ROOT/"docs").resolve()
        if docs not in output.parents or self.run in output.parents:
            raise ValueError("audit outputs must stay under repository docs and outside formal run")
        verdict="FAIL" if self.errors else "PARTIAL" if self.pending else "METADATA_COMPLETE_WITH_DEFERRED_BINARY_VERIFICATION"
        code=1 if self.errors else 2 if self.pending else 0
        script=Path(__file__).resolve()
        payload={"schema":"c3_final_test_metadata_audit_v1","verdict":verdict,"exit_code":code,
            "started_utc":self.started,"ended_utc":now(),"run":str(self.run),"checks_total":sum(self.counts.values()),
            "checks_by_group":dict(self.counts),"discrepancies":self.errors,"missing_or_pending":self.pending,
            "execution":{"argv":sys.argv,"python_executable":sys.executable,"cwd":str(Path.cwd()),"elapsed_wall_s":time.perf_counter()-self.timer,
                "retained_script_relative_path":script.relative_to(ROOT).as_posix(),"retained_script_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),
                "core_imports_tests_inference_physics_training_DDIM_encoding_or_export":False,"weight_NPZ_largearchive_read_or_hash":False},
            "source_identity":{"expected_algorithm_producer_commit":PRODUCER,"observed_algorithm_producer_commit":getattr(self,"identity",{}).get("algorithm_producer_commit"),
                "source_identity_sha256":getattr(self,"identity_sha",None),
                "reviewed_current_sources":[name for name in SOURCES if str((ROOT/"v6_4"/name).resolve()) in self.inputs],
                "all430_current_source_hashes_verified_by_this_script":False},
            "scope":"Independent small-metadata TEST audit. No current trace/NPZ/weight/large interval-log byte revalidation; full frozen source and archive validation are separate required work.",
            "deferred_current_byte_verification":self.deferred,"experiment_completion_verdict":"NOT_SUPPLIED_BY_THIS_METADATA_ONLY_AUDIT",
            "results":self.results,"input_sha256_inventory":self.inputs}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
        lines=["# C.3 final TEST metadata audit","",f"Verdict: **{verdict}** (exit {code}).","",
            f"Observed {self.started} to {payload['ended_utc']} UTC. {sum(self.counts.values())} checks, {len(self.errors)} contradictions, {len(self.pending)} missing/pending inputs.","",
            "The JSON independently reconstructs request/count/work accounting, raw outcomes, original actual acceptance and alias identities. Missing files never pass. Original report completion declarations are not evidence. Four-slot curves are nominal predictions; R8 has no separately measured cold-request total.","",
            "Current weight, trace/NPZ and large interval-log bytes and the complete 430-source inventory are outside this script. Recorded SHA strings and existence are retained as deferred evidence; they are not labeled current-byte verified. No experiment core is imported or executed and formal run evidence is unchanged.",""]
        if self.results:
            lines += [f"Candidate counts: `{self.results['candidate_counts']}`; TEST disjoint logical partition: `{self.results['TEST_logical_partition_disjoint']}`.","",
                "| Endpoint | Preference | Full27s + five gates /4 | B30 /4 | NO_PLAN | Pre-refusal | Actual failure | Alias | Near qualified R12 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in self.results["endpoint_summary"]:
                b="N/A" if row["actual_B30"] is None else str(row["actual_B30"])
                lines.append(f"| {row['endpoint']} | {row['preference']} | {row['full_27s_five_gates']} | {b} | {row['NO_PLAN']} | {row['actual_pre_rejected']} | {row['actual_failed']} | {row['actual_aliases']} | {row['near_R12_count']}/{row['R12_qualified_reference_count']} |")
            lines += ["","Quality and capability precede costs. Illegal proposals and shortened failed runs can reduce integrations without establishing a learning benefit. This audit supplies no benefit, checkpoint-reselection, deployment or overall research-completion verdict.",""]
        if self.pending:
            lines += ["Missing/pending:",""]+["- "+item["path"]+": "+item["reason"] for item in self.pending]+[""]
        if self.errors:
            lines += ["Contradictions:","","```json",json.dumps(self.errors,indent=2,ensure_ascii=False),"```",""]
        output.with_suffix(".md").write_text("\n".join(lines),encoding="utf8")
        print(json.dumps({"verdict":verdict,"exit_code":code,"checks":sum(self.counts.values()),"errors":len(self.errors),"pending":len(self.pending),"output_prefix":str(output)},ensure_ascii=False))
        return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--output-prefix",type=Path,default=ROOT/"docs/C3_FINAL_TEST_METADATA_AUDIT")
    parser.add_argument("--report-summary",type=Path,help="Optional completed run/summary.json; compare independently reconstructed numbers only")
    args=parser.parse_args(argv)
    audit=Audit(args.run)
    try:
        audit.audit(args.report_summary)
    except FileNotFoundError as error:
        audit.pending.append({"path":str(error.filename),"reason":"required evidence disappeared or was never completed"})
    except (ValueError,KeyError,TypeError,IndexError,StopIteration,OSError) as error:
        audit.errors.append({"group":"schema_or_evidence","check":"audit stopped on inconsistent or unsupported metadata","type":type(error).__name__,"message":str(error)})
    return audit.finish(args.output_prefix)


if __name__ == "__main__":
    raise SystemExit(main())
