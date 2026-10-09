"""Standalone read-only C.3 metadata audit: no repository or torch imports."""
import collections
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
import zipfile
import numpy as np

ROOT = Path(r"E:\v64c3")
RUN = ROOT / "v6_4/output/search_aware_warmstart_20261008_01"
DOCS = ROOT / "docs"
START = datetime.datetime.now(datetime.timezone.utc).isoformat()
inventory = {}
errors = []
checks = collections.Counter()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def record(path):
    path = Path(path).resolve()
    size = path.stat().st_size
    if size > 24_000_000:
        raise ValueError("Large archive reads/hashes forbidden: " + str(path))
    inventory[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": size}
    return inventory[str(path)]["sha256"]

def read(path):
    path = Path(path)
    path = path if path.is_absolute() else RUN / path
    record(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def check(kind, truth, detail):
    checks[kind] += 1
    if not truth:
        errors.append({"kind": kind, "detail": detail})
        print("DISCREPANCY", kind, detail, flush=True)

def finite(value):
    return type(value) in (float, int) and math.isfinite(value)

def full(row):
    m = row.get("prediction_metrics") or {}
    return bool(row.get("prediction_admissible") is True and row.get("prediction_task_passed") is True and
        row.get("online_guards_passed") is True and row.get("prediction_steps") == 13500 and
        m.get("native_state_count") == 13501 and finite(m.get("saved_horizon_s")) and
        abs(m["saved_horizon_s"] - 27) < 1e-8 and all(finite(m.get(k)) for k in ("I_support", "L_full")))

def qualified(row, pref):
    m = row.get("prediction_metrics") or {}
    return full(row) and (pref == "A" or m.get("clearance_status") == "MEASURED" and
        finite(m.get("d_support")) and m["d_support"] >= .030)

def near(row, reference, pref):
    if reference is None or not qualified(row, pref):
        return False
    a, b = row["prediction_metrics"], reference["prediction_metrics"]
    return a["L_full"] <= b["L_full"] + .005 and (pref == "B" or a["I_support"] <= b["I_support"] + .001)

def rank(rows, pref):
    valid = [r for r in rows if r.get("prediction_admissible") is True and
        all(finite((r.get("prediction_metrics") or {}).get(k)) for k in ("I_support", "L_full"))]
    norm = lambda r: float(np.linalg.norm(r["x_m"]))
    if pref == "B":
        return min((r for r in valid if finite(r["prediction_metrics"].get("d_support")) and
            r["prediction_metrics"]["d_support"] >= .03), key=lambda r: (
            r["prediction_metrics"]["L_full"], r["prediction_metrics"]["I_support"], norm(r), r["candidate_id"]), default=None)
    best = min((r["prediction_metrics"]["I_support"] for r in valid), default=None)
    return min((r for r in valid if r["prediction_metrics"]["I_support"] <= best + .001), key=lambda r: (
        r["prediction_metrics"]["L_full"], norm(r), ("v1", "v2").index(r["family"]), r["candidate_id"]), default=None) if best is not None else None

prior = read(DOCS / "C3_TEACHER_EVIDENCE_AUDIT_10_STREAMS.json")
record(DOCS / "C3_TEACHER_EVIDENCE_AUDIT_10_STREAMS.md")
record(Path(r"C:\Users\admin\.codex\attachments\1917d637-236f-4883-8674-4c3f5a6ff3e0\goal-objective.md"))
record(__file__)
plan = read("plan.json")
identity = read("source_identity.json")
split = read("learning_split_manifest.json")
hist = read("historical_import/history.json")
pairs = read("teacher_search/frozen_seed_pairs.json")
results = read("teacher_search/results.json")
manifest = read("dataset/manifest.json")
labels = read("dataset/labels.json")
facts = read("dataset/candidate_facts.json")
cond = read("dataset/condition_scaler.json")
residual = read("dataset/residual_scaler.json")
ds_schema = read("dataset/condition_schema.json")
config = read("models/training_config.json")
model_cond = read("models/condition_normalizer.json")
model_residual = read("models/residual_normalizer.json")
schema = read("models/condition_schema.json")
training_summary = read("models/training_summary.json")
train = {t["task_id"]: t for t in split["tasks"] if t["split"] == "train"}
train_sha = {t["task_sha256"] for t in train.values()}
train_mothers = {t["mother_id"] for t in train.values()}
check("external_split", len(train) == 6 and len(train_mothers) == 3 and
    set(train) == {t["task_id"] for t in hist["original_train_tasks"]}, "six original C2 TRAIN Tasks")
check("external_split", set(hist["train_mothers"]) == train_mothers and
    hist["excluded_C2_VAL_TEST_candidate_count"] == 24 and hist["original_C2_candidate_count"] == 96, "old VAL/TEST excluded")
check("identity", identity["algorithm_producer_commit"] == "9f39b42775283432eb933f63a9047c488ba22070", "frozen producer")
for value, name in ((pairs, "pairs"), (results, "results")):
    check("self_seal", digest({k: v for k, v in value.items() if k != "content_sha256"}) == value["content_sha256"], name)
check("self_seal", results["pairs_sha256"] == record(RUN / "teacher_search/frozen_seed_pairs.json") and
    pairs["history_sha256"] == record(RUN / "historical_import/history.json"), "teacher/history bindings")
check("teacher_terminal", len(results["summaries"]) == len(pairs["pairs"]) == 12 and results["slots_consumed"] == 96 and
    results["actual_logical_slots"] == 0 and results["protocol_completed"] is True and
    results["formal_actual_validation"] == "NOT_RUN_TEACHER_SEARCH", "12 groups/96 slots/no teacher actual")
check("history", facts == hist["candidates"] and len(facts) == 72, "same 72 historical physical facts")
route = hist["route_quality_labels"]
route_by = {s["sample_id"]: s for s in route}
physical_keys = set()
for fact in facts:
    key = digest({k: fact[k] for k in ("task_sha256", "execution_identity", "reference_version", "plan_sha256")})
    check("physical_dedup", key == fact["physical_candidate_id"] and key not in physical_keys, fact["physical_candidate_id"])
    physical_keys.add(key)
    check("history_train_only", fact["task_id"] in train and fact["task_sha256"] in train_sha and fact["split"] == "train" and
        fact["mother_id"] == train[fact["task_id"]]["mother_id"], fact["physical_candidate_id"])
    check("history_metadata", fact["nominal_archive_seal"]["complete"] is True, fact["physical_candidate_id"])
    if fact["evidence_tier"] in ("PREDICTED_COMPLETE", "VALIDATED_EXECUTION"):
        check("history_complete", full(fact["prediction"]), fact["physical_candidate_id"])
    if fact["evidence_tier"] == "VALIDATED_EXECUTION":
        def gate_pass(binding):
            gates = binding.get("original_five_gates", {})
            return binding.get("independently_bound_and_validated") is True and all(
                gates.get(g, {}).get("passed") is True if isinstance(gates.get(g), dict) else gates.get(g) is True
                for g in ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding"))
        check("history_actual_metadata", any(gate_pass(b) for b in fact["actual_bindings"]), fact["physical_candidate_id"])

expected_route = []
for tid, task in train.items():
    for pref in ("A", "B"):
        for family in ("v1", "v2"):
            pool = [f for f in facts if f["task_id"] == tid and f["family"] == family and
                f["evidence_tier"] in ("PREDICTED_COMPLETE", "VALIDATED_EXECUTION") and qualified(f["prediction"], pref)]
            metric, band = ("I_support", .001) if pref == "A" else ("L_full", .005)
            best = min((f["prediction_metrics"][metric] for f in pool), default=None)
            expected_route += [(f["physical_candidate_id"], pref) for f in pool if f["prediction_metrics"][metric] <= best + band]
    pool = [{**f["prediction"], "candidate_id": min(f["original_candidate_ids"]), "family": f["family"],
        "x_m": np.asarray(f["z_m"])[np.asarray(f["search_interval_mask"], bool)].reshape(-1).tolist(),
        "physical_candidate_id": f["physical_candidate_id"]} for f in facts if f["task_id"] == tid and
        f["evidence_tier"] in ("PREDICTED_COMPLETE", "VALIDATED_EXECUTION")]
    for pref in ("A", "B"):
        winner, reference = rank(pool, pref), pairs["historical_references"][tid][pref]
        check("historical_reference", (winner is None and reference is None) or (winner is not None and
            reference == {k: winner[k] for k in ("physical_candidate_id", "prediction_metrics", "plan_sha256")}), tid + "/" + pref)
check("route_quality", set(expected_route) == {(s["physical_candidate_id"], s["preference"]) for s in route} and
    len(expected_route) == len(route), "family near-optimal route pool reconstructed")
for sample in route:
    check("route_provenance", sample["source_types"] == ["route_quality_example"] and sample["source_refs"] == [{
        "source_type": "route_quality_example", "physical_candidate_id": sample["physical_candidate_id"], "original_C2_TRAIN": True}], sample["sample_id"])

raw_rows, fallbacks, streams, recomputed = [], [], [], []
all_rows = []
work = collections.Counter()
for pair in pairs["pairs"]:
    tid = pair["task_id"]
    tag = tid + "/" + pair["combination"]
    mask = np.zeros(6, bool)
    mask[train[tid]["active_intervals"]] = True
    check("pair_identity", pair["task_sha256"] == train[tid]["task_sha256"] and pair["mother_id"] == train[tid]["mother_id"] and
        pair["seed_pair_id"] == digest({k: v for k, v in pair.items() if k != "seed_pair_id"}), tag)
    for slot, pref, family in ((1, "A", "v1"), (3, "B", "v2")):
        proposal, raw = pair["proposals"][str(slot)], pair["raw_qualification"][str(slot)]
        z = np.asarray(proposal["raw_z_m"], float)
        check("frozen_raw", z.shape == (6, 2) and np.isfinite(z).all() and not np.any(z[~mask]) and
            np.all(np.linalg.norm(z, axis=1) <= .02) and raw["raw_legal"] is True and raw["raw_repaired"] is False and
            raw["resampled"] is False and raw["raw_z_m"] == proposal["raw_z_m"] and raw["search_interval_mask"] == mask.tolist() and
            proposal["preference"] == pref and proposal["family"] == family, tag + "/" + pref)
        construction = proposal["teacher_construction"]
        status = construction["construction_status"]
        if status == "RULE_CONSTRUCTION_NO_MATCHING_TEACHER":
            fallbacks.append({"task_id": tid, "slot": slot})
            check("local_fallback", not any(s["task_id"] == tid and s["preference"] == pref and s["family"] == family for s in route), tag + "/" + pref)
        elif pair["combination"] == "T_local":
            source = route_by[construction["route_quality_sample_id"]]
            check("local_source", source["task_id"] == tid and source["preference"] == pref and source["family"] == family and
                source["z_m"] == proposal["raw_z_m"] and source["physical_candidate_id"] == construction["physical_candidate_id"], tag + "/" + pref)
            local_ids = {s["physical_candidate_id"] for s in route if s["task_id"] == tid and s["preference"] == pref and s["family"] == family}
            local_rows = [{**f["prediction"], "candidate_id": min(f["original_candidate_ids"]), "family": f["family"],
                "x_m": np.asarray(f["z_m"])[mask].reshape(-1).tolist(), "physical_candidate_id": f["physical_candidate_id"]}
                for f in facts if f["physical_candidate_id"] in local_ids]
            local_winner = rank(local_rows, pref)
            check("local_teacher_ranking", local_winner is not None and local_winner["physical_candidate_id"] == source["physical_candidate_id"], tag + "/" + pref)
        else:
            source = route_by[proposal["retrieval_source"]["sample_id"]]
            rules = proposal["retrieval_rules"]
            check("transfer_exclusion", source["mother_id"] != pair["mother_id"] and source["mother_id"] in train_mothers and
                construction["excluded_mother_id"] == pair["mother_id"] and set(construction["eligible_mothers"]) == train_mothers - {pair["mother_id"]} and
                rules["leave_out_mother_id"] == pair["mother_id"] and rules["initial_historical_pool_only"] is True and
                source["preference"] == pref and source["family"] == family and source["z_m"] == proposal["raw_z_m"], tag + "/" + pref)
            check("transfer_schema", rules["schema"] == "v64_c2_train_retrieval_v1" and rules["distance"] ==
                "mean_squared_all_train_normalized_condition_dimensions_v1" and rules["tie_order"] == ["task_sha256", "plan_sha256", "sample_id"] and
                rules["condition_scaler_sha256"] == digest(pairs["condition_scaler"]), tag + "/" + pref)
        raw_rows.append({"task_id": tid, "combination": pair["combination"], "slot": slot, "raw_legal": raw["raw_legal"], "construction": status})
    summary = next(s for s in results["summaries"] if s["seed_pair_id"] == pair["seed_pair_id"])
    stream = Path(summary["stream_path"])
    planning = stream / "planning" / tid
    rows = read(planning / "candidate_registry.json")
    proposals = read(planning / "proposals.json")
    selection = read(planning / "selection.json")
    effect = read(stream / "teacher_effect.json")
    all_rows += rows
    check("central_effect", summary == effect, tag)
    for name, expected in summary["source_files"].items():
        check("teacher_source_seal", record(RUN / name) == expected, name)
    check("teacher_stream", len(rows) == 8 and len(proposals) == selection["budget"]["proposal_attempts"] and
        selection["registry_content_sha256"] == digest(rows) and selection["selection_reads_final_actual"] is False and
        selection["budget"]["stop_reason"] != "TOOL_ERROR", tag)
    by_id = {r["candidate_id"]: r for r in rows}
    endpoints = {}
    for index, row in enumerate(rows):
        check("candidate_metadata", row["candidate_id"] == f"C{index:02d}" and row["plan_sha256"] == digest(row["plan"]) and
            row["plan"]["definition"]["task_sha256"] == pair["task_sha256"], tag + "/" + row["candidate_id"])
        if row.get("parent_candidate_id"):
            check("lineage", row["parent_candidate_id"] in by_id and int(row["parent_candidate_id"][1:]) < index, tag + "/" + row["candidate_id"])
            parent_lineage = by_id[row["parent_candidate_id"]].get("proposal_lineage", [])
            check("lineage_inheritance", row.get("proposal_lineage", [])[:-1] == parent_lineage and
                row["proposal_lineage"][-1].get("parent_candidate_id") == row["parent_candidate_id"], tag + "/" + row["candidate_id"])
        for name, count in (row.get("costs") or {}).items():
            if type(count) == int:
                work[name] += count
    for budget in (4, 8):
        prefix = read(planning / f"prefix_{budget:02d}.json")
        check("prefix", prefix["registry_content_sha256"] == digest(rows[:budget]) and prefix["snapshot_content_sha256"] ==
            digest({k: v for k, v in prefix.items() if k != "snapshot_content_sha256"}) and prefix["later_slots_read"] is False, tag + "/" + str(budget))
    for slot, pref, family in ((1, "A", "v1"), (3, "B", "v2")):
        selected = selection["preferences"][pref]
        cid = selected.get("source_candidate_id")
        row = by_id.get(cid)
        reference = pairs["historical_references"][tid][pref]
        seed = next(p for p in proposals if p.get("initializer_slot") == slot)
        seed_id = seed.get("cache_hit_candidate_id") or seed.get("candidate_id") or next((r["candidate_id"] for r in rows if r.get("search_content_key") == seed.get("content_key")), None)
        direct = bool(seed["raw_seed_diagnostics"]["raw_legal"] and seed_id in by_id and by_id[seed_id].get("initializer_slot") == slot and qualified(by_id[seed_id], pref))
        from_slot = bool(row and any(v.get("initial_position") == slot for v in row.get("proposal_lineage", [])))
        accepted = bool(row and qualified(row, pref))
        rule_only = bool(accepted and not from_slot and not direct)
        positive = bool(seed["raw_seed_diagnostics"]["raw_legal"] and accepted and (direct or from_slot) and
            pair["proposals"][str(slot)]["teacher_construction"]["construction_status"] != "RULE_CONSTRUCTION_NO_MATCHING_TEACHER")
        first = next((i + 1 for i, r in enumerate(rows) if qualified(r, pref)), None)
        first_near = next((i + 1 for i, r in enumerate(rows) if near(r, reference, pref)), None) if reference else None
        expected = {"qualified_endpoint": accepted, "near_historical_reference": bool(accepted and near(row, reference, pref)),
            "historical_reference_available": reference is not None, "selected_candidate_id": cid, "first_qualified_slot": first,
            "first_qualified_right_censored_budget": 8 if first is None else None, "first_near_quality_slot": first_near,
            "first_near_quality_status": "N/A_NO_HISTORICAL_REFERENCE" if reference is None else "HIT" if first_near else "RIGHT_CENSORED",
            "first_near_quality_right_censored_budget": 8 if reference and first_near is None else None, "raw_legal": True,
            "direct_seed_qualified": direct, "selected_seed_or_descendant": from_slot, "rule_only": rule_only, "positive_supervision": positive,
            "label_status": "INITIALIZER_EFFECT_EVIDENCED" if positive else "RULE_ONLY" if rule_only else "NO_QUALIFIED_INITIALIZER_EFFECT_LABEL"}
        check("teacher_endpoint", all(summary["endpoints"][pref][k] == v for k, v in expected.items()), tag + "/" + pref)
        endpoints[pref] = expected
        winner = rank(rows, pref)
        check("backend_ranking", cid == (winner["candidate_id"] if winner else None), tag + "/" + pref)
        own = [s for s in summary["initializer_effect_labels"] if s["preference"] == pref]
        check("effect_eligibility", len(own) == int(positive), tag + "/" + pref)
        for label in own:
            ref = label["source_refs"][0]
            check("effect_original_seed", label["z_m"] == pair["proposals"][str(slot)]["raw_z_m"] and label["family"] == family and
                label["task_sha256"] == pair["task_sha256"] and label["evidence_tier"] == "PREDICTED_COMPLETE" and
                label["formal_actual_validation"] == "NOT_RUN_TEACHER_SEARCH" and label["prediction_metrics"] == row["prediction_metrics"] and
                ref["seed_pair_id"] == pair["seed_pair_id"] and ref["slot"] == slot and ref["seed_candidate_id"] == seed_id and
                ref["selected_candidate_id"] == cid and ref["fixed_partner"] == pair["proposals"][str(3 if slot == 1 else 1)] and
                ref["shared_search_is_not_independent_counterfactuals"] is True, tag + "/" + pref)
        check("raw_consumed", seed["raw_seed_diagnostics"]["raw_z_m"] == pair["proposals"][str(slot)]["raw_z_m"] and
            seed["raw_seed_diagnostics"]["raw_legal"] is True, tag + "/" + pref)
    recomputed.append({"seed_pair_id": pair["seed_pair_id"], "task_id": tid, "combination": pair["combination"], "endpoints": endpoints})
    streams.append({"task_id": tid, "combination": pair["combination"], "slots": len(rows), "proposals": len(proposals),
        "cache_hits": sum(bool(p.get("cache_hit_candidate_id")) for p in proposals), "endpoints": endpoints})

selected_ids = {}
for tid in train:
    survivors = [s for s in recomputed if s["task_id"] == tid]
    for name in ("qualified_endpoint", "near_historical_reference"):
        best = max(sum(e[name] for e in s["endpoints"].values()) for s in survivors)
        survivors = [s for s in survivors if sum(e[name] for e in s["endpoints"].values()) == best]
    groups = {}
    for summary in survivors:
        achieved = tuple(p for p, e in summary["endpoints"].items() if e["qualified_endpoint"])
        near_set = tuple(p for p, e in summary["endpoints"].items() if e["near_historical_reference"])
        score = sum(summary["endpoints"][p]["first_near_quality_slot" if p in near_set else "first_qualified_slot"] or 9 for p in achieved)
        groups.setdefault((achieved, near_set), []).append((score, summary["seed_pair_id"]))
    selected_ids[tid] = [pid for group in groups.values() for score, pid in group if score == min(v[0] for v in group)]
    check("pair_hierarchy", selected_ids[tid] == results["selected_teacher_pair_ids"][tid], tid)
expected_samples = route + [s for summary in results["summaries"] if summary["seed_pair_id"] in
    selected_ids[summary["task_id"]] for s in summary["initializer_effect_labels"]]
unique = {}
for sample in expected_samples:
    key = digest({k: sample[k] for k in ("task_sha256", "family", "preference", "z_m")})
    row = unique.setdefault(key, {"source_types": set(), "refs": []})
    row["source_types"].update(sample["source_types"])
    row["refs"] += sample["source_refs"]
check("pool_union", set(unique) == {s["sample_id"] for s in labels} and len(labels) == len(unique), "all and only deduplicated route plus selected effects")
for sample in labels:
    key = digest({k: sample[k] for k in ("task_sha256", "family", "preference", "z_m")})
    expected, z, mask = unique[key], np.asarray(sample["z_m"]), np.asarray(sample["search_interval_mask"], bool)
    check("label_identity", key == sample["sample_id"] and sample["task_id"] in train and sample["task_sha256"] in train_sha and
        sample["split"] == "train" and sample["mother_id"] == train[sample["task_id"]]["mother_id"] and mask.sum() <= 2 and
        not np.any(z[~mask]) and np.all(np.linalg.norm(z, axis=1) <= .02), key)
    check("pool_provenance", set(sample["source_types"]) == expected["source_types"] and
        {digest(v) for v in sample["source_refs"]} == {digest(v) for v in expected["refs"]}, key)
for name, expected in manifest["files"].items():
    check("dataset_seal", record(RUN / "dataset" / name) == expected, name)
check("dataset_seal", read("dataset/dataset_manifest.json") == manifest and manifest["teacher_results_sha256"] == record(RUN / "teacher_search/results.json") and
    manifest["history_sha256"] == record(RUN / "historical_import/history.json") and
    manifest["learning_split_manifest_sha256"] == record(RUN / "learning_split_manifest.json"), "dataset roots")
counts = {"unique_labels": len(labels), "route_quality_labels": sum("route_quality_example" in s["source_types"] for s in labels),
    "initializer_effect_labels": sum("initializer_effect_example" in s["source_types"] for s in labels),
    "both_source_labels": sum(len(s["source_types"]) == 2 for s in labels), "zero_labels": sum(not np.any(s["z_m"]) for s in labels),
    "pre_hierarchy_effect_rows": sum(len(s["initializer_effect_labels"]) for s in results["summaries"]), "selected_pair_effect_rows": len(expected_samples) - len(route)}
check("dataset_counts", manifest["label_count_views"] == counts["unique_labels"] and manifest["route_quality_label_count"] == counts["route_quality_labels"] and
    manifest["initializer_effect_label_count"] == counts["initializer_effect_labels"] and manifest["zero_label_views"] == counts["zero_labels"] and
    manifest["D_S_N_identical_pool"] is True, "manifest counts/shared pool")
check("scaler_identity", cond == model_cond and residual == model_residual and set(cond["fit_task_sha256"]) == train_sha and
    cond["fit_split"] == residual["fit_split"] == "train", "shared TRAIN-only normalizers")

unique_conditions = {}
for sample in labels:
    key, raw = (sample["task_id"], sample["preference"], sample["family"]), np.asarray(sample["condition_raw"], float)
    check("condition_consistency", key not in unique_conditions or np.array_equal(unique_conditions[key], raw), str(key))
    unique_conditions[key] = raw
names = cond["feature_names_with_units"]
name_index = {name: i for i, name in enumerate(names)}
def valid_mask(raw):
    valid = np.ones(len(names), bool)
    for j, name in enumerate(names):
        match = re.match(r"(base_path|requirement|obstacle|interval|protected)\.(\d+)\.(.*)", name)
        if match:
            group, num, field = match.groups()
            prefix = group + "." + num
            if group == "base_path" and field.startswith("position"):
                valid[j] = bool(raw[name_index[prefix + ".present.0:dimensionless"]])
            elif group == "base_path" and field.startswith("segment_duration"):
                valid[j] = bool(raw[name_index[prefix + ".segment_present.0:dimensionless"]])
            elif group in ("requirement", "obstacle") and not field.startswith("present"):
                valid[j] = bool(raw[name_index[prefix + ".present.0:dimensionless"]])
            elif group == "interval" and not field.startswith("mask"):
                valid[j] = bool(raw[name_index[prefix + ".mask.0:dimensionless"]])
            elif group == "protected" and field.startswith("times"):
                valid[j] = bool(raw[name_index[prefix + ".present.0:dimensionless"]])
        if name == "preferred_clearance_m":
            valid[j] = bool(raw[name_index["preference.B"]])
    return valid
raw = np.stack(list(unique_conditions.values()))
valid = np.stack([valid_mask(v) for v in raw])
count = valid.sum(0)
mean = np.divide((raw * valid).sum(0), count, out=np.zeros(raw.shape[1]), where=count > 0)
variance = np.divide(((raw - mean)**2 * valid).sum(0), count, out=np.zeros_like(mean), where=count > 0)
std = np.maximum(np.sqrt(variance), cond["std_floors"])
literal = np.asarray(cond["literal"], bool)
mean[literal], std[literal] = 0., 1.
check("condition_refit", np.allclose(mean, cond["mean"], rtol=0, atol=1e-12) and
    np.allclose(std, cond["std"], rtol=0, atol=1e-12), "independent TRAIN-only condition fit")
for sample in labels:
    raw = np.asarray(sample["condition_raw"])
    normalized = np.where(valid_mask(raw), (raw - cond["mean"]) / np.asarray(cond["std"]), 0).astype(np.float32)
    check("condition_transform", np.array_equal(normalized, np.asarray(sample["condition_normalized"], np.float32)), sample["sample_id"])
normalized_initial_route = []
for sample in route:
    raw = unique_conditions[(sample["task_id"], sample["preference"], sample["family"])]
    scaler = pairs["condition_scaler"]
    normalized = np.where(valid_mask(raw), (raw - scaler["mean"]) / np.asarray(scaler["std"]), 0).astype(np.float32)
    normalized_initial_route.append({**sample, "condition_raw": raw.tolist(), "condition_normalized": normalized.tolist()})
for pair in pairs["pairs"]:
    if pair["combination"] == "T_transfer":
        expected_pool = [s for s in normalized_initial_route if s["mother_id"] != pair["mother_id"]]
        for slot, proposal in pair["proposals"].items():
            check("transfer_full_pool_identity", proposal["retrieval_rules"]["train_labels_sha256"] == digest(expected_pool), pair["task_id"] + "/" + slot)
residual_unique = {digest({"task": s["task_sha256"], "family": s["family"], "z": s["z_m"]}): s for s in labels}
rv = np.concatenate([np.asarray(s["z_m"])[np.asarray(s["search_interval_mask"], bool)] for s in residual_unique.values()])
check("residual_refit", np.allclose(rv.mean(0), residual["mean_m"], atol=1e-14, rtol=0) and
    np.allclose(np.maximum(rv.std(0), .001), residual["std_m"], atol=1e-14, rtol=0) and
    len(rv) == residual["count_active_intervals"], "unique residual parameter statistics")
check("model_schema", schema["condition_dim"] == len(names) == 896 and schema["latent_dim"] == 12 and
    schema["feature_names_with_units"] == names and schema["literal"] == cond["literal"] and
    schema["ids_or_outcomes_encoded"] is False and ds_schema["feature_names_with_units"] == names, "same declared features and masks")
expected_supported = {}
for sample in labels:
    key = sample["preference"] + "/" + sample["reference_family"] + "/" + "".join(str(int(v)) for v in sample["search_interval_mask"])
    row = expected_supported.setdefault(key, {"preference": sample["preference"], "reference_family": sample["reference_family"],
        "search_mask": sample["search_interval_mask"], "sample_ids": [], "mother_ids": []})
    row["sample_ids"].append(sample["sample_id"])
    row["mother_ids"].append(sample["mother_id"])
for row in expected_supported.values():
    row["sample_ids"] = sorted(set(row["sample_ids"]))
    row["mother_ids"] = sorted(set(row["mother_ids"]))
check("supported_condition_IDs", config["supported_conditions"] == expected_supported, "all condition buckets and exact TRAIN sample/mother IDs")
check("model_architecture", all(config["models"][m]["condition_dim"] == 896 and config["models"][m]["latent_dim"] == 12 and
    config["models"][m]["hidden_dim"] == 128 and config["models"][m]["hidden_layers"] == 2 for m in ("D", "S")) and
    config["models"]["S"]["activation"] == "SiLU" and config["models"]["D"]["objective"] == "v_mse" and
    config["models"]["D"]["diffusion_steps"] == 100 and config["models"]["D"]["noise_schedule"] == "cosine" and
    config["models"]["D"]["ddim_steps"] == 20, "shared2x128 architecture;unmodified cosine100/v-MSE/DDIM20 D")
check("training_config", config["dataset_manifest_sha256"] == record(RUN / "dataset/manifest.json") and
    config["sample_ids_in_dataset_order"] == [s["sample_id"] for s in labels] and config["optimizer_updates"] == 4000 and
    config["batch_size"] == 32 and config["checkpoint_updates"] == [250, 4000] and
    config["optimizer"] == {"name": "AdamW", "lr": .0001, "weight_decay": .01, "gradient_norm_clip": 1.} and
    config["fresh_initialization"] is True and config["old_weights_loaded"] is False and
    config["initialization_seeds"] == {"D": 64321, "S": 64331}, "fixed paired training protocol")
for name, expected in config["normalizer_sha256"].items():
    check("training_normalizer_seal", record(RUN / "models" / name) == expected, name)
for name, expected in config["source_files"].items():
    check("training_source_seal", record(name) == expected, name)
refs_path = RUN / "models/paired_reference_draws.npz"
check("paired_draw_seal", record(refs_path) == config["paired_reference_draws_file_sha256"], "small 1 MB paired index file")
with np.load(refs_path, allow_pickle=False) as archive:
    refs = archive["reference_indices"]
check("paired_draws", refs.shape == (4000, 32) and refs.dtype.kind == "i" and refs.min() >= 0 and refs.max() < len(labels) and
    hashlib.sha256(refs.astype("<i8", copy=False).tobytes()).hexdigest() == config["paired_reference_indices_sha256"], "128000 paired index bytes")
exposures = dict(collections.Counter(labels[int(i)]["sample_id"] for i in refs.reshape(-1)))

class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *rest: dict(storage=storage, offset=offset, size=size, stride=stride)
        raise ValueError("Forbidden checkpoint pickle global: " + module + "." + name)
    def persistent_load(self, value):
        return value

checkpoint_rows, model_rows = [], {}
for model in ("D", "S"):
    report = read(f"models/{model}/training_report.json")
    status = read(f"models/{model}/training_status.json")
    curves_path = RUN / f"models/{model}/curves.jsonl"
    record(curves_path)
    curves = [json.loads(v) for v in curves_path.read_text().splitlines()]
    check("training_terminal", report["status"] == "COMPLETED" and report["optimizer_updates_total"] == 4000 and
        report["sample_exposures_total"] == 128000 and report["sample_exposures"] == exposures and
        report["paired_reference_indices_sha256"] == config["paired_reference_indices_sha256"] and
        report["training_config_sha256"] == record(RUN / "models/training_config.json") and report["physics_steps"] == report["ddim_sample_units"] == 0 and
        report["selected_checkpoint_update"] is None and report["test_used_for_training_or_selection"] is False and
        report["val_used_for_training_or_selection"] is False, model)
    check("training_status", status["status"] == "COMPLETED" and status["optimizer_updates_started"] == status["optimizer_updates_completed"] == 4000 and
        status["training_report_sha256"] == record(RUN / f"models/{model}/training_report.json"), model)
    check("training_curves", [v["update"] for v in curves] == list(range(50, 4001, 50)) and
        all(finite(v["loss"]) and finite(v["gradient_norm_before_clip"]) for v in curves), model)
    for update in (250, 4000):
        path = RUN / f"models/{model}/checkpoint_{update:04d}.pt"
        file_sha = record(path)
        check("checkpoint_file", file_sha == report["checkpoint_sha256"][str(update)], model + str(update))
        with zipfile.ZipFile(path) as archive:
            prefix = next(n.rsplit("/", 1)[0] for n in archive.namelist() if n.endswith("/data.pkl"))
            meta = MetadataReader(io.BytesIO(archive.read(prefix + "/data.pkl"))).load()
            state_hash, parameters, shapes = hashlib.sha256(), 0, {}
            for name, tensor in sorted(meta["state_dict"].items()):
                storage = tensor["storage"]
                check("checkpoint_tensor", storage[1] == "FloatStorage" and tensor["offset"] == 0, model + str(update) + "/" + name)
                raw_bytes = archive.read(prefix + "/data/" + storage[2])
                array = np.frombuffer(raw_bytes, dtype="<f4").reshape(tensor["size"])
                check("checkpoint_tensor", np.isfinite(array).all(), "finite " + model + str(update) + "/" + name)
                state_hash.update(name.encode() + b"\0")
                state_hash.update(array.tobytes())
                shapes[name] = list(tensor["size"])
                if name.endswith(("weight", "bias")):
                    parameters += array.size
        check("checkpoint_metadata", meta["schema"] == "v64_c3_paired_warmstart_checkpoint_v1" and meta["model_name"] == model and
            meta["update"] == meta["optimizer_updates_experienced"] == update and meta["training_config"] == config and
            meta["training_config_sha256"] == record(RUN / "models/training_config.json") and meta["condition_normalizer"] == cond and
            meta["residual_normalizer"] == residual and meta["condition_schema"] == schema and meta["supported_conditions"] == config["supported_conditions"] and
            meta["fresh_initialization"] is True and meta["old_weights_loaded"] is False and meta["checkpoint_selected"] is False and
            meta["physics_steps"] == meta["ddim_sample_units"] == 0, model + str(update))
        prefix_exposures = collections.Counter(labels[int(i)]["sample_id"] for i in refs[:update].reshape(-1))
        prefix_exposures = {s["sample_id"]: prefix_exposures.get(s["sample_id"], 0) for s in labels}
        check("checkpoint_paired_exposure", meta["paired_reference_indices_prefix_sha256"] ==
            hashlib.sha256(refs[:update].astype("<i8", copy=False).tobytes()).hexdigest() and meta["sample_exposures"] == prefix_exposures and
            sum(meta["sample_exposures"].values()) == update * 32, model + str(update))
        check("checkpoint_weights", state_hash.hexdigest() == meta["state_sha256"] and meta["initial_state_sha256"] == report["initial_state_sha256"] and
            parameters == meta["effective_parameter_count"] == report["effective_parameter_count"] and
            (update != 4000 or meta["state_sha256"] == report["final_state_sha256"]), model + str(update))
        checkpoint_rows.append({"model": model, "update": update, "bytes": path.stat().st_size, "file_sha256": file_sha,
            "state_sha256": state_hash.hexdigest(), "initial_state_sha256": meta["initial_state_sha256"], "parameter_count": parameters,
            "sample_exposures": sum(meta["sample_exposures"].values()), "paired_reference_prefix_sha256": meta["paired_reference_indices_prefix_sha256"], "tensor_shapes": shapes})
    model_rows[model] = {k: report[k] for k in ("status", "optimizer_updates_total", "sample_exposures_total", "initialization_seed",
        "effective_parameter_count", "elapsed_s", "initial_state_sha256", "final_state_sha256", "paired_reference_indices_sha256", "selected_checkpoint_update")}
check("paired_models", model_rows["D"]["paired_reference_indices_sha256"] == model_rows["S"]["paired_reference_indices_sha256"] and
    all(v["initial_state_sha256"] != v["final_state_sha256"] for v in model_rows.values()) and training_summary["paired_exposures_verified"] is True, "paired fresh-training records")
for model in ("D", "S"):
    check("checkpoint_progress", len({v["state_sha256"] for v in checkpoint_rows if v["model"] == model}) == 2, "distinct 250/4000 " + model)
phase_receipts = {}
for name in ("teacher_phase.json", "command_logs/phase_build-dataset.json", "command_logs/phase_train.json"):
    phase_receipts[name] = read(name)
    check("phase_exit", phase_receipts[name]["exit_code"] == 0, name)

endpoint_counts = collections.Counter(e["label_status"] for s in recomputed for e in s["endpoints"].values())
result = {"schema": "v64_c3_final_teacher_dataset_training_audit_v1", "audit_start_utc": START,
    "audit_end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "argv": [sys.executable, *sys.argv], "exit_code": 0 if not errors else 1,
    "audit_script_identity": {"path": str(Path(__file__).resolve()), "sha256": record(__file__), "imports_repository_core": False, "imports_torch": False},
    "audit_tooling_note": "An initial overlength command failed before process creation (Windows os error206). The first temporary-script pass exited1 at receipt lookup because build/train receipts are under command_logs,not run root;all prior metadata checks emitted no discrepancy and no audit artifacts had been written. The corrected final pass is recorded here. No experiment terminal/cost records were modified by any audit attempt.",
    "run": str(RUN), "source_identity_sha256": record(RUN / "source_identity.json"), "frozen_producer": identity["algorithm_producer_commit"],
    "verdict": "NO_DISCREPANCY_IN_REVIEWED_EVIDENCE" if not errors else "DISCREPANCIES_FOUND",
    "prior_interim_scope": {"path": "docs/C3_TEACHER_EVIDENCE_AUDIT_10_STREAMS.json", "streams": prior["completed_stream_count_reviewed"], "is_final_dataset_or_training_audit": False},
    "scope": "Final12 teacher streams,TRAIN-only shared D/S/N pool,TRAIN scaler refits,paired draw indices and four small checkpoint metadata/weights;VAL/TEST pending.",
    "teacher": {"streams": 12, "slots": 96, "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH", "endpoint_label_status_counts": dict(endpoint_counts),
        "qualified_endpoints": sum(e["qualified_endpoint"] for s in recomputed for e in s["endpoints"].values()),
        "near_historical_endpoints": sum(e["near_historical_reference"] for s in recomputed for e in s["endpoints"].values()),
        "selected_teacher_pair_ids": selected_ids, "raw_inputs": raw_rows, "rule_construction_fallbacks": fallbacks, "streams_detail": streams, "recorded_work_totals": dict(work)},
    "historical_import": {"physical_candidates": 72, "excluded_old_VAL_TEST": 24,
        "evidence_tier_counts": dict(collections.Counter(f["evidence_tier"] for f in facts)), "raw_archives_rehashed_by_this_audit": False},
    "dataset": {**counts, "train_tasks": sorted(train), "train_mothers": sorted(train_mothers), "condition_unique_rows": len(unique_conditions),
        "condition_dim": len(names), "residual_unique_parameters": len(residual_unique), "residual_active_intervals": len(rv),
        "pool_manifest_sha256": record(RUN / "dataset/manifest.json"), "condition_scaler_sha256": record(RUN / "dataset/condition_scaler.json"),
        "residual_scaler_sha256": record(RUN / "dataset/residual_scaler.json"), "D_S_N_shared_pool_identity_verified": not errors, "TRAIN_only_scaler_refit_verified": not errors},
    "training": {"models": model_rows, "paired_index_shape": list(refs.shape), "paired_draw_file_sha256": record(refs_path),
        "paired_index_sha256": config["paired_reference_indices_sha256"], "checkpoint_whitelist": [250, 4000], "checkpoints": checkpoint_rows, "model_selection_performed_by_this_audit": False},
    "phase_receipts": phase_receipts, "checks": dict(checks), "check_total": sum(checks.values()), "errors": errors, "source_inventory": inventory,
    "limitations": ["No core imports,tests,physics,DDIM,training,rendering or export.",
        "No large physical NPZ/archive reads or hashes;historical saved seals are metadata evidence,not renewed byte-level physical verification.",
        "Teacher effects are nominal finite shared-search evidence,not final actual validation or independent causal necessity.",
        "Small checkpoint tensor bytes and paired draw arrays were read without torch/core imports;optimizer execution was not replayed.",
        "One seed per model and3 TRAIN mothers remain DATA_LIMITED;VAL/TEST completion and learning benefit are pending."],
    "added_physics_steps": 0, "added_training_updates": 0, "added_DDIM_samples": 0, "added_tests": 0, "research_completion_claim": False}
json_path, md_path = DOCS / "C3_FINAL_TEACHER_DATASET_AUDIT.json", DOCS / "C3_FINAL_TEACHER_DATASET_AUDIT.md"
with json_path.open("w", encoding="utf8") as handle:
    json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
lines = ["# C.3 finalized teacher, TRAIN dataset and paired-training audit", "",
    f"Audit UTC: {result['audit_end_utc']}. Frozen producer: `{result['frozen_producer']}`.", "",
    f"**{result['verdict']}**. {result['check_total']} read-only evidence checks; {len(errors)} discrepancies.", "",
    "This finalizes teacher/TRAIN/training only. Earlier10-stream interim documents are preserved. Formal VAL is running;VAL checkpoint choice,TEST and overall completion are outside this verdict.", "",
    "All12 teacher groups completed96 shared candidate slots and0 teacher actual slots. All24 original raw seeds were legal,unrepaired and unresampled. Pair identities,original seed targets,fixed partners,source references,seed/cache/lineage association,B30,near bands,first-hit censoring and the predeclared hierarchy were independently recomputed.", "",
    f"The24 nominal preference endpoints contain {result['teacher']['qualified_endpoints']} qualified endpoints and {result['teacher']['near_historical_endpoints']} near historical references. Statuses: `{json.dumps(dict(endpoint_counts),sort_keys=True)}`. RULE_ONLY,NO_PLAN,failed/incomplete predictions and2 explicit local B rule fallbacks receive no positive effect credit.", "",
    "The historical pool contains72 unique physical TRAIN candidates from3 mothers/6 Tasks;24 old C.2 VAL/TEST facts remain excluded. Transfer sources exclude the query mother and retain original historical-pool,family/mask and retrieval declarations.", "",
    f"The D/S/N pool contains {counts['unique_labels']} unique Task/family/preference/z labels: {counts['route_quality_labels']} with route provenance,{counts['initializer_effect_labels']} with effect provenance,{counts['both_source_labels']} with both. {counts['selected_pair_effect_rows']} selected-pair effect rows deduplicate into those effect-bearing labels;all source references remain. Zero labels: {counts['zero_labels']}. No parameter averaging or extra physical sample credit was found.", "",
    f"Both896-dimensional model condition normalizers and the residual normalizer match the dataset exactly. Condition statistics were independently refitted from {len(unique_conditions)} unique TRAIN declarations;residual statistics from {len(residual_unique)} unique Task/family/z parameters and {len(rv)} active intervals. Model schema and supported label IDs bind to the same TRAIN pool.", "",
    "| Model | Updates | Paired exposures | Parameters | Seed | Training seconds |", "|---|---:|---:|---:|---:|---:|"]
for model, value in model_rows.items():
    lines.append(f"| {model} | {value['optimizer_updates_total']} | {value['sample_exposures_total']} | {value['effective_parameter_count']} | {value['initialization_seed']} | {value['elapsed_s']:.6f} |")
lines += ["", "Each model records one fresh4000-update AdamW run,batch32,lr1e-4,weight_decay0.01 and clip1. The shared4000x32 array matches both exposure maps and hashes. Four update250/4000 checkpoint metadata,normalizers,schema,paired-prefix exposures,parameter counts,finite tensor bytes,file hashes and state hashes were independently verified without importing torch or repository core. The250/4000 states differ per model;training performs no model selection,DDIM or physics.", "",
    "## Scope limits", ""] + ["- " + limit for limit in result["limitations"]] + ["",
    "The companion JSON records UTC,argv,exit,temporary audit-script/source identity,input SHA inventory,per-stream findings,pair hierarchy,checkpoint identities and phase receipts. Audit tooling notes separately preserve an overlength dispatch rejection and a corrected receipt-path lookup error;neither changed an experiment record. Default C.1,DATA_LIMITED and deployment=NOT_MET remain in force."]
if errors:
    lines += ["", "## Discrepancies", ""] + [f"- {e['kind']}: {e['detail']}" for e in errors]
with md_path.open("w", encoding="utf8") as handle:
    handle.write("\n".join(lines) + "\n")
print(json.dumps({"verdict": result["verdict"], "checks": result["check_total"], "errors": errors,
    "dataset": result["dataset"], "teacher_endpoint_counts": dict(endpoint_counts), "documents": [str(json_path), str(md_path)]}, ensure_ascii=False, indent=2))
sys.exit(result["exit_code"])
