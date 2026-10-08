"""C.3 finite TRAIN search-effect teacher, with immutable evidence provenance.

Route-quality labels describe routes. Search-effect labels describe the exact
two-seed combination actually tested by the unchanged eight-slot optimizer.
Neither nominal evidence nor lineage is an actual execution certificate.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import numpy as np

from .continuous_route_optimizer import rank_candidates
from .preference_teacher_dataset import (ConditionNormalizer, DATASET_SCHEMA,
    EVIDENCE_TIERS, _coverage, _fact, _validate_actual, classify_evidence,
    deduplicate_candidates, encode_condition, load_preference_dataset,
    physical_candidate_key, preference_labels)
from .preference_warmstart_protocol import validate_learning_splits
from .residual_dataset import ResidualNormalizer
from .route_initializers import (OVERRIDE_CONDITIONS, RetrievalInitializer,
    json_raw, raw_seed_plan, search_mask)
from .route_optimizer_protocol import (ROOT, VERSIONS, digest, initial_candidates,
    parameter_plan, read, sha, write)
from .task_anchored_reference import TaskAnchoredResidualPlan, build_reference_definition
from .task_protocol import TaskSpec

C2_RELEASE = ROOT / "v6_4/releases/preference_warmstart_20261008_01"
C1_RELEASE = ROOT / "v6_4/releases/continuous_route_optimizer_20261007_01"
HISTORY_SCHEMA = "v64_c3_train_history_v1"
PAIR_SCHEMA = "v64_c3_frozen_teacher_seed_pairs_v1"
TEACHER_BUDGET = 8
TRAIN_TASK_COUNT = 6
TRAIN_MOTHER_COUNT = 3


class BoundArchive:
    """Resolve published paths to real bytes; a path string is never evidence.

    Raw local files omitted from portable releases remain usable only if their
    bytes match the original publication SHA. Missing files are reported and
    never replaced by another candidate, task, or newer archive.
    """
    def __init__(self, release):
        self.release = Path(release).resolve()
        self.mapping = read(self.release / "portable_paths.json")
        self.by_relative = {}
        for original, row in self.mapping.items():
            relative = row.get("path")
            if not relative:
                name = str(original).replace("\\", "/")
                marker = "/v6_4/output/"
                if marker in name:
                    relative = "snapshot/" + name.split(marker, 1)[1].split("/", 1)[1]
            if relative:
                self.by_relative[relative] = (original, row)
        self.inventory = []
        self._cache = {}

    def path(self, relative, expected=None, required=True):
        relative = str(relative).replace("\\", "/")
        if not relative.startswith("snapshot/"):
            relative = "snapshot/" + relative
        if relative not in self.by_relative:
            if required:
                raise ValueError("unbound publication path: " + relative)
            return None
        original, row = self.by_relative[relative]
        bound_sha = row["sha256"]
        if expected is not None and expected != bound_sha:
            raise ValueError("publication and evidence SHA disagree: " + relative)
        key = (relative, bound_sha)
        if key in self._cache:
            path = self._cache[key]
            if path is None and required:
                raise FileNotFoundError("bound evidence missing: " + relative)
            return path
        paths = [self.release / relative, Path(original)]
        found = next((p for p in paths if p.is_file()), None)
        status = "MISSING"
        if found is not None:
            if sha(found) != bound_sha:
                raise ValueError("published evidence SHA mismatch: " + str(found))
            found = found.resolve()
            status = "PORTABLE_SHA_VERIFIED" if self.release in found.parents else "LOCAL_ARCHIVE_SHA_VERIFIED"
        self.inventory.append({"release": str(self.release), "release_relative_path": relative,
            "original_path": str(original), "resolved_path": str(found) if found else None,
            "sha256": bound_sha, "availability": status})
        self._cache[key] = found
        if found is None and required:
            raise FileNotFoundError("bound evidence missing: " + relative)
        return found

    def read(self, relative):
        return read(self.path(relative))

    def verify_tree_seal(self, relative):
        manifest = self.read(relative.rstrip("/") + "/manifest.json")
        missing = []
        for name, expected in manifest.items():
            if self.path(relative.rstrip("/") + "/" + name, expected, required=False) is None:
                missing.append(name)
        return {"complete": not missing, "missing": missing, "sealed_file_count": len(manifest)}


def _load_tasks(run):
    run = Path(run)
    manifest = read(run / "learning_split_manifest.json")
    splits = validate_learning_splits(manifest)
    tasks = {row["task_id"]: TaskSpec.from_dict(read(run / row["task_path"])) for row in manifest["tasks"]}
    for row in manifest["tasks"]:
        if tasks[row["task_id"]].sha256() != row["task_sha256"]:
            raise ValueError("external Task SHA mismatch")
    return manifest, splits, tasks


def _verify_saved_files(run, result):
    for name, expected in result.get("files", {}).items():
        if sha(Path(run) / name) != expected:
            raise ValueError("immutable C.3 artifact changed: " + name)
    return result


def deduplicate_physical_facts(facts):
    """Deduplicate physical plans while retaining all evidence/alias views."""
    result = deduplicate_candidates(facts)
    originals = {}
    for fact in facts:
        originals.setdefault(fact["physical_candidate_id"], []).append(fact)
    for fact in result:
        for field in ("actual_bindings", "source_inventory"):
            combined = []
            for original in originals[fact["physical_candidate_id"]]:
                for row in original.get(field, []):
                    if row not in combined:
                        combined.append(copy.deepcopy(row))
            fact[field] = combined
    return result


def import_history(run, release=C2_RELEASE, c1_release=C1_RELEASE):
    """Import precisely C.2's original TRAIN pool, without new physics.

    Rechecks candidate bindings, published source files, nominal raw seals,
    and the five historical actual gates. Historical VAL/TEST are excluded
    before candidate processing. Missing raw bytes downgrade evidence honestly.
    """
    run = Path(run).resolve()
    target = run / "historical_import/history.json"
    if target.exists():
        return _verify_saved_files(run, read(target))
    external, splits, tasks = _load_tasks(run)
    archive, c1 = BoundArchive(release), BoundArchive(c1_release)
    old_splits = archive.read("learning_split_manifest.json")
    original_train = [r for r in old_splits["tasks"] if r["split"] == "train"]
    old_train = {r["task_sha256"]: r for r in original_train}
    if len(old_train) != TRAIN_TASK_COUNT or len({r["mother_source_sha256"] for r in original_train}) != TRAIN_MOTHER_COUNT:
        raise ValueError("C.2 TRAIN pool is not six Tasks / three mother scenes")
    if {r["task_sha256"] for r in external["tasks"] if r["split"] == "train"} != set(old_train):
        raise ValueError("C.3 TRAIN must equal original C.2 TRAIN exactly")
    dataset_manifest = archive.read("dataset/manifest.json")
    fact_path = archive.path("dataset/candidate_facts.json", dataset_manifest["files"]["candidate_facts.json"])
    candidates = read(fact_path)
    all_facts, missing = [], []
    registry_cache = {}
    source_c2 = archive.read("source_identity.json")
    for frozen in original_train:
        tid, task_sha = frozen["task_id"], frozen["task_sha256"]
        task = tasks[tid]
        old_task = TaskSpec.from_dict(archive.read(frozen["task_path"]))
        if old_task.sha256() != task.sha256():
            raise ValueError("historical TaskSpec changed")
        origin = c1 if frozen.get("historical") else archive
        registry_relative = f"planning/{tid}/candidate_registry.json" if frozen.get("historical") else f"teacher_predictions/{tid}/planning/{tid}/candidate_registry.json"
        registry_path = origin.path(registry_relative)
        rows = origin.read(registry_relative)
        registry_cache[tid] = (origin, registry_relative, rows)
        for imported in [f for f in candidates if f["task_sha256"] == task_sha]:
            if imported.get("split") != "train" or imported["mother_id"] != frozen["mother_id"]:
                raise ValueError("C.2 fact grouping disagrees with authoritative TRAIN split")
            plan = TaskAnchoredResidualPlan.from_dict(imported["plan"])
            matches = [r for r in rows if r.get("plan_sha256") == plan.sha256()]
            if not matches:
                raise ValueError("C.2 fact has no bound original candidate registry row")
            row = matches[0]
            if (digest(row["execution_identity"]) != digest(imported["execution_identity"]) or
                    digest(row.get("prediction_metrics")) != digest(imported.get("prediction_metrics"))):
                raise ValueError("C.2 fact contradicts original physical candidate")
            root = registry_relative.rsplit("/", 1)[0] + "/predictions/" + row["candidate_id"]
            nominal = origin.verify_tree_seal(root)
            bindings = []
            for previous in imported.get("actual_bindings", []):
                method = previous["method"]
                try:
                    slot = origin.read(f"actual/{tid}/{method}/slot.json")
                    bound = _validate_actual(origin, origin.release / "snapshot", tid, slot, task, plan, row)
                    evidence_method = bound["evidence_method"]
                    actual_root = f"actual/{tid}/{evidence_method}"
                    actual_seal = origin.verify_tree_seal(actual_root)
                    bound["raw_archive_seal"] = actual_seal
                    bound["independently_bound_and_validated"] &= actual_seal["complete"]
                    bindings.append(bound)
                except FileNotFoundError as error:
                    bindings.append({"method": method, "independently_bound_and_validated": False,
                        "missing_evidence": str(error)})
            fact = _fact(row, task, splits[task_sha], bindings,
                imported.get("producer_commit"), [{"path": str(registry_path), "sha256": sha(registry_path)}])
            fact["historical_evidence_tier"] = imported["evidence_tier"]
            fact["nominal_archive_seal"] = nominal
            fact["historical_source_inventory"] = imported.get("source_inventory", [])
            if not nominal["complete"]:
                fact["evidence_tier"] = "MISSING_OR_UNBOUND"
                missing.append({"task_id": tid, "plan_sha256": plan.sha256(), "missing": nominal["missing"]})
            all_facts.append(fact)
    facts = deduplicate_physical_facts(all_facts)
    if len(facts) > 72:
        raise ValueError("original C.2 TRAIN physical candidate bound exceeded")
    labels, buckets = preference_labels(facts, original_train)
    for sample in labels:
        sample.update(source_types=["route_quality_example"], source_refs=[{
            "source_type": "route_quality_example", "physical_candidate_id": sample["physical_candidate_id"],
            "original_C2_TRAIN": True}])
    output = target.parent
    write(output / "candidate_facts.json", facts)
    write(output / "route_quality_labels.json", labels)
    write(output / "bucket_coverage.json", buckets)
    write(output / "source_inventory.json", archive.inventory + c1.inventory)
    result = {"schema": HISTORY_SCHEMA, "original_train_tasks": original_train,
        "candidates": facts, "route_quality_labels": labels, "buckets": buckets,
        "original_C2_candidate_count": len(candidates), "candidate_count_unique": len(facts),
        "excluded_C2_VAL_TEST_candidate_count": sum(f["task_sha256"] not in old_train for f in candidates),
        "train_mothers": sorted({r["mother_id"] for r in original_train}),
        "evidence_tier_counts": dict(Counter(f["evidence_tier"] for f in facts)),
        "missing_evidence": missing, "new_physics_steps": 0, "new_actual_executions": 0,
        "historical_actual_producer": source_c2["historical_actual_producer"],
        "source_release": str(Path(release).resolve()), "source_release_manifest_sha256": sha(Path(release) / "release_manifest.json"),
        "learning_split_manifest_sha256": sha(run / "learning_split_manifest.json"),
        "files": {p.relative_to(run).as_posix(): sha(p) for p in output.iterdir() if p.is_file() and p != target}}
    write(target, result)
    return result


def _normalized_labels(labels, tasks, split_manifest):
    keys = sorted({(s["task_id"], s["preference"], s["family"]) for s in labels})
    scaler = ConditionNormalizer.fit([(tasks[t], build_reference_definition(tasks[t], version=VERSIONS[f]), p, f)
        for t, p, f in keys], split_manifest)
    result = copy.deepcopy(labels)
    for s in result:
        encoded = encode_condition(tasks[s["task_id"]], build_reference_definition(tasks[s["task_id"]], version=VERSIONS[s["family"]]), s["preference"], s["family"])
        s["condition_raw"] = encoded["values"].tolist()
        s["condition_normalized"] = scaler.transform_condition(encoded).tolist()
    return result, scaler


def _rank_fact(fact):
    return {**fact["prediction"], "candidate_id": min(fact["original_candidate_ids"]),
        "family": fact["family"], "x_m": np.asarray(fact["z_m"])[np.asarray(fact["search_interval_mask"], dtype=bool)].reshape(-1).tolist(),
        "physical_candidate_id": fact["physical_candidate_id"]}


def historical_references(facts, train_rows):
    result = {}
    for frozen in train_rows:
        pool = [_rank_fact(f) for f in facts if f["task_sha256"] == frozen["task_sha256"] and
            f["evidence_tier"] in ("PREDICTED_COMPLETE", "VALIDATED_EXECUTION")]
        ranks = rank_candidates(pool)
        result[frozen["task_id"]] = {pref: {"physical_candidate_id": row["physical_candidate_id"],
            "prediction_metrics": row["prediction_metrics"], "plan_sha256": row["plan_sha256"]} if row else None
            for pref, row in (("A", ranks["tie_selected"]), ("B", ranks["B"]))}
    return result


def freeze_teacher_pairs(run, history=None):
    run = Path(run).resolve()
    target = run / "teacher_search/frozen_seed_pairs.json"
    if target.exists():
        result = read(target)
        body = {k: v for k, v in result.items() if k != "content_sha256"}
        if digest(body) != result["content_sha256"] or sha(run / "historical_import/history.json") != result["history_sha256"]:
            raise ValueError("frozen teacher pairs or history changed")
        return result
    history = history or import_history(run)
    manifest, _, tasks = _load_tasks(run)
    labels, scaler = _normalized_labels(history["route_quality_labels"], tasks, manifest)
    facts_by_id = {f["physical_candidate_id"]: f for f in history["candidates"]}
    pairs = []
    for frozen in history["original_train_tasks"]:
        tid, mother = frozen["task_id"], frozen["mother_id"]
        task = tasks[tid]
        local, origins = {}, {}
        rules = initial_candidates(task)
        for slot, (pref, family) in OVERRIDE_CONDITIONS.items():
            candidates = [s for s in labels if s["task_sha256"] == task.sha256() and s["preference"] == pref and s["family"] == family]
            ranked = rank_candidates([_rank_fact(facts_by_id[s["physical_candidate_id"]]) for s in candidates])
            winner = ranked["tie_selected" if pref == "A" else "B"]
            if winner:
                sample = next(s for s in candidates if s["physical_candidate_id"] == winner["physical_candidate_id"])
                z = sample["z_m"]
                origin = {"construction_status": "LOCAL_QUALIFIED_ROUTE_TEACHER", "route_quality_sample_id": sample["sample_id"],
                    "physical_candidate_id": sample["physical_candidate_id"], "source_mother_id": mother}
            else:
                z = parameter_plan(task, family, rules[slot]["x_m"]).z_m.tolist()
                origin = {"construction_status": "RULE_CONSTRUCTION_NO_MATCHING_TEACHER", "source_mother_id": mother}
            local[slot] = {"source": "retrieval", "preference": pref, "family": family,
                "raw_z_m": z, "teacher_construction": origin}
            origins[str(slot)] = origin
        transfer_labels = [s for s in labels if s["mother_id"] != mother]
        if any(s["mother_id"] == mother for s in transfer_labels):
            raise AssertionError("leave-one-mother exclusion failed")
        retrieval = RetrievalInitializer(transfer_labels, scaler, {"leave_out_mother_id": mother,
            "initial_historical_pool_only": True})
        transfer = retrieval(task)
        for proposal in transfer.values():
            proposal["teacher_construction"] = {"construction_status": "TRANSFER_LEAVE_ONE_MOTHER",
                "excluded_mother_id": mother, "eligible_mothers": sorted({s["mother_id"] for s in transfer_labels})}
        for name, proposals in (("T_local", local), ("T_transfer", transfer)):
            pair = {"task_id": tid, "task_sha256": task.sha256(), "mother_id": mother,
                "combination": name, "proposals": json_raw(proposals),
                "common_rule_seeds": [rules[0], rules[2]], "candidate_budget": TEACHER_BUDGET,
                "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH", "training_only": True,
                "raw_qualification": {str(slot): raw_seed_plan(task, proposal, slot)[1] for slot, proposal in proposals.items()}}
            pair["seed_pair_id"] = digest(pair)
            pairs.append(pair)
    result = {"schema": PAIR_SCHEMA, "history_sha256": sha(run / "historical_import/history.json"),
        "all_pairs_frozen_before_new_search": True, "pairs": pairs,
        "historical_references": historical_references(history["candidates"], history["original_train_tasks"]),
        "condition_scaler": scaler.to_dict(), "maximum_new_candidate_slots": 96,
        "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH"}
    result["content_sha256"] = digest(result)
    write(target, result)
    return result


def _complete(row):
    m = row.get("prediction_metrics") or {}
    return bool(row.get("prediction_admissible") is True and row.get("prediction_task_passed") is True and
        row.get("online_guards_passed") is True and row.get("prediction_steps") == 13500 and
        m.get("native_state_count") == 13501 and type(m.get("saved_horizon_s")) in (float, int) and
        abs(m["saved_horizon_s"] - 27.) < 1e-8 and
        all(type(m.get(k)) in (float, int) and np.isfinite(m[k]) for k in ("I_support", "L_full")))


def preference_qualified(row, preference):
    if not _complete(row):
        return False
    metrics = row["prediction_metrics"]
    return preference == "A" or (metrics.get("clearance_status") == "MEASURED" and
        type(metrics.get("d_support")) in (int, float) and np.isfinite(metrics["d_support"]) and metrics["d_support"] >= .030)


def near_reference(row, reference, preference):
    if reference is None or not preference_qualified(row, preference):
        return False
    m, r = row["prediction_metrics"], reference["prediction_metrics"]
    return bool(m["L_full"] <= r["L_full"] + .005 and
        (preference == "B" or m["I_support"] <= r["I_support"] + .001))


def summarize_teacher_pair(pair, selection, rows, references, proposals=None):
    """Strict endpoints, right-censored first hits, and seed participation.

    A legal duplicate is kept in proposals despite consuming no extra slot.
    A direct qualified seed may earn supervision; descendants count only when
    a qualified descendant from the same original slot is actually selected.
    """
    if len(rows) > TEACHER_BUDGET or selection["budget"]["candidate_budget"] != TEACHER_BUDGET:
        raise ValueError("teacher search exceeds frozen eight-slot budget")
    proposals = proposals or rows
    endpoints, labels = {}, []
    by_id = {r["candidate_id"]: r for r in rows}
    for slot, (pref, family) in OVERRIDE_CONDITIONS.items():
        selected_id = selection["preferences"][pref].get("source_candidate_id")
        selected = by_id.get(selected_id)
        qualified = selected is not None and preference_qualified(selected, pref)
        reference = references.get(pref)
        first = next((i + 1 for i, r in enumerate(rows) if preference_qualified(r, pref)), None)
        near = next((i + 1 for i, r in enumerate(rows) if near_reference(r, reference, pref)), None) if reference else None
        seed = next((p for p in proposals if p.get("initializer_slot") == slot), None)
        raw_legal = bool(seed and (seed.get("raw_seed_diagnostics") or {}).get("raw_legal"))
        seed_id = seed.get("cache_hit_candidate_id") or seed.get("candidate_id") if seed else None
        if seed_id is None and seed:
            seed_id = next((r["candidate_id"] for r in rows if r.get("plan_sha256") == seed.get("plan_sha256") or
                r.get("search_content_key") == seed.get("content_key")), None)
        # An exact cache alias of common slot0/2 is still the common rule
        # candidate, not a separately tested initializer contribution.
        direct = (raw_legal and seed_id in by_id and
            by_id[seed_id].get("initializer_slot") == slot and preference_qualified(by_id[seed_id], pref))
        lineage = selected.get("proposal_lineage", []) if selected else []
        from_slot = any(p.get("initial_position") == slot for p in lineage)
        common_only = bool(qualified and not from_slot and not direct)
        construction = pair["proposals"][str(slot)].get("teacher_construction", {})
        rule_construction = construction.get("construction_status") == "RULE_CONSTRUCTION_NO_MATCHING_TEACHER"
        positive = bool(raw_legal and qualified and (direct or from_slot) and not rule_construction)
        endpoints[pref] = {"qualified_endpoint": qualified, "near_historical_reference": bool(qualified and near_reference(selected, reference, pref)),
            "historical_reference_available": reference is not None, "selected_candidate_id": selected_id,
            "prediction_metrics": selected.get("prediction_metrics") if selected else None,
            "first_qualified_slot": first, "first_qualified_right_censored_budget": len(rows) if first is None else None,
            "first_near_quality_slot": near, "first_near_quality_status": "N/A_NO_HISTORICAL_REFERENCE" if reference is None else "HIT" if near else "RIGHT_CENSORED",
            "first_near_quality_right_censored_budget": len(rows) if reference and near is None else None,
            "raw_legal": raw_legal, "direct_seed_qualified": bool(direct), "selected_seed_or_descendant": bool(from_slot),
            "rule_only": common_only, "positive_supervision": positive,
            "label_status": "INITIALIZER_EFFECT_EVIDENCED" if positive else "RULE_ONLY" if common_only else "NO_QUALIFIED_INITIALIZER_EFFECT_LABEL"}
        if positive:
            labels.append({"task_id": pair["task_id"], "task_sha256": pair["task_sha256"],
                "mother_id": pair["mother_id"], "group_id": pair["mother_id"], "split": "train",
                "preference": pref, "family": family, "reference_family": family,
                "reference_version": VERSIONS[family], "z_m": copy.deepcopy(pair["proposals"][str(slot)]["raw_z_m"]),
                "source_types": ["initializer_effect_example"], "source_refs": [{"source_type": "initializer_effect_example",
                    "seed_pair_id": pair["seed_pair_id"], "combination": pair["combination"], "slot": slot,
                    "seed_candidate_id": seed_id, "selected_candidate_id": selected_id,
                    "direct_seed_qualified": bool(direct), "selected_seed_or_descendant": bool(from_slot),
                    "fixed_partner": pair["proposals"][str(3 if slot == 1 else 1)],
                    "shared_search_is_not_independent_counterfactuals": True}],
                "evidence_tier": "PREDICTED_COMPLETE", "label_scope": "finite_shared_search_initializer_effect",
                "prediction_metrics": selected["prediction_metrics"], "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH"})
    return {"seed_pair_id": pair["seed_pair_id"], "task_id": pair["task_id"], "combination": pair["combination"],
        "endpoints": endpoints, "initializer_effect_labels": labels, "budget": selection["budget"],
        "protocol_completed": selection["budget"].get("stop_reason") != "TOOL_ERROR",
        "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH"}


def compare_teacher_pairs(summaries):
    """The frozen hierarchical rule, retaining ties and matched hit sets."""
    if not summaries:
        return []
    survivors = list(summaries)
    for key in ("qualified_endpoint", "near_historical_reference"):
        best = max(sum(e[key] for e in s["endpoints"].values()) for s in survivors)
        survivors = [s for s in survivors if sum(e[key] for e in s["endpoints"].values()) == best]
    # Never compare a fast A hit to a fast B hit: qualification sets must match.
    groups = {}
    for s in survivors:
        achieved = tuple(p for p, e in s["endpoints"].items() if e["qualified_endpoint"])
        near = tuple(p for p, e in s["endpoints"].items() if e["near_historical_reference"])
        key = (achieved, near)
        fields = [(p, "first_near_quality_slot" if p in near else "first_qualified_slot") for p in achieved]
        score = sum(s["endpoints"][p][f] if s["endpoints"][p][f] is not None else TEACHER_BUDGET + 1 for p, f in fields)
        groups.setdefault(key, []).append((score, s))
    return [s["seed_pair_id"] for group in groups.values() for score, s in group if score == min(x[0] for x in group)]


def run_teacher_search(run, stream_runner=None):
    """Run twelve frozen eight-slot searches; never append teacher actuals."""
    run = Path(run).resolve()
    pairs = freeze_teacher_pairs(run)
    target = run / "teacher_search/results.json"
    if target.exists():
        result = read(target)
        if (digest({k: v for k, v in result.items() if k != "content_sha256"}) != result["content_sha256"] or
                result["pairs_sha256"] != sha(run / "teacher_search/frozen_seed_pairs.json")):
            raise ValueError("teacher search result seal changed")
        for summary in result["summaries"]:
            _verify_saved_files(run, {"files": summary["source_files"]})
        return result
    if stream_runner is None:
        from .search_aware_warmstart_experiment import run_stream
        stream_runner = run_stream
    frozen_by_id = {r["task_id"]: r for r in read(run / "plan.json")["tasks"]}
    summaries = []
    for pair in pairs["pairs"]:
        frozen = frozen_by_id[pair["task_id"]]
        if frozen["split"] != "train":
            raise ValueError("teacher stream Task outside external TRAIN")
        proposals = {int(k): copy.deepcopy(v) for k, v in pair["proposals"].items()}
        stream = stream_runner(run, frozen, pair["combination"], TEACHER_BUDGET, proposals=proposals, stage="teacher")
        root = Path(stream["path"])
        planning = root / "planning" / pair["task_id"]
        selection = stream["selection"]
        rows = read(planning / "candidate_registry.json")
        summary = summarize_teacher_pair(pair, selection, rows, pairs["historical_references"][pair["task_id"]], read(planning / "proposals.json"))
        summary["stream_path"] = str(root)
        summary["source_files"] = {str(p.relative_to(run).as_posix()): sha(p) for p in
            (planning / "candidate_registry.json", planning / "proposals.json", planning / "selection.json")}
        effect_path = root / "teacher_effect.json"
        if effect_path.exists():
            if digest(read(effect_path)) != digest(summary):
                raise ValueError("retained teacher effect summary changed")
        else:
            write(effect_path, summary)
        summaries.append(summary)
    winners = {tid: compare_teacher_pairs([s for s in summaries if s["task_id"] == tid]) for tid in frozen_by_id if frozen_by_id[tid]["split"] == "train"}
    result = {"schema": "v64_c3_teacher_search_results_v1", "pairs_sha256": sha(run / "teacher_search/frozen_seed_pairs.json"),
        "summaries": summaries, "selected_teacher_pair_ids": winners,
        "slots_consumed": sum(s["budget"]["slots_consumed"] for s in summaries),
        "maximum_candidate_slots": 96, "actual_logical_slots": 0,
        "formal_actual_validation": "NOT_RUN_TEACHER_SEARCH",
        "protocol_completed": all(s["protocol_completed"] for s in summaries)}
    if result["slots_consumed"] > 96:
        raise ValueError("finite TRAIN teacher budget exceeded")
    result["content_sha256"] = digest(result)
    write(target, result)
    return result


def deduplicate_supervision(samples, tasks):
    """One Task/family/preference/z row, retaining every source reference."""
    unique = {}
    for original in samples:
        sample = copy.deepcopy(original)
        if sample["split"] != "train":
            raise ValueError("formal supervision is TRAIN only")
        task = tasks[sample["task_id"]]
        family, pref = sample["family"], sample["preference"]
        z = np.asarray(sample["z_m"], dtype=float).reshape(6, 2)
        mask = np.asarray(search_mask(task), dtype=bool)
        if not np.isfinite(z).all() or np.any(z[~mask] != 0.) or np.any(np.linalg.norm(z, axis=1) > .020):
            raise ValueError("supervision violates raw residual space")
        if sample["task_sha256"] != task.sha256():
            raise ValueError("supervision task binding differs")
        definition = build_reference_definition(task, version=VERSIONS[family])
        semantic = {"task_sha256": task.sha256(), "family": family, "preference": pref, "z_m": z.tolist()}
        key = digest(semantic)
        sample.update(sample_id=key, z_m=z.tolist(), search_interval_mask=mask.tolist(),
            reference_interval_mask=definition["interval_mask"], bucket_id=digest({k: semantic[k] for k in ("task_sha256", "family", "preference")}),
            plan_sha256=TaskAnchoredResidualPlan.from_definition(definition, z).sha256())
        sample.setdefault("physical_candidate_id", "initializer:" + key)
        sample.setdefault("source_inventory", [])
        if key in unique:
            existing = unique[key]
            existing["source_types"] = sorted(set(existing["source_types"] + sample["source_types"]))
            for ref in sample["source_refs"]:
                if ref not in existing["source_refs"]:
                    existing["source_refs"].append(ref)
        else:
            unique[key] = sample
    return [unique[k] for k in sorted(unique)]


def build_dataset(run):
    """Create a shared TRAIN-only D/S/N dataset in the original C.2 format."""
    run = Path(run).resolve()
    output = run / "dataset"
    target = output / "manifest.json"
    if target.exists():
        return load_search_aware_dataset(target)
    history = import_history(run)
    manifest, _, tasks = _load_tasks(run)
    results = read(run / "teacher_search/results.json")
    if digest({k: v for k, v in results.items() if k != "content_sha256"}) != results["content_sha256"]:
        raise ValueError("teacher result seal differs")
    if not results["protocol_completed"]:
        raise ValueError("incomplete technical teacher run cannot be hidden in formal dataset")
    pairs_path = run / "teacher_search/frozen_seed_pairs.json"
    if sha(pairs_path) != results["pairs_sha256"]:
        raise ValueError("teacher frozen pairs changed")
    samples = copy.deepcopy(history["route_quality_labels"])
    for summary in results["summaries"]:
        for name, expected in summary["source_files"].items():
            if sha(run / name) != expected:
                raise ValueError("teacher trace/source changed")
        if summary["seed_pair_id"] in results["selected_teacher_pair_ids"][summary["task_id"]]:
            samples.extend(summary["initializer_effect_labels"])
    samples = deduplicate_supervision(samples, tasks)
    if not samples:
        raise ValueError("NO_LEGAL_TRAIN_SUPERVISION: no checkpoint may be fabricated")
    samples, condition = _normalized_labels(samples, tasks, manifest)
    # Multi-preference/source views do not duplicate residual scaler statistics.
    residual_unique = {digest({"task": s["task_sha256"], "family": s["family"], "z": s["z_m"]}): s for s in samples}
    residual = ResidualNormalizer.fit([s["z_m"] for s in residual_unique.values()],
        [np.asarray(s["search_interval_mask"], dtype=bool) for s in residual_unique.values()])
    output.mkdir(parents=True, exist_ok=True)
    write(output / "labels.json", samples)
    write(output / "candidate_facts.json", history["candidates"])
    write(output / "condition_scaler.json", condition.to_dict())
    write(output / "residual_scaler.json", residual.to_dict())
    write(output / "bucket_coverage.json", history["buckets"])
    write(output / "condition_schema.json", {"schema": condition.to_dict()["schema"], "dimension": len(condition.names),
        "feature_names_with_units": condition.names, "literal": condition.literal.tolist(), "ids_or_outcomes_encoded": False})
    supported = [{"preference": pref, "reference_family": family,
        "train_labels": sum(s["preference"] == pref and s["family"] == family for s in samples),
        "status": "DATA_LIMITED" if any(s["preference"] == pref and s["family"] == family for s in samples) else "UNSUPPORTED_TRAINING_CONDITION"}
        for pref, family in (("A", "v1"), ("B", "v2"))]
    effects = sum("initializer_effect_example" in s["source_types"] for s in samples)
    dataset_manifest = {"schema": DATASET_SCHEMA, "c3_schema": "v64_c3_shared_train_supervision_v1",
        "learning_split_manifest_path": "../learning_split_manifest.json",
        "learning_split_manifest_sha256": sha(run / "learning_split_manifest.json"), "tasks": manifest["tasks"],
        "supervision_available": True, "scalers_train_only": True, "test_leakage_checks_passed": True,
        "candidate_count_unique": len(history["candidates"]), "label_count_views": len(samples),
        "zero_label_views": sum(not np.any(s["z_m"]) for s in samples),
        "route_quality_label_count": sum("route_quality_example" in s["source_types"] for s in samples),
        "initializer_effect_label_count": effects, "search_effect_teacher_evidence_available": effects > 0,
        "search_effect_status": "FINITE_SEARCH_EFFECT_EVIDENCE_AVAILABLE" if effects else "SEARCH_EFFECT_SUPERVISION_NOT_ESTABLISHED",
        "data_status": "DATA_LIMITED", "supported_preference_family_conditions": supported,
        "coverage": _coverage(history["candidates"], samples),
        "teacher_results_sha256": sha(run / "teacher_search/results.json"),
        "history_sha256": sha(run / "historical_import/history.json"),
        "training_distribution": "uniform mother then Task then supported preference/family then 1:1 source type if both then unique original reference",
        "D_S_N_identical_pool": True, "source_inventory": history["files"],
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name not in ("manifest.json", "dataset_manifest.json")}}
    write(target, dataset_manifest)
    write(output / "dataset_manifest.json", dataset_manifest)
    return load_search_aware_dataset(target)


def load_search_aware_dataset(path):
    dataset = load_preference_dataset(path)
    if dataset.manifest.get("c3_schema") != "v64_c3_shared_train_supervision_v1" or any(s["split"] != "train" for s in dataset.samples):
        raise ValueError("C.3 requires its immutable TRAIN-only shared supervision pool")
    if set(dataset.condition_scaler.fit_task_sha256) - {r["task_sha256"] for r in dataset.manifest["tasks"] if r["split"] == "train"}:
        raise ValueError("condition scaler fitted outside TRAIN")
    return dataset
