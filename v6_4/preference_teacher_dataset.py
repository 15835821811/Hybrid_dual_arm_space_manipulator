"""Physical-candidate facts, evidence tiers and family-specific C.2 labels.

This module reads declarations and sealed evidence only. It never simulates,
repairs residuals, or treats preference-row duplication as extra trajectories.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .preference_warmstart_protocol import (C1_PRODUCER, HISTORY, ROOT,
    search_interval_mask, validate_learning_splits)
from .route_optimizer_protocol import digest, read, sha, write, VERSIONS
from .residual_dataset import (ResidualNormalizer, encode_residual_condition, _array)
from .task_anchored_reference import TaskAnchoredResidualPlan, build_reference_definition
from .task_protocol import TaskSpec

CONDITION_SCHEMA = "v64_c2_preference_family_search_condition_v1"
DATASET_SCHEMA = "v64_c2_physical_candidate_preference_dataset_v1"
EVIDENCE_TIERS = ("VALIDATED_EXECUTION", "PREDICTED_COMPLETE", "FAILED_OR_INCOMPLETE", "MISSING_OR_UNBOUND")


class PortableEvidence:
    """Read published bytes by portable path and check original publication SHA."""
    def __init__(self, release=HISTORY):
        self.release = Path(release).resolve()
        self.mapping = read(self.release / "portable_paths.json")
        self.by_relative = {r["path"]: r for r in self.mapping.values() if r.get("available") and r.get("path")}
        self.inventory = []

    def path(self, value):
        name = str(value).replace("\\", "/")
        mapped = self.mapping.get(str(value))
        if mapped is None:
            mapped = next((v for k, v in self.mapping.items() if k.replace("\\", "/") == name), None)
        if mapped is not None:
            if not mapped.get("available") or not mapped.get("path"):
                raise FileNotFoundError("published evidence bytes unavailable: " + str(value))
            relative, expected = mapped["path"], mapped["sha256"]
        else:
            relative = name
            if not relative.startswith("snapshot/"):
                relative = "snapshot/" + relative
            mapped = self.by_relative.get(relative)
            if mapped is None:
                raise ValueError("evidence is not bound by portable publication mapping: " + relative)
            expected = mapped["sha256"]
        path = (self.release / relative).resolve()
        if self.release not in path.parents or sha(path) != expected:
            raise ValueError("portable evidence SHA mismatch: " + relative)
        row = {"release_relative_path": relative, "sha256": expected}
        if row not in self.inventory:
            self.inventory.append(row)
        return path

    def read(self, value):
        return read(self.path(value))


def physical_candidate_key(task_sha256, execution_identity, reference_version, plan_sha256):
    required = ("source_identity_sha256", "config_sha256", "run_config_sha256", "model_contract_sha256", "task_sha256")
    if not all(execution_identity.get(k) for k in required) or execution_identity["task_sha256"] != task_sha256:
        raise ValueError("missing or mismatched physical execution/model/config identity")
    return digest({"task_sha256": task_sha256, "execution_identity": execution_identity,
        "reference_version": reference_version, "plan_sha256": plan_sha256})


def _finite(value):
    return type(value) in (float, int) and np.isfinite(value)


def _prediction_binding(row, task, plan):
    metrics = row.get("prediction_metrics") or {}
    binding = metrics.get("consumed_reference_binding") or {}
    identity = row.get("execution_identity") or {}
    return bool(row.get("plan_sha256") == plan.sha256() and
        plan.definition.get("task_sha256") == task.sha256() and
        identity.get("task_sha256") == task.sha256() and
        identity.get("model_contract_sha256") == task.model_contract_sha256 and
        binding.get("available") is True and binding.get("passed") is True and
        binding.get("plan_sha256") == plan.sha256() and
        binding.get("definition_sha256") == plan.definition["definition_sha256"] and
        binding.get("representation_version") == plan.representation_version and
        np.array_equal(np.asarray(binding.get("z_m")), plan.z_m))


def classify_evidence(row, task, plan, actual_bindings=()):
    """Actual qualification and B clearance are deliberately independent."""
    identity = row.get("execution_identity") or {}
    if (row.get("plan_sha256") != plan.sha256() or identity.get("task_sha256") != task.sha256()
            or identity.get("model_contract_sha256") != task.model_contract_sha256 or row.get("tool_error")):
        return "MISSING_OR_UNBOUND"
    metrics = row.get("prediction_metrics") or {}
    complete = bool(row.get("prediction_rollout_started") is True and row.get("prediction_steps") == 13500 and
        row.get("prediction_task_passed") is True and row.get("online_guards_passed") is True and
        row.get("prediction_admissible") is True and
        _finite(metrics.get("saved_horizon_s")) and abs(metrics["saved_horizon_s"] - 27.) < 1e-8 and
        metrics.get("native_state_count") == 13501)
    if not complete:
        return "FAILED_OR_INCOMPLETE"
    if not _prediction_binding(row, task, plan):
        return "MISSING_OR_UNBOUND"
    if any(b.get("independently_bound_and_validated") is True for b in actual_bindings):
        return "VALIDATED_EXECUTION"
    return "PREDICTED_COMPLETE"


def _validate_actual(portable, snapshot, tid, slot, task, plan, registry_row):
    """Validate the five original independent gates and exact actual bindings."""
    source_method = slot.get("evidence_method") or slot.get("alias_of_method") or slot["method"]
    root = f"actual/{tid}/{source_method}"
    report = portable.read(root + "/attempt/actual/evaluation/report.json")
    attempt = portable.read(root + "/attempt/attempt_result.json")
    bound_plan = TaskAnchoredResidualPlan.from_dict(portable.read(root + "/attempt/plan.json"))
    binding = report.get("reference_binding") or {}
    consumed = binding.get("consumed_reference_identity") or {}
    gates = {k: (report.get(k) or {}).get("passed") is True for k in
        ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")}
    checks = {"slot_task": slot.get("task_sha256") == task.sha256(),
        "slot_plan": slot.get("plan_sha256") == plan.sha256(),
        "report_task": report.get("task_sha256") == task.sha256(),
        "actual_plan": bound_plan.sha256() == plan.sha256(),
        "report_plan": binding.get("plan_sha256") == plan.sha256(),
        "consumed_plan": consumed.get("available") is True and consumed.get("passed") is True and consumed.get("plan_sha256") == plan.sha256(),
        "report_definition": binding.get("definition_sha256") == plan.definition["definition_sha256"],
        "full_report": report.get("evidence_valid") is True and report.get("full_task_success") is True,
        "full_steps": report.get("actual_physics_steps") == 13500 and slot.get("actual_steps") == 13500,
        "full_horizon": _finite(report.get("actual_saved_horizon_s")) and abs(report["actual_saved_horizon_s"] - 27.) < 1e-8,
        "independent_replay": report.get("replayed_physics_steps") == 13500 and bool(report.get("fresh_replay_sha256")),
        "slot_full": slot.get("full_task_success") is True and slot.get("original_independent_gates_passed") is True,
        "prediction_actual": slot.get("prediction_actual_consistent") is True,
        "attempt_full": attempt.get("full_task_success") is True,
        "all_five_original_gates": all(gates.values())}
    quality = slot.get("quality") or {}
    predicted = registry_row.get("prediction_metrics") or {}
    checks["numeric_prediction_actual_consistency"] = all(_finite(quality.get(k)) and _finite(predicted.get(k)) and
        abs(quality[k] - predicted[k]) <= 1e-9 for k in ("I_support", "L_full", "d_support"))
    checks["actual_consumed_z"] = np.array_equal(np.asarray(consumed.get("z_m")), plan.z_m)
    return {"method": slot["method"], "evidence_method": source_method,
        "alias_of": slot.get("alias_of_method"), "independently_bound_and_validated": all(checks.values()),
        "checks": checks, "original_five_gates": gates, "quality": quality,
        "slot_sha256": sha(snapshot / "actual" / tid / slot["method"] / "slot.json")}


def _fact(row, task, split_row, actual_bindings=(), producer=None, source_inventory=None):
    plan = TaskAnchoredResidualPlan.from_dict(row["plan"])
    family = "v1" if plan.representation_version == VERSIONS["v1"] else "v2"
    if plan.representation_version not in VERSIONS.values():
        raise ValueError("unsupported reference family")
    mask = search_interval_mask(plan.definition)
    z = np.asarray(plan.z_m)
    if np.any(z[~mask] != 0):
        raise ValueError("teacher residual contains dimensions outside C.1 search mask")
    key = physical_candidate_key(task.sha256(), row["execution_identity"], plan.representation_version, plan.sha256())
    return {"physical_candidate_id": key, "task_id": task.task_id, "task_sha256": task.sha256(),
        "group_id": split_row["mother_id"], "mother_id": split_row["mother_id"],
        "mother_source_sha256": split_row["mother_source_sha256"], "split": split_row["split"],
        "family": family, "reference_family": family, "reference_version": plan.representation_version,
        "plan_sha256": plan.sha256(), "plan": plan.to_dict(), "z_m": z.tolist(),
        "reference_interval_mask": plan.interval_mask.tolist(), "search_interval_mask": mask.tolist(),
        "original_candidate_ids": [row["candidate_id"]], "parent_candidate_ids": [row.get("parent_candidate_id")],
        "source": row.get("source"), "producer_commit": producer,
        "execution_identity": row["execution_identity"], "prediction": row,
        "prediction_metrics": row.get("prediction_metrics"), "actual_bindings": list(actual_bindings),
        "evidence_tier": classify_evidence(row, task, plan, actual_bindings),
        "source_inventory": source_inventory or []}


def deduplicate_candidates(facts):
    result = {}
    for fact in facts:
        key = physical_candidate_key(fact["task_sha256"], fact["execution_identity"], fact["reference_version"], fact["plan_sha256"])
        if key in result:
            previous = result[key]
            for field in ("z_m", "prediction_metrics", "task_sha256", "split", "reference_version"):
                if previous[field] != fact[field]:
                    raise ValueError("same physical candidate has contradictory evidence")
            previous["original_candidate_ids"] = sorted(set(previous["original_candidate_ids"] + fact["original_candidate_ids"]))
            previous["parent_candidate_ids"] = list(dict.fromkeys(previous["parent_candidate_ids"] + fact["parent_candidate_ids"]))
        else:
            result[key] = dict(fact)
    return sorted(result.values(), key=lambda r: (r["task_sha256"], r["plan_sha256"], r["physical_candidate_id"]))


def import_c1(run, release=HISTORY):
    run, release = Path(run), Path(release)
    target = run / "historical_import" / "candidate_facts.json"
    if target.exists():
        return read(target)
    portable = PortableEvidence(release)
    snapshot = release / "snapshot"
    source = portable.read("source_identity.json")
    if source.get("algorithm_producer_commit") != C1_PRODUCER:
        raise ValueError("C.1 publication is not its actual algorithm producer")
    splits = validate_learning_splits(read(run / "learning_split_manifest.json"))
    teacher_path = portable.path("teacher_records.jsonl")
    teacher_rows = [json.loads(line) for line in teacher_path.read_text(encoding="utf8").splitlines() if line]
    plan = portable.read("plan.json")
    actual_complete = portable.read("actual_complete.json")
    facts, row_bindings, global_selections = [], [], {}
    for frozen in plan["tasks"]:
        tid = frozen["task_id"]
        task_path = portable.path(f"frozen_tasks/{tid}/task.json")
        task = TaskSpec.from_dict(read(task_path))
        split = splits[task.sha256()]
        if not split["historical"] or split["split"] != "train":
            raise ValueError("historical candidate leaked outside TRAIN")
        rows = portable.read(f"planning/{tid}/candidate_registry.json")
        global_selections[tid] = portable.read(f"planning/{tid}/selection.json")
        slots = []
        for method in ("Z0", "G0", "OI", "OC"):
            path = portable.path(f"actual/{tid}/{method}/slot.json")
            if sha(path) != actual_complete["slot_hashes"][f"{tid}/{method}"]:
                raise ValueError("actual completion seal mismatch")
            slots.append(read(path))
        for row in rows:
            plan_object = TaskAnchoredResidualPlan.from_dict(row["plan"])
            exported = [t for t in teacher_rows if t["task_sha256"] == task.sha256() and t["plan_sha256"] == plan_object.sha256()]
            if len(exported) != 2 or {r["preference"] for r in exported} != {"A", "B"}:
                raise ValueError("released physical candidate does not bind its two preference exports")
            for teacher in exported:
                if teacher["z_m"] != plan_object.z_m.tolist():
                    raise ValueError("teacher/registry physical plan differs")
            bindings = []
            for slot in slots:
                if slot.get("plan_sha256") == plan_object.sha256() and slot.get("full_task_success"):
                    bindings.append(_validate_actual(portable, snapshot, tid, slot, task, plan_object, row))
            bound_source = [{"release_relative_path": f"snapshot/planning/{tid}/candidate_registry.json",
                "sha256": sha(snapshot / "planning" / tid / "candidate_registry.json")},
                {"release_relative_path": "snapshot/source_identity.json", "sha256": sha(snapshot / "source_identity.json")}]
            fact = _fact(row, task, split, bindings, C1_PRODUCER, bound_source)
            fact["historical_preference_annotations"] = [{k: r.get(k) for k in
                ("preference", "strict_A_member", "tie_A_member", "tie_A_selected", "B_available_member", "finite_non_dominated_member")}
                for r in exported]
            facts.append(fact)
            row_bindings.extend({"export_row": teacher_rows.index(r), "preference": r["preference"],
                "physical_candidate_id": fact["physical_candidate_id"], "original_candidate_id": r["candidate_id"]} for r in exported)
    unique = deduplicate_candidates(facts)
    if len(unique) != 48 or len(teacher_rows) != 96:
        raise ValueError("frozen C.1 inventory must be 48 physical candidates / 96 preference exports")
    result = {"schema": DATASET_SCHEMA, "historical_candidate_count_unique": len(unique),
        "historical_preference_export_rows": len(teacher_rows), "historical_new_physics_steps": 0,
        "historical_actual_producer": C1_PRODUCER, "candidates": unique,
        "global_original_selections": global_selections,
        "preference_export_bindings": row_bindings, "source_inventory": portable.inventory,
        "evidence_tier_counts": {tier: sum(f["evidence_tier"] == tier for f in unique) for tier in EVIDENCE_TIERS}}
    write(target, result)
    return result


def encode_condition(task, definition, preference, family, search_mask=None):
    if preference not in ("A", "B") or family not in VERSIONS:
        raise ValueError("unknown preference/reference family")
    if definition["representation_version"] != VERSIONS[family]:
        raise ValueError("conditioning family differs from analytic definition")
    e = encode_residual_condition(task, definition)
    mask = search_interval_mask(definition) if search_mask is None else np.asarray(search_mask, dtype=bool)
    if mask.shape != (6,) or not np.array_equal(mask, search_interval_mask(definition)):
        raise ValueError("search mask is not the frozen C.1 key/predecessor mask")
    additions = [float(preference == "A"), float(preference == "B"),
        float(family == "v1"), float(family == "v2"), .030 if preference == "B" else 0.,
        float(preference == "B"), *mask.astype(float), *mask.astype(float)]
    names = ["preference.A", "preference.B", "reference_family.v1", "reference_family.v2",
        "preferred_clearance_m", "preferred_clearance.present",
        *[f"search_interval.{i}.mask" for i in range(6)], *[f"open_slot.{i}" for i in range(6)]]
    literal = np.ones(len(additions), dtype=bool); literal[4] = False
    valid = np.ones(len(additions), dtype=bool); valid[4] = preference == "B"
    floors = np.full(len(additions), .01)
    return {"schema": CONDITION_SCHEMA, "values": np.r_[e["values"], additions],
        "valid": np.r_[e["valid"], valid], "literal": np.r_[e["literal"], literal],
        "std_floors": np.r_[e["std_floors"], floors], "names": e["names"] + names}


@dataclass
class ConditionNormalizer:
    mean: np.ndarray
    std: np.ndarray
    literal: np.ndarray
    floors: np.ndarray
    names: list
    fit_task_sha256: list

    @classmethod
    def fit(cls, task_condition_tuples, split_manifest):
        splits = validate_learning_splits(split_manifest)
        encodings, shas = [], []
        for task, definition, preference, family in task_condition_tuples:
            task_sha = task.sha256()
            if task_sha not in splits or splits[task_sha]["split"] != "train":
                raise ValueError("C.2 condition scaler may only fit external TRAIN Task SHAs")
            encodings.append(encode_condition(task, definition, preference, family))
            shas.append(task_sha)
        if not encodings:
            raise ValueError("no TRAIN supervision for condition scaler")
        e = encodings[0]
        if any(x["names"] != e["names"] for x in encodings):
            raise ValueError("condition schema changed")
        values = np.stack([x["values"] for x in encodings]); valid = np.stack([x["valid"] for x in encodings])
        count = valid.sum(0)
        mean = np.divide((values * valid).sum(0), count, out=np.zeros(values.shape[1]), where=count > 0)
        variance = np.divide(((values - mean)**2 * valid).sum(0), count, out=np.zeros_like(mean), where=count > 0)
        std = np.maximum(np.sqrt(variance), e["std_floors"])
        mean[e["literal"]], std[e["literal"]] = 0., 1.
        return cls(mean, std, e["literal"], e["std_floors"], e["names"], sorted(set(shas)))

    def transform_condition(self, e):
        if e["names"] != self.names or not np.array_equal(e["literal"], self.literal):
            raise ValueError("condition schema mismatch")
        return np.where(e["valid"], (e["values"] - self.mean) / self.std, 0.).astype(np.float32)

    def transform(self, task, definition, preference, family, search_mask=None):
        return self.transform_condition(encode_condition(task, definition, preference, family, search_mask))

    def to_dict(self):
        return {"schema": CONDITION_SCHEMA, "fit_split": "train", "external_TaskSHA_split": True,
            "fit_task_sha256": self.fit_task_sha256, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "literal": self.literal.tolist(), "std_floors": self.floors.tolist(), "feature_names_with_units": self.names,
            "literal_types_and_masks": True}

    @classmethod
    def from_dict(cls, value):
        if value["schema"] != CONDITION_SCHEMA or value["fit_split"] != "train":
            raise ValueError("unsupported C.2 scaler")
        n = len(value["feature_names_with_units"])
        mean, std, floors = [_array(value[k], (n,), k) for k in ("mean", "std", "std_floors")]
        literal = np.asarray(value["literal"], dtype=bool)
        if literal.shape != (n,) or np.any(std < floors) or np.any(mean[literal] != 0) or np.any(std[literal] != 1):
            raise ValueError("discrete/scaler contract violated")
        return cls(mean, std, literal, floors, value["feature_names_with_units"], value["fit_task_sha256"])


def preference_labels(facts, task_inventory):
    """Near-optimal labels within each family, never relabel global optimum."""
    eligible = [f for f in facts if f["evidence_tier"] in ("VALIDATED_EXECUTION", "PREDICTED_COMPLETE")]
    samples, buckets = [], []
    for task in task_inventory:
        for preference in ("A", "B"):
            for family in ("v1", "v2"):
                pool = [f for f in eligible if f["task_sha256"] == task["task_sha256"] and f["family"] == family]
                metric, band = ("I_support", .001) if preference == "A" else ("L_full", .005)
                pool = [f for f in pool if _finite((f.get("prediction_metrics") or {}).get(metric))]
                if preference == "B":
                    pool = [f for f in pool if _finite(f["prediction_metrics"].get("d_support")) and
                        f["prediction_metrics"]["d_support"] >= .030 and
                        f["prediction_metrics"].get("clearance_status") == "MEASURED"]
                best = min((f["prediction_metrics"][metric] for f in pool), default=None)
                near = [f for f in pool if f["prediction_metrics"][metric] <= best + band] if best is not None else []
                bucket_id = digest({"task_sha256": task["task_sha256"], "preference": preference, "family": family})
                bucket = {"bucket_id": bucket_id, "task_id": task["task_id"], "task_sha256": task["task_sha256"],
                    "mother_id": task["mother_id"], "split": task["split"], "preference": preference,
                    "reference_family": family, "near_optimal_metric": metric, "near_optimal_band": band,
                    "family_minimum": best, "qualified_pool_unique": len(pool), "label_count_unique": len(near),
                    "status": "SUPPORTED" if near else "NO_PREFERENCE_LABEL_WITHIN_TEACHER_BUDGET"}
                buckets.append(bucket)
                for fact in near:
                    samples.append({"sample_id": digest({"physical_candidate": fact["physical_candidate_id"], "preference": preference}),
                        "physical_candidate_id": fact["physical_candidate_id"], "task_id": fact["task_id"],
                        "task_sha256": fact["task_sha256"], "mother_id": fact["mother_id"], "group_id": fact["mother_id"],
                        "split": fact["split"], "preference": preference, "family": family, "reference_family": family,
                        "reference_version": fact["reference_version"], "plan_sha256": fact["plan_sha256"],
                        "z_m": fact["z_m"], "search_interval_mask": fact["search_interval_mask"],
                        "reference_interval_mask": fact["reference_interval_mask"], "evidence_tier": fact["evidence_tier"],
                        "bucket_id": bucket_id, "label_scope": "family_near_optimum_only",
                        "prediction_metrics": fact["prediction_metrics"], "source_inventory": fact["source_inventory"]})
    return samples, buckets


def _coverage(facts, samples):
    rows = []
    # Joint and marginal counts preserve unique candidates separately from views.
    for dimension in ("evidence_tier", "mother_id", "task_id", "preference", "reference_family"):
        source = samples if dimension == "preference" else facts
        for value in sorted({r.get(dimension) for r in source if r.get(dimension) is not None}):
            subset = [r for r in source if r.get(dimension) == value]
            rows.append({"dimension": dimension, "value": value, "records": len(subset),
                "unique_candidates": len({r["physical_candidate_id"] for r in subset})})
    joint = defaultdict(list)
    for s in samples:
        joint[(s["evidence_tier"], s["mother_id"], s["task_id"], s["preference"], s["reference_family"])].append(s)
    return {"marginal": rows, "joint": [{"evidence_tier": k[0], "mother_id": k[1], "task_id": k[2],
        "preference": k[3], "reference_family": k[4], "records": len(v),
        "unique_candidates": len({x["physical_candidate_id"] for x in v})} for k, v in sorted(joint.items())]}


def build_dataset(run):
    run = Path(run).resolve()
    target = run / "dataset" / "manifest.json"
    if target.exists():
        return load_preference_dataset(target)
    split_manifest = read(run / "learning_split_manifest.json")
    splits = validate_learning_splits(split_manifest)
    historical = import_c1(run)
    facts = historical["candidates"]
    global_selections = dict(historical["global_original_selections"])
    tasks = {r["task_id"]: TaskSpec.from_dict(read(run / r["task_path"])) for r in split_manifest["tasks"]}
    for frozen in read(run / "plan.json")["tasks"]:
        if frozen["split"] not in ("train", "val"):
            continue
        root = run / "teacher_predictions" / frozen["task_id"]
        registry = root / "planning" / frozen["task_id"] / "candidate_registry.json"
        if not registry.exists():
            raise FileNotFoundError("finite teacher task is not sealed: " + str(registry))
        if sha(root / "frozen_execution_config.json") != sha(run / "frozen_execution_config.json"):
            raise ValueError("teacher config differs from frozen C.2 config")
        global_selections[frozen["task_id"]] = read(registry.parent / "selection.json")
        for row in read(registry):
            task = tasks[frozen["task_id"]]
            source = [{"path": registry.relative_to(run).as_posix(), "sha256": sha(registry)}]
            facts.append(_fact(row, task, splits[task.sha256()], producer=read(run / "source_identity.json")["algorithm_producer_commit"], source_inventory=source))
    facts = deduplicate_candidates(facts)
    if len(facts) > 96:
        raise ValueError("finite physical candidate teacher budget exceeded")
    samples, buckets = preference_labels(facts, split_manifest["tasks"])
    # TEST buckets are inventory only and have no candidate quality or labels.
    if any(s["split"] == "test" for s in samples):
        raise ValueError("TEST teacher outcome leaked into dataset")
    definitions = {(tid, family): build_reference_definition(t, version=VERSIONS[family])
        for tid, t in tasks.items() for family in ("v1", "v2")}
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    scalers = {}
    if train and val:
        # One fit row per Task/pref/family avoids overweighting duplicate routes.
        unique_conditions = {(s["task_id"], s["preference"], s["family"]) for s in train}
        condition = ConditionNormalizer.fit([(tasks[tid], definitions[(tid, fam)], pref, fam)
            for tid, pref, fam in sorted(unique_conditions)], split_manifest)
        # One residual per physical TRAIN label; multi-preference views do not
        # duplicate the physical statistics fitted on open dimensions.
        physical_train = {s["physical_candidate_id"]: s for s in train}.values()
        residual = ResidualNormalizer.fit([s["z_m"] for s in physical_train],
            [np.asarray(s["search_interval_mask"], dtype=bool) for s in physical_train])
        scalers = {"condition": condition.to_dict(), "residual": residual.to_dict()}
        for sample in samples:
            enc = encode_condition(tasks[sample["task_id"]], definitions[(sample["task_id"], sample["family"])],
                sample["preference"], sample["family"])
            sample["condition_raw"] = enc["values"].tolist()
            sample["condition_normalized"] = condition.transform_condition(enc).tolist()
    supported = []
    for pref, family in (("A", "v1"), ("B", "v2")):
        train_support = [s for s in train if s["preference"] == pref and s["family"] == family]
        val_support = [s for s in val if s["preference"] == pref and s["family"] == family]
        supported.append({"preference": pref, "reference_family": family,
            "train_labels": len(train_support), "val_labels": len(val_support),
            "train_mothers": sorted({s["mother_id"] for s in train_support}),
            "status": "UNSUPPORTED_TRAINING_CONDITION" if not train_support else
                "DATA_LIMITED" if len({s["mother_id"] for s in train_support}) < 2 or not val_support else "SUPPORTED"})
    output = target.parent
    output.mkdir(parents=True, exist_ok=True)
    write(output / "candidate_facts.json", facts)
    write(output / "labels.json", samples)
    write(output / "bucket_coverage.json", buckets)
    write(output / "global_original_selections.json", global_selections)
    if scalers:
        write(output / "condition_scaler.json", scalers["condition"])
        write(output / "residual_scaler.json", scalers["residual"])
        write(output / "condition_schema.json", {"schema": CONDITION_SCHEMA, "dimension": len(condition.names),
            "feature_names_with_units": condition.names, "literal": condition.literal.tolist(),
            "ids_or_outcomes_encoded": False, "reference_and_search_masks_distinct": True})
    manifest = {"schema": DATASET_SCHEMA, "learning_split_manifest_path": "../learning_split_manifest.json",
        "learning_split_manifest_sha256": sha(run / "learning_split_manifest.json"),
        "tasks": split_manifest["tasks"], "candidate_count_unique": len(facts),
        "historical_candidate_count_unique": 48, "label_count_views": len(samples),
        "zero_label_views": sum(not np.any(s["z_m"]) for s in samples),
        "supervision_available": bool(train and val), "scalers_train_only": bool(scalers),
        "test_leakage_checks_passed": True, "supported_preference_family_conditions": supported,
        "data_status": "NO_SUPERVISION" if not train or not val else "DATA_LIMITED" if any(x["status"] != "SUPPORTED" for x in supported) else "SUPPORTED",
        "coverage": _coverage(facts, samples), "source_inventory": import_c1(run)["source_inventory"],
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "training_distribution": "uniform mother, then Task, then supported preference/family bucket, then unique near-optimal reference"}
    write(target, manifest)
    write(output / "dataset_manifest.json", manifest)
    return load_preference_dataset(target)


@dataclass
class PreferenceDataset:
    manifest_path: Path
    manifest: dict
    samples: list
    tasks: dict
    definitions: dict
    z_m: np.ndarray
    search_masks: np.ndarray
    condition_scaler: object
    residual_scaler: object
    task_splits: dict
    task_mothers: dict

    @property
    def masks(self):
        return self.search_masks

    @property
    def labels(self):
        return self.samples

    @property
    def scaler(self):
        return self.condition_scaler

    def indices(self, split):
        return np.asarray([i for i, s in enumerate(self.samples) if s["split"] == split], dtype=np.int64)


def load_preference_dataset(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = read(path)
    if manifest["schema"] != DATASET_SCHEMA:
        raise ValueError("unsupported C.2 dataset")
    for name, expected in manifest["files"].items():
        if sha(path.parent / name) != expected:
            raise ValueError("immutable dataset artifact changed: " + name)
    split_path = (path.parent / manifest["learning_split_manifest_path"]).resolve()
    if sha(split_path) != manifest["learning_split_manifest_sha256"]:
        raise ValueError("external split manifest changed")
    splits = validate_learning_splits(read(split_path))
    run = split_path.parent
    tasks = {r["task_id"]: TaskSpec.from_dict(read(run / r["task_path"])) for r in manifest["tasks"]}
    for task in tasks.values():
        if task.sha256() not in splits:
            raise ValueError("dataset Task SHA changed")
    samples = read(path.parent / "labels.json")
    definitions = {(tid, fam): build_reference_definition(t, version=VERSIONS[fam])
        for tid, t in tasks.items() for fam in ("v1", "v2")}
    for sample in samples:
        if sample["split"] != splits[sample["task_sha256"]]["split"] or sample["split"] == "test":
            raise ValueError("sample leaks external split")
        if np.any(np.asarray(sample["z_m"])[~np.asarray(sample["search_interval_mask"], dtype=bool)] != 0):
            raise ValueError("sample inactive output nonzero")
    condition = ConditionNormalizer.from_dict(read(path.parent / "condition_scaler.json")) if manifest["scalers_train_only"] else None
    residual = ResidualNormalizer.from_dict(read(path.parent / "residual_scaler.json")) if manifest["scalers_train_only"] else None
    return PreferenceDataset(path.resolve(), manifest, samples, tasks, definitions,
        np.asarray([s["z_m"] for s in samples], dtype=float).reshape(-1, 6, 2),
        np.asarray([s["search_interval_mask"] for s in samples], dtype=bool).reshape(-1, 6), condition, residual,
        {r["task_id"]: r["split"] for r in manifest["tasks"]},
        {r["task_id"]: r["mother_id"] for r in manifest["tasks"]})


load_dataset = load_preference_dataset
