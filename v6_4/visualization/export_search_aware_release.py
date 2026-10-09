"""Byte-preserving C.3 portable release; no physics, training or inference.

The final snapshot keeps every model, scientific summary, raw initializer,
selection and unique actual trace/replay. Large private nominal rollouts and
raw timing/certificate ledgers stay in their original local archives, explicitly
inventoried with SHA/size and available=false in the portable path map.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
FILE_LIMIT = 90_000_000
RELEASE_SCHEMA = "v64_c3_portable_search_aware_release_v1"
REPORT_SCHEMA = "v64_c3_report_artifact_identity_v1"
REPORT_TABLES = ("endpoint_summary.csv", "per_task_endpoints.csv", "first_hits.csv",
                 "predicted_prefixes.csv", "paired_comparisons.json", "cost_accounting.json", "status.json")
REPORT_INPUTS = ("plan.json", "source_identity.json", "validation/protocol_validation.json",
                 "model_selection.json", "model_freeze.json", "actual_complete.json",
                 "closed_loop_val/sealed_selections/all_selections.json",
                 "frozen_test/sealed_selections/all_selections.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("existing release metadata differs: " + str(path))
        return
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def copy_exact(source, target, expected=None):
    source, target = Path(source), Path(target)
    expected = expected or sha(source)
    if source.stat().st_size > FILE_LIMIT:
        raise ValueError("required portable file exceeds90MB: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError("existing release bytes differ: " + str(target))
    else:
        shutil.copyfile(source, target)
    if sha(target) != expected:
        raise ValueError("copied release bytes changed")


def qualification(relative, size):
    """The portable/local boundary is fixed by artifact role, never result."""
    path = Path(relative)
    parts = set(path.parts)
    is_actual = "actual" in parts and path.parts[0] in ("closed_loop_val", "frozen_test")
    if "__pycache__" in parts or path.suffix.lower() in (".pyc", ".tmp"):
        return False, "temporary_or_python_cache"
    if "pre_execution_identity_revision" in parts:
        return (path.suffix.lower() == ".json"), "pre_execution_identity_revision_receipt"
    if "private_prediction" in parts:
        keep = path.name in ("run_metadata.json", "runner_result.json", "execution_failure.json")
        return keep, "nominal_summary" if keep else "private_nominal_raw_local_archive_only"
    if "timing" in parts or "pcc_monitor" in parts or "certificate" in path.name.lower():
        return False, "raw_timing_geometry_or_certificate_local_archive_only"
    if path.suffix.lower() in (".npz", ".npy"):
        keep = is_actual or (path.parts[0] == "models" and path.name == "paired_reference_draws.npz")
        if keep and size > FILE_LIMIT:
            raise ValueError("required actual/exposure array exceeds90MB; use bound lossless chunks before export: " + relative)
        return keep, "unique_actual_or_paired_training_array" if keep else "private_nominal_or_derived_array_local_archive_only"
    if size > FILE_LIMIT:
        if path.suffix.lower() in (".pt", ".pth", ".json"):
            raise ValueError("required scientific model/JSON exceeds90MB: " + relative)
        return False, "over90MB_local_archive_only"
    if path.suffix.lower() in (".json", ".csv", ".md", ".pt", ".pth"):
        return True, "portable_scientific_receipt"
    if path.suffix.lower() == ".jsonl":
        keep = is_actual and "evaluation" in parts or path.name == "curves.jsonl"
        return keep, "independent_actual_evaluation_or_training_curve" if keep else "raw_event_ledger_local_archive_only"
    if path.suffix.lower() in (".png", ".svg") and path.parts[0] in ("figures", "result_tables"):
        return True, "core_scientific_figure"
    if path.parts[0] == "command_logs" and path.suffix.lower() in (".log", ".txt", ".ps1"):
        return True, "execution_command_record"
    return False, "diagnostic_or_media_managed_separately"


def _verify_original_seals(run):
    """Verify original evidence bytes without replay or independent re-audit."""
    run = Path(run).resolve()
    seals = 0
    for path in sorted(run.rglob("manifest.json")):
        manifest = read(path)
        if not isinstance(manifest, dict) or not manifest or not all(isinstance(v, str) and len(v) == 64 for v in manifest.values()):
            continue
        for relative, expected in manifest.items():
            target = (path.parent / relative).resolve()
            if run not in target.parents or sha(target) != expected:
                raise ValueError("original evidence seal changed: " + str(target))
        seals += 1
    return seals


def _contained_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("report artifact path escapes its declared directory: " + str(relative))
    return path


def _verify_complete_report(run):
    """Require a complete, current report; completion labels alone are insufficient."""
    run = Path(run).resolve(); tables = run / "result_tables"
    provenance_path = tables / "report_artifact_identity.json"
    for path in (run / "REPORT.md", run / "summary.json", provenance_path):
        if not path.is_file():
            raise ValueError("final release requires current complete report and provenance: " + str(path))
    summary, provenance = read(run / "summary.json"), read(provenance_path)
    status = summary.get("status") or {}
    completion = ("research_execution_completed", "independent_test_completed", "closed_loop_val_completed",
                  "training_completed_D", "training_completed_S")
    if any(status.get(key) is not True for key in completion) or (summary.get("evidence_verification") or {}).get("complete") is not True:
        raise ValueError("final release requires report completion backed by verified current evidence")
    if provenance.get("schema") != REPORT_SCHEMA:
        raise ValueError("final report provenance schema is missing or unsupported")
    reporter = Path(__file__).with_name("report_search_aware_warmstart.py")
    if provenance.get("reporter_source_sha256") != sha(reporter):
        raise ValueError("final report was produced by a different reporting helper")
    roots, derived, inputs = (provenance.get(key) or {} for key in ("root_reports", "derived_artifacts", "input_files"))
    if set(roots) != {"REPORT.md", "summary.json"}:
        raise ValueError("report provenance must bind REPORT.md and summary.json")
    for relative, expected in roots.items():
        if sha(_contained_path(run, relative)) != expected:
            raise ValueError("root report changed after report provenance: " + relative)
    current_derived = {p.relative_to(run).as_posix() for p in tables.rglob("*") if p.is_file() and p != provenance_path}
    required_derived = {"result_tables/" + name for name in REPORT_TABLES}
    if set(derived) != current_derived or not required_derived <= set(derived):
        raise ValueError("report derived artifact inventory is incomplete or stale")
    for relative, expected in derived.items():
        path = _contained_path(run, relative)
        if tables not in path.parents or sha(path) != expected:
            raise ValueError("derived report artifact changed: " + relative)
    normalized_inputs = {}
    for name, expected in inputs.items():
        path = Path(name)
        if not path.is_absolute():
            path = _contained_path(run, path)
        path = path.resolve(); normalized_inputs[str(path)] = expected
        if not path.is_file() or sha(path) != expected:
            raise ValueError("report scientific input changed or missing: " + str(path))
    if any(str((run / relative).resolve()) not in normalized_inputs for relative in REPORT_INPUTS):
        raise ValueError("report provenance lacks required final scientific input identities")
    return sha(provenance_path)


def _run_inventory(run):
    inventory = []
    for source in sorted(Path(run).resolve().rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(run).as_posix()
        expected, size = sha(source), source.stat().st_size
        keep, reason = qualification(relative, size)
        inventory.append({"path": relative, "source_path": str(source), "sha256": expected, "size": size,
            "copied": keep, "availability": "portable_bytes" if keep else "local_archive_only", "reason": reason})
    return inventory


def _inventory_sha(inventory):
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def _verify_current_inventory(run, saved):
    current = _run_inventory(run)
    if current != saved:
        raise ValueError("existing release source-run inventory is stale; use a fresh destination")
    return _inventory_sha(current)


def verify_inputs(run, require_complete=True):
    run = Path(run).resolve()
    identity = read(run / "source_identity.json")
    for relative, expected in identity["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("algorithm producer source changed: " + relative)
    for original, expected in identity.get("protected_artifacts", {}).items():
        if sha(original) != expected:
            raise ValueError("frozen Task/config changed: " + original)
    if require_complete:
        validation = read(run / "validation/protocol_validation.json")
        if validation.get("all_terminal") is not True or validation.get("actual_logical_slots") != {"val": 20, "test": 40}:
            raise ValueError("final release requires terminal C.3 20 VAL +40 TEST actual logical slots")
        for model in ("D", "S"):
            report = read(run / "models" / model / "training_report.json")
            if report.get("training_executed") is not True or report.get("optimizer_updates_total") != 4000:
                raise ValueError("real fixed-budget model training incomplete: " + model)
            for update in (250, 4000):
                if sha(run / "models" / model / f"checkpoint_{update:04d}.pt") != report["checkpoint_sha256"][str(update)]:
                    raise ValueError("model checkpoint changed")
        for original, expected in read(run / "model_freeze.json")["artifacts"].items():
            if sha(original) != expected:
                raise ValueError("pre-TEST model/data freeze changed")
        for phase in ("closed_loop_val", "frozen_test"):
            for original, expected in read(run / phase / "sealed_selections/all_selections.json")["files"].items():
                if sha(original) != expected:
                    raise ValueError("sealed phase selections changed")
        _verify_complete_report(run)
    return identity


def export_release(run, release, *, require_complete=True):
    run, release = Path(run).resolve(), Path(release).resolve()
    if run == release or run in release.parents or release in run.parents:
        raise ValueError("release must be outside the original experiment run")
    target = release / "release_manifest.json"
    retained = read(target) if target.exists() else None
    if retained is not None and require_complete and retained.get("stage") != "final":
        raise ValueError("partial release cannot be reused as final; use a fresh destination")
    # A retained final must remain final even when a later caller passes
    # --allow-partial; that flag must not weaken its current-evidence checks.
    effective_complete = require_complete or bool(retained and retained.get("stage") == "final")
    identity = verify_inputs(run, require_complete=effective_complete)
    if target.exists():
        value = retained
        if value["source_run"] != str(run) or value["source_identity_sha256"] != sha(run / "source_identity.json"):
            raise ValueError("existing release belongs to a different producer/run")
        inventory_sha = _verify_current_inventory(run, value["inventory"])
        if value.get("source_run_inventory_sha256", inventory_sha) != inventory_sha:
            raise ValueError("existing release source-run inventory digest differs")
        if effective_complete and value.get("report_artifact_identity_sha256") != sha(run / "result_tables/report_artifact_identity.json"):
            raise ValueError("existing final release report provenance is stale; use a fresh destination")
        PortableResolver(release).verify_all()
        return value
    release.mkdir(parents=True, exist_ok=True)
    inventory, mapping = _run_inventory(run), {}
    for row in inventory:
        relative, expected, size, keep, reason = (row[k] for k in ("path", "sha256", "size", "copied", "reason"))
        source = Path(row["source_path"])
        mapped = {"available": keep, "path": "snapshot/" + relative if keep else None,
            "sha256": expected, "size": size, "archive_path": str(source), "reason": reason}
        mapping[str(source)] = mapped
        if keep:
            copy_exact(source, release / "snapshot" / relative, expected)
    frozen_sources = {}
    for relative, expected in identity["source_sha256"].items():
        source = ROOT / relative
        copy_exact(source, release / "frozen_source" / relative, expected)
        frozen_sources[relative] = expected
        mapping[str(source.resolve())] = {"available": True, "path": "frozen_source/" + relative,
            "sha256": expected, "size": source.stat().st_size}
    immutable_json(release / "frozen_source_manifest.json", {
        "algorithm_producer_commit": identity["algorithm_producer_commit"], "source_sha256": frozen_sources,
        "publication_commit_is_not_experiment_producer": True})
    # This inherited inventory was checked at import. Do not assert that its
    # raw original absolute paths are portable or still available now.
    inherited = run / "historical_import/source_inventory.json"
    if inherited.exists():
        rows = read(inherited)
        immutable_json(release / "historical_evidence_inventory.json", rows)
        for row in rows:
            original = row.get("resolved_path") or row.get("original_path")
            if original and original not in mapping:
                mapping[original] = {"available": False, "path": None, "sha256": row["sha256"],
                    "archive_path": original, "reason": "inherited_C1_C2_evidence_verified_at_import_not_duplicated_in_C3_snapshot"}
    immutable_json(release / "portable_paths.json", mapping)
    original_seals = _verify_original_seals(run)
    inventory_sha = _verify_current_inventory(run, inventory)
    report_identity = _verify_complete_report(run) if require_complete else None
    receipt = {"schema": RELEASE_SCHEMA, "stage": "final" if require_complete else "partial",
        "source_run": str(run), "source_identity_sha256": sha(run / "source_identity.json"),
        "algorithm_producer_commit": identity["algorithm_producer_commit"],
        "base_publication_commit": identity["base_publication_commit"],
        "publication_commit_is_not_experiment_producer": True,
        "exporter_sha256": sha(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": inventory, "copied_count": sum(r["copied"] for r in inventory),
        "source_run_inventory_sha256": inventory_sha, "report_artifact_identity_sha256": report_identity,
        "omitted_count": sum(not r["copied"] for r in inventory),
        "original_evidence_seals_verified": original_seals,
        "unique_actual_trace_replay_arrays_included": True,
        "standalone_media_regeneration_scope": "saved-state rendering inputs retained; original certificate/timing archives may still be required by media validation; render original run before publishing media",
        "private_nominal_raw_local_archive_only": True,
        "new_physics_steps": 0, "new_geometry_queries": 0, "new_training_updates": 0,
        "new_DDIM_samples": 0, "deployment": "NOT_MET"}
    immutable_json(target, receipt)
    PortableResolver(release).verify_all()
    return receipt


class PortableResolver:
    """Resolve only copied release bytes; never follow an original E: path."""
    def __init__(self, release):
        self.release = Path(release).resolve()
        self.manifest = read(self.release / "release_manifest.json")
        if self.manifest.get("schema") != RELEASE_SCHEMA:
            raise ValueError("not a C.3 portable release")
        self.mapping = read(self.release / "portable_paths.json")

    def resolve(self, original, expected=None):
        row = self.mapping.get(str(original))
        if row is None or row.get("available") is not True or not row.get("path"):
            raise FileNotFoundError("artifact is not portable; consult explicit local archive inventory: " + str(original))
        if expected is not None and row["sha256"] != expected:
            raise ValueError("original frozen SHA disagrees with portable mapping")
        path = (self.release / row["path"]).resolve()
        if self.release not in path.parents or sha(path) != row["sha256"]:
            raise ValueError("portable artifact bytes changed")
        return path

    def snapshot(self, relative):
        matches = [r for r in self.manifest["inventory"] if r["path"] == relative]
        if len(matches) != 1:
            raise ValueError("snapshot artifact not uniquely inventoried")
        return self.resolve(matches[0]["source_path"], matches[0]["sha256"])

    def verify_all(self):
        inventory = self.manifest["inventory"]
        if len({r["path"] for r in inventory}) != len(inventory) or len({r["source_path"] for r in inventory}) != len(inventory):
            raise ValueError("portable inventory contains duplicate artifact identities")
        if self.manifest.get("source_run_inventory_sha256") is not None and self.manifest["source_run_inventory_sha256"] != _inventory_sha(inventory):
            raise ValueError("portable source-run inventory digest changed")
        for row in self.manifest["inventory"]:
            if row.get("copied"):
                mapped = self.mapping.get(row["source_path"]) or {}
                if mapped.get("path") != "snapshot/" + row["path"] or mapped.get("size") != row["size"]:
                    raise ValueError("copied artifact path/size mapping differs from immutable inventory")
                path = self.resolve(row["source_path"], row["sha256"])
                if path.stat().st_size != row["size"]:
                    raise ValueError("portable artifact size differs from immutable inventory")
        for original, row in self.mapping.items():
            if row.get("available") is True:
                self.resolve(original, row["sha256"])
        source_manifest = self.release / "frozen_source_manifest.json"
        if not source_manifest.is_file():
            raise ValueError("portable frozen producer source manifest missing")
        frozen = read(source_manifest)
        source_identity = read(self.snapshot("source_identity.json"))
        if frozen.get("source_sha256") != source_identity.get("source_sha256"):
            raise ValueError("frozen producer source manifest differs from snapshotted execution identity")
        for relative, expected in frozen["source_sha256"].items():
            matching = [(original, row) for original, row in self.mapping.items() if row.get("path") == "frozen_source/" + relative]
            if len(matching) != 1 or matching[0][1].get("available") is not True or matching[0][1]["sha256"] != expected:
                raise ValueError("frozen producer source mapping changed")
            path = self.resolve(matching[0][0], expected)
            if path.stat().st_size != matching[0][1].get("size"):
                raise ValueError("frozen producer source byte size changed")
        return {"portable_bytes_verified": True, "new_physics_steps": 0, "new_DDIM_samples": 0}

    def verify_sources(self):
        """Require the imported clone to contain the frozen numerical sources."""
        manifest = read(self.release / "frozen_source_manifest.json")
        for relative, expected in manifest["source_sha256"].items():
            source = self.release / "frozen_source" / relative
            if sha(source) != expected or sha(ROOT / relative) != expected:
                raise ValueError("loaded clone differs from frozen producer input: " + relative)
        return manifest

    def load_dataset(self):
        self.verify_sources()
        from ..search_effect_teacher import load_search_aware_dataset
        return load_search_aware_dataset(self.snapshot("dataset/manifest.json"))

    def load_selected_sampler(self, model, device="cpu"):
        """Load exact pre-TEST chosen weights with the original sampler math.

        Only path/byte verification is relocated. The returned object is the
        original SearchAwareSampler; this loader performs zero DDIM samples,
        regression forwards, optimization updates, or physics steps.
        """
        started = time.perf_counter()
        if model not in ("D", "S"):
            raise ValueError("model must be D or S")
        self.verify_sources()
        freeze = read(self.snapshot("model_freeze.json"))
        for original, expected in freeze["artifacts"].items():
            self.resolve(original, expected)
        checkpoint_path = self.resolve(freeze["selected_checkpoints"][model])
        import torch
        from ..dataset import object_sha
        from ..preference_teacher_dataset import ConditionNormalizer
        from ..residual_dataset import ResidualNormalizer
        from ..residual_diffusion import ResidualDDPM, ResidualDiffusionConfig, _configure, _state_sha
        from ..simple_warmstart_regression import CHECKPOINT_SCHEMA, SearchAwareSampler, SimpleWarmstartRegression
        _configure()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("model_name") != model or checkpoint.get("update") not in (250, 4000):
            raise ValueError("not the selected fixed C.3 model/checkpoint")
        for original, expected in checkpoint["training_config"]["source_files"].items():
            self.resolve(original, expected)
        if sha(self.snapshot("models/training_config.json")) != checkpoint["training_config_sha256"]:
            raise ValueError("checkpoint training protocol changed")
        if sha(self.resolve(checkpoint["training_config"]["dataset_manifest_path"])) != checkpoint["training_config"]["dataset_manifest_sha256"]:
            raise ValueError("checkpoint's shared TRAIN dataset changed")
        sampler = SearchAwareSampler.__new__(SearchAwareSampler)
        sampler.checkpoint_path = checkpoint_path
        sampler.checkpoint_sha256 = sha(checkpoint_path)
        sampler.device, sampler.model_name = torch.device(device), model
        sampler.model = (ResidualDDPM(ResidualDiffusionConfig.from_dict(checkpoint["model_config"])) if model == "D" else
            SimpleWarmstartRegression(checkpoint["model_config"]["condition_dim"])).to(sampler.device)
        sampler.model.load_state_dict(checkpoint["state_dict"], strict=True)
        sampler.model.eval()
        if _state_sha(sampler.model) != checkpoint["state_sha256"]:
            raise ValueError("portable checkpoint tensor identity changed")
        for parameter in sampler.model.parameters():
            parameter.requires_grad_(False)
        sampler.checkpoint = checkpoint
        sampler.condition_normalizer = ConditionNormalizer.from_dict(checkpoint["condition_normalizer"])
        sampler.residual_normalizer = ResidualNormalizer.from_dict(checkpoint["residual_normalizer"])
        sampler.identity = {"checkpoint_sha256": sampler.checkpoint_sha256,
            "checkpoint_update": checkpoint["update"], "model": model,
            "training_config_sha256": checkpoint["training_config_sha256"],
            "condition_scaler_sha256": object_sha(checkpoint["condition_normalizer"]),
            "residual_scaler_sha256": object_sha(checkpoint["residual_normalizer"]),
            "condition_schema_sha256": object_sha(checkpoint["condition_schema"]),
            "effective_parameter_count": checkpoint["effective_parameter_count"]}
        sampler.model_load_wall_s = time.perf_counter() - started
        sampler.sample_units = 0
        return sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("--run")
    parser.add_argument("--release", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.action == "export":
        if not args.run:
            parser.error("export requires --run")
        value = export_release(args.run, args.release, require_complete=not args.allow_partial)
        print(json.dumps({k: v for k, v in value.items() if k != "inventory"}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(PortableResolver(args.release).verify_all(), indent=2))


if __name__ == "__main__":
    main()
