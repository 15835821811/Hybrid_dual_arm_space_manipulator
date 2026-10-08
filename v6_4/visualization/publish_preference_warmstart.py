"""Curated, byte-preserving C.2 publication; no experiment/model/physics imports.

Each stage has an immutable manifest. Final publication adds evidence to the
training snapshot and refuses to replace any previously copied bytes. Large
native traces and raw ledgers remain local with full SHA/size/path inventories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
FILE_LIMIT = 90_000_000
JSON_LIMIT = 10 * 1024 * 1024
TRAIN_ROOTS = {"frozen_tasks", "historical_import", "teacher_predictions", "dataset",
               "model", "budget_ledger", "validation_preflight"}
FINAL_ROOTS = TRAIN_ROOTS | {"benchmark_search", "sealed_selections", "actual",
                            "validation", "tables", "figures", "teacher_update", "command_logs"}
ROOT_FILES = {"plan.json", "source_identity.json", "learning_split_manifest.json",
    "frozen_execution_config.json", "frozen_run_config.json", "prepare_receipt.json",
    "teacher_phase.json", "model_freeze.json"}
FINAL_FILES = ROOT_FILES | {"search_phase.json", "actual_phase.json", "actual_complete.json",
    "REPORT.md", "summary.json", "manifest.json", "teacher_records.jsonl"}
REQUIRED_FACT_DICTIONARIES = {
    "historical_import/candidate_facts.json", "historical_import/source_inventory.json",
    "dataset/candidate_facts.json", "dataset/labels.json", "dataset/source_inventory.json",
    "teacher_update/candidate_facts.json", "teacher_update/rejected_proposals.json",
    "teacher_update/source_inventory.json", "teacher_update/manifest.json",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("immutable publication metadata differs: " + str(path))
    else:
        with path.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


def copy_exact(source, target, expected=None):
    source, target = Path(source), Path(target)
    expected = expected or sha(source)
    if source.stat().st_size > FILE_LIMIT:
        raise ValueError("publication file exceeds 90MB: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError("previous publication bytes differ: " + str(target))
    else:
        shutil.copyfile(source, target)
    if sha(target) != expected:
        raise ValueError("copied bytes changed: " + str(target))


def verify_bindings(run, stage):
    """Byte/receipt verification only; this never reruns scientific validation."""
    identity = read(run / "source_identity.json")
    for relative, expected in identity["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("algorithm producer source changed: " + relative)
    for absolute, expected in identity.get("protected_artifacts", {}).items():
        if sha(absolute) != expected:
            raise ValueError("frozen task/config changed: " + absolute)
    teacher = read(run / "teacher_phase.json")
    if teacher.get("passed") is not True:
        raise ValueError("teacher phase is not terminal and validated")
    freeze = read(run / "model_freeze.json")
    for path, expected in freeze["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("model freeze artifact changed: " + path)
    training = read(run / "model" / "training_summary.json")
    if not training["training_executed"] or training["optimizer_updates_total"] != 4000:
        raise ValueError("publication requires the completed real fixed-budget model")
    if training["validation_checkpoint_count"] != 16:
        raise ValueError("the sixteen predeclared checkpoints are missing")
    checkpoints = list((run / "model" / "model").glob("checkpoint_*.pt"))
    if len(checkpoints) != 16:
        raise ValueError("all sixteen checkpoint files must be retained")
    if stage == "final":
        validation = read(run / "validation" / "validation.json")
        summary = read(run / "summary.json")
        if validation.get("all_terminal") is not True or summary.get("research_delivery_complete") is not True:
            raise ValueError("final publication requires completed original protocol validation")
        if validation.get("logical_actual_slots") != 32:
            raise ValueError("final publication must retain all32 logical actual slots")
        selections = read(run / "sealed_selections" / "all_selections.json")
        if sha(run / "model_freeze.json") != selections["model_freeze_sha256"]:
            raise ValueError("model freeze changed after endpoint selection")
        for relative, expected in selections["files"].items():
            if sha(run / relative) != expected:
                raise ValueError("sealed endpoint/search file changed: " + relative)
        actual = read(run / "actual_complete.json")
        for relative, expected in actual["slot_hashes"].items():
            if sha(run / relative) != expected:
                raise ValueError("actual logical slot changed: " + relative)
        for phase in ("search", "actual"):
            if read(run / (phase + "_phase.json")).get("passed") is not True:
                raise ValueError("required final phase did not finish: " + phase)
    return identity


def qualification(relative, size, stage):
    """Keep scientific receipts, omit duplicate/raw execution arrays and logs."""
    path = Path(relative)
    parts = set(path.parts)
    roots = FINAL_ROOTS if stage == "final" else TRAIN_ROOTS
    files = FINAL_FILES if stage == "final" else ROOT_FILES
    if path.parts[0] not in roots and relative not in files:
        return False, "outside_curated_stage_scope"
    if size > FILE_LIMIT:
        if relative in REQUIRED_FACT_DICTIONARIES:
            raise ValueError("required scientific fact dictionary exceeds90MB; create a compact source-bound fact export before publication: " + relative)
        return False, "over_90MB_local_archive_only"
    if relative in REQUIRED_FACT_DICTIONARIES:
        return True, "required_physical_fact_label_or_source_inventory_exact_bytes"
    if path.suffix.lower() in (".npz", ".npy", ".mjb", ".bin", ".h5", ".hdf5"):
        return False, "native_physical_arrays_local_archive_only"
    lower = relative.lower()
    if ("certificate" in lower or "preview" in lower or "pcc_monitor" in parts
            or ("ledger" in path.name.lower() and "budget_ledger" not in parts)):
        return False, "raw_certificate_preview_or_duplicate_ledger_local_only"
    if path.suffix.lower() == ".jsonl":
        keep = ("model" in parts and path.name == "curves.jsonl") or "teacher_records" in path.name
        return (True, "bounded_training_or_teacher_evidence") if keep and size <= JSON_LIMIT else (False, "raw_jsonl_local_archive_only")
    if path.parts[0] == "command_logs" and path.suffix.lower() in (".log", ".txt"):
        return (True, "bounded_command_environment_or_phase_receipt") if size <= JSON_LIMIT else (False, "large_command_log_local_archive_only")
    if path.suffix.lower() == ".json" and size > JSON_LIMIT:
        return False, "large_json_raw_log_local_archive_only"
    if path.suffix.lower() in (".json", ".md", ".csv", ".pt", ".pth"):
        return True, "portable_scientific_evidence"
    if path.suffix.lower() in (".png", ".svg") and path.parts[0] == "figures":
        return True, "core_scientific_figure"
    return False, "raw_diagnostic_or_unrequested_format_local_only"


def verify_seals(run, rows):
    """Inspect original manifest hashes without requiring omitted files in clone."""
    seals = 0
    for row in rows:
        if not row["copied"] or not row["path"].endswith("/manifest.json"):
            continue
        path = run / row["path"]
        value = read(path)
        if not isinstance(value, dict) or not value or not all(isinstance(v, str) and len(v) == 64 for v in value.values()):
            continue
        for relative, expected in value.items():
            target = (path.parent / relative).resolve()
            if not target.is_relative_to(run):
                raise ValueError("seal escapes frozen run")
            if sha(target) != expected:
                raise ValueError("original execution/evaluation seal differs: " + str(target))
        seals += 1
    return seals


def bind_media(manifest, release, run):
    manifest = Path(manifest).resolve()
    value = read(manifest)
    if value.get("schema") != "v64_c2_saved_actual_media_v1":
        raise ValueError("not a C.2 saved-actual visualization manifest")
    for name in ("plan", "actual_complete"):
        if value["run_identity"][name + "_sha256"] != sha(run / (name + ".json")):
            raise ValueError("media manifest belongs to a different scientific run")
    if any(value.get(k, 0) != 0 for k in ("physics_steps_executed", "geometry_queries", "QP_solves", "optimizer_updates", "new_model_samples")):
        raise ValueError("media generation consumed forbidden new scientific work")
    inventory = []
    for relative, row in sorted(value["files"].items()):
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError("media asset escapes its manifest directory")
        expected = row["sha256"] if isinstance(row, dict) else row
        if sha(path) != expected:
            raise ValueError("saved-actual media asset changed: " + relative)
        inventory.append({"source_path": str(path), "repository_path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else None,
            "sha256": expected, "size": path.stat().st_size,
            "publication_size_eligible": path.stat().st_size <= FILE_LIMIT,
            "media_role": "video" if path.suffix.lower() == ".mp4" else "figure_or_media_receipt"})
    copy_exact(manifest, release / "media_bindings" / "visualization_manifest.json")
    binding = {"schema": "v64_c2_separate_saved_actual_media_binding_v1",
        "source_manifest_path": str(manifest), "source_manifest_sha256": sha(manifest),
        "video_count": value["video_count"], "unique_actual_media": value["unique_actual_media"],
        "inventory": inventory, "assets_remain_in_visualization_directory": True,
        "scientific_acceptance_replaced_by_media": False,
        "new_physics_steps": 0, "new_model_samples": 0}
    immutable_json(release / "media_bindings.json", binding)
    return binding


def publish(run, *, stage="final", release=None, media_manifest=None):
    if stage not in ("train", "final"):
        raise ValueError("publication stage must be train or final")
    run = Path(run).resolve()
    release = Path(release).resolve() if release else ROOT / "v6_4" / "releases" / run.name
    if release.is_relative_to(run) or run.is_relative_to(release):
        raise ValueError("publication snapshot must be separate from raw local archive")
    identity = verify_bindings(run, stage)
    existing = release / ("release_manifest_" + stage + ".json")
    if existing.exists():
        retained = read(existing)
        for row in retained["inventory"]:
            if sha(row["source_path"]) != row["sha256"]:
                raise ValueError("published-stage raw source bytes changed")
            if row["copied"] and sha(release / "snapshot" / row["path"]) != row["sha256"]:
                raise ValueError("published-stage copied bytes changed")
        if media_manifest:
            if stage != "final":
                raise ValueError("media binding belongs to final publication")
            bind_media(media_manifest, release, run)
        return retained
    rows = []
    for source in sorted(p for p in run.rglob("*") if p.is_file()):
        relative = source.relative_to(run).as_posix()
        size = source.stat().st_size
        copied, reason = qualification(relative, size, stage)
        row = {"path": relative, "source_path": str(source), "sha256": sha(source), "size": size,
            "copied": copied, "availability": "portable_exact_bytes" if copied else "raw_local_archive_only",
            "reason": reason}
        rows.append(row)
    seals = verify_seals(run, rows)
    for row in rows:
        if row["copied"]:
            copy_exact(row["source_path"], release / "snapshot" / row["path"], row["sha256"])
    source_manifest = {}
    for relative, expected in sorted(identity["source_sha256"].items()):
        if Path(relative).suffix.lower() != ".py":
            continue
        # Exact frozen inputs establish the actual algorithm producer. Assets
        # are bound by source_identity rather than duplicated into the release.
        if Path(relative).parts[0] in ("v6_4", "v6_lite", "model_test"):
            copy_exact(ROOT / relative, release / "frozen_source" / relative, expected)
            source_manifest[relative] = expected
    immutable_json(release / "frozen_source_manifest.json", {
        "algorithm_producer_commit": identity["algorithm_producer_commit"],
        "base_publication_commit": identity["base_publication_commit"],
        "historical_actual_producer": identity["historical_actual_producer"],
        "source_sha256": source_manifest, "publication_commit_is_not_algorithm_producer": True})
    raw = {row["path"]: {"sha256": row["sha256"], "size": row["size"], "source_path": row["source_path"]} for row in rows}
    immutable_json(release / ("raw_manifest_" + stage + ".json"), raw)
    portable = {row["source_path"]: {"available": row["copied"],
        "path": "snapshot/" + row["path"] if row["copied"] else None,
        "sha256": row["sha256"], "size": row["size"]} for row in rows}
    for relative, expected in source_manifest.items():
        portable[str((ROOT / relative).resolve())] = {"available": True,
            "path": "frozen_source/" + relative, "sha256": expected,
            "size": (ROOT / relative).stat().st_size,
            "artifact_kind": "frozen_algorithm_producer_source"}
    immutable_json(release / ("portable_paths_" + stage + ".json"), portable)
    result = {"schema": "v64_c2_portable_release_v1", "stage": stage,
        "source_run": str(run), "algorithm_producer_commit": identity["algorithm_producer_commit"],
        "base_publication_commit": identity["base_publication_commit"],
        "historical_actual_producer": identity["historical_actual_producer"],
        "publication_commit_is_not_algorithm_producer": True,
        "publication_script_sha256": sha(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": rows, "copied_count": sum(r["copied"] for r in rows),
        "omitted_count": sum(not r["copied"] for r in rows), "original_seals_verified": seals,
        "raw_local_archive_preserved": True, "native_replay_archive_complete_in_clone": False,
        "verification_scope": "exact bytes, source identity, original report/seal bindings and explicit omissions; no physical replay",
        "new_physics_steps": 0, "new_geometry_queries": 0, "new_QP_calls": 0, "new_model_samples": 0}
    immutable_json(existing, result)
    immutable_json(release / ("verification_" + stage + ".json"), {"passed": True,
        "stage": stage, "release_manifest_sha256": sha(existing),
        "copied_count": result["copied_count"], "omitted_count": result["omitted_count"],
        "new_physics_steps": 0, "physical_replay_performed_by_export": False})
    if stage == "final":
        immutable_json(release / "release_manifest.json", result)
        immutable_json(release / "portable_paths.json", portable)
        if media_manifest:
            bind_media(media_manifest, release, run)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "final"), default="final")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--media-manifest", type=Path)
    args = parser.parse_args()
    result = publish(args.run, stage=args.stage, release=args.release, media_manifest=args.media_manifest)
    print(json.dumps({k: result[k] for k in ("stage", "copied_count", "omitted_count", "algorithm_producer_commit", "new_physics_steps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
