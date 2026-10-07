"""Export sealed B.2 evidence without altering its historical bytes or claims.

This is an artifact copier and checksum verifier. It never imports the simulator,
trainer, sampler, or experimental execution entry point.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any


SCHEMA = "v64_b2_portable_lightweight_release_v2"
LEGACY_SCHEMA = "v64_b2_portable_lightweight_release_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = ("manifest.json", "verification.json")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": _sha(path), "bytes": path.stat().st_size}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    actual = _identity(path)
    if actual != {key: expected[key] for key in ("sha256", "bytes")}:
        raise ValueError(f"Checksum or byte-count mismatch: {path}")


def _relative_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(p in ("..", ".") for p in parts):
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes directory: {relative}")
    return path


def _include(relative: str) -> bool:
    """The fixed 718-file light-evidence selection, including both seal controls."""
    parts = PurePosixPath(relative).parts
    if parts[0] != "attempts":
        return True
    name = parts[-1]
    return (
        len(parts) == 3
        or name.endswith(".png")
        or (parts[-2] == "evaluation" and name in {
            "manifest.json", "report.json", "native_geometry.json", "trace_view_identity.json"
        })
        or (len(parts) == 4 and name in {
            "run_metadata.json", "execution_failure.json", "run_failure.json"
        })
        or (parts[-2] == "failures" and name.endswith(".json"))
    )


def _normal(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _under(path: str, root: str) -> str | None:
    candidate, prefix = _normal(path), _normal(root)
    if candidate.casefold().startswith(prefix.casefold() + "/"):
        return candidate[len(prefix) + 1:]
    return None


def _historical_join(root: str, relative: str) -> str:
    """Keep recorded Windows spelling stable when verifying from Linux/macOS."""
    separator = "\\" if "\\" in root else "/"
    return root.rstrip("/\\") + separator + relative.replace("/", separator)


def _mapping_record(
    original_path: str,
    identity: dict[str, Any],
    source: str,
    repository: str,
    included: dict[str, Any],
    sealed_source: dict[str, str],
    frozen_external: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {"original_path": original_path, **identity}
    relative = _under(original_path, source)
    if relative is not None:
        record.update({
            "kind": "INCLUDED_LIGHTWEIGHT" if relative in included else "OMITTED_HEAVY",
            "source_relative_path": relative,
            "release_relative_path": f"snapshot/{relative}" if relative in included else None,
            "repository_relative_path": None,
        })
        return record
    relative = _under(original_path, repository)
    if relative is not None:
        if frozen_external is not None and relative in frozen_external:
            record.update({
                "kind": "FROZEN_REPOSITORY_SOURCE" if relative in sealed_source else "FROZEN_REPOSITORY_SEALED_EXTERNAL",
                "source_relative_path": None,
                "release_relative_path": frozen_external[relative]["release_relative_path"],
                "repository_relative_path": None,
                "corresponding_repository_relative_path": relative,
            })
            return record
        record.update({
            "kind": "REPOSITORY_SOURCE" if relative in sealed_source else "REPOSITORY_SEALED_EXTERNAL",
            "source_relative_path": None,
            "release_relative_path": None,
            "repository_relative_path": relative,
        })
        return record
    record.update({
        "kind": "OMITTED_EXTERNAL",
        "source_relative_path": None,
        "release_relative_path": None,
        "repository_relative_path": None,
    })
    return record


def _verify_mapped_record(record: dict[str, Any], output: Path, repository: Path) -> bool:
    if record["release_relative_path"] is not None:
        _check(_relative_file(output, record["release_relative_path"]), record)
        return True
    if record["repository_relative_path"] is not None:
        _check(_relative_file(repository, record["repository_relative_path"]), record)
        return True
    if record["kind"] not in {"OMITTED_HEAVY", "OMITTED_EXTERNAL"}:
        raise ValueError("Unrecognized unavailable mapping state")
    return False


def _seal_inventory(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read(source / "manifest.json")
    verification = _read(source / "verification.json")
    if verification["status"] != "PASS" or verification["mismatches"]:
        raise ValueError("The historical seal does not report a clean PASS")
    if verification["manifest_sha256"] != _sha(source / "manifest.json"):
        raise ValueError("Historical verification is not bound to this sealed manifest")
    inventory = dict(manifest["payload"])
    for name in CONTROL_NAMES:
        inventory[name] = _identity(source / name)
    actual_paths = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual_paths != set(inventory):
        raise ValueError("The local sealed directory inventory changed")
    for relative, expected in inventory.items():
        _check(_relative_file(source, relative), expected)
    for path, expected in manifest["external"].items():
        _check(Path(path), expected)
    return manifest, verification, inventory


def _make_report(source: Path, included_count: int, omitted_count: int, omitted_bytes: int,
                 frozen_count: int, frozen_bytes: int) -> str:
    report = (source / "REPORT.md").read_text(encoding="utf-8")
    report = report.replace(_normal(source) + "/figures/", "snapshot/figures/")
    report = report.replace(str(source) + "\\figures\\", "snapshot/figures/")
    banner = f"""# GitHub evidence view — V6.4-B.2

This portable release contains {included_count} original lightweight files with unchanged
bytes. The original sealed report is [snapshot/REPORT.md](snapshot/REPORT.md); the text
below changes only its two image destinations to portable relative paths.

**The Git clone does not contain complete replay evidence.** {omitted_count} large files
({omitted_bytes:,} bytes), including physics/replay traces and per-cycle records, remain
in the sealed local archive. Their exact hashes and sizes are retained in
[release_manifest.json](release_manifest.json). Historical JSON absolute paths remain
unchanged; [portable_paths.json](portable_paths.json) resolves available artifacts and
explicitly identifies unavailable heavy or external evidence.

An additional {frozen_count} frozen repository source/document inputs ({frozen_bytes:,}
bytes) are preserved under `frozen_repository_inputs/`. This keeps the original
working-file hashes verifiable across Git checkout newline settings. The current
repository files remain the executable source; [repository_blob_review.json](repository_blob_review.json)
records each original working-file SHA and Git blob SHA separately. Where they differ,
the only byte difference is CRLF versus LF; these distinct byte hashes are not equated.

Research outcome: E0 4/4, E1 4/4, E2 3/4; learning advantage not established;
deployment NOT_MET. Refreshed visualizations do not add trials, training, or samples.

Verify this lightweight release from a repository checkout:

```sh
python -m v6_4.export_residual_release --verify --output v6_4/releases/task_anchored_residual_20261007_01
```

The verifier checks all included bytes, the complete omitted-file ledger, and available
repository mappings. It does not claim to verify omitted physical traces from a clone.

---

"""
    return banner + report


def _repository_blob_review(repository: Path, frozen_external: dict[str, Any]) -> dict[str, Any]:
    """Read Git blobs without touching the index, worktree, or experimental source."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                          stdout=subprocess.PIPE).stdout.decode("ascii").strip()
    relatives = list(frozen_external)
    request = "".join(f"{head}:{relative}\n" for relative in relatives).encode("utf-8")
    response = subprocess.run(["git", "cat-file", "--batch"], cwd=repository,
                              input=request, stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    records: dict[str, Any] = {}
    for relative in relatives:
        end = response.index(b"\n", cursor)
        header = response[cursor:end].decode("ascii")
        cursor = end + 1
        if header.endswith(" missing"):
            raise ValueError(f"Missing historical Git blob for sealed input: {relative}")
        size = int(header.rsplit(" ", 1)[1])
        blob = response[cursor:cursor + size]
        cursor += size + 1
        working = _relative_file(repository, relative).read_bytes()
        identical = blob == working
        only_eol = not identical and blob.replace(b"\r\n", b"\n") == working.replace(b"\r\n", b"\n")
        if not identical and not only_eol:
            raise ValueError(f"Sealed input differs from Git beyond newline spelling: {relative}")
        records[relative] = {
            "frozen_working_sha256": hashlib.sha256(working).hexdigest(),
            "frozen_working_bytes": len(working),
            "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "git_blob_bytes": len(blob),
            "raw_bytes_identical": identical,
            "difference": "IDENTICAL" if identical else "CRLF_LF_ONLY",
            "CRLF_to_LF_bytes_equal": blob.replace(b"\r\n", b"\n") == working.replace(b"\r\n", b"\n"),
        }
    return {
        "schema": "v64_b2_sealed_working_bytes_git_blob_review_v1",
        "reviewed_git_head": head,
        "reviewed_repository_files": len(records),
        "identical_raw_bytes": sum(row["raw_bytes_identical"] for row in records.values()),
        "CRLF_LF_only_differences": sum(row["difference"] == "CRLF_LF_ONLY" for row in records.values()),
        "other_byte_differences": 0,
        "raw_sha256_values_are_never_equated_by_normalization": True,
        "source_or_index_modified": False,
        "records": records,
    }


def export(source: Path, output: Path, refresh_frozen_inputs: bool = False) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists() and not refresh_frozen_inputs:
        raise ValueError("Release output already exists; use --verify, not an overwrite")
    if refresh_frozen_inputs:
        if not output.exists() or _read(output / "release_manifest.json")["schema"] != LEGACY_SCHEMA:
            raise ValueError("Refresh only upgrades the existing unpublished v1 light release")
        verify(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Release and sealed-source directories must be separate")
    sealed, historical_verification, inventory = _seal_inventory(source)
    source_identity = _read(source / "source_identity.json")
    # The source root is recorded by the seal rather than inferred from today's cwd.
    frozen_repository = str(source.parents[2])
    review_candidates = [
        Path(path) for path in sealed["external"]
        if _normal(path).endswith("/2026-10-07_run01/inputs.json")
    ]
    if len(review_candidates) != 1:
        raise ValueError("Expected one independently sealed result-to-claim input inventory")
    review_inputs = _read(review_candidates[0])
    for row in review_inputs["paths"]:
        _check(Path(row["path"]), row)
    included = {
        relative: {**expected, "release_relative_path": f"snapshot/{relative}"}
        for relative, expected in inventory.items() if _include(relative)
    }
    omitted = {
        relative: {**expected, "reason": "heavy replay, per-cycle, or per-sample evidence retained in sealed local archive"}
        for relative, expected in inventory.items() if relative not in included
    }
    if (len(included), len(omitted)) != (718, 294):
        raise ValueError("This exporter only accepts the fixed, sealed B.2 inventory")
    output.mkdir(parents=True, exist_ok=refresh_frozen_inputs)
    for relative, row in included.items():
        destination = _relative_file(output, row["release_relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _check(destination, row)
        else:
            shutil.copyfile(_relative_file(source, relative), destination)
        _check(destination, row)
    frozen_external: dict[str, Any] = {}
    for original_path, row in sealed["external"].items():
        relative = _under(original_path, frozen_repository)
        if relative is None:
            continue
        frozen_external[relative] = {**row, "release_relative_path": f"frozen_repository_inputs/{relative}"}
        destination = _relative_file(output, frozen_external[relative]["release_relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(original_path), destination)
        _check(destination, row)
    artifact_mappings = {
        relative: _mapping_record(_historical_join(str(source), relative), row,
                                 str(source), frozen_repository, included, source_identity["source_sha256"])
        for relative, row in inventory.items()
    }
    external_mappings = [
        _mapping_record(path, row, str(source), frozen_repository, included, source_identity["source_sha256"], frozen_external)
        for path, row in sealed["external"].items()
    ]
    review_mappings = [
        _mapping_record(row["path"], {"sha256": row["sha256"], "bytes": row["bytes"]},
                        str(source), frozen_repository, included, source_identity["source_sha256"], frozen_external)
        for row in review_inputs["paths"]
    ]
    mappings = {
        "schema": "v64_b2_portable_artifact_path_map_v1",
        "historical_paths_are_preserved": True,
        "repository_paths_resolve_from_current_checkout_root": False,
        "repository_inputs_resolve_from_exact_frozen_copies": True,
        "source_archive_local_path": str(source),
        "frozen_repository_local_path": frozen_repository,
        "artifacts": artifact_mappings,
        "external": external_mappings,
        "review_inputs": review_mappings,
        "unavailable_states": ["OMITTED_HEAVY", "OMITTED_EXTERNAL"],
    }
    _write(output / "portable_paths.json", mappings)
    (output / "report.md").write_text(
        _make_report(source, len(included), len(omitted), sum(row["bytes"] for row in omitted.values()),
                     len(frozen_external), sum(row["bytes"] for row in frozen_external.values())),
        encoding="utf-8",
    )
    summary = _read(source / "summary.json")
    analysis = _read(source / "delivery_analysis.json")
    budget = _read(source / "budget_ledger.json")
    blob_review = _repository_blob_review(Path(frozen_repository), frozen_external)
    _write(output / "repository_blob_review.json", blob_review)
    release_summary = {
        "schema": "v64_b2_release_summary_v1",
        "report": "report.md",
        "sealed_original_report": "snapshot/REPORT.md",
        "source_producer_commit": sealed["source_producer_commit"],
        "historical_delivery_git_head": sealed["delivery_git_head"],
        "sealed_manifest_sha256": _sha(source / "manifest.json"),
        "sealed_verification_sha256": _sha(source / "verification.json"),
        "included_files": len(included),
        "included_bytes": sum(row["bytes"] for row in included.values()),
        "omitted_files": len(omitted),
        "omitted_bytes": sum(row["bytes"] for row in omitted.values()),
        "frozen_repository_input_files": len(frozen_external),
        "frozen_repository_input_bytes": sum(row["bytes"] for row in frozen_external.values()),
        "repository_Git_blob_CRLF_LF_only_differences": blob_review["CRLF_LF_only_differences"],
        "complete_replay_evidence_in_git": False,
        "full_evidence_retained_at_local_sealed_source": str(source),
        "TEST_full_task_success": historical_verification["TEST_full_task_success"],
        "teacher_full_task_success": 23,
        "teacher_attempts": 24,
        "TRAIN_label_references": 17,
        "TRAIN_label_tasks": 6,
        "VAL_label_references": 6,
        "VAL_label_tasks": 2,
        "raw_candidates": 16,
        "valid_reference_candidates": 14,
        "K4_actual": "NOT_RUN",
        "training_updates": analysis["training_updates"],
        "training_sample_exposures": analysis["training_sample_exposures"],
        "selected_update": analysis["selected_update"],
        "selected_checkpoint_training_exposures": analysis["selected_checkpoint_training_exposures"],
        "actual_attempts": budget["terminal_slots"],
        "actual_physics_steps": budget["actual_physics_steps"],
        "verdict": summary["verdict"],
        "scope_notes": analysis["scope_notes"],
        "new_physics_steps": 0,
        "new_optimizer_updates": 0,
        "new_samples": 0,
    }
    _write(output / "release_summary.json", release_summary)
    derived = {name: _identity(output / name) for name in (
        "portable_paths.json", "report.md", "release_summary.json", "repository_blob_review.json"
    )}
    release_manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_repository_path": "v6_4/export_residual_release.py",
        "exporter_sha256": _sha(Path(__file__)),
        "source_producer_commit": sealed["source_producer_commit"],
        "source_sealed_manifest": _identity(source / "manifest.json"),
        "source_sealed_verification": _identity(source / "verification.json"),
        "source_payload_files": len(sealed["payload"]),
        "source_external_files": len(sealed["external"]),
        "source_review_input_paths": len(review_inputs["paths"]),
        "source_local_full_seal_verified_before_copy": True,
        "source_local_review_inputs_verified_before_copy": True,
        "included": included,
        "omitted": omitted,
        "frozen_repository_inputs": frozen_external,
        "derived": derived,
        "complete_replay_evidence_in_git": False,
        "historical_result_fields_modified": False,
        "new_physics_steps": 0,
        "new_optimizer_updates": 0,
        "new_samples": 0,
        "excluded_self_referential_control_files": ["release_manifest.json", "release_verification.json"],
    }
    _write(output / "release_manifest.json", release_manifest)
    return verify(output, refresh_verification=refresh_frozen_inputs)


def verify(output: Path, refresh_verification: bool = False) -> dict[str, Any]:
    """Verify only available portable bytes, explicitly account for unavailable data."""
    output = output.resolve()
    manifest = _read(output / "release_manifest.json")
    if manifest["schema"] not in {SCHEMA, LEGACY_SCHEMA}:
        raise ValueError("Unsupported release manifest schema")
    for relative, row in manifest["included"].items():
        _check(_relative_file(output, row["release_relative_path"]), row)
    for name, row in manifest["derived"].items():
        _check(_relative_file(output, name), row)
    frozen_external = manifest.get("frozen_repository_inputs")
    if frozen_external is not None:
        for row in frozen_external.values():
            _check(_relative_file(output, row["release_relative_path"]), row)
    historical = _read(output / "snapshot/manifest.json")
    historical_verification = _read(output / "snapshot/verification.json")
    _check(output / "snapshot/manifest.json", manifest["source_sealed_manifest"])
    _check(output / "snapshot/verification.json", manifest["source_sealed_verification"])
    if historical_verification["manifest_sha256"] != manifest["source_sealed_manifest"]["sha256"]:
        raise ValueError("Historical verification binding changed")
    inventory = dict(historical["payload"])
    inventory.update({name: manifest["included"][name] for name in CONTROL_NAMES})
    if set(manifest["included"]) & set(manifest["omitted"]):
        raise ValueError("Included and omitted inventory overlap")
    if set(inventory) != set(manifest["included"]) | set(manifest["omitted"]):
        raise ValueError("The included/omitted inventory does not cover the historical archive")
    for relative, row in inventory.items():
        published = manifest["included"].get(relative, manifest["omitted"].get(relative))
        if {key: row[key] for key in ("sha256", "bytes")} != {key: published[key] for key in ("sha256", "bytes")}:
            raise ValueError(f"Historical inventory identity changed: {relative}")
        if _include(relative) != (relative in manifest["included"]):
            raise ValueError(f"Release selection changed: {relative}")
    actual_snapshot = {p.relative_to(output / "snapshot").as_posix() for p in (output / "snapshot").rglob("*") if p.is_file()}
    if actual_snapshot != set(manifest["included"]):
        raise ValueError("Snapshot contains unaccounted or missing files")
    mappings = _read(output / "portable_paths.json")
    if set(mappings["artifacts"]) != set(inventory):
        raise ValueError("Artifact path mapping is incomplete")
    source_identity = _read(output / "snapshot/source_identity.json")
    external_expected = list(historical["external"].items())
    if frozen_external is not None:
        expected_frozen = {
            relative: {**row, "release_relative_path": f"frozen_repository_inputs/{relative}"}
            for path, row in external_expected
            if (relative := _under(path, mappings["frozen_repository_local_path"])) is not None
        }
        if frozen_external != expected_frozen:
            raise ValueError("Frozen repository copies do not exactly cover the sealed repository inputs")
        review = _read(output / "repository_blob_review.json")
        if set(review["records"]) != set(frozen_external):
            raise ValueError("Git blob review does not cover every frozen repository input")
        for relative, row in review["records"].items():
            expected = frozen_external[relative]
            if (row["frozen_working_sha256"], row["frozen_working_bytes"]) != (expected["sha256"], expected["bytes"]):
                raise ValueError("Git blob review is not bound to the frozen working-file identity")
            if row["difference"] not in {"IDENTICAL", "CRLF_LF_ONLY"} or not row["CRLF_to_LF_bytes_equal"]:
                raise ValueError("Git blob review reports a non-newline change")
    review_input_map = next(row for row in mappings["external"] if _normal(row["original_path"]).endswith("/2026-10-07_run01/inputs.json"))
    review_inputs_path = (
        _relative_file(output, review_input_map["release_relative_path"])
        if review_input_map["release_relative_path"] is not None
        else _relative_file(REPOSITORY_ROOT, review_input_map["repository_relative_path"])
    )
    review_inputs = _read(review_inputs_path)
    expected_mapping_groups = {
        "artifacts": [
            _mapping_record(_historical_join(mappings["source_archive_local_path"], relative), row,
                            mappings["source_archive_local_path"], mappings["frozen_repository_local_path"],
                            manifest["included"], source_identity["source_sha256"])
            for relative, row in inventory.items()
        ],
        "external": [
            _mapping_record(path, row, mappings["source_archive_local_path"], mappings["frozen_repository_local_path"],
                            manifest["included"], source_identity["source_sha256"], frozen_external)
            for path, row in external_expected
        ],
        "review_inputs": [
            _mapping_record(row["path"], {"sha256": row["sha256"], "bytes": row["bytes"]},
                            mappings["source_archive_local_path"], mappings["frozen_repository_local_path"],
                            manifest["included"], source_identity["source_sha256"], frozen_external)
            for row in review_inputs["paths"]
        ],
    }
    available_counts: dict[str, int] = {}
    unavailable_counts: dict[str, int] = {}
    for group, expected in expected_mapping_groups.items():
        actual = list(mappings[group].values()) if group == "artifacts" else mappings[group]
        if actual != expected:
            raise ValueError(f"Historical mapping identity or destination changed: {group}")
        available_counts[group] = sum(_verify_mapped_record(row, output, REPOSITORY_ROOT) for row in actual)
        unavailable_counts[group] = len(actual) - available_counts[group]
    expected_output = {row["release_relative_path"] for row in manifest["included"].values()}
    expected_output.update(manifest["derived"])
    if frozen_external is not None:
        expected_output.update(row["release_relative_path"] for row in frozen_external.values())
    expected_output.update(manifest["excluded_self_referential_control_files"])
    actual_output = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual_output - expected_output:
        raise ValueError("Release contains unaccounted files")
    result = {
        "schema": "v64_b2_portable_release_verification_v1",
        "status": "PASS",
        "release_manifest_sha256": _sha(output / "release_manifest.json"),
        "included_files_checked": len(manifest["included"]),
        "included_bytes_checked": sum(row["bytes"] for row in manifest["included"].values()),
        "omitted_ledger_entries_checked": len(manifest["omitted"]),
        "omitted_bytes_accounted": sum(row["bytes"] for row in manifest["omitted"].values()),
        "derived_files_checked": len(manifest["derived"]),
        "available_mapping_entries_checked": available_counts,
        "unavailable_mapping_entries_accounted": unavailable_counts,
        "source_local_full_seal_verified_during_export": manifest["source_local_full_seal_verified_before_copy"],
        "complete_replay_evidence_in_git": False,
        "omitted_evidence_reverified_from_clone": False,
        "mismatches": [],
        "new_physics_steps": 0,
        "new_optimizer_updates": 0,
        "new_samples": 0,
    }
    if frozen_external is not None:
        result["frozen_repository_input_files_checked"] = len(frozen_external)
        result["frozen_repository_input_bytes_checked"] = sum(row["bytes"] for row in frozen_external.values())
    verification_path = output / "release_verification.json"
    if verification_path.exists() and not refresh_verification:
        if _read(verification_path) != result:
            raise ValueError("Existing portable verification does not match fresh verification")
    else:
        _write(verification_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Sealed full local B.2 output (export only)")
    parser.add_argument("--output", type=Path, required=True, help="New release directory, or existing directory for --verify")
    parser.add_argument("--verify", action="store_true", help="Verify portable release without reading omitted local traces")
    parser.add_argument("--refresh-frozen-inputs", action="store_true",
                        help="Upgrade the unpublished v1 release with exact frozen source/document copies")
    args = parser.parse_args()
    if args.verify:
        if args.source is not None or args.refresh_frozen_inputs:
            parser.error("--source and --refresh-frozen-inputs are only used for export")
        result = verify(args.output)
    else:
        if args.source is None:
            parser.error("Export requires --source")
        result = export(args.source, args.output, args.refresh_frozen_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
