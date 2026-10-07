"""Copy sealed B.3.1 evidence into a self-verifying portable GitHub release.

This module imports no experiment, simulator, optimizer, or sampler. Export is
exclusive: it never updates an existing release. Verify needs only the release
directory, including exact frozen source/input copies, and never follows local
absolute paths recorded in the original experiment's JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
from typing import Any


SCHEMA = "v64_b31_portable_conditional_release_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
SOURCE_CONTROLS = ("manifest.json", "verification.json")
RELEASE_CONTROLS = ("release_manifest.json", "release_verification.json")
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".csv", ".tsv", ".txt", ".toml",
                 ".yaml", ".yml", ".py", ".html", ".css", ".js", ".svg"}
VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".webp"}
SMALL_ARRAY_NAMES = {"raw_outputs.npz", "dataset.npz", "labels.npz", "exposure_counts.npz"}
SMALL_WEIGHT_NAMES = {"selected.pt", "last.pt"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def relative_file(root, name):
    value = str(name)
    parts = PurePosixPath(value).parts
    if (not parts or "\\" in value or ":" in value or PurePosixPath(value).is_absolute()
            or any(part in (".", "..") for part in parts)):
        raise ValueError(f"Unsafe relative artifact path: {value}")
    result = Path(root).joinpath(*parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Artifact path escapes its directory: {value}")
    return result


def check(path, expected):
    if not Path(path).is_file():
        raise ValueError(f"Missing artifact: {path}")
    if identity(path) != {key: expected[key] for key in ("sha256", "bytes")}:
        raise ValueError(f"Artifact SHA256 or size mismatch: {path}")


def normal(path):
    return str(path).replace("\\", "/").rstrip("/")


def under(path, root):
    candidate, prefix = normal(path), normal(root)
    if candidate.casefold().startswith(prefix.casefold() + "/"):
        return candidate[len(prefix) + 1:]
    return None


def historical_join(root, name):
    separator = "\\" if "\\" in str(root) else "/"
    return str(root).rstrip("/\\") + separator + str(name).replace("/", separator)


def _record(value):
    if (not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
            or not isinstance(value.get("bytes"), int) or isinstance(value["bytes"], bool)
            or value["bytes"] < 0):
        raise ValueError("The source inventory requires literal SHA256 and byte counts")
    return {key: value[key] for key in ("sha256", "bytes")}


def payload(manifest):
    raw = manifest.get("payload", manifest.get("files"))
    if isinstance(raw, list):
        if len({row["path"] for row in raw}) != len(raw):
            raise ValueError("Duplicate source payload path")
        raw = {row["path"]: row for row in raw}
    if not isinstance(raw, dict):
        raise ValueError("Source seal needs a payload or files inventory")
    return {name: _record(row) for name, row in raw.items()}


def external(manifest):
    raw = manifest.get("external", {})
    if isinstance(raw, list):
        if len({row["path"] for row in raw}) != len(raw):
            raise ValueError("Duplicate source external input path")
        raw = {row["path"]: row for row in raw}
    if not isinstance(raw, dict):
        raise ValueError("Invalid sealed external input inventory")
    return {name: _record(row) for name, row in raw.items()}


def clean_verification(value, manifest_sha):
    passed = value.get("status") == "PASS" or value.get("passed") is True
    bound = value.get("manifest_sha256", value.get("source_manifest_sha256"))
    if not passed or value.get("mismatches") or bound != manifest_sha:
        raise ValueError("The source verification is not a clean PASS bound to this manifest")


def seal_inventory(source):
    manifest, verification = read(source/"manifest.json"), read(source/"verification.json")
    clean_verification(verification, sha(source/"manifest.json"))
    inventory = payload(manifest)
    if set(inventory) & set(SOURCE_CONTROLS):
        raise ValueError("Self-referential source controls must not be payload entries")
    inventory.update({name: identity(source/name) for name in SOURCE_CONTROLS})
    actual = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    if actual != set(inventory):
        raise ValueError("The sealed source directory contains missing or unaccounted files")
    for name, row in inventory.items():
        check(relative_file(source, name), row)
    for name, row in external(manifest).items():
        check(Path(name), row)
    if "source_identity.json" not in inventory or "REPORT.md" not in inventory:
        raise ValueError("The terminal source must include source_identity.json and REPORT.md")
    return manifest, verification, inventory


def selection(name, size, limit):
    """A frozen selection rule; no result-dependent curation or file rewriting."""
    path = PurePosixPath(name)
    if name in SOURCE_CONTROLS:
        return True, "source seal control"
    if size > limit:
        return False, "file exceeds the fixed portable per-file byte limit"
    if path.suffix.lower() in TEXT_SUFFIXES | VISUAL_SUFFIXES:
        return True, "light text, tabular, source, or static visual evidence"
    if path.name in SMALL_ARRAY_NAMES:
        return True, "small declared raw-output or dataset array"
    if path.name in SMALL_WEIGHT_NAMES:
        return True, "small selected or final checkpoint"
    return False, "heavy replay/per-sample binary or non-curated binary format"


def source_references(source_identity, sealed_external, source_root, repository_root):
    """Reconstruct expected hashes without opening paths, including in verify."""
    result = {name: {**row, "roles": ["sealed_external"]} for name, row in sealed_external.items()}
    def add(name, expected_sha, role):
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("Frozen source/protected input SHA256 is malformed")
        row = result.setdefault(name, {"sha256": expected_sha, "bytes": None, "roles": []})
        if row["sha256"] != expected_sha:
            raise ValueError(f"Conflicting frozen input identity: {name}")
        if role not in row["roles"]:
            row["roles"].append(role)
    source_hashes = source_identity.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Frozen source identity needs nonempty source_sha256")
    for name, digest in source_hashes.items():
        relative_file(Path(repository_root), name)
        add(historical_join(repository_root, name), digest, "execution_source")
    for name, digest in source_identity.get("protected_artifacts", {}).items():
        add(name, digest, "protected_input")
    return result


def _copy(source, output, destination, expected):
    target = relative_file(output, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    check(target, expected)


def _absolute_references(snapshot, included):
    """Index local provenance strings; unsealed paths never become proof files."""
    result = {}
    def visit(value, location, origin):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, location+"/"+str(key), origin)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, location+"/"+str(index), origin)
        elif isinstance(value, str) and (re.match(r"^[A-Za-z]:[/\\]", value) or value.startswith("/")):
            result.setdefault(value, []).append({"snapshot_path": origin, "json_location": location})
    for name in included:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in (".json", ".jsonl"):
            continue
        try:
            if suffix == ".jsonl":
                with relative_file(snapshot, name).open(encoding="utf8") as stream:
                    for index, line in enumerate(stream):
                        if line.strip():
                            visit(json.loads(line), "/line/"+str(index+1), "snapshot/"+name)
                continue
            value = read(relative_file(snapshot, name))
        except (UnicodeError, json.JSONDecodeError):
            continue
        visit(value, "", "snapshot/"+name)
    return result


def _portable_report(source, mappings):
    report = (source/"REPORT.md").read_text(encoding="utf8")
    lookup = {normal(row["original_path"]).casefold(): row for row in mappings.values()}
    def replacement(match):
        label, destination = match.groups()
        destination = destination.strip("<>")
        key = normal(destination).casefold()
        row = lookup.get(key)
        if row is None and not re.match(r"^[A-Za-z]:[/\\]|^/|^[a-z]+://", destination):
            resolved = posixpath.normpath(normal(historical_join(str(source), destination)))
            row = lookup.get(resolved.casefold())
        if row is None:
            if re.match(r"^[A-Za-z]:[/\\]|^/", destination):
                return label.lstrip("!")+" (local provenance path; unavailable from this clone)"
            return match.group(0)
        if row.get("release_relative_path"):
            return label+"("+row["release_relative_path"]+")"
        return label.lstrip("!")+" (omitted; see portable_paths.json)"
    report = re.sub(r"(!?\[[^\]\n]*\])\((<[^>\n]*>|[^)\n]*)\)", replacement, report)
    banner = """# Portable evidence view — V6.4-B.3.1

The original sealed [REPORT.md](snapshot/REPORT.md) and all copied files retain
their original bytes. The report below changes only local link destinations.
Absolute paths in the original JSON are provenance strings. Available files
resolve through [portable_paths.json](portable_paths.json); omitted or unbound
local references are explicitly unavailable from a Git clone.

This is a curated release, not a complete physics/replay archive. The complete
source inventory, omitted file SHA256/size ledger, exact frozen source copies,
and source/producer relationships are preserved in [release_manifest.json](release_manifest.json).
Verification checks copied bytes and ledger fidelity; it does not re-execute
physics or certify the unavailable replay files.

```sh
python -m v6_4.export_execution_aware_release --verify --output v6_4/releases/execution_aware_route_teacher_20261007_01
```

Export adds zero physics steps, optimizer updates, and samples. It does not
change any research conclusion or deployment status.

---

"""
    return banner+report


def _git_head(repository):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout.decode("ascii").strip()


def export(source, output, *, repository=None, max_file_bytes=DEFAULT_MAX_FILE_BYTES):
    source, output = Path(source).resolve(), Path(output).resolve()
    repository = Path(repository or REPOSITORY_ROOT).resolve()
    if not isinstance(max_file_bytes, int) or max_file_bytes < 1:
        raise ValueError("The fixed per-file byte limit must be a positive integer")
    if output.exists():
        raise FileExistsError("Release output exists; use --verify, never overwrite")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and release directories must be separate")
    sealed, verification, inventory = seal_inventory(source)
    execution_identity = read(source/"source_identity.json")
    references = source_references(execution_identity, external(sealed), str(source), str(repository))
    for name, row in references.items():
        actual = identity(Path(name))
        if actual["sha256"] != row["sha256"] or row["bytes"] not in (None, actual["bytes"]):
            raise ValueError(f"Frozen execution source/input changed: {name}")
        row["bytes"] = actual["bytes"]
    source_controls = {name: identity(source/name) for name in SOURCE_CONTROLS}
    included, omitted, mappings, frozen = {}, {}, {}, {}
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in sorted(inventory.items()):
        retained, reason = selection(name, expected["bytes"], max_file_bytes)
        original = historical_join(str(source), name)
        row = {**expected, "reason": reason, "original_path": original}
        if retained:
            row["release_relative_path"] = "snapshot/"+name
            included[name] = row
            _copy(relative_file(source, name), output, row["release_relative_path"], expected)
        else:
            row["release_relative_path"] = None
            omitted[name] = row
        mappings[original] = {**row, "kind": "INCLUDED_SOURCE_ARTIFACT" if retained else "OMITTED_HEAVY",
            "source_relative_path": name, "portable_available": retained}
    for index, (original, row) in enumerate(sorted(references.items())):
        source_relative = under(original, str(source))
        if source_relative is not None:
            if source_relative not in inventory or _record(row) != inventory[source_relative]:
                raise ValueError("A protected source input is not bound to the sealed payload")
            mappings[original] = {**mappings[historical_join(str(source), source_relative)],
                "original_path": original, "roles": row["roles"]}
            continue
        repository_relative = under(original, str(repository))
        retained, reason = selection(PurePosixPath(normal(original)).name, row["bytes"], max_file_bytes)
        if "execution_source" in row["roles"]:
            retained, reason = True, "exact frozen execution source bytes"
        destination = None
        if retained:
            destination = ("frozen_repository_inputs/"+repository_relative if repository_relative is not None
                else f"frozen_external_inputs/{index:04d}_{row['sha256'][:16]}/"+PurePosixPath(normal(original)).name)
            _copy(Path(original), output, destination, row)
        mapped = {**row, "original_path": original, "release_relative_path": destination,
            "repository_relative_path": repository_relative, "reason": reason,
            "kind": "FROZEN_INPUT_COPY" if retained else "OMITTED_EXTERNAL",
            "portable_available": retained}
        mappings[original] = mapped
        frozen[original] = mapped
    exporter_destination = "frozen_publication_inputs/export_execution_aware_release.py"
    _copy(Path(__file__), output, exporter_destination, identity(Path(__file__)))
    absolute_refs = _absolute_references(output/"snapshot", included)
    mapped_normal = {normal(name).casefold() for name in mappings}
    unbound = {name: {"kind": "UNBOUND_LOCAL_REFERENCE", "portable_available": False,
        "reason": "local path not an identity-bound sealed artifact; never followed by portable verify",
        "occurrences": occurrences}
        for name, occurrences in sorted(absolute_refs.items()) if normal(name).casefold() not in mapped_normal}
    paths = {"schema": "v64_b31_portable_paths_v1", "original_paths_are_provenance_only": True,
        "absolute_paths_are_never_opened_by_portable_verify": True,
        "original_source_root": str(source), "original_repository_root": str(repository),
        "records": mappings, "unbound_absolute_references": unbound}
    write(output/"portable_paths.json", paths)
    (output/"report.md").write_text(_portable_report(source, mappings), encoding="utf8", newline="\n")
    relationship = {key: sealed[key] for key in (
        "source_producer_commit", "algorithm_producer_commit", "development_git_head", "delivery_git_head") if key in sealed}
    relationship["execution_source_identity_git_head"] = execution_identity.get("git_head")
    relationship["publication_checkout_git_head"] = _git_head(repository)
    relationship["source_identity_sha256"] = inventory["source_identity.json"]["sha256"]
    relationship["exact_execution_source_copies_preserve_checkout_bytes"] = True
    summary = {"schema": "v64_b31_release_summary_v1", "report": "report.md",
        "original_report": "snapshot/REPORT.md", "source_producer_relationship": relationship,
        "source_manifest_sha256": source_controls["manifest.json"]["sha256"],
        "source_verification_sha256": source_controls["verification.json"]["sha256"],
        "included_files": len(included), "included_bytes": sum(row["bytes"] for row in included.values()),
        "omitted_files": len(omitted), "omitted_bytes": sum(row["bytes"] for row in omitted.values()),
        "frozen_input_files": sum(row["portable_available"] for row in frozen.values()),
        "omitted_external_files": sum(not row["portable_available"] for row in frozen.values()),
        "complete_replay_evidence_in_git": False,
        "scientific_results_reinterpreted": False, "new_physics_steps": 0,
        "new_optimizer_updates": 0, "new_samples": 0}
    write(output/"release_summary.json", summary)
    derived = {name: identity(output/name) for name in ("portable_paths.json", "report.md", "release_summary.json")}
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"max_file_bytes": max_file_bytes, "text_suffixes": sorted(TEXT_SUFFIXES),
            "visual_suffixes": sorted(VISUAL_SUFFIXES), "small_array_names": sorted(SMALL_ARRAY_NAMES),
            "small_weight_names": sorted(SMALL_WEIGHT_NAMES)},
        "source_controls": source_controls, "source_producer_relationship": relationship,
        "source_local_full_seal_verified_before_copy": True,
        "included": included, "omitted": omitted, "frozen_inputs": frozen,
        "frozen_publication_inputs": {exporter_destination: identity(output/exporter_destination)},
        "derived": derived, "historical_result_fields_modified": False,
        "new_physics_steps": 0, "new_optimizer_updates": 0, "new_samples": 0,
        "excluded_self_referential_control_files": list(RELEASE_CONTROLS)}
    # Detect an in-progress run/source change before asserting a portable seal.
    _, _, final_inventory = seal_inventory(source)
    if final_inventory != inventory:
        raise ValueError("The source archive changed during export")
    for name, row in references.items():
        check(Path(name), row)
    write(output/"release_manifest.json", manifest)
    return verify(output)


def verify(output):
    """Verify copied bytes and omitted identities without any original archive."""
    output = Path(output).resolve()
    manifest = read(output/"release_manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported execution-aware release schema")
    policy = manifest["selection"]
    if (policy.get("text_suffixes") != sorted(TEXT_SUFFIXES)
            or policy.get("visual_suffixes") != sorted(VISUAL_SUFFIXES)
            or policy.get("small_array_names") != sorted(SMALL_ARRAY_NAMES)
            or policy.get("small_weight_names") != sorted(SMALL_WEIGHT_NAMES)
            or not isinstance(policy.get("max_file_bytes"), int) or policy["max_file_bytes"] < 1):
        raise ValueError("Frozen curation policy changed")
    included, omitted = manifest["included"], manifest["omitted"]
    if set(included) & set(omitted):
        raise ValueError("The included/omitted source inventories overlap")
    artifacts = {}
    for name, row in included.items():
        if row["release_relative_path"] != "snapshot/"+name:
            raise ValueError("Source artifact destination is not its exact snapshot path")
        artifacts[row["release_relative_path"]] = _record(row)
    for original, row in manifest["frozen_inputs"].items():
        if original != row["original_path"]:
            raise ValueError("Frozen input identity key differs")
        if row["portable_available"] != (row["release_relative_path"] is not None):
            raise ValueError("Frozen input availability and destination disagree")
        if row["portable_available"]:
            artifacts[row["release_relative_path"]] = _record(row)
    artifacts.update(manifest["frozen_publication_inputs"])
    artifacts.update(manifest["derived"])
    for name, row in artifacts.items():
        check(relative_file(output, name), row)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual - set(RELEASE_CONTROLS) != set(artifacts):
        raise ValueError("The release contains missing or unaccounted files")
    for name, expected in manifest["source_controls"].items():
        check(output/"snapshot"/name, expected)
    sealed = read(output/"snapshot/manifest.json")
    verification = read(output/"snapshot/verification.json")
    clean_verification(verification, manifest["source_controls"]["manifest.json"]["sha256"])
    original_inventory = payload(sealed)
    original_inventory.update(manifest["source_controls"])
    if set(original_inventory) != set(included) | set(omitted):
        raise ValueError("The portable source partition does not cover the complete original seal")
    for name, expected in original_inventory.items():
        row = included.get(name, omitted.get(name))
        if _record(row) != expected:
            raise ValueError("An included or omitted original SHA256/size changed")
        retained, reason = selection(name, expected["bytes"], policy["max_file_bytes"])
        if retained != (name in included) or row["reason"] != reason:
            raise ValueError("The included/omitted selection rule changed")
    paths = read(output/"portable_paths.json")
    execution_identity = read(output/"snapshot/source_identity.json")
    references = source_references(execution_identity, external(sealed),
        paths["original_source_root"], paths["original_repository_root"])
    for original, expected in references.items():
        if original not in paths["records"]:
            raise ValueError("A sealed execution source/input has no portable mapping")
        actual_record = paths["records"][original]
        if actual_record["sha256"] != expected["sha256"] or expected["bytes"] not in (None, actual_record["bytes"]):
            raise ValueError("A portable frozen input identity changed")
        if under(original, paths["original_source_root"]) is None:
            if manifest["frozen_inputs"].get(original) != actual_record:
                raise ValueError("Frozen input ledger and portable mapping disagree")
            retained, reason = selection(PurePosixPath(normal(original)).name, actual_record["bytes"], policy["max_file_bytes"])
            if "execution_source" in expected["roles"]:
                retained, reason = True, "exact frozen execution source bytes"
            if actual_record["portable_available"] != retained or actual_record["reason"] != reason:
                raise ValueError("Frozen input copy/omission rule changed")
    for name, expected in original_inventory.items():
        original = historical_join(paths["original_source_root"], name)
        row = paths["records"].get(original)
        target = included.get(name, omitted.get(name))
        if (row is None or _record(row) != expected
                or row["release_relative_path"] != target["release_relative_path"]
                or row["portable_available"] != (name in included)):
            raise ValueError("Source artifact mapping changed or is missing")
    expected_mapping_keys = set(references) | {historical_join(paths["original_source_root"], name) for name in original_inventory}
    if set(paths["records"]) != expected_mapping_keys:
        raise ValueError("Portable mapping contains extra or missing records")
    expected_frozen = {name for name in references if under(name, paths["original_source_root"]) is None}
    if set(manifest["frozen_inputs"]) != expected_frozen:
        raise ValueError("Frozen source/input ledger does not exactly cover external references")
    for row in paths["unbound_absolute_references"].values():
        if row["kind"] != "UNBOUND_LOCAL_REFERENCE" or row["portable_available"] is not False:
            raise ValueError("An unbound local path is being represented as portable evidence")
    summary = read(output/"release_summary.json")
    if summary["source_producer_relationship"] != manifest["source_producer_relationship"]:
        raise ValueError("Source/producer relationship differs between release controls")
    expected_summary_counts = {
        "included_files": len(included), "included_bytes": sum(row["bytes"] for row in included.values()),
        "omitted_files": len(omitted), "omitted_bytes": sum(row["bytes"] for row in omitted.values()),
        "frozen_input_files": sum(row["portable_available"] for row in manifest["frozen_inputs"].values()),
        "omitted_external_files": sum(not row["portable_available"] for row in manifest["frozen_inputs"].values())}
    if any(summary[key] != value for key, value in expected_summary_counts.items()):
        raise ValueError("Release summary counts differ from the verified portable inventory")
    for record in (summary, manifest):
        if (record["new_physics_steps"] != 0 or record["new_optimizer_updates"] != 0
                or record["new_samples"] != 0):
            raise ValueError("Artifact publication cannot claim new execution or learning work")
    result = {"schema": "v64_b31_portable_release_verification_v1", "status": "PASS",
        "release_manifest_sha256": sha(output/"release_manifest.json"),
        "verified_portable_files": len(artifacts), "included_source_files": len(included),
        "omitted_source_files": len(omitted), "omitted_source_bytes": sum(row["bytes"] for row in omitted.values()),
        "frozen_input_files": sum(row["portable_available"] for row in manifest["frozen_inputs"].values()),
        "omitted_external_files": sum(not row["portable_available"] for row in manifest["frozen_inputs"].values()),
        "unbound_local_reference_paths": len(paths["unbound_absolute_references"]),
        "original_absolute_paths_opened": 0,
        "omitted_physical_replay_bytes_verified_from_clone": False,
        "new_physics_steps": 0, "new_optimizer_updates": 0, "new_samples": 0}
    control = output/"release_verification.json"
    if control.exists():
        if read(control) != result:
            raise ValueError("The retained release verification does not match this release")
    else:
        write(control, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args()
    if args.verify:
        if args.source:
            parser.error("--verify uses portable copies only; do not supply --source")
        result = verify(args.output)
    else:
        if not args.source:
            parser.error("export requires --source")
        result = export(args.source, args.output, max_file_bytes=args.max_file_bytes)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
