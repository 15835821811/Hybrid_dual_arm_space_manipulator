"""Observe B3.1's remote commit and prove publication-tree byte parity.

This standalone Git/file utility does not import experiment code. It reads
origin refs and fetches only the exact new branch (local FETCH_HEAD update).
It never pushes or modifies a branch, experiment, release, or source file.
An optional receipt records an observed commit; it cannot contain its own
future receipt commit. Omit --output to repeat identical checks to stdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

BRANCH = "v6.4-b3-1-execution-aware-route-teacher"
NEW_REF = "refs/heads/"+BRANCH
OLD_REFS = {
    "refs/heads/main": "84faf3063faeeaf5914e5acf4bacca55d0f02e80",
    "refs/heads/v6.4-b2-task-anchored-residual": "28ef7889be16b4a504d9449f52c50e3a99e82a31",
    "refs/heads/v6.4-b3-conditional-route-value": "bdd6df6b4c68a6c95872f4639b8b8269fb6856c7",
}
RUN = "execution_aware_route_teacher_20261007_01"
RELEASE = "v6_4/releases/"+RUN
VISUAL = "v6_4/visualization/"+RUN
REVIEW = "paper/review-traces/experiment-result-to-claim/2026-10-07_run03"
MODULES = ("v6_4/evaluate_execution_aware_teacher.py", "v6_4/plot_execution_aware_teacher.py",
           "v6_4/deliver_execution_aware_teacher.py", "v6_4/export_execution_aware_release.py")
DIRECT = (
    "README.md", "v6_lite/README.md", "docs/V6_4_B31_VISUALIZATION.md",
    "docs/V6_2_LATEST_VISUALIZATION.md", "v6_lite/visualization/index.html",
    "v6_4/visualization/index.html", ".gitattributes", "MANIFEST.md", "findings.md",
    "paper/EXPERIMENT_PLAN.md", "paper/EXPERIMENT_TRACKER.md", *MODULES,
)
VISUAL_FILES = ("fig_reference_actual.png", "fig_route_quality.png", "all_candidates.csv",
                "index.html", "dashboard_data.json", "teacher_records.json", "visualization_manifest.json")
RELEASE_FILES = ("release_manifest.json", "release_verification.json", "release_summary.json",
                 "portable_paths.json", "report.md")
RELEASE_CONTROLS = ("release_manifest.json", "release_verification.json")
SNAPSHOT_FILES = (
    "manifest.json", "verification.json", "report.json", "REPORT.md", "quality_matrix.json",
    "teacher_records.json", "all_candidates.csv", "final_conclusions.json", "execution_complete.json",
    "plan.json", "task_manifest.json", "source_identity.json", "baseline_diagnosis.json",
    "baseline_inputs_manifest.json", "reference_tests.json", "execution_diagnostics_checks.json",
    "preflight_checks.json", "protocol_clarifications.json", "figures/plot_manifest.json",
    "figures/fig_reference_actual.png", "figures/fig_route_quality.png",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def safe_error(value):
    # Keep a repository's configured credential URL out of failure receipts.
    return re.sub(r"(https?://)[^/\s@]+@", r"\1<redacted>@", str(value))[-12000:]


def git(repository, *arguments):
    result = subprocess.run(["git", *arguments], cwd=repository, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=False, check=False)
    if result.returncode:
        raise RuntimeError("git "+arguments[0]+" failed: "+safe_error(result.stderr.decode("utf8", errors="replace")))
    return result.stdout


def relative_name(value):
    name = str(value)
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and "\\" not in name and ":" not in name
            and not any(character in name for character in ("\x00", "\r", "\n"))
            and not any(part in (".", "..") for part in p.parts), "Unsafe publication path: "+name)
    return p.as_posix()


def path_in(repository, name):
    name = relative_name(name)
    path = repository.joinpath(*PurePosixPath(name).parts)
    require(path.resolve().is_relative_to(repository), "Publication path escapes repository")
    require(path.is_file(), "Required publication file missing: "+name)
    return path


def read(repository, name):
    return json.loads(path_in(repository, name).read_text(encoding="utf-8-sig"))


def observe_remote(repository, expected):
    refs = (*OLD_REFS, NEW_REF)
    output = git(repository, "ls-remote", "--heads", "origin", *refs).decode("ascii")
    observed = {}
    for line in output.splitlines():
        head, ref = line.split("\t", 1)
        require(ref in refs and ref not in observed and re.fullmatch(r"[0-9a-f]{40}", head),
                "Unexpected or duplicate origin ref response")
        observed[ref] = head
    require(set(observed) == set(refs), "Origin does not expose all four exact requested branches")
    for ref, head in OLD_REFS.items():
        require(observed[ref] == head, "Historical remote branch changed: "+ref)
    require(observed[NEW_REF] == expected, "New remote branch is not expected local HEAD")
    return {"observed_utc": utc(), "remote": "origin", "refs": observed}


def release_artifact_inventory(repository, manifest):
    """Select every curated copied artifact, without opening original paths.

    This mirrors only the exporter's membership contract. Full source-seal,
    omission-policy and external-ledger validation remain publisher duties.
    """
    require(manifest.get("schema") == "v64_b31_portable_conditional_release_v1",
            "Unsupported release manifest schema")
    controls = manifest["excluded_self_referential_control_files"]
    require(len(controls) == len(RELEASE_CONTROLS) and set(controls) == set(RELEASE_CONTROLS),
            "Release self-referential control inventory changed")
    artifacts = {}

    def add(relative, row, role):
        relative = relative_name(relative)
        require(relative not in RELEASE_CONTROLS, "Artifact collides with release control: "+relative)
        expected = row.get("sha256")
        size = row.get("bytes")
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
                and isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                "Invalid release artifact identity: "+relative)
        if relative in artifacts:
            require(artifacts[relative]["sha256"] == expected and artifacts[relative]["bytes"] == size,
                    "Conflicting release artifact identities: "+relative)
            artifacts[relative]["roles"].append(role)
        else:
            artifacts[relative] = {"sha256": expected, "bytes": size, "roles": [role]}

    for original, row in manifest["included"].items():
        source_relative = relative_name(original)
        require(row["release_relative_path"] == "snapshot/"+source_relative,
                "Included artifact destination is not its exact snapshot path: "+source_relative)
        add(row["release_relative_path"], row, "sealed_included_source_artifact")
    for original, row in manifest["frozen_inputs"].items():
        require(original == row["original_path"], "Frozen input identity key differs")
        available = row["portable_available"]
        require(isinstance(available, bool) and available == (row["release_relative_path"] is not None),
                "Frozen input availability and destination disagree")
        if available:
            add(row["release_relative_path"], row, "sealed_exact_frozen_input_copy")
    for relative, row in manifest["frozen_publication_inputs"].items():
        add(relative, row, "sealed_publication_generator_copy")
    for relative, row in manifest["derived"].items():
        add(relative, row, "sealed_derived_release_artifact")

    expected = set(artifacts) | set(RELEASE_CONTROLS)
    release_dir = repository.joinpath(*PurePosixPath(RELEASE).parts)
    require(release_dir.is_dir(), "Release directory missing")
    actual = {path.relative_to(release_dir).as_posix() for path in release_dir.rglob("*") if path.is_file()}
    missing, extra = sorted(expected-actual), sorted(actual-expected)
    require(not missing and not extra,
            "Release disk membership differs from manifest; missing="+repr(missing)+", extra="+repr(extra))
    names, expected_sha = {}, {}
    for relative, row in sorted(artifacts.items()):
        name = RELEASE+"/"+relative
        path = path_in(repository, name)
        require(path.stat().st_size == row["bytes"], "Release manifest byte count differs: "+name)
        names[name] = ";".join(row["roles"])
        expected_sha[name] = row["sha256"]
    for relative in RELEASE_CONTROLS:
        names[RELEASE+"/"+relative] = "self_referential_release_control"
    coverage = {"release_directory": RELEASE, "manifest_sha256": file_sha(release_dir/"release_manifest.json"),
                "artifact_files": len(artifacts), "control_files": len(RELEASE_CONTROLS),
                "complete_file_membership": sorted(expected), "disk_membership_exact": True,
                "complete_file_membership_sha256": canonical_sha(sorted(expected))}
    return names, expected_sha, coverage


def inventory(repository):
    names = {relative_name(name): "current_publication_entry_or_module" for name in DIRECT}
    names.update({VISUAL+"/"+name: "current_dashboard_or_figure" for name in VISUAL_FILES})
    names.update({RELEASE+"/"+name: "release_control_or_report" for name in RELEASE_FILES})
    names.update({RELEASE+"/snapshot/"+name: "sealed_snapshot_critical_report" for name in SNAPSHOT_FILES})
    for name in ("verdict.json", "inputs.json", "response.md", "prompt.md", "handoff.md"):
        names[REVIEW+"/"+name] = "independent_review_run03"
    review_dir = repository.joinpath(*PurePosixPath(REVIEW).parts)
    require(review_dir.is_dir(), "Independent review run03 missing")
    for path in review_dir.rglob("*"):
        if path.is_file():
            names[path.relative_to(repository).as_posix()] = "independent_review_run03_with_exact_inputs"
    release_manifest = read(repository, RELEASE+"/release_manifest.json")
    release_names, frozen_expected, release_coverage = release_artifact_inventory(repository, release_manifest)
    names.update(release_names)
    # Critical paths remain mandatory, but must also belong to the full manifest.
    mandatory_release = {RELEASE+"/"+name for name in RELEASE_FILES}
    mandatory_release.update(RELEASE+"/snapshot/"+name for name in SNAPSHOT_FILES)
    require(mandatory_release.issubset(release_names), "Critical release file is outside complete manifest membership")
    identity = read(repository, RELEASE+"/snapshot/source_identity.json")
    current_frozen_sources = []
    for relative, expected_sha in sorted(identity["source_sha256"].items()):
        relative = relative_name(relative)
        copy = RELEASE+"/frozen_repository_inputs/"+relative
        names[copy] = "exact_actual_producer_source_copy"
        if copy in frozen_expected:
            require(frozen_expected[copy] == expected_sha, "Release/source identity digest differs: "+relative)
        frozen_expected[copy] = expected_sha
        # Legacy Git text attributes can normalize original root source blobs.
        # The release's -text frozen copies retain exact executed checkout bytes.
        # Check current original bytes against their execution identity separately.
        original = path_in(repository, relative)
        require(file_sha(original) == expected_sha, "Current actual-producer source changed: "+relative)
        current_frozen_sources.append({"path": relative, "sha256": expected_sha, "committed_exact_copy": copy})
    require({name for name in names if name.startswith(RELEASE+"/")} == set(release_names),
            "Frozen actual source copy is outside complete release manifest membership")
    return names, frozen_expected, current_frozen_sources, identity, release_coverage


def verify_blob_bytes(repository, expected, names, frozen_expected):
    """Stream size-framed binary blobs from one git cat-file --batch process."""
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repository,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=False)
    blobs = []
    try:
        for name, role in sorted(names.items()):
            path = path_in(repository, name)
            current_sha, current_bytes = file_sha(path), path.stat().st_size
            if name in frozen_expected:
                require(current_sha == frozen_expected[name], "Frozen publication copy digest changed: "+name)
            process.stdin.write((expected+":"+name+"\n").encode("utf8"))
            process.stdin.flush()
            header = process.stdout.readline(1024*1024)
            match = re.fullmatch(rb"([0-9a-f]{40}|[0-9a-f]{64}) blob ([0-9]+)\n", header)
            require(match is not None, "Required path is not a committed blob: "+name)
            blob_bytes = int(match.group(2))
            require(blob_bytes == current_bytes, "Committed blob byte count differs from current file: "+name)
            remaining, h = blob_bytes, hashlib.sha256()
            while remaining:
                block = process.stdout.read(min(1024*1024, remaining))
                require(bool(block), "Truncated cat-file binary body: "+name)
                h.update(block)
                remaining -= len(block)
            require(process.stdout.read(1) == b"\n", "Invalid cat-file binary frame delimiter: "+name)
            blob_sha = h.hexdigest()
            require(blob_sha == current_sha, "Committed raw bytes differ from current file: "+name)
            blobs.append({"path": name, "role": role, "sha256": blob_sha, "bytes": blob_bytes,
                          "HEAD_blob_equals_current_file_bytes": True})
        process.stdin.close()
        require(process.stdout.read() == b"", "Unexpected trailing cat-file batch output")
        stderr = process.stderr.read()
        require(process.wait() == 0, "git cat-file batch failed: "+safe_error(stderr.decode("utf8", errors="replace")))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    return blobs


def verify(repository, expected):
    started = utc()
    repository = Path(repository).resolve()
    require(repository.is_dir(), "Repository directory missing")
    require(re.fullmatch(r"[0-9a-f]{40}", expected) is not None, "--expected-head requires exact lowercase 40-character SHA")
    top = Path(git(repository, "rev-parse", "--show-toplevel").decode("utf8").strip()).resolve()
    require(top == repository, "--repository must be the actual worktree root")
    local = git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    require(local == expected, "Local HEAD differs from --expected-head")
    before = observe_remote(repository, expected)
    git(repository, "fetch", "--no-tags", "--no-recurse-submodules", "--refmap=", "origin", NEW_REF)
    fetched = git(repository, "rev-parse", "FETCH_HEAD").decode("ascii").strip()
    require(fetched == expected, "Exact new ref fetch differs from expected commit")
    names, frozen_expected, current_sources, identity, release_coverage = inventory(repository)
    release_tree = git(repository, "ls-tree", "-r", "--name-only", "-z", expected, "--", RELEASE)
    committed_release = {name.decode("utf8") for name in release_tree.split(b"\x00") if name}
    expected_release = {RELEASE+"/"+name for name in release_coverage["complete_file_membership"]}
    require(committed_release == expected_release,
            "Committed release membership differs from complete manifest/disk inventory; missing="+
            repr(sorted(expected_release-committed_release))+", extra="+repr(sorted(committed_release-expected_release)))
    release_coverage["committed_tree_membership_exact"] = True
    blobs = verify_blob_bytes(repository, expected, names, frozen_expected)
    # Detect a concurrent checkout/source edit or remote move during the proof.
    for row in blobs:
        require(file_sha(path_in(repository, row["path"])) == row["sha256"], "Local artifact changed during verification")
    for row in current_sources:
        require(file_sha(path_in(repository, row["path"])) == row["sha256"], "Actual source changed during verification")
    require(git(repository, "rev-parse", "HEAD").decode("ascii").strip() == expected, "Local HEAD moved during verification")
    after = observe_remote(repository, expected)
    require(before["refs"] == after["refs"], "Remote heads changed during proof")
    return {"schema": "v64_b31_publication_commit_remote_tree_proof_v1", "status": "PASS",
        "started_utc": started, "completed_utc": utc(), "repository": str(repository),
        "proof_subject": "Observed origin commit and its published artifact tree; this receipt does not know or claim its own future receipt commit",
        "observed_remote_commit": expected, "expected_local_HEAD": expected, "fetched_exact_new_ref_HEAD": fetched,
        "remote_branch": BRANCH, "remote_observations": [before, after],
        "historical_remote_heads_unchanged": OLD_REFS,
        "actual_algorithm_producer_commit": identity["algorithm_producer_commit"],
        "verifier_source_sha256": digest(Path(__file__).read_bytes()),
        "publication_modules": list(MODULES), "verified_raw_blob_files": len(blobs),
        "release_coverage": release_coverage,
        "complete_curated_release_membership_and_commit_bytes_verified": True,
        "verified_current_frozen_source_files": len(current_sources),
        "blob_inventory_sha256": canonical_sha(blobs), "blob_checks": blobs,
        "current_frozen_source_inventory_sha256": canonical_sha(current_sources),
        "current_frozen_source_checks": current_sources,
        "byte_scope": "git cat-file --batch size-framed raw blob bytes equal current file bytes for every curated release artifact/control and selected publication file; no line-ending/filter normalization. Legacy executed source byte identities are carried in exact -text release copies, with current originals separately checked against execution SHA.",
        "local_complete_manifest_or_external_validation_repeated": False,
        "physics_steps": 0, "geometry_queries": 0, "QP_solves": 0, "DDIM_samples": 0,
        "training_runs": 0, "pushes": 0, "remote_writes": 0,
        "local_git_mutation": "FETCH_HEAD and object database only; exact new branch fetched, no tags/branch checkout/update"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Exclusive JSON receipt; omit to verify the receipt-containing commit to stdout only")
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    if output is not None and output.exists():
        parser.error("Exclusive output already exists; do not overwrite a remote proof")
    try:
        proof = verify(args.repository, args.expected_head)
        code = 0
    except Exception as error:
        proof = {"schema": "v64_b31_publication_commit_remote_tree_proof_v1", "status": "FAIL",
            "completed_utc": utc(), "expected_local_HEAD": args.expected_head,
            "error_type": type(error).__name__, "error": safe_error(error),
            "proof_subject": "Failed verification; no successful remote/artifact proof asserted"}
        code = 1
    if output is not None:
        # Parent directory must already exist; creating it is a publisher action.
        with output.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    summary = {key: value for key, value in proof.items()
               if key not in ("blob_checks", "current_frozen_source_checks")}
    summary["receipt_written"] = output is not None
    if output is not None:
        summary["receipt_path"] = str(output)
        summary["receipt_sha256"] = file_sha(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
