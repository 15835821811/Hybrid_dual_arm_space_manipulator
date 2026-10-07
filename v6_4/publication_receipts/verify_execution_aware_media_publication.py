"""Prove that the B3.1 media supplement and preserved research are on origin.

Reuses the original research publication verifier without changing its
historical receipts, manifests, or frozen artifact membership.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

BASE = "e67f8cfee583fb3d42fad69b8372fbb8dd0e8d16"
SOURCE_MANIFEST_SHA = "6903dcc58ee793bb576d59b71732d6c73af2440e9e0d96b7950e6eeae8fbe9e5"
ACTUAL_PRODUCER = "d4464c8ae2aa7913a730ebbd775917e7a3b1af71"
MEDIA = "v6_4/visualization/execution_aware_media_20261007_01"
ENTRY_FILES = {"README.md", "docs/V6_4_B31_VISUALIZATION.md", "docs/V6_2_LATEST_VISUALIZATION.md",
               "v6_lite/README.md", "v6_4/visualization/index.html", "v6_lite/visualization/index.html"}
MODULES = (
    "v6_4/visualization/render_execution_aware_replays.py",
    "v6_4/plot_execution_aware_media.py",
    "v6_4/visualization/build_execution_aware_media_gallery.py",
    "v6_4/visualization/validate_execution_aware_media.py",
    "v6_4/visualization/extract_execution_aware_route_posters.py",
    "v6_4/publication_receipts/verify_execution_aware_media_publication.py",
    "docs/V6_4_B31_MEDIA_SUPPLEMENT.md",
    "v6_4/publication_receipts/execution_aware_media_20261007_01_original_seal_before_entries.json",
    "v6_4/publication_receipts/execution_aware_media_20261007_01_current_entries.json",
)


def load_verifier():
    path = Path(__file__).with_name("verify_execution_aware_publication.py")
    spec = importlib.util.spec_from_file_location("research_publication_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(repository, expected):
    original = load_verifier()
    repository = Path(repository).resolve()
    research = original.verify(repository, expected)
    require = original.require
    preserved = ("v6_4/output/"+original.RUN, original.RELEASE, original.VISUAL, original.REVIEW, *original.MODULES)
    changed = original.git(repository, "diff", "--name-only", BASE, expected, "--", *preserved).decode("utf8").splitlines()
    require(not changed, "Sealed research publication artifacts changed: "+repr(changed))
    original.git(repository, "merge-base", "--is-ancestor", BASE, expected)
    folder = original.path_in(repository, MEDIA+"/media_manifest.json").parent
    manifest = original.read(repository, MEDIA+"/media_manifest.json")
    validation = original.read(repository, MEDIA+"/validation.json")
    require(validation["status"] == "PASS", "Media validation is not PASS")
    require(validation["media_manifest_sha256"] == original.file_sha(folder/"media_manifest.json"), "PASS receipt belongs to a different media manifest")
    require(validation["validator_source"]["sha256"] == original.file_sha(original.path_in(repository, "v6_4/visualization/validate_execution_aware_media.py")), "PASS receipt belongs to a different validator source")
    require(validation["source_manifest_sha256"] == SOURCE_MANIFEST_SHA
            and validation["actual_algorithm_producer_commit"] == ACTUAL_PRODUCER,
            "PASS receipt scientific source identity differs")
    require(validation["fixed_slots_checked"] == 28 and validation["mp4_files_probed_and_decoded"] == 196
            and validation["decoded_frames_checked"] == 588
            and validation["route_poster_count"] == validation["decoded_route_poster_frames_checked"] == 56
            and validation["selected_png_figures"] == validation["selected_pdf_figures"] == 117
            and validation["selected_CSV_files"] == 84, "PASS receipt media/numeric coverage is incomplete")
    require(validation["new_physics_steps"] == validation["new_geometry_queries"]
            == validation["new_QP_solves"] == validation["new_model_calls"] == 0,
            "PASS receipt contains new simulation/solver/model work")
    require(manifest["record_count"] == 28 and manifest["video_count"] == 196, "Incomplete fixed media cohort")
    require(manifest["base_research_publication_commit"] == BASE, "Wrong research base")
    controls = {"media_manifest.json", "validation.json"}
    require(set(manifest["excluded_self_referential_files"]) == controls, "Control inventory differs")
    require(set(manifest["payload"]) | controls == {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}, "Media membership differs from seal")
    names = {MEDIA+"/"+name: "sealed_media_supplement_artifact" for name in manifest["payload"]}
    names.update({MEDIA+"/"+name: "media_supplement_control" for name in controls})
    names.update({name: "media_generator_validator_or_documentation" for name in MODULES})
    for path in (repository/"v6_4/publication_receipts").glob("execution_aware_media_20261007_01*.json"):
        names[path.relative_to(repository).as_posix()] = "media_supplement_publication_or_audit_receipt"
    expected_hashes = {MEDIA+"/"+name: row["sha256"] for name, row in manifest["payload"].items()}
    tree = original.git(repository, "ls-tree", "-r", "--name-only", "-z", expected, "--", MEDIA)
    require({p.decode("utf8") for p in tree.split(b"\x00") if p} == {name for name in names if name.startswith(MEDIA+"/")}, "Committed media membership is incomplete")
    blobs = original.verify_blob_bytes(repository, expected, names, expected_hashes)
    for entry in ("v6_4/visualization/index.html", "v6_lite/visualization/index.html"):
        text = original.path_in(repository, entry).read_text(encoding="utf8")
        require("execution_aware_media_20261007_01/index.html" in text, "Current entry does not link to media gallery")
    entry_proof = original.read(repository, "v6_4/publication_receipts/execution_aware_media_20261007_01_current_entries.json")
    require(entry_proof["status"] == "PASS"
            and entry_proof["media_manifest_sha256"] == original.file_sha(folder/"media_manifest.json")
            and entry_proof["media_validation_sha256"] == original.file_sha(folder/"validation.json"),
            "Current entries receipt is not bound to this validated media")
    require(set(entry_proof["updated_entries"]) == ENTRY_FILES, "Current entry receipt omits or adds publication entry paths")
    before_entries = original.read(repository, "v6_4/publication_receipts/execution_aware_media_20261007_01_original_seal_before_entries.json")
    require(before_entries["status"] == "PASS" and before_entries["before_media_current_entry_updates"]
            and before_entries["manifest_sha256"] == SOURCE_MANIFEST_SHA
            and before_entries["payload_files"] == 737 and before_entries["external_files"] == 699,
            "Historical full-seal-before-entry-update evidence differs")
    for name, row in entry_proof["updated_entries"].items():
        require(original.file_sha(original.path_in(repository, name)) == row["sha256"], "Entry differs from update receipt")
        if name.endswith(".md"):
            before = original.git(repository, "show", BASE+":"+name)
            require(original.path_in(repository, name).read_bytes().endswith(before), "Original documentation body changed beyond the supplement prefix")
    observed = original.observe_remote(repository, expected)
    return {"schema": "v64_b31_media_remote_tree_proof_v1", "status": "PASS",
            "completed_utc": original.utc(), "observed_remote_commit": expected,
            "base_research_publication_commit": BASE,
            "proof_subject": "Observed artifact commit; receipt does not claim its own future commit",
            "preserved_research_paths": list(preserved), "preserved_research_diff": changed,
            "complete_research_release_verified_files": research["release_coverage"]["artifact_files"]+research["release_coverage"]["control_files"],
            "frozen_actual_source_files_verified": research["verified_current_frozen_source_files"],
            "original_publication_verified_files": research["verified_raw_blob_files"],
            "media_publication_verified_files": len(blobs),
            "media_manifest_sha256": original.file_sha(folder/"media_manifest.json"),
            "media_validation_sha256": original.file_sha(folder/"validation.json"),
            "media_record_count": 28, "media_video_count": 196,
            "media_raw_blob_inventory_sha256": original.canonical_sha(blobs),
            "media_blob_checks": blobs, "historical_remote_heads_unchanged": original.OLD_REFS,
            "remote_observation": observed, "verifier_source_sha256": original.file_sha(__file__),
            "local_original_full_external_path_seal_repeated": False,
            "historical_seal_scope": "Original mutable documentation paths evolve in this later supplement; historical receipts and exact frozen portable copies remain unchanged.",
            "physics_steps": 0, "QP_solves": 0, "training_updates": 0,
            "remote_writes": 0, "local_git_mutation": "FETCH_HEAD and object database only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Receipt exists; refuse overwrite")
    original = load_verifier()
    try:
        proof = verify(args.repository, args.expected_head)
        code = 0
    except Exception as error:
        proof = {"schema": "v64_b31_media_remote_tree_proof_v1", "status": "FAIL",
                 "expected_commit": args.expected_head, "completed_utc": original.utc(),
                 "error_type": type(error).__name__, "error": original.safe_error(error)}
        code = 1
    if args.output:
        with args.output.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "media_blob_checks"}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
