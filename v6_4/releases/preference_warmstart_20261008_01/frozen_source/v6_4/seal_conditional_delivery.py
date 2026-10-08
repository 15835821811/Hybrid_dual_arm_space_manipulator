"""Seal a completed B.3 local delivery using only saved files and hashes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


CONTROLS = {"manifest.json", "verification.json"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--visualization", required=True, type=Path)
    args = parser.parse_args()
    source, repository, visualization = (x.resolve() for x in
        (args.source, args.repository, args.visualization))
    assert not any((source/name).exists() for name in CONTROLS), "Exclusive seal already exists"
    report = read(source/"report.json")
    assert report["research_delivery_complete"] and report["research_execution_complete"]
    assert report["route_value_identifiable"] is False and report["training_executed"] is False
    assert report["pilot_full_task_success"] == {"numerator": 6, "denominator": 6}
    assert report["deployment"] == "NOT_MET"
    checks = read(source/"preseal_checks.json")
    assert checks["passed"] and not checks["allow_pending_delivery"]
    assert checks["report_sha256"] == identity(source/"report.json")["sha256"]
    for row in checks["input_bindings"]:
        assert identity(Path(row["path"]))["sha256"] == row["sha256"], row["path"]
    static = read(source/"visualization_validation.json")
    assert static["passed"], "Static publication checks did not pass"
    trace = repository/"paper/review-traces/experiment-result-to-claim/2026-10-07_run02"
    verdict = read(trace/"verdict.json")
    assert verdict["claim_supported"] == "no" and verdict["blocking_issues"] == []
    for name in ("prompt.md", "response.md", "inputs.json", "handoff.md"):
        assert (trace/name).is_file(), name
    review_completion = read(source/"review_completion.json")
    assert review_completion["review_verdict_sha256"] == identity(trace/"verdict.json")["sha256"]
    before = read(trace/"reviewed_inputs/report.json")
    unchanged = dict(report)
    unchanged["research_delivery_complete"] = False
    assert unchanged == before, "Scientific payload changed after independent review"
    frozen = read(source/"source_identity.json")
    external = {}

    def add(path, expected_sha=None):
        path = path.resolve()
        record = identity(path)
        assert expected_sha is None or record["sha256"] == expected_sha, str(path)
        if not path.is_relative_to(source):
            external[str(path)] = record

    for name, digest in frozen["source_sha256"].items():
        add(repository/name, digest)
    for name, digest in frozen["protected_artifacts"].items():
        add(Path(name), digest)
    preflight_attempt = Path(read(source/"preflight_geometry_binding_check.json")["source_attempt"])
    for name in ("task.json", "actual/run_metadata.json", "actual/evaluation/report.json",
                 "actual/evaluation/fresh_replay.npz",
                 "actual/timing/b2_continuum_route_test_000.jsonl"):
        add(preflight_attempt/name)
    # Review/source identities are exact bytes; no mutable HEAD is substituted.
    for root in (trace, visualization):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                add(path)
    for name in ("README.md", ".gitattributes", "findings.md", "MANIFEST.md",
                 "paper/EXPERIMENT_PLAN.md", "paper/EXPERIMENT_TRACKER.md",
                 "paper/EXPERIMENT_TRACKER.20261007_pre_results.md",
                 "docs/V6_4_B3_VISUALIZATION.md", "docs/V6_2_LATEST_VISUALIZATION.md",
                 "v6_lite/README.md", "v6_lite/visualization/index.html",
                 "v6_4/evaluate_conditional_route_value.py", "v6_4/export_conditional_release.py",
                 "v6_4/verify_conditional_delivery.py", "v6_4/seal_conditional_delivery.py",
                 "v6_4/visualization/build_conditional_dashboard.py",
                 "v6_4/visualization/build_conditional_figures.py",
                 "v6_4/visualization/audit_conditional_publication.py"):
        add(repository/name)
    payload = {path.relative_to(source).as_posix(): identity(path)
               for path in sorted(source.rglob("*")) if path.is_file()}
    assert not set(payload) & CONTROLS
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository).decode().strip()
    manifest = {"schema": "v64_b3_final_local_delivery_manifest_v1", "status": "FINAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_producer_commit": frozen["algorithm_producer_commit"],
        "algorithm_producer_commit": frozen["algorithm_producer_commit"],
        "published_B2_base": frozen["published_B2_base"],
        "development_git_head": head, "delivery_git_head": head,
        "payload": payload, "external": external,
        "excluded_self_referential_controls": sorted(CONTROLS),
        "publication_status": "local sealed delivery; final GitHub commit/remote proof recorded separately",
        "additional_physics_steps": 0, "additional_geometry_queries": 0,
        "additional_DDIM_calls": 0, "additional_optimizer_updates": 0}
    write(source/"manifest.json", manifest)
    mismatches = []
    for name, expected in payload.items():
        if identity(source/name) != expected:
            mismatches.append(name)
    for name, expected in external.items():
        if identity(Path(name)) != expected:
            mismatches.append(name)
    assert not mismatches, mismatches
    verification = {"schema": "v64_b3_final_local_delivery_verification_v1", "status": "PASS",
        "manifest_sha256": identity(source/"manifest.json")["sha256"],
        "verified_payload_files": len(payload), "verified_external_files": len(external),
        "mismatches": [], "frozen_source_guard_passed": True,
        "fixed_six_slots_and_negative_stop_verified": True,
        "scientific_payload_unchanged_after_review": True,
        "independent_review_blocking_issues": [],
        "full_physics_or_learning_rerun_for_delivery": False,
        "browser_render_check": "NOT_VERIFIED_TOOL_POLICY",
        "additional_physics_steps": 0, "additional_geometry_queries": 0,
        "additional_DDIM_calls": 0, "additional_optimizer_updates": 0}
    write(source/"verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
