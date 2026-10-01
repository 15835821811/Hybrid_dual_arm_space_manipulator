"""Verify immutable pressure-trial raw bytes on disk and in Git's index."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DIRECTORIES = {
    "research_velocity_stress_01": "manifest.json",
    "research_velocity_stress_independent_audit_01": "artifact_manifest.json",
    "research_velocity_failure_feasibility_01": "manifest.json",
    "research_velocity_failure_feasibility_02": "manifest.json",
    "research_velocity_failure_scalar_independent_01": "manifest.json",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    checks, inventory = [], []
    for directory, manifest_name in DIRECTORIES.items():
        base = ROOT / "v6_lite/output/runs" / directory
        manifest = json.loads((base / manifest_name).read_text(encoding="utf-8"))
        actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
        checks.append({"name": f"{directory}:complete_manifest_file_set", "passed": actual == set(manifest) | {manifest_name}})
        for name in sorted(actual):
            p = base / name
            raw = p.read_bytes()
            record = manifest.get(name)
            expected_sha = record if isinstance(record, str) else record.get("sha256") if record else sha(raw)
            expected_size = record.get("bytes", record.get("size_bytes")) if isinstance(record, dict) else None
            checks.append({"name": f"{directory}:{name}:disk_sha", "passed": sha(raw) == expected_sha})
            if expected_size is not None:
                checks.append({"name": f"{directory}:{name}:disk_bytes", "passed": len(raw) == expected_size})
            relative = p.relative_to(ROOT).as_posix()
            staged = subprocess.check_output(["git", "show", ":" + relative], cwd=ROOT)
            checks.append({"name": f"{directory}:{name}:index_raw_bytes", "passed": staged == raw})
            inventory.append({"path": relative, "sha256": sha(raw), "bytes": len(raw), "staged_git_sha256": sha(staged)})
    baseline_status = json.loads(subprocess.check_output(["git", "show", "9c7120e8ca8711cd58dc65471e58763c965148ff:v6_lite/controller_status.json"], cwd=ROOT).decode("utf-8"))
    current_status = json.loads((ROOT / "v6_lite/controller_status.json").read_text(encoding="utf-8"))
    checks.append({"name": "all_53_historical_status_values_preserved", "passed": len(baseline_status) == 53 and all(current_status.get(k) == v for k, v in baseline_status.items())})
    report = {
        "schema": "research_velocity_stress_disk_and_staged_git_verification_v1",
        "evidence_valid": all(x["passed"] for x in checks),
        "trial_acceptance_passed": False,
        "trial_source_commit": "9c7120e8ca8711cd58dc65471e58763c965148ff",
        "archive_parent_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "raw_artifact_count": len(inventory),
        "raw_artifact_bytes": sum(x["bytes"] for x in inventory),
        "check_count": len(checks),
        "failed_checks": [x for x in checks if not x["passed"]],
        "checks": checks,
        "raw_artifacts": inventory,
        "scope": "Raw producer/auditor bytes including manifests and failed diagnostic 01; excludes this archive verifier and its own report. Manifest self-hashes are recorded here rather than claimed as producer self-verification. No physics rerun or new functional acceptance.",
    }
    (OUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("evidence_valid", "raw_artifact_count", "raw_artifact_bytes", "check_count", "failed_checks")}))
    return 0 if report["evidence_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
