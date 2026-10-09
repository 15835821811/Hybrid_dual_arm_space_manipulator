"""Classify each of the sixteen original C.1 historical-source failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from v6_lite.run_test_profiles import BASE, FROZEN, RAW_EOL, historical_ids


def run(output, historical_report):
    root = Path(__file__).resolve().parent.parent
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads(subprocess.check_output(["git", "show",
        BASE + ":v6_lite/output/v6_2_c1_release/acceptance_summary.json"], cwd=root))
    ids = historical_ids(root)
    proof = old["regression"]["source_hash_baseline_proof"]
    error_sources = ["v6_lite/run_v6_lite.py", "v6_lite/hierarchical_qp.py"]
    records = []
    result = json.loads(historical_report.read_text(encoding="utf-8"))
    for i, test_id in enumerate(ids):
        source = error_sources[i] if i < 2 else proof[i - 2]["source"]
        commit, expected = FROZEN[source]
        raw = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=root)
        actual_frozen = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        if actual_frozen != expected:
            raise ValueError(f"immutable historical object mismatch: {source}")
        current = hashlib.sha256((root / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        prior = hashlib.sha256(subprocess.check_output(["git", "show", f"{BASE}:{source}"], cwd=root).replace(b"\r\n", b"\n")).hexdigest()
        records.append({"test_id": test_id, "checked_object": source,
            "expected_version_commit": commit, "expected_sha256": expected,
            "actual_sha256_at_c1_head": prior, "actual_current_sha256": current,
            "frozen_snapshot_actual_sha256": actual_frozen,
            "failure_category": "frozen_historical_check_applied_to_current_development_source",
            "disposition": "execute unchanged original test and golden against reconstructed fixed Git-source snapshot",
            "reason": "the saved report belongs to the earlier source object; C.1 and B.2 changed current control code",
            "historical_profile_passed": result["historical"]["passed"],
            "additional_snapshot_overrides": FROZEN,
            "historical_raw_eol_goldens": RAW_EOL if "test_b2_shadow" in test_id else {},
        })
    payload = {"schema": "c11_historical_hash_classification_v1", "records": records,
        "original_regression": {"passed": 122, "total": 138, "failures": 14, "errors": 2},
        "historical_report": historical_report.as_posix(),
        "historical_report_sha256": hashlib.sha256(historical_report.read_bytes()).hexdigest(),
        "goldens_modified": False, "current_hash_used_as_expected": False,
        "current_version_physical_evidence_passed": False,
        "scope": "historical source disposition; new wall trajectories still require original independent physical acceptance"}
    (output / "classification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--historical-report", type=Path, required=True)
    args = p.parse_args()
    report = run(args.output_dir, args.historical_report)
    print(len(report["records"]), "original failures classified")
