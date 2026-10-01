"""Require exact equality for every non-timing array of all five scenes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def run(reference_dir, candidate_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for scene in range(5):
        name = f"v6_lite_scenario_{scene:02d}.npz"
        old_path, new_path = [Path(p) / "traces" / name for p in (reference_dir, candidate_dir)]
        with np.load(old_path) as old, np.load(new_path) as new:
            mismatches = []
            keys = sorted(set(old.files) | set(new.files))
            checked = []
            for key in keys:
                if "latency" in key or "wall_time" in key:
                    continue
                checked.append(key)
                if key not in old or key not in new:
                    mismatches.append(key)
                    continue
                a, b = old[key], new[key]
                same = (a.shape == b.shape and a.dtype == b.dtype and
                        np.array_equal(a, b, equal_nan=True)
                        if np.issubdtype(a.dtype, np.number) else
                        a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
                if not same:
                    mismatches.append(key)
            records.append({"scenario": name, "checked_array_count": len(checked),
                            "mismatches": mismatches, "exact_equal": not mismatches,
                            "reference_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                            "candidate_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()})
    report = {"schema": "c1_exact_nontiming_parity_v1", "records": records,
              "all_passed": all(r["exact_equal"] for r in records),
              "reference_dir": str(reference_dir), "candidate_dir": str(candidate_dir)}
    (output_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = run(args.reference_dir, args.candidate_dir, args.output_dir)
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)
