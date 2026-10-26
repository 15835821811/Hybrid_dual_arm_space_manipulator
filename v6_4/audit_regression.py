"""Compare all non-timing trace arrays with frozen protocol-A evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(reference: Path, current: Path, output: Path):
    rows = []
    for scene in range(5):
        name = f"v6_lite_scenario_{scene:02d}.npz"
        before, after = Path(reference) / "traces" / name, Path(current) / "traces" / name
        with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
            names = [k for k in a.files if not any(x in k for x in ("latency", "wall_time"))]
            mismatches = []
            for key in names:
                if key not in b:
                    mismatches.append(key)
                    continue
                equal = (np.array_equal(a[key], b[key], equal_nan=True) if a[key].dtype.kind in "fc"
                         else np.array_equal(a[key], b[key]))
                if not equal:
                    mismatches.append(key)
            unexpected = [k for k in b.files if k not in a and not any(x in k for x in ("latency", "wall_time"))]
            rows.append({"scenario_id": name[:-4], "non_timing_arrays": len(names),
                         "mismatches": mismatches, "unexpected_non_timing_arrays": unexpected,
                         "reference_trace_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
                         "current_trace_sha256": hashlib.sha256(after.read_bytes()).hexdigest()})
    report = {"schema": "v6_4_protocol_a_non_timing_parity_v1",
              "passed": all(not x["mismatches"] and not x["unexpected_non_timing_arrays"] for x in rows),
              "reference": str(Path(reference).resolve()), "current": str(Path(current).resolve()),
              "exclusions": "names containing latency or wall_time only", "rows": rows}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(a.reference, a.current, a.output)
    print(json.dumps({"passed": report["passed"], "arrays_per_scene": [r["non_timing_arrays"] for r in report["rows"]]}))
    if not report["passed"]:
        raise SystemExit(1)
