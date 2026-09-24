"""Consolidate V6.1-B baseline, enabled-control, and monitor evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v6_lite.pcc_monitor import PCCMonitor, PCCMonitorSample

DEFAULT_OUTPUT = Path("v6_lite/output/v6_1_b")


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _aggregate_monitors(output_dir: Path) -> dict[str, Any]:
    monitor = PCCMonitor(scenario_id="v6_1_b_enabled_five_scenario_aggregate")
    time_offset = 0.0
    paths = sorted(
        (output_dir / "enabled_root" / "output" / "pcc_monitor").glob(
            "*/clearance_compare.json"
        )
    )
    for path in paths:
        payload = _read(path)
        rows = payload.get("samples", [])
        for row in rows:
            monitor.samples.append(
                PCCMonitorSample(
                    time=time_offset + float(row["time"]),
                    d_pcc=_float_or_nan(row["d_pcc"]),
                    d_capsule=_float_or_nan(row["d_capsule"]),
                    d_mujoco=_float_or_nan(row["d_mujoco"]),
                    distance_error=_float_or_nan(row["distance_error"]),
                    gradient_error=_float_or_nan(row["gradient_error"]),
                    capsule_gradient_error=_float_or_nan(
                        row["capsule_gradient_error"]
                    ),
                    pcc_constraint_active_count=int(
                        row["pcc_constraint_active_count"]
                    ),
                    capsule_constraint_active_count=int(
                        row["capsule_constraint_active_count"]
                    ),
                    pcc_binding_constraint_count=int(
                        row["pcc_binding_constraint_count"]
                    ),
                    capsule_binding_constraint_count=int(
                        row["capsule_binding_constraint_count"]
                    ),
                    pcc_avoidance_intervention=float(
                        row["pcc_avoidance_intervention"]
                    ),
                    pcc_segment_id=int(row["pcc_segment_id"]),
                    pcc_arc_length_m=_float_or_nan(row["pcc_arc_length_m"]),
                    shape_clearance_latency_s=float(
                        row["shape_clearance_latency_s"]
                    ),
                )
            )
        if rows:
            time_offset += float(rows[-1]["time"]) + 0.02
    if not paths or not monitor.samples:
        raise ValueError("enabled five-scenario PCC monitor outputs are missing")
    payload = monitor.write(output_dir)
    payload["source_scenarios"] = [path.parent.name for path in paths]
    _write(output_dir / "clearance_compare.json", payload)
    return payload


def finalize(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    baseline_metrics = _read(
        output_dir / "baseline_root" / "output" / "v6_lite_metrics.json"
    )
    baseline_validation = _read(
        output_dir / "baseline_root" / "output" / "validation.json"
    )
    enabled_metrics = _read(
        output_dir / "enabled_root" / "output" / "v6_lite_metrics.json"
    )
    enabled_validation = _read(
        output_dir / "enabled_root" / "output" / "validation.json"
    )
    pcc_audit = _read(output_dir / "pcc_audit.json")
    comparison = _aggregate_monitors(output_dir)
    baseline_aggregate = baseline_metrics["aggregate_metrics"]
    enabled_aggregate = enabled_metrics["aggregate_metrics"]
    baseline_checks = (
        f"{baseline_validation['passed_count']}/"
        f"{baseline_validation['total_count']}"
    )
    enabled_checks = (
        f"{enabled_validation['passed_count']}/"
        f"{enabled_validation['total_count']}"
    )
    checks = {
        "formal_pcc_audit_passes": bool(pcc_audit["passed"]),
        "baseline_suite_passes": bool(baseline_metrics["passed"]),
        "baseline_independent_validation_is_26_of_26": bool(
            baseline_validation["passed"] and baseline_checks == "26/26"
        ),
        "baseline_qp_failure_count_is_zero": int(
            baseline_aggregate["total_qp_failures"]
        )
        == 0,
        "enabled_suite_passes": bool(enabled_metrics["passed"]),
        "enabled_independent_validation_is_26_of_26": bool(
            enabled_validation["passed"] and enabled_checks == "26/26"
        ),
        "enabled_qp_failure_count_is_zero": int(
            enabled_aggregate["total_qp_failures"]
        )
        == 0,
        "enabled_task_latency_p95_below_20_ms": float(
            enabled_aggregate["task_full_latency_p95_ms_max"]
        )
        < 20.0,
        "pcc_constraint_activates": int(
            enabled_aggregate["pcc_constraint_active_count"]
        )
        > 0,
        "pcc_changes_the_command": float(
            enabled_aggregate["pcc_avoidance_intervention_max"]
        )
        > 1e-3,
        "enabled_real_mujoco_clearance_above_5_mm": float(
            enabled_validation["recomputed_aggregate"][
                "continuum_target_500hz_minimum_clearance_m"
            ]
        )
        >= 0.005,
        "five_scenario_monitor_is_complete": int(comparison["sample_count"])
        == 5 * 27 * 50,
    }
    payload = {
        "contract_version": "v6.1-b.regression.1",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "baseline": {
            "mode": "enable_pcc_cbf=False, enable_capsule_cbf=False",
            "scenario_count": int(baseline_aggregate["scenario_count"]),
            "validation_checks": baseline_checks,
            "qp_failure_count": int(baseline_aggregate["total_qp_failures"]),
            "task_full_latency_p95_ms": float(
                baseline_aggregate["task_full_latency_p95_ms_max"]
            ),
            "continuum_target_500hz_minimum_clearance_m": float(
                baseline_validation["recomputed_aggregate"][
                    "continuum_target_500hz_minimum_clearance_m"
                ]
            ),
        },
        "enabled_control": {
            "mode": "enable_pcc_cbf=True, enable_capsule_cbf=True",
            "scenario_count": int(enabled_aggregate["scenario_count"]),
            "validation_checks": enabled_checks,
            "qp_failure_count": int(enabled_aggregate["total_qp_failures"]),
            "task_full_latency_p95_ms": float(
                enabled_aggregate["task_full_latency_p95_ms_max"]
            ),
            "shape_clearance_latency_p95_ms": float(
                enabled_aggregate["shape_clearance_latency_p95_ms_max"]
            ),
            "pcc_constraint_active_count": int(
                enabled_aggregate["pcc_constraint_active_count"]
            ),
            "pcc_constraint_binding_count": int(
                enabled_aggregate["pcc_constraint_binding_count"]
            ),
            "capsule_constraint_active_count": int(
                enabled_aggregate["capsule_constraint_active_count"]
            ),
            "capsule_constraint_binding_count": int(
                enabled_aggregate["capsule_constraint_binding_count"]
            ),
            "pcc_avoidance_intervention_max": float(
                enabled_aggregate["pcc_avoidance_intervention_max"]
            ),
            "continuum_target_500hz_minimum_clearance_m": float(
                enabled_validation["recomputed_aggregate"][
                    "continuum_target_500hz_minimum_clearance_m"
                ]
            ),
        },
        "ab_interpretation": (
            "The default-off run preserves the frozen V6-lite acceptance behavior. "
            "The enabled run enters the same single 50 Hz QP and 500 Hz torque "
            "loop, activates PCC constraints, changes the command, and remains "
            "clear under independent MuJoCo replay."
        ),
        "artifacts": {
            "pcc_audit": "pcc_audit.json",
            "clearance_compare": "clearance_compare.json",
            "regression_report": "regression_report.json",
            "plots": [
                "plots/distance_comparison.png",
                "plots/gradient_comparison.png",
                "plots/minimum_clearance_comparison.png",
            ],
        },
        "local_reproduction_sources": {
            "versioned": False,
            "reason": "raw A/B traces are reproducible and total about 275 MB",
            "baseline_metrics": (
                "baseline_root/output/v6_lite_metrics.json"
            ),
            "baseline_validation": "baseline_root/output/validation.json",
            "enabled_metrics": "enabled_root/output/v6_lite_metrics.json",
            "enabled_validation": "enabled_root/output/validation.json",
        },
    }
    report_path = output_dir / "regression_report.json"
    _write(report_path, payload)
    manifest_paths = [
        output_dir / "pcc_audit.json",
        output_dir / "clearance_compare.json",
        report_path,
        output_dir / "plots" / "distance_comparison.png",
        output_dir / "plots" / "gradient_comparison.png",
        output_dir / "plots" / "minimum_clearance_comparison.png",
    ]
    _write(
        output_dir / "v6_1_b_manifest.json",
        {
            "contract_version": payload["contract_version"],
            "passed": payload["passed"],
            "artifacts": [
                {
                    "path": path.as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in manifest_paths
            ],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = finalize(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
