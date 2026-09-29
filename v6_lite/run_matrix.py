"""Predeclare and retain every baseline/enabled timing trial."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_evidence import new_run_id
from v6_lite.run_v6_lite import V6LiteRunConfig, run_suite


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_matrix(output_root: Path, *, matrix_id: str | None = None,
               repeats: int = 3, seed: int = 20260801,
               duration_s: float = 27.0) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeat count must be positive")
    run_config = V6LiteRunConfig(seed=seed, duration_s=duration_s)
    run_config.validate()
    configs = {
        "baseline": HierarchicalQPConfig(),
        "enabled": HierarchicalQPConfig(enable_pcc_cbf=True, enable_capsule_cbf=True),
    }
    name = matrix_id or f"matrix-{new_run_id()}"
    matrix_dir = Path(output_root) / name
    matrix_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "v6_2_b1_predeclared_timing_matrix_v1",
        "matrix_id": name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repeats_per_mode": repeats,
        "run_order": [
            {"mode": mode, "repeat": repeat}
            for repeat in range(1, repeats + 1)
            for mode in ("baseline", "enabled")
        ],
        "load_condition": (
            "one benchmark run at a time on the local desktop; "
            "background system load is uncontrolled and recorded wall-clock samples are retained"
        ),
        "run_config": asdict(run_config),
        "qp_configs": {name: asdict(cfg) for name, cfg in configs.items()},
        "selection_policy": "all predeclared trials are reported, including failures and aborts",
    }
    _write(matrix_dir / "plan.json", plan)
    result: dict[str, Any] = {
        "schema": "v6_2_b1_timing_matrix_result_v1",
        "matrix_id": name,
        "plan": (matrix_dir / "plan.json").as_posix(),
        "trials": [],
        "complete": False,
    }
    _write(matrix_dir / "matrix_result.json", result)
    for item in plan["run_order"]:
        mode, repeat = item["mode"], item["repeat"]
        trial_id = f"{mode}-r{repeat:02d}-{new_run_id()}"
        trial_dir = matrix_dir / trial_id
        print(f"[matrix] {mode} repeat {repeat}/{repeats}: {trial_id}", flush=True)
        try:
            payload = run_suite(run_config, configs[mode], trial_dir)
            trial = {
                "mode": mode, "repeat": repeat, "run_id": trial_id,
                "status": "PASSED" if payload["passed"] else "FAILED",
                "passed": bool(payload["passed"]),
                "task_full_latency_p95_ms_max": payload["aggregate_metrics"][
                    "task_full_latency_p95_ms_max"
                ],
                "metrics": (trial_dir / "v6_lite_metrics.json").as_posix(),
            }
        except KeyboardInterrupt:
            result["trials"].append({
                "mode": mode, "repeat": repeat, "run_id": trial_id,
                "status": "INTERRUPTED", "passed": False,
            })
            _write(matrix_dir / "matrix_result.json", result)
            raise
        except Exception as error:
            trial = {
                "mode": mode, "repeat": repeat, "run_id": trial_id,
                "status": "FAILED", "passed": False,
                "error_type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            }
        result["trials"].append(trial)
        _write(matrix_dir / "matrix_result.json", result)
    result["complete"] = True
    result["all_trials_passed"] = all(item["passed"] for item in result["trials"])
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write(matrix_dir / "matrix_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("v6_lite/output/runs"))
    parser.add_argument("--matrix-id")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--duration", type=float, default=27.0)
    args = parser.parse_args()
    result = run_matrix(args.output_root, matrix_id=args.matrix_id,
                        repeats=args.repeats, seed=args.seed,
                        duration_s=args.duration)
    print(json.dumps({"matrix_id": result["matrix_id"],
                      "complete": result["complete"],
                      "all_trials_passed": result["all_trials_passed"],
                      "trials": result["trials"]}, ensure_ascii=False, indent=2))
    if not result["all_trials_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
