"""Immutable run-directory allocation and start-of-run provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "v6_lite/hierarchical_qp.py",
    "v6_lite/safety_contract.py",
    "v6_lite/execution_ramp.py",
    "v6_lite/run_v6_lite.py",
    "v6_lite/run_evidence.py",
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True,
            text=True, encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): safe(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(part) for part in item]
        if isinstance(item, np.ndarray):
            return safe(item.tolist())
        if isinstance(item, (np.floating, float)):
            scalar = float(item)
            return scalar if math.isfinite(scalar) else None
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, (np.integer, int)):
            return int(item)
        return item

    encoded = json.dumps(safe(value), ensure_ascii=False, indent=2,
                         allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    os.replace(temporary, path)


def start_run(
    output_dir: Path,
    *,
    run_config: Any,
    qp_config: Any,
    spec: Any,
    scenarios: Sequence[Any],
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """Claim a new directory atomically before any physics or output write."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    scenario_definitions = [item.to_dict() for item in scenarios]
    source_hashes = {name: _source_hash(ROOT / name) for name in SOURCE_FILES}
    commit = _git("rev-parse", "HEAD")
    tracked_changes = _git("status", "--porcelain", "--untracked-files=no")
    host_label = hashlib.sha256(platform.node().encode("utf-8")).hexdigest()
    metadata: dict[str, Any] = {
        "schema": "v6_2_b1_immutable_run_v1",
        "run_id": output_dir.name,
        "parent_run_id": parent_run_id,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir.as_posix(),
        "source": {
            "git_commit": commit,
            "tracked_worktree_dirty": bool(tracked_changes),
            "source_normalized_lf_sha256": source_hashes,
        },
        "model_identity": spec.identity().to_dict(),
        "run_config": asdict(run_config),
        "qp_config": asdict(qp_config),
        "configuration_sha256": _canonical_hash({
            "run": asdict(run_config), "qp": asdict(qp_config),
        }),
        "scenario_definitions_sha256": _canonical_hash(scenario_definitions),
        "scenarios": scenario_definitions,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mujoco": mujoco.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "host_label_sha256": host_label,
        },
        "timing_protocol": {
            "clock": "time.perf_counter",
            "load_condition": "uncontrolled_local_desktop",
            "replay_can_reproduce_wall_clock": False,
        },
    }
    _write_json(output_dir / "run_metadata.json", metadata, exclusive=True)
    return metadata


def finish_run(output_dir: Path, metadata: dict[str, Any], *, passed: bool,
               summary: dict[str, Any] | None = None) -> None:
    if not passed:
        _write_json(Path(output_dir) / "run_failure.json", {
            "run_id": metadata["run_id"],
            "status": "FAILED",
            "reason": "acceptance_gate_failed",
            "summary": summary or {},
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }, exclusive=True)
    metadata["status"] = "PASSED" if passed else "FAILED"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["summary"] = summary or {}
    _write_json(Path(output_dir) / "run_metadata.json", metadata)


def fail_run(output_dir: Path, metadata: dict[str, Any], error: BaseException,
             *, scenario_id: str | None = None) -> None:
    status = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
    record = {
        "run_id": metadata["run_id"],
        "status": status,
        "scenario_id": scenario_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(Path(output_dir) / "run_failure.json", record, exclusive=True)
    metadata["status"] = status
    metadata["finished_utc"] = record["time_utc"]
    metadata["failure"] = {key: record[key] for key in (
        "scenario_id", "error_type", "message",
    )}
    _write_json(Path(output_dir) / "run_metadata.json", metadata)
