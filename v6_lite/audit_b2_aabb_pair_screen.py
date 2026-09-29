"""Read-only AABB screen audit on every scene-01 private task state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.audit_b2_mujoco_sphere_screen import SPHERE_SCREEN_PAD_M
from v6_lite.b2_aabb_pair_screen import ConservativePairAABBScreen
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(np.max(array))}


def run(output_dir: Path, private_dir: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = a1_root / "enabled_root/output/v6_lite_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = V6LiteRunConfig(**metrics["run_config"])
    qp_cfg = HierarchicalQPConfig(**metrics["qp_config"])
    saved = next(item for item in metrics["scenarios"]
                 if item["scenario"]["scenario_id"] == "v6_lite_scenario_01")
    robot = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        robot, _obstacles(saved["scenario"]), WholeBodyVerificationConfig(
            minimum_clearance=cfg.whole_body_minimum_clearance_m,
            query_distance_max=2.5,
            adaptive_subdivisions=cfg.verification_subdivisions,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    pairs = verifier.pairs
    a = np.asarray([item.geom_a for item in pairs], dtype=np.int32)
    b = np.asarray([item.geom_b for item in pairs], dtype=np.int32)
    radius_a = np.asarray(model.geom_rbound[a], dtype=np.float64)
    radius_b = np.asarray(model.geom_rbound[b], dtype=np.float64)
    aabb = ConservativePairAABBScreen(model, a, b)
    activation = qp_cfg.clearance_activation_m
    query_max = qp_cfg.clearance_query_max_m
    threshold = activation + SPHERE_SCREEN_PAD_M
    trace_path = private_dir / "private_rollout_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos_states = trace["qpos_states"][::10].copy()
        qvel_states = trace["task_qvel_states"].copy()
    if len(qpos_states) != 1351 or len(qvel_states) != 1351:
        raise ValueError("private trace incomplete")
    data = mujoco.MjData(model)
    all_indexes = np.arange(len(pairs), dtype=np.int32)
    fromto = np.zeros(6, dtype=np.float64)

    def sphere_far() -> np.ndarray:
        center_a = np.asarray(data.geom_xpos[a], dtype=np.float64)
        center_b = np.asarray(data.geom_xpos[b], dtype=np.float64)
        finite = (np.isfinite(radius_a) & np.isfinite(radius_b)
                  & (radius_a > 0.0) & (radius_b > 0.0)
                  & np.all(np.isfinite(center_a), axis=1)
                  & np.all(np.isfinite(center_b), axis=1))
        lower = np.linalg.norm(center_a - center_b, axis=1)
        lower -= radius_a
        lower -= radius_b
        return finite & (lower > threshold)

    def exact(indexes: np.ndarray) -> np.ndarray:
        distances = np.empty(len(indexes), dtype=np.float64)
        for offset, index in enumerate(indexes):
            distances[offset] = mujoco.mj_geomDistance(
                model, data, int(a[index]), int(b[index]), query_max, fromto)
        return distances

    rows = []
    for tick, (qpos, qvel) in enumerate(zip(qpos_states, qvel_states)):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        full_distances = exact(all_indexes)
        active = full_distances <= activation
        measurements = {}
        for label in (("sphere", "aabb") if tick % 2 == 0
                      else ("aabb", "sphere")):
            started = time.perf_counter()
            far = sphere_far()
            if label == "aabb":
                far |= aabb.far_mask(data, threshold)
            retained = np.flatnonzero(~far)
            screened = exact(retained)
            measurements[label] = {
                "far": far,
                "retained_count": len(retained),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "active_count": int(np.count_nonzero(screened <= activation)),
            }
        if (np.any(active & measurements["sphere"]["far"])
                or np.any(active & measurements["aabb"]["far"])
                or measurements["sphere"]["active_count"] != int(np.sum(active))
                or measurements["aabb"]["active_count"] != int(np.sum(active))):
            raise ValueError(f"screen omitted an original active pair at {tick}")
        rows.append({
            "tick": tick,
            "all_pair_count": len(pairs),
            "original_active_count": int(np.sum(active)),
            "sphere_retained_count": measurements["sphere"]["retained_count"],
            "aabb_retained_count": measurements["aabb"]["retained_count"],
            "sphere_ms": measurements["sphere"]["elapsed_ms"],
            "aabb_ms": measurements["aabb"]["elapsed_ms"],
        })
    rows_path = output_dir / "aabb_pair_screen_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_private_aabb_pair_screen_v1",
        "scenario_id": "v6_lite_scenario_01",
        "state_count": len(rows),
        "all_pair_count": len(pairs),
        "activation_distance_m": activation,
        "query_max_m": query_max,
        "screen_pad_m": SPHERE_SCREEN_PAD_M,
        "missed_original_active_pair_count": 0,
        "sphere_retained": _stats([row["sphere_retained_count"] for row in rows]),
        "aabb_retained": _stats([row["aabb_retained_count"] for row in rows]),
        "sphere_screen_and_exact_ms": _stats([row["sphere_ms"] for row in rows]),
        "aabb_screen_and_exact_ms": _stats([row["aabb_ms"] for row in rows]),
        "full_private_cycle_evaluated": False,
        "production_online_controller_changed": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_aabb_pair_screen.py",
                              "b2_aabb_pair_screen.py",
                              "b2_screened_next_start_rows.py")},
        "inputs_sha256": {"a1_metrics": _sha(metrics_path),
                           "private_trace": _sha(trace_path)},
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "aabb_pair_screen_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    args = parser.parse_args()
    report = run(args.output_dir, args.private_dir, args.a1_root)
    print(json.dumps({key: report[key] for key in (
        "state_count", "missed_original_active_pair_count",
        "sphere_retained", "aabb_retained", "sphere_screen_and_exact_ms",
        "aabb_screen_and_exact_ms")}, indent=2))


if __name__ == "__main__":
    main()
