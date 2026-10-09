"""Read-only MuJoCo pair screening trial on native A.1 torque replay states.

Use model.geom_rbound only to identify pairs whose bounding spheres are beyond
the unchanged 0.09 m distance query. The trial still calls the original
mj_geomDistance for every retained pair and verifies active rows, witnesses,
and global/target minima against an all-pair reference. Nothing selects this
path in the controller. This is an empirical implementation parity audit,
not a new safety proof or complete control-cycle timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig,
)
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.recompute_execution_constraints import _obstacles
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec,
)


TICKS = (50, 100, 150, 250, 350, 500, 1000)
SPHERE_SCREEN_PAD_M = 1e-6


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        raise ValueError("empty sphere-screen sample")
    return {"count": len(data), "min": float(np.min(data)),
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data))}


def _distances(model: mujoco.MjModel, data: mujoco.MjData,
               a: np.ndarray, b: np.ndarray,
               indexes: np.ndarray, query_max_m: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(len(indexes), dtype=np.float64)
    witnesses = np.empty((len(indexes), 6), dtype=np.float64)
    fromto = np.zeros(6, dtype=np.float64)
    for offset, index in enumerate(indexes):
        values[offset] = mujoco.mj_geomDistance(
            model, data, int(a[index]), int(b[index]), query_max_m, fromto,
        )
        witnesses[offset] = fromto
    return values, witnesses


def _screened(model: mujoco.MjModel, data: mujoco.MjData,
              a: np.ndarray, b: np.ndarray,
              query_max_m: float) -> dict:
    started = time.perf_counter()
    radii_a = np.asarray(model.geom_rbound[a], dtype=np.float64)
    radii_b = np.asarray(model.geom_rbound[b], dtype=np.float64)
    centers_a = np.asarray(data.geom_xpos[a], dtype=np.float64)
    centers_b = np.asarray(data.geom_xpos[b], dtype=np.float64)
    finite = (np.isfinite(radii_a) & np.isfinite(radii_b)
              & (radii_a > 0) & (radii_b > 0)
              & np.all(np.isfinite(centers_a), axis=1)
              & np.all(np.isfinite(centers_b), axis=1))
    lower = np.linalg.norm(centers_a - centers_b, axis=1) - radii_a - radii_b
    far = finite & (lower > query_max_m + SPHERE_SCREEN_PAD_M)
    indexes = np.flatnonzero(~far)
    kept_distances, kept_witnesses = _distances(
        model, data, a, b, indexes, query_max_m,
    )
    fallback = bool(not len(kept_distances)
                    or np.min(kept_distances) >= query_max_m)
    if fallback:
        # MuJoCo may report either query_max_m or a larger sentinel for a far
        # pair. Preserve its exact global minimum when the retained set has
        # not witnessed anything strictly inside the query radius.
        omitted = np.flatnonzero(far)
        omitted_distances, omitted_witnesses = _distances(
            model, data, a, b, omitted, query_max_m,
        )
    else:
        omitted = np.empty(0, dtype=np.int32)
        omitted_distances = np.empty(0, dtype=np.float64)
        omitted_witnesses = np.empty((0, 6), dtype=np.float64)
    distances = np.full(len(a), query_max_m, dtype=np.float64)
    witnesses = np.zeros((len(a), 6), dtype=np.float64)
    distances[indexes], witnesses[indexes] = kept_distances, kept_witnesses
    if fallback:
        distances[omitted], witnesses[omitted] = omitted_distances, omitted_witnesses
    return {
        "distances": distances, "witnesses": witnesses, "far_mask": far,
        "fallback": fallback,
        "exact_call_count": len(indexes) + len(omitted),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def _at_state(model: mujoco.MjModel, data: mujoco.MjData, pairs: tuple,
              config: HierarchicalQPConfig) -> dict:
    a = np.asarray([pair.geom_a for pair in pairs], dtype=np.int32)
    b = np.asarray([pair.geom_b for pair in pairs], dtype=np.int32)
    all_indexes = np.arange(len(pairs), dtype=np.int32)
    is_target = np.asarray([pair.pair_class == "continuum_target"
                            for pair in pairs], dtype=bool)
    reference_times = []
    screened_times = []
    comparisons = []
    for label in ("reference", "screened", "screened", "reference"):
        if label == "reference":
            started = time.perf_counter()
            distances, witnesses = _distances(
                model, data, a, b, all_indexes,
                config.clearance_query_max_m,
            )
            reference_times.append((time.perf_counter() - started) * 1000.0)
            comparisons.append((label, distances, witnesses, None))
        else:
            result = _screened(model, data, a, b,
                               config.clearance_query_max_m)
            screened_times.append(result["elapsed_ms"])
            comparisons.append((label, result["distances"],
                                result["witnesses"], result))
    ref = comparisons[0]
    original_active = ref[1] <= config.clearance_activation_m
    original_target_min = float(np.min(np.minimum(
        ref[1][is_target], config.clearance_query_max_m)))
    max_active_distance_error = 0.0
    max_active_witness_error = 0.0
    maximum_minimum_error = 0.0
    missed_active = 0
    screened_calls = []
    skipped = []
    fallback = []
    for label, distances, witnesses, result in comparisons[1:]:
        if label == "reference":
            if (not np.array_equal(distances, ref[1])
                    or not np.array_equal(witnesses[original_active],
                                          ref[2][original_active])):
                raise ValueError("same-state all-pair reference changed across ABBA order")
            continue
        candidate_active = distances <= config.clearance_activation_m
        missed_active += int(np.count_nonzero(original_active & ~candidate_active))
        if not np.array_equal(candidate_active, original_active):
            raise ValueError("sphere screen changed activated MuJoCo pair IDs")
        max_active_distance_error = max(max_active_distance_error, float(
            np.max(np.abs(distances[original_active] - ref[1][original_active]))
            if np.any(original_active) else 0.0))
        max_active_witness_error = max(max_active_witness_error, float(
            np.max(np.abs(witnesses[original_active]
                              - ref[2][original_active]))
            if np.any(original_active) else 0.0))
        minimum_error = abs(float(np.min(distances) - np.min(ref[1])))
        target_error = abs(float(np.min(np.minimum(
            distances[is_target], config.clearance_query_max_m))
            - original_target_min))
        maximum_minimum_error = max(maximum_minimum_error,
                                    minimum_error, target_error)
        if not np.isfinite(maximum_minimum_error):
            raise ValueError("sphere screen changed a global or target minimum")
        screened_calls.append(result["exact_call_count"])
        skipped.append(int(np.count_nonzero(result["far_mask"])))
        fallback.append(result["fallback"])
    if (max_active_distance_error > 0.0
            or max_active_witness_error > 0.0
            or maximum_minimum_error > 0.0 or missed_active):
        raise ValueError("sphere screen did not preserve exact MuJoCo row evidence")
    return {
        "pair_count": len(pairs),
        "active_pair_count": int(np.count_nonzero(original_active)),
        "sphere_far_count": skipped[0],
        "exact_call_count": screened_calls[0],
        "fallback_to_all_pairs": fallback[0],
        "reference_pair_time_ms": float(np.median(reference_times)),
        "screened_pair_time_ms": float(np.median(screened_times)),
        "maximum_active_distance_error_m": max_active_distance_error,
        "maximum_active_witness_error_m": max_active_witness_error,
        "maximum_minimum_error_m": maximum_minimum_error,
        "missed_active_count": missed_active,
        "far_reference_below_activation_count": int(np.count_nonzero(
            comparisons[1][3]["far_mask"] & original_active)),
        "far_reference_uncapped_sentinel_count": int(np.count_nonzero(
            comparisons[1][3]["far_mask"]
            & (ref[1] > config.clearance_query_max_m))),
    }


def run(output_dir: Path, a1_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = default_v6_lite_robot_spec()
    records = []
    input_records = []
    failures = []
    for mode in ("baseline", "enabled"):
        root = a1_root / f"{mode}_root" / "output"
        metrics_path = root / "v6_lite_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = HierarchicalQPConfig(**metrics["qp_config"])
        run_cfg = V6LiteRunConfig(**metrics["run_config"])
        scenarios = {item.scenario_id: item for item in build_scenarios(robot, run_cfg)}
        if (cfg.clearance_query_max_m != .09 or cfg.clearance_activation_m != .08):
            raise ValueError("frozen original MuJoCo distance thresholds changed")
        for item in metrics["scenarios"]:
            scenario_id = item["scenario"]["scenario_id"]
            if scenarios[scenario_id].seed != item["scenario"]["seed"]:
                raise ValueError("frozen A.1 scenario seed changed")
            verifier = WholeBodyCollisionVerifier(
                robot, _obstacles(item["scenario"]), WholeBodyVerificationConfig(
                    minimum_clearance=run_cfg.whole_body_minimum_clearance_m,
                    query_distance_max=2.5,
                    adaptive_subdivisions=run_cfg.verification_subdivisions,
                    self_collision_ancestor_exclusion_depth=3,
                    include_target_satellite_pairs=True,
                ),
            )
            model = verifier.model
            model.geom_contype[:] = 0
            model.geom_conaffinity[:] = 0
            data = mujoco.MjData(model)
            trace_path = Path(item["trace"]["path"])
            trace_hash = _sha(trace_path)
            if trace_hash != item["trace"]["sha256"]:
                raise ValueError("A.1 torque trace hash changed")
            input_records.append({"mode": mode, "scenario_id": scenario_id,
                                  "metrics_sha256": _sha(metrics_path),
                                  "trace_path": trace_path.as_posix(),
                                  "trace_sha256": trace_hash})
            with np.load(trace_path, allow_pickle=False) as trace:
                data.qpos[:] = trace["initial_qpos"]
                data.qvel[:] = trace["initial_qvel"]
                torque = trace["torque"][:TICKS[-1] * 10 + 1].copy()
                saved_qpos = trace["task_qpos"].copy()
                saved_time = trace["task_time"].copy()
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            max_replay_error = 0.0
            for step, control in enumerate(torque):
                if step % 10 == 0:
                    tick = step // 10
                    mujoco.mj_forward(model, data)
                    max_replay_error = max(
                        max_replay_error,
                        float(np.max(np.abs(data.qpos - saved_qpos[tick]))),
                        abs(float(data.time - saved_time[tick])),
                    )
                    if tick in TICKS:
                        try:
                            record = _at_state(model, data, verifier.pairs, cfg)
                            record.update({"mode": mode, "scenario_id": scenario_id,
                                           "tick": tick,
                                           "native_replay_error": max_replay_error})
                            records.append(record)
                        except Exception as exc:
                            failures.append({
                                "mode": mode, "scenario_id": scenario_id,
                                "tick": tick,
                                "exception": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            })
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
            if max_replay_error > 1e-8:
                failures.append({"mode": mode, "scenario_id": scenario_id,
                                 "exception": "native A.1 torque replay diverged",
                                 "maximum_replay_error": max_replay_error})
            print(f"[b2-sphere-screen] {mode} {scenario_id}: "
                  f"{len(TICKS)} audited ticks, replay_error={max_replay_error:.2e}",
                  flush=True)
    records_path = output_dir / "sphere_screen_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    failures_path = output_dir / "sphere_screen_failures.jsonl"
    with failures_path.open("x", encoding="utf-8", newline="\n") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    summary = {
        "schema": "v6_2_b2_read_only_mujoco_sphere_screen_v1",
        "scope": "ABBA_pair_query_on_70_native_torque_replayed_old_A1_states",
        "mujoco_version": mujoco.__version__,
        "probe_ticks": list(TICKS),
        "sphere_screen_pad_m": SPHERE_SCREEN_PAD_M,
        "query_max_m": .09, "activation_m": .08,
        "new_interval_mode_executed": False,
        "online_controller_changed": False,
        "full_control_cycle_measured": False,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_mujoco_sphere_screen.py", "hierarchical_qp.py",
            "recompute_execution_constraints.py",
        )},
        "inputs": input_records,
        "record_count": len(records), "failure_count": len(failures),
        "records_sha256": _sha(records_path),
        "failures_sha256": _sha(failures_path),
        "modes": {},
    }
    for mode in ("baseline", "enabled"):
        own = [record for record in records if record["mode"] == mode]
        if not own:
            continue
        summary["modes"][mode] = {
            "state_count": len(own),
            "pair_count_per_state": 2927,
            "sphere_far_count": _stats([record["sphere_far_count"] for record in own]),
            "exact_call_count": _stats([record["exact_call_count"] for record in own]),
            "reference_pair_time_ms": _stats([
                record["reference_pair_time_ms"] for record in own]),
            "screened_pair_time_ms": _stats([
                record["screened_pair_time_ms"] for record in own]),
            "max_active_distance_error_m": max(
                record["maximum_active_distance_error_m"] for record in own),
            "max_active_witness_error_m": max(
                record["maximum_active_witness_error_m"] for record in own),
            "max_minimum_error_m": max(
                record["maximum_minimum_error_m"] for record in own),
            "missed_active_count": sum(record["missed_active_count"] for record in own),
            "fallback_count": sum(record["fallback_to_all_pairs"] for record in own),
            "far_uncapped_sentinel_count": sum(
                record["far_reference_uncapped_sentinel_count"] for record in own),
        }
    summary_path = output_dir / "sphere_screen_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 原 MuJoCo 碰撞对的包围球筛选只读试验", "",
        "从旧 A.1 两组五场景原生 500 Hz 力矩重放到 tick "
        "50/100/150/250/350/500/1000，"
        "固定原 0.09 m 查询上限和 0.08 m 激活距离。每状态按全对→筛选→筛选→全对"
        "测量；只对包围球下界严格大于查询上限加 1 μm 的碰撞对跳过"
        "精确调用。若保留对没有任何距离严格小于查询上限，则回退完整查询。", "",
        "| 模式 | 状态 | 每状态原碰撞对 | 筛选后精确调用 p95 | 原全对 p95 ms | 筛选 p95 ms | 激活行/见证点/最小值差异 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = summary["modes"].get(mode)
        if item is None:
            continue
        lines.append(
            f"| {mode} | {item['state_count']} | {item['pair_count_per_state']} | "
            f"{item['exact_call_count']['p95']:.1f} | "
            f"{item['reference_pair_time_ms']['p95']:.3f} | "
            f"{item['screened_pair_time_ms']['p95']:.3f} | "
            f"{item['missed_active_count']} / "
            f"{item['max_active_witness_error_m']:.1e} / "
            f"{item['max_minimum_error_m']:.1e} |"
        )
    lines += [
        "", f"完整记录 {len(records)}/70；失败快照 {len(failures)}。"
        f"MuJoCo {mujoco.__version__} 的某些远距离查询返回极大哨兵值，而不是 0.09 m；"
        "筛选方案因此只要求激活行、见证点、全局及目标最小值一致，"
        "并保留缺少近距见证时的全对回退。逐状态计数和差异保留于 JSONL。",
        "只读结果说明计算复用的潜力；没有修改原碰撞约束、在线控制、"
        "PCC 半径或安全门槛。包围球筛选耗时仅覆盖 MuJoCo 对查询，"
        "不包含梯度、QP、监控或 67 路力矩伺服；不能用于 20 ms 全链准入。", "",
    ]
    document_path = output_dir / "MUJOCO_SPHERE_SCREEN.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_sphere_screen_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "records_sha256": _sha(records_path),
        "failures_sha256": _sha(failures_path),
        "document_sha256": _sha(document_path),
    }
    with (output_dir / "sphere_screen_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    args = parser.parse_args()
    result = run(args.output_dir, args.a1_root)
    print(json.dumps({"record_count": result["record_count"],
                      "failure_count": result["failure_count"],
                      "modes": result["modes"]}, indent=2))
    if result["record_count"] != 70 or result["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
