"""Recreate the V6.1-A 10k target protocol and audit frozen new holdouts.

This is an offline, read-only experiment. Original V6.1-A individual inputs
were not stored, so the historical cohort is regenerated from its published
seed and source algorithm; the source/report artifacts are historical checks,
not a claim that a saved 10k input file was physically replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from v6_lite.audit_v6_1a import (
    DEFAULT_SEED, _random_rotation, _set_target_geom_pose,
    build_frozen_configurations,
)
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    OrientedBox, ShapeClearanceShadow, minimum_mujoco_geom_clearance,
    point_obb_signed_distance, target_box_from_mujoco,
)


HOLDOUT_SEED = 20260929
SAFETY_GATE_M = 0.005


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_holdout(path: Path, count: int) -> str:
    if count < 1:
        raise ValueError("held-out count must be positive")
    rng = np.random.default_rng(HOLDOUT_SEED)
    spec = default_continuum_model_spec()
    shape = ContinuumShapeModel(spec)
    configurations = rng.uniform(spec.work_domain_lower_rad,
                                 spec.work_domain_upper_rad, size=(count, 10))
    centers = np.zeros((count, 3))
    rotations = np.zeros((count, 3, 3))
    half_extents = np.zeros((count, 3))
    categories = np.zeros(count, dtype=np.int64)
    for i, q in enumerate(configurations):
        segment = int(rng.integers(0, 5))
        local_s = float(rng.uniform(0, spec.segment_lengths_m[segment]))
        s = float(spec.segment_boundaries_m[segment] + local_s)
        centerline = shape.evaluate(q, np.eye(4), s,
                                    with_jacobians=False).position_world
        axis = int(rng.integers(0, 3))
        category = i % 4
        categories[i] = category
        offset = np.zeros(3)
        offset[axis] = (0.0, 0.04, 0.10, 0.20)[category]
        offset += rng.uniform(-0.02, 0.02, 3)
        centers[i] = centerline + offset
        rotations[i] = _random_rotation(rng)
        half_extents[i] = [0.085, 0.075, 0.065]
    with path.open("xb") as stream:
        np.savez_compressed(stream, configurations=configurations,
                            centers=centers, rotations=rotations,
                            half_extents=half_extents, categories=categories,
                            seed=np.asarray(HOLDOUT_SEED))
    return _sha(path)


def _historical_cases(count: int):
    """Reproduce the target-case generator in V6.1-A, without shadow calls."""
    robot = default_v6_lite_robot_spec()
    spec = default_continuum_model_spec(robot)
    model = robot.compile_dynamic_model()
    data = mujoco.MjData(model)
    configurations = build_frozen_configurations(spec, 10_000, DEFAULT_SEED)[8000:]
    qpos_ids, _ = joint_addresses(model, robot)
    base_qpos, _ = free_joint_slices(model, robot.base_joint_name)
    target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
    target_geom_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
    ))
    shadow = ShapeClearanceShadow(model, target_geom_id,
                                  V61A_PCC_TUBE_RADII_M, spec=spec,
                                  samples_per_segment=25,
                                  query_distance_max_m=0.5)
    rng = np.random.default_rng(DEFAULT_SEED + 3)
    categories = ("face", "edge", "corner", "rotated", "mid_arm", "penetration")
    mid_arm = [x for x in shadow.envelopes.capsules
               if x.segment_index in (1, 2, 3) and x.body_name.startswith("link_")]
    for case_index in range(count):
        q = configurations[case_index % len(configurations)].copy()
        data.qpos[:] = model.qpos0
        full = robot.planner_zero.copy()
        full[:10] = q
        data.qpos[qpos_ids] = robot.encode_position(full)
        data.qpos[base_qpos] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[target_qpos] = [5, 5, 5, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        category = categories[case_index % len(categories)]
        capsule = (mid_arm[case_index % len(mid_arm)] if category == "mid_arm"
                   else shadow.envelopes.capsules[
                       int(rng.integers(0, len(shadow.envelopes.capsules)))])
        body_rotation = np.asarray(data.xmat[capsule.body_id]).reshape(3, 3)
        body_position = np.asarray(data.xpos[capsule.body_id])
        local_midpoint = 0.5 * (capsule.local_start + capsule.local_end)
        arm_point = body_position + body_rotation @ local_midpoint
        target_rotation = (np.eye(3) if category in
                           ("face", "edge", "corner", "penetration")
                           else _random_rotation(rng))
        half_extents = np.asarray(model.geom_size[target_geom_id], dtype=np.float64)
        if category == "penetration":
            target_center = arm_point.copy()
        else:
            active_count = {"face": 1, "edge": 2, "corner": 3,
                            "rotated": int(rng.integers(1, 4)),
                            "mid_arm": 1}[category]
            axes = rng.choice(3, size=active_count, replace=False)
            gap = float(rng.uniform(-0.02, 0.18))
            offset = np.zeros(3)
            for axis in axes:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                offset[axis] = sign * (
                    half_extents[axis] + (capsule.radius_m + gap)
                    / np.sqrt(active_count)
                )
            target_center = arm_point - target_rotation @ offset
        _set_target_geom_pose(model, data, target_qpos, target_geom_id,
                              target_center, target_rotation)
        mujoco.mj_forward(model, data)
        box = target_box_from_mujoco(model, data, target_geom_id)
        actual = minimum_mujoco_geom_clearance(
            model, data, shadow.enveloped_geom_ids, target_geom_id,
            query_distance_max_m=0.5,
        )
        yield q, box, category, actual.signed_distance_m, actual.query_truncated


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(arr.min()),
            "p50": float(np.quantile(arr, 0.5)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max())}


def run_audit(output_dir: Path, *, historical_count: int = 10_000,
              heldout_count: int = 1024, max_evaluations: int = 31,
              tolerance_m: float = 0.001) -> dict:
    if historical_count < 1 or historical_count > 10_000:
        raise ValueError("historical count must be within 1..10000")
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "independent_heldout_inputs.npz"
    frozen_sha = freeze_holdout(frozen_path, heldout_count)
    evaluator = PCCBoundedClearanceEvaluator()
    spec = evaluator.shape_model.spec
    input_digest = hashlib.sha256()
    report = {
        "schema": "v6_2_b1_bounded_clearance_audit_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            "v6_lite/pcc_bounded_clearance.py": _sha(
                Path(__file__).with_name("pcc_bounded_clearance.py")),
            "v6_lite/audit_b1_bounded_clearance.py": _sha(Path(__file__)),
            "v6_lite/audit_v6_1a.py": _sha(
                Path(__file__).with_name("audit_v6_1a.py")),
        },
        "historical_protocol": {
            "generator": "v6_lite.audit_v6_1a.audit_shape_vs_geometry",
            "seed": DEFAULT_SEED,
            "original_10000_case_inputs_saved": False,
            "regeneration_scope": "deterministic historical target-case protocol",
            "case_count": historical_count,
        },
        "independent_holdout": {"seed": HOLDOUT_SEED,
                                "case_count": heldout_count,
                                "frozen_inputs_sha256": frozen_sha},
        "query": {"max_evaluations": max_evaluations,
                  "tolerance_m": tolerance_m,
                  "safety_gate_m": SAFETY_GATE_M,
                  "tube_radii_m": V61A_PCC_TUBE_RADII_M.tolist(),
                  "section_arclength_lipschitz": 1.0,
                  "numerical_pad_m": 1e-9,
                  "online_controller_changed": False},
        "model_contract_sha256": spec.contract_sha256(),
        "cohorts": {},
    }
    with np.load(frozen_path, allow_pickle=False) as frozen:
        heldout_cases = [
            (frozen["configurations"][i],
             OrientedBox(frozen["centers"][i], frozen["rotations"][i],
                         frozen["half_extents"][i]),
             f"category_{int(frozen['categories'][i])}", None, False)
            for i in range(heldout_count)
        ]
    for cohort_name, cases in (("historical_regenerated",
                                _historical_cases(historical_count)),
                               ("independent_heldout", heldout_cases)):
        started = time.perf_counter()
        status = Counter()
        categories = Counter()
        gaps: list[float] = []
        times: list[float] = []
        evaluations: list[float] = []
        interval_counts: list[float] = []
        sampled_lower_violations = 0
        empirical_false_safe = 0
        geometry_checked = 0
        examples = []
        for i, (q, box, category, actual, truncated) in enumerate(cases):
            categories[category] += 1
            input_digest.update(np.asarray(q, dtype="<f8").tobytes())
            for part in (box.center, box.rotation, box.half_extents):
                input_digest.update(np.asarray(part, dtype="<f8").tobytes())
            result = evaluator.evaluate(
                q, np.eye(4), box, max_evaluations=max_evaluations,
                tolerance_m=tolerance_m, safety_gate_m=SAFETY_GATE_M,
            )
            status[result.proxy_clearance_status] += 1
            status["bounds_invalid"] += int(not result.bounds_valid)
            status["budget_exhausted"] += int(result.budget_exhausted)
            status["tolerance_met"] += int(result.tolerance_met)
            gaps.append(result.bound_gap_m)
            times.append(result.elapsed_ms)
            evaluations.append(result.evaluation_count)
            interval_counts.append(result.interval_count)
            if actual is not None and not truncated:
                geometry_checked += 1
                empirical_false_safe += int(
                    result.proxy_clearance_status == "PROXY_CLEARANCE_AT_LEAST_GATE"
                    and actual < SAFETY_GATE_M
                )
            # Dense samples can falsify an incorrect lower bound, but do not
            # establish the exact continuous minimum.
            if i < 64:
                sample_min = math.inf
                for segment in range(5):
                    a = spec.segment_boundaries_m[segment]
                    for local in np.linspace(0, spec.segment_lengths_m[segment], 129):
                        point = evaluator.shape_model.evaluate(
                            q, np.eye(4), float(a + local),
                            with_jacobians=False).position_world
                        signed = point_obb_signed_distance(point, box).signed_distance_m
                        sample_min = min(sample_min,
                                         signed - V61A_PCC_TUBE_RADII_M[segment])
                sampled_lower_violations += int(
                    sample_min < result.distance_lower_bound_m - 1e-8
                )
            if result.proxy_clearance_status == "UNKNOWN_CROSSES_GATE" and len(examples) < 10:
                examples.append({"index": i, "category": category,
                                 "lower_m": result.distance_lower_bound_m,
                                 "upper_m": result.distance_upper_bound_m,
                                 "budget_exhausted": result.budget_exhausted})
            if (i + 1) % 1000 == 0:
                print(f"[b1] {cohort_name}: {i + 1}", flush=True)
        report["cohorts"][cohort_name] = {
            "count": sum(categories.values()),
            "categories": dict(categories), "status_counts": dict(status),
            "bound_gap_m": _summary(gaps),
            "query_elapsed_ms": _summary(times),
            "evaluation_count": _summary(evaluations),
            "interval_count": _summary(interval_counts),
            "dense_sample_lower_violations_first_64": sampled_lower_violations,
            "dense_sample_scope": "diagnostic samples, not continuous truth",
            "mujoco_geometry_checked": geometry_checked,
            "empirical_false_safe_against_mujoco": empirical_false_safe,
            "unknown_examples": examples,
            "elapsed_wall_s": time.perf_counter() - started,
        }
    report["input_stream_sha256"] = input_digest.hexdigest()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["checks"] = {
        "all_queries_return_valid_bounds": all(
            x["status_counts"].get("bounds_invalid", 0) == 0
            for x in report["cohorts"].values()),
        "sampled_points_respect_lower_bounds": all(
            x["dense_sample_lower_violations_first_64"] == 0
            for x in report["cohorts"].values()),
        "historical_geometry_empirical_false_safe_zero":
            report["cohorts"]["historical_regenerated"][
                "empirical_false_safe_against_mujoco"] == 0,
        "historical_full_count": historical_count == 10_000,
    }
    report["passed"] = all(report["checks"].values())
    report_path = output_dir / "bounded_clearance_audit.json"
    _write(report_path, report)
    historic = report["cohorts"]["historical_regenerated"]
    holdout = report["cohorts"]["independent_heldout"]
    doc = f"""# V6.2-B.1 离线 PCC 有界距离审计

审计结果：**{'通过' if report['passed'] else '未通过'}**。在线 17 维 QP、67 路力矩和 MuJoCo/PCC/胶囊约束均未接入此模块。

| 指标 | 历史协议重生成 | 独立冻结留出 |
| --- | ---: | ---: |
| 案例数 | {historic['count']} | {holdout['count']} |
| 有效界失败 | {historic['status_counts'].get('bounds_invalid', 0)} | {holdout['status_counts'].get('bounds_invalid', 0)} |
| 代理安全 | {historic['status_counts'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | {holdout['status_counts'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} |
| 代理低于门槛 | {historic['status_counts'].get('PROXY_CLEARANCE_BELOW_GATE', 0)} | {holdout['status_counts'].get('PROXY_CLEARANCE_BELOW_GATE', 0)} |
| 未判定 | {historic['status_counts'].get('UNKNOWN_CROSSES_GATE', 0)} | {holdout['status_counts'].get('UNKNOWN_CROSSES_GATE', 0)} |
| 预算耗尽 | {historic['status_counts'].get('budget_exhausted', 0)} | {holdout['status_counts'].get('budget_exhausted', 0)} |
| 间隙 p95 (m) | {historic['bound_gap_m']['p95']:.6g} | {holdout['bound_gap_m']['p95']:.6g} |
| 查询时间 p95 (ms) | {historic['query_elapsed_ms']['p95']:.3f} | {holdout['query_elapsed_ms']['p95']:.3f} |
| 求值次数 p95 | {historic['evaluation_count']['p95']:.0f} | {holdout['evaluation_count']['p95']:.0f} |

历史组使用 V6.1-A 源码定义的种子与生成顺序重建案例；原始 10,000 输入未单独保存，因此这是历史协议回归，不是已保存输入文件的真实重放。独立留出输入冻结于 `independent_heldout_inputs.npz`，SHA-256 为 `{frozen_sha}`。

每段中心线的弧长导数范数在精确数学模型中为 1；点到 OBB 的有符号距离是 1-Lipschitz。区间中点给出每段下界，所有叶区间覆盖五段，采样值给出上界。数值结果用 1 nm 外扩和 `nextafter`；未做形式化浮点区间认证。原有半径及 5 mm 门槛未变。局部优化仅能改进上界。

“代理安全”仅表示当前 PCC 管体代理净空下界达到门槛；包络证据仍是 V6.1-A 有限样本回归。历史组中可比较的 MuJoCo 几何案例 {historic['mujoco_geometry_checked']} 个，代理假安全 {historic['empirical_false_safe_against_mujoco']} 个。密采样只用于寻找违反下界的反例，不被称为连续最小值真值。连续时间、全域包络和在线导数一致性仍待后续工作。
"""
    doc_path = output_dir / "BOUNDED_CLEARANCE_AUDIT.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(doc)
    files = (frozen_path, report_path, doc_path)
    manifest = {"schema": "v6_2_b1_audit_hash_manifest_v1",
                "passed": report["passed"],
                "artifacts": [{"path": p.name, "sha256": _sha(p),
                               "bytes": p.stat().st_size} for p in files]}
    _write(output_dir / "audit_manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("v6_lite/output/v6_2_b1/formal_audit"))
    parser.add_argument("--historical-count", type=int, default=10_000)
    parser.add_argument("--heldout-count", type=int, default=1024)
    parser.add_argument("--max-evaluations", type=int, default=31)
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()
    result = run_audit(args.output_dir,
                       historical_count=args.historical_count,
                       heldout_count=args.heldout_count,
                       max_evaluations=args.max_evaluations,
                       tolerance_m=args.tolerance)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]},
                     indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
