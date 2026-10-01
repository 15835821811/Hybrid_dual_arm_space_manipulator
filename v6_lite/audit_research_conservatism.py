"""Attribute frozen holdout rejections before spending more query budget.

Offline geometry only. The controller, tube radii, gate and numerical pads
are unchanged. High-budget queries run only on the predeclared UNKNOWN set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import mujoco
import numpy as np
import scipy

from v6_lite.audit_b2_holdout_geometry import DEFAULT_INPUT
from v6_lite.audit_v6_1a import _set_target_geom_pose
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.pcc_bounded_clearance import PCCBoundedClearanceEvaluator
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id
from v6_lite.shape_clearance import OrientedBox, ShapeClearanceShadow, minimum_mujoco_geom_clearance

ROOT = Path(__file__).resolve().parents[1]
FROZEN_INPUT_SHA = "ec620f3396bc06f98597e76270501a1bd7a88a941e5740239bb10d775d3efe8c"
OLD_REPORT = ROOT / "v6_lite/output/v6_2_b2/heldout_geometry/heldout_geometry_report.json"
OLD_B1_REPORT = ROOT / "v6_lite/output/v6_2_b1/formal_audit_r02/bounded_clearance_audit.json"
SAFE = "PROXY_CLEARANCE_AT_LEAST_GATE"
BELOW = "PROXY_CLEARANCE_BELOW_GATE"
UNKNOWN = "UNKNOWN_CROSSES_GATE"
GATE = .005
TOLERANCE = .001
PAD = 1e-9
B1_BUDGETS = (255, 510, 1020)
ONLINE_BUDGETS = ((255, 128), (255, 512), (510, 512), (1020, 512))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite summary samples")
    return {"count": len(values), "min": float(values.min()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def counts(rows):
    result = Counter({key: 0 for key in (
        "checked_count", SAFE, BELOW, UNKNOWN, "bounds_invalid", "actual_geometry_safe",
        "actual_geometry_below_gate", "empirical_proxy_false_safe", "empirical_proxy_false_reject")})
    for row in rows:
        status = row["proxy_status"]
        actual_safe = row["actual_mujoco_distance_m"] >= GATE
        result["checked_count"] += 1
        result[status] += 1
        result["bounds_invalid"] += int(not row["bounds_valid"])
        result["actual_geometry_safe"] += int(actual_safe)
        result["actual_geometry_below_gate"] += int(not actual_safe)
        result["empirical_proxy_false_safe"] += int(status == SAFE and not actual_safe)
        result["empirical_proxy_false_reject"] += int(status != SAFE and actual_safe)
    return dict(result)


def discrepancy(actual, truncated, lower, upper):
    # At distmax MuJoCo supplies a lower bound, not an exact minimum. For
    # truncated cases actual-proxy has no finite measured upper endpoint.
    return {"actual_minus_proxy_lower_m": actual - (1e-12 if truncated else 0.0) - upper,
            "actual_minus_proxy_upper_m": None if truncated else actual - lower,
            "interpretation": "CENSORED_LOWER_BOUND" if truncated else "BOUNDED_DIFFERENCE"}


def run(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = current_source_snapshot(ROOT)
    report = {"schema": "research_conservatism_attribution_v1", "complete": False,
              "evidence_valid": False, "online_controller_changed": False,
              "closed_loop_claim": False, "hardware_claim": False,
              "supplements_previous_goal": True}
    try:
        if frozen["capture_errors"] or frozen["tracked_worktree_dirty"]:
            raise RuntimeError("commit the experiment tool before execution")
        input_path = ROOT / DEFAULT_INPUT
        source_manifest_path = input_path.parent / "audit_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        input_entry = next(item for item in source_manifest["artifacts"] if item["path"] == input_path.name)
        if sha(input_path) != FROZEN_INPUT_SHA or sha(input_path) != input_entry["sha256"] or input_path.stat().st_size != input_entry["bytes"]:
            raise ValueError("original frozen input identity mismatch")
        old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
        old_b1 = json.loads(OLD_B1_REPORT.read_text(encoding="utf-8"))
        if old["frozen_input_sha256"] != FROZEN_INPUT_SHA or old["query_max_evaluations"] != 31 or old["proxy_gate_m"] != GATE:
            raise ValueError("historical comparison protocol mismatch")
        if old_b1["query"]["tube_radii_m"] != list(V61A_PCC_TUBE_RADII_M) or old_b1["query"]["tolerance_m"] != TOLERANCE or old_b1["query"]["numerical_pad_m"] != PAD:
            raise ValueError("historical query parameters changed")
        robot = default_v6_lite_robot_spec()
        model = robot.compile_dynamic_model()
        data = mujoco.MjData(model)
        qpos_ids, _ = joint_addresses(model, robot)
        target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
        target_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"))
        shadow = ShapeClearanceShadow(model, target_id, V61A_PCC_TUBE_RADII_M)
        bounded = PCCBoundedClearanceEvaluator(shadow.shape_model)
        online = BatchedDistanceDecisionQuery(shadow.shape_model)
        spec = shadow.shape_model.spec
        identity_salt = spec.contract_sha256()
        plan = {"schema": "research_conservatism_plan_v1", "declared_before_queries": True,
                "started_utc": datetime.now(timezone.utc).isoformat(), "source": frozen,
                "frozen_input": {"path": str(input_path), "sha256": FROZEN_INPUT_SHA, "bytes": input_entry["bytes"]},
                "historical_report": {"path": str(OLD_REPORT), "sha256": sha(OLD_REPORT)},
                "historical_b1": {"path": str(OLD_B1_REPORT), "sha256": sha(OLD_B1_REPORT),
                                  "model_contract_sha256": old_b1["model_contract_sha256"]},
                "input_manifest": {"path": str(source_manifest_path), "sha256": sha(source_manifest_path)},
                "model_contract_sha256": identity_salt, "model_contract": spec.to_dict(),
                "nominal_compiled_model_sha256": model_id(model, identity_salt),
                "mujoco_reference_geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in shadow.enveloped_geom_ids],
                "gate_m": GATE, "tube_radii_m": list(V61A_PCC_TUBE_RADII_M),
                "tolerance_m": TOLERANCE, "numerical_pad_m": PAD, "query_distance_max_m": .5,
                "baseline": {"count": 1024, "B1_max_evaluations": 31},
                "subset_selection": "all and only baseline UNKNOWN_CROSSES_GATE; save indices before refinement",
                "b1_subset_max_evaluations": list(B1_BUDGETS), "b1_leaf_cap": None,
                "online_subset_point_leaf_budgets": [list(pair) for pair in ONLINE_BUDGETS],
                "online_partition_each_query": "fresh five roots, no cross-case reuse",
                "high_budget_non_unknown_status": "NOT_RUN",
                "timing_scope": "single offline query observations; subset is not population or control-cycle timing",
                "wall_deployment_status": "NOT_MET", "administrator_branch": "DEFERRED",
                "failure_policy": "exclusive directory; retain partial rows and failure report",
                "environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "mujoco": mujoco.__version__,
                                "numpy": np.__version__, "scipy": scipy.__version__}}
        write(output_dir / "plan.json", plan)
        (output_dir / "producer.py").write_bytes(Path(__file__).read_bytes())
        with np.load(input_path, allow_pickle=False) as saved:
            inputs = {key: saved[key].copy() for key in ("configurations", "centers", "rotations", "half_extents")}
        expected_shapes = {"configurations": (1024, 10), "centers": (1024, 3),
                           "rotations": (1024, 3, 3), "half_extents": (1024, 3)}
        if any(value.shape != expected_shapes[key] or not np.all(np.isfinite(value)) for key, value in inputs.items()):
            raise ValueError("frozen input layout invalid")
        baseline = []
        with (output_dir / "baseline.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index, q in enumerate(inputs["configurations"]):
                center, rotation, extents = (inputs[key][index] for key in ("centers", "rotations", "half_extents"))
                model.geom_size[target_id] = extents
                data.qpos[:] = model.qpos0
                planner = robot.planner_zero.copy()
                planner[:10] = q
                data.qpos[qpos_ids] = robot.encode_position(planner)
                _set_target_geom_pose(model, data, target_qpos, target_id, center, rotation)
                mujoco.mj_forward(model, data)
                box = OrientedBox(center, rotation, extents)
                result = bounded.evaluate(q, np.eye(4), box, max_evaluations=31,
                                          safety_gate_m=GATE, tolerance_m=TOLERANCE, numerical_pad_m=PAD,
                                          local_refinement_count=0)
                actual = minimum_mujoco_geom_clearance(model, data, shadow.enveloped_geom_ids, target_id, query_distance_max_m=.5)
                row = {"index": index, "lower_m": result.distance_lower_bound_m,
                       "upper_m": result.distance_upper_bound_m, "gap_m": result.bound_gap_m,
                       "bounds_valid": result.bounds_valid, "proxy_status": result.proxy_clearance_status,
                       "actual_mujoco_distance_m": actual.signed_distance_m,
                       "actual_query_truncated": actual.query_truncated,
                       "actual_source_geom": actual.source_name,
                       "compiled_case_model_sha256": model_id(model, identity_salt),
                       "evaluation_count": result.evaluation_count, "interval_count": result.interval_count,
                       "tolerance_met": result.tolerance_met, "budget_exhausted": result.budget_exhausted,
                       "elapsed_ms": result.elapsed_ms, "high_budget_status": "PENDING_UNKNOWN_SUBSET" if result.proxy_clearance_status == UNKNOWN else "NOT_RUN",
                       "query_result": result.to_dict(),
                       "discrepancy": discrepancy(actual.signed_distance_m, actual.query_truncated,
                                                  result.distance_lower_bound_m, result.distance_upper_bound_m)}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                baseline.append(row)
                if (index + 1) % 128 == 0:
                    print(f"[conservatism] baseline {index + 1}/1024", flush=True)
        base_counts = counts(baseline)
        if base_counts != old["counts"]:
            report["baseline_counts"] = base_counts
            raise ValueError("current baseline does not reproduce historical counts")
        unknown_indices = [row["index"] for row in baseline if row["proxy_status"] == UNKNOWN]
        false_rejects = [row["index"] for row in baseline if row["proxy_status"] != SAFE and row["actual_mujoco_distance_m"] >= GATE]
        geometric_rejects = [row["index"] for row in baseline if row["proxy_status"] == BELOW and row["actual_mujoco_distance_m"] >= GATE]
        selections = {"unknown_indices": unknown_indices, "false_reject_indices": false_rejects,
                      "proven_proxy_below_actual_safe_indices": geometric_rejects,
                      "high_budget_NOT_RUN_indices": [row["index"] for row in baseline if row["proxy_status"] != UNKNOWN]}
        write(output_dir / "selections.json", selections)
        if len(unknown_indices) > 18:
            raise ValueError("predeclared subset size exceeded")
        subset = []
        with (output_dir / "refinement.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index in unknown_indices:
                q = inputs["configurations"][index]
                box = OrientedBox(inputs["centers"][index], inputs["rotations"][index], inputs["half_extents"][index])
                for budget in B1_BUDGETS:
                    result = bounded.evaluate(q, np.eye(4), box, max_evaluations=budget,
                                              safety_gate_m=GATE, tolerance_m=TOLERANCE, numerical_pad_m=PAD,
                                              local_refinement_count=0)
                    row = {"index": index, "family": "B1_distance_bounds", "point_budget": budget,
                           "leaf_budget": None, "proxy_status": result.proxy_clearance_status,
                           "lower_m": result.distance_lower_bound_m, "upper_m": result.distance_upper_bound_m,
                           "gap_m": result.bound_gap_m, "bounds_valid": result.bounds_valid,
                           "point_count": result.evaluation_count, "leaf_count": result.interval_count,
                           "elapsed_ms": result.elapsed_ms, "tolerance_met": result.tolerance_met,
                           "budget_exhausted": result.budget_exhausted, "query_result": result.to_dict()}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    subset.append(row)
                for point_budget, leaf_budget in ONLINE_BUDGETS:
                    result = online.evaluate(q, np.eye(4), box, IntervalPartition.uniform(), gate_m=GATE,
                                             max_point_evaluations=point_budget, max_leaves=leaf_budget, numerical_pad_m=PAD)
                    row = {"index": index, "family": "BatchedDistanceDecisionQuery", "point_budget": point_budget,
                           "leaf_budget": leaf_budget, "proxy_status": result.proxy_clearance_status,
                           "lower_m": result.distance_lower_bound_m, "upper_m": result.distance_upper_bound_m,
                           "gap_m": result.bound_gap_m, "bounds_valid": result.bounds_valid,
                           "point_count": result.point_evaluation_count, "leaf_count": result.interval_count,
                           "split_count": result.split_count, "elapsed_ms": result.elapsed_ms,
                           "budget_exhausted": result.budget_exhausted, "failure_reason": result.failure_reason,
                           "partition_ids": [leaf.interval_id for leaf in result.partition.leaves],
                           "coverage_complete": result.partition.coverage(spec.segment_lengths_m).coverage_complete,
                           "lower_by_interval_id": result.lower_by_interval_id,
                           "midpoint_upper_by_interval_id": result.midpoint_upper_by_interval_id}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    subset.append(row)
                print(f"[conservatism] refined UNKNOWN index {index}", flush=True)
        variant_summaries = []
        variants = [("B1_distance_bounds", b, None) for b in B1_BUDGETS] + [("BatchedDistanceDecisionQuery", b, l) for b, l in ONLINE_BUDGETS]
        for family, budget, leaf_budget in variants:
            rows = [row for row in subset if (row["family"], row["point_budget"], row["leaf_budget"]) == (family, budget, leaf_budget)]
            variant_summaries.append({"family": family, "point_budget": budget, "leaf_budget": leaf_budget,
                                      "sample_count": len(rows), "status_counts": dict(Counter(row["proxy_status"] for row in rows)),
                                      "point_count": distribution([row["point_count"] for row in rows]),
                                      "leaf_count": distribution([row["leaf_count"] for row in rows]),
                                      "gap_m": distribution([row["gap_m"] for row in rows]),
                                      "query_elapsed_ms": distribution([row["elapsed_ms"] for row in rows]),
                                      "timing_population": "baseline UNKNOWN subset only, first sample included",
                                      "resolved_safe_indices": [row["index"] for row in rows if row["proxy_status"] == SAFE],
                                      "resolved_below_indices": [row["index"] for row in rows if row["proxy_status"] == BELOW],
                                      "remaining_unknown_indices": [row["index"] for row in rows if row["proxy_status"] == UNKNOWN]})
        after = current_source_snapshot(ROOT)
        provenance = source_provenance(frozen, after)
        write(output_dir / "source_provenance.json", provenance)
        checks = {"all_original_1024_compared": len(baseline) == 1024,
                  "historical_counts_reproduced": base_counts == old["counts"],
                  "all_bounds_valid": all(row["bounds_valid"] for row in baseline + subset),
                  "no_empirical_false_safe_baseline": base_counts["empirical_proxy_false_safe"] == 0,
                  "all_predeclared_subset_variants_complete": len(subset) == len(unknown_indices) * 7,
                  "subset_only_unknown": all(row["index"] in unknown_indices for row in subset),
                  "online_complete_coverage": all(row.get("coverage_complete", True) for row in subset),
                  "source_unchanged": provenance["source_unchanged"] and not after["tracked_worktree_dirty"],
                  "input_unchanged": sha(input_path) == FROZEN_INPUT_SHA,
                  "historical_report_unchanged": sha(OLD_REPORT) == plan["historical_report"]["sha256"],
                  "all_subset_safe_predictions_actual_safe": all(baseline[row["index"]]["actual_mujoco_distance_m"] >= GATE for row in subset if row["proxy_status"] == SAFE)}
        report.update({"complete": True, "checks": checks, "evidence_valid": all(checks.values()),
                       "git_commit": frozen["git_commit"], "frozen_input_sha256": FROZEN_INPUT_SHA,
                       "baseline_counts": base_counts, "unknown_count": len(unknown_indices),
                       "status_by_actual_gate": {status: {
                           "actual_safe": sum(row["proxy_status"] == status and row["actual_mujoco_distance_m"] >= GATE for row in baseline),
                           "actual_below_gate": sum(row["proxy_status"] == status and row["actual_mujoco_distance_m"] < GATE for row in baseline)}
                           for status in (SAFE, BELOW, UNKNOWN)},
                       "actual_safe_proxy_below_count": len(geometric_rejects),
                       "actual_safe_proxy_unknown_count": len(set(false_rejects) & set(unknown_indices)),
                       "actual_distance_censored_count": sum(row["actual_query_truncated"] for row in baseline),
                       "baseline_query_gap_m": distribution([row["gap_m"] for row in baseline]),
                       "baseline_offline_query_elapsed_ms": distribution([row["elapsed_ms"] for row in baseline]),
                       "subset_variants": variant_summaries,
                       "claim_scope": "finite frozen offline geometry attribution; no robust/continuous-time/closed-loop or wall deployment guarantee"})
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["source_after_failure"] = current_source_snapshot(ROOT)
    write(output_dir / "report.json", report)
    artifacts = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(output_dir.iterdir()) if path.is_file()]
    write(output_dir / "artifact_manifest.json", {"schema": "research_conservatism_artifact_manifest_v1", "artifacts": artifacts})
    print(json.dumps({key: report.get(key) for key in ("complete", "evidence_valid", "baseline_counts", "actual_safe_proxy_below_count", "actual_safe_proxy_unknown_count", "failure")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir)["evidence_valid"] else 1)
