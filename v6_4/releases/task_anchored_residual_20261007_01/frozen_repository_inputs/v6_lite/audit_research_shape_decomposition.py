"""Read-only decomposition of the frozen 172 actual-safe proxy-below cases.

No controller, tube radius, safety distance or query tolerance is changed.
Same-radius discrete backbone and statewise envelope bounds are diagnostics,
not authorization to replace online constraints or shrink collision margins.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np

from v6_lite.audit_research_conservatism import FROZEN_INPUT_SHA, GATE, distribution, sha, write
from v6_lite.audit_v6_1a import (
    CAPSULE_AXIS_SAMPLES, PCC_ENVELOPE_SAMPLES_PER_SEGMENT,
    _capsule_axis_world_points, _pcc_points_by_segment, _set_target_geom_pose,
)
from v6_lite.continuum_shape_model import DiscreteContinuumKinematics
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id
from v6_lite.shape_clearance import (
    OrientedBox, ShapeClearanceShadow, minimum_capsule_clearance,
    minimum_mujoco_geom_clearance, point_obb_signed_distance,
    segment_obb_signed_distance,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "v6_lite/output/runs/research_conservatism_01"
REFERENCE_AUDIT = ROOT / "v6_lite/output/runs/research_conservatism_independent_audit_01"
NUMERICAL_MARGIN_M = 1e-4


def verify_manifest(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("reference artifact changed: " + str(path))


def diagnose(shape, discrete, envelopes, q, box, witness, model, data):
    spec = shape.spec
    actual = spec.planner_to_actuated @ q
    transforms = discrete.body_transforms(actual, np.eye(4))
    chain_rows = []
    for i, (name, length) in enumerate(zip(spec.module_body_names, spec.module_lengths_m)):
        transform = transforms[name]
        start = transform[:3, 3]
        end = start + transform[:3, :3] @ np.asarray([length, 0., 0.])
        result = segment_obb_signed_distance(start, end, box)
        point_distance = result.signed_distance_m
        chain_rows.append({"module_index": i, "body_name": name, "segment_id": i // 6,
                           "start_world": start.tolist(), "end_world": end.tolist(),
                           "centerline_distance_or_unsafe_indicator_m": point_distance,
                           "centerline_exterior_exact": point_distance > 0,
                           "tube_radius_m": float(V61A_PCC_TUBE_RADII_M[i // 6]),
                           "same_radius_net_clearance_indicator_m": float(point_distance - V61A_PCC_TUBE_RADII_M[i // 6]),
                           "witness_parameter": result.segment_parameter})
    chain_minimum = min(chain_rows, key=lambda row: row["same_radius_net_clearance_indicator_m"])
    segment_id = witness["best_segment_id"]
    s = float(spec.segment_boundaries_m[segment_id] + witness["best_local_arclength_m"])
    pcc = shape.evaluate(q, np.eye(4), s, with_jacobians=False).position
    actual_point = discrete.evaluate(actual, np.eye(4), s, transforms=transforms)
    backbone = actual_point.position
    pcc_sd = point_obb_signed_distance(pcc, box).signed_distance_m
    chain_sd = point_obb_signed_distance(backbone, box).signed_distance_m
    witness_row = {"segment_id": segment_id, "material_arclength_m": s,
                   "discrete_module_index": actual_point.module_index,
                   "discrete_module_arclength_m": actual_point.module_arclength_m,
                   "pcc_world": pcc.tolist(), "discrete_world": backbone.tolist(),
                   "pcc_point_sd_m": pcc_sd, "discrete_point_sd_m": chain_sd,
                   "position_discrepancy_norm_m": float(np.linalg.norm(backbone - pcc)),
                   "directional_distance_effect_m": chain_sd - pcc_sd,
                   "pcc_same_radius_sample_clearance_m": float(pcc_sd - V61A_PCC_TUBE_RADII_M[segment_id]),
                   "discrete_same_radius_sample_clearance_m": float(chain_sd - V61A_PCC_TUBE_RADII_M[segment_id]),
                   "radius_m": float(V61A_PCC_TUBE_RADII_M[segment_id]),
                   "matches_saved_proxy_upper": abs(float(pcc_sd - V61A_PCC_TUBE_RADII_M[segment_id] + 1e-9) - witness["distance_upper_bound_m"]) <= 1e-12,
                   "matches_saved_pcc_point": bool(np.allclose(pcc, witness["best_point_world"], rtol=0, atol=1e-12))}
    pcc_points = _pcc_points_by_segment(shape, q, PCC_ENVELOPE_SAMPLES_PER_SEGMENT)
    capsule_rows = []
    grouped = [[] for _ in range(5)]
    for cap in envelopes.capsules:
        axis = _capsule_axis_world_points(cap, transforms[cap.body_name], CAPSULE_AXIS_SAMPLES)
        distances = np.linalg.norm(axis[:, None, :] - pcc_points[cap.segment_index][None, :, :], axis=2)
        sampled_max = float(np.max(np.min(distances, axis=1)))
        row = {"geom_name": cap.geom_name, "body_name": cap.body_name, "segment_id": cap.segment_index,
               "sampled_axis_to_pcc_max_m": sampled_max, "capsule_radius_m": cap.radius_m,
               "sampled_required_radius_m": sampled_max + cap.radius_m,
               "axis_sampling_allowance_m": cap.axis_length_m / (2 * (CAPSULE_AXIS_SAMPLES - 1)),
               "source_vertex_excess_m": cap.maximum_vertex_excess_m}
        grouped[cap.segment_index].append(row)
        capsule_rows.append(row)
    segment_rows = []
    for i, group in enumerate(grouped):
        raw_required = max(row["sampled_required_radius_m"] for row in group)
        axis_allowance = max(row["axis_sampling_allowance_m"] for row in group)
        pcc_allowance = float(spec.segment_lengths_m[i] / (2 * (PCC_ENVELOPE_SAMPLES_PER_SEGMENT - 1)))
        sufficient_radius = raw_required + axis_allowance + pcc_allowance + NUMERICAL_MARGIN_M
        declared = float(V61A_PCC_TUBE_RADII_M[i])
        segment_rows.append({"segment_id": i, "sampled_required_radius_m": raw_required,
                             "axis_sampling_allowance_m": axis_allowance, "pcc_sampling_allowance_m": pcc_allowance,
                             "numerical_margin_m": NUMERICAL_MARGIN_M,
                             "sufficient_radius_upper_original_protocol_m": sufficient_radius,
                             "declared_radius_m": declared, "sufficient_radius_margin_m": declared - sufficient_radius})
    capsule_clearance = minimum_capsule_clearance(model, data, envelopes, box, spec=spec, compute_planner_gradient=False)
    position_errors = []
    rotation_errors = []
    for name, transform in transforms.items():
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if body_id < 0:
            raise ValueError("missing FK body: " + name)
        position_errors.append(float(np.linalg.norm(data.xpos[body_id] - transform[:3, 3])))
        rotation_errors.append(float(np.max(np.abs(data.xmat[body_id].reshape(3, 3) - transform[:3, :3]))))
    return {"same_radius_discrete_chain_gate_status": "SAFE" if chain_minimum["same_radius_net_clearance_indicator_m"] >= GATE else "BELOW",
            "same_radius_discrete_chain_clearance_indicator_m": chain_minimum["same_radius_net_clearance_indicator_m"],
            "chain_minimum_module_index": chain_minimum["module_index"],
            "all_chain_centerlines_exterior": all(row["centerline_exterior_exact"] for row in chain_rows),
            "chain_modules": chain_rows, "refusal_witness": witness_row,
            "last_module_endpoint_matches_terminal_frame": bool(np.allclose(chain_rows[-1]["end_world"], transforms[spec.end_effector_body_name][:3, 3], rtol=0, atol=1e-12)),
            "original_protocol_segment_envelope": segment_rows, "capsule_envelope_samples": capsule_rows,
            "original_protocol_sufficient_coverage_at_this_state": all(row["sufficient_radius_margin_m"] >= 0 for row in segment_rows),
            "minimum_sufficient_radius_margin_m": min(row["sufficient_radius_margin_m"] for row in segment_rows),
            "capsule_clearance_indicator_m": capsule_clearance.signed_distance_m,
            "capsule_gate_status": "SAFE" if capsule_clearance.signed_distance_m >= GATE else "BELOW",
            "capsule_minimum_source_geom": capsule_clearance.source_name,
            "capsule_witness_arm_world": capsule_clearance.point_on_arm.tolist(),
            "capsule_witness_box_world": capsule_clearance.point_on_box.tolist(),
            "discrete_fk_vs_mujoco_position_error_max_m": max(position_errors),
            "discrete_fk_vs_mujoco_rotation_entry_error_max": max(rotation_errors)}


def run(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = current_source_snapshot(ROOT)
    report = {"schema": "research_shape_envelope_decomposition_v1", "complete": False,
              "evidence_valid": False, "safety_contract_changed": False,
              "closed_loop_claim": False, "hardware_claim": False,
              "wall_deployment_status": "NOT_MET"}
    try:
        if frozen["capture_errors"] or frozen["tracked_worktree_dirty"]:
            raise RuntimeError("freeze the study source before execution")
        verify_manifest(REFERENCE)
        verify_manifest(REFERENCE_AUDIT)
        old = json.loads((REFERENCE / "plan.json").read_text(encoding="utf-8"))
        selections = json.loads((REFERENCE / "selections.json").read_text(encoding="utf-8"))
        old_rows = [json.loads(row) for row in (REFERENCE / "baseline.jsonl").read_text(encoding="utf-8").splitlines()]
        audit = json.loads((REFERENCE_AUDIT / "report.json").read_text(encoding="utf-8"))
        if audit["evidence_valid"] is not True:
            raise ValueError("reference independent evidence failed")
        indices = selections["proven_proxy_below_actual_safe_indices"]
        independently_selected = [r["index"] for r in old_rows if r["proxy_status"] == "PROXY_CLEARANCE_BELOW_GATE" and r["actual_mujoco_distance_m"] >= GATE]
        if len(indices) != 172 or indices != independently_selected or len(set(indices)) != 172:
            raise ValueError("the exact frozen 172-case cohort changed")
        input_path = Path(old["frozen_input"]["path"])
        if sha(input_path) != FROZEN_INPUT_SHA:
            raise ValueError("frozen geometry input changed")
        robot = default_v6_lite_robot_spec()
        model = robot.compile_dynamic_model()
        data = mujoco.MjData(model)
        qpos_ids, _ = joint_addresses(model, robot)
        target_qpos, _ = free_joint_slices(model, robot.target_free_joint_name)
        target_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"))
        shadow = ShapeClearanceShadow(model, target_id, V61A_PCC_TUBE_RADII_M)
        shape, spec = shadow.shape_model, shadow.spec
        if spec.contract_sha256() != old["model_contract_sha256"] or list(V61A_PCC_TUBE_RADII_M) != old["tube_radii_m"]:
            raise ValueError("frozen model or radii changed")
        discrete = DiscreteContinuumKinematics(spec)
        reference_files = [REFERENCE / name for name in ("artifact_manifest.json", "plan.json", "baseline.jsonl", "selections.json")]
        reference_files += [REFERENCE_AUDIT / "artifact_manifest.json", REFERENCE_AUDIT / "report.json"]
        reference_identity = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)} for path in reference_files]
        plan = {"schema": "research_shape_decomposition_plan_v1", "declared_before_diagnosis": True,
                "started_utc": datetime.now(timezone.utc).isoformat(), "source": frozen,
                "frozen_input_sha256": FROZEN_INPUT_SHA, "references": reference_identity,
                "indices": indices, "indices_sha256": hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
                "cohort": "exact 172 original actual-safe PCC proxy BELOW cases; no newly sampled inputs",
                "gate_m": GATE, "tube_radii_m": list(V61A_PCC_TUBE_RADII_M),
                "model_contract_sha256": spec.contract_sha256(), "model_contract": spec.to_dict(),
                "nominal_compiled_model_sha256": model_id(model, spec.contract_sha256()),
                "diagnostics": ["same-radius 30-module discrete centerline", "same-material-coordinate rejection witness",
                                "original statewise 17x5 capsule/PCC radius support", "actual chain capsule comparison"],
                "same_radius_chain_claim": "classification counterfactual only; not an independently certified envelope or online replacement",
                "negative_clearance_rule": "intersecting centerline and capsule negative values are unsafe indicators, not exact penetration depths",
                "negative_envelope_margin_rule": "sufficient upper bound exceeds fixed radius; inconclusive, not proof of actual noncontainment",
                "PCC_samples_per_section": PCC_ENVELOPE_SAMPLES_PER_SEGMENT,
                "capsule_axis_samples": CAPSULE_AXIS_SAMPLES, "original_envelope_numerical_margin_m": NUMERICAL_MARGIN_M,
                "capsule_count": len(shadow.envelopes.capsules), "fallback_geom_names": list(shadow.envelopes.fallback_geom_names),
                "fallback_included_in_envelope_radius_or_distances": False,
                "safety_contract_changed": False, "online_radius_reduction_authorized": False,
                "simulation_periods_unchanged": {"planning_s": .020, "physics_s": .002},
                "compute_performance_or_wall_test": False, "failure_policy": "exclusive output; preserve incomplete rows/failures"}
        write(output_dir / "plan.json", plan)
        (output_dir / "producer.py").write_bytes(Path(__file__).read_bytes())
        with np.load(input_path, allow_pickle=False) as saved:
            inputs = {key: saved[key].copy() for key in ("configurations", "centers", "rotations", "half_extents")}
        rows = []
        with (output_dir / "cases.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for ordinal, index in enumerate(indices):
                q = inputs["configurations"][index]
                center, rotation, extents = (inputs[key][index] for key in ("centers", "rotations", "half_extents"))
                model.geom_size[target_id] = extents
                data.qpos[:] = model.qpos0
                planner = robot.planner_zero.copy()
                planner[:10] = q
                data.qpos[qpos_ids] = robot.encode_position(planner)
                _set_target_geom_pose(model, data, target_qpos, target_id, center, rotation)
                mujoco.mj_forward(model, data)
                actual = minimum_mujoco_geom_clearance(model, data, shadow.enveloped_geom_ids, target_id)
                original = old_rows[index]
                diagnostic = diagnose(shape, discrete, shadow.envelopes, q, OrientedBox(center, rotation, extents),
                                      original["query_result"], model, data)
                witness = diagnostic["refusal_witness"]
                total_gap = actual.signed_distance_m - witness["pcc_same_radius_sample_clearance_m"]
                remaining = actual.signed_distance_m - witness["discrete_point_sd_m"]
                witness.update({"actual_minus_raw_pcc_witness_m": total_gap,
                                "remaining_global_surface_correspondence_term_m": remaining,
                                "decomposition_arithmetic_residual_m": total_gap - (
                                    witness["radius_m"] + witness["directional_distance_effect_m"] + remaining),
                                "remaining_term_is_pure_physical_radius": False})
                row = {"index": index, "original_lower_m": original["lower_m"], "original_upper_m": original["upper_m"],
                       "q_continuum_rad": q.tolist(), "target_center_m": center.tolist(),
                       "target_rotation": rotation.tolist(), "target_half_extents_m": extents.tolist(),
                       "actual_mujoco_distance_m": actual.signed_distance_m, "actual_query_truncated": actual.query_truncated,
                       "actual_source_geom": actual.source_name, "compiled_case_model_sha256": model_id(model, spec.contract_sha256()),
                       "matches_original_distance": abs(actual.signed_distance_m - original["actual_mujoco_distance_m"]) <= 1e-12,
                       "matches_original_model": model_id(model, spec.contract_sha256()) == original["compiled_case_model_sha256"],
                       **diagnostic}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if (ordinal + 1) % 32 == 0:
                    print(f"[shape-decomposition] {ordinal + 1}/172", flush=True)
        after = current_source_snapshot(ROOT)
        provenance = source_provenance(frozen, after)
        write(output_dir / "source_provenance.json", provenance)
        checks = {"exact_original_172_indices": [row["index"] for row in rows] == indices,
                  "same_actual_distances_and_case_models": all(row["matches_original_distance"] and row["matches_original_model"] for row in rows),
                  "all_original_actual_safe_proxy_below": all(row["actual_mujoco_distance_m"] >= GATE and row["original_upper_m"] < GATE for row in rows),
                  "all_saved_witnesses_reconstructed": all(row["refusal_witness"]["matches_saved_pcc_point"] and row["refusal_witness"]["matches_saved_proxy_upper"] for row in rows),
                  "no_terminal_offset_double_counted": all(row["last_module_endpoint_matches_terminal_frame"] for row in rows),
                  "witness_decomposition_identity_valid": all(abs(row["refusal_witness"]["decomposition_arithmetic_residual_m"]) <= 1e-12 for row in rows),
                  "discrete_fk_matches_mujoco": all(row["discrete_fk_vs_mujoco_position_error_max_m"] <= 1e-6 and row["discrete_fk_vs_mujoco_rotation_entry_error_max"] <= 1e-6 for row in rows),
                  "reference_files_unchanged": all(sha(ROOT / item["path"]) == item["sha256"] for item in reference_identity),
                  "input_unchanged": sha(input_path) == FROZEN_INPUT_SHA,
                  "source_unchanged": provenance["source_unchanged"] and not after["tracked_worktree_dirty"]}
        report.update({"complete": True, "evidence_valid": all(checks.values()), "checks": checks,
                       "source_commit": frozen["git_commit"], "case_count": len(rows), "indices_sha256": plan["indices_sha256"],
                       "same_radius_chain_status_counts": dict(Counter(row["same_radius_discrete_chain_gate_status"] for row in rows)),
                       "capsule_gate_status_counts": dict(Counter(row["capsule_gate_status"] for row in rows)),
                       "centerline_replacement_alone_flips_case_count": sum(row["same_radius_discrete_chain_gate_status"] == "SAFE" for row in rows),
                       "sufficient_envelope_support_at_this_state_count": sum(row["original_protocol_sufficient_coverage_at_this_state"] for row in rows),
                       "witness_position_discrepancy_norm_m": distribution([row["refusal_witness"]["position_discrepancy_norm_m"] for row in rows]),
                       "witness_directional_distance_effect_m": distribution([row["refusal_witness"]["directional_distance_effect_m"] for row in rows]),
                       "minimum_sufficient_radius_margin_m": distribution([row["minimum_sufficient_radius_margin_m"] for row in rows]),
                       "actual_minus_capsule_indicator_m": distribution([row["actual_mujoco_distance_m"] - row["capsule_clearance_indicator_m"] for row in rows]),
                       "claim_scope": "172 selected offline cases; proxy/shape/radius effects coupled; no independent contribution percentages or online radius reduction"})
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["source_after_failure"] = current_source_snapshot(ROOT)
    write(output_dir / "report.json", report)
    write(output_dir / "artifact_manifest.json", {"schema": "research_shape_decomposition_manifest_v1", "artifacts": [
        {"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir()) if path.is_file()]})
    print(json.dumps({key: report.get(key) for key in ("complete", "evidence_valid", "case_count", "same_radius_chain_status_counts", "capsule_gate_status_counts", "sufficient_envelope_support_at_this_state_count", "failure")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir)["evidence_valid"] else 1)
