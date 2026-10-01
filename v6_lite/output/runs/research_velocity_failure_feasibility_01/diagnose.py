"""Independent finite LP diagnosis of the three frozen 2x failures.

No controller solve, geometry query, physics step, parameter change, or old
artifact write occurs. Model helpers supply the pinned named-joint contract;
all constraint assembly, endpoint reconstruction, LPs, and residual checks
below are independent of the producer QP and action-validation functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import scipy
from scipy.optimize import linprog
from model_test.robot_model_spec_v5 import default_robot_model_spec_v5
from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere,
)

TRIAL = ROOT / "v6_lite/output/runs/research_velocity_stress_01"
OUT = Path(__file__).resolve().parent
TRIAL_COMMIT = "9c7120e8ca8711cd58dc65471e58763c965148ff"
CASES = ("v6_lite_scenario_00", "v6_lite_scenario_03", "v6_lite_scenario_04")
CHECKS: list[dict] = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=True, allow_nan=False)
        f.write("\n")


def check(name, condition, detail=None):
    passed = bool(condition)
    CHECKS.append({"name": name, "passed": passed, "detail": detail})
    if not passed:
        raise ValueError(f"Evidence integrity check failed: {name}")


def vector(value, size, name):
    a = np.asarray(value, dtype=np.float64)
    check(name, a.shape == (size,) and np.all(np.isfinite(a)))
    return a


def exact_dot(row, value):
    # The certificate applies to the exact rational values of the recorded
    # float64 coefficients, avoiding an optimistic rounded dot product.
    return sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                for a, b in zip(row, value)), Fraction(0))


def exact_residuals(A, lo, up, x, tol):
    lower, upper = [], []
    within = True
    for row, l, u, t in zip(A, lo, up, tol):
        product = exact_dot(row, x)
        lr = product - Fraction.from_float(float(l)) if np.isfinite(l) else None
        ur = Fraction.from_float(float(u)) - product if np.isfinite(u) else None
        lower.append(float(lr) if lr is not None else None)
        upper.append(float(ur) if ur is not None else None)
        threshold = -Fraction.from_float(float(t))
        within &= ((lr is None or lr >= threshold)
                   and (ur is None or ur >= threshold))
    finite = [r for r in lower + upper if r is not None]
    return {"lower_residuals": lower, "upper_residuals": upper,
            "minimum_residual": min(finite),
            "max_violation": max(0.0, -min(finite)),
            "within_original_row_tolerances_exact_binary_rational": bool(within)}


def upper_form(A, lo, up, tol):
    rows, bounds, tolerances, ids = [], [], [], []
    for i in range(len(lo)):
        if np.isfinite(up[i]):
            rows.append(A[i]); bounds.append(up[i]); tolerances.append(tol[i])
            ids.append({"row_index": i, "side": "upper"})
        if np.isfinite(lo[i]):
            rows.append(-A[i]); bounds.append(-lo[i]); tolerances.append(tol[i])
            ids.append({"row_index": i, "side": "lower"})
    return np.asarray(rows), np.asarray(bounds), np.asarray(tolerances), ids


def lp_record(result, A, lo, up, tol):
    return {"status": int(result.status), "success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "objective": float(result.fun) if result.fun is not None else None,
            "witness": result.x.tolist() if result.x is not None else None,
            "independent_exact_residual_check": (
                exact_residuals(A, lo, up, result.x, tol)
                if result.x is not None else None)}


def analyse_system(A, lo, up, tol, options):
    G, b, rowtol, ids = upper_form(A, lo, up, tol)
    strict = linprog(np.zeros(17), A_ub=G, b_ub=b,
                     bounds=[(None, None)] * 17, method="highs", options=options)
    # This is the unchanged declared validator tolerance, not a new contract.
    original_tolerance = linprog(np.zeros(17), A_ub=G, b_ub=b + rowtol,
                                bounds=[(None, None)] * 17,
                                method="highs", options=options)
    # Auxiliary delta diagnoses the unavoidable worst absolute row violation.
    # It is neither a replacement command nor a weakened safety constraint.
    phase = linprog(np.r_[np.zeros(17), 1.0],
                    A_ub=np.c_[G, -np.ones(len(G))], b_ub=b,
                    bounds=[(None, None)] * 17 + [(0.0, None)],
                    method="highs", options=options)
    phase_record = {"status": int(phase.status), "success": bool(phase.success),
                    "message": str(phase.message), "iterations": int(phase.nit),
                    "minimum_uniform_absolute_violation_lp": (
                        float(phase.x[-1]) if phase.x is not None else None),
                    "velocity_witness": phase.x[:17].tolist() if phase.x is not None else None,
                    "witness_exact_residuals": (
                        exact_residuals(A, lo, up, phase.x[:17], tol)
                        if phase.x is not None else None),
                    "inequality_marginals": (
                        phase.ineqlin.marginals.tolist() if phase.x is not None else None)}
    return {"row_count": len(lo), "finite_inequality_count": len(b),
            "strict_inequalities_lp": lp_record(strict, A, lo, up, tol),
            "original_tolerance_inequalities_lp": lp_record(
                original_tolerance, A, lo, up, tol),
            "phase_one_minimax": phase_record, "upper_form_row_mapping": ids}


def run():
    check("exclusive_output", not (OUT / "report.json").exists())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    check("trial_commit_is_current", head == TRIAL_COMMIT)
    plan, report = load(TRIAL / "plan.json"), load(TRIAL / "report.json")
    metadata = load(TRIAL / "simulation/run_metadata.json")
    producer_manifest = load(TRIAL / "manifest.json")
    inputs = [{"path": (TRIAL / "manifest.json").relative_to(ROOT).as_posix(),
               "sha256": sha(TRIAL / "manifest.json"),
               "size_bytes": (TRIAL / "manifest.json").stat().st_size,
               "role": "producer_manifest"}]
    for relative, expected in producer_manifest.items():
        path = TRIAL / relative
        actual = sha(path)
        check("producer_artifact_sha:" + relative, actual == expected)
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": actual,
                       "size_bytes": path.stat().st_size, "role": "producer_artifact"})
    check("trial_retains_failure", report["passed"] is False and report["complete"] is False)
    check("plan_source_commit", plan["source"]["git_commit"] == TRIAL_COMMIT)
    check("source_before_after_identical",
          report["source_provenance"]["before"]["inventory_sha256"]
          == report["source_provenance"]["after"]["inventory_sha256"]
          == plan["source"]["inventory_sha256"])
    source_checks = []
    for relative, expected in plan["source"]["files"].items():
        path = ROOT / relative
        actual = sha(path)
        check("frozen_source_sha:" + relative,
              actual == expected["sha256_raw"] and path.stat().st_size == expected["size_bytes"])
        source_checks.append({"path": relative, "sha256": actual,
                              "size_bytes": path.stat().st_size})
    cfg = plan["qp_config"]
    check("original_tolerances", cfg["feasibility_tolerance"] == 1e-4
          and cfg["clearance_rate_tolerance_m_s"] == 1e-4
          and cfg["velocity_tolerance_rad_s"] == 1e-4
          and cfg["qp_ftol"] == 5e-5 and cfg["qp_max_iterations"] == 1200)
    options = {"primal_feasibility_tolerance": cfg["feasibility_tolerance"],
               "dual_feasibility_tolerance": cfg["qp_ftol"]}
    spec = default_robot_model_spec_v5()
    model_contract, model_bundle = spec.runtime_contract_sha256(), spec.source_bundle_sha256()
    check("model_identity_matches_run", metadata["model_identity"]["runtime_contract_sha256"]
          == model_contract and metadata["model_identity"]["source_bundle_sha256"] == model_bundle)
    model_assets = []
    for path in spec._source_assets():
        model_assets.append({"path": str(path.resolve()), "sha256": sha(path),
                             "size_bytes": path.stat().st_size})
    scenes = {s["scenario_id"]: s for s in load(TRIAL / "scenario_definitions.json")["scale_two"]}
    weights = np.asarray([(1.0 - step / 10.0, step / 10.0)
                          for step in range(1, 11)], dtype=np.float64).mean(axis=0)
    check("original_ramp_weights", abs(weights[0] - .45) <= 1e-16
          and abs(weights[1] - .55) <= 1e-16)
    # continuum_model_spec.py:352,373-374 fixes these ten domain coordinates.
    work_lower, work_upper = -np.ones(10), np.ones(10)
    summaries = []
    for sid in CASES:
        snap_path = TRIAL / "simulation/failures" / (sid + "_counterexample.json")
        failure_path = snap_path.with_name(sid + "_execution_failure.json")
        trace_path = snap_path.with_name(sid + "_partial_trace.npz")
        snap, failure = load(snap_path), load(failure_path)
        states = snap["planning_snapshots"]
        s = states[-1]
        check(sid + ":last_failure", len(states) == 3 and s["solver_status"] == "maximum_iterations"
              and s["failure_reason"] == "iteration_limit" and s["selected_command"] is None
              and snap["next_servo_step_executed"] is False
              and failure["next_servo_step_executed"] is False)
        check(sid + ":model_identity", snap["model_runtime_contract_sha256"] == model_contract
              and snap["model_source_bundle_sha256"] == model_bundle)
        check(sid + ":configuration", snap["qp_config"] == cfg
              and snap["run_config"] == plan["run_config"]
              and snap["configuration_sha256"] == metadata["configuration_sha256"]
              == canonical_sha({"run": plan["run_config"], "qp": cfg}))
        obstacles = tuple(WorkspaceSphere(o["name"], np.asarray(o["center_w"]), o["radius_m"])
                          for o in scenes[sid]["workspace_obstacles"])
        model = WholeBodyCollisionVerifier(spec, obstacles, WholeBodyVerificationConfig(
            minimum_clearance=plan["run_config"]["whole_body_minimum_clearance_m"],
            query_distance_max=2.5, adaptive_subdivisions=4,
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True)).model
        ids = []
        for name in spec.low_level_joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            check(sid + ":named_scalar_joint:" + name,
                  jid >= 0 and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE))
            ids.append(int(model.jnt_qposadr[jid]))
        qpos = vector(s["qpos"], model.nq, sid + ":qpos")
        vector(s["qvel"], model.nv, sid + ":qvel")
        q = spec.low_level_to_planner @ qpos[ids]
        old = vector(s["old_command"], 17, sid + ":old_command")
        candidate = vector(s["solver_candidate"], 17, sid + ":candidate")
        check(sid + ":old_equals_previous_selected", np.array_equal(
            old, np.asarray(states[-2]["selected_command"])))
        vl = vector(s["velocity_lower"], 17, sid + ":velocity_lower")
        vu = vector(s["velocity_upper"], 17, sid + ":velocity_upper")
        speed = cfg["velocity_limit_scale"] * spec.planner_velocity_limits
        reconstructed_lower = np.maximum(-speed, old - spec.planner_acceleration_limits * cfg["task_period_s"])
        reconstructed_upper = np.minimum(speed, old + spec.planner_acceleration_limits * cfg["task_period_s"])
        reconstructed_lower = np.maximum(reconstructed_lower, -cfg["joint_barrier_gain"] * (
            q - (spec.planner_lower + cfg["joint_position_margin_rad"])))
        reconstructed_upper = np.minimum(reconstructed_upper, cfg["joint_barrier_gain"] * (
            (spec.planner_upper - cfg["joint_position_margin_rad"]) - q))
        check(sid + ":velocity_bounds_reconstruct_exact",
              np.array_equal(vl, reconstructed_lower) and np.array_equal(vu, reconstructed_upper))
        with np.load(trace_path, allow_pickle=False) as trace:
            check(sid + ":frozen_state_equals_trace", np.array_equal(trace["task_qpos"][-1], qpos)
                  and float(trace["task_time"][-1]) == s["time_s"]
                  and float(trace["failure_time_s"]) == s["time_s"]
                  and int(trace["task_iterations"][-1]) == cfg["qp_max_iterations"]
                  and len(trace["time"]) == snap["partial_trace"]["physics_steps_saved"])
        check(sid + ":partial_trace_sha", sha(trace_path) == snap["partial_trace"]["sha256"])
        C = np.asarray([r["gradient_m_per_rad"] for r in s["rows"]], dtype=np.float64)
        b = np.asarray([r["lower_m_s"] for r in s["rows"]], dtype=np.float64)
        H = np.asarray([r["lookahead_gradient_m_per_rad"] for r in s["rows"]], dtype=np.float64)
        h = np.asarray([r["lookahead_lower_m_s"] for r in s["rows"]], dtype=np.float64)
        check(sid + ":complete_rows_finite", C.shape == (38, 17) and H.shape == C.shape
              and b.shape == (38,) and h.shape == (38,)
              and all(np.all(np.isfinite(a)) for a in (C, b, H, h)))
        gains = np.asarray([r["barrier_gain_s_inv"] for r in s["rows"]])
        drifts = np.asarray([r["target_drift_m_s"] for r in s["rows"]])
        gain_dt = gains * cfg["task_period_s"]
        check(sid + ":lookahead_reconstruct_exact", np.array_equal(
            H, C * (1.0 + gain_dt * weights[1])[:, None]) and np.array_equal(
            h, b - gain_dt * (weights[0] * (C @ old) + drifts)
            + cfg["lookahead_model_margin_m_s"]))
        check(sid + ":stored_candidate_residuals", all(
            math.isclose(float(C[i] @ candidate - b[i]), r["candidate_residual_m_s"],
                         rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(C[i] @ old - b[i]), r["old_command_residual_m_s"],
                             rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(H[i] @ candidate - h[i]), r["lookahead_candidate_residual_m_s"],
                             rel_tol=0.0, abs_tol=1e-12)
            for i, r in enumerate(s["rows"])))
        offset = q[:10] + cfg["task_period_s"] * weights[0] * old[:10]
        denominator = cfg["task_period_s"] * weights[1]
        dl, du = (work_lower - offset) / denominator, (work_upper - offset) / denominator
        A = np.vstack([C, H, np.eye(17), np.eye(17)[:10]])
        lo = np.r_[b, h, vl, dl]
        up = np.r_[np.full(76, np.inf), vu, du]
        tol = np.full(len(lo), cfg["feasibility_tolerance"])
        labels = (["instantaneous:" + r["source"] for r in s["rows"]]
                  + ["lookahead:" + r["source"] for r in s["rows"]]
                  + [f"velocity_box:{j}" for j in range(17)]
                  + [f"domain_endpoint:{j}" for j in range(10)])
        ramp_A, ramp_lo, ramp_up, ramp_tol, ramp_labels = [A], [lo], [up], [tol], labels.copy()
        for step in range(11):
            alpha = step / 10.0
            offset_velocity = (1.0 - alpha) * old
            ramp_A.extend([alpha * C, alpha * np.eye(17)])
            ramp_lo.extend([b - C @ offset_velocity, vl - offset_velocity])
            ramp_up.extend([np.full(38, np.inf), vu - offset_velocity])
            ramp_tol.extend([np.full(38, cfg["clearance_rate_tolerance_m_s"]),
                             np.full(17, cfg["velocity_tolerance_rad_s"])])
            ramp_labels.extend([f"ramp_step_{step}:" + r["source"] for r in s["rows"]]
                               + [f"ramp_step_{step}:velocity_box:{j}" for j in range(17)])
        RA, RL, RU, RT = (np.vstack(ramp_A), np.concatenate(ramp_lo),
                          np.concatenate(ramp_up), np.concatenate(ramp_tol))
        gap = vl[:10] - du
        j = int(np.argmax(gap))
        exact_gap = (Fraction.from_float(float(vl[j])) - Fraction.from_float(float(du[j]))
                     - 2 * Fraction.from_float(cfg["feasibility_tolerance"]))
        check(sid + ":exact_scalar_infeasibility_certificate", exact_gap > 0)
        contradiction = {"planner_index": j, "planner_name": spec.planner_coordinate_names[j],
            "measured_shape_q_rad": float(q[j]), "old_command_rad_s": float(old[j]),
            "velocity_lower_rad_s": float(vl[j]), "domain_endpoint_upper_rad_s": float(du[j]),
            "unadjusted_gap_rad_s": float(gap[j]),
            "sum_of_original_two_row_tolerances_rad_s": 2 * cfg["feasibility_tolerance"],
            "tolerance_adjusted_gap_rad_s": float(exact_gap),
            "exact_binary_rational_adjusted_gap": {"numerator": str(exact_gap.numerator),
                                                    "denominator": str(exact_gap.denominator)},
            "proof": "The same scalar must satisfy v >= velocity_lower - 1e-4 and "
                     "v <= domain_endpoint_upper + 1e-4. Their exact recorded binary-rational "
                     "lower-minus-upper is positive; no value can satisfy both.",
            "unavoidable_max_uniform_violation_lower_bound_rad_s": float(gap[j] / 2)}
        qp_analysis = analyse_system(A, lo, up, tol, options)
        ramp_analysis = analyse_system(RA, RL, RU, RT, options)
        check(sid + ":highs_corroborates_exact_certificate", all(
            a[k]["status"] == 2 for a in (qp_analysis, ramp_analysis)
            for k in ("strict_inequalities_lp", "original_tolerance_inequalities_lp")))
        candidate_qp = exact_residuals(A, lo, up, candidate, tol)
        candidate_ramp = exact_residuals(RA, RL, RU, candidate, RT)
        case = {"scenario_id": sid, "failure_time_s": s["time_s"],
            "status": "INFEASIBLE_WITHIN_UNCHANGED_ORIGINAL_TOLERANCES",
            "snapshot_sha256": sha(snap_path), "partial_trace_sha256": sha(trace_path),
            "execution_failure_sha256": sha(failure_path),
            "model_runtime_contract_sha256": model_contract,
            "model_source_bundle_sha256": model_bundle,
            "configuration_sha256": snap["configuration_sha256"],
            "joint_qpos_addresses_in_low_level_name_order": ids,
            "measured_planner_q": q.tolist(), "work_domain_lower_rad": work_lower.tolist(),
            "work_domain_upper_rad": work_upper.tolist(), "ramp_mean_weights": weights.tolist(),
            "domain_endpoint_lower_rad_s": dl.tolist(), "domain_endpoint_upper_rad_s": du.tolist(),
            "qp_row_labels": labels, "ramp_row_labels": ramp_labels,
            "exact_scalar_contradiction_certificate": contradiction,
            "producer_candidate_full_qp_residuals": candidate_qp,
            "producer_candidate_frozen_ramp_residuals": candidate_ramp,
            "qp_linear_feasibility": qp_analysis, "qp_plus_frozen_ramp_feasibility": ramp_analysis,
            "missing_objective_and_solver_history": "Snapshots omit Hessian, linear objective, "
            "ADMM dual/penalty/residual history. They are unnecessary for the demonstrated "
            "linear infeasibility, but this audit cannot reproduce optimality or explain "
            "iteration-by-iteration solver behavior.",
            "not_executed": ["LP candidate dispatch", "ten-step nonlinear preview", "MuJoCo step",
                             "full cohort", "wall timing", "parameter or tolerance change"]}
        np.savez_compressed(OUT / (sid + "_constraints.npz"),
                            qp_matrix=A, qp_lower=lo, qp_upper=up, qp_tolerance=tol,
                            frozen_ramp_matrix=RA, frozen_ramp_lower=RL, frozen_ramp_upper=RU,
                            frozen_ramp_tolerance=RT, producer_candidate=candidate)
        write(sid + "_diagnosis.json", case)
        summaries.append({"scenario_id": sid, "failure_time_s": s["time_s"],
            "classification": case["status"], "qp_row_count": len(lo),
            "qp_plus_frozen_ramp_row_count": len(RL),
            "conflicting_planner_index": j, "endpoint_velocity_gap_rad_s": float(gap[j]),
            "tolerance_adjusted_gap_rad_s": float(exact_gap),
            "phase_one_minimum_uniform_violation": qp_analysis["phase_one_minimax"][
                "minimum_uniform_absolute_violation_lp"],
            "producer_candidate_max_violation": candidate_qp["max_violation"],
            "diagnosis_file": sid + "_diagnosis.json"})
        print(json.dumps(summaries[-1], ensure_ascii=True), flush=True)
    # Re-read every input after diagnosis; ignored output is excluded by design.
    check("all_producer_inputs_unchanged", all(sha(ROOT / p["path"]) == p["sha256"] for p in inputs))
    check("all_frozen_sources_unchanged", all(sha(ROOT / p["path"]) == p["sha256"] for p in source_checks))
    check("all_model_assets_unchanged", all(sha(p["path"]) == p["sha256"] for p in model_assets))
    check("head_unchanged", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                   text=True).strip() == head)
    write("input_manifest.json", {"producer_inputs": inputs, "frozen_source_files": source_checks,
                                  "model_source_assets": model_assets})
    final = {"schema": "research_velocity_failure_linear_feasibility_diagnosis_v1",
        "evidence_valid": True, "diagnostic_execution_completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial_produced_commit": TRIAL_COMMIT, "auditor_source_commit": head,
        "auditor_script_sha256": sha(__file__), "trial_report_sha256": sha(TRIAL / "report.json"),
        "trial_report_passed": False, "trial_report_complete": False,
        "producer_manifest_items_verified": len(producer_manifest),
        "source_inventory_files_verified": len(source_checks),
        "model_source_assets_verified": len(model_assets),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "mujoco": mujoco.__version__, "platform": platform.platform()},
        "lp_method": "scipy.optimize.linprog(method='highs')", "lp_options": options,
        "original_qp_config": cfg,
        "constraint_completeness": "All 38 saved current rows, 38 lookahead rows, "
        "17 saved velocity-box rows, and 10 additionally reconstructed endpoint-domain rows "
        "form the actual 103-row QP operator. Separate combined system includes step 0..10 "
        "frozen current-row and velocity-box tests from the original action contract.",
        "independence": "No producer QP/admission/solver/action-validator function imported or "
        "called. Named model/spec compilation supplies joint addresses and pinned model "
        "parameters only. No geometry query or physics integration performed.",
        "counts": {"frozen_failure_cases": 3, "complete_qp_linear_infeasible": 3,
                   "original_tolerance_infeasible": 3, "exact_scalar_contradictions": 3,
                   "frozen_ramp_combined_infeasible": 3, "feasible_witnesses": 0},
        "cases": summaries, "checks": CHECKS,
        "interpretation": "At all three recorded terminal states the unchanged endpoint "
        "domain restriction and acceleration velocity box conflict beyond the original "
        "1e-4 tolerances. The iteration-limit code alone did not distinguish this, but "
        "independent scalar certificates and HiGHS show these are infeasible declared "
        "linear constraints, not merely an unlocated feasible solution. Increasing the "
        "iteration budget alone cannot make these recorded polytopes feasible.",
        "claim_limits": "This finite diagnosis does not prove that every 2x trajectory is "
        "impossible, identify a safe alternative trajectory, certify nonlinear ten-step "
        "execution or tracking, establish delay/model robustness, or meet the old wall "
        "C11/hardware goal. The full 2x cohort remains failed and incomplete."}
    write("report.json", final)
    manifest = {p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
                for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write("manifest.json", manifest)
    print(json.dumps({"evidence_valid": True, "checks": len(CHECKS),
                      "manifest_items": len(manifest), "cases": len(summaries)}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        if not (OUT / "report.json").exists():
            write("report.json", {"evidence_valid": False, "diagnostic_execution_completed": False,
                                  "error_type": type(e).__name__, "error": str(e), "checks": CHECKS})
        raise
