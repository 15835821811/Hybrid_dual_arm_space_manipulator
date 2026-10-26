"""Finite intrinsic Teacher prior construction, never an execution permission.

The common center uses one frozen historical scene04 bootstrap.  It may fail
Task or safety checks and still define residual coordinates.  Every generated
proposal must subsequently pass the unchanged independent proposal gate.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import Bounds, minimize

from v6_4.contracts import TrajectoryProposal
from v6_4.proposal_gate import gate_proposal, requirement_results
from v6_4.reference_adapter import SplineReferenceProvider, scenario_from_task
from v6_4.run_teacher_comparison import static_score
from v6_4.teacher_planner import TeacherPlanner, load_bootstrap
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_lite.continuum_model_spec import default_continuum_model_spec

ROOT = Path(__file__).resolve().parents[1]
PRIOR_ID = "finite_teacher_arm_anchor_rate_prior_early_stop_v1"
SOURCE_SCOPE = "three-package runtime PY; excludes uncalled tests/output and uncalled prior_diffusion.py; other runtime code conservatively included"
BUDGET = {"seed": 64, "starts": 8, "pre_anchor_rate_rounds": 8,
          "post_geometry_anchor_rate_rounds": 8, "maximum_anchor_rate_rounds": 16,
          "ik_nfev_per_anchor": 35, "geometry_rounds": 8,
          "geometry_lines_per_round": 4, "geometry_witnesses_per_round": 8,
          "geometry_stride": 10, "rate_coordinate_dimension": 30,
          "rate_maxiter_per_offending_coordinate": 35, "rate_retry_count": 0,
          "rate_grid_states": 1351, "internal_velocity_scale": .95,
          "velocity_scale_scope": "rate_projection_target; intermediate_geometry_uses_original_limits",
          "original_acceleration_limits": [2.5] * 10 + [4.] * 7,
          "anchor_rounds_are_maxima": True,
          "anchor_early_stop": "fresh_private_prediction_original_Task_all_requirements_and_CP_PCC_domain_internal_0.95_dq_original_ddq; geometry_and_final_full_gate_still_required"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _record(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}


def _source_records():
    paths = [p for package in ("v6_4", "v6_lite", "model_test")
             for p in (ROOT / package).rglob("*.py")
             if "output" not in p.relative_to(ROOT / package).parts
             and "tests" not in p.relative_to(ROOT / package).parts
             and not p.name.startswith("test_") and p.name != "prior_diffusion.py"]
    return {p.relative_to(ROOT).as_posix(): _record(p) for p in sorted(paths)}


def rate_project_coordinate(codec, controls, coordinate, anchor_times, anchor_values,
                            *, lower, upper, velocity, acceleration):
    """One strictly convex 30D closest-CP problem; failed iterates are retained.

    Only the original input column is returned on a failed solve. Neither a
    returned SLSQP iterate nor optimizer success is an admission certificate.
    """
    full = codec._controls(controls)
    if not isinstance(coordinate, int) or isinstance(coordinate, bool) or not 0 <= coordinate < 17:
        raise ValueError("coordinate must be an integer in [0,17)")
    scalars = np.asarray([lower, upper, velocity, acceleration], dtype=float)
    if not np.all(np.isfinite(scalars)) or lower >= upper or min(velocity, acceleration) <= 0:
        raise ValueError("finite nonempty position domain and positive rate limits required")
    anchor_times = np.asarray(anchor_times, dtype=float)
    anchor_values = np.asarray(anchor_values, dtype=float)
    if anchor_values.shape != anchor_times.shape or anchor_times.ndim != 1 or not np.all(np.isfinite(anchor_values)):
        raise ValueError("finite aligned coordinate anchors required")
    grid = np.arange(1351) * .02
    d1, d2 = codec.basis(grid, 1), codec.basis(grid, 2)
    matrix = np.vstack([d1[:, 2:], d2[:, 2:]])
    fixed = np.r_[d1[:, :2] @ full[:2, coordinate], d2[:, :2] @ full[:2, coordinate]]
    limits = np.r_[np.full(1351, velocity), np.full(1351, acceleration)]
    low, high = -limits - fixed, limits - fixed
    anchor = codec.basis(anchor_times)
    eq = anchor[:, 2:]
    target = anchor_values - anchor[:, :2] @ full[:2, coordinate]
    if len(eq) and np.linalg.matrix_rank(eq) != len(eq):
        raise ValueError("coordinate anchor rows are rank deficient")
    original = full[2:, coordinate].copy()
    inequalities = np.vstack([matrix, -matrix])
    rhs = np.r_[low, -high]
    constraints = [{"type": "ineq", "fun": lambda x: inequalities @ x - rhs,
                    "jac": lambda x: inequalities}]
    if len(eq):
        constraints.append({"type": "eq", "fun": lambda x: eq @ x - target,
                            "jac": lambda x: eq})
    started = time.perf_counter()
    result = minimize(lambda x: .5 * np.dot(x-original, x-original), original,
                      jac=lambda x: x-original, bounds=Bounds(lower, upper),
                      constraints=constraints, method="SLSQP",
                      options={"maxiter": 35, "ftol": 1e-10, "disp": False})
    candidate = np.asarray(result.x, dtype=float)
    finite = candidate.shape == (30,) and bool(np.all(np.isfinite(candidate)))
    violation = float(max(0., np.max(low-matrix@candidate), np.max(matrix@candidate-high),
                          np.max(lower-candidate), np.max(candidate-upper))) if finite else None
    eq_error = float(np.max(abs(eq@candidate-target))) if finite and len(eq) else (0. if finite else None)
    fixed_domain_ok = bool(np.all(full[:2, coordinate] >= lower) and np.all(full[:2, coordinate] <= upper))
    accepted = bool(result.success and finite and fixed_domain_ok and violation <= 1e-12 and eq_error <= 1e-10)
    report = {"coordinate": coordinate, "accepted": accepted, "optimizer_success": bool(result.success),
              "status": int(result.status), "message": str(result.message), "nit": int(result.nit),
              "nfev": int(result.nfev), "njev": int(result.njev), "maxiter": 35,
              "retry_count": 0, "analytic_objective_and_constraint_jacobians": True,
              "grid_states": 1351, "fixed_domain_ok": fixed_domain_ok,
              "inequality_violation_max": violation, "anchor_error_max_rad": eq_error,
              "anchor_times_s": anchor_times.tolist(), "anchor_values_rad": anchor_values.tolist(),
              "lower_rad": float(lower), "upper_rad": float(upper),
              "internal_velocity_limit_rad_s": float(velocity), "acceleration_limit_rad_s2": float(acceleration),
              "candidate_free_column": candidate.tolist() if finite else None,
              "input_free_column": original.tolist(), "failed_solution_applied": False,
              "wall_s": time.perf_counter()-started}
    return candidate.copy() if accepted else original, report


class FiniteTeacherPrior(TeacherPlanner):
    """Legacy orchestration plus explicit arm-specific anchor/rate operations."""
    def __init__(self, spec):
        super().__init__(spec, seed=64, max_starts=8, optimization_rounds=8,
                         ik_max_evaluations=35, geometry_stride=10,
                         geometry_optimization_rounds=8, geometry_line_search_steps=4)
        domain = default_continuum_model_spec(spec)
        self.lower, self.upper = spec.planner_lower.copy(), spec.planner_upper.copy()
        self.lower[:10] = np.maximum(self.lower[:10], domain.work_domain_lower_rad)
        self.upper[:10] = np.minimum(self.upper[:10], domain.work_domain_upper_rad)
        self.costs = {"anchor_optimization_calls": 0, "anchor_rate_rounds": 0,
                      "ik_calls": 0, "ik_nfev_completed": 0, "ik_nfev_exact": True,
                      "ik_wall_s": 0., "rate_calls": 0, "rate_nit_completed": 0,
                      "rate_nfev_completed": 0, "rate_njev_completed": 0,
                      "rate_evaluations_exact": True, "rate_wall_s": 0.,
                      "native_calls": 0, "native_queries_completed": 0,
                      "native_queries_exact": True, "native_wall_s": 0.,
                      "anchor_private_predictions_started": 0, "physics_steps": 0}
        self.costs["ik_nfev_scope"] = "SciPy least_squares nfev excludes finite-difference residual callbacks and final diagnostic residual calls"

    def _anchor_configuration(self, *args):
        self.costs["ik_calls"] += 1; started = time.perf_counter()
        try:
            value, report = super()._anchor_configuration(*args)
            self.costs["ik_nfev_completed"] += report["nfev"]
            return value, report
        except Exception:
            self.costs["ik_nfev_exact"] = False
            raise
        finally:
            self.costs["ik_wall_s"] += time.perf_counter()-started

    def _native_geometry(self, *args):
        self.costs["native_calls"] += 1; started = time.perf_counter()
        try:
            result = super()._native_geometry(*args)
            self.costs["native_queries_completed"] += result["query_count"]
            return result
        except Exception:
            self.costs["native_queries_exact"] = False
            raise
        finally:
            self.costs["native_wall_s"] += time.perf_counter()-started

    def _geometry_gradient(self, *args):
        self.costs["native_calls"] += 1; started = time.perf_counter()
        try:
            gradient, report = super()._geometry_gradient(*args)
            self.costs["native_queries_completed"] += report["native_distance_queries"]
            return gradient, report
        except Exception:
            self.costs["native_queries_exact"] = False
            raise
        finally:
            self.costs["native_wall_s"] += time.perf_counter()-started

    def _optimize(self, task, codec, controls, prior_times, prior_q, model, initial_data, scenario):
        groups = {}
        for point in task.requirements:
            groups.setdefault(point.time_s, []).append(point)
        times = np.asarray(sorted(groups))
        arm_times = {arm: np.asarray(sorted({p.time_s for p in task.requirements if p.arm == arm}))
                     for arm in ("continuum", "rigid")}
        logs = []
        self.costs["anchor_optimization_calls"] += 1
        for outer in range(8):
            self.costs["anchor_rate_rounds"] += 1
            log = {"round": outer, "ik": [], "rate": [], "completed": False}
            started = time.perf_counter()
            try:
                self.costs["anchor_private_predictions_started"] += 1
                provider = SplineReferenceProvider(task, controls).prepare(self.spec, model, initial_data, scenario)
                anchored = {}
                for t in times:
                    guess = codec.sample(controls, np.array([t]))["q"][0]
                    value, report = self._anchor_configuration(model, provider.prediction["full_qpos"][int(round(t/.02))],
                                                                guess, groups[float(t)])
                    anchored[float(t)] = value
                    log["ik"].append({"time_s": float(t), **report})
                free = codec.encode_free(controls)
                for j in range(17):
                    ts = arm_times["continuum" if j < 10 else "rigid"]
                    if not len(ts):
                        continue
                    basis = codec.basis(ts)
                    matrix = basis[:, 2:]
                    if np.linalg.matrix_rank(matrix) != len(matrix):
                        raise ValueError("required-arm anchor rows are rank deficient")
                    wanted = np.array([anchored[float(t)][j] for t in ts]) - basis[:, :2] @ codec.fixed_controls[:, j]
                    free[:, j] += matrix.T @ np.linalg.solve(matrix@matrix.T, wanted-matrix@free[:, j])
                controls = codec.decode_free(free)
                samples = codec.sample(controls, np.arange(1351)*.02)
                internal_v = .95*self.spec.planner_velocity_limits
                offending = np.flatnonzero(np.any(controls < self.lower, axis=0) | np.any(controls > self.upper, axis=0)
                    | np.any(abs(samples["dq"]) > internal_v+1e-12, axis=0)
                    | np.any(abs(samples["ddq"]) > self.spec.planner_acceleration_limits+1e-12, axis=0))
                for index in offending:
                    j = int(index); ts = arm_times["continuum" if j < 10 else "rigid"]
                    values = np.array([anchored[float(t)][j] for t in ts])
                    self.costs["rate_calls"] += 1; rate_started = time.perf_counter()
                    try:
                        column, report = rate_project_coordinate(codec, controls, j, ts, values,
                            lower=self.lower[j], upper=self.upper[j], velocity=internal_v[j],
                            acceleration=self.spec.planner_acceleration_limits[j])
                        self.costs["rate_nit_completed"] += report["nit"]
                        self.costs["rate_nfev_completed"] += report["nfev"]
                        self.costs["rate_njev_completed"] += report["njev"]
                    except Exception:
                        self.costs["rate_evaluations_exact"] = False
                        raise
                    finally:
                        self.costs["rate_wall_s"] += time.perf_counter()-rate_started
                    log["rate"].append(report)
                    if not report["accepted"]:
                        log["failure"] = {"type": "RateProjectionFailure", "coordinate": j,
                                          "message": "finite solve failed; unaccepted iterate not applied"}
                        break
                    free = codec.encode_free(controls); free[:, j] = column
                    controls = codec.decode_free(free)
                log["completed"] = "failure" not in log
                if log["completed"]:
                    self.costs["anchor_private_predictions_started"] += 1
                    fresh = SplineReferenceProvider(task, controls).prepare(self.spec, model, initial_data, scenario)
                    requirements = requirement_results(task, fresh.prediction)
                    values = codec.sample(controls, np.arange(1351)*.02)
                    checks = {
                        "original_Task_all_requirements": bool(requirements["passed"]),
                        "CP_and_PCC_domain": bool(np.all(controls >= self.lower) and np.all(controls <= self.upper)
                            and np.all(values["q"] >= self.lower) and np.all(values["q"] <= self.upper)),
                        "internal_0_95_velocity": bool(np.all(abs(values["dq"]) <= internal_v+1e-12)),
                        "original_acceleration": bool(np.all(abs(values["ddq"]) <= self.spec.planner_acceleration_limits+1e-12))}
                    log["early_stop"] = {"passed": all(checks.values()), "checks": checks,
                                         "task_requirements": requirements, "geometry_checked_here": False,
                                         "geometry_and_final_full_gate_still_required": True}
            except Exception as error:
                log["failure"] = {"type": type(error).__name__, "message": str(error)}
            log["wall_s"] = time.perf_counter()-started
            logs.append(log)
            if not log["completed"]:
                break
            if log["early_stop"]["passed"]:
                break
        return controls, logs


def task_max_ratio(requirements):
    if not isinstance(requirements, dict) or not requirements.get("requirements"):
        return None
    ratios = []
    for row in requirements["requirements"]:
        if row.get("position_error_m") is None or row.get("orientation_error_rad") is None:
            return None
        ratios.append(max(row["position_error_m"]/row["position_tolerance_m"],
                          row["orientation_error_rad"]/row["orientation_tolerance_rad"]))
    result = max(ratios)
    return float(result) if np.isfinite(result) else None


def select_center(records):
    finite = [r for r in records if r["center_finite"] and r["start_boundary_exact"]]
    passed = [r for r in finite if r["original_gate_passed"] and r.get("static_score") is not None]
    if passed:
        return min(passed, key=lambda r: (r["static_score"], r["attempt_index"]))["attempt_index"]
    if finite:
        return min(finite, key=lambda r: (r["task_max_ratio"] if r["task_max_ratio"] is not None else float("inf"),
                                          r["attempt_index"]))["attempt_index"]
    return None


def build_task_center(task, output, *, spec=None, bootstrap_path=None):
    """Save eight starts and one representation center. Never execute physics."""
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    spec = default_v6_lite_robot_spec() if spec is None else spec
    frozen = ROOT/"v6_lite/output/runs/research_acceptance_01/simulation/traces/v6_lite_scenario_04.npz"
    bootstrap_path = frozen if bootstrap_path is None else Path(bootstrap_path).resolve()
    if bootstrap_path.resolve() != frozen.resolve():
        raise ValueError("this prior identity fixes its generic legacy scene04 bootstrap")
    before = _source_records(); _write(output/"source_before.json", before)
    assets_before = {str(p.resolve()): _record(p) for p in spec._source_assets()}
    _write(output/"model_assets_before.json", assets_before)
    contract_before = spec.runtime_contract_sha256()
    if contract_before != task.model_contract_sha256:
        raise ValueError("prior Task changes the declared nominal model contract")
    times, prior, bootstrap = load_bootstrap(bootstrap_path)
    scenario = scenario_from_task(task)
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=1,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    model = verifier.model; model.geom_contype[:] = 0; model.geom_conaffinity[:] = 0
    initial = mujoco.MjData(model); initial.qpos[:], initial.qvel[:] = task.initial_qpos, task.initial_qvel
    initial.ctrl[:] = 0.; mujoco.mj_forward(model, initial)
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    started = time.perf_counter()
    planner = FiniteTeacherPrior(spec)
    attempts = planner.propose(task, times, prior, bootstrap_metadata=bootstrap,
                      model=model, initial_data=initial, pairs=verifier.pairs)
    planning_wall = time.perf_counter()-started; records = []; gate_queries = 0; gate_wall = 0.; gate_queries_exact = True
    for attempt in attempts:
        directory = output/f"candidate_{attempt.attempt_index:03d}"; directory.mkdir()
        _write(directory/"planning.json", attempt.to_dict())
        controls = attempt.controls
        finite = bool(controls is not None and np.shape(controls) == (32,17) and np.all(np.isfinite(controls)))
        exact = bool(finite and np.array_equal(controls[:2], codec.fixed_controls))
        row = {"attempt_index": attempt.attempt_index, "center_finite": finite,
               "start_boundary_exact": exact, "original_gate_passed": False,
               "task_max_ratio": None, "static_score": None, "failure": None}
        gate = {"passed": False, "raw_passed": False, "status": "PRIOR_NOT_CONSTRUCTIBLE",
                "task_id": task.task_id, "task_sha256": task.sha256()}
        acceleration = {"schema": "teacher_prior_original_reference_ddq_guard_v1",
                        "task_id": task.task_id, "task_sha256": task.sha256(),
                        "passed": False, "status": "NOT_RUN", "physics_steps": 0,
                        "limits_rad_s2": spec.planner_acceleration_limits.tolist(),
                        "numeric_tolerance": 1e-12, "scope": "private supplied 1351 states; not continuous time"}
        if finite and exact:
            np.save(directory/"full.npy", controls, allow_pickle=False)
            np.save(directory/"free.npy", codec.encode_free(controls), allow_pickle=False)
            proposal = TrajectoryProposal.from_controls(task, codec.encode_free(controls), origin="teacher",
                seed=64+attempt.attempt_index, postprocessing=(), metadata={"intrinsic_algorithm": PRIOR_ID,
                "prior_budget": BUDGET, "bootstrap_identity": bootstrap,
                "representation_center_only": True, "planning_metadata": attempt.metadata})
            _write(directory/"proposal.json", proposal.to_dict())
            try:
                provider = SplineReferenceProvider(task, controls).prepare(spec, model, initial, scenario)
                np.savez_compressed(directory/"prediction.npz", **provider.prediction, control_points=controls)
                task_result = requirement_results(task, provider.prediction)
                _write(directory/"task_requirements.json", task_result)
                row["task_max_ratio"] = task_max_ratio(task_result)
                row["static_score"], row["score_components"] = static_score(task, provider.prediction)
                gate = gate_proposal(task, proposal, spec, provider=provider)
                gate_wall += float(gate["elapsed_wall_s"])
                geometry = gate["nominal_screen"].get("geometry")
                if geometry is not None: gate_queries += int(geometry["query_count"])
                if gate.get("errors") or gate["nominal_screen"].get("errors"): gate_queries_exact = False
                # The shared gate does not independently check ddq. Keep the
                # original acceleration guard alongside it, never infer it.
                row["original_acceleration_passed"] = bool(np.all(abs(provider.prediction["ddq"]) <= spec.planner_acceleration_limits+1e-12))
                acceleration.update(passed=row["original_acceleration_passed"], status="COMPLETE",
                    grid_state_count=len(provider.prediction["time"]), period_s=.02,
                    max_abs_by_coordinate_rad_s2=np.max(abs(provider.prediction["ddq"]),axis=0).tolist(),
                    violation_count=int(np.count_nonzero(abs(provider.prediction["ddq"]) > spec.planner_acceleration_limits+1e-12)),
                    time_array_sha256=hashlib.sha256(provider.prediction["time"].tobytes()).hexdigest(),
                    ddq_array_sha256=hashlib.sha256(provider.prediction["ddq"].tobytes()).hexdigest(),
                    reference_sha256=_sha(directory/"prediction.npz"))
                row["original_gate_passed"] = bool(gate["passed"] and row["original_acceleration_passed"])
            except Exception as error:
                row["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                gate_queries_exact = False
                gate = {"passed": False, "raw_passed": False, "status": "PRIOR_EVIDENCE_FAILURE", "failure": row["failure"],
                        "task_id": task.task_id, "task_sha256": task.sha256()}
            _write(directory/"acceleration_guard.json", acceleration)
        _write(directory/"gate.json", gate); _write(directory/"record.json", row); records.append(row)
    chosen = select_center(records)
    row = {"task_id": task.task_id, "task_sha256": task.sha256(), "split": task.split,
           "group_id": task.group_id, "chosen_attempt_index": chosen,
           "center_finite": chosen is not None, "start_boundary_exact": chosen is not None,
           "intrinsic_algorithm": PRIOR_ID, "prior_budget": BUDGET, "bootstrap_identity": bootstrap,
           "source": before, "source_scope": SOURCE_SCOPE, "attempts": records, "costs": {**planner.costs,
           "planning_wall_s": planning_wall, "original_gate_wall_s": gate_wall,
           "original_gate_native_queries_completed": gate_queries,
           "original_gate_native_queries_exact": gate_queries_exact,
           "prediction_count_scope": "anchor calls counted; inherited geometry/provider calls additionally retained in planning logs",
           "total_wall_s": time.perf_counter()-started, "physics_steps": 0},
           "center_is_execution_permission": False, "task_or_safety_failure_does_not_drop_labels": True}
    if chosen is not None:
        directory = output/f"candidate_{chosen:03d}"
        for stem, name in (("center_free", "free.npy"), ("center_full", "full.npy"),
                           ("center_proposal", "proposal.json"), ("prior_only_gate", "gate.json"),
                           ("prior_only_acceleration_guard", "acceleration_guard.json")):
            record = _record(directory/name)
            row.update({stem+"_path": record["path"], stem+"_sha256": record["sha256"], stem+"_bytes": record["bytes"]})
        proposal = TrajectoryProposal.from_dict(json.loads((directory/"proposal.json").read_text(encoding="utf-8")))
        row["center_proposal_file_sha256"] = row["center_proposal_sha256"]
        row["center_proposal_semantic_sha256"] = proposal.sha256()
        row["proposal_sha256"] = proposal.sha256()
        gate = json.loads((directory/"gate.json").read_text(encoding="utf-8"))
        row["prior_only_gate_raw_passed"] = bool(gate["raw_passed"])
        row["prior_only_gate_passed"] = bool(records[chosen]["original_gate_passed"])
        if (directory/"prediction.npz").is_file(): row["provider"] = _record(directory/"prediction.npz")
    after = _source_records(); _write(output/"source_after.json", after)
    assets_after = {str(p.resolve()): _record(p) for p in spec._source_assets()}
    _write(output/"model_assets_after.json", assets_after)
    row["source_unchanged"] = before == after
    row["model_assets_unchanged"] = assets_before == assets_after
    row["model_contract_before"] = contract_before
    row["model_contract_after"] = spec.runtime_contract_sha256()
    _write(output/"center.json", row)
    _write(output/"artifact_manifest.json", {p.relative_to(output).as_posix(): _record(p)
           for p in sorted(output.rglob("*")) if p.is_file()})
    if before != after or assets_before != assets_after or row["model_contract_after"] != contract_before:
        raise RuntimeError("prior source or model changed during construction; evidence retained")
    return row


def build_prior_centers(tasks, output, *, spec=None, bootstrap_path=None):
    tasks = list(tasks)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("nonempty unique predeclared Task cohort required")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    _write(output/"tasks.json", {"tasks": [t.to_dict() for t in tasks]})
    centers = [build_task_center(task, output/f"t{index}", spec=spec, bootstrap_path=bootstrap_path)
               for index, task in enumerate(tasks)]
    manifest = {"schema": "teacher_prior_centers_v1", "intrinsic_algorithm": PRIOR_ID,
                "prior_budget": BUDGET, "centers": centers, "task_count": len(tasks),
                "all_tasks_attempted": len(centers) == len(tasks),
                "all_centers_constructible": all(r["center_finite"] for r in centers),
                "complete": len(centers) == len(tasks), "source_unchanged": all(r["source_unchanged"] for r in centers),
                "physics_steps": 0, "labels_dropped": False, "centers_are_execution_permissions": False}
    _write(output/"manifest.json", manifest)
    if not manifest["all_centers_constructible"]:
        raise RuntimeError("at least one Task has no finite prior center; full mechanism stops without dropping labels")
    return manifest
