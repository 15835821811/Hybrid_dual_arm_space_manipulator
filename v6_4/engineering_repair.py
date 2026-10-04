"""Attributed task projection, separate from the primary raw comparison.

This finite engineering stage consumes an immutable joint-spline proposal.
It changes only free controls, uses private nominal prediction and never steps
physics.  A repaired proposal still needs the unchanged full shared gate and
the original torque executor; it is not direct model success or a certificate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_4.contracts import TrajectoryProposal
from v6_4.proposal_gate import gate_proposal
from v6_4.reference_adapter import scenario_from_task
from v6_4.run_planning import prepare_provider, save_reference
from v6_4.run_teacher_comparison import SCORE_DEFINITION, static_score
from v6_4.teacher_planner import TeacherPlanner, load_bootstrap
from v6_4.trajectory_codec import CubicBSplineCodec

ROOT = Path(__file__).resolve().parents[1]
REPAIR_TAG = "task_projection_v1"
REPAIR_TAG_V2 = "task_native_projection_v2"
BUDGET = {"task_projection_outer_rounds": 8, "ik_max_nfev_per_requirement_time": 35,
          "geometry_optimization_rounds": 0, "maximum_candidate_count": 8,
          "gate_state_count": 1351, "gate_period_s": .02,
          "physics_period_s": .002, "duration_s": 27., "minimum_clearance_m": .005}
BUDGET_V2 = {**BUDGET, "post_geometry_task_projection_outer_rounds": 8,
             "geometry_optimization_rounds": 8, "geometry_line_search_steps": 4,
             "geometry_active_witnesses_per_round": 8, "geometry_stride": 10,
             "native_gradient_queries_per_round_budget": 272}


def _version_contract(version):
    if version == "v1":
        return REPAIR_TAG, dict(BUDGET)
    if version == "v2":
        return REPAIR_TAG_V2, dict(BUDGET_V2)
    raise ValueError("repair version must be explicitly v1 or v2; no other budget supported")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _manifest(directory):
    return {p.relative_to(directory).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def _source_records():
    # Archive complete local Python bytes, including transitive imports.  Output
    # scripts are evidence, not production code, and are deliberately excluded.
    paths = []
    for package in ("v6_4", "v6_lite", "model_test"):
        paths.extend(p for p in (ROOT / package).rglob("*.py")
                     if "output" not in p.relative_to(ROOT / package).parts)
    return {p.relative_to(ROOT).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
            for p in sorted(paths)}


def _asset_records(spec):
    return {str(path.resolve()): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in spec._source_assets()}


def _save_snapshot(directory, sources):
    for relative, identity in sources.items():
        content = (ROOT / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise RuntimeError("source changed while saving repair snapshot")
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)


def _domain_checks(task, codec, controls, spec):
    samples = codec.sample(controls, np.arange(1351) * .02)
    domain = default_continuum_model_spec(spec)
    return {
        "finite_full_controls": bool(np.all(np.isfinite(controls))),
        "fixed_C0_C1": bool(np.array_equal(controls[:2], codec.fixed_controls)),
        "control_point_position_domain": bool(np.all(controls >= spec.planner_lower)
                                              and np.all(controls <= spec.planner_upper)),
        "sampled_position_domain": bool(np.all(samples["q"] >= spec.planner_lower)
                                        and np.all(samples["q"] <= spec.planner_upper)),
        "original_pcc_work_domain": bool(np.all(samples["q"][:, :10] >= domain.work_domain_lower_rad)
                                         and np.all(samples["q"][:, :10] <= domain.work_domain_upper_rad)),
        "original_reference_velocity_limits": bool(np.all(abs(samples["dq"]) <= spec.planner_velocity_limits + 1e-12)),
        "original_reference_acceleration_limits": bool(np.all(abs(samples["ddq"]) <= spec.planner_acceleration_limits + 1e-12)),
        "fixed_initial_q_dq": bool(np.allclose(samples["q"][0], task.initial_planner_q, atol=1e-12, rtol=0.)
                                    and np.allclose(samples["dq"][0], task.initial_planner_dq, atol=1e-12, rtol=0.)),
    }


class _ProjectionPlanner(TeacherPlanner):
    """Use the frozen task optimizer, journaling even a partial IK failure."""
    def __init__(self, spec, seed, journal, *, geometry_rounds=0):
        super().__init__(spec, seed=seed, max_starts=1, optimization_rounds=8,
                         ik_max_evaluations=35, geometry_optimization_rounds=geometry_rounds,
                         geometry_line_search_steps=4, geometry_stride=10)
        self.journal = journal
        self.anchor_calls = []
        self.projection_stage = "task_projection"
        self.geometry_journal = journal.with_name("native_geometry_calls.jsonl")
        self.geometry_calls = []

    def _anchor_configuration(self, model, predicted_qpos, guess, points):
        row = {"call_index": len(self.anchor_calls), "stage": self.projection_stage,
               "point_ids": [p.point_id for p in points],
               "time_s": float(points[0].time_s), "status": "STARTED"}
        self.anchor_calls.append(row)
        started = time.perf_counter()
        try:
            value, report = super()._anchor_configuration(model, predicted_qpos, guess, points)
            row.update(report, call_status="COMPLETED", returned_q=value.tolist())
            return value, report
        except Exception as error:
            row.update(call_status="FAILED", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            row["wall_s"] = time.perf_counter() - started
            # An append-only record survives a later IK/prediction exception.
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _native_geometry(self, task, prediction, model, pairs):
        row = {"call_index": len(self.geometry_calls), "kind": "coarse_geometry",
               "status": "STARTED", "native_distance_queries": None}
        self.geometry_calls.append(row)
        started = time.perf_counter()
        try:
            result = super()._native_geometry(task, prediction, model, pairs)
            row.update(status="COMPLETED", result=result, native_distance_queries=result["query_count"])
            return result
        except Exception as error:
            row.update(status="FAILED", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            row["wall_s"] = time.perf_counter() - started
            with self.geometry_journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _geometry_gradient(self, model, predicted_qpos, pair):
        row = {"call_index": len(self.geometry_calls), "kind": "native_gradient",
               "status": "STARTED", "native_distance_queries": None,
               "geom_a": pair.geom_a_name, "geom_b": pair.geom_b_name,
               "predicted_qpos_sha256": hashlib.sha256(np.asarray(predicted_qpos).tobytes()).hexdigest()}
        self.geometry_calls.append(row)
        started = time.perf_counter()
        try:
            gradient, result = super()._geometry_gradient(model, predicted_qpos, pair)
            row.update(status="COMPLETED", result=result, gradient_17d=gradient.tolist(),
                       native_distance_queries=result["native_distance_queries"])
            return gradient, result
        except Exception as error:
            row.update(status="FAILED", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            row["wall_s"] = time.perf_counter() - started
            with self.geometry_journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")


def _repair_one(task, raw, directory, spec, codec, version):
    tag, budget = _version_contract(version)
    row = {"input_proposal_sha256": raw.sha256() if raw is not None else None,
           "origin": raw.origin if raw is not None else None, "failure": None,
           "raw_gate_passed": False, "raw_proposal_passed": False,
           "repaired_full_gate_passed": False, "repaired_raw_gate_passed": False,
           "eligible_for_selection": False, "score": None, "score_components": None,
           "repaired_proposal_path": None, "repaired_proposal_sha256": None,
           "postprocessing": [tag], "repair_version": version, "fallback_used": False, "physics_steps": 0,
           "cost": {}, "status": "NOT_STARTED"}
    started = time.perf_counter()
    phase = "generation"
    planner = None
    try:
        if raw is None:
            row["status"] = "GENERATION_FAILED_NO_PROPOSAL"
            row["failure"] = {"phase": phase, "message": "Absent input retains this candidate slot"}
            return row
        _write(directory / "input_raw_proposal.json", raw.to_dict())
        controls = codec.decode_free(raw.free_controls)
        np.savez_compressed(directory / "input_raw_control_points.npz", control_points=controls,
                            controls_free=raw.free_controls)
        if raw.task_id != task.task_id or raw.task_sha256 != task.sha256():
            raise ValueError("input proposal differs from frozen task")
        if raw.postprocessing or raw.origin == "fallback":
            raise ValueError("this stage requires a raw, non-fallback input proposal")
        phase = "raw_nominal_prediction"
        tick = time.perf_counter()
        provider, verifier = prepare_provider(task, controls, spec)
        row["cost"]["raw_prediction_wall_s"] = time.perf_counter() - tick
        save_reference(provider, directory / "input_raw_nominal_prediction.npz")
        _write(directory / "input_raw_reference_identity.json", provider.metadata)
        phase = "raw_gate"
        tick = time.perf_counter()
        raw_gate = gate_proposal(task, raw, spec, provider=provider, output_path=directory / "input_raw_gate.json")
        row["cost"]["raw_gate_wall_s"] = time.perf_counter() - tick
        row["raw_gate_passed"], row["raw_proposal_passed"] = bool(raw_gate["passed"]), bool(raw_gate["raw_passed"])
        tick = time.perf_counter()
        row["raw_score"], row["raw_score_components"] = static_score(task, provider.prediction)
        row["cost"]["raw_scoring_wall_s"] = time.perf_counter() - tick
        phase = "task_projection"
        initial = mujoco.MjData(verifier.model)
        initial.qpos[:], initial.qvel[:], initial.ctrl[:] = task.initial_qpos, task.initial_qvel, 0.
        mujoco.mj_forward(verifier.model, initial)
        planner = _ProjectionPlanner(spec, raw.seed, directory / "ik_calls.jsonl",
                                     geometry_rounds=budget["geometry_optimization_rounds"])
        planner.projection_stage = "pre_geometry_task_projection" if version == "v2" else "task_projection"
        tick = time.perf_counter()
        try:
            repaired_controls, logs = planner._optimize(task, codec, controls.copy(), None, None,
                                                       verifier.model, initial, scenario_from_task(task))
        finally:
            row["cost"]["task_projection_wall_s"] = time.perf_counter() - tick
        _write(directory / "optimization_rounds.json", logs)
        if len(logs) != 8:
            raise RuntimeError("task projection did not complete the declared eight rounds")
        if version == "v2":
            row["cost"]["pre_geometry_task_projection_wall_s"] = row["cost"]["task_projection_wall_s"]
            np.savez_compressed(directory / "pre_geometry_control_points.npz", control_points=repaired_controls)
            phase = "native_geometry_optimization"
            tick = time.perf_counter()
            try:
                repaired_controls, geometry_log = planner._optimize_geometry(task, codec, repaired_controls,
                    verifier.model, initial, scenario_from_task(task), verifier.pairs)
            finally:
                row["cost"]["native_geometry_optimization_wall_s"] = time.perf_counter() - tick
            _write(directory / "geometry_optimization.json", geometry_log)
            row["cost"]["geometry_private_prediction_count"] = geometry_log["private_prediction_count"]
            row["cost"]["geometry_native_distance_queries"] = geometry_log["native_distance_queries"]
            row["cost"]["geometry_private_configuration_perturbations"] = geometry_log["private_configuration_perturbations"]
            np.savez_compressed(directory / "post_geometry_control_points.npz", control_points=repaired_controls)
            phase = "post_geometry_task_projection"
            planner.projection_stage = phase
            tick = time.perf_counter()
            try:
                repaired_controls, refinement = planner._optimize(task, codec, repaired_controls, None, None,
                    verifier.model, initial, scenario_from_task(task))
            finally:
                row["cost"]["post_geometry_task_projection_wall_s"] = time.perf_counter() - tick
            _write(directory / "post_geometry_task_projection_rounds.json", refinement)
            if len(refinement) != 8:
                raise RuntimeError("post-geometry projection did not complete the declared eight rounds")
        repaired = TrajectoryProposal.from_controls(task, codec.encode_free(repaired_controls),
            origin=raw.origin, seed=raw.seed, postprocessing=(tag,), metadata={
                "repair": tag, "repair_version": version, "budget": budget, "input_proposal_sha256": raw.sha256(),
                "input_raw_metadata": raw.metadata, "fallback_used": False,
                "geometry_optimization_used": version == "v2", "deadline_shift_s": 0., "terminal_settling_added": False})
        _write(directory / "repaired_proposal.json", repaired.to_dict())
        np.savez_compressed(directory / "repaired_control_points.npz", control_points=repaired_controls,
                            controls_free=repaired.free_controls)
        row["repaired_proposal_path"] = "repaired_proposal.json"
        row["repaired_proposal_sha256"] = repaired.sha256()
        row["modification"] = {"free_control_l2_rad": float(np.linalg.norm(repaired.free_controls - raw.free_controls)),
            "full_control_l2_rad": float(np.linalg.norm(repaired_controls - controls)),
            "control_max_abs_rad": float(np.max(abs(repaired_controls - controls))),
            "fixed_C0_C1_change_max_rad": float(np.max(abs(repaired_controls[:2] - controls[:2])))}
        phase = "repaired_nominal_prediction"
        tick = time.perf_counter()
        repaired_provider, _ = prepare_provider(task, repaired_controls, spec)
        row["cost"]["repaired_prediction_wall_s"] = time.perf_counter() - tick
        save_reference(repaired_provider, directory / "repaired_nominal_prediction.npz")
        _write(directory / "repaired_reference_identity.json", repaired_provider.metadata)
        row["domain_checks"] = _domain_checks(task, codec, repaired_controls, spec)
        phase = "repaired_full_gate"
        tick = time.perf_counter()
        gate = gate_proposal(task, repaired, spec, provider=repaired_provider, output_path=directory / "repaired_full_gate.json")
        row["cost"]["repaired_full_gate_wall_s"] = time.perf_counter() - tick
        row["repaired_full_gate_passed"], row["repaired_raw_gate_passed"] = bool(gate["passed"]), bool(gate["raw_passed"])
        if gate["raw_passed"]:
            raise RuntimeError("tagged engineering proposal was incorrectly attributed to raw success")
        tick = time.perf_counter()
        row["score"], row["score_components"] = static_score(task, repaired_provider.prediction)
        row["cost"]["repaired_scoring_wall_s"] = time.perf_counter() - tick
        row["modification"]["path_and_base_metrics_before"] = row["raw_score_components"]
        row["modification"]["path_and_base_metrics_after"] = row["score_components"]
        row["eligible_for_selection"] = bool(gate["passed"] and all(row["domain_checks"].values()))
        row["status"] = "ADMISSIBLE_POSTPROCESSED_PROPOSAL" if row["eligible_for_selection"] else "REPAIR_REJECTED"
    except Exception as error:
        row["status"] = "REPAIR_EVIDENCE_OR_OPTIMIZATION_FAILURE"
        row["failure"] = {"phase": phase, "type": type(error).__name__, "message": str(error),
                          "traceback": traceback.format_exc()}
        row["eligible_for_selection"] = False
    finally:
        row["cost"]["total_wall_s"] = time.perf_counter() - started
        calls = planner.anchor_calls if planner is not None else []
        row["cost"]["anchor_calls_started"] = len(calls)
        row["cost"]["anchor_calls_completed"] = sum(c.get("call_status") == "COMPLETED" for c in calls)
        row["cost"]["reported_ik_nfev"] = sum(c.get("nfev", 0) for c in calls)
        row["cost"]["nfev_scope"] = "SciPy nfev excludes some numerical-Jacobian residual calls; not an exact native-FK count"
        geometry_calls = planner.geometry_calls if planner is not None else []
        known = sum(c["native_distance_queries"] for c in geometry_calls if c["native_distance_queries"] is not None)
        complete = all(c["status"] == "COMPLETED" for c in geometry_calls)
        row["cost"]["journaled_geometry_native_queries_known_completed"] = known
        row["cost"]["journaled_geometry_native_queries"] = known if complete else None
        row["cost"]["geometry_query_count_complete"] = complete
        row["cost"]["geometry_calls_started"] = len(geometry_calls)
    return row


def evaluate_engineering_candidate_set(task, raw_proposals_or_None, output, *, spec=None, version="v1"):
    """Repair 1..8 ordered slots and select using only the common static score.

    K1 is always slot zero, not the first survivor. K8 uses at most eight slots.
    Every slot receives the same fixed projection budget; raw selector code is
    untouched. No actual outcome or future actual trajectory is an input.
    """
    tag, budget = _version_contract(version)
    invocation_started = time.perf_counter()
    proposals = list(raw_proposals_or_None)
    if not 1 <= len(proposals) <= 8:
        raise ValueError("engineering candidate count must be 1..8 without dropping failure slots")
    if any(p is not None and not isinstance(p, TrajectoryProposal) for p in proposals):
        raise TypeError("immutable TrajectoryProposal or None is required in every slot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = _source_records()
    _write(output / "source_before.json", sources)
    _save_snapshot(output / "source_snapshot", sources)
    _write(output / "task.json", task.to_dict())
    _write(output / "protocol.json", {"budget": budget, "score_definition": SCORE_DEFINITION,
        "repair_tag": tag, "repair_version": version, "raw_comparison_unchanged": True, "fallback_permitted": False,
        "future_actual_trace_used": False, "terminal_settling_added": False, "deadline_shift_s": 0.,
        "scope": "finite task-anchor projection; acceptance established only by unchanged full gate and original domain/rate limits"})
    spec = default_v6_lite_robot_spec() if spec is None else spec
    assets = _asset_records(spec)
    _write(output / "model_assets_before.json", assets)
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    shared_setup_wall_s = time.perf_counter() - invocation_started
    started = time.perf_counter()
    records = []
    for index, proposal in enumerate(proposals):
        directory = output / f"candidate_{index:03d}"
        directory.mkdir()
        row = _repair_one(task, proposal, directory, spec, codec, version)
        row["candidate_index"] = index
        if row["repaired_proposal_path"] is not None:
            row["repaired_proposal_path"] = (directory.relative_to(output) / row["repaired_proposal_path"]).as_posix()
        _write(directory / "record.json", row)
        records.append(row)
        print(json.dumps({"candidate": index, "status": row["status"], "full_gate_passed": row["repaired_full_gate_passed"],
                          "raw_passed": row["repaired_raw_gate_passed"], "wall_s": row["cost"]["total_wall_s"]}), flush=True)
    after, assets_after = _source_records(), _asset_records(spec)
    _write(output / "source_after.json", after)
    _write(output / "model_assets_after.json", assets_after)
    frozen = sources == after and assets == assets_after
    eligible = [row for row in records if row["eligible_for_selection"]] if frozen else []
    chosen = min(eligible, key=lambda r: (r["score"], r["candidate_index"])) if eligible else None
    report = {"schema": "v6_4_engineering_candidate_selection_v1", "task_id": task.task_id,
        "task_sha256": task.sha256(), "records": records, "candidate_count": len(proposals),
        "selectedIndex": chosen["candidate_index"] if chosen else None,
        "selectedIndexK1": 0 if frozen and records[0]["eligible_for_selection"] else None,
        "sources_unchanged": sources == after, "model_assets_unchanged": assets == assets_after,
        "evidence_valid": frozen and all(row["failure"] is None or row["status"] == "GENERATION_FAILED_NO_PROPOSAL" for row in records),
        "repair_used": True, "repair_tag": tag, "repair_version": version, "budget": budget,
        "score_definition": SCORE_DEFINITION, "selection_tie_break": "lowest candidate_index",
        "selection_uses_actual_outcomes": False, "fallback_used": False, "physics_steps": 0,
        "total_screen_wall_s": time.perf_counter() - started,
        "shared_setup_wall_s": shared_setup_wall_s,
        "total_invocation_wall_s": time.perf_counter() - invocation_started,
        "prefix_K1_screen_wall_s": records[0]["cost"]["total_wall_s"],
        "cost_scope": "all retained slots and shared setup counted in total invocation; K1 is prefix cost excluding shared setup; projection is explicit engineering cost",
        "raw_success_claimed": False, "closed_loop_success_established": False}
    _write(output / "selection.json", report)
    _write(output / "artifact_manifest.json", _manifest(output))
    return report


def repair_proposal(task, raw_proposal_or_None, output, *, spec=None, version="v1"):
    """Single-candidate wrapper returning the same retained selection schema."""
    return evaluate_engineering_candidate_set(task, [raw_proposal_or_None], output, spec=spec, version=version)


def load_selected_proposal(report, output, *, K=8):
    """Load the selected tagged proposal with byte and task identity checks."""
    if K not in (1, 8):
        raise ValueError("only predeclared K1/K8 selection is supported")
    output = Path(output)
    if report != json.loads((output / "selection.json").read_text(encoding="utf-8")):
        raise ValueError("selection report differs from immutable saved report")
    if not report["sources_unchanged"] or not report["model_assets_unchanged"]:
        raise ValueError("source or model freeze changed during repair")
    sources = json.loads((output / "source_before.json").read_text(encoding="utf-8"))
    if sources != _source_records():
        raise ValueError("current source differs from saved repair source freeze")
    assets = json.loads((output / "model_assets_before.json").read_text(encoding="utf-8"))
    if any(_sha(path) != identity["sha256"] or Path(path).stat().st_size != identity["bytes"]
           for path, identity in assets.items()):
        raise ValueError("model asset bytes changed since repair")
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    for relative, identity in manifest.items():
        artifact = (output / relative).resolve()
        if (not artifact.is_relative_to(output.resolve()) or _sha(artifact) != identity["sha256"]
                or artifact.stat().st_size != identity["bytes"]):
            raise ValueError("repair artifact bytes changed")
    index = report["selectedIndexK1" if K == 1 else "selectedIndex"]
    if index is None:
        return None
    row = report["records"][index]
    path = (output / row["repaired_proposal_path"]).resolve()
    if not path.is_relative_to(output.resolve()) or not row["eligible_for_selection"]:
        raise ValueError("selected proposal path or eligibility invalid")
    if _sha(path) != manifest[path.relative_to(output.resolve()).as_posix()]["sha256"]:
        raise ValueError("selected proposal bytes changed")
    proposal = TrajectoryProposal.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if (proposal.sha256() != row["repaired_proposal_sha256"] or proposal.task_sha256 != report["task_sha256"]
            or proposal.postprocessing != (report["repair_tag"],) or proposal.origin == "fallback"):
        raise ValueError("selected tagged proposal identity changed")
    return proposal


def fixed_joint_prior_seed(task, output):
    """Fit the declared old scene04 joint prior, never the fixed Cartesian arm.

    This is an explicit traditional joint-prior comparator seed. No task IK,
    repair, gate, execution, success label or target future actual is supplied.
    Pass the returned proposal to the same attributed repair API afterwards.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "v6_lite/output/runs/research_acceptance_01/simulation/traces/v6_lite_scenario_04.npz"
    started = time.perf_counter()
    sources = _source_records()
    _write(output / "source_before.json", sources)
    _save_snapshot(output / "source_snapshot", sources)
    times, q, lineage = load_bootstrap(source)
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    # Start elimination uses this task's declared initial state, not the old
    # trace's first post-integration state or a copied physical state.
    times, q = np.r_[0., times], np.vstack([task.initial_planner_q, q])
    controls = codec.fit(times, q)
    metadata = {"comparator": "fixed_joint_prior", "original_fixed_cartesian_baseline": False,
                "bootstrap_source": lineage, "bootstrap_is_success_label": False,
                "intrinsic_seed_operations": ["constrained_32_control_cubic_spline_fit_with_fixed_C0_C1"],
                "future_actual_trace_used": False, "gate_or_actual_success_established": False,
                "fitting_wall_s": time.perf_counter() - started,
                "seed_module_sha256": _sha(Path(__file__)),
                "initial_state_source": "immutable TaskSpec; no old physical state transplanted"}
    proposal = TrajectoryProposal.from_controls(task, codec.encode_free(controls), origin="fixed_joint_prior",
                                               seed=0, metadata=metadata)
    _write(output / "proposal.json", proposal.to_dict())
    np.savez_compressed(output / "control_points.npz", control_points=controls, controls_free=proposal.free_controls)
    _write(output / "source_lineage.json", metadata)
    after = _source_records()
    _write(output / "source_after.json", after)
    _write(output / "freeze_status.json", {"sources_unchanged": sources == after})
    _write(output / "artifact_manifest.json", _manifest(output))
    if sources != after:
        raise RuntimeError("fixed joint-prior seed source freeze changed")
    return proposal
