"""Read-only B.3 route quality and task-local elite labels.

The frozen QP already logs ||u_selected - u_unconstrained||. Its nominal
diagnostic is the Hessian optimum after the original velocity-bound clipping,
before clearance constraints. No raw proposal difference is used here.
This module never steps physics, changes z, or changes a controller result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA = "v64_b3_route_quality_v1"
GATE_NAMES = ("task_requirements", "execution_contract", "independent_interval",
              "native_geometry", "reference_binding")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def intervals_array(intervals_s):
    """The frozen union, never inferred from a nonzero z or observed outcome."""
    intervals = np.asarray(intervals_s, dtype=float)
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
            or not np.isfinite(intervals).all() or np.any(intervals[:, 0] < 0)
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(intervals[1:, 0] < intervals[:-1, 1])):
        raise ValueError("route intervals must be finite, ordered, positive-width and disjoint")
    return intervals


def window_mask(times, intervals_s, *, endpoint=True):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("time vector must be finite")
    selected = np.zeros(len(times), dtype=bool)
    for a, b in intervals_array(intervals_s):
        selected |= (times >= a - 1e-10) & (times <= b + 1e-10 if endpoint else times < b - 1e-10)
    return selected


def command_intervention_metrics(trace: Mapping, consumed_ticks: int, intervals_s):
    """RMS of existing vector-norm diagnostic equals sqrt(mean(||du||^2))."""
    times = np.asarray(trace["task_time"], dtype=float)[:consumed_ticks]
    norms = np.asarray(trace["task_avoidance_intervention"], dtype=float)[:consumed_ticks]
    if (times.shape != (consumed_ticks,) or norms.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(norms).all()
            or np.any(norms < 0)):
        raise ValueError("consumed task-clock / QP intervention diagnostic is invalid")
    selected = window_mask(times, intervals_s)
    def rms(values):
        return float(np.sqrt(np.mean(values ** 2))) if len(values) else None
    return {"I_route_rad_s": rms(norms[selected]), "I_full_rad_s": rms(norms),
            "route_planning_sample_count": int(np.sum(selected)),
            "consumed_planning_sample_count": consumed_ticks,
            "definition": "sqrt(mean(||u_unconstrained-u_selected||^2))",
            "producer_field": "task_avoidance_intervention",
            "source_vectors_available": False,
            "source_scope": "saved scalar norm of 17-D QP nominal diagnostic minus actual selected command",
            "nominal_diagnostic": "np.clip(-_unconstrained_solve(hessian,linear),velocity_lower,velocity_upper); original producer semantics",
            "producer": "v6_lite/hierarchical_qp.py:HierarchicalVelocityQP.solve -> unconstrained_to_command_norm; v6_lite/run_v6_lite.py -> task_avoidance_intervention",
            "proposal_minus_selected_is_used": False,
            "window_sampling": "50 Hz consumed planning ticks in frozen closed [start,end] interval union"}


def safety_eligibility(attempt, evaluation):
    gates = {name: evaluation.get(name, {}).get("passed") is True for name in GATE_NAMES}
    complete = bool(evaluation.get("complete") is True and attempt.get("actual_steps") == 13500)
    eligible = bool(complete and attempt.get("full_task_success") is True
                    and evaluation.get("full_task_success") is True
                    and evaluation.get("evidence_valid") is True
                    and not attempt.get("execution_failure") and not attempt.get("pipeline_failure")
                    and attempt.get("fallback_used") is False and all(gates.values()))
    return {"full_complete": complete, "full_task_and_original_safety_passed": eligible,
            "gates": gates, "path_quality_comparison_eligible": eligible,
            "historical_runtime_reported_passed": evaluation.get("runtime_reported_passed"),
            "historical_strict_curve_is_success_gate": False}


def quality_replay_evidence(attempt, evaluation, evaluation_path):
    """Distinguish a captured evaluator failure from missing/tampered evidence.

    A failed replay legitimately leaves no fresh state file. Its producer still
    saves the failure report and an exact manifest of what it did produce. A
    file that the manifest claims, or any file required by a successful
    evaluation, may never be silently treated as an unavailable metric.
    """
    success_claimed = bool(attempt.get("full_task_success") is True
                           or attempt.get("full_27s_success") is True
                           or evaluation.get("full_task_success") is True
                           or evaluation.get("task_success") is True
                           or evaluation.get("evidence_valid") is True)
    captured_failure = evaluation.get("evidence_valid") is False and bool(evaluation.get("errors"))
    if evaluation_path is None:
        if success_claimed:
            raise ValueError("successful slot/evaluation lacks independent evaluation path")
        return None, [], {"reason": "INDEPENDENT_EVALUATION_PATH_UNAVAILABLE",
                          "scope": "no independent state evidence; no complete or prefix quality is credited"}
    evaluation_dir = Path(evaluation_path).resolve().parent
    manifest_path = evaluation_dir / "manifest.json"
    manifest = read(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("independent evaluation manifest must map files to digests")
    for name, digest in manifest.items():
        artifact = (evaluation_dir / name).resolve()
        if evaluation_dir not in artifact.parents:
            raise ValueError("independent evaluation manifest path escapes its directory")
        if not artifact.is_file():
            raise ValueError("declared independent evaluation artifact is missing: " + name)
        if sha(artifact) != digest:
            raise ValueError("independent evaluation artifact digest differs: " + name)
    replay_path = evaluation_dir / "fresh_replay.npz"
    if replay_path.exists():
        if "fresh_replay.npz" not in manifest:
            raise ValueError("present fresh replay lacks independent manifest digest")
        native = evaluation.get("native_geometry")
        whole_body = native.get("whole_body") if isinstance(native, dict) else None
        pair_policy = whole_body.get("pair_policy_sha256") if isinstance(whole_body, dict) else None
        if not pair_policy:
            if captured_failure and not success_claimed:
                return None, [manifest_path, replay_path], {
                    "reason": "INDEPENDENT_GEOMETRY_BINDING_UNAVAILABLE_AFTER_EVALUATION_FAILURE",
                    "evaluation_errors": evaluation["errors"],
                    "scope": "fresh states were saved but native geometry/pair-policy evidence was not completed; no complete or prefix quality is credited"}
            raise ValueError("independent replay lacks required native geometry/pair-policy evidence")
        return replay_path, [manifest_path, replay_path], None
    if success_claimed:
        raise ValueError("successful independent evaluation lacks fresh replay")
    if captured_failure:
        return None, [manifest_path], {
            "reason": "INDEPENDENT_REPLAY_UNAVAILABLE_AFTER_EVALUATION_FAILURE",
            "evaluation_errors": evaluation["errors"],
            "scope": "producer saved evaluation failure without a fresh replay; no complete or prefix quality is credited"}
    raise ValueError("missing fresh replay is not explained by a captured evaluation failure")


def path_and_base_metrics(state, intervals_s):
    times = np.asarray(state["time"], dtype=float)
    position = np.asarray(state["continuum_position"], dtype=float)
    base = np.asarray(state["base_pose"], dtype=float)
    if (position.shape != (len(times), 3) or base.shape != (len(times), 7)
            or not len(times) or not np.isfinite(position).all() or not np.isfinite(base).all()):
        raise ValueError("fresh path/base state shape or finiteness invalid")
    route_edges = np.zeros(max(0, len(times) - 1), dtype=bool)
    for a, b in intervals_array(intervals_s):
        route_edges |= (times[:-1] >= a - 1e-10) & (times[1:] <= b + 1e-10)
    distances = np.linalg.norm(np.diff(position, axis=0), axis=1)
    displacement = np.linalg.norm(base[:, :3] - base[0, :3], axis=1)
    quaternions = base[:, 3:]
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    if np.any(quaternion_norm < 1e-12):
        raise ValueError("zero base quaternion")
    quaternions = quaternions / quaternion_norm[:, None]
    angles = 2 * np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0., 1.))
    route = window_mask(times, intervals_s, endpoint=True)
    return {"continuum_path_length_m": float(np.sum(distances)),
            "continuum_route_window_path_length_m": float(np.sum(distances[route_edges])),
            "base_translation_peak_m": float(np.max(displacement)),
            "base_rotation_peak_rad": float(np.max(angles)),
            "base_translation_route_peak_m": float(np.max(displacement[route])) if route.any() else None,
            "base_rotation_route_peak_rad": float(np.max(angles[route])) if route.any() else None,
            "saved_horizon_s": float(times[-1]), "saved_native_state_count": len(times)}


def local_continuum_clearance(task, state, obstacle_name, intervals_s, evaluation, actual_dir):
    """Additional same-state geometry only; no integration, no global-min proxy."""
    import mujoco
    from dataclasses import asdict
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
    from v6_4.reference_adapter import scenario_from_task
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    from v6_lite.runtime_command import model_id
    spec = default_v6_lite_robot_spec()
    if spec.runtime_contract_sha256() != task.model_contract_sha256:
        raise ValueError("quality robot contract differs from task")
    scene = scenario_from_task(task)
    names = [o.name for o in scene.obstacles]
    if names.count(obstacle_name) != 1:
        raise ValueError("route obstacle must identify exactly one frozen scene sphere")
    verifier = WholeBodyCollisionVerifier(spec, scene.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    model = verifier.model
    native = evaluation["native_geometry"]
    if verifier._pair_policy_sha256 != native["whole_body"]["pair_policy_sha256"]:
        raise ValueError("quality pair policy differs from independent safety evaluation")
    continuum_bodies = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
    geom_name = f"v5_workspace_sphere_{names.index(obstacle_name):03d}_{obstacle_name}"
    pairs = [p for p in verifier.pairs if p.pair_class == "arm_obstacle"
             and p.geom_b_name == geom_name and int(model.geom_bodyid[p.geom_a]) in continuum_bodies]
    if not pairs:
        raise ValueError("empty declared continuum/route-obstacle pair set")
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    compiled = model_id(model, spec.runtime_contract_sha256())
    historic = Path(actual_dir) / "historical_metric_observations.json"
    model_identity_sources = {}
    if historic.exists():
        if read(historic)["execution_contract"]["source_compiled_model_sha256"] != compiled:
            raise ValueError("quality scene model differs from actual compiled scene identity")
        model_identity_sources[str(historic.resolve())] = sha(historic)
    certificate_count = 0
    for timing in sorted((Path(actual_dir) / "timing").glob("*.jsonl")):
        for line in timing.read_text(encoding="utf8").splitlines():
            certificate = json.loads(line).get("certificate")
            if certificate:
                if certificate["source_model_hash"] != compiled:
                    raise ValueError("quality scene model differs from actual command certificate")
                certificate_count += 1
        model_identity_sources[str(timing.resolve())] = sha(timing)
    if not certificate_count:
        raise ValueError("actual scene identity lacks command-certificate evidence")
    times = np.asarray(state["time"])
    data = mujoco.MjData(model)
    minima = np.empty(len(times))
    witnesses = []
    global_min = float("inf")
    for index, t in enumerate(times):
        data.qpos[:] = state["qpos"][index]
        data.qvel[:] = state["qvel"][index]
        data.time = float(t)
        mujoco.mj_forward(model, data)
        distances = [float(mujoco.mj_geomDistance(model, data, p.geom_a, p.geom_b, 2.5, np.zeros(6))) for p in pairs]
        if not np.isfinite(distances).all():
            raise ValueError("nonfinite continuum/route-obstacle signed distance")
        which = int(np.argmin(distances))
        minima[index] = distances[which]
        if minima[index] < global_min:
            global_min = float(minima[index])
            witnesses.append({"state_index": index, "time_s": float(t), "signed_distance_m": global_min,
                              "pair": asdict(pairs[which])})
    route = window_mask(times, intervals_s, endpoint=True)
    return {"obstacle_name": obstacle_name, "obstacle_geom_name": geom_name,
            "pair_scope": "continuum collision geoms versus the declared route sphere only",
            "pair_count": len(pairs), "pairs": [asdict(p) for p in pairs],
            "full_saved_horizon_minimum_m": float(np.min(minima)),
            "route_window_minimum_m": float(np.min(minima[route])) if route.any() else None,
            "minimum_witness": witnesses[-1], "native_period_s": .002,
            "query_count": len(times) * len(pairs), "state_count": len(times),
            "minimum_censored": bool(np.min(minima) >= 2.5 - 1e-12),
            "source_compiled_model_sha256": compiled,
            "scene_identity_definition": "runtime_command.model_id binds robot contract and mutable compiled dynamics/geometry; not a serialized MJB-file hash",
            "scene_identity_sources": model_identity_sources,
            "actual_model_certificates_checked": certificate_count,
            "model_contract_sha256": task.model_contract_sha256,
            "model_source_bundle_sha256": spec.source_bundle_sha256(),
            "new_physics_steps": 0, "new_optimizer_updates": 0,
            "clearance_query_role": "additional route-quality description; original independent safety gates unchanged",
            "continuous_time_certified": False}, minima


def build_route_quality(attempt_dir, route_intervals_s, obstacle_name, output_dir, *, candidate_name=None):
    """Bind a frozen slot to complete metrics or an isolated failed-prefix record."""
    from .task_protocol import TaskSpec
    from .task_anchored_reference import TaskAnchoredResidualPlan
    attempt_dir, output_dir = Path(attempt_dir).resolve(), Path(output_dir).resolve()
    if output_dir == attempt_dir or attempt_dir in output_dir.parents:
        raise ValueError("derived quality output must be separate from the actual evidence directory")
    intervals = intervals_array(route_intervals_s)
    result = read(attempt_dir / "attempt_result.json")
    task = TaskSpec.from_dict(read(attempt_dir / "task.json"))
    if task.sha256() != result["task_sha256"]:
        raise ValueError("quality task differs from attempted task")
    plan_path = attempt_dir / "plan.json"
    plan = TaskAnchoredResidualPlan.from_dict(read(plan_path)) if plan_path.exists() else None
    if plan is not None:
        if plan.definition["task_id"] != task.task_id or plan.definition["task_sha256"] != task.sha256():
            raise ValueError("quality residual definition differs from attempted task")
        if result.get("plan_file_sha256") and sha(plan_path) != result["plan_file_sha256"]:
            raise ValueError("quality plan file differs from attempted input")
        declared = np.asarray(plan.definition["intervals_s"])[plan.interval_mask]
        if any(not any(np.array_equal(interval, allowed) for allowed in declared) for interval in intervals):
            raise ValueError("frozen route window must comprise complete predeclared allowed intervals, including for z0")
    evaluation_path = Path(result["evaluation_path"]) if result.get("evaluation_path") else None
    evaluation = read(evaluation_path) if evaluation_path else {}
    if evaluation_path and sha(evaluation_path) != result["evaluation_sha256"]:
        raise ValueError("quality evaluation digest differs from slot")
    safety = safety_eligibility(result, evaluation)
    report = {"schema": SCHEMA, "slot_id": result["slot_id"], "task_id": task.task_id,
              "task_sha256": task.sha256(), "split": task.split,
              "candidate_name": candidate_name, "status": result["status"],
              "route_intervals_s": intervals.tolist(), "route_obstacle_name": obstacle_name,
              "z_m": plan.z_m.tolist() if plan else None, "safety": safety,
              "independent_safety_and_task_metrics": {name: evaluation.get(name) for name in GATE_NAMES},
              "full_metrics": None, "failed_prefix_metrics": None,
              "quality_label_eligible": False, "metric_unavailable": None,
              "sources": {}, "deployment": "NOT_MET"}
    sources = [attempt_dir / "attempt_result.json", attempt_dir / "task.json"]
    if plan_path.exists():
        sources.append(plan_path)
    if evaluation_path:
        sources.append(evaluation_path)
    trace_path = Path(result["trace_path"]) if result.get("trace_path") else None
    if result.get("actual_steps", 0) and trace_path is None:
        raise ValueError("slot declares consumed actual steps without a source trace")
    costs = {"actual_physics_steps": int(result.get("actual_steps", 0)),
             "independent_saved_torque_replay_steps": int(evaluation.get("replayed_physics_steps", 0)),
             "independent_robot_target_geometry_queries": int(evaluation.get("native_geometry", {}).get("robot_target_500hz", {}).get("query_count", 0)),
             "independent_whole_body_geometry_queries": int(evaluation.get("native_geometry", {}).get("whole_body", {}).get("query_count", 0)),
             "additional_route_quality_geometry_queries": 0,
             "additional_route_quality_physics_steps": 0,
             "private_preview": {"status": "NOT_COUNTED_BY_THIS_MODULE", "steps": None,
                                 "reason": "retain executor instrumentation; trace alone omits rejected/unconsumed previews"}}
    if trace_path and result.get("actual_steps", 0):
        if sha(trace_path) != result["trace_sha256"]:
            raise ValueError("quality trace digest differs from slot")
        sources.append(trace_path)
        replay_path, replay_sources, unavailable = quality_replay_evidence(result, evaluation, evaluation_path)
        sources.extend(replay_sources)
        report["metric_unavailable"] = unavailable
    else:
        replay_path = None
        report["metric_unavailable"] = {"reason": "NO_CONSUMED_ACTUAL_TRACE",
                                        "scope": "no consumed trajectory; no complete or prefix quality is credited"}
    if replay_path is not None:
        with np.load(trace_path, allow_pickle=False) as data:
            trace = {key: data[key] for key in ("task_time", "task_avoidance_intervention", "torque")}
        n = len(trace["torque"])
        if n != result["actual_steps"] or n % 10:
            raise ValueError("consumed complete ramps differ from attempted physics steps")
        with np.load(replay_path, allow_pickle=False) as data:
            state = {key: data[key] for key in ("time", "continuum_position", "base_pose", "qpos", "qvel")}
        metrics = {**command_intervention_metrics(trace, n // 10, intervals),
                   **path_and_base_metrics(state, intervals)}
        geometry, minima = local_continuum_clearance(task, state, obstacle_name, intervals, evaluation, attempt_dir / "actual")
        metrics["continuum_route_obstacle_clearance"] = geometry
        costs["additional_route_quality_geometry_queries"] = geometry["query_count"]
        output_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output_dir / "route_clearance.npz", time=state["time"], continuum_obstacle_minimum_m=minima)
        if safety["path_quality_comparison_eligible"] and metrics["I_route_rad_s"] is not None:
            report["full_metrics"] = metrics
            report["quality_label_eligible"] = True
        else:
            report["failed_prefix_metrics"] = metrics
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    report["costs"] = costs
    report["sources"] = {str(path): sha(path) for path in sources}
    write(output_dir / "route_quality.json", report)
    write(output_dir / "manifest.json", {p.name: sha(p) for p in sorted(output_dir.iterdir()) if p.is_file()})
    return report


def quality_filter(records):
    """Task-local elite sets; failures never enter successful denoising labels."""
    if len({r["slot_id"] for r in records}) != len(records):
        raise ValueError("duplicate teacher slot identifiers")
    grouped = {}
    for record in records:
        grouped.setdefault(record["task_id"], []).append(record)
    tasks = []
    for task_id, rows in sorted(grouped.items()):
        eligible = [r for r in rows if r.get("quality_label_eligible") is True
                    and r.get("safety", {}).get("full_task_and_original_safety_passed") is True
                    and r.get("full_metrics", {}).get("I_route_rad_s") is not None]
        for row in eligible:
            value = row["full_metrics"]["I_route_rad_s"]
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid successful route intervention")
        minimum = min((r["full_metrics"]["I_route_rad_s"] for r in eligible), default=None)
        threshold = minimum + max(.1 * minimum, .001) if minimum is not None else None
        elites = [r for r in eligible if r["full_metrics"]["I_route_rad_s"] <= threshold]
        rank = sorted(eligible, key=lambda r: (r["full_metrics"]["I_route_rad_s"],
                      r["full_metrics"]["continuum_path_length_m"], r["slot_id"]))
        tasks.append({"task_id": task_id, "attempt_count": len(rows), "successful_safe_count": len(eligible),
                      "I_min_rad_s": minimum, "elite_limit_rad_s": threshold,
                      "elite_slot_ids": [r["slot_id"] for r in rank if r in elites],
                      "ranked_successful_slot_ids": [r["slot_id"] for r in rank],
                      "best_slot_id": rank[0]["slot_id"] if rank else None,
                      "failure_slot_ids": [r["slot_id"] for r in rows if r not in eligible]})
    elite_ids = {slot for task in tasks for slot in task["elite_slot_ids"]}
    return {"schema": "v64_b3_quality_elite_set_v1", "formula": "I_route <= I_min + max(0.1*I_min,0.001 rad/s)",
            "scope": "better references among four declared teacher candidates per task; not globally optimal trajectories",
            "tie_break": "I_route, then continuum path length, then fixed slot_id lexical order",
            "zero_residual_allowed": True, "coordinate_averaging_used": False,
            "tasks": tasks, "elite_records": [r for r in records if r["slot_id"] in elite_ids]}


def data_qualification(elite_set, *, required_val_tasks=2):
    if not isinstance(required_val_tasks, int) or required_val_tasks < 1:
        raise ValueError("the frozen VAL task requirement must be a positive integer")
    elites = elite_set["elite_records"]
    train = {r["task_id"] for r in elites if r["split"].lower() == "train"}
    val = {r["task_id"] for r in elites if r["split"].lower() == "val"}
    passed = len(train) >= 4 and len(val) >= required_val_tasks
    return {"status": "DATA_QUALIFIED_FOR_THIS_PILOT" if passed else "DATA_INSUFFICIENT_FOR_THIS_PILOT",
            "eligible_train_task_count": len(train), "eligible_val_task_count": len(val),
            "minimum_train_tasks": 4, "minimum_val_tasks": required_val_tasks,
            "val_has_evaluation_labels": bool(val),
            "additional_teacher_attempts_authorized": False}


def pilot_discriminability(records, paired_task_ids):
    """Exactly z0/z+/z- per side; opposite nonzero improvements on both sides."""
    if len(paired_task_ids) != 2 or len(set(paired_task_ids)) != 2 or len(records) != 6:
        raise ValueError("pilot requires one pair and exactly six terminal slots")
    outcomes = []
    for task_id in paired_task_ids:
        rows = {r["candidate_name"]: r for r in records if r["task_id"] == task_id}
        if set(rows) != {"z0", "z+", "z-"} or sum(r["task_id"] == task_id for r in records) != 3:
            raise ValueError("pilot per-side z0/z+/z- slots are incomplete or duplicated")
        valid = {name: r["quality_label_eligible"] for name, r in rows.items()}
        nonzero = [name for name in ("z+", "z-") if valid[name]]
        preferred = None
        nonzero_gap = None
        if len(nonzero) == 1:
            preferred = nonzero[0]
        elif len(nonzero) == 2:
            plus = rows["z+"]["full_metrics"]["I_route_rad_s"]
            minus = rows["z-"]["full_metrics"]["I_route_rad_s"]
            nonzero_gap = abs(plus - minus)
            if nonzero_gap > 1e-12:
                preferred = "z+" if plus < minus else "z-"
        favorable = []
        evidence = []
        for direction, opposite in (("z+", "z-"), ("z-", "z+")):
            if not valid[direction]:
                continue
            for comparator in ("z0", opposite):
                if not valid[comparator]:
                    favorable.append(direction)
                    evidence.append({"direction": direction, "comparator": comparator, "criterion": "A_complete_vs_failure"})
                else:
                    better, baseline = rows[direction]["full_metrics"]["I_route_rad_s"], rows[comparator]["full_metrics"]["I_route_rad_s"]
                    difference = baseline - better
                    relative = difference / baseline if baseline > 0 else 0.
                    if difference >= .001 - 1e-12 and relative >= .1 - 1e-12:
                        favorable.append(direction)
                        evidence.append({"direction": direction, "comparator": comparator, "criterion": "B_intervention_gain",
                                         "absolute_difference_rad_s": difference, "relative_reduction": relative})
        qualifying_preferred = preferred if preferred in favorable else None
        opposite = "z-" if preferred == "z+" else "z+" if preferred == "z-" else None
        direct_opposite = [e for e in evidence if e["direction"] == preferred and e["comparator"] == opposite]
        outcomes.append({"task_id": task_id, "favorable_nonzero_directions": sorted(set(favorable)),
                         "preferred_nonzero_direction": preferred,
                         "preferred_qualifying_direction": qualifying_preferred,
                         "nonzero_intervention_absolute_difference_rad_s": nonzero_gap,
                         "preferred_direct_opposite_meets_A_or_B": bool(direct_opposite),
                         "preferred_direct_opposite_evidence": direct_opposite,
                         "direction_tie_tolerance_rad_s": 1e-12,
                         "direction_tie_does_not_use_path_or_slot": True,
                         "complete_safe_by_candidate": valid, "evidence": evidence})
    directions = [r["preferred_qualifying_direction"] for r in outcomes]
    identifiable = directions in (["z+", "z-"], ["z-", "z+"])
    return {"route_value_identifiable": identifiable,
            "status": "ROUTE_VALUE_IDENTIFIABLE" if identifiable else "ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION",
            "sides": outcomes, "thresholds": {"relative_reduction": .1, "absolute_difference_rad_s": .001},
            "training_authorized_by_pilot": identifiable, "additional_pilot_slots_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--route-intervals-json", required=True, help="JSON file containing the frozen key route interval list; identical for every method on a task")
    parser.add_argument("--obstacle", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_route_quality(args.attempt, read(args.route_intervals_json), args.obstacle, args.output, candidate_name=args.candidate)
    print(json.dumps({"slot_id": result["slot_id"], "quality_label_eligible": result["quality_label_eligible"]}))


if __name__ == "__main__":
    main()
