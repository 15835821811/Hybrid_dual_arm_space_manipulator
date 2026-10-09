"""Private nominal closed-loop screening and fresh native quality extraction.

The original runner constructs a new model, MjData, controller, reference state
and QP for each call. Search never replays another candidate's torque/state.
Independent torque/interval/whole-body validation remains the final executor's
unchanged default, and is explicitly NOT_RUN for search screening.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

import mujoco
import numpy as np

from .route_optimizer_protocol import read, sha, write, digest, related_pairs
from .route_quality_dataset import window_mask, path_and_base_metrics


def command_metrics(trace, ticks, windows):
    times = np.asarray(trace["task_time"])[:ticks]
    nominal = np.asarray(trace["task_qp_box_nominal_velocity"])[:ticks]
    selected = np.asarray(trace["task_qp_selected_velocity"])[:ticks]
    if nominal.shape != (ticks, 17) or selected.shape != nominal.shape or times.shape != (ticks,):
        raise ValueError("consumed command vector shapes differ")
    if not all(np.isfinite(a).all() for a in (times, nominal, selected)):
        raise ValueError("consumed command vector is nonfinite")
    if not np.array_equal(selected, np.asarray(trace["task_selected_command"])[:ticks]):
        raise ValueError("selected diagnostic differs from consumed 17D command")
    du = nominal - selected
    squared = np.sum(du ** 2, axis=1)
    scalar = np.asarray(trace["task_avoidance_intervention"])[:ticks]
    if scalar.shape != (ticks,) or not np.allclose(np.sqrt(squared), scalar, atol=1e-12, rtol=0.):
        raise ValueError("original scalar norm differs from source command vectors")
    result = {}
    for name, intervals in windows.items():
        mask = window_mask(times, intervals)
        if mask.any():
            total = float(np.sqrt(np.mean(squared[mask])))
            continuum = float(np.sqrt(np.mean(np.sum(du[mask, :10] ** 2, axis=1))))
            rigid = float(np.sqrt(np.mean(np.sum(du[mask, 10:] ** 2, axis=1))))
        else:
            total = continuum = rigid = None
        result[name] = total
        result[name + "_10D"] = continuum
        result[name + "_7D"] = rigid
        result[name + "_sample_count"] = int(mask.sum())
    result["command_definition"] = "box-clipped nominal 17D velocity minus actually consumed selected 17D command"
    result["component_square_identity_checked"] = all(
        result[name] is None or abs(result[name] ** 2 - result[name + "_10D"] ** 2 - result[name + "_7D"] ** 2) < 1e-12
        for name in windows)
    return result, {"time": times, "box_nominal_17D": nominal, "selected_17D": selected,
                    "du_17D": du, "du_10D": du[:, :10], "du_7D": du[:, 10:]}


def fresh_quality(task, plan, trace_path, frozen, output, *, evidence_root, replay_path=None):
    """Forward geometry at saved native 2ms states; no new integration.

    Search uses its own saved physical states. Final actual uses its independent
    saved-torque replay, whose original safety gates have already been retained.
    """
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    from v6_lite.hierarchical_qp import joint_addresses, free_joint_slices
    from v6_lite.runtime_command import model_id
    from .proposal_gate import requirement_results
    from .residual_execution import consumed_reference_identity_binding
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with np.load(trace_path, allow_pickle=False) as f:
        trace = {key: f[key] for key in f.files}
    n = len(trace["torque"]); ticks = n // 10
    if n % 10 or not n:
        raise ValueError("quality needs at least one complete consumed original ramp")
    if not np.array_equal(trace["initial_qpos"], task.initial_qpos) or not np.array_equal(trace["initial_qvel"], task.initial_qvel):
        raise ValueError("candidate initial physical state differs from frozen Task")
    if not np.allclose(trace["time"], np.arange(1, n + 1) * .002, atol=1e-9, rtol=0.):
        raise ValueError("native saved physical clock differs")
    identity = consumed_reference_identity_binding(plan, trace, ticks)
    if not identity["available"] or not identity["passed"]:
        raise ValueError("missing consumed reference identity")
    windows = {"I_support": frozen["W_support"], "I_route_key_legacy": frozen["W_key"], "I_full": [[0., 27.]]}
    metrics, vectors = command_metrics(trace, ticks, windows)
    np.savez_compressed(output / "command_vectors.npz", **vectors)
    if replay_path is None:
        qpos = np.concatenate((np.asarray(task.initial_qpos)[None], trace["actual_full_qpos"]))
        qvel = np.concatenate((np.asarray(task.initial_qvel)[None], trace["actual_full_qvel"]))
        times = np.r_[0., trace["time"]]
        state_source = "own_saved_native_prediction_states"
    else:
        with np.load(replay_path, allow_pickle=False) as f:
            qpos, qvel, times = f["qpos"], f["qvel"], f["time"]
        state_source = "independent_saved_torque_actual_replay"
        if (np.max(np.abs(qpos[1:] - trace["actual_full_qpos"])) > 1e-9
                or np.max(np.abs(qvel[1:] - trace["actual_full_qvel"])) > 1e-8):
            raise ValueError("actual replay state mismatch")
    if qpos.shape != (n + 1, 81) or qvel.shape != (n + 1, 79) or not all(np.isfinite(a).all() for a in (qpos, qvel)):
        raise ValueError("native state array is invalid")
    spec = default_v6_lite_robot_spec()
    if spec.runtime_contract_sha256() != task.model_contract_sha256:
        raise ValueError("quality model contract differs")
    verifier, pairs = related_pairs(spec, task)
    pair_ids = [[p.geom_a_name, p.geom_b_name] for p in pairs]
    if pair_ids != [list(p) for p in frozen["related_pair_ids"]] or verifier._pair_policy_sha256 != frozen["pair_policy_sha256"]:
        raise ValueError("quality related pair scope/policy changed")
    model = verifier.model; model.geom_contype[:] = 0; model.geom_conaffinity[:] = 0
    compiled = model_id(model, spec.runtime_contract_sha256())
    certificate_count = 0
    for path in sorted(Path(evidence_root).rglob("timing/*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            certificate = __import__("json").loads(line).get("certificate")
            if certificate:
                if certificate["source_model_hash"] != compiled:
                    raise ValueError("candidate command certificate binds a different compiled model")
                certificate_count += 1
    if not certificate_count:
        raise ValueError("no command certificate binds the quality model")
    data = mujoco.MjData(model)
    qids, vids = joint_addresses(model, spec)
    base_ids, _ = free_joint_slices(model, spec.base_joint_name)
    target_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.target_free_joint_name)
    bodies = {arm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, getattr(spec, arm + "_tip_body_name"))
              for arm in ("rigid", "continuum")}
    bodies["target"] = int(model.jnt_bodyid[target_joint])
    values = {name: [] for name in ("time", "q", "dq", "qpos", "qvel", "base_pose",
        "target_position", "target_rotation", "rigid_position", "rigid_rotation", "continuum_position", "continuum_rotation")}
    support = window_mask(times, frozen["W_support"])
    minima = np.full(len(times), np.nan); witness = None; query_count = 0
    best = float("inf")
    for i, t in enumerate(times):
        data.qpos[:] = qpos[i]; data.qvel[:] = qvel[i]; data.time = float(t)
        mujoco.mj_forward(model, data)
        for name, value in (("time", float(t)), ("q", spec.decode_position(data.qpos[qids])),
                ("dq", spec.decode_velocity(data.qvel[vids])), ("qpos", data.qpos),
                ("qvel", data.qvel), ("base_pose", data.qpos[base_ids])):
            values[name].append(np.copy(value))
        for arm, body in bodies.items():
            rotation = data.xmat[body].reshape(3, 3).copy(); position = data.xpos[body].copy()
            if arm == "continuum":
                position += rotation @ np.array([.0475, 0., 0.])
            values[arm + "_position"].append(position); values[arm + "_rotation"].append(rotation)
        if support[i]:
            local = float("inf")
            for pair in pairs:
                fromto = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(model, data, pair.geom_a, pair.geom_b, 2.5, fromto))
                query_count += 1
                if not np.isfinite(distance):
                    raise ValueError("nonfinite related pair signed distance")
                local = min(local, distance)
                if distance < best:
                    best = distance
                    witness = {"state_index": i, "time_s": float(t), "pair": asdict(pair),
                               "signed_distance_m": distance, "fromto": fromto.tolist(),
                               "censored_lower_bound": distance >= 2.5 - 1e-12}
            minima[i] = local
    state = {key: np.asarray(value) for key, value in values.items()}
    path = path_and_base_metrics(state, frozen["W_support"])
    metrics.update(L_full=path["continuum_path_length_m"], L_support=path["continuum_route_window_path_length_m"],
        base_translation_peak_m=path["base_translation_peak_m"], base_rotation_peak_rad=path["base_rotation_peak_rad"],
        d_support=best if witness is not None else None, clearance_status="MEASURED" if witness else "NOT_MEASURED",
        clearance_is_censored_lower_bound=bool(witness and witness["censored_lower_bound"]),
        clearance_witness=witness, related_pair_count=len(pairs), clearance_query_count=query_count,
        source_compiled_model_sha256=compiled, command_certificates_checked=certificate_count,
        consumed_reference_binding=identity, native_state_count=len(times), saved_horizon_s=float(times[-1]),
        torque_saturation_count=int(np.sum(trace["torque_saturation_count"])),
        peak_torque_abs_nm=float(np.max(np.abs(trace["torque"]))), state_source=state_source)
    requirements = requirement_results(task, state)
    np.savez_compressed(output / "fresh_states.npz", **state)
    np.savez_compressed(output / "support_clearance.npz", time=times[support], signed_distance_m=minima[support])
    write(output / "quality.json", {"metrics": metrics, "task_requirements": requirements,
        "trace_sha256": sha(trace_path), "replay_sha256": sha(replay_path) if replay_path else None,
        "physics_steps_added": 0, "continuous_time_safety": "NOT_ESTABLISHED"})
    return metrics, requirements


def seal(directory):
    directory = Path(directory)
    write(directory / "manifest.json", {p.relative_to(directory).as_posix(): sha(p)
        for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"})


def verify_seal(directory):
    directory = Path(directory)
    for relative, expected in read(directory / "manifest.json").items():
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or sha(path) != expected:
            raise ValueError("retained evidence hash mismatch: " + relative)


class NominalCandidateEvaluator:
    def __init__(self, run, task, frozen):
        from .route_optimizer_protocol import verify_frozen
        self.run, self.task, self.frozen = Path(run).resolve(), task, frozen
        verify_frozen(self.run)
        self.execution_identity = {"source_identity_sha256": sha(self.run / "source_identity.json"),
            "config_sha256": sha(self.run / "frozen_execution_config.json"),
            "run_config_sha256": sha(self.run / "frozen_run_config.json"),
            "task_sha256": task.sha256(), "model_contract_sha256": task.model_contract_sha256}

    def __call__(self, plan, candidate_id):
        from .route_optimizer_protocol import verify_frozen
        from .conditional_execution import ExecutionCostLedger
        from . import residual_execution as execution
        from v6_lite.run_v6_lite import V6LiteRunConfig, run_synchronous_scenario, default_v6_lite_robot_spec
        from v6_lite.hierarchical_qp import HierarchicalQPConfig
        from .reference_adapter import scenario_from_task
        from .task_anchored_reference import reference_precheck, TaskAnchoredResidualReferenceProvider
        from .evaluate_planning import _execution_trace_checks
        verify_frozen(self.run)
        directory = self.run / "planning" / self.task.task_id / "predictions" / candidate_id
        key = digest({**self.execution_identity, "plan_sha256": plan.sha256()})
        if directory.exists():
            if not (directory / "result.json").exists():
                raise RuntimeError("unfinished consumed candidate retained; inspect and record recovery, never retry: " + str(directory))
            verify_seal(directory)
            retained = read(directory / "result.json")
            if retained["content_key"] != key:
                raise ValueError("retained candidate model/QP/reference/config identity mismatch")
            return retained
        directory.mkdir(parents=True)
        write(directory / "started.json", {"content_key": key, "argv": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()), "started_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate_id, "initial_state_sha256": digest([self.task.initial_qpos, self.task.initial_qvel]),
            "state_isolation": "new runner/model/MjData/controller/provider/integrator/QP and ADMM cache per candidate"})
        write(directory / "plan.json", plan.to_dict())
        ledger = ExecutionCostLedger(); start = time.perf_counter()
        result = {"candidate_id": candidate_id, "content_key": key, "prediction_rollout_started": False,
            "prediction_steps": 0, "prediction_task_passed": False, "online_guards_passed": False,
            "prediction_admissible": False, "prediction_metrics": None, "status": "STARTED",
            "formal_actual_validation": "NOT_RUN", "independent_validation": "NOT_RUN_SEARCH_SCREENING",
            "plan_sha256": plan.sha256(), "execution_identity": self.execution_identity}
        try:
            check = reference_precheck(self.task, plan, cartesian_speed_limit_m_s=.24)
            write(directory / "reference_precheck.json", check)
            if not check["passed"]:
                result["status"] = "REFERENCE_PRECHECK_REJECTED"
            elif not self.frozen["geometry_precheck_passed"]:
                result["status"] = "TASK_INITIAL_OR_ANCHOR_PRECHECK_REJECTED"
            else:
                cfg = V6LiteRunConfig(**read(self.run / "frozen_run_config.json"))
                cfg.validate()
                qp = HierarchicalQPConfig(**read(self.run / "frozen_execution_config.json"))
                spec = default_v6_lite_robot_spec(); scene = scenario_from_task(self.task)
                physical = directory / "private_prediction"
                metadata = execution.start_run(physical, run_config=cfg, qp_config=qp, spec=spec, scenarios=(scene,))
                result["prediction_rollout_started"] = True
                scene_result = None; failure = None
                with ledger.installed():
                    try:
                        with ledger.scope("prediction"):
                            scene_result = run_synchronous_scenario(spec, cfg, qp, scene, physical / "traces",
                                reference_provider=TaskAnchoredResidualReferenceProvider(self.task, plan),
                                execution_diagnostics=True, diagnostic_obstacle_name=self.frozen["obstacle_name"])
                        write(physical / "runner_result.json", scene_result)
                        trace_path = physical / "traces" / (scene.scenario_id + ".npz")
                    except Exception as error:
                        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                        write(physical / "execution_failure.json", failure)
                        partials = sorted((physical / "failures").glob("*partial_trace.npz"))
                        trace_path = partials[-1] if partials else None
                    if trace_path is not None:
                        with np.load(trace_path, allow_pickle=False) as f:
                            trace = {key: f[key] for key in f.files}
                        n = len(trace["torque"]); result["prediction_steps"] = n
                        result["trace_relative_path"] = trace_path.relative_to(directory).as_posix()
                        if n:
                            with ledger.scope("prediction_quality"):
                                metrics, requirements = fresh_quality(self.task, plan, trace_path, self.frozen,
                                    directory / "quality", evidence_root=physical)
                            execution_check = _execution_trace_checks(self.task, trace, spec, qp)
                            result.update(prediction_metrics=metrics, task_requirements=requirements,
                                online_execution_checks=execution_check,
                                prediction_task_passed=bool(n == 13500 and requirements["passed"]),
                                online_guards_passed=bool(not failure and execution_check["passed"]))
                            result["prediction_admissible"] = bool(result["prediction_task_passed"] and result["online_guards_passed"])
                    if failure:
                        result["execution_failure"] = failure
                        # The original runner's structured failure artifact is
                        # the distinction between guard refusal and tool error.
                        receipts = list((physical / "failures").glob("*.json"))
                        result["failure_receipts"] = [p.relative_to(directory).as_posix() for p in receipts]
                        if not receipts:
                            result["tool_error"] = failure
                    result["status"] = ("PREDICTION_ADMISSIBLE" if result["prediction_admissible"] else
                        "EXECUTION_REFUSED" if failure and not result.get("tool_error") else
                        "TOOL_ERROR" if result.get("tool_error") else "PREDICTION_TASK_UNMET")
                execution.finish_run(physical, metadata, passed=result["prediction_admissible"], summary={"role": "private_search_prediction", "status": result["status"]})
        except Exception as error:
            result.update(status="TOOL_ERROR", tool_error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        result["costs"] = ledger.to_dict()
        result["costs"]["prediction_physics_steps"] = ledger.physics_steps("prediction")
        result["elapsed_wall_s"] = time.perf_counter() - start
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        verify_frozen(self.run)
        write(directory / "result.json", result); seal(directory)
        return result
