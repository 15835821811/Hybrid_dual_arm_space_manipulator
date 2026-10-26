"""Immutable V6.4-A attempts using the existing physical research executor.

The learned or traditional trajectory supplies references only.  This entry
does not write execution qpos/qvel after initialization or produce torque.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_evidence import start_run, finish_run, _write_json
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, default_v6_lite_robot_spec, run_synchronous_scenario,
)
from v6_4.contracts import TrajectoryProposal
from v6_4.reference_adapter import SplineReferenceProvider, scenario_from_task
from v6_4.task_protocol import TaskSpec, freeze_task_suite, generate_task_suite, validate_task_splits
from v6_4.trajectory_codec import CubicBSplineCodec

ROOT = Path(__file__).resolve().parents[1]


def load_task_suite(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = tuple(TaskSpec.from_dict(x) for x in payload["tasks"])
    splits = validate_task_splits(tasks)
    if payload["task_hashes"] != {t.task_id: t.sha256() for t in tasks} or payload["splits"] != splits:
        raise ValueError("frozen task suite content/hash/split mismatch")
    return tasks


def prepare_provider(task, controls, spec):
    """Prepare only private prediction states with the actual model settings."""
    scenario = scenario_from_task(task)
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=1,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    initial = mujoco.MjData(model)
    initial.qpos[:], initial.qvel[:] = task.initial_qpos, task.initial_qvel
    initial.ctrl[:] = 0
    mujoco.mj_forward(model, initial)
    provider = SplineReferenceProvider(task, controls).prepare(spec, model, initial, scenario)
    return provider, verifier


def save_reference(provider, path):
    np.savez_compressed(path, **provider.prediction, control_points=provider.controls)


def gate_allows_execution(gate, *, require_raw_gate=True):
    """An engineering postprocessed proposal still requires every full gate."""
    return bool(gate['passed'] and (gate['raw_passed'] or not require_raw_gate))


def run_attempt(task: TaskSpec, output_dir: Path, *, method: str,
                proposal: TrajectoryProposal | None = None,
                require_raw_gate: bool = True, attribution: dict | None = None):
    """Keep every gate failure, complete run, and physical refusal separately."""
    spec = default_v6_lite_robot_spec()
    scenario = scenario_from_task(task)
    run_config = V6LiteRunConfig(pcc_mode="bounded_interval_pcc",
                                dispatch_clock_policy="research_simulation")
    run_config.validate()
    qp_config = HierarchicalQPConfig(enable_pcc_cbf=False, enable_capsule_cbf=True)
    output_dir = Path(output_dir)
    metadata = start_run(output_dir, run_config=run_config, qp_config=qp_config,
                         spec=spec, scenarios=(scenario,))
    started = time.perf_counter()
    _write_json(output_dir / "task.json", task.to_dict(), exclusive=True)
    source_hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "v6_4").rglob("*.py"))
        if "output" not in p.relative_to(ROOT / "v6_4").parts
    }
    identity = {"schema": "v6_4_trial_identity_v1", "task_sha256": task.sha256(),
                "method": method, "source_sha256": source_hashes,
                "frozen_base_commit": "070e7008923bf2642dcecf9806745564388669b9",
                "run_config": asdict(run_config), "qp_config": asdict(qp_config),
                "attribution": attribution or {}, "raw_gate_required": require_raw_gate,
                "future_actual_state_is_planner_input": False,
                "wall_20ms_is_acceptance_gate": False}
    _write_json(output_dir / "planning_identity.json", identity, exclusive=True)
    provider, gate, scene_result, evaluation = None, None, None, None
    reference_path = None
    result = {"schema": "v6_4_trial_result_v1", "task_id": task.task_id,
              "task_sha256": task.sha256(), "split": task.split, "family": task.family,
              "method": method, "proposal_accepted": False,
              "raw_proposal_passed": False if proposal is not None else None,
              "task_success": False, "complete": False, "fallback_used": False,
              "postprocessing": [], "status": "STARTED"}
    try:
        if proposal is not None:
            if proposal.task_id != task.task_id or proposal.task_sha256 != task.sha256():
                raise ValueError("proposal identity differs from the frozen task")
            _write_json(output_dir / "raw_proposal.json", proposal.to_dict(), exclusive=True)
            codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
            controls = codec.decode_free(proposal.free_controls)
            np.savez_compressed(output_dir / "raw_proposal.npz", controls_free=proposal.free_controls,
                                control_points=controls)
            result["postprocessing"] = list(proposal.postprocessing)
            result["fallback_used"] = proposal.origin == "fallback"
            provider, verifier = prepare_provider(task, controls, spec)
            reference_path = output_dir / "candidate_reference.npz"
            save_reference(provider, reference_path)
            _write_json(output_dir / "reference_identity.json", provider.metadata, exclusive=True)
            from v6_4.proposal_gate import gate_proposal
            gate = gate_proposal(task, proposal, spec, provider=provider,
                                 output_path=output_dir / "proposal_gate.json")
            result["raw_proposal_passed"] = gate["raw_passed"]
            if not gate_allows_execution(gate, require_raw_gate=require_raw_gate):
                result["status"] = "PROPOSAL_REJECTED"
                result["failure"] = {"phase": "proposal_gate", "gate": gate}
            else:
                result["proposal_accepted"] = True
                import shutil
                selected = output_dir / "selected_reference.npz"
                shutil.copyfile(reference_path, selected)
                reference_path = selected
        elif method == "fixed_reference":
            # Protocol-B tasks retain the declared current deterministic curve.
            # Its availability is distinct from a learned joint-proposal gate.
            result["proposal_accepted"] = True
            result["raw_proposal_metric_scope"] = "fixed Cartesian reference availability; no joint proposal gate"
        else:
            raise ValueError("a trajectory proposal is required for this method")

        if result["proposal_accepted"]:
            scene_result = run_synchronous_scenario(
                spec, run_config, qp_config, scenario, output_dir / "traces",
                reference_provider=provider)
            _write_json(output_dir / "historical_metric_observations.json", scene_result, exclusive=True)
            trace_path = output_dir / "traces" / f"{scenario.scenario_id}.npz"
            from v6_4.evaluate_planning import evaluate_trial
            evaluation = evaluate_trial(task, trace_path, reference_path,
                scenario_result=scene_result, qp_config=qp_config,
                output_dir=output_dir / "evaluation")
            result.update(task_success=bool(evaluation["task_success"]),
                          complete=bool(evaluation["complete"]),
                          status="TASK_COMPLETED" if evaluation["task_success"] else "TASK_FAILED",
                          trace_path=trace_path.resolve().as_posix(), evaluation=evaluation)
    except Exception as error:
        result["status"] = "EXECUTION_OR_PIPELINE_FAILURE"
        result["task_success"] = False
        result["complete"] = False
        result["failure"] = {"phase": "execution_or_evaluation", "type": type(error).__name__,
                             "message": str(error), "traceback": traceback.format_exc()}
        partials = sorted((output_dir / "failures").glob("*partial_trace.npz"))
        if partials:
            result["partial_trace_paths"] = [p.resolve().as_posix() for p in partials]
        _write_json(output_dir / "pipeline_failure.json", result["failure"], exclusive=True)
        partial = output_dir / "failures" / f"{scenario.scenario_id}_partial_trace.npz"
        result["partial_evaluation_diagnostic"] = {
            "status": "NOT_RUN", "reason": "no current accepted consumed prefix"}
        if result["proposal_accepted"] and partial in partials:
            try:
                with np.load(partial, allow_pickle=False) as saved:
                    torque = saved["torque"]
                if (torque.ndim != 2 or torque.shape[1] != 67 or not len(torque)
                        or len(torque) % 10 or not np.all(np.isfinite(torque))):
                    raise ValueError("partial torque must contain complete finite ten-step ramps with 67 channels")
                result["trace_path"] = partial.resolve().as_posix()
                if (output_dir / "evaluation").exists():
                    result["partial_evaluation_diagnostic"] = {
                        "status": "NOT_RUN", "reason": "existing evaluation preserved"}
                else:
                    from v6_4.evaluate_planning import evaluate_trial
                    evaluation = evaluate_trial(task, partial, reference_path,
                        scenario_result=scene_result, qp_config=qp_config,
                        output_dir=output_dir / "evaluation")
                    result["evaluation"] = evaluation
                    result["partial_evaluation_diagnostic"] = {
                        "status": "EVALUATED", "trace_path": partial.resolve().as_posix(),
                        "evidence_valid": bool(evaluation["evidence_valid"]),
                        "original_execution_failure_preserved": True}
            except Exception as evaluation_error:
                result["partial_evaluation_diagnostic"] = {
                    "status": "EVALUATION_FAILED", "type": type(evaluation_error).__name__,
                    "message": str(evaluation_error)}
    result["elapsed_wall_s"] = time.perf_counter() - started
    result["gate"] = gate
    result["reference_path"] = reference_path.resolve().as_posix() if reference_path else None
    _write_json(output_dir / "result.json", result, exclusive=True)
    finish_run(output_dir, metadata, passed=result["task_success"], summary={
        "v6_4_status": result["status"], "task_success": result["task_success"],
        "old_curve_rmse_is_new_success_gate": False})
    return result


def evaluate_saved(task, trial_dir, output_dir):
    """Supplement an earlier evaluation failure without rerunning its physics."""
    trial_dir, output_dir = Path(trial_dir), Path(output_dir)
    previous = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    if previous["task_sha256"] != task.sha256():
        raise ValueError("saved attempt has a different task identity")
    scenario = scenario_from_task(task)
    trace = trial_dir / "traces" / f"{scenario.scenario_id}.npz"
    scene_result = json.loads((trial_dir / "historical_metric_observations.json").read_text(encoding="utf-8"))
    reference = Path(previous["reference_path"]) if previous.get("reference_path") else None
    from v6_4.evaluate_planning import evaluate_trial
    report = evaluate_trial(task, trace, reference, scenario_result=scene_result,
        qp_config=HierarchicalQPConfig(enable_pcc_cbf=False, enable_capsule_cbf=True),
        output_dir=output_dir)
    _write_json(output_dir / "evaluation_lineage.json", {
        "prior_trial_result": (trial_dir / "result.json").resolve().as_posix(),
        "prior_trial_result_sha256": hashlib.sha256((trial_dir / "result.json").read_bytes()).hexdigest(),
        "prior_status": previous["status"], "new_physics_steps_executed": 0,
        "prior_failure_preserved": True}, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--seed", type=int, default=20261002)
    freeze.add_argument("--counts", type=int, nargs=3, default=(2, 1, 1))
    execute = sub.add_parser("execute")
    execute.add_argument("--tasks", required=True, type=Path)
    execute.add_argument("--task-id", required=True)
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument("--method", choices=("fixed_reference", "teacher", "diffusion"), required=True)
    execute.add_argument("--proposal", type=Path)
    execute.add_argument("--allow-postprocessed", action="store_true",
                         help="explicit engineering group; all task/domain/geometry gates remain required")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--tasks", required=True, type=Path)
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--trial", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        tasks = generate_task_suite(default_v6_lite_robot_spec(), args.counts, args.seed)
        freeze_task_suite(tasks, args.output)
        print(json.dumps({"task_count": len(tasks), "output": args.output.as_posix()}), flush=True)
    else:
        tasks = load_task_suite(args.tasks)
        task = next((t for t in tasks if t.task_id == args.task_id), None)
        if task is None:
            raise ValueError("task ID absent from the frozen suite")
        if args.command == "evaluate":
            report = evaluate_saved(task, args.trial, args.output)
            print(json.dumps({k: report[k] for k in ("task_id", "complete", "evidence_valid", "task_success")}), flush=True)
            return
        proposal = (TrajectoryProposal.from_dict(json.loads(args.proposal.read_text(encoding="utf-8")))
                    if args.proposal else None)
        result = run_attempt(task, args.output, method=args.method, proposal=proposal,
                             require_raw_gate=not args.allow_postprocessed)
        print(json.dumps({k: result[k] for k in ("task_id", "status", "task_success", "elapsed_wall_s")}), flush=True)


if __name__ == "__main__":
    main()
