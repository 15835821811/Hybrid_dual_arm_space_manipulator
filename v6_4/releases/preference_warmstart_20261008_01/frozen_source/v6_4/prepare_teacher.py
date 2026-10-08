"""Save every finite traditional planning attempt before physical filtering."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_4.contracts import TaskSpec
from v6_4.run_planning import load_task_suite
from v6_4.teacher_planner import TeacherPlanner, load_bootstrap


def load_declared_bootstrap(task, prior_path: Path, *, bootstrap_task=None, source_evaluation=None):
    """Bind a new nominal prior to its saved source TRAIN task identity."""
    prior_path = Path(prior_path)
    historical = Path(__file__).resolve().parents[1] / "v6_lite/output/runs/research_acceptance_01/simulation/traces"
    historical_paths = {(historical / f"v6_lite_scenario_{i:02d}.npz").resolve() for i in range(5)}
    source_task = bootstrap_task
    if prior_path.resolve() not in historical_paths:
        # A prior belongs to its source TaskSpec, not the task being planned.
        # Read its saved immutable identity before declaring its split/group.
        source_trial = prior_path.resolve().parent.parent
        saved_task = TaskSpec.from_dict(json.loads((source_trial / "task.json").read_text(encoding="utf-8")))
        saved_result = json.loads((source_trial / "result.json").read_text(encoding="utf-8"))
        if source_task is None:
            if saved_task.sha256() != task.sha256():
                raise ValueError("a different training prior requires explicit bootstrap_task / --prior-task-id")
            source_task = task
        if (source_task.split != "train" or saved_task.sha256() != source_task.sha256()
                or saved_result.get("task_sha256") != source_task.sha256()
                or saved_result.get("task_id") != source_task.task_id):
            raise ValueError("bootstrap source trial does not bind the declared training TaskSpec")
        trace_hash = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        evaluation_path = Path(source_evaluation) if source_evaluation else source_trial / 'evaluation/report.json'
        if not evaluation_path.is_file():
            raise ValueError('new TRAIN prior requires its complete independent evaluation; retry requires explicit source_evaluation')
        evaluation_path = evaluation_path.resolve()
        if source_trial not in evaluation_path.parents:
            raise ValueError('bootstrap evaluation must belong to the source trial')
        evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if (not evaluation.get('complete') or not evaluation.get('evidence_valid')
                or evaluation.get('task_sha256') != source_task.sha256()
                or evaluation.get('model_contract_sha256') != source_task.model_contract_sha256
                or Path(evaluation['trace_path']).resolve() != prior_path.resolve()
                or evaluation.get('trace_sha256') != trace_hash):
            raise ValueError('bootstrap trace raw SHA/path/task/model is not bound to full independent source evidence')
        manifest_path = evaluation_path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        fresh = evaluation_path.parent / 'fresh_replay.npz'
        if (manifest.get('report.json') != hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
                or manifest.get('fresh_replay.npz') != hashlib.sha256(fresh.read_bytes()).hexdigest()
                or manifest['fresh_replay.npz'] != evaluation.get('fresh_replay_sha256')):
            raise ValueError('bootstrap source evaluation artifact identity changed')
        if saved_result.get('trace_path') is None:
            lineage = json.loads((evaluation_path.parent / 'evaluation_lineage.json').read_text(encoding='utf-8'))
            if (Path(lineage['prior_trial_result']).resolve() != source_trial / 'result.json'
                    or lineage['prior_trial_result_sha256'] != hashlib.sha256((source_trial / 'result.json').read_bytes()).hexdigest()
                    or lineage['new_physics_steps_executed'] != 0 or not lineage['prior_failure_preserved']):
                raise ValueError('bootstrap evaluation retry must preserve and bind the original failed result')
        elif Path(saved_result['trace_path']).resolve() != prior_path.resolve():
            raise ValueError('bootstrap prior path differs from the actual source result')
        with np.load(prior_path, allow_pickle=False) as data:
            if (data['time'].shape != (13500,) or not np.all(np.isfinite(data['time']))
                    or not np.allclose(data['time'], .002*np.arange(1,13501), atol=1e-8, rtol=0.)
                    or data['planner_q'].shape != (13500,17) or not np.all(np.isfinite(data['planner_q']))
                    or data['torque'].shape != (13500,67) or not np.all(np.isfinite(data['torque']))
                    or not np.allclose(data['initial_qpos'], source_task.initial_qpos, atol=1e-12, rtol=0.)
                    or not np.allclose(data['initial_qvel'], source_task.initial_qvel, atol=1e-12, rtol=0.)):
                raise ValueError('bootstrap actual force trace initial state/grid/finite/full horizon changed')
    times, q, identity = load_bootstrap(prior_path, source_task=source_task)
    if prior_path.resolve() not in historical_paths:
        identity.update(source_evaluation_path=evaluation_path.as_posix(),
            source_evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            source_fresh_replay_sha256=manifest['fresh_replay.npz'],
            scope='complete actual TRAIN nominal trajectory as a planning prior only; refit needs new physical validation')
    return times, q, identity


def prepare(task, prior_path: Path, output_dir: Path, *, seed=64, starts=3, rounds=2,
            bootstrap_task=None, source_evaluation=None, geometry_rounds=8, geometry_line_search_steps=4,
            geometry_stride=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    times, q, bootstrap_identity = load_declared_bootstrap(task, prior_path,
        bootstrap_task=bootstrap_task, source_evaluation=source_evaluation)
    planner = TeacherPlanner(default_v6_lite_robot_spec(), seed=seed,
                             max_starts=starts, optimization_rounds=rounds,
                             geometry_stride=geometry_stride,
                             geometry_optimization_rounds=geometry_rounds,
                             geometry_line_search_steps=geometry_line_search_steps)
    identity = {"task_id": task.task_id, "task_sha256": task.sha256(),
                "prior_path": Path(prior_path).resolve().as_posix(),
                "prior_sha256": hashlib.sha256(Path(prior_path).read_bytes()).hexdigest(),
                "seed": seed, "starts": starts, "optimization_rounds": rounds,
                "geometry_rounds": geometry_rounds, "geometry_line_search_steps": geometry_line_search_steps,
                "geometry_stride": geometry_stride,
                "prior_is_demo_label": False,
                "bootstrap_identity": bootstrap_identity,
                "physics_after_fitting_required": True,
                "teacher_source_sha256": hashlib.sha256(Path(__import__(
                    "v6_4.teacher_planner", fromlist=["__file__"]).__file__).read_bytes()).hexdigest()}
    (output_dir / "identity.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    attempts = planner.propose(task, times, q, bootstrap_metadata=bootstrap_identity)
    for attempt in attempts:
        path = output_dir / f"candidate_{attempt.attempt_index:03d}"
        path.mkdir()
        (path / "planning_attempt.json").write_text(json.dumps(attempt.to_dict(), indent=2,
            allow_nan=False)+"\n", encoding="utf-8")
        if attempt.controls is not None:
            np.savez_compressed(path / "control_points.npz", control_points=attempt.controls)
        if attempt.proposal is not None:
            (path / "proposal.json").write_text(json.dumps(attempt.proposal.to_dict(), indent=2,
                allow_nan=False)+"\n", encoding="utf-8")
        if attempt.provider is not None:
            attempt.provider.save_prediction(path / "nominal_prediction.npz")
        print(json.dumps({"task_id": task.task_id, "candidate": attempt.attempt_index,
                          "screening_passed": attempt.screening_passed,
                          "planning_wall_s": attempt.metadata["planning_wall_time_s"],
                          "failure": attempt.metadata["failure"]}), flush=True)
    return attempts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--prior-task-id", help="explicit source TRAIN task for a new saved nominal prior")
    p.add_argument("--prior-evaluation", type=Path, help="explicit independently validated retry report for a preserved source failure")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=64)
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--geometry-rounds", type=int, default=8)
    p.add_argument("--geometry-line-search-steps", type=int, default=4)
    p.add_argument("--geometry-stride", type=int, default=10)
    a = p.parse_args()
    tasks = load_task_suite(a.tasks)
    task = next(t for t in tasks if t.task_id == a.task_id)
    bootstrap_task = next(t for t in tasks if t.task_id == a.prior_task_id) if a.prior_task_id else None
    prepare(task, a.prior, a.output, seed=a.seed, starts=a.starts, rounds=a.rounds,
            bootstrap_task=bootstrap_task, source_evaluation=a.prior_evaluation, geometry_rounds=a.geometry_rounds,
            geometry_line_search_steps=a.geometry_line_search_steps, geometry_stride=a.geometry_stride)
