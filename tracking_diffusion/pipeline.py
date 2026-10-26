"""Run: python -m tracking_diffusion.pipeline --help"""
from __future__ import annotations

# Set before importing numpy/MuJoCo/PyTorch; no dependency on a particular venv.
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .environment import CONTROL_DT, Task, TrackingEnv, rotation_error
from .trajectory import HORIZON, REPLAN, JointChunk, fit_chunk


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body=json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    # Progress readers must see either the old or the new complete document.
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
            prefix=path.name+".",suffix=".tmp",delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(body)
    try:
        for attempt in range(10):
            try:
                os.replace(temporary,path)
                break
            except PermissionError:
                if attempt==9: raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def error_metrics(error, times):
    result = {}
    for arm, pos, rot in (("continuum", slice(0, 3), slice(3, 6)),
                           ("rigid", slice(6, 9), slice(9, 12))):
        p = np.linalg.norm(error[:, pos], axis=1)*1000
        r = np.rad2deg(np.linalg.norm(error[:, rot], axis=1))
        active = (times >= 4.5) & (times <= 25.5)
        steady = times >= max(4.5, times[-1]-1.5)
        result[arm] = {"position_rmse_mm": float(np.sqrt(np.mean(p*p))),
                       "position_p95_mm": float(np.percentile(p, 95)),
                       "position_max_mm": float(np.max(p)),
                       "final_position_mm": float(p[-1]),
                       "orientation_max_deg": float(np.max(r)),
                       "path_rmse_mm": float(np.sqrt(np.mean(p[active]**2))) if active.any() else None,
                       "steady_rmse_mm": float(np.sqrt(np.mean(p[steady]**2))) if steady.any() else None}
    return result


def select_chunk(env, free_candidates, horizon=HORIZON, chunk_type=JointChunk,
                 scoring="legacy", previous_acceleration=None, continuity_weight=0.):
    q, dq = env.state()
    g, drift = env.generalized_map()
    j, offset = env.task_jacobian(g, drift)
    times = np.linspace(0, horizon, 6) if scoring=="legacy" else np.unique(np.r_[
        np.linspace(0,min(.1,horizon),6),np.linspace(0,horizon,11)])
    current = [env.pose(0), env.pose(1)]
    target = []
    for delta in times:
        ref = env.reference(float(env.data.time)+delta)
        values = []
        for arm, k in ((0, 0), (1, 4)):
            p, r = current[arm]
            values.extend([ref[k]-p, rotation_error(ref[k+1], r)])
        target.append(np.concatenate(values))
    target = np.array(target)
    weights = np.repeat([1., 2., 1.4, 2.4], 3)
    admissible = []
    reports=[]
    for candidate_index,free in enumerate(free_candidates):
        chunk = chunk_type.decode(free, q, dq, horizon)
        if not np.all(np.isfinite(chunk.controls)) or not chunk.bounds_ok(env.spec):
            reports.append({"candidate":candidate_index,"admissible":False})
            continue
        positions, _, accelerations = chunk.sample(times)
        residual = (positions-q) @ j.T + times[:, None]*offset - target
        if scoring=="nonlinear":
            from .candidate_fk import residuals
            residual=residuals(env,positions,times,g,drift)
        if scoring=="legacy":
            score = float(np.mean((residual*weights)**2) + 1e-5*np.mean(accelerations**2))
        else:
            task_scale=np.repeat([.005,.02,.005,.02],3)
            weighted=(residual*weights/task_scale)**2
            tracking=.7*np.mean(weighted[times<=min(.1,horizon)+1e-9])+.3*np.mean(weighted)
            previous=np.zeros(17) if previous_acceleration is None else previous_acceleration
            continuity=np.mean(((accelerations[0]-previous)/env.spec.planner_acceleration_limits)**2)
            score=float(tracking+1e-5*np.mean((accelerations/env.spec.planner_acceleration_limits)**2)+continuity_weight*continuity)
        reports.append({"candidate":candidate_index,"admissible":True,"score":score})
        admissible.append((score, chunk,candidate_index))
    if not admissible:
        raise RuntimeError("NO_PLAN: all diffusion chunks violate position/velocity/acceleration bounds")
    # Local linearization is a ranking heuristic, not a full-horizon safety test.
    selected=min(admissible,key=lambda v:v[0])
    env.last_selection_report={"candidates":reports,"selected_index":selected[2],"scoring":scoring}
    return selected[1], len(admissible)


def run_episode(task, output, policy=None, mode="teacher", feedback=1., progress=False,
                observer=None, control=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"result.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing episode: {output}")
    start = time.perf_counter()
    env = TrackingEnv(task)
    if policy is not None and getattr(policy, "task_domain", "legacy-wide-v1") != task.domain:
        raise ValueError("Checkpoint and execution task domain differ")
    if policy is not None and getattr(policy,"metadata",{}).get("profile_sha256",env.task_profile_hash) != env.task_profile_hash:
        raise ValueError("Checkpoint and task profile hash differ")
    if policy is not None and hasattr(policy, "bind_environment"):
        policy.bind_environment(env)
    def notify(kind, payload=None):
        if observer is not None:
            observer(kind, env, payload or {})
    notify("initialized")
    log = {key: [] for key in ("time", "q", "dq", "qpos", "qvel", "condition", "error",
           "q_reference", "planned_velocity", "feedback_norm", "filter_delta", "selected_velocity")}
    sampling_times, admissions = [], []
    def record(planned, selected):
        q, dq = env.state()
        values = {"time": float(env.data.time), "q": q, "dq": dq,
                  "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                  "condition": env.condition(), "error": env.task_error(),
                  "q_reference": env.reference_q.copy(), "planned_velocity": planned.copy(),
                  "feedback_norm": getattr(env, "last_feedback_norm", 0.),
                  "filter_delta": env.last_filter_delta, "selected_velocity": selected.copy()}
        for key, value in values.items():
            log[key].append(value)
        notify("metrics", {"time":values["time"],
            "error_c_mm":float(1000*np.linalg.norm(values["error"][:3])),
            "error_r_mm":float(1000*np.linalg.norm(values["error"][6:9]))})
    record(np.zeros(17), np.zeros(17))
    notify("state")
    failure, chunk, chunk_start = None, None, 0.
    ticks = int(round(task.duration/CONTROL_DT))
    try:
        for tick in range(ticks):
            if control is not None and not control():
                raise RuntimeError("USER_CANCELLED")
            planned = np.zeros(17)
            if mode == "teacher":
                velocity = env.command_velocity()
            elif mode == "feedback-only":
                velocity = env.command_velocity(planned, cartesian_feedback=feedback)
            elif mode in ("diffusion", "label-replay"):
                if tick % int(round(REPLAN/CONTROL_DT)) == 0:
                    sampling_start = time.perf_counter()
                    previous_acceleration=(chunk.sample(float(env.data.time)-chunk_start)[2]
                        if chunk is not None else np.zeros(17))
                    policy.previous_acceleration=previous_acceleration
                    def denoising(step, total, value):
                        if control is not None and not control():
                            raise RuntimeError("USER_CANCELLED")
                        notify("denoising", {"step": step, "total": total, "free": value})
                    if hasattr(policy, "chunk_type"):
                        horizon = min(policy.horizon, task.duration-float(env.data.time))
                        env.active_chunk_type, env.active_horizon = policy.chunk_type, horizon
                        candidates = policy.predict(env.condition(), horizon=horizon,
                            time=float(env.data.time), on_step=denoising if observer else None)
                        chunk, accepted = select_chunk(env,candidates,horizon,policy.chunk_type,
                            getattr(policy,"scoring","legacy"),previous_acceleration,
                            getattr(policy,"continuity_weight",0.))
                    else:
                        candidates = policy.predict(env.condition(), on_step=denoising if observer else None)
                        chunk, accepted = select_chunk(env, candidates)
                    notify("selected", {"controls": chunk.controls, "accepted": accepted,
                        "guidance": getattr(policy,"last_guidance",[]),
                        "selected_index":env.last_selection_report.get("selected_index"),
                        "candidate_stages":getattr(policy,"last_stages",{})})
                    sampling_times.append(time.perf_counter()-sampling_start)
                    if getattr(policy,"diagnostic",False) and env.data.time<=1.+1e-8:
                        q0,dq0=env.state()
                        g0,d0=env.generalized_map()
                        j0,o0=env.task_jacobian(g0,d0)
                        dt=np.linspace(0.,horizon,26)
                        np.savez_compressed(output/f"planning_{tick:04d}.npz",
                            time=float(env.data.time),horizon=horizon,q=q0,dq=dq0,
                            jacobian=j0,drift=o0,preview_times=dt,
                            target=np.array([env.task_error(float(env.data.time)+x) for x in dt]),
                            previous_acceleration=previous_acceleration,
                            acceleration_limits=env.spec.planner_acceleration_limits,
                            selected_controls=chunk.controls,final_candidates=candidates,
                            **getattr(policy,"last_stages",{}))
                        write_json(output/f"planning_{tick:04d}.json",env.last_selection_report)
                    admissions.append(accepted)
                    chunk_start = float(env.data.time)
                # Feedforward is from the generated joint trajectory. Feedback
                # uses task error; no teacher invocation in this branch.
                qp, vp, _ = chunk.sample(float(env.data.time)-chunk_start+CONTROL_DT)
                q, _ = env.state()
                planned = vp + 2.*(qp-q)
                velocity = env.command_velocity(planned, cartesian_feedback=feedback)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            env.advance(velocity)
            record(planned, velocity)
            if tick % 5 == 0 or tick == ticks-1:
                notify("state", {"planned_velocity": planned, "selected_velocity": velocity})
            if progress and (tick+1) % 250 == 0:
                print(f"{mode} seed={task.seed} sim={env.data.time:.1f}s wall={time.perf_counter()-start:.1f}s", flush=True)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        failure = str(exc)
        # Preserve the failure state even if it occurred inside a 500 Hz tick.
        if float(env.data.time) > log["time"][-1]+1e-9:
            record(planned, env.command)
        notify("state")
    arrays = {key: np.asarray(value) for key, value in log.items()}
    arrays["planning_seconds"]=np.asarray(sampling_times)
    np.savez_compressed(output/"trace.npz", **arrays)
    metrics = error_metrics(arrays["error"], arrays["time"])
    completed = failure is None and env.data.time >= task.duration-1e-8
    full_task = completed and task.duration >= 27.
    # New short smoke runs cannot be used to claim full-task tracking success.
    old_tracking = bool(full_task and metrics["continuum"]["path_rmse_mm"] <= .18
                        and metrics["rigid"]["final_position_mm"] <= .10
                        and metrics["rigid"]["steady_rmse_mm"] <= .15
                        and max(m["orientation_max_deg"] for m in metrics.values()) <= .25)
    # Teacher-admission threshold is explicitly separate from final performance.
    accepted_teacher = bool(full_task and env.minimum_margin >= -1e-6
        and env.torque_saturations == 0 and env.qp_failures == 0 and env.joint_limit_violations == 0
        and all(m["position_max_mm"] <= 3. and m["orientation_max_deg"] <= .5 for m in metrics.values()))
    result = {"task": asdict(task), "mode": mode, "completed": completed,
              "full_task_executed": full_task, "failure": failure,
              "cartesian_feedback_multiplier": feedback if mode != "teacher" else None,
              "metrics": metrics, "execution": env.diagnostics(),
              "meets_original_tracking_precision": old_tracking,
              "meets_stage_tracking": accepted_teacher,
              "accepted_for_teacher_dataset": accepted_teacher if mode == "teacher" else False,
              "teacher_admission_thresholds": {"both_arms_max_position_mm": 3., "orientation_max_deg": .5},
              "sampling_calls": len(sampling_times),
              "sampling_and_selection_p95_ms": float(1e3*np.percentile(sampling_times, 95)) if sampling_times else None,
              "sampling_and_selection_p99_ms": float(1e3*np.percentile(sampling_times,99)) if sampling_times else None,
              "sampling_and_selection_max_ms": float(1e3*max(sampling_times)) if sampling_times else None,
              "planning_100ms_deadline_misses": sum(t>REPLAN for t in sampling_times),
              "mean_admitted_candidates": float(np.mean(admissions)) if admissions else None,
              "mean_generated_velocity_norm": float(np.mean(np.linalg.norm(arrays["planned_velocity"], axis=1))),
              "mean_feedback_velocity_norm": float(np.mean(arrays["feedback_norm"])),
              "mean_filter_correction_norm": float(np.mean(arrays["filter_delta"])),
              "policy_metadata": getattr(policy, "metadata", None),
              "wall_seconds": time.perf_counter()-start,
              "model_runtime_contract_sha256": env.spec.runtime_contract_sha256(),
              "task_profile_sha256": env.task_profile_hash, "task_parameters": env.task_parameters,
              "path": env.path.to_dict()}
    write_json(output/"result.json", result)
    notify("finished", {"result": result})
    print(json.dumps({"mode": mode, "seed": task.seed, "completed": completed,
                      "failure": failure, "metrics": metrics,
                      "teacher_accepted": accepted_teacher, "wall_s": result["wall_seconds"]}), flush=True)
    return result, arrays, env.spec


def collect(output, train_episodes=8, validation_episodes=2, seed=20261012, obstacles=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"dataset.npz").exists():
        raise FileExistsError("Use a fresh output directory; existing dataset will not be overwritten")
    items, rejected, episodes = [], [], []
    # Split by complete task BEFORE generating overlapping windows.
    for split, count in ((0, train_episodes), (1, validation_episodes)):
        for index in range(count):
            task_seed = seed + split*100000 + index*104729
            folder = output/"episodes"/f"{'train' if split == 0 else 'val'}_{task_seed}"
            if (folder/"result.json").exists():
                result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                with np.load(folder/"trace.npz", allow_pickle=False) as f:
                    arrays = {k: f[k] for k in f.files}
                spec = None
                if result["task"] != asdict(Task(task_seed, 27., obstacles)):
                    raise ValueError("Existing episode belongs to a different task")
            else:
                result, arrays, spec = run_episode(Task(task_seed, 27., obstacles), folder, progress=True)
            episodes.append({"seed": task_seed, "split": split, "result": str(folder/"result.json"),
                             "accepted": result["accepted_for_teacher_dataset"]})
            if not result["accepted_for_teacher_dataset"]:
                continue
            if spec is None:
                # Only the immutable limits are required for resumed collection.
                from .environment import ROOT
                from model_test.robot_model_spec_v5 import RobotModelSpecV5
                spec = RobotModelSpecV5.from_urdf(Path("dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf"))
            window = int(round(HORIZON/CONTROL_DT))
            for k in range(0, len(arrays["q"])-window, 5):
                free, fit_error = fit_chunk(arrays["q"][k:k+window+1], arrays["dq"][k])
                chunk = JointChunk.decode(free, arrays["q"][k], arrays["dq"][k])
                if fit_error > .002 or not chunk.bounds_ok(spec):
                    rejected.append({"seed": task_seed, "start": float(arrays["time"][k]),
                                     "fit_error_rad": fit_error})
                    continue
                items.append((arrays["condition"][k], free, split, task_seed, arrays["time"][k], fit_error))
            write_json(output/"collection_progress.json", {"episodes": episodes, "windows": len(items)})
    split_counts = {str(s): sum(x[2] == s for x in items) for s in (0, 1)}
    report = {"episodes": episodes, "window_counts": split_counts,
              "rejected_windows": rejected, "horizon_s": HORIZON, "window_stride_s": .1,
              "warning": "Overlapping windows are not independent tasks. TEST is never included in dataset.npz."}
    write_json(output/"dataset_report.json", report)
    if not items or min(split_counts.values()) == 0:
        raise RuntimeError("No accepted data in at least one split; inspect teacher failures before training")
    columns = list(zip(*items))
    np.savez_compressed(output/"dataset.npz", condition=np.asarray(columns[0], dtype=np.float32),
                        free=np.asarray(columns[1], dtype=np.float32), split=np.asarray(columns[2]),
                        seed=np.asarray(columns[3]), time=np.asarray(columns[4]), fit_error=np.asarray(columns[5]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "collect", "train", "evaluate", "demo"))
    parser.add_argument("--output", type=Path, default=Path("tracking_output/pilot"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--duration", type=float, default=27.)
    parser.add_argument("--obstacles", type=int, choices=(0, 2), default=2)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--feedback", type=float, default=1.)
    parser.add_argument("--mode", choices=("diffusion", "feedback-only"), default="diffusion")
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--domain", choices=("legacy-wide-v1", "local10mm-v1"), default="local10mm-v1")
    parser.add_argument("--nominal", action="store_true")
    args = parser.parse_args()
    if args.train_episodes < 1 or args.validation_episodes < 1:
        parser.error("At least one complete task per data split is required")
    if args.sampling_steps < 2 or args.candidates < 1 or not 0 <= args.feedback <= 2:
        parser.error("Invalid sampling or feedback parameter")
    if args.stage == "baseline":
        run_episode(Task(args.seed, args.duration, args.obstacles, args.domain, args.nominal), args.output, progress=True)
    elif args.stage == "collect":
        collect(args.output, args.train_episodes, args.validation_episodes, args.seed, args.obstacles)
    elif args.stage == "train":
        from .model import train
        report = train(args.dataset or args.output/"dataset.npz", args.output/"model.pt",
                       args.steps, args.seed, args.device)
        write_json(args.output/"training_report.json", report)
    elif args.stage == "evaluate":
        policy = None
        if args.mode == "diffusion":
            from .model import DiffusionPolicy
            policy = DiffusionPolicy(args.checkpoint or args.output/"model.pt", args.device,
                                     args.seed, args.candidates, args.sampling_steps)
            if args.seed in policy.train_seeds + policy.val_seeds:
                parser.error("evaluate requires a held-out task seed (not TRAIN or VAL)")
        run_episode(Task(args.seed, args.duration, args.obstacles), args.output,
                    policy, args.mode, args.feedback, progress=True)
    elif args.stage == "demo":
        from .model import train, DiffusionPolicy
        collect(args.output/"data", args.train_episodes, args.validation_episodes, args.seed, args.obstacles)
        report = train(args.output/"data/dataset.npz", args.output/"model.pt", args.steps, args.seed, args.device)
        write_json(args.output/"training_report.json", report)
        # Same held-out task for teacher, feedback only and diffusion comparisons.
        test = Task(args.seed+9000001, 27., args.obstacles)
        results = {}
        for mode in ("teacher", "feedback-only", "diffusion"):
            policy = DiffusionPolicy(args.output/"model.pt", args.device, test.seed,
                                     args.candidates, args.sampling_steps) if mode == "diffusion" else None
            result, _, _ = run_episode(test, args.output/"test"/mode, policy, mode, args.feedback, progress=True)
            results[mode] = result
        write_json(args.output/"comparison.json", results)


if __name__ == "__main__":
    main()
