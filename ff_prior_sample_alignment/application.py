"""Predeclared static capture approach benchmark; no outcome-based scene selection."""
import time
import numpy as np
import mujoco
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.pipeline import audit, acceptance
from ff_prior_complex.tasks import task_for
from ff_prior_origin_scene.runtime import at_scene_origin
from ff_prior.scenes import SceneSpec, Obstacle
from ff_prior.guidance import public_goal_initializer
from ff_prior_delivery_guidance.sampling import candidate, LatentProjector
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from ff_prior_feasible.validation import validate
from ff_prior_generalization.experiment import execute
from .protocol import BASE, P

APP_METHODS = ("plain", "aligned", "structured", "learned_mean", "aligned_unguided")


def freeze_application():
    config = read_json(BASE / "run.json")["config"]
    _, robot, initial, _ = audit(config)
    profile = acceptance(config)
    d = initial.reset(robot)
    robot.forward(d)
    target = robot.id(mujoco.mjtObj.mjOBJ_BODY, "target_satellite")
    target_p = d.xpos[target].copy()
    target_R = d.xmat[target].reshape(3, 3).copy()
    goals = robot.ee(d).copy()
    goals[1, :3, 3] = target_p + target_R @ np.array([0., -.235, 0.])
    # Static non-contact pregrasp; not grasp contact or dynamic capture.
    # Fixed target-local geometry, independent of any planner or trained output.
    centers = [(0., -.40, .02), (-.07, -.34, .08), (.07, -.34, -.06),
               (-.07, -.46, -.07), (.07, -.46, .10)]
    entries = []
    for count in (3, 5):
        scene = SceneSpec(tuple(Obstacle(f"capture_{j}", "sphere",
                               tuple(target_p+target_R@np.array(p)), (.025, 0., 0.))
                                for j, p in enumerate(centers[:count])), f"capture_{count}_nested")
        entry = {"sample_id": f"capture_{count}_obstacles", "split": "APPLICATION_DEV",
                 "scene": scene.to_dict(), "seed": 20263521,
                 "requested_goals": goals, "duration": 27.,
                 "rigid_goal": "static non-contact pregrasp at target-local [0,-.235,0]m; preserve initial tip orientation",
                 "continuum_goal": "hold initial world pose", "obstacle_count": count,
                 "frozen_before_training": True, "planner_results_used": False}
        try:
            rr, ii, cc, proof = at_scene_origin(robot, initial, scene, robot.spec.planner_zero)
            entry.update(task=task_for(rr, ii, cc, profile, goals, 27.), origin_proof=proof, initial_status="PASS")
        except Exception as error:
            entry.update(initial_status="BLOCKED", reason=repr(error))
        entries.append(entry)
    return {"entries": entries, "methods": APP_METHODS, "candidates_per_method": 5,
            "new_candidates_max": 50, "new_executions_max": 10,
            "reference_logic": "fixed nested three/five spherical obstacles; common tasks and strict verifier",
            "not_reference_coordinates_or_robot": True, "not_dynamic_waypoints": True, "TEST": False}


def run(out, bundle_map, selection, cache, profile, ledger, budget, progress):
    entries = selection["entries"]
    for ti, e in enumerate(entries):
        sid = e["sample_id"]
        if e["initial_status"] != "PASS":
            atomic_json(out / "application" / sid / "BLOCKED.json", e)
            continue
        rr, ii, cc, _ = cache.get(e)
        task = e["task"]
        base = public_goal_initializer(task, rr, ii)
        for method in APP_METHODS:
            records = []
            bundle = bundle_map["plain" if method == "plain" else "aligned"]
            for k in range(5):
                path = out / "application" / sid / method / "candidates" / f"{k:02d}.json"
                key = f"application_{sid}_{method}_{k}"
                if key not in ledger.data["completed"]:
                    ledger.begin(key, "application_candidate", P["wall_budget_s"], 50)
                    budget.reserve(113)
                    progress("M4_static_capture", obstacles=e["obstacle_count"], method=method, candidate=k+1)
                    seed = e["seed"] + k*997
                    if method != "aligned_unguided":
                        name = "late4_diffusion" if method in ("plain", "aligned") else method
                        r = candidate(name, k, task, base, seed, bundle, rr, ii, cc, profile, budget.deadline)
                    else:
                        before = rr.calls.copy()
                        started = time.perf_counter()
                        raw, steps = unguided_sample(bundle, base, task, seed)
                        proj = LatentProjector(task, rr, ii, cc, profile, budget.deadline, codec=bundle[1], base=base)
                        z, prep = proj.prepare(raw)
                        r = dict(validate(z, task, rr, ii, cc, profile, budget.deadline), status="DONE",
                                 Z=z, raw_Z=raw, steps=steps, preparation=prep, physics_guidance_calls=0,
                                 seed=seed, candidate_id=k, task_hash=task["task_hash"],
                                 actual_network_calls=16, private_witness_used=False,
                                 model_calls={n: rr.calls[n]-before[n] for n in before},
                                 wall_s=time.perf_counter()-started)
                    r.update(method=method, sample_id=sid, split="APPLICATION_DEV")
                    atomic_json(path, r)
                    ledger.finish({"status": r["status"], "eligible": r["eligible"]}, r["model_calls"])
                records.append(read_json(path))
            chosen = min((r for r in records if r["eligible"]), key=lambda r: (r["score"], r["candidate_id"]), default=None)
            key = f"application_execute_{sid}_{method}"
            if key in ledger.data["completed"]:
                continue
            ledger.begin(key, "application_execution", P["wall_budget_s"], 10)
            directory = out / "application" / sid / method / "execution"
            atomic_json(directory / "selection.json", {"selected_id": chosen["candidate_id"] if chosen else None,
                        "selection_before_execution": True, "task_hash": task["task_hash"],
                        "candidate_pool_digest": digest(records)})
            before = rr.calls.copy()
            budget.check()
            if chosen:
                result = execute(directory, task, chosen["Z"], rr, ii, cc, profile, budget.deadline)
            else:
                result = {"status": "NO_CANDIDATE"}
                atomic_json(directory / "outcome.json", result)
            ledger.finish(result, {n: rr.calls[n]-before[n] for n in before})
