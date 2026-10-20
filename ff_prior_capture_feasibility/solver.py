"""Coupled finite differences with fixed geometry pairs; no fixed-base shortcut."""
import time
import mujoco
import numpy as np
from scipy.optimize import minimize, lsq_linear
from ff_prior.storage import digest
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled
from ff_prior.proposal_generator import quick_gates
from ff_prior_fixed.correction import pose_vector
from ff_prior_feasible.guidance import group_norms
from .protocol import P


def basis(kind, codec):
    if kind == "full323":
        return np.eye(323).reshape(323, 19, 17)
    if kind == "codec102":
        return np.einsum("tk,kjv->jtv", codec.basis, np.eye(102).reshape(102, 6, 17).transpose(1, 0, 2))
    if kind == "legacy24":
        s = np.r_[np.linspace(0, 1, 20)[1:-1], 1.]
        h, bow = s*s*(3-2*s), np.sin(np.pi*s)
        bow[-1] = 0
        d = np.concatenate([np.eye(17)[:, None, :]*h[None, :, None],
                            np.eye(17)[10:, None, :]*bow[None, :, None]])
        return np.einsum("tk,sk,nsj->ntj", codec.basis, codec.basis, d)
    raise ValueError(kind)


def admissible_difference(z, direction, lower, upper):
    capacities = []
    for sign in (1., -1.):
        d = direction * sign
        caps = np.full(d.shape, np.inf)
        np.divide(np.broadcast_to(upper, z.shape)-z, d, out=caps, where=d > 0)
        np.divide(np.broadcast_to(lower, z.shape)-z, d, out=caps, where=d < 0)
        capacities.append(float(np.min(caps)))
    sign = 1. if capacities[0] >= capacities[1] else -1.
    capacity = max(capacities)
    return sign*min(P["difference_rad"], capacity*.5) if capacity > 1e-9 else 0.


class Evaluator:
    def __init__(self, task, robot, initial, collision, budget):
        self.task, self.robot, self.initial, self.cc, self.budget = task, robot, initial, collision, budget
        self.scratch = initial.reset(robot)
    def predict(self, z, intervals=128):
        self.budget.check(1)
        c = self.task["condition"]
        return predict_coupled(self.initial, Reference(z, c["initial_features"], c["duration"]),
                               self.robot, intervals, self.budget.deadline)
    def evaluate(self, z, active=None):
        pred = self.predict(z)
        e = pose_vector(pred["ee"][-1], self.task["condition"]["goals"])
        if active is not None:
            distances = []
            last = None
            for frame, ga, gb in active:
                if frame != last:
                    self.scratch.qpos[:] = pred["qpos"][frame]
                    self.robot.kinematics(self.scratch)
                    last = frame
                distances.append(float(mujoco.mj_geomDistance(self.robot.model, self.scratch, ga, gb, 2.5, None)))
            self.robot.calls["distance"] += len(active)
            return {"goal": e, "active_distances": np.asarray(distances), "ee": pred["ee"][-1]}
        minima, pairs, active_values = [], [], {}
        for frame, state in enumerate(pred["qpos"]):
            self.budget.check()
            self.scratch.qpos[:] = state
            self.robot.kinematics(self.scratch)
            r = self.cc.scan(self.scratch, P["active_buffer_m"])
            minima.append(r["minimum_clearance_m"])
            pairs.append(r["minimum_pair"])
            for v in r["violations"]:
                ga = self.robot.id(mujoco.mjtObj.mjOBJ_GEOM, v["geom_a"])
                gb = self.robot.id(mujoco.mjtObj.mjOBJ_GEOM, v["geom_b"])
                active_values[(frame, ga, gb)] = v["distance_m"]
        chosen = sorted(active_values, key=lambda k: (active_values[k], k))[:P["active_pair_cap"]]
        chosen.sort()
        return {"goal": e, "minima": np.asarray(minima), "pairs": pairs, "active": chosen,
                "active_distances": np.array([active_values[k] for k in chosen]),
                "ee": pred["ee"][-1], "qpos": pred["qpos"],
                "reference_hash": Reference(z, self.task["condition"]["initial_features"],
                                            self.task["condition"]["duration"]).reference_hash}


def metric(record):
    e, d = record["goal"], record["minima"]
    return {"goal_groups": group_norms(e), "Q": float(max(group_norms(e))),
            "minimum_clearance_m": float(min(d)),
            "collision_violation": float(np.mean(np.minimum((d-P["repair_clearance_m"])/P["collision_scale_m"], 0.)**2))}


def accept(old, new):
    a, b = metric(old), metric(new)
    if a["Q"] <= P["accept_goal_fraction"]:
        return b["Q"] <= P["accept_goal_fraction"] and b["collision_violation"] < a["collision_violation"]-1e-10
    return float(new["goal"]@new["goal"] + b["collision_violation"]) < float(old["goal"]@old["goal"]+a["collision_violation"])-1e-9


def differentiate(ev, z, current, directions):
    goal, distances, steps, terminal, raw_distances = [], [], [], [], []
    for d in directions:
        step = admissible_difference(z, d, ev.robot.lower, ev.robot.upper)
        if step:
            value = ev.evaluate(z+step*d, current["active"])
            goal.append((value["goal"]-current["goal"])/step)
            distances.append((value["active_distances"]-current["active_distances"])/step)
            terminal.append(value["ee"])
            raw_distances.append(value["active_distances"])
        else:
            goal.append(np.zeros(12))
            distances.append(np.zeros(len(current["active"])))
            terminal.append(current["ee"])
            raw_distances.append(current["active_distances"])
        steps.append(step)
    return np.array(goal).T, np.array(distances).T, np.array(steps), np.array(terminal), np.array(raw_distances)


def solve_step(z, directions, e, distances, jg, jd, lower, upper):
    n = len(directions)
    matrix = directions.reshape(n, -1).T
    scale = P["collision_scale_m"]
    b = (distances-P["repair_clearance_m"])/scale
    # Full collision+goal restoration, never the old goal-only fallback.
    target = np.r_[-e, np.maximum(-b, 0.), np.zeros(n)]
    design = np.vstack([jg, jd/scale, .05*np.eye(n)])
    start = lsq_linear(design, target, bounds=(-P["trust_rad"], P["trust_rad"]),
                       tol=1e-6, max_iter=80).x
    curvature = np.diff(directions, n=2, axis=1).reshape(n, -1).T
    def fun(x):
        g = e+jg@x
        v = np.minimum(b+jd@x/scale, 0.)
        c = curvature@x
        return float(v@v + .01*g@g + .001*x@x + .01*c@c)
    def jac(x):
        g = e+jg@x
        v = np.minimum(b+jd@x/scale, 0.)
        return 2*jd.T@v/scale + .02*jg.T@g + .002*x + .02*curvature.T@(curvature@x)
    def con(x):
        return P["solver_goal_fraction"]**2 - np.sum((e+jg@x).reshape(4, 3)**2, axis=1)
    def cjac(x):
        return -2*np.einsum("gi,gij->gj", (e+jg@x).reshape(4, 3), jg.reshape(4, 3, n))
    lo, hi = (np.broadcast_to(lower, z.shape)-z).ravel(), (np.broadcast_to(upper, z.shape)-z).ravel()
    if n == 323:
        bounds = list(zip(np.maximum(lo, -.15), np.minimum(hi, .15)))
        start = np.maximum(np.minimum(start, np.minimum(hi, .15)), np.maximum(lo, -.15))
        constraints = [{"type": "ineq", "fun": con, "jac": cjac}]
    else:
        bounds = [(-.15, .15)] * n
        constraints = [{"type": "ineq", "fun": con, "jac": cjac},
                       {"type": "ineq", "fun": lambda x: np.r_[matrix@x-lo, hi-matrix@x],
                        "jac": lambda x: np.vstack([matrix, -matrix])}]
    solved = minimize(fun, start, jac=jac, bounds=bounds, constraints=constraints,
                      method="SLSQP", options={"maxiter": P["solver_max_iterations"], "ftol": 1e-8})
    feasible = bool(np.isfinite(solved.x).all() and min(con(solved.x)) >= -1e-7)
    x = solved.x if feasible else start
    if not np.isfinite(x).all():
        raise ValueError("NONFINITE_SOLVER_STEP")
    return x, {"success": bool(solved.success), "linear_pose_feasible": feasible,
               "message": str(solved.message), "iterations": int(solved.nit),
               "fallback": None if feasible else "JOINT_COLLISION_GOAL_RESTORATION",
               "goal_jacobian_rank": int(np.linalg.matrix_rank(jg)), "variables": n}


def optimize(directory, start, kind, codec, ev, progress, max_iterations=10):
    z = np.asarray(start).copy()
    directions = basis(kind, codec)
    history = []
    current = ev.evaluate(z)
    atomic_json(directory / "initial.json", dict(metric(current), Z=z, reference_hash=current["reference_hash"]))
    for iteration in range(max_iterations):
        progress("bounded_path_search", variant=kind, path=directory.name, iteration=iteration+1)
        m = metric(current)
        if m["Q"] <= .65 and m["minimum_clearance_m"] >= P["repair_clearance_m"]:
            break
        jg, jd, steps, perturb_ee, perturb_distances = differentiate(ev, z, current, directions)
        ev.robot.calls["sqp"] += 1
        step, solve = solve_step(z, directions, current["goal"], current["active_distances"],
                                 jg, jd, ev.robot.lower, ev.robot.upper)
        delta = np.einsum("i,ijk->jk", step, directions)
        trials, accepted = [], False
        before = z.copy()
        for fraction in P["line_scales"]:
            trial = z+fraction*delta
            c = ev.task["condition"]
            gates = quick_gates(ev.robot, Reference(trial, c["initial_features"], c["duration"]))
            if not all(gates.values()):
                trials.append({"scale": fraction, "status": "QUICK_REJECTED", "gates": gates})
                continue
            value = ev.evaluate(trial)
            ok = accept(current, value)
            trials.append(dict(metric(value), scale=fraction, status="EVALUATED", accepted=ok))
            if ok:
                z, after, accepted = trial, value, True
                break
        row = {"iteration": iteration, "before": m, "solver": solve, "trials": trials,
               "accepted": accepted, "before_Z_hash": digest(before), "after_Z_hash": digest(z),
               "active_pairs": current["active"], "direction_basis": kind,
               "blocked_difference_columns": int(sum(steps == 0))}
        pd = directory / "iterations" / f"{iteration:02d}"
        atomic_npz(pd / "linearization.npz", Z=before, next_Z=z, goal=current["goal"],
                   active_distances=current["active_distances"], steps=steps, perturb_ee=perturb_ee,
                   jg=jg, jd=jd, step=step, perturb_distances=perturb_distances, current_qpos=current["qpos"],
                   current_ee=current["ee"], current_minima=current["minima"])
        atomic_json(pd / "record.json", row)
        history.append(row)
        if not accepted:
            break
        current = after
    result = dict(metric(current), Z=z, kind=kind, iterations=len(history),
                  reference_hash=current["reference_hash"], status="SEARCH_COMPLETE",
                  feasibility_claim=False, goal_relabel=False)
    atomic_npz(directory / "final_path.npz", Z=z, qpos=current["qpos"], ee=current["ee"],
               minima=current["minima"])
    atomic_json(directory / "final_scan.json", {"pairs": current["pairs"], "minima": current["minima"]})
    atomic_json(directory / "result.json", result)
    return result
