"""Reference validation; model evidence cannot certify real execution."""
import time
import mujoco
import numpy as np
from .contracts import MODEL_GATES, all_pass
from .proposal_generator import quick_gates
from .coupled_predictor import converge
from .coupled_predictor import pose_errors


def path_geometry(robot, initial, states, collision, threshold, wall_deadline=None):
    d = initial.reset(robot)
    minimum, pair, count = 2.5, None, 0
    for state in states:
        if wall_deadline is not None and time.time() >= wall_deadline:
            raise RuntimeError("BUDGET_EXHAUSTED")
        d.qpos[:] = state  # independent geometry scratch, never executor state
        robot.kinematics(d)
        report = collision.scan(d, threshold)
        count += 1
        if report["minimum_clearance_m"] < minimum:
            minimum, pair = report["minimum_clearance_m"], report["minimum_pair"]
    return {"minimum_clearance_m":minimum,"pair":pair,"checked_states":count,
            "passed":minimum >= threshold,"continuous_time_certified":False}


def validate_model(reference, prediction, profile, robot, initial, collision, numerical=None, wall_deadline=None):
    gates = quick_gates(robot, reference)
    gates["identity"] = initial.model_hash == robot.identity()["scene_model_hash"]
    gates["coupled_convergence"] = (numerical is not None and numerical.get("passed") is True
        and prediction["numerical"]["solve_residual"] < 1e-10
        and prediction["numerical"]["robot_momentum_residual"] < 1e-10
        and prediction["numerical"]["quaternion_norm_error"] < 1e-12)
    coarse = path_geometry(robot,initial,prediction["qpos"][::2],collision,profile.planning_clearance_m,wall_deadline)
    mid = path_geometry(robot,initial,prediction["qpos"][1::2],collision,profile.planning_clearance_m,wall_deadline)
    minimum = min(coarse["minimum_clearance_m"],mid["minimum_clearance_m"])
    geometry_delta = abs(coarse["minimum_clearance_m"] - minimum)
    gates["geometry"] = coarse["passed"] and mid["passed"]
    def geometry_motion(p):
        centers = np.linalg.norm(np.diff(p["geom"],axis=0),axis=-1)
        rotations = np.diff(p["geom_rotation"],axis=0).reshape(-1,robot.model.ngeom,3,3)
        return float(np.max(centers+np.linalg.norm(rotations,axis=(-1,-2))*robot.model.geom_rbound))
    maximum_geometry_step = geometry_motion(prediction)
    gates["geometry_refinement"] = geometry_delta <= .001 and maximum_geometry_step <= .002
    # If separate geometry grid has not converged, double independently of
    # the endpoint integrator's numerical success.
    extra = None
    if not gates["geometry_refinement"] and len(prediction["phase"]) <= 257:
        from .coupled_predictor import predict_coupled
        finer = predict_coupled(initial,reference,robot,512,wall_deadline)
        extra = path_geometry(robot,initial,finer["qpos"][1::2],collision,profile.planning_clearance_m,wall_deadline)
        maximum_geometry_step = geometry_motion(finer)
        gates["geometry_refinement"] = abs(minimum-min(minimum,extra["minimum_clearance_m"])) <= .001 and maximum_geometry_step <= .002
        minimum = min(minimum,extra["minimum_clearance_m"])
        gates["geometry"] = gates["geometry"] and extra["passed"]
    label_data=initial.reset(robot)
    label_data.qpos[:]=prediction["qpos"][-1]
    robot.kinematics(label_data)
    label_position,label_rotation=pose_errors(robot.ee(label_data),prediction["ee"][-1])
    gates["label_consistency"] = bool(np.isfinite(prediction["ee"][-1]).all()
        and np.max(label_position)<1e-10 and np.max(label_rotation)<1e-10
        and np.max(abs(label_data.qpos[robot.qpos_ids]-robot.B@reference.evaluate(reference.duration)))<1e-10)
    gates = {k:bool(v) for k,v in gates.items()}
    return {"validation_level":"MODEL_VALID" if all_pass(gates,MODEL_GATES) else "REJECTED",
            "gates":gates,"required":MODEL_GATES,"geometry":{"coarse":coarse,"midpoints":mid,"extra":extra,
            "minimum_clearance_m":minimum,"refinement_difference_m":geometry_delta,
            "maximum_geometry_step_m":maximum_geometry_step,"geometry_step_budget_m":.002,
            "hold_geometry":"constant_at_zero_terminal_velocity_static_environment"},
            "numerical":numerical,"reference_torque_feasibility":"NOT_EVALUATED",
            "validation_profile_id":profile.profile_id}


def evaluate_reference(robot,initial,reference,collision,profile,wall_deadline=None):
    gates = quick_gates(robot,reference)
    if not all(gates.values()):
        return None, {"validation_level":"REJECTED","gates":gates,"reason":"QUICK_GATE_FAILED"}
    initial_report = collision.scan(initial.reset(robot),profile.planning_clearance_m)
    if not initial_report["passed"]:
        return None, {"validation_level":"REJECTED","gates":dict(gates,geometry=False),
                      "reason":"INITIAL_CLEARANCE_FAILED","initial_geometry":initial_report}
    prediction,numerical = converge(initial,reference,robot,wall_deadline)
    return prediction,validate_model(reference,prediction,profile,robot,initial,collision,numerical,wall_deadline)
