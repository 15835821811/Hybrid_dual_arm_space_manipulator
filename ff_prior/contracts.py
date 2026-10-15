"""Fail-closed levels, frozen public tasks and strict pool lineage rules."""
from dataclasses import dataclass, asdict
import numpy as np
from .storage import digest

MODEL_GATES = ("identity", "finite", "domain", "native_range", "boundary", "motion",
    "coupled_convergence", "geometry", "geometry_refinement", "time_hold", "label_consistency")
CONDITION_KEYS = {"initial_features", "goals", "duration", "pose_mask"}
SCHEMA = {"version": "ff_prior_condition_1", "representation": "Z19x17_C24_degree3",
          "pose": "world_position_rotation6d_columns", "fixed_initial_state": True}

EXECUTION_GATES = ("identity", "finite", "dual_goal_hold", "whole_body_clearance", "joint_limits",
    "derivative_limits", "raw_torque_limits", "tracking", "manifold", "momentum",
    "independent_replay", "complete_native_time", "no_external_wrench", "no_hidden_reset")
all_gates = lambda gates, required: all_pass(gates, required)


@dataclass(frozen=True)
class AcceptanceConfig:
    position_error_m: float = .005
    rotation_error_rad: float = np.pi/180
    planning_clearance_m: float = 0.
    actual_clearance_m: float = 0.
    hold_s: float = .5
    deadline_s: float = 30.
    tracking_error_rad: float = .02
    manifold_residual_rad: float = .02
    replay_atol: float = 1e-10
    momentum_atol: float = 1e-5
    momentum_relative_tolerance: float = .002
    subdivisions: int = 2
    calibration_status: str = "INHERITED_DEV_FROZEN_FOR_THIS_SMOKE_NOT_PRODUCTION"

    @property
    def profile_id(self):
        return "v7_1_dev_"+digest(asdict(self))[:16]

    def __post_init__(self):
        for k,v in asdict(self).items():
            if isinstance(v,str):
                continue
            if not np.isfinite(v) or v < 0 or (v == 0 and k not in ("planning_clearance_m", "actual_clearance_m")):
                raise ValueError(f"invalid tolerance {k}")


def all_pass(gates, required):
    return all(gates.get(k) is True for k in required)


def public_condition(condition):
    if set(condition) != CONDITION_KEYS:
        raise ValueError("condition schema violation/future field injection")
    goals = np.asarray(condition["goals"], float)
    if goals.shape != (2, 4, 4) or not np.isfinite(goals).all():
        raise ValueError("dual full world poses required")
    for p in goals:
        if not np.allclose(p[3], [0, 0, 0, 1]) or not np.allclose(p[:3,:3].T @ p[:3,:3], np.eye(3), atol=1e-8) or np.linalg.det(p[:3,:3]) < 0.999999:
            raise ValueError("SO(3) pose required")
    features = np.asarray(condition["initial_features"], float)
    mask = np.asarray(condition["pose_mask"], float)
    duration = float(condition["duration"])
    if features.shape != (17,) or not np.isfinite(features).all() or mask.shape != (2, 6) or not np.all(mask == 1) or not 0 < duration <= 29.5:
        raise ValueError("fixed initial q17, full pose mask and legal duration required")
    return np.r_[features, goals[:, :3, 3].ravel(), goals[:, :3, :2].transpose(0,2,1).ravel(), duration, mask.ravel()]


def frozen_task(condition, contract):
    public_condition(condition)
    payload = {"condition": condition, "contract": contract, "source": "MODEL_DERIVED"}
    return dict(payload, task_hash=digest(payload))


def assert_frozen(task):
    if task["task_hash"] != digest({k: v for k, v in task.items() if k != "task_hash"}):
        raise ValueError("frozen task relabel/mutation rejected")


def assert_task_compatible(task,robot,initial,collision,profile):
    assert_frozen(task)
    expected = {"initial_state_id":initial.state_id,"robot_hash":initial.model_hash,
                "pair_policy_hash":collision.policy_hash,"acceptance":asdict(profile)}
    if task["contract"] != expected or not np.array_equal(task["condition"]["initial_features"],robot.spec.planner_zero):
        raise ValueError("frozen task robot/initial/profile mismatch")


def strict_pool(records, terminal_results):
    if not records:
        return "INSUFFICIENT_CANDIDATES"
    if len({r["task_hash"] for r in records}) != 1:
        return "TASK_MISMATCH"
    roots, paths = set(), []
    for r in records:
        path = np.asarray(r["path"], float)
        if r["validation_level"] != "MODEL_VALID" or r["root_id"] in roots or any(np.max(np.abs(path - x)) < 0.025 for x in paths):
            continue
        roots.add(r["root_id"])
        paths.append(path)
    if len(roots) < 5:
        return "INSUFFICIENT_CANDIDATES"
    if any(terminal_results.get(r["sample_id"]) not in ("EXECUTION_VERIFIED", "EXECUTION_FAILED") for r in records):
        return "INCOMPLETE_POOL_VERIFICATION"
    return "COMPLETE"
