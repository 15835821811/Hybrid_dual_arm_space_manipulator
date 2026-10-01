"""One ten-step reference ramp shared by QP, gate, and 500 Hz servo.

The QP uses its exact affine velocity weights. Reference-position clipping
uses the measured position at each real servo step; a frozen-state preview is
only a prediction and is not a claim about future measured motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceStep:
    velocity: np.ndarray
    feedforward_acceleration: np.ndarray
    unclipped_position: np.ndarray
    position_after_joint_limits: np.ndarray
    position: np.ndarray
    joint_limit_clip_count: int
    measured_window_clip_count: int


def step_weights(step: int, count: int) -> tuple[float, float]:
    if count != 10 or step < 0 or step > count:
        raise ValueError("V6 reference ramp requires steps 0..10")
    fraction = float(step) / float(count)
    return 1.0 - fraction, fraction


def ramp_velocity(start: np.ndarray, endpoint: np.ndarray, step: int, count: int = 10) -> np.ndarray:
    old_weight, new_weight = step_weights(step, count)
    return old_weight * np.asarray(start) + new_weight * np.asarray(endpoint)


def ramp_mean_weights(count: int = 10) -> tuple[float, float]:
    """Mean of the ten velocities actually sent at steps 1..10."""
    weights = np.asarray([step_weights(step, count) for step in range(1, count + 1)])
    return float(np.mean(weights[:, 0])), float(np.mean(weights[:, 1]))


def advance_reference(
    position: np.ndarray,
    start_velocity: np.ndarray,
    endpoint_velocity: np.ndarray,
    step: int,
    measured_position: np.ndarray,
    planner_lower: np.ndarray,
    planner_upper: np.ndarray,
    *,
    physics_period_s: float = 0.002,
    task_period_s: float = 0.02,
    measured_window_rad: float = 0.012,
) -> ReferenceStep:
    """Apply the exact existing interpolation, integration, and two clips."""
    if abs(task_period_s / physics_period_s - 10.0) > 1e-12:
        raise ValueError("V6 ramp requires ten physics steps per task step")
    velocity = ramp_velocity(start_velocity, endpoint_velocity, step)
    acceleration = (np.asarray(endpoint_velocity) - np.asarray(start_velocity)) / task_period_s
    unclipped = np.asarray(position) + velocity * physics_period_s
    joint_limited = np.clip(unclipped, planner_lower, planner_upper)
    window_limited = np.clip(
        joint_limited,
        np.asarray(measured_position) - measured_window_rad,
        np.asarray(measured_position) + measured_window_rad,
    )
    return ReferenceStep(
        velocity=velocity,
        feedforward_acceleration=acceleration,
        unclipped_position=unclipped,
        position_after_joint_limits=joint_limited,
        position=window_limited,
        joint_limit_clip_count=int(np.count_nonzero(joint_limited != unclipped)),
        measured_window_clip_count=int(np.count_nonzero(window_limited != joint_limited)),
    )
