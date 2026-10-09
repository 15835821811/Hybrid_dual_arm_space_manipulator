"""Nominal command braking viability inside the unchanged PCC work domain.

This computes an additional endpoint velocity box for the same 17-D QP. It
uses the exact shared ten-step reference ramp, then maximum deceleration at
the original planner acceleration limit until the outward command is zero.
It is a reference viability repair, not an actual dynamics certificate. The
existing torque preview, next-start and actual execution gates remain required.
"""
from __future__ import annotations

import numpy as np

from v6_lite.execution_ramp import ramp_mean_weights


def stopping_displacement(velocity: float, acceleration: float, period_s: float) -> float:
    """Sum exact ten-step ramps to nonnegative max-deceleration endpoints."""
    velocity = max(float(velocity), 0.)
    decrement = float(acceleration) * float(period_s)
    if decrement <= 0.:
        raise ValueError("braking acceleration and task period must be positive")
    count = int(np.ceil(velocity / decrement))
    _, new_weight = ramp_mean_weights()
    return (period_s * ((count - new_weight) * velocity
                        - decrement * count * (count - 1) / 2.)
            if count else 0.)


def peak_outward_displacement(start: float, endpoint: float,
                              acceleration: float, period_s: float) -> float:
    """Maximum displacement during this ramp and subsequent command braking."""
    fractions = np.arange(1, 11, dtype=float) / 10.
    velocities = (1. - fractions) * float(start) + fractions * float(endpoint)
    positions = np.cumsum(velocities) * (period_s / 10.)
    return max(0., float(np.max(positions)),
               float(positions[-1]) + stopping_displacement(endpoint, acceleration, period_s))


def _largest_endpoint(remaining: float, start: float, acceleration: float,
                      period_s: float, speed: float) -> float:
    """Monotone piecewise-linear displacement inversion, with no gate tolerance."""
    left, right = -float(speed), float(speed)
    if peak_outward_displacement(start, right, acceleration, period_s) <= remaining:
        return right
    if peak_outward_displacement(start, left, acceleration, period_s) > remaining:
        # Preserve an explicit finite contradiction with the original speed
        # box. A reference already outside viability cannot be repaired here.
        return left - acceleration * period_s
    for _ in range(64):
        middle = .5 * (left + right)
        if peak_outward_displacement(start, middle, acceleration, period_s) <= remaining:
            left = middle
        else:
            right = middle
    return left


def ramp_braking_endpoint_velocity_box(shape_q, start_velocity, work_lower,
                                       work_upper, acceleration, speed,
                                       task_period_s):
    """Return a conservative command-stop box; never widen existing bounds."""
    inputs = [np.asarray(x, dtype=float) for x in
              (shape_q, start_velocity, work_lower, work_upper, acceleration, speed)]
    if any(x.shape != (10,) or not np.all(np.isfinite(x)) for x in inputs):
        raise ValueError("PCC braking inputs must be finite ten-dimensional arrays")
    q, start, lo, hi, accel, limit = inputs
    if np.any(lo >= hi) or np.any(accel <= 0.) or np.any(limit <= 0.):
        raise ValueError("PCC braking domain, acceleration and speed are invalid")
    if abs(float(task_period_s) - .02) > 1e-12:
        raise ValueError("PCC braking preserves the original 20ms task period")
    upper = np.asarray([_largest_endpoint(hi[i] - q[i], start[i], accel[i],
                                          task_period_s, limit[i]) for i in range(10)])
    lower = -np.asarray([_largest_endpoint(q[i] - lo[i], -start[i], accel[i],
                                           task_period_s, limit[i]) for i in range(10)])
    return lower, upper
