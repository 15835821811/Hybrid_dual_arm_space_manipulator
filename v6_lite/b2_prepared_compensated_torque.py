"""Private identical compensated torque from an already-forwarded state.

The ten-step private branch calls ``mj_forward`` before each envelope check.
The original compensated torque then calls it again at the same unchanged
state. This function reuses that prepared MuJoCo state and otherwise performs
the same 67-channel inverse-dynamics arithmetic and diagnostics.
"""

from __future__ import annotations

import mujoco
import numpy as np


def prepared_compensated_torque(model, data, robot, qpos_ids, dof_ids,
                                base_dof, reference_position,
                                reference_velocity, feedforward_acceleration,
                                full_mass):
    # Caller invariant: mj_forward(model, data) ran at this same state.
    mujoco.mj_fullM(model, full_mass, data.qM)
    measured_q = np.asarray(data.qpos)[qpos_ids]
    measured_dq = np.asarray(data.qvel)[dof_ids]
    reference_low_q = robot.encode_position(reference_position)
    reference_low_dq = robot.encode_velocity(reference_velocity)
    feedforward_low_ddq = robot.encode_velocity(feedforward_acceleration)
    natural_frequency = np.concatenate([np.full(60, 42.0), np.full(7, 34.0)])
    desired = (
        feedforward_low_ddq
        + natural_frequency**2 * (reference_low_q - measured_q)
        + 2.0 * natural_frequency * (reference_low_dq - measured_dq)
    )
    acceleration_limit = np.concatenate([np.full(60, 45.0), np.full(7, 70.0)])
    unclipped = desired.copy()
    desired = np.clip(desired, -acceleration_limit, acceleration_limit)
    base_ids = np.arange(base_dof.start, base_dof.stop, dtype=np.int32)
    bias = np.asarray(data.qfrc_bias)
    passive = np.asarray(data.qfrc_passive)
    original_base = -np.linalg.solve(
        full_mass[np.ix_(base_ids, base_ids)],
        full_mass[np.ix_(base_ids, dof_ids)] @ desired
        + bias[base_ids] - passive[base_ids],
    )
    original_required = (
        full_mass[np.ix_(dof_ids, base_ids)] @ original_base
        + full_mass[np.ix_(dof_ids, dof_ids)] @ desired
        + bias[dof_ids] - passive[dof_ids]
    )
    original = np.clip(original_required, -robot.torque_limits,
                       robot.torque_limits)
    modified = full_mass + model.opt.timestep * np.diag(model.dof_damping)
    compensated_base = -np.linalg.solve(
        modified[np.ix_(base_ids, base_ids)],
        modified[np.ix_(base_ids, dof_ids)] @ desired
        + bias[base_ids] - passive[base_ids],
    )
    required = (
        modified[np.ix_(dof_ids, base_ids)] @ compensated_base
        + modified[np.ix_(dof_ids, dof_ids)] @ desired
        + bias[dof_ids] - passive[dof_ids]
    )
    command = np.clip(required, -robot.torque_limits, robot.torque_limits)
    return command, desired, {
        "original_torque_saturation_count": int(np.count_nonzero(
            original != original_required)),
        "compensated_torque_saturation_count": int(np.count_nonzero(
            command != required)),
        "acceleration_clip_count": int(np.count_nonzero(desired != unclipped)),
        "torque_change_linf_nm": float(np.max(np.abs(command - original))),
        "required_torque_abs_max_nm": float(np.max(np.abs(required))),
    }
