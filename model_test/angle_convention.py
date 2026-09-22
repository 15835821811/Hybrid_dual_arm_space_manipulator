from __future__ import annotations

import numpy as np


CONTINUUM_INPUT_ORDER_10 = (
    "theta1",
    "theta2",
    "theta3",
    "theta4",
    "theta5",
    "theta6",
    "theta7",
    "theta8",
    "theta9",
    "theta10",
)

RIGID_INPUT_ORDER_7 = (
    "theta_R1",
    "theta_R2",
    "theta_R3",
    "theta_R4",
    "theta_R5",
    "theta_R6",
    "theta_R7",
)

INPUT_ORDER_17 = CONTINUUM_INPUT_ORDER_10 + RIGID_INPUT_ORDER_7

ZERO_COMMAND_DEG_17 = np.zeros(17, dtype=np.float64)
CONTINUUM_HOME_DEG = np.zeros(10, dtype=np.float64)
RIGID_HOME_DEG = np.array([0.0, 45.0, 0.0, 90.0, 0.0, 45.0, 0.0], dtype=np.float64)
HOME_OFFSETS_DEG_17 = np.concatenate((CONTINUUM_HOME_DEG, RIGID_HOME_DEG))

MATLAB_CONTINUUM_MAP_12X2 = (1.0 / 6.0) * np.array(
    [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
    ],
    dtype=np.float64,
)

MUJOCO_CONTINUUM_MAP_12X2 = (1.0 / 6.0) * np.array(
    [
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
    ],
    dtype=np.float64,
)


def _repeat_segment_map(map12: np.ndarray) -> np.ndarray:
    return np.kron(np.eye(5, dtype=np.float64), map12)


def continuum_matlab_map_60_to_10() -> np.ndarray:
    return _repeat_segment_map(MATLAB_CONTINUUM_MAP_12X2)


def continuum_mujoco_map_60_to_10() -> np.ndarray:
    return _repeat_segment_map(MUJOCO_CONTINUUM_MAP_12X2)


def actuation_map_67_to_17() -> np.ndarray:
    out = np.zeros((67, 17), dtype=np.float64)
    out[:60, :10] = continuum_matlab_map_60_to_10()
    out[60:, 10:] = np.eye(7, dtype=np.float64)
    return out
