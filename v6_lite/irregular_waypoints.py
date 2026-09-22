"""Learning-free copy of the original irregular-waypoint target contract.

The geometry and seeded randomization match the preserved V4/V5
``irregular_waypoints`` experiment, while the runtime reference is a smooth
minimum-jerk transition followed by length-proportional waypoint segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ORIGINAL_CIRCLE_CENTER_W = np.asarray([1.800, 0.626, 0.000], dtype=np.float64)
ORIGINAL_CIRCLE_RADIUS_M = 0.150
ORIGINAL_ARC_ANGLE_RAD = 2.0 * np.pi
ORIGINAL_WAYPOINT_COUNT = 7
ORIGINAL_TRANSITION_DURATION_S = 4.5
ORIGINAL_PATH_DURATION_S = 21.0
ORIGINAL_CENTER_Y_NEGATIVE_SHIFT_RATIO = 0.15


def _minimum_jerk_progress(value: float) -> float:
    tau = float(np.clip(value, 0.0, 1.0))
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def _minimum_jerk_progress_rate(value: float) -> float:
    tau = float(np.clip(value, 0.0, 1.0))
    return 30.0 * tau**2 * (1.0 - tau) ** 2


def _sample_polyline_by_fraction(
    points: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    fractions = np.clip(np.asarray(fractions, dtype=np.float64), 0.0, 1.0)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(lengths)]
    )
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], fractions.size, axis=0)
    result = np.zeros((fractions.size, 3), dtype=np.float64)
    for index, distance in enumerate(fractions * total):
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = int(np.clip(segment, 0, lengths.size - 1))
        alpha = (distance - cumulative[segment]) / max(lengths[segment], 1e-12)
        result[index] = (
            (1.0 - alpha) * points[segment] + alpha * points[segment + 1]
        )
    return result


def _template() -> tuple[np.ndarray, np.ndarray]:
    fractions = np.asarray(
        [0.0, 0.08, 0.17, 0.30, 0.43, 0.58, 0.72, 0.87, 1.0],
        dtype=np.float64,
    )
    offsets = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.010, -0.008, 0.006],
            [-0.014, 0.010, 0.018],
            [0.016, 0.018, -0.004],
            [-0.018, -0.012, 0.016],
            [0.014, 0.014, 0.008],
            [-0.012, 0.020, -0.006],
            [0.008, -0.016, 0.012],
            [-0.006, 0.012, -0.004],
        ],
        dtype=np.float64,
    )
    return fractions, offsets


def sample_original_randomization(
    waypoint_count: int, random_seed: int
) -> dict[str, np.ndarray]:
    """Use the original V4/V5 ranges and random-number draw order."""

    count = max(int(waypoint_count), 2)
    rng = np.random.default_rng(int(random_seed))
    return {
        "radial_scales": 1.0 + rng.uniform(-0.10, 0.40, size=count),
        "front_back_scales": rng.uniform(0.6, 2.0, size=count),
        "front_back_offsets_m": rng.uniform(-0.030, 0.020, size=count),
    }


def build_original_waypoints(
    orientation_world: np.ndarray,
    random_seed: int,
    waypoint_count: int = ORIGINAL_WAYPOINT_COUNT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the original randomized irregular waypoint geometry."""

    rotation = np.asarray(orientation_world, dtype=np.float64).reshape(3, 3)
    anchor_fractions, anchor_offsets = _template()
    start_direction = rotation[:, 2]
    counterclockwise_direction = -rotation[:, 1]
    anchor_points = []
    for fraction, offset in zip(anchor_fractions, anchor_offsets):
        theta = fraction * ORIGINAL_ARC_ANGLE_RAD
        point = ORIGINAL_CIRCLE_CENTER_W + ORIGINAL_CIRCLE_RADIUS_M * (
            np.cos(theta) * start_direction
            + np.sin(theta) * counterclockwise_direction
        )
        anchor_points.append(point + rotation @ offset)
    source_anchors = np.asarray(anchor_points, dtype=np.float64)
    sample_fractions = np.linspace(0.0, 1.0, int(waypoint_count))
    source_points = _sample_polyline_by_fraction(
        source_anchors, sample_fractions
    )
    randomization = sample_original_randomization(waypoint_count, random_seed)
    local = (
        rotation.T
        @ (source_points - ORIGINAL_CIRCLE_CENTER_W.reshape(1, 3)).T
    ).T
    local[:, 0] = (
        local[:, 0] * randomization["front_back_scales"]
        + randomization["front_back_offsets_m"]
    )
    local[:, 1:] *= randomization["radial_scales"].reshape(-1, 1)
    points = ORIGINAL_CIRCLE_CENTER_W + (rotation @ local.T).T
    unshifted_centroid = np.mean(points, axis=0)
    shift_m = (
        abs(float(unshifted_centroid[1]))
        * ORIGINAL_CENTER_Y_NEGATIVE_SHIFT_RATIO
    )
    points = points + np.asarray([0.0, -shift_m, 0.0])
    metadata: dict[str, Any] = {
        "preset": "irregular_generalization_v3_randomized",
        "random_seed": int(random_seed),
        "circle_center_w": ORIGINAL_CIRCLE_CENTER_W.tolist(),
        "circle_radius_m": ORIGINAL_CIRCLE_RADIUS_M,
        "arc_angle_deg": 360.0,
        "waypoint_count": int(waypoint_count),
        "radial_expansion_range": [-0.10, 0.40],
        "front_back_scale_range": [0.6, 2.0],
        "front_back_jitter_range_m": [-0.030, 0.020],
        "center_y_negative_shift_ratio": (
            ORIGINAL_CENTER_Y_NEGATIVE_SHIFT_RATIO
        ),
        "center_y_negative_shift_m": shift_m,
        "radial_scales": randomization["radial_scales"].tolist(),
        "front_back_scales": randomization["front_back_scales"].tolist(),
        "front_back_offsets_m": randomization["front_back_offsets_m"].tolist(),
        "waypoint_points_m": points.tolist(),
    }
    return points, metadata


def allocate_segment_durations(
    waypoint_points: np.ndarray, total_seconds: float
) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(np.asarray(waypoint_points), axis=0), axis=1)
    total_length = float(np.sum(lengths))
    if total_length <= 1e-12:
        return np.full(lengths.shape, float(total_seconds) / lengths.size)
    return float(total_seconds) * lengths / total_length


@dataclass(frozen=True)
class IrregularWaypointTarget:
    initial_position_w: np.ndarray
    waypoint_points_w: np.ndarray
    target_rotation_world: np.ndarray
    transition_duration_s: float
    path_duration_s: float
    segment_durations_s: np.ndarray
    metadata: dict[str, Any]

    @property
    def path_start_s(self) -> float:
        return float(self.transition_duration_s)

    @property
    def path_end_s(self) -> float:
        return float(self.transition_duration_s + self.path_duration_s)

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        time_s = max(float(time_s), 0.0)
        if time_s < self.transition_duration_s:
            duration = max(float(self.transition_duration_s), 1e-12)
            tau = time_s / duration
            alpha = _minimum_jerk_progress(tau)
            alpha_rate = _minimum_jerk_progress_rate(tau) / duration
            delta = self.waypoint_points_w[0] - self.initial_position_w
            return (
                self.initial_position_w + alpha * delta,
                alpha_rate * delta,
            )
        elapsed = time_s - self.transition_duration_s
        if elapsed >= self.path_duration_s:
            return self.waypoint_points_w[-1].copy(), np.zeros(3, dtype=np.float64)
        cumulative = np.concatenate(
            [np.zeros(1), np.cumsum(self.segment_durations_s)]
        )
        segment = int(np.searchsorted(cumulative, elapsed, side="right") - 1)
        segment = int(np.clip(segment, 0, self.segment_durations_s.size - 1))
        duration = max(float(self.segment_durations_s[segment]), 1e-12)
        tau = (elapsed - cumulative[segment]) / duration
        alpha = _minimum_jerk_progress(tau)
        alpha_rate = _minimum_jerk_progress_rate(tau) / duration
        delta = self.waypoint_points_w[segment + 1] - self.waypoint_points_w[segment]
        return (
            self.waypoint_points_w[segment] + alpha * delta,
            alpha_rate * delta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "irregular_waypoints",
            "reference_profile": "minimum_jerk_c2",
            "initial_position_w": self.initial_position_w.tolist(),
            "target_rotation_world": self.target_rotation_world.tolist(),
            "transition_duration_s": float(self.transition_duration_s),
            "path_duration_s": float(self.path_duration_s),
            "path_start_s": self.path_start_s,
            "path_end_s": self.path_end_s,
            "segment_durations_s": self.segment_durations_s.tolist(),
            **self.metadata,
        }


def build_original_irregular_target(
    initial_position_w: np.ndarray,
    target_rotation_world: np.ndarray,
    random_seed: int,
) -> IrregularWaypointTarget:
    points, metadata = build_original_waypoints(
        target_rotation_world, random_seed, ORIGINAL_WAYPOINT_COUNT
    )
    durations = allocate_segment_durations(points, ORIGINAL_PATH_DURATION_S)
    return IrregularWaypointTarget(
        initial_position_w=np.asarray(initial_position_w, dtype=np.float64).copy(),
        waypoint_points_w=points,
        target_rotation_world=np.asarray(
            target_rotation_world, dtype=np.float64
        ).reshape(3, 3).copy(),
        transition_duration_s=ORIGINAL_TRANSITION_DURATION_S,
        path_duration_s=ORIGINAL_PATH_DURATION_S,
        segment_durations_s=durations,
        metadata=metadata,
    )
