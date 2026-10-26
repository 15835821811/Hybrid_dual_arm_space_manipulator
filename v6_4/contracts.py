"""Immutable proposal records and the shared V6.4-A representation contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from v6_4.task_protocol import TaskPoint, TaskSpec

PLANNER_DIM = 17
CONTROL_POINT_COUNT = 32
FREE_CONTROL_POINT_COUNT = 30
DURATION_S = 27.0
TASK_PERIOD_S = 0.02
PHYSICS_PERIOD_S = 0.002
REPRESENTATION_VERSION = "v6_4_clamped_cubic_32x17_start_eliminated_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class TrajectoryProposal:
    """A proposal is not a successful demonstration or an execution certificate."""

    task_id: str
    task_sha256: str
    controls_free: tuple[tuple[float, ...], ...]
    origin: str
    seed: int = 0
    postprocessing: tuple[str, ...] = ()
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        values = np.asarray(self.controls_free, dtype=float)
        if values.shape != (FREE_CONTROL_POINT_COUNT, PLANNER_DIM) or not np.all(np.isfinite(values)):
            raise ValueError("proposal free controls must be finite 30x17 values")
        if not self.task_id or len(self.task_sha256) != 64 or not self.origin:
            raise ValueError("task identity, SHA-256 and proposal origin are required")
        int(self.task_sha256, 16)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("proposal seed must be an integer")
        if any(not isinstance(item, str) or not item for item in self.postprocessing):
            raise ValueError("postprocessing entries must be explicit nonempty strings")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("proposal metadata must be a JSON object")
        object.__setattr__(self, "controls_free", tuple(tuple(float(x) for x in row) for row in values))
        object.__setattr__(self, "postprocessing", tuple(self.postprocessing))
        object.__setattr__(self, "metadata_json", canonical_json(metadata))

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)

    @property
    def free_controls(self) -> np.ndarray:
        """Return a copy, so callers cannot change the immutable proposal."""
        return np.asarray(self.controls_free, dtype=float).copy()

    def to_dict(self) -> dict:
        return {"schema": "v6_4_trajectory_proposal_v1", "representation": REPRESENTATION_VERSION,
                "task_id": self.task_id, "task_sha256": self.task_sha256,
                "controls_free": [list(row) for row in self.controls_free], "origin": self.origin,
                "seed": self.seed, "postprocessing": list(self.postprocessing), "metadata": self.metadata,
                "closed_loop_success_established": False}

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_controls(cls, task: TaskSpec, controls_free: np.ndarray, *, origin: str,
                      seed: int = 0, postprocessing: tuple[str, ...] = (),
                      metadata: dict | None = None) -> "TrajectoryProposal":
        return cls(task.task_id, task.sha256(), controls_free, origin, seed,
                   postprocessing, canonical_json(metadata or {}))

    @classmethod
    def from_dict(cls, value: dict) -> "TrajectoryProposal":
        if (value.get("schema") != "v6_4_trajectory_proposal_v1"
                or value.get("representation") != REPRESENTATION_VERSION):
            raise ValueError("unknown proposal schema or trajectory representation")
        return cls(value["task_id"], value["task_sha256"], value["controls_free"], value["origin"],
                   value["seed"], tuple(value.get("postprocessing", ())), canonical_json(value.get("metadata", {})))
