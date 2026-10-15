"""Full MuJoCo integration-state snapshots; every reset starts with new data."""
from dataclasses import dataclass
import mujoco
import numpy as np
from .storage import digest

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class InitialState:
    values: np.ndarray
    model_hash: str

    @classmethod
    def capture(cls, robot, data):
        values = np.empty(mujoco.mj_stateSize(robot.model, STATE))
        mujoco.mj_getState(robot.model, data, values, STATE)
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite initial state")
        values.setflags(write=False)
        return cls(values, robot.identity()["scene_model_hash"])

    @property
    def state_id(self):
        return digest({"values": self.values, "model_hash": self.model_hash, "state_mask": int(STATE)})

    def reset(self, robot):
        # Checking at reset prevents applying a snapshot to different geometry
        # or model dimensions. No state writing occurs in the actual step loop.
        if self.model_hash != robot.identity()["scene_model_hash"]:
            raise ValueError("initial-state model identity mismatch")
        data = mujoco.MjData(robot.model)
        mujoco.mj_resetData(robot.model, data)
        mujoco.mj_setState(robot.model, data, self.values.copy(), STATE)
        robot.forward(data)
        return data
