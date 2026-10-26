"""Private zero-momentum prediction and the V6.4 reference-provider interface.

``mj_integratePos`` below advances only a private kinematic prediction. This
is not torque execution, a physics replay, a safety certificate, or a reading
of future actual states. The original executor remains responsible for them.
"""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from v6_lite.hierarchical_qp import CONTINUUM_EE_OFFSET_M, free_joint_slices, joint_addresses
from v6_lite.runtime_command import model_id
from v6_4.contracts import TaskSpec, canonical_json
from v6_4.trajectory_codec import CubicBSplineCodec


def scenario_from_task(task: TaskSpec):
    """Reconstruct only frozen declared task inputs; no RNG or actual trace."""
    from model_test.whole_body_verifier_v5 import WorkspaceSphere
    from v6_lite.irregular_waypoints import IrregularWaypointTarget
    from v6_lite.run_v6_lite import V6LiteScenario
    source = task.scenario
    target_source = source["continuum_target"]
    metadata = dict(target_source)
    for name in ("initial_position_w","target_rotation_world","transition_duration_s",
                 "path_duration_s","path_start_s","path_end_s","segment_durations_s"):
        metadata.pop(name,None)
    target = IrregularWaypointTarget(
        np.asarray(target_source["initial_position_w"],dtype=float),
        np.asarray(target_source["waypoint_points_m"],dtype=float),
        np.asarray(target_source["target_rotation_world"],dtype=float),
        float(target_source["transition_duration_s"]),float(target_source["path_duration_s"]),
        np.asarray(target_source["segment_durations_s"],dtype=float),metadata)
    scenario = V6LiteScenario(
        scenario_id=source["scenario_id"],seed=source["seed"],
        target_satellite_position_shift_m=np.asarray(source["target_satellite_position_shift_m"]),
        target_satellite_linear_velocity_m_s=np.asarray(source["target_satellite_linear_velocity_m_s"]),
        target_satellite_angular_velocity_rad_s=np.asarray(source["target_satellite_angular_velocity_rad_s"]),
        grasp_point_target_frame_m=np.asarray(source["grasp_point_target_frame_m"]),
        grasp_rotation_target_frame=np.asarray(source["grasp_rotation_target_frame"]),
        continuum_target_rotation_world=np.asarray(source["continuum_target_rotation_world"]),
        continuum_target=target,obstacles=tuple(WorkspaceSphere(item["name"],np.asarray(item["center_w"]),item["radius_m"])
                                               for item in source["workspace_obstacles"]))
    if canonical_json(scenario.to_dict()) != canonical_json(source):
        raise ValueError("TaskSpec scenario cannot be reconstructed without changing its declaration")
    return scenario


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]), a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


class _RotationReference:
    """Normalized quaternion Hermite interpolation with world angular rates."""

    def __init__(self, time: np.ndarray, rotations: np.ndarray, world_rates: np.ndarray) -> None:
        # SciPy's Cython buffer requires a writable input even though this
        # constructor does not conceptually modify the saved prediction.
        xyzw = Rotation.from_matrix(np.asarray(rotations,dtype=float).copy()).as_quat()
        quaternions = xyzw[:, [3,0,1,2]]
        for index in range(1, len(quaternions)):
            if np.dot(quaternions[index-1], quaternions[index]) < 0.:
                quaternions[index] *= -1.
        derivatives = np.stack([.5*_multiply(np.r_[0., rate], quaternion)
                                for rate, quaternion in zip(world_rates, quaternions)])
        self.spline = CubicHermiteSpline(time, quaternions, derivatives, extrapolate=False)

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        value, derivative = self.spline(time), self.spline(time, 1)
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("rotation reference quaternion is degenerate")
        quaternion = value/norm
        qdot = (derivative-quaternion*np.dot(quaternion, derivative))/norm
        conjugate = quaternion*np.array([1.,-1.,-1.,-1.])
        world_rate = 2.*_multiply(qdot, conjugate)[1:]
        return Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix(), world_rate


def _pose(model: mujoco.MjModel, data: mujoco.MjData, body: int,
          local_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(data.xmat[body]).reshape(3,3).copy()
    position = np.asarray(data.xpos[body]).copy() + rotation @ local_offset
    jacobian, angular = np.zeros((3,model.nv)), np.zeros((3,model.nv))
    mujoco.mj_jac(model, data, jacobian, angular, position, body)
    return position, jacobian @ data.qvel, rotation, angular @ data.qvel


class SplineReferenceProvider:
    """Convert a joint/shape proposal into coupled intermediate arm references.

    Fixed task targets remain in TaskSpec and are checked independently. The
    candidate's intermediate poses are never substituted for the task goal.
    """

    def __init__(self, task: TaskSpec, control_points: np.ndarray) -> None:
        self.task = task
        self.codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
        self.controls = self.codec.decode_free(self.codec.encode_free(control_points))
        self.controls.setflags(write=False)
        self._prediction = None
        self._prepared_identity = None
        self._position_interpolators = {}
        self._rotation_interpolators = {}
        self.metadata = {"schema": "v6_4_reference_adapter_v1", "task_id": task.task_id,
                         "task_sha256": task.sha256(), "prediction_is_execution": False,
                         "future_actual_trace_read": False, "prediction_period_s": .02,
                         "prediction": "private current mass-matrix zero-momentum map; kinematic configuration integration",
                         "target_prediction": "initial declared target generalized twist held constant",
                         "interpolation": "Cartesian/quaternion Hermite; spline posture sampled analytically",
                         "reference_is_task_success": False, "physics_steps_executed": 0}

    @property
    def prediction(self) -> dict[str, np.ndarray]:
        if self._prediction is None:
            raise RuntimeError("reference provider has not been prepared")
        # Read-only views prevent a gate/consumer from mutating prepared data.
        return dict(self._prediction)

    def prepare(self, spec, model: mujoco.MjModel, initial_data: mujoco.MjData, scenario) -> "SplineReferenceProvider":
        contract = spec.runtime_contract_sha256()
        if contract != self.task.model_contract_sha256:
            raise ValueError("TaskSpec and provider model contract differ")
        if canonical_json(scenario.to_dict()) != canonical_json(self.task.scenario):
            raise ValueError("runtime scenario does not match immutable TaskSpec")
        if (not np.array_equal(initial_data.qpos, np.asarray(self.task.initial_qpos))
                or not np.array_equal(initial_data.qvel, np.asarray(self.task.initial_qvel))
                or abs(float(initial_data.time)) > 1e-12):
            raise ValueError("provider requires the exact declared initial state at time zero")
        identity = (self.task.sha256(), model_id(model, contract),
                    hashlib.sha256(np.r_[initial_data.qpos,initial_data.qvel].tobytes()).hexdigest())
        if self._prepared_identity is not None:
            if identity != self._prepared_identity:
                raise ValueError("prepared provider cannot be rebound to another task/model/state")
            return self
        qpos_ids, dof_ids = joint_addresses(model, spec)
        base_qpos, base_dofs = free_joint_slices(model, spec.base_joint_name)
        target_qpos, target_dofs = free_joint_slices(model, spec.target_free_joint_name)
        decoded_q = spec.decode_position(np.asarray(initial_data.qpos)[qpos_ids])
        decoded_dq = spec.decode_velocity(np.asarray(initial_data.qvel)[dof_ids])
        if (not np.allclose(decoded_q, self.codec.q0, rtol=0., atol=1e-12)
                or not np.allclose(decoded_dq, self.codec.dq0, rtol=0., atol=1e-12)):
            raise ValueError("named planner mapping differs from the fixed spline start")
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.time = initial_data.qpos, initial_data.qvel, 0.
        data.ctrl[:] = 0.
        mujoco.mj_forward(model, data)
        full_mass = np.empty((model.nv, model.nv))
        base_ids = np.arange(base_dofs.start, base_dofs.stop)
        mujoco.mj_fullM(model, full_mass, data.qM)
        initial_momentum_rows = full_mass[base_ids] @ data.qvel
        if np.linalg.norm(initial_momentum_rows) > 1e-8:
            raise ValueError("zero-momentum nominal predictor requires zero initial robot base momentum")
        ids = {name: mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body_name)
               for name,body_name in (("rigid",spec.rigid_tip_body_name),
                                      ("continuum",spec.continuum_tip_body_name),
                                      ("target","target_satellite"))}
        base_joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.base_joint_name)
        base_body = int(model.jnt_bodyid[base_joint])
        if min(ids.values()) < 0:
            raise ValueError("nominal reference body missing")
        time = np.linspace(0.,27.,1351)
        spline = self.codec.sample(self.controls,time)
        keys = ("full_qpos","full_qvel","base_pose","base_twist","target_position","target_rotation",
                "rigid_position","rigid_velocity","rigid_rotation","rigid_angular_velocity",
                "continuum_position","continuum_velocity","continuum_rotation","continuum_angular_velocity")
        recorded = {key: [] for key in keys}
        target_twist = np.asarray(initial_data.qvel)[target_dofs].copy()
        momentum_residual = []
        for index,t in enumerate(time):
            data.qpos[qpos_ids] = spec.encode_position(spline["q"][index])
            data.time = float(t)
            mujoco.mj_forward(model, data)
            mujoco.mj_fullM(model, full_mass, data.qM)
            mass_bb = full_mass[np.ix_(base_ids,base_ids)]
            mass_ba = full_mass[np.ix_(base_ids,dof_ids)]
            base_map = -np.linalg.solve(mass_bb, mass_ba @ spec.planner_to_low_level)
            data.qvel[:] = 0.
            data.qvel[dof_ids] = spec.encode_velocity(spline["dq"][index])
            data.qvel[base_dofs] = base_map @ spline["dq"][index]
            data.qvel[target_dofs] = target_twist
            momentum_residual.append(float(np.linalg.norm(mass_bb @ base_map + mass_ba @ spec.planner_to_low_level)))
            recorded["full_qpos"].append(data.qpos.copy()); recorded["full_qvel"].append(data.qvel.copy())
            recorded["base_pose"].append(data.qpos[base_qpos].copy())
            bp,bv,br,bw = _pose(model,data,base_body,np.zeros(3))
            recorded["base_twist"].append(np.r_[bv,bw])
            tp,tv,tr,tw = _pose(model,data,ids["target"],np.zeros(3))
            recorded["target_position"].append(tp); recorded["target_rotation"].append(tr)
            for branch,offset in (("rigid",np.zeros(3)),("continuum",CONTINUUM_EE_OFFSET_M)):
                position,velocity,rotation,angular = _pose(model,data,ids[branch],offset)
                for key,value in (("position",position),("velocity",velocity),("rotation",rotation),("angular_velocity",angular)):
                    recorded[f"{branch}_{key}"].append(value)
            if index < len(time)-1:
                # Explicit private zero-momentum prediction, not mj_step or execution.
                mujoco.mj_integratePos(model,data.qpos,data.qvel,float(time[index+1]-t))
        prediction = {"time":time,"q":spline["q"],"dq":spline["dq"],"ddq":spline["ddq"],
                      **{key:np.asarray(value) for key,value in recorded.items()},
                      "zero_momentum_map_residual":np.asarray(momentum_residual)}
        if any(not np.all(np.isfinite(value)) for value in prediction.values()):
            raise ValueError("nominal coupled prediction became nonfinite")
        for value in prediction.values():
            value.setflags(write=False)
        self._prediction, self._prepared_identity = prediction, identity
        for branch in ("rigid","continuum"):
            self._position_interpolators[branch] = CubicHermiteSpline(time,prediction[f"{branch}_position"],prediction[f"{branch}_velocity"],extrapolate=False)
            self._rotation_interpolators[branch] = _RotationReference(time,prediction[f"{branch}_rotation"],prediction[f"{branch}_angular_velocity"])
        self.metadata.update({"source_model_id":identity[1],"prediction_state_count":1351,
                              "maximum_zero_momentum_map_residual":max(momentum_residual),
                              "continuum_tip_local_offset_body_m":CONTINUUM_EE_OFFSET_M.tolist()})
        return self

    def sample(self, time_s: float) -> dict[str, np.ndarray]:
        if self._prediction is None:
            raise RuntimeError("reference provider has not been prepared")
        if not np.isfinite(time_s) or time_s < -1e-9 or time_s > 27.+1e-9:
            raise ValueError("reference time outside the declared complete 27s horizon")
        time = min(27.,max(0.,float(time_s)))  # Endpoint clock roundoff only; not proposal clipping.
        posture = self.codec.sample(self.controls,np.asarray([time]))
        result = {"posture_reference_q":posture["q"][0],"posture_reference_dq":posture["dq"][0]}
        for branch in ("rigid","continuum"):
            position = self._position_interpolators[branch]
            rotation,angular = self._rotation_interpolators[branch].sample(time)
            result.update({f"{branch}_target_position":position(time),f"{branch}_target_velocity":position(time,1),
                           f"{branch}_target_rotation":rotation,f"{branch}_target_angular_velocity":angular})
        return result

    def save_prediction(self, path) -> None:
        np.savez_compressed(path, **self.prediction)
