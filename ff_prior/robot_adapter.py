"""Name-resolved V7 adapter to the audited mapping and physical profile."""
from __future__ import annotations

from pathlib import Path
import os
import mujoco
import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from .storage import digest
from v6_lite.continuum_model_spec import default_continuum_model_spec

ROOT = Path(__file__).resolve().parents[1]


def physical_spec(physics_dt=0.002):
    # Same numerical physical profile as default_v6_lite_robot_spec; no old
    # controller or task module is imported.
    return RobotModelSpecV5.from_urdf(
        Path(os.path.relpath(ROOT / "dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf")),
        simulation_timestep=physics_dt,
        continuum_kp=100.0, continuum_kd=5.0, continuum_torque_limit=4.0,
        rigid_kp=200.0, rigid_kd=20.0, rigid_torque_limit=80.0)


def compile_model(contract, scene=None):
    spec = mujoco.MjSpec.from_file(str(contract.source_urdf))
    # Preserve named fixed frames for the tips, mounts and pair policy. This
    # changes bookkeeping only; the audit compares mass/response to fusion.
    spec.compiler.fusestatic = False
    spec.option.gravity[:] = 0
    spec.option.timestep = contract.simulation_timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Adjacent continuum modules share a two-axis mechanical joint. Their CAD
    # collision hulls overlap by 10 mm at the bearing even at physical home.
    # MuJoCo's direct-parent filter misses the intervening axis body. Use the
    # same explicit connection exclusions as the distance verifier, not a
    # blanket collision disable or ancestor-depth exemption.
    connected = [("base_link", "link_1")] + [(f"link_{i}", f"link_{i+1}") for i in range(1, 30)]
    for i, (body1, body2) in enumerate(connected):
        spec.add_exclude(name=f"v7_module_connection_{i:02d}", bodyname1=body1, bodyname2=body2)
    if scene is not None:
        for obstacle in scene.objects:
            spec.worldbody.add_geom(name=f"v7_{obstacle.object_id}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE if obstacle.shape == "sphere" else mujoco.mjtGeom.mjGEOM_BOX,
                size=obstacle.size, pos=obstacle.position, quat=obstacle.quaternion_wxyz,
                contype=1, conaffinity=1, rgba=[0.8, 0.25, 0.15, 1])
    for name, armature, damping, limit in zip(contract.low_level_joint_names,
            contract.joint_armature, contract.joint_damping, contract.torque_limits):
        joint = spec.joint(name)
        joint.armature, joint.damping = float(armature), float(damping)
        a = spec.add_actuator(name=f"v5_torque_{name}", target=name)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = a.gear[0] = 1
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.ctrllimited = a.forcelimited = True
        a.ctrlrange[:] = a.forcerange[:] = (-float(limit), float(limit))
    return spec.compile()


class RobotAdapter:
    def __init__(self, model=None, scene=None, physics_dt=0.002):
        self.spec = physical_spec(physics_dt)
        if model is not None and scene is not None:
            raise ValueError("pass either compiled model or SceneSpec, never both")
        self.scene = scene
        self.model = model if model is not None else compile_model(self.spec, scene)
        m = self.model
        self.joints = np.array([self.id(mujoco.mjtObj.mjOBJ_JOINT, n)
                               for n in self.spec.low_level_joint_names])
        self.qpos_ids = m.jnt_qposadr[self.joints].copy()
        self.dof_ids = m.jnt_dofadr[self.joints].copy()
        self.actuator_ids = np.array([self.id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"v5_torque_{n}")
                                     for n in self.spec.low_level_joint_names])
        self.base_joint = self.id(mujoco.mjtObj.mjOBJ_JOINT, self.spec.base_joint_name)
        self.base_qpos = np.arange(m.jnt_qposadr[self.base_joint], m.jnt_qposadr[self.base_joint] + 7)
        self.base_dofs = np.arange(m.jnt_dofadr[self.base_joint], m.jnt_dofadr[self.base_joint] + 6)
        self.base_body = int(m.jnt_bodyid[self.base_joint])
        self.tip_c = self.id(mujoco.mjtObj.mjOBJ_BODY, self.spec.continuum_tip_body_name)
        self.tip_r = self.id(mujoco.mjtObj.mjOBJ_BODY, self.spec.rigid_tip_body_name)
        self.B = self.spec.planner_to_low_level
        self.Bplus = self.spec.low_level_to_planner
        self.calls = {"mass_matrix": 0, "fk": 0, "forward": 0, "distance": 0, "coupled": 0,
                      "physics_steps": 0, "rrt": 0, "sqp": 0, "torque_rollout": 0}
        self.shape_spec = default_continuum_model_spec(self.spec)
        self.lower = self.spec.planner_lower.copy()
        self.upper = self.spec.planner_upper.copy()
        self.lower[:10] = np.maximum(self.lower[:10], self.shape_spec.work_domain_lower_rad)
        self.upper[:10] = np.minimum(self.upper[:10], self.shape_spec.work_domain_upper_rad)
        self.index_map = {n: {"joint": int(j), "qpos": int(q), "dof": int(v), "actuator": int(a)}
                          for n, j, q, v, a in zip(self.spec.low_level_joint_names, self.joints,
                                                 self.qpos_ids, self.dof_ids, self.actuator_ids)}
        assert len(self.actuator_ids) == 67 and len(set(self.dof_ids)) == 67
        assert int(m.jnt_type[self.base_joint]) == int(mujoco.mjtJoint.mjJNT_FREE)

    def id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"missing model object {name}")
        return int(index)

    def new_data(self):
        d = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qpos_ids] = self.B @ self.spec.planner_zero
        self.forward(d)
        return d

    def forward(self,data):
        self.calls["forward"] += 1
        mujoco.mj_forward(self.model,data)

    def ee(self, data):
        poses = np.tile(np.eye(4), (2, 1, 1))
        for i, body in enumerate((self.tip_c, self.tip_r)):
            poses[i, :3, :3] = data.xmat[body].reshape(3, 3)
            poses[i, :3, 3] = data.xpos[body]
        poses[0, :3, 3] += poses[0, :3, :3] @ np.array([0.0475, 0., 0.])
        return poses

    def mass(self, data):
        self.calls["mass_matrix"] += 1
        mass = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass, data.qM)
        return mass

    def kinematics(self, data, mass=False):
        self.calls["fk"] += 1
        mujoco.mj_kinematics(self.model, data)
        if mass:
            mujoco.mj_comPos(self.model, data)
            mujoco.mj_crb(self.model, data)

    def identity(self):
        # Adapter model is compiled once and is immutable by contract. Scene
        # changes construct a fresh adapter. Avoid serializing the 564 MB mesh
        # bundle at every scratch-state reset during numerical optimization.
        if hasattr(self, "_identity"):
            return self._identity
        m = self.model
        # Include complete compiled dynamics and geometry in addition to source
        # assets. Binary model bytes are persisted by the audit command.
        packed = np.empty(mujoco.mj_sizeModel(m), dtype=np.uint8)
        mujoco.mj_saveModel(m, None, packed)
        import hashlib
        self._identity = {"robot": self.spec.identity().to_dict(),
                # Hash the contiguous buffer directly; tobytes() duplicates the
                # entire ~538 MiB model without changing any hash input bytes.
                "scene_model_hash": hashlib.sha256(memoryview(packed)).hexdigest(),
                "nq": m.nq, "nv": m.nv, "nu": m.nu, "joint_index_map": self.index_map,
                "planner_lower": self.lower, "planner_upper": self.upper,
                "base_qpos_indices": self.base_qpos, "base_dof_indices": self.base_dofs,
                "quaternion_order": "wxyz", "angle_unit": "rad", "dynamic_support": "NOT_VALIDATED"}
        return self._identity
