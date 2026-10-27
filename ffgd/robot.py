"""Explicit physical adapter; imports only the pinned parameter authority.

Base tangent: world translation, body-frame angular velocity (MuJoCo).
q17 are absolute physical radians; rigid home is a configuration, not an offset.
"""
from pathlib import Path
import hashlib
import mujoco
import numpy as np
from model_test.robot_model_spec_v5 import RobotModelSpecV5
from .contracts import Snapshot

ROOT = Path(__file__).resolve().parents[1]


class Robot:
    def __init__(self,physics_dt=.002):
        self.spec = RobotModelSpecV5.from_urdf(
            ROOT / "dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf",
            simulation_timestep=physics_dt, continuum_kp=100., continuum_kd=5.,
            continuum_torque_limit=4., rigid_kp=200., rigid_kd=20., rigid_torque_limit=80.)
        s = mujoco.MjSpec.from_file(str(self.spec.source_urdf))
        s.compiler.fusestatic = False
        s.option.gravity[:] = 0
        s.option.timestep = self.spec.simulation_timestep
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.connected = [("base_link", "link_1")] + [(f"link_{i}", f"link_{i+1}") for i in range(1,30)]
        for i, (a,b) in enumerate(self.connected):
            s.add_exclude(name=f"v7_module_connection_{i:02d}", bodyname1=a, bodyname2=b)
        for name, arm, damp, limit in zip(self.spec.low_level_joint_names, self.spec.joint_armature, self.spec.joint_damping, self.spec.torque_limits):
            j = s.joint(name); j.armature = float(arm); j.damping = float(damp)
            a = s.add_actuator(name=f"v5_torque_{name}", target=name)
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = a.gear[0] = 1
            a.biastype = mujoco.mjtBias.mjBIAS_NONE
            a.ctrllimited = a.forcelimited = True
            a.ctrlrange[:] = a.forcerange[:] = (-float(limit), float(limit))
        self.model = m = s.compile()
        self.joints = np.array([m.joint(n).id for n in self.spec.low_level_joint_names])
        self.qpos_ids = m.jnt_qposadr[self.joints].copy()
        self.dof_ids = m.jnt_dofadr[self.joints].copy()
        self.actuator_ids = np.array([m.actuator(f"v5_torque_{n}").id for n in self.spec.low_level_joint_names])
        self.base_joint = m.joint(self.spec.base_joint_name).id
        self.base_body = int(m.jnt_bodyid[self.base_joint])
        self.base_qpos = np.arange(m.jnt_qposadr[self.base_joint], m.jnt_qposadr[self.base_joint]+7)
        self.base_dofs = np.arange(m.jnt_dofadr[self.base_joint], m.jnt_dofadr[self.base_joint]+6)
        self.B = self.spec.planner_to_low_level.copy()
        self.Bplus = self.spec.low_level_to_planner.copy()
        self.lower = self.spec.planner_lower.copy(); self.lower[:10] = np.maximum(self.lower[:10], -1.)
        self.upper = self.spec.planner_upper.copy(); self.upper[:10] = np.minimum(self.upper[:10], 1.)
        self.tips = [m.body(self.spec.continuum_tip_body_name).id, m.body(self.spec.rigid_tip_body_name).id]
        self._hash = None
        if m.nu != 67 or np.linalg.matrix_rank(self.B) != 17: raise ValueError("mapping mismatch")

    @property
    def model_hash(self):
        if self._hash is None:
            buf = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
            mujoco.mj_saveModel(self.model, None, buf)
            self._hash = hashlib.sha256(memoryview(buf)).hexdigest()
        return self._hash

    def new_data(self):
        d = mujoco.MjData(self.model)
        d.qpos[self.qpos_ids] = self.B @ self.spec.planner_zero
        mujoco.mj_forward(self.model, d)
        return d

    def snapshot(self, d):
        mask = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        value = np.empty(mujoco.mj_stateSize(self.model, mask))
        mujoco.mj_getState(self.model, d, value, mask)
        return Snapshot(self.model_hash, mask, value)

    def restore(self, snapshot):
        if snapshot.model_hash != self.model_hash: raise ValueError("snapshot model identity mismatch")
        if snapshot.state.size != mujoco.mj_stateSize(self.model, snapshot.mask): raise ValueError("state size mismatch")
        d = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, d, snapshot.state.copy(), snapshot.mask)
        # No mj_forward here: it changes warmstart in the integration snapshot.
        return d

    def mass(self, d):
        result = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, result, d.qM)
        return result

    def metadata(self):
        m = self.model
        return dict(model_hash=self.model_hash, source_bundle_sha256=self.spec.source_bundle_sha256(),
                    mujoco_version=mujoco.__version__, nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, ngeom=m.ngeom,
                    joint_names=list(self.spec.low_level_joint_names), B=self.B.tolist(), offset=np.zeros(67).tolist(),
                    qpos_ids=self.qpos_ids.tolist(), dof_ids=self.dof_ids.tolist(),
                    joint_limited=m.jnt_limited[self.joints].tolist(), joint_range=m.jnt_range[self.joints].tolist(),
                    planner_lower=self.lower.tolist(), planner_upper=self.upper.tolist(),
                    torque_limits=self.spec.torque_limits.tolist(), armature=m.dof_armature.tolist(), damping=m.dof_damping.tolist(),
                    base_tangent="world_linear_body_angular", quaternion="wxyz", angle_unit="radian",
                    state_mask=int(mujoco.mjtState.mjSTATE_INTEGRATION), state_length=mujoco.mj_stateSize(m,int(mujoco.mjtState.mjSTATE_INTEGRATION)))
