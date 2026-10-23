"""Same 836-state handoff. Only forces and robot controls are actual inputs."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from ff_prior_balanced_brake.environment import Environment as PreviousEnvironment
from .protocol import P
from .force import force_at,satellite_state
from .diagnostics import BodyObstacleRecorder
from .compact_storage import json_size

class Environment(PreviousEnvironment):
    def __init__(self,robot,terminal,contract,budget,record_initial=True):
        self.satellite_history=[];self.force_inputs=[];self.qfrc_inputs=[]
        self.force_points=[];self.impulse=[np.zeros(3)];self.angular_impulse=[np.zeros(3)]
        self.work=[0.];self.full_state_restore_count=1
        super().__init__(robot,terminal,contract,budget,record_initial=False)
        self.target_id=self.body_ids[2]
        self.body_recorder=BodyObstacleRecorder(robot,contract['scene'],source_mode='actual_control_tick',deadline=budget.physics_deadline if budget else None)
        self.json_estimate=0;self.estimated_events=0
        if record_initial:self.record()

    def reference(self,t):
        goal=np.asarray(self.contract['hold_continuum_pose'])
        pt,rt=self.pose(2);lead=float(t-self.data.time)
        v=self.data.qvel[self.tv.start:self.tv.start+3]
        w=self.data.qvel[self.tv.start+3:self.tv.stop]
        future=rt@Rotation.from_rotvec(w*lead).as_matrix()
        return (goal[:3,3],goal[:3,:3],np.zeros(3),np.zeros(3),
                pt+v*lead+future@self.grasp_local,future@self.grasp_rotation_local,
                v+np.cross(future@w,future@self.grasp_local),future@w)

    def record(self):
        before=len(self.integration_history)
        try:super().record()
        finally:
            if len(self.integration_history)>before:
                self.satellite_history.append(satellite_state(self.robot,self.data,self.target_id))

    def recording_guard(self):
        # Conservative uncompressed envelope, including the new tick before it
        # advances physics. Fail closed instead of discarding active records.
        ticks=len(self.clearance_observations)
        rows=sum(len(o['row_pairs']) for o in self.clearance_observations)
        native=len(self.controls)
        for i in range(self.estimated_events,len(self.filter_history)):
            self.json_estimate+=json_size(self.filter_history[i])+json_size(self.body_recorder.events[i])+4
        self.estimated_events=len(self.filter_history)
        estimate=64*1024**2+native*26000+ticks*len(self.pairs)*18+rows*240+self.json_estimate+len(self.collision.states)*1000
        if rows>P['max_active_rows'] or estimate>P['max_recording_estimate_bytes']:
            raise RuntimeError('RECORDING_RESERVATION_EXHAUSTED')

    def advance_native(self,velocity):
        self.recording_guard()
        r,d=self.robot,self.data;old=self.command.copy();acceleration=(velocity-old)/.02
        for substep in range(10):
            v=old+(substep+1)/10*(velocity-old);next_reference=self.reference_q+.002*v
            r.forward(d);mass=r.mass(d);w=np.r_[np.full(60,42.),np.full(7,34.)]
            aref=r.B@acceleration+w*w*(r.B@next_reference-d.qpos[r.qpos_ids])+2*w*(r.B@v-d.qvel[r.dof_ids])
            aref=np.clip(aref,-np.r_[np.full(60,45.),np.full(7,70.)],np.r_[np.full(60,45.),np.full(7,70.)])
            b,a=r.base_dofs,r.dof_ids;force=d.qfrc_bias-d.qfrc_passive
            ba=-np.linalg.solve(mass[np.ix_(b,b)],mass[np.ix_(b,a)]@aref+force[b])
            raw=mass[np.ix_(a,b)]@ba+mass[np.ix_(a,a)]@aref+force[a]
            if not np.isfinite(raw).all():raise RuntimeError('NONFINITE_TORQUE')
            if np.any(abs(raw)>r.spec.torque_limits+1e-12):raise RuntimeError('RAW_TORQUE_LIMIT')
            # No qpos/qvel/time writes. The first full snapshot is preserved.
            self.budget.reserve('native');self.reference_q=next_reference
            d.ctrl[r.actuator_ids]=raw
            d.xfrc_applied[:]=0.
            f=force_at((len(self.controls)+.5)*P['dt'])
            d.xfrc_applied[self.target_id,:3]=f
            if np.any(d.qfrc_applied):raise RuntimeError('UNAUTHORIZED_GENERALIZED_FORCE')
            point=d.xipos[self.target_id].copy();before=satellite_state(r,d,self.target_id)
            self.force_inputs.append(d.xfrc_applied.copy());self.qfrc_inputs.append(d.qfrc_applied.copy())
            self.force_points.append(point)
            self.controls.append(d.ctrl.copy());self.raw.append(raw.copy())
            self.command_qd.append(v.copy());self.command_qdd.append(acceleration.copy())
            mujoco.mj_step(r.model,d);self.budget.complete('native')
            try:self.record()
            finally:
                after=satellite_state(r,d,self.target_id)
                self.impulse.append(self.impulse[-1]+P['dt']*f)
                self.angular_impulse.append(self.angular_impulse[-1]+P['dt']*np.cross(point,f))
                self.work.append(self.work[-1]+P['dt']*(f@((before[12:15]+after[12:15])*.5)))
            if np.any(d.warning.number):raise RuntimeError('MUJOCO_WARNING')
        self.command=velocity.copy()
