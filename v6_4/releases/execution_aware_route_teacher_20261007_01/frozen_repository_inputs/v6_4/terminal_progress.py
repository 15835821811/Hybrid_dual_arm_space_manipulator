"""Disclosed terminal progress repair with a target that keeps physical time.

The frozen path and weights are unchanged. Private nominal configurations are
only a gate input; runtime Cartesian target registration reads current feedback
through an independent MjData, and never overwrites execution state or caches.
"""
from __future__ import annotations
import numpy as np
import mujoco
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from v6_4.reference_adapter import _RotationReference, _pose
from v6_4.task_protocol import canonical_json
from v6_lite.hierarchical_qp import joint_addresses, free_joint_slices, CONTINUUM_EE_OFFSET_M

REFERENCE_MODE='terminal_progress_v1'
PROGRESS_START_S=24.0
PROGRESS_END_S=26.5
DURATION_S=27.0


def terminal_progress(times):
    """Fixed C2 quintic, including only the original 1ns endpoint roundoff."""
    t=np.asarray(times,dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t< -1e-9) or np.any(t>27.+1e-9):
        raise ValueError('terminal progress time must be finite within the 27s horizon')
    t=np.minimum(27.,np.maximum(0.,t));u=np.clip((t-24.)/2.5,0.,1.)
    s=np.where(t<=24.,t,24.+2.5*u+15.*u**3-25.*u**4+10.5*u**5)
    ds=np.where(t<=24.,1.,(1.-u)**2*(21.*u*u+2.*u+1.))
    dds=np.where((t>24.)&(t<26.5),(90.*u-300.*u*u+210.*u**3)/6.25,0.)
    s=np.where(t>=26.5,27.,s);ds=np.where(t>=26.5,0.,ds)
    # The monotone polynomial is mathematically in [0,27]. Cancellation at
    # u~=1 can overshoot by a few ulps; use only the existing endpoint rule.
    if np.any(s< -1e-9) or np.any(s>27.+1e-9):raise ValueError('terminal polynomial escaped its declared horizon')
    s=np.minimum(27.,np.maximum(0.,s))
    if t.ndim==0:return float(s),float(ds),float(dds)
    return s,ds,dds


def register_rigid_reference(base_sample,nominal_target,current_target,ds):
    """All angular rates are world-frame; target position is body origin."""
    pn,vn,rn,wn=nominal_target;pa,va,ra,wa=current_target
    offset=np.asarray(base_sample['rigid_target_position'])-pn
    relative=rn.T@offset
    relative_rate=rn.T@(np.asarray(base_sample['rigid_target_velocity'])-vn-np.cross(wn,offset))
    relative_angular=rn.T@(np.asarray(base_sample['rigid_target_angular_velocity'])-wn)
    world_offset=ra@relative
    return {'rigid_target_position':pa+world_offset,
        'rigid_target_velocity':va+np.cross(wa,world_offset)+ra@relative_rate*ds,
        'rigid_target_rotation':ra@rn.T@np.asarray(base_sample['rigid_target_rotation']),
        'rigid_target_angular_velocity':wa+ra@relative_angular*ds}


class TerminalProgressReferenceProvider:
    """Wrap one prepared frozen spline with the fixed disclosed repair."""
    def __init__(self,task,base_provider):
        if task.sha256()!=base_provider.task.sha256():raise ValueError('terminal repair task identity mismatch')
        # Access requires the original frozen provider to have been prepared.
        self._base_prediction=base_provider.prediction
        self.task,self.base,self.codec=task,base_provider,base_provider.codec
        self.controls=base_provider.controls
        self._prediction=None;self._registered_prediction=None
        self._model=self._live_data=self._private_live_data=None
        self._target_position=self._target_rotation=None
        self.metadata={'schema':'v64_a1_terminal_progress_reference_v1','reference_mode':REFERENCE_MODE,
            'task_sha256':task.sha256(),'progress_start_s':24.,'progress_end_s':26.5,'path_horizon_s':27.,
            'terminal_hold_s':.5,'maximum_progress_rate':640./343.,'not_pure_slowdown':True,
            'frozen_control_points_changed':False,'weights_changed':False,'explicit_repair':True,
            'future_actual_trace_read':False,'physics_steps_executed':0,
            'rigid_reconstruction':'nominal path relative to nominal target at s, registered to current actual target body origin at physical t; all transport derivatives included',
            'continuum_reconstruction':'original world Cartesian path at s; its time derivatives multiply ds/dt',
            'posture_reconstruction':'q_path(s), dq_path(s)*(ds/dt), ddq_path(s)*(ds/dt)^2+dq_path(s)*(d2s/dt2)',
            'nominal_gate_reconstruction':'base old private path pose(s), exact arm q_path(s), physical-time target pose(t), recomputed zero-momentum base rates; fresh FK',
            'nominal_fk_is_registered_runtime_cartesian':False,
            'live_state_and_cache_modified':False}

    @property
    def prediction(self):
        if self._prediction is None:raise RuntimeError('terminal repair has not been prepared')
        return dict(self._prediction)

    def posture_at_physical_time(self,times):
        s,ds,dds=terminal_progress(times)
        s=np.atleast_1d(s);ds=np.atleast_1d(ds);dds=np.atleast_1d(dds)
        value=self.codec.sample(self.controls,s)
        return {'q':value['q'],'dq':value['dq']*ds[:,None],
            'ddq':value['ddq']*ds[:,None]**2+value['dq']*dds[:,None]}

    def _nominal_target(self,time_s):
        rotation,angular=self._target_rotation.sample(float(time_s))
        return self._target_position(time_s),self._target_position(time_s,1),rotation,angular

    def prepare(self,spec,model,initial_data,scenario):
        if (spec.runtime_contract_sha256()!=self.task.model_contract_sha256
            or canonical_json(scenario.to_dict())!=canonical_json(self.task.scenario)
            or not np.array_equal(initial_data.qpos,self.task.initial_qpos)
            or not np.array_equal(initial_data.qvel,self.task.initial_qvel)
            or abs(float(initial_data.time))>1e-12):
            raise ValueError('terminal repair requires the original declared model/task/initial state')
        self.base.prepare(spec,model,initial_data,scenario)
        self._model,self._live_data=model,initial_data
        self._private_live_data=mujoco.MjData(model)
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.target_free_joint_name)
        if joint<0:raise ValueError('terminal repair target joint missing')
        self._target_body=int(model.jnt_bodyid[joint])
        if self._prediction is None:self._build_prediction(spec,model)
        return self

    def _build_prediction(self,spec,model):
        original=self._base_prediction;times=np.asarray(original['time'])
        data=mujoco.MjData(model);target_rates=[];target_velocities=[]
        for qpos,qvel in zip(original['full_qpos'],original['full_qvel']):
            data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data)
            _,velocity,_,angular=_pose(model,data,self._target_body,np.zeros(3))
            target_velocities.append(velocity);target_rates.append(angular)
        self._target_position=CubicHermiteSpline(times,original['target_position'],target_velocities,extrapolate=False)
        self._target_rotation=_RotationReference(times,original['target_rotation'],np.asarray(target_rates))
        base_position=CubicHermiteSpline(times,original['base_pose'][:,:3],original['base_twist'][:,:3],extrapolate=False)
        base_rotations=Rotation.from_quat(original['base_pose'][:,[4,5,6,3]].copy()).as_matrix()
        base_rotation=_RotationReference(times,base_rotations,original['base_twist'][:,3:])
        qids,vids=joint_addresses(model,spec)
        base_qpos,base_dofs=free_joint_slices(model,spec.base_joint_name)
        target_qpos,target_dofs=free_joint_slices(model,spec.target_free_joint_name)
        base_ids=np.arange(base_dofs.start,base_dofs.stop);mass=np.empty((model.nv,model.nv))
        target_twist=np.asarray(self.task.initial_qvel)[target_dofs]
        source_s,ds,dds=terminal_progress(times);posture=self.posture_at_physical_time(times)
        arms={a:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,getattr(spec,a+'_tip_body_name')) for a in ['rigid','continuum']}
        keys=['full_qpos','full_qvel','base_pose','base_twist','target_position','target_rotation','target_velocity','target_angular_velocity',
            'rigid_position','rigid_velocity','rigid_rotation','rigid_angular_velocity',
            'continuum_position','continuum_velocity','continuum_rotation','continuum_angular_velocity']
        recorded={k:[] for k in keys};registered={k:[] for k in ['target_position','target_rotation','rigid_position','rigid_rotation','continuum_position','continuum_rotation']}
        for index,(t,s) in enumerate(zip(times,source_s)):
            data.qpos[:]=self.task.initial_qpos
            data.qpos[qids]=spec.encode_position(posture['q'][index])
            br,_=base_rotation.sample(float(s));xyzw=Rotation.from_matrix(br).as_quat()
            data.qpos[base_qpos]=np.r_[base_position(s),xyzw[[3,0,1,2]]]
            tp,tv,tr,tw=self._nominal_target(float(t));xyzw=Rotation.from_matrix(tr).as_quat()
            data.qpos[target_qpos]=np.r_[tp,xyzw[[3,0,1,2]]]
            # Preserve the exact eliminated initial state, including quaternion signs.
            if index==0:data.qpos[:]=self.task.initial_qpos
            data.qvel[:]=0.;data.time=float(t);mujoco.mj_forward(model,data)
            mujoco.mj_fullM(model,mass,data.qM)
            base_map=-np.linalg.solve(mass[np.ix_(base_ids,base_ids)],mass[np.ix_(base_ids,vids)]@spec.planner_to_low_level)
            data.qvel[vids]=spec.encode_velocity(posture['dq'][index])
            data.qvel[base_dofs]=base_map@posture['dq'][index];data.qvel[target_dofs]=target_twist
            if index==0:data.qvel[:]=self.task.initial_qvel
            recorded['full_qpos'].append(data.qpos.copy());recorded['full_qvel'].append(data.qvel.copy())
            recorded['base_pose'].append(data.qpos[base_qpos].copy())
            # Rates are world-frame, independently read from this fresh private FK.
            base_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.base_joint_name)
            _,bv,_,bw=_pose(model,data,int(model.jnt_bodyid[base_joint]),np.zeros(3))
            recorded['base_twist'].append(np.r_[bv,bw])
            recorded['target_position'].append(tp);recorded['target_rotation'].append(tr)
            recorded['target_velocity'].append(tv);recorded['target_angular_velocity'].append(tw)
            for arm,body in arms.items():
                p,v,r,w=_pose(model,data,body,CONTINUUM_EE_OFFSET_M if arm=='continuum' else np.zeros(3))
                for k,val in [('position',p),('velocity',v),('rotation',r),('angular_velocity',w)]:recorded[arm+'_'+k].append(val)
            base_sample=self.base.sample(float(s))
            desired=register_rigid_reference(base_sample,self._nominal_target(float(s)),(tp,tv,tr,tw),float(ds[index]))
            registered['target_position'].append(tp);registered['target_rotation'].append(tr)
            registered['rigid_position'].append(desired['rigid_target_position']);registered['rigid_rotation'].append(desired['rigid_target_rotation'])
            registered['continuum_position'].append(base_sample['continuum_target_position']);registered['continuum_rotation'].append(base_sample['continuum_target_rotation'])
        prediction={'time':times.copy(),**posture,**{k:np.asarray(v) for k,v in recorded.items()},
            'source_progress_s':source_s,'progress_rate':ds,'progress_acceleration':dds,
            'reference_mode':np.asarray(REFERENCE_MODE),'progress_start_s':np.asarray(24.),'progress_end_s':np.asarray(26.5),
            'progress_source':np.asarray('frozen_selected_path_with_fixed_terminal_time_repair')}
        for key,value in prediction.items():
            if value.dtype.kind not in 'US' and not np.all(np.isfinite(value)):raise ValueError('terminal private prediction nonfinite '+key)
            value.setflags(write=False)
        self._prediction=prediction
        self._registered_prediction={'time':times.copy(),**{k:np.asarray(v) for k,v in registered.items()}}
        for value in self._registered_prediction.values():value.setflags(write=False)

    def sample(self,time_s):
        if self._live_data is None:raise RuntimeError('terminal repair not bound to current feedback')
        if not np.isfinite(time_s) or abs(float(time_s)-float(self._live_data.time))>1e-9:
            raise ValueError('terminal repair may only sample the current physical feedback time')
        s,ds,dds=terminal_progress(float(time_s));base=self.base.sample(s)
        private=self._private_live_data
        private.qpos[:]=self._live_data.qpos;private.qvel[:]=self._live_data.qvel;private.time=float(self._live_data.time)
        mujoco.mj_forward(self._model,private)
        current_target=_pose(self._model,private,self._target_body,np.zeros(3))
        result=register_rigid_reference(base,self._nominal_target(s),current_target,ds)
        posture=self.posture_at_physical_time(np.asarray([float(time_s)]))
        result.update(posture_reference_q=posture['q'][0],posture_reference_dq=posture['dq'][0],
            continuum_target_position=base['continuum_target_position'],continuum_target_velocity=base['continuum_target_velocity']*ds,
            continuum_target_rotation=base['continuum_target_rotation'],continuum_target_angular_velocity=base['continuum_target_angular_velocity']*ds)
        return result

    def gate_repair(self,spec):
        """One extra original-policy gate; never replace the frozen proposal gate."""
        from v6_4.proposal_gate import gate_nominal_prediction,requirement_results
        from v6_lite.continuum_model_spec import default_continuum_model_spec
        prediction=self.prediction;domain=default_continuum_model_spec(spec)
        checks={'unchanged_frozen_controls':np.array_equal(self.controls,self.base.controls),
            'original_q_range':bool(np.all(prediction['q']>=spec.planner_lower)&np.all(prediction['q']<=spec.planner_upper)),
            'original_control_point_domain':bool(np.all(self.controls>=spec.planner_lower)&np.all(self.controls<=spec.planner_upper)),
            'original_dq_limits':bool(np.all(np.abs(prediction['dq'])<=spec.planner_velocity_limits+1e-12)),
            'original_ddq_limits':bool(np.all(np.abs(prediction['ddq'])<=spec.planner_acceleration_limits+1e-12)),
            'original_PCC_domain':bool(np.all(prediction['q'][:,:10]>=domain.work_domain_lower_rad)&np.all(prediction['q'][:,:10]<=domain.work_domain_upper_rad))}
        registered=requirement_results(self.task,self._registered_prediction)
        checks['registered_nominal_desired_Task_requirements']=registered['passed']
        nominal=gate_nominal_prediction(self.task,prediction,spec) if all(checks.values()) else None
        checks['fresh_nominal_FK_Task_and_native_geometry']=bool(nominal and nominal['passed'])
        return {'schema':'v64_a1_terminal_progress_extra_gate_v1','passed':all(checks.values()),
            'checks':checks,'registered_nominal_desired_Task':registered,'nominal_FK_gate':nominal,
            'metadata':self.metadata,'changes_original_proposal_gate':False,
            'physics_steps_executed':0,'registered_runtime_reference_is_nominal_FK':False,
            'scope':'Additional fixed repair gate only. Original q/dq/ddq/PCC/Task windows and fresh 50Hz native pair policy remain. It is not an actual execution certificate.'}
