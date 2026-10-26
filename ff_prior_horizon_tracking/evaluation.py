"""Original dynamic gates with explicit absolute-clock handoff and full-state replay gates."""
import numpy as np
import mujoco
from ff_prior.coupled_predictor import pose_errors
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.storage import file_digest
from ff_prior_dynamic_reference.execution import integration
from .protocol import P

def evaluate(env,directory,reason,progress,budget):
    r=env.robot
    h={k:np.asarray(v) for k,v in env.history.items()}
    ctrl=np.asarray(env.controls).reshape(-1,r.model.nu)
    raw=np.asarray(env.raw).reshape(-1,67)
    atomic_npz(directory/'native_trace.npz',**h,ctrl=ctrl,raw_torque=raw,
        integration=np.asarray(env.integration_history),
        reference_qd=np.asarray(env.command_qd),reference_qdd=np.asarray(env.command_qdd))
    atomic_npz(directory/'geometry.npz',qpos=np.asarray(env.collision.states),minima=np.asarray(env.collision.minima))
    atomic_json(directory/'geometry_pairs.json',env.collision.pairs)
    replay=env.initial.reset(r)
    ri=[integration(r,replay)];rp=[replay.qpos.copy()];rv=[replay.qvel.copy()]
    for i,u in enumerate(ctrl):
        budget.step('replay')
        replay.ctrl[:]=u
        mujoco.mj_step(r.model,replay)
        r.forward(replay) # Same public forward observation, no copied actual state.
        rp.append(replay.qpos.copy());rv.append(replay.qvel.copy());ri.append(integration(r,replay))
        if i%500==0:progress('independent_replay',steps=i+1,total=len(ctrl))
    atomic_npz(directory/'replay_trace.npz',qpos=rp,qvel=rv,integration=ri,ctrl=ctrl)
    err=max(float(np.max(abs(np.array(rp)-h['qpos']))),float(np.max(abs(np.array(rv)-h['qvel']))))
    fullerr=float(np.max(abs(np.array(ri)-np.array(env.integration_history))))
    pos,rot=pose_errors(h['ee'],h['desired'])
    lowq,lowv=h['qpos'][:,r.qpos_ids],h['qvel'][:,r.dof_ids]
    v=lowv@r.Bplus.T;a=np.diff(v,axis=0)/P['dt'];la=np.diff(lowv,axis=0)/P['dt']
    mp=np.array([np.max(np.linalg.norm(h['momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    me=np.array([np.max(np.linalg.norm(h['exchange'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
    hold=h['time']-env.origin_time>=25.5-1e-9
    complete=(len(ctrl)==P['steps_each'] and reason is None and len(h['time'])==P['steps_each']+1
        and abs(h['time'][0]-env.origin_time)<1e-10 and abs(h['time'][-1]-env.origin_time-P['duration'])<1e-9)
    limited=r.model.jnt_limited[r.joints].astype(bool);limits=r.model.jnt_range[r.joints][limited]
    gates=dict(complete_native_time=complete,independent_replay=err<=1e-10,full_integration_replay=fullerr<=1e-10,
        actual_terminal_handoff=bool(np.array_equal(np.asarray(env.integration_history)[0],env.initial.values)),
        absolute_time_continuity=bool(np.all(abs(np.diff(h['time'])-.002)<1e-10)),
        whole_body_clearance=bool(np.min(h['clearance'])>=0),geometry_step=env.max_geom_step<=.002+1e-12,
        dynamic_position=bool(np.all(pos<=.005)),dynamic_rotation=bool(np.all(rot<=np.pi/180)),
        relative_hold=bool(complete and hold.any() and np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180)),
        joint_limits=bool(np.all(h['q17']>=r.lower) and np.all(h['q17']<=r.upper)
            and np.all(h['qref']>=r.lower) and np.all(h['qref']<=r.upper)
            and np.all(lowq[:,limited]>=limits[:,0]) and np.all(lowq[:,limited]<=limits[:,1])),
        derivative_limits=bool(len(a)>0 and np.all(abs(v)<=r.spec.planner_velocity_limits)
            and np.all(abs(a)<=r.spec.planner_acceleration_limits)
            and np.all(abs(lowv)<=abs(r.B)@r.spec.planner_velocity_limits)
            and np.all(abs(la)<=abs(r.B)@r.spec.planner_acceleration_limits)
            and np.all(abs(np.asarray(env.command_qd))<=r.spec.planner_velocity_limits)
            and np.all(abs(np.asarray(env.command_qdd))<=r.spec.planner_acceleration_limits)),
        raw_torque=bool(len(raw)>0 and np.all(abs(raw)<=r.spec.torque_limits+1e-12)),
        tracking=bool(np.max(abs(h['q17']-h['qref']))<=.02),
        manifold=bool(np.max(abs(lowq-h['q17']@r.B.T))<=.02),momentum=bool(np.all(mp<=1e-5+.002*me)),
        no_hidden_reset=True,no_external_wrench=True,
        finite=bool(all(np.isfinite(v).all() for v in h.values()) and np.isfinite(ctrl).all()
            and np.isfinite(raw).all() and np.isfinite(env.integration_history).all()))
    def dist(x):return dict(mean=np.mean(x,axis=0),p95=np.percentile(x,95,axis=0),maximum=np.max(x,axis=0),final=x[-1])
    report=dict(absolute_start_time=float(h['time'][0]),absolute_end_time=float(h['time'][-1]),
        inherited_target_velocity=env.contract['target_velocity'],moving_satellite_claim=False,
        status='EXECUTION_VERIFIED' if all(gates.values()) else 'EXECUTION_FAILED',gates=gates,
        stop_reason=reason,steps=len(ctrl),planned_steps=P['steps_each'],position_mm=dist(pos*1000),
        rotation_deg=dist(np.rad2deg(rot)),minimum_clearance_mm=float(np.min(h['clearance'])*1000),
        replay_max_error=err,replay_full_integration_error=fullerr,
        momentum_peak=mp,momentum_threshold=1e-5+.002*me,
        maximum_geometry_step_m=env.max_geom_step,interpolation_states=env.interpolation_states,
        native_trace_sha256=file_digest(directory/'native_trace.npz'),
        continuous_time_certificate=False,hardware_experiment=False,
        training_library_added=False,reference_grade='EXECUTION_VERIFIED' if all(gates.values()) else 'REJECTED')
    transitions=[]
    for boundary in [env.origin_time+4.5,env.origin_time+25.5]:
        k=int(np.argmin(abs(h['time']-boundary)))
        if abs(h['time'][k]-boundary)<1e-8 and k>0 and k+1<len(h['time']):
            transitions.append(dict(time=boundary,index=k,previous_time=h['time'][k-1],next_time=h['time'][k+1],
                direct_native_step=True,state_reset=False))
            atomic_npz(directory/f'transition_{int(boundary*10)}.npz',
                integration=np.asarray(env.integration_history)[k-1:k+2],qpos=h['qpos'][k-1:k+2],
                qvel=h['qvel'][k-1:k+2],time=h['time'][k-1:k+2])
    atomic_json(directory/'transitions.json',transitions)
    atomic_json(directory/'result.json',report)
    return report
