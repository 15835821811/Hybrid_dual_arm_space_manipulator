"""Full-frequency force, impulse, angular momentum and work accounting."""
import numpy as np
import mujoco
from ff_prior_pilot_train.storage import atomic_npz,atomic_json
from .force import force_at,analytic_motion,STATE_COLUMNS
from .protocol import P

def save_ledger(env,directory):
    a=dict(time=np.asarray(env.history['time']),state=np.asarray(env.satellite_history),
        xfrc_applied=np.asarray(env.force_inputs),qfrc_applied=np.asarray(env.qfrc_inputs),
        force_point_world=np.asarray(env.force_points),impulse=np.asarray(env.impulse),
        angular_impulse=np.asarray(env.angular_impulse),work_J=np.asarray(env.work),
        input_midpoint_local_time=(np.arange(len(env.controls))+.5)*P['dt'])
    atomic_npz(directory/'satellite_ledger.npz',**a)
    atomic_json(directory/'satellite_ledger_metadata.json',dict(state_columns=STATE_COLUMNS,
        target_body_id=env.target_id,target_body_name='target_satellite',
        satellite_mass_kg=float(env.model.body_mass[env.target_id]),
        body_names=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(env.model.nbody)],
        inputs_apply_to_next_state=True,input_frame=P['force_frame'],sampling=P['force_sampling'],
        model='constant-mass rigid satellite with disclosed external propulsion input; no fuel/exhaust simulation',
        original_robot_momentum_gate_unchanged=True,full_state_restore_count=env.full_state_restore_count))
    return a

def ledger_gates(env,a):
    s=a['state'];n=len(env.controls);x=a['xfrc_applied'];qid=a['qfrc_applied'];tid=env.target_id
    aligned=s.shape==(n+1,25) and a['impulse'].shape==(n+1,3) and x.shape==(n,env.model.nbody,6)
    if not aligned:return dict(satellite_ledger_complete=False),dict(error='LEDGER_SHAPE_MISMATCH')
    expected=np.asarray([force_at(t) for t in a['input_midpoint_local_time']])
    other=x.copy();other[:,tid,:]=0
    linear=s[:,18:21]-s[0,18:21]-a['impulse']
    angular=s[:,21:24]-s[0,21:24]-a['angular_impulse']
    work=s[:,24]-s[0,24]-a['work_J']
    mass=float(env.model.body_mass[tid]);t=a['time']-a['time'][0]
    z,v=analytic_motion(t,mass);axis=np.asarray(P['force_axis_world'])
    expected_p=s[0,:3]+t[:,None]*s[0,12:15]+z[:,None]*axis
    expected_v=s[0,12:15]+v[:,None]*axis
    coast_index=int(round(P['burn_s']/P['dt']));coast=np.arange(n)>=coast_index
    full=n==P['steps_each'];maxabs=lambda y:float(np.max(abs(y))) if np.size(y) else 0.
    balance=maxabs(linear);abalance=maxabs(angular);wbalance=maxabs(work)
    perr=maxabs(s[:,:3]-expected_p);verr=maxabs(s[:,12:15]-expected_v)
    coast_change=maxabs(s[coast_index:,12:18]-s[coast_index,12:18]) if n>=coast_index else float('inf')
    gates=dict(satellite_ledger_complete=bool(aligned and all(np.isfinite(y).all() for y in a.values())),
        no_robot_external_wrench=bool(np.count_nonzero(other)==0 and np.count_nonzero(qid)==0),
        authorized_satellite_force=bool(maxabs(x[:,tid,:3]-expected)<=P['force_atol_N']
            and np.count_nonzero(x[:,tid,3:])==0 and maxabs(x[:,tid,:3])<=P['peak_force_N']),
        coast_no_external_wrench=bool(full and coast.any() and np.count_nonzero(x[coast])==0),
        satellite_linear_impulse_balance=balance<=P['linear_balance_atol'],
        satellite_angular_impulse_balance=abalance<=P['angular_balance_atol'],
        satellite_work_energy_balance=wbalance<=P['work_balance_atol_J'],
        satellite_analytic_origin=bool(perr<=P['analytic_position_atol_m'] and verr<=P['analytic_velocity_atol_m_s']),
        satellite_inertial_coast=bool(full and coast_change<=P['coast_velocity_atol_m_s']),
        actual_nonzero_satellite_motion=bool(full and np.linalg.norm(s[-1,12:15]-s[0,12:15])>.0019
            and np.linalg.norm(s[-1,:3]-s[0,:3])>.055),
        full_origin_and_coast=bool(full and abs(t[-1]-29.)<1e-9))
    report=dict(linear_balance_max_abs=balance,angular_balance_max_abs=abalance,
        work_energy_balance_max_abs_J=wbalance,analytic_position_max_abs_m=perr,
        analytic_velocity_max_abs_m_s=verr,coast_velocity_change_max_abs=coast_change,
        actual_displacement_m=s[-1,:3]-s[0,:3],initial_velocity_m_s=s[0,12:15],
        final_velocity_m_s=s[-1,12:15],impulse_Ns=a['impulse'][-1],
        angular_impulse_Nms=a['angular_impulse'][-1],work_J=a['work_J'][-1],
        original_no_external_wrench_scope='robot unchanged; satellite exception only during authorized 2s origin',
        threshold_source='pre-run frozen protocol; no post-run tuning')
    return gates,report
