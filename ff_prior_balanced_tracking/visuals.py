"""Saved-state media for the continuous static + dynamic episode, no new physics."""
import csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
from ff_prior_horizon_restore.visuals import savefig
from ff_prior_balanced_evidence.media import generate_episode_media
from .protocol import SOURCE
def generate(run,robot,contract,notify,deadline):
    out=run/'visuals';out.mkdir(exist_ok=False)
    a=dict(np.load(run/'execution/native_trace.npz'));b=np.load(run/'execution/replay_trace.npz')
    t=a['time'];local=t-t[0];pos,rot=pose_errors(a['ee'],a['desired']);files=[]
    result=read_json(run/'execution/result.json')
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    axs[0,0].plot(t,1000*pos);axs[0,0].axhline(5,color='red',ls='--');axs[0,0].set(title='Dynamic world position error (gate at ALL times)',ylabel='mm')
    axs[0,1].plot(t,np.rad2deg(rot));axs[0,1].axhline(1,color='red',ls='--');axs[0,1].set(title='Dynamic world orientation error',ylabel='degree')
    axs[1,0].plot(t,a['clearance']*1000);axs[1,0].axhline(0,color='red');axs[1,0].set(title='Whole-body clearance',ylabel='mm')
    axs[1,1].plot(t,(a['base'][:,:3]-a['base'][0,:3])*1000);axs[1,1].set(title='Free-base displacement after handoff',ylabel='mm')
    if len(a['raw_torque']):axs[2,0].plot(t[:-1],np.max(abs(a['raw_torque'])/robot.spec.torque_limits,axis=1))
    axs[2,0].axhline(1,color='red',ls='--');axs[2,0].set(title='Raw torque / limit')
    axs[2,1].plot(t[:len(b['integration'])],np.max(abs(a['integration'][:len(b['integration'])]-b['integration']),axis=1));axs[2,1].axhline(1e-10,color='red',ls='--');axs[2,1].set(title='Full INTEGRATION independent replay error')
    for ax in axs.flat:ax.set_xlabel('absolute simulation time (s)');ax.grid(alpha=.2)
    savefig(fig,out,'01_dynamic_summary',files)
    fig,axs=plt.subplots(3,1,figsize=(13,10));v=a['qvel'][:,robot.dof_ids]@robot.Bplus.T
    axs[0].plot(t,a['q17']);axs[0].set(title='All 17 actual coordinates',ylabel='rad')
    axs[1].plot(t,v/robot.spec.planner_velocity_limits);axs[1].set(ylabel='velocity / limit')
    axs[2].plot(t[1:],np.diff(v,axis=0)/.002/robot.spec.planner_acceleration_limits);axs[2].set(ylabel='acceleration / limit',xlabel='absolute time (s)')
    for ax in axs[1:]:ax.axhline(1,color='red',ls='--');ax.axhline(-1,color='red',ls='--')
    savefig(fig,out,'02_joint_motion',files)
    fig,axs=plt.subplots(2,1,figsize=(13,7));axs[0].plot(t,a['q17']-a['qref']);axs[0].set(title='Actual minus commanded reference',ylabel='rad')
    axs[1].plot(t,a['momentum']);axs[1].set(title='Free-base generalized momentum',xlabel='absolute time (s)')
    savefig(fig,out,'03_tracking_momentum',files)
    fig=plt.figure(figsize=(12,7));ax=fig.add_subplot(111,projection='3d')
    for arm in range(2):
        ax.plot(*a['desired'][:,arm,:3,3].T,ls='--',label=f'desired arm{arm}')
        ax.plot(*a['ee'][:,arm,:3,3].T,label=f'actual arm{arm}')
    points=np.asarray(contract['waypoints']['waypoint_points_m']);ax.scatter(*points.T,marker='*',s=60,label='original 7 waypoints')
    ax.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='world z (m)',title='Frozen world waypoints and actual tracking');ax.legend()
    savefig(fig,out,'04_world_paths',files)
    fig,axs=plt.subplots(2,1,figsize=(13,8))
    axs[0].plot(t[:-1],a['raw_torque']);axs[0].set(title='All 67 actual raw torques',ylabel='N m')
    axs[1].plot(t[:-1],a['raw_torque']/robot.spec.torque_limits);axs[1].axhline(1,color='red',ls='--');axs[1].axhline(-1,color='red',ls='--')
    axs[1].set(title='All 67 raw torque / original limit',xlabel='absolute time (s)')
    savefig(fig,out,'05_all_torques',files)
    fig,axs=plt.subplots(3,1,figsize=(13,10))
    axs[0].plot(t,a['base'][:,:3]);axs[0].set(title='Free base world translation',ylabel='m')
    axs[1].plot(t,a['base'][:,3:]);axs[1].set(title='Free base orientation quaternion')
    axs[2].plot(t,np.max(abs(a['qpos'][:,robot.qpos_ids]-a['q17']@robot.B.T),axis=1))
    axs[2].set(title='67-to-17 manifold residual',ylabel='rad',xlabel='absolute time (s)')
    savefig(fig,out,'06_base_manifold',files)
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    for arm in range(2):
        for axis in range(3):
            axs[axis,arm].plot(t,a['ee'][:,arm,axis,3],label='actual')
            axs[axis,arm].plot(t,a['desired'][:,arm,axis,3],'--',label='target')
            axs[axis,arm].set(title=f'Arm {arm} world {"XYZ"[axis]}',ylabel='m',xlabel='absolute time (s)')
            axs[axis,arm].legend()
    savefig(fig,out,'07_world_coordinates',files)
    fig,axs=plt.subplots(2,1,figsize=(13,7))
    axs[0].plot(t[:len(b['integration'])],np.max(abs(a['integration'][:len(b['integration'])]-b['integration']),axis=1))
    axs[0].set(title='All 836 INTEGRATION components: maximum replay difference at each state')
    geometry=np.load(run/'execution/geometry.npz')
    axs[1].plot(np.arange(len(geometry['minima'])),geometry['minima']*1000)
    axs[1].axhline(0,color='red');axs[1].set(title='Every native and interpolated full-machine geometry scan',xlabel='saved scan index',ylabel='mm')
    savefig(fig,out,'08_replay_geometry',files)
    with (out/'timeseries.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['absolute_time_s','dynamic_local_time_s','c_position_mm','r_position_mm','c_rotation_deg','r_rotation_deg','clearance_mm'])
        for i in range(len(t)):w.writerow([t[i],local[i],*(pos[i]*1000),*np.rad2deg(rot[i]),a['clearance'][i]*1000])
    previous=np.load(SOURCE/'units/u00/execution/native_trace.npz')
    assert np.array_equal(previous['integration'][-1],a['integration'][0]),'MEDIA_HANDOFF_NOT_IDENTICAL'
    full={k:np.concatenate((previous[k],a[k][1:]),axis=0) for k in ('time','qpos','qvel','ee','base')}
    goals=np.concatenate((np.repeat(np.asarray(contract['static_goals'])[None],len(previous['time']),axis=0),a['desired'][1:]))
    full['clearance_m']=np.r_[previous['clearance_m'],a['clearance'][1:]]
    static_figures=[]
    for source in sorted((SOURCE/'units/u00/visuals').glob('0*')):
        if source.suffix in ('.png','.pdf'):
            shutil.copy2(source,out/('static_'+source.name))
            if source.suffix=='.png':static_figures.append('static_'+source.stem)
    rendered=generate_episode_media(out,robot,full,goals,contract['scene'],
        contract['waypoints']['waypoint_points_m'],
        [SOURCE/'units/u00/body_diagnostics',run/'execution'],deadline,notify,
        result['status'],'新七障碍：完整静态接近与真实末态接原七不规则航点',max_frames=820)
    manifest=dict(rendered,source_dynamic_hash=file_digest(run/'execution/native_trace.npz'),
        source_static_hash=contract['source_native_sha256'],native_steps_for_visualization=0,
        combined_actual_duration_s=float(full['time'][-1]-full['time'][0]),
        actual_handoff_full836_exact=True,moving_satellite_claim=False,
        waypoint_times_rotations_preserved=True,geometric_reference_not_network_autonomy=True,
        files={p.name:dict(sha256=file_digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    atomic_json(out/'manifest.json',manifest);return manifest
