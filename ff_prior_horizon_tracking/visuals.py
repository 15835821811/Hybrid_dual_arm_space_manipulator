"""Saved-state media for the continuous static + dynamic episode, no new physics."""
import json,csv,shutil,subprocess,time
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
from ff_prior_horizon_restore.visuals import savefig,HTML
from .media_geometry import add_saved_overlays,JS_OVERLAYS
from .protocol import SOURCE
P=dict(render_width=960,render_height=720,render_fps=15)

def render_video(out,robot,arrays,goals,indices,view,deadline,notify):
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFMPEG_NOT_FOUND')
    width,height=P['render_width'],P['render_height']
    m=robot.model;m.vis.global_.offwidth=max(m.vis.global_.offwidth,width)
    m.vis.global_.offheight=max(m.vis.global_.offheight,height)
    d=mujoco.MjData(m);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    xyz=np.concatenate([arrays['ee'][:,:3,:3,3].reshape(-1,3),arrays['base'][:,:3]],axis=0)
    center=(np.min(xyz,axis=0)+np.max(xyz,axis=0))/2
    camera.lookat[:]=center;camera.distance=max(2.,float(np.ptp(xyz,axis=0).max())*1.8)
    camera.azimuth=125 if view=='overview' else 90;camera.elevation=-25 if view=='overview' else -85
    option=mujoco.MjvOption();option.geomgroup[:]=1
    target=out/f'{view}.mp4'
    argv=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(P['render_fps']),'-i','-',
        '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(target)]
    proc=None
    with mujoco.Renderer(m,height=height,width=width) as renderer:
        try:
            proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            for j,k in enumerate(indices):
                if time.time()>=deadline:raise TimeoutError('VISUALIZATION_WALL_BUDGET')
                d.qpos[:]=arrays['qpos'][k];d.qvel[:]=arrays['qvel'][k];d.time=float(arrays['time'][k])
                mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera,scene_option=option)
                trajectory_indices=np.unique(np.linspace(0,len(arrays['time'])-1,min(400,len(arrays['time']))).round().astype(int))
                add_saved_overlays(renderer.scene,arrays['ee'],goals[k],int(k),trajectory_indices)
                for arm in range(2):
                    scn=renderer.scene
                    if scn.ngeom<scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3,.012),goals[k,arm,:3,3],np.eye(3).ravel(),
                            np.array([.95,.2,.2,1.] if arm else [.1,.75,.95,1.],dtype=np.float32))
                        scn.ngeom+=1
                frame=renderer.render();proc.stdin.write(frame.tobytes())
                if j in (0,len(indices)//2,len(indices)-1):plt.imsave(out/f'{view}_{j:03d}.png',frame)
                if j%50==0:notify('render_saved_states',view=view,frame=j+1,total=len(indices))
            proc.stdin.close();proc.stdin=None
            _,error=proc.communicate(timeout=60)
            if proc.returncode:raise RuntimeError(error.decode('utf-8',errors='replace'))
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate();proc.wait(timeout=10)
    return dict(file=target.name,frames=len(indices),fps=P['render_fps'],
        recorded_duration_s=float(arrays['time'][-1]-arrays['time'][0]),
        playback_duration_s=len(indices)/P['render_fps'],new_native_steps=0)

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
    axs[2,1].plot(t,np.max(abs(a['integration']-b['integration']),axis=1));axs[2,1].axhline(1e-10,color='red',ls='--');axs[2,1].set(title='Full INTEGRATION independent replay error')
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
    axs[0].plot(t,np.max(abs(a['integration']-b['integration']),axis=1))
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
    clearance=np.r_[previous['clearance_m'],a['clearance'][1:]];fp,_=pose_errors(full['ee'],goals)
    indices=np.unique(np.linspace(0,len(full['time'])-1,min(820,len(full['time']))).round().astype(int))
    data=mujoco.MjData(robot.model);bodies=[]
    for k in indices:
        data.qpos[:]=full['qpos'][k];data.qvel[:]=full['qvel'][k];mujoco.mj_forward(robot.model,data);bodies.append(data.xpos.copy())
    payload=dict(time=full['time'][indices].tolist(),bodies=np.asarray(bodies).tolist(),parents=robot.model.body_parentid.tolist(),
        goals=goals[indices,:,:3,3].tolist(),ee=full['ee'][indices,:,:3,3].tolist(),ee_poses=full['ee'][indices].tolist(),goal_poses=goals[indices].tolist(),obstacles=contract['scene']['objects'],
        position_mm=(fp[indices]*1000).tolist(),clearance_mm=(clearance[indices]*1000).tolist(),execution_level=result['status'])
    static_figures=[]
    for source in sorted((SOURCE/'units/u00/visuals').glob('0*')):
        if source.suffix in ('.png','.pdf'):
            shutil.copy2(source,out/('static_'+source.name))
            if source.suffix=='.png':static_figures.append('static_'+source.stem)
    videos=[];errors=[]
    for view in ('overview','top'):
        try:videos.append(render_video(out,robot,full,goals,indices,view,deadline,notify))
        except Exception as exc:errors.append(dict(view=view,error=repr(exc)))
    html=HTML.replace('dot(D.goals[k]','dot(D.goals[i][k]').replace('__OVERLAY_JS__',JS_OVERLAYS)
    html=html.replace('原固定五障碍、原世界目标与27秒运动。动画来自实际保存状态；动态连续交接尚属后续阶段。系统通过不等于裸网络独立通过，不进行相对优势评判。',
        '原五障碍完整静态接近＋实际终态接原七不规则航点。27.5秒处为真实连续交接，动态段等级见下。视频仅实际已记录前缀；若失败并未伪造完成。原五障碍贯穿两段；卫星继承零速度，尚未证明实体漂移。动态段由反馈控制器完成，不归因于网络。')
    html=html.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).replace('__STATUS__',result['status'])
    html=html.replace('__VIDEOS__',''.join(f'<h2>{v["file"]}</h2><video src="{v["file"]}" controls loop muted></video>' for v in videos))
    html=html.replace('__GALLERY__',''.join(f'<section><h2>{n}</h2><a href="{n}.pdf">PDF</a><img src="{n}.png"></section>' for n in ['01_dynamic_summary','02_joint_motion','03_tracking_momentum','04_world_paths','05_all_torques','06_base_manifold','07_world_coordinates','08_replay_geometry']+static_figures))
    (out/'index.html').write_text(html,encoding='utf-8')
    manifest=dict(status='COMPLETE' if len(videos)==2 else 'PARTIAL_RENDER_ERRORS',videos=videos,errors=errors,
        source_dynamic_hash=file_digest(run/'execution/native_trace.npz'),source_static_hash=contract['source_native_sha256'],
        native_steps_for_visualization=0,overlays=dict(actual_ee_frames=True,time_varying_goal_frames=True,actual_ee_trajectories=True,axis_colors='RGB = XYZ'),combined_actual_duration_s=float(full['time'][-1]),
        files={p.name:dict(sha256=file_digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    atomic_json(out/'manifest.json',manifest);return manifest

