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
from ff_prior_diffusion_demo.visuals import savefig,HTML

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
    
    for ob in contract['scene']['objects']:
        ax.scatter(*ob['position'],marker='o',s=60,c='red')
    ax.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='world z (m)',title='Frozen world waypoints and actual tracking');ax.legend()
    savefig(fig,out,'04_world_paths',files)
    with (out/'timeseries.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['absolute_time_s','dynamic_local_time_s','c_position_mm','r_position_mm','c_rotation_deg','r_rotation_deg','clearance_mm'])
        for i in range(len(t)):w.writerow([t[i],local[i],*(pos[i]*1000),*np.rad2deg(rot[i]),a['clearance'][i]*1000])
    previous=np.load(run.parent/'static/units/u00/execution/native_trace.npz')
    full={k:np.concatenate((previous[k],a[k][1:]),axis=0) for k in ('time','qpos','qvel','ee','base')}
    goals=np.concatenate((np.repeat(np.asarray(contract['static_goals'])[None],len(previous['time']),axis=0),a['desired'][1:]))
    clearance=np.r_[previous['clearance_m'],a['clearance'][1:]];fp,_=pose_errors(full['ee'],goals)
    indices=np.unique(np.linspace(0,len(full['time'])-1,min(400,len(full['time']))).round().astype(int))
    data=mujoco.MjData(robot.model);bodies=[]
    for k in indices:
        data.qpos[:]=full['qpos'][k];data.qvel[:]=full['qvel'][k];mujoco.mj_forward(robot.model,data);bodies.append(data.xpos.copy())
    payload=dict(time=full['time'][indices].tolist(),bodies=np.asarray(bodies).tolist(),parents=robot.model.body_parentid.tolist(),
        goals=goals[indices,:,:3,3].tolist(),ee=full['ee'][indices,:,:3,3].tolist(),obstacles=contract['scene']['objects'],
        position_mm=(fp[indices]*1000).tolist(),clearance_mm=(clearance[indices]*1000).tolist(),execution_level=result['status'])
    videos=[];errors=[]
    for view in ('overview','top'):
        try:videos.append(render_video(out,robot,full,goals,indices,view,deadline,notify))
        except Exception as exc:errors.append(dict(view=view,error=repr(exc)))
    html=HTML.replace('dot(D.goals[k]','dot(D.goals[i][k]')
    html=html.replace('固定三障碍、原世界目标与27秒运动。动画来自真实保存状态；动态连续交接和五障碍尚属后续阶段。系统通过不等于裸网络独立通过，不进行相对优势评判。',
        '双侧障碍新场景：刚性臂3球＋连续体七航点之间6球，共9球。原七航点与时序保留，段间采用预定义C2绕行。27.5秒实际终态连续交接。视频只展示真实执行前缀；状态失败时不能称为合格。动态绕行由几何参考与反馈控制完成，不归因于网络；卫星继承零速度。')
    html=html.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).replace('__STATUS__',result['status'])
    html=html.replace('__VIDEOS__',''.join(f'<h2>{v["file"]}</h2><video src="{v["file"]}" controls loop muted></video>' for v in videos))
    html=html.replace('__GALLERY__',''.join(f'<section><h2>{n}</h2><a href="{n}.pdf">PDF</a><img src="{n}.png"></section>' for n in ('01_dynamic_summary','02_joint_motion','03_tracking_momentum','04_world_paths')))
    (out/'index.html').write_text(html,encoding='utf-8')
    manifest=dict(status='COMPLETE' if len(videos)==2 else 'PARTIAL_RENDER_ERRORS',videos=videos,errors=errors,
        source_dynamic_hash=file_digest(run/'execution/native_trace.npz'),source_static_hash=contract['source_native_sha256'],
        native_steps_for_visualization=0,combined_actual_duration_s=float(full['time'][-1]),
        files={p.name:dict(sha256=file_digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    atomic_json(out/'manifest.json',manifest);return manifest

