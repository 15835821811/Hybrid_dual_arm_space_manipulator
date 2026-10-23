"""Visualize saved native states only. Rendering never advances the simulation."""
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.coupled_predictor import pose_errors
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import P,ROOT
from .media_geometry import add_saved_overlays,JS_OVERLAYS


def savefig(fig,out,name,files):
    fig.tight_layout()
    for ext in ('png','pdf'):
        p=out/f'{name}.{ext}';fig.savefig(p,dpi=170,bbox_inches='tight');files.append(p.name)
    plt.close(fig)


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
                add_saved_overlays(renderer.scene,arrays['ee'],goals,int(k),indices)
                for arm in range(2):
                    scn=renderer.scene
                    if scn.ngeom<scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3,.012),goals[arm,:3,3],np.eye(3).ravel(),
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


def generate(run,robot,entry,notify,deadline,unit_id):
    unit=run/'units'/unit_id;out=unit/'visuals';out.mkdir(exist_ok=False)
    task=entry['task'];goals=np.asarray(task['condition']['goals']);files=[]
    native=unit/'execution/native_trace.npz'
    if not native.exists():
        result=dict(status='BLOCKED_NO_NATIVE_TRACE',new_native_steps=0,reason='No qualified candidate reached execution')
        atomic_json(out/'manifest.json',result)
        (out/'index.html').write_text('<meta charset="utf-8"><h1>本轮未进入实际执行</h1><p>保留原始失败；没有伪造完整运行动画。</p>',encoding='utf-8')
        return result
    a=dict(np.load(native))
    if len(a['time'])<=1:
        result=dict(status='BLOCKED_NO_NATIVE_STEPS',new_native_steps=0,source_native_hash=file_digest(native))
        atomic_json(out/'manifest.json',result)
        (out/'index.html').write_text('<meta charset="utf-8"><h1>未产生原生执行步</h1><p>仅保存初态；不生成运行动画。执行拒绝原因见实际报告。</p>',encoding='utf-8')
        return result
    t=a['time']-a['time'][0];pos,rot=pose_errors(a['ee'],goals)
    torque=a['raw_torque'];q=a['q_actual_17'];v=a['qvel'][:,robot.dof_ids]@robot.Bplus.T
    acc=np.diff(v,axis=0)/float(robot.model.opt.timestep);hold=t>=task['condition']['duration']-1e-9
    clearance=a.get('clearance_m',np.empty(0));report=read_json(unit/'execution/report.json')
    indices=np.unique(np.linspace(0,len(t)-1,min(P['render_frames'],len(t))).round().astype(int))
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    axs[0,0].plot(t,pos*1000);axs[0,0].axhline(5,color='red',ls='--');axs[0,0].set(title='World goal position error (gate applies during hold)',ylabel='mm')
    axs[0,1].plot(t,np.rad2deg(rot));axs[0,1].axhline(1,color='red',ls='--');axs[0,1].set(title='World goal orientation error',ylabel='degree')
    if len(clearance)==len(t):axs[1,0].plot(t,clearance*1000)
    axs[1,0].axhline(0,color='red');axs[1,0].set(title='Whole-body minimum clearance',ylabel='mm')
    axs[1,1].plot(t,1000*(a['base'][:,:3]-a['base'][0,:3]));axs[1,1].set(title='Free-base displacement',ylabel='mm')
    axs[2,0].plot(t[:-1],np.max(abs(torque)/robot.spec.torque_limits,axis=1));axs[2,0].axhline(1,color='red',ls='--');axs[2,0].set(title='Maximum raw torque / limit')
    axs[2,1].plot(t,np.linalg.norm(a['base_generalized_momentum'][:,:3],axis=1),label='linear')
    axs[2,1].plot(t,np.linalg.norm(a['base_generalized_momentum'][:,3:],axis=1),label='angular');axs[2,1].legend();axs[2,1].set(title='Base generalized momentum (units differ)')
    for ax in axs.flat:
        ax.set_xlabel('time (s)');ax.grid(alpha=.25);ax.axvspan(task['condition']['duration'],t[-1],alpha=.08,color='green')
    savefig(fig,out,'01_summary',files)
    fig,axs=plt.subplots(3,1,figsize=(13,10))
    axs[0].plot(t,q);axs[0].set(ylabel='rad',title='All 17 actual planner coordinates')
    axs[1].plot(t,v/robot.spec.planner_velocity_limits);axs[1].set(ylabel='velocity / limit')
    axs[2].plot(t[1:],acc/robot.spec.planner_acceleration_limits);axs[2].set(ylabel='acceleration / limit',xlabel='time (s)')
    for ax in axs[1:]:ax.axhline(1,color='red',ls='--');ax.axhline(-1,color='red',ls='--')
    for ax in axs:ax.grid(alpha=.2)
    savefig(fig,out,'02_joint_motion',files)
    fig,axs=plt.subplots(2,1,figsize=(13,7))
    im=axs[0].imshow((torque/robot.spec.torque_limits).T,aspect='auto',origin='lower',extent=[t[0],t[-1],0,67],cmap='coolwarm',vmin=-1,vmax=1)
    fig.colorbar(im,ax=axs[0],label='raw torque / limit');axs[0].set(title='All 67 actuator torques',ylabel='actuator')
    axs[1].plot(t[:-1],q[:-1]-a['reference_q']);axs[1].axhline(.02,color='red',ls='--');axs[1].axhline(-.02,color='red',ls='--');axs[1].set(title='Actual minus commanded reference',ylabel='rad',xlabel='time (s)');axs[1].grid(alpha=.2)
    savefig(fig,out,'03_torque_tracking',files)
    fig,axs=plt.subplots(2,1,figsize=(13,7));axs[0].plot(t,a['base'][:,3:]);axs[0].set(title='Free-base quaternion',ylabel='component')
    axs[1].plot(t,np.max(abs(a['manifold_residual']),axis=1));axs[1].set(title='17 to 67 manifold residual',ylabel='rad',xlabel='time (s)')
    for ax in axs:ax.grid(alpha=.2)
    savefig(fig,out,'04_base_manifold',files)
    fig=plt.figure(figsize=(12,6));ax=fig.add_subplot(111,projection='3d')
    pd=unit
    stages=[('initial guess',pd/'restore/initial.npz'),('restored',pd/'validated/prediction.npz')]
    for name,path in stages:
        if path.exists():
            x=np.load(path)['ee']
            for arm in range(2):ax.plot(*x[:,arm,:3,3].T,label=f'{name} arm{arm}',alpha=.65)
    for arm in range(2):ax.plot(*a['ee'][:,arm,:3,3].T,lw=2,label=f'executed arm{arm}');ax.scatter(*goals[arm,:3,3],marker='*',s=120)
    ax.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='world z (m)',title='Converted / physical guidance / executed world paths');ax.legend(fontsize=8)
    savefig(fig,out,'05_world_paths',files)
    replay=unit/'execution/replay_trace.npz'
    if replay.exists():
        b=np.load(replay);common=min(len(a['integration']),len(b['integration']));delta=np.max(abs(a['integration'][:common]-b['integration'][:common]),axis=1)
        fig,ax=plt.subplots(figsize=(12,4));ax.plot(t[:common],delta);ax.axhline(1e-10,color='red',ls='--');ax.set(title=f'Independent replay: {common}/{len(t)} states; max error across 836 components',xlabel='time (s)',ylabel='absolute error');ax.grid(alpha=.2)
        savefig(fig,out,'06_independent_replay',files)
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for arm in range(2):
        for j in range(3):
            ax=axs[arm,j];ax.plot(t,a['ee'][:,arm,j,3]);ax.axhline(goals[arm,j,3],color='red',ls='--');ax.set(title=f'Arm {arm} world {"xyz"[j]}',xlabel='time (s)',ylabel='m');ax.grid(alpha=.2)
    savefig(fig,out,'07_world_coordinates',files)
    with (out/'timeseries.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','arm0_position_mm','arm1_position_mm','arm0_rotation_deg','arm1_rotation_deg','clearance_mm','hold_gate_active'])
        for i in range(len(t)):w.writerow([t[i],*(pos[i]*1000),*np.rad2deg(rot[i]),clearance[i]*1000 if len(clearance)==len(t) else '',bool(hold[i])])
    from ff_prior_balanced_evidence.media import generate_episode_media
    layout=read_json(ROOT/'ff_prior_balanced_scene/evidence/layout.json')
    manifest=generate_episode_media(out,robot,a,goals,entry['scene'],layout['waypoints'],
        [unit/'body_diagnostics'],deadline,notify,report['level'],
        '2+5 same-scene static approach: actual 27.5 seconds',max_frames=P['render_frames'])
    manifest.update(source_native_hash=file_digest(native),
        labels='balanced 2R5C static approach; W1-W7 are upcoming dynamic targets, not visited in this stage',
        new_native_steps=0,new_training_updates=0)
    atomic_json(out/'manifest.json',manifest)
    return manifest
