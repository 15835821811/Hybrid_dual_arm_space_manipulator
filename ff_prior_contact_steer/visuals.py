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
from .protocol import P


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
    a=dict(np.load(native));t=a['time']-a['time'][0];pos,rot=pose_errors(a['ee'],goals)
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
    stages=[('converted',pd/'before_repair/prediction.npz'),('guided',pd/'validated/prediction.npz')]
    for name,path in stages:
        if path.exists():
            x=np.load(path)['ee']
            for arm in range(2):ax.plot(*x[:,arm,:3,3].T,label=f'{name} arm{arm}',alpha=.65)
    for arm in range(2):ax.plot(*a['ee'][:,arm,:3,3].T,lw=2,label=f'executed arm{arm}');ax.scatter(*goals[arm,:3,3],marker='*',s=120)
    ax.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='world z (m)',title='Converted / physical guidance / executed world paths');ax.legend(fontsize=8)
    savefig(fig,out,'05_world_paths',files)
    replay=unit/'execution/replay_trace.npz'
    if replay.exists():
        b=np.load(replay);delta=np.max(abs(a['integration']-b['integration']),axis=1)
        fig,ax=plt.subplots(figsize=(12,4));ax.plot(t,delta);ax.axhline(1e-10,color='red',ls='--');ax.set(title='Independent replay: max error across full integration state',xlabel='time (s)',ylabel='absolute error');ax.grid(alpha=.2)
        savefig(fig,out,'06_independent_replay',files)
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for arm in range(2):
        for j in range(3):
            ax=axs[arm,j];ax.plot(t,a['ee'][:,arm,j,3]);ax.axhline(goals[arm,j,3],color='red',ls='--');ax.set(title=f'Arm {arm} world {"xyz"[j]}',xlabel='time (s)',ylabel='m');ax.grid(alpha=.2)
    savefig(fig,out,'07_world_coordinates',files)
    with (out/'timeseries.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','arm0_position_mm','arm1_position_mm','arm0_rotation_deg','arm1_rotation_deg','clearance_mm','hold_gate_active'])
        for i in range(len(t)):w.writerow([t[i],*(pos[i]*1000),*np.rad2deg(rot[i]),clearance[i]*1000 if len(clearance)==len(t) else '',bool(hold[i])])
    data=mujoco.MjData(robot.model);body=[]
    for i in indices:
        data.qpos[:]=a['qpos'][i];data.qvel[:]=a['qvel'][i];mujoco.mj_forward(robot.model,data);body.append(data.xpos.copy())
    payload=dict(time=t[indices].tolist(),bodies=np.asarray(body).tolist(),parents=robot.model.body_parentid.tolist(),
        goals=goals[:,:3,3].tolist(),ee=a['ee'][indices,:,:3,3].tolist(),obstacles=entry['scene']['objects'],
        position_mm=(pos[indices]*1000).tolist(),clearance_mm=(clearance[indices]*1000).tolist() if len(clearance)==len(t) else [],
        execution_level=report['level'])
    videos=[];errors=[]
    for view in ('overview','top'):
        try:videos.append(render_video(out,robot,a,goals,indices,view,deadline,notify))
        except Exception as exc:errors.append(dict(view=view,error=repr(exc)))
    html=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    gallery=''.join(f'<section><h2>{name}</h2><a href="{name}.pdf">PDF</a> · <a href="{name}.png">PNG</a><img src="{name}.png"></section>' for name in [f'{i:02d}_{s}' for i,s in enumerate(['summary','joint_motion','torque_tracking','base_manifold','world_paths','independent_replay','world_coordinates'],1)] if (out/f'{name}.png').exists())
    html=html.replace('__GALLERY__',gallery).replace('__STATUS__',report['level']).replace('__VIDEOS__',''.join(f'<h2>{v["file"]}</h2><video src="{v["file"]}" controls loop muted></video>' for v in videos))
    (out/'index.html').write_text(html,encoding='utf-8')
    manifest=dict(status='COMPLETE' if len(videos)==2 else 'PARTIAL_RENDER_ERRORS',videos=videos,errors=errors,
        source_native_hash=file_digest(native),new_native_steps=0,new_training_updates=0,
        labels='original five-obstacle static system; no dynamic handoff in this run',
        files={p.name:dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in out.iterdir() if p.is_file()})
    atomic_json(out/'manifest.json',manifest);return manifest


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Diffusion 完整运行可视化</title>
<style>body{margin:0;background:#101722;color:#dce5ee;font:16px system-ui}main{max-width:1200px;margin:auto;padding:28px}section{background:#182333;border:1px solid #304255;border-radius:12px;padding:18px;margin:18px 0}h1{font-size:30px}h2{font-size:19px}a{color:#64cbe7}img,video{width:100%;border-radius:8px}canvas{width:100%;height:560px;background:#101923;touch-action:none}input{width:65%}button{padding:9px 18px}small{color:#a8b8c7}.badge{color:#ffcd78}</style>
<main><h1>历史 Diffusion 模型前缀 → 碰撞反馈修正 → 67通道执行 → 独立重放</h1><p class="badge">本轮执行等级：__STATUS__</p><p>原固定五障碍、原世界目标与27秒运动。动画来自实际保存状态；动态连续交接尚属后续阶段。系统通过不等于裸网络独立通过，不进行相对优势评判。</p>
<section><h2>交互三维状态回看</h2><small>拖动旋转，滚轮缩放；骨架示意来自模型body位置。完整实体和障碍见下方MuJoCo视频。没有重新运行物理。</small><canvas id="c"></canvas><button id="play">播放/暂停</button> <input id="seek" type="range" min="0" value="0"><p id="info"></p></section>
__VIDEOS__<p><a href="timeseries.csv">下载逐时刻指标 CSV</a> · <a href="manifest.json">可视化来源与文件hash</a></p>__GALLERY__</main>
<script>const D=__DATA__;const c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('seek');s.max=D.time.length-1;let i=0,playing=false,yaw=.7,pitch=.45,zoom=1,drag=null,last=0;const all=D.bodies.flat(),lo=[0,1,2].map(k=>Math.min(...all.map(x=>x[k]))),hi=[0,1,2].map(k=>Math.max(...all.map(x=>x[k]))),center=lo.map((x,k)=>(x+hi[k])/2),span=Math.max(...hi.map((x,k)=>x-lo[k]),1);
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);return[c.width/2+a*c.width*.55/span*zoom,c.height/2-(z*Math.cos(pitch)-b*Math.sin(pitch))*c.width*.55/span*zoom]}
function dot(p,color,r=5){let a=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],r,0,7);ctx.fill()}
function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=560*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);let b=D.bodies[i];ctx.lineWidth=2*devicePixelRatio;for(let j=1;j<b.length;j++){let p=D.parents[j];if(p<=0)continue;let a=project(b[p]),v=project(b[j]);ctx.strokeStyle='#77c4d9';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...v);ctx.stroke()}for(let o of D.obstacles){let a=project(o.position);ctx.fillStyle='#ed8a5155';ctx.strokeStyle='#ed8a51';ctx.beginPath();ctx.arc(...a,Math.max(2,o.size[0]*c.width*.55/span*zoom),0,7);ctx.fill();ctx.stroke()}for(let k=0;k<2;k++){dot(D.goals[k],k?'#ff6f75':'#4bdfe8',7*devicePixelRatio);dot(D.ee[i][k],'#f7f7d4',4*devicePixelRatio)}document.getElementById('info').textContent=`t=${D.time[i].toFixed(3)} s · 双臂位置误差 ${D.position_mm[i].map(x=>x.toFixed(3)).join(' / ')} mm · 间隙 ${D.clearance_mm.length?D.clearance_mm[i].toFixed(3):'NA'} mm`;s.value=i}
s.oninput=()=>{i=+s.value;draw()};document.getElementById('play').onclick=()=>playing=!playing;c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;function tick(t){if(playing&&t-last>66){i=(i+1)%D.time.length;last=t;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);</script></html>'''
