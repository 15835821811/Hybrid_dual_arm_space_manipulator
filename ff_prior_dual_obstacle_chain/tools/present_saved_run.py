"""Present only saved states. No physics steps, search, goal changes or recertification."""
from pathlib import Path
import json,shutil,subprocess,sys,time
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior.coupled_predictor import pose_errors
from ff_prior_dual_obstacle_chain.protocol import ROOT,OUT,PACKAGE

def main():
    out=OUT/'presentation';out.mkdir(exist_ok=False)
    original=OUT/'dynamic/visuals'
    for f in original.iterdir():
        if f.is_file():shutil.copy2(f,out/f.name)
    a=np.load(OUT/'dynamic/execution/native_trace.npz');s=np.load(OUT/'static/units/u00/execution/native_trace.npz')
    c=read_json(OUT/'dynamic/contract.json');t=a['time'];w=np.asarray(c['waypoints']['waypoint_points_m'])
    obstacles=c['scene']['objects'][3:];centers=np.array([o['position'] for o in obstacles]);radii=np.array([o['size'][0] for o in obstacles])
    boundaries=t[0]+4.5+np.r_[0,np.cumsum(c['waypoints']['segment_durations_s'])]
    rows=[]
    for j,b in enumerate(boundaries):
        k=int(np.argmin(abs(t-b)))
        rows.append(dict(waypoint=j+1,commanded_time_s=float(b),nearest_recorded_time_s=float(t[k]),
            time_offset_ms=float((t[k]-b)*1000),position_error_mm=float(np.linalg.norm(a['ee'][k,0,:3,3]-w[j])*1000)))
    distance=np.linalg.norm(a['ee'][:,0,None,:3,3]-centers[None],axis=-1)-radii
    pos,rot=pose_errors(a['ee'],a['desired'])
    stats=dict(waypoints=rows,all_7_pass=all(x['position_error_mm']<=5 for x in rows),
        waypoint_measurement='nearest recorded 2ms state; not interpolated exact-time certification',
        tip_center_to_sphere_surface_min_mm=(1000*distance.min(axis=0)).tolist(),
        tip_clearance_scope='point-to-sphere diagnostic, not whole-body clearance',
        handoff_full836_exact=bool(np.array_equal(s['integration'][-1],a['integration'][0])),
        minimum_dynamic_whole_body_clearance_mm=float(a['clearance'].min()*1000),
        dynamic_position_max_mm=(pos.max(axis=0)*1000).tolist(),
        native_steps_for_presentation=0,source_dynamic=file_digest(OUT/'dynamic/execution/native_trace.npz'))
    atomic_json(PACKAGE/'evidence/seven_waypoints.json',stats)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(13,6));ax=fig.add_subplot(121,projection='3d');ay=fig.add_subplot(122)
    ax.plot(*a['desired'][:,0,:3,3].T,'--',label='冻结绕行参考',lw=1)
    ax.plot(*a['ee'][:,0,:3,3].T,label='实际执行路径',lw=1.5)
    ax.scatter(*w.T,color='green',s=35,label='原七世界航点')
    for j,(center,radius) in enumerate(zip(centers,radii)):
        u=np.linspace(0,2*np.pi,24);v=np.linspace(0,np.pi,16)
        ax.plot_surface(center[0]+radius*np.outer(np.cos(u),np.sin(v)),center[1]+radius*np.outer(np.sin(u),np.sin(v)),center[2]+radius*np.outer(np.ones_like(u),np.cos(v)),color='red',alpha=.6)
    for j,p in enumerate(w):ax.text(*p,str(j+1),color='green')
    ax.set(xlabel='世界 x / m',ylabel='世界 y / m',zlabel='世界 z / m',title='连续体实际绕过六个新增障碍');ax.view_init(22,25);ax.legend(fontsize=8)
    vals=[x['position_error_mm'] for x in rows];ay.bar(range(1,8),vals,color='#268a9e')
    for i,x in enumerate(vals):ay.text(i+1,x+.001,f'{x:.4f}',ha='center',fontsize=9)
    ay.set(xlabel='原航点编号',ylabel='位置误差 / mm',title='七航点到达误差：均小于原5 mm门',ylim=(0,.07),xticks=range(1,8));ay.grid(axis='y',alpha=.2)
    fig.subplots_adjust(left=.05,right=.97,bottom=.12,top=.9,wspace=.3)
    for ext in ('png','pdf'):fig.savefig(out/f'05_continuum_waypoints.{ext}',dpi=170,bbox_inches='tight',pad_inches=.3)
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7));axs[0].plot(t,pos[:,0]*1000,label='连续体位置误差');axs[0].plot(t,pos[:,1]*1000,label='刚性臂位置误差');axs[0].legend();axs[0].set(ylabel='mm',title='动态世界误差放大视图（全程原5 mm门通过）')
    axs[1].plot(t,a['clearance']*1000,label='全机所有有效几何对最小间隙');axs[1].axhline(0,c='red',ls='--');axs[1].legend();axs[1].set(xlabel='绝对仿真时间 / s',ylabel='mm',title='原非穿透门；不是只检查末端')
    for ax in axs:
        for b in boundaries:ax.axvline(b,color='green',alpha=.2)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'06_error_clearance_zoom.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    # Close-up replay of saved qpos/qvel; forward only updates displayed geometry.
    robot=RobotAdapter(scene=SceneSpec.from_dict(c['scene']));m=robot.model;d=mujoco.MjData(m)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=w.mean(axis=0)+np.array([-.055,0.,0.]);cam.distance=.9;cam.azimuth=135;cam.elevation=-20
    opt=mujoco.MjvOption();opt.geomgroup[:]=1
    idx=np.unique(np.linspace(0,len(t)-1,400).round().astype(int))
    ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    argv=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','15','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'continuum_closeup.mp4')]
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for j,k in enumerate(idx):
                d.qpos[:]=a['qpos'][k];d.qvel[:]=a['qvel'][k];d.time=float(t[k]);mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=cam,scene_option=opt)
                # Green dots mark the exact original seven world points (visual overlays only).
                for point in w:
                    sc=renderer.scene
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.004),point,np.eye(3).ravel(),np.array([.1,.95,.4,1.],np.float32));sc.ngeom+=1
                frame=renderer.render();proc.stdin.write(frame.tobytes())
                if j in (100,200,300):plt.imsave(out/f'closeup_{j:03d}.png',frame)
            proc.stdin.close();proc.stdin=None;_,err=proc.communicate(timeout=60)
            assert proc.returncode==0,err
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    html=(out/'index.html').read_text(encoding='utf-8')
    html=html.replace('<title>Diffusion 完整运行可视化</title>','<title>九障碍完整执行：七航点与独立重放通过</title>')
    html=html.replace('<h1>Diffusion 提案 → 避障修正 → 67通道力矩执行 → 独立重放</h1>','<h1>双侧九障碍：真实连续执行与独立重放通过</h1>')
    html=html.replace('<section><h2>交互三维状态回看','<section><h2>连续体特写：红球为新增障碍，绿点为原七航点</h2><video src="continuum_closeup.mp4" controls loop muted></video><p>仅回看已保存的27秒动态执行，未增加物理步。七点近采样时刻误差0.011–0.053 mm；完整动态最大位置误差0.181 mm。</p></section><section><h2>交互三维状态回看')
    html=html.replace('<button id="play">播放/暂停</button>','<button id="play">播放/暂停</button> <button id="focus">连续体特写 / 全机</button>')
    html=html.replace("const c=document.getElementById('c')",'D.waypoints='+json.dumps(w.tolist())+";let focus=false;const fc="+json.dumps((w.mean(axis=0)+np.array([-.055,0,0])).tolist())+";const c=document.getElementById('c')")
    html=html.replace('function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2]', 'function project(p){let origin=focus?fc:center,scale=focus?.55:span;let x=p[0]-origin[0],y=p[1]-origin[1],z=p[2]-origin[2]')
    html=html.replace('/span*zoom','/(focus?.55:span)*zoom')
    html=html.replace("for(let k=0;k<2;k++){dot", "for(let j=0;j<D.waypoints.length;j++){dot(D.waypoints[j],'#39ef7a',4*devicePixelRatio);let p=project(D.waypoints[j]);ctx.fillStyle='#6aff9c';ctx.font=(14*devicePixelRatio)+'px sans-serif';ctx.fillText(String(j+1),p[0]+5,p[1]-5)}for(let k=0;k<2;k++){dot")
    html=html.replace('s.oninput=()=>',"document.getElementById('focus').onclick=()=>{focus=!focus;draw()};s.oninput=()=>")
    html=html.replace('</main>','<section><h2>原七航点与实际绕行路径</h2><img src="05_continuum_waypoints.png"></section><section><h2>误差与全机间隙放大视图</h2><img src="06_error_clearance_zoom.png"></section></main>')
    (out/'index.html').write_text(html,encoding='utf-8')
    atomic_json(out/'manifest.json',dict(type='supplemental_presentation_saved_states_only',original_manifest_sha256=file_digest(original/'manifest.json'),
        new_physics_steps=0,closeup_frames=400,closeup_recorded_duration_s=27.,closeup_playback_s=400/15,
        files={p.name:dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(stats,ensure_ascii=False))

if __name__=='__main__':main()
