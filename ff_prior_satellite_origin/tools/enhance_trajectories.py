"""New saved-state presentation, never physics or requalification of sealed runs."""
from pathlib import Path
import csv, hashlib, json, re, subprocess, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PKG=ROOT/'ff_prior_satellite_origin'
PREV=PKG/'delivery_001'
DEST=PKG/'delivery_002'
COLORS=[(45,225,244),(255,178,64)]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def trajectory_figure(ee,goals,waypoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,8))
    for arm in (0,1):
        ax=fig.add_subplot(1,2,arm+1,projection='3d');color=np.array(COLORS[arm])/255
        ax.plot(*ee[:,arm].T,color=color,lw=2,label='actual saved EE')
        ax.plot(*goals[:,arm].T,color='grey',lw=1,ls='--',label='saved target')
        ax.scatter(*ee[0,arm],c='green',s=35,label='start (0 s)')
        ax.scatter(*ee[-1,arm],c='black',marker='x',s=40,label='end (83.5 s)')
        if arm==0:
            wp=np.asarray(waypoints);ax.scatter(*wp.T,c='#6e83d7',s=15)
            for j,p in enumerate(wp):ax.text(*(p+[0,0,.006]),'W'+str(j+1),fontsize=9)
        ax.set(xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title=['Continuum EE0','Rigid EE1'][arm])
        ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Actual EE trajectories 0–83.5 s | new 54.5–83.5 s segment FAILED momentum gate',y=.97)
    fig.text(.5,.04,'0–27.5 s: approach   |   27.5–54.5 s: seven waypoints   |   54.5–56.5 s: disclosed satellite thrust   |   56.5–83.5 s: coast tracking',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.98,bottom=.10,top=.91,wspace=.08)
    for ext in ('png','pdf'):fig.savefig(DEST/f'21_ee_trajectories.{ext}',dpi=150)
    plt.close(fig)

def main():
    started=time.time();deadline=started+900
    DEST.mkdir(exist_ok=False)
    def budget():
        if time.time()>deadline:raise TimeoutError('READ_ONLY_MEDIA_900S')
        if sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())>64*1024**2:raise RuntimeError('READ_ONLY_MEDIA_64MIB')
    page=(PREV/'index.html').read_text(encoding='utf-8')
    D=json.loads(page.split('<script>const D=',1)[1].split(';const c=',1)[0])
    oldmanifest=read(PREV/'manifest.json')
    # Read only saved poses for the user's requested rendering/trajectory export.
    sources=[ROOT/'artifacts/v7_1/balanced_restore_001/units/u00/execution/native_trace.npz',ROOT/'artifacts/v7_1/balanced_brake_001/execution/native_trace.npz',ROOT/'artifacts/v7_1/moving_satellite_001/execution/native_trace.npz']
    entry=read(ROOT/'ff_prior_balanced_restore/evidence/selection.json')['entries'][0]
    chunks=[];before={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    for k,p in enumerate(sources):
        with np.load(p) as z:
            chunk={n:z[n] for n in ('time','ee','qpos')}
            chunk['desired']=z['desired'] if k else np.repeat(np.asarray(entry['task']['condition']['goals'])[None],len(z['time']),axis=0)
        chunks.append(chunk)
    arrays={n:np.concatenate([a[n][1:] if k else a[n] for k,a in enumerate(chunks)]) for n in chunks[0]}
    times=np.asarray(D['time']);ids=np.searchsorted(arrays['time'],times)
    assert np.max(abs(arrays['time'][ids]-times))<1e-10
    assert np.array_equal(arrays['ee'][ids],np.asarray(D['ee_poses']))
    ee=arrays['ee'][:,:,:3,3];goals=arrays['desired'][:,:,:3,3]
    with (DEST/'full_actual_and_target_ee_xyz.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['time_s']+[f'{kind}_{arm}_{axis}_m' for kind in ('actual','target') for arm in ('continuum','rigid') for axis in 'xyz'])
        w.writerows(np.c_[arrays['time'],ee.reshape(-1,6),goals.reshape(-1,6)])
    trajectory_figure(ee,goals,D['waypoints']);budget()
    # Kinematics-only scene display; make every dynamic integration entry point fail.
    import mujoco
    def forbidden(*a,**kw):raise RuntimeError('DYNAMIC_INTEGRATION_FORBIDDEN_IN_MEDIA')
    for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_forward'):
        if hasattr(mujoco,name):setattr(mujoco,name,forbidden)
    from ff_prior.robot_adapter import RobotAdapter
    from ff_prior.scenes import SceneSpec
    from ff_prior_balanced_evidence.media import connector,sphere,frame_segments,AXIS_COLORS
    from PIL import Image,ImageDraw,ImageFont
    robot=RobotAdapter(scene=SceneSpec.from_dict(entry['scene']))
    model=robot.model;data=mujoco.MjData(model)
    model.vis.global_.offwidth=1100;model.vis.global_.offheight=780
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    pts=ee[ids];target=goals[ids]
    bounds=[]
    for arm in (0,1):
        p=np.r_[pts[:,arm],target[:,arm]];lo=p.min(0);hi=p.max(0);gap=np.maximum(hi-lo,.03)*.10
        bounds.append((lo-gap,hi+gap))
    trail=np.unique(np.linspace(0,len(times)-1,min(700,len(times))).round().astype(int))
    xyz=np.r_[pts.reshape(-1,3),np.asarray(D['bodies'])[:,1],np.asarray(D['waypoints'])]
    videos=[]
    def line_panel(draw,j,arm,plane,box):
        x0,y0,x1,y1=box;axes=(0,1) if plane=='XY' else (0,2);lo,hi=bounds[arm]
        draw.rounded_rectangle(box,8,fill='#142234',outline='#46556a')
        draw.text((x0+10,y0+8),('连续体 EE0' if arm==0 else '刚性臂 EE1')+'  '+plane+'投影',font=small,fill=COLORS[arm])
        def proj(p):return (x0+28+(p[axes[0]]-lo[axes[0]])/(hi[axes[0]]-lo[axes[0]])*(x1-x0-50),y1-25-(p[axes[1]]-lo[axes[1]])/(hi[axes[1]]-lo[axes[1]])*(y1-y0-70))
        future=[proj(p) for p in pts[:,arm]]
        draw.line(future,fill='#324253',width=1)
        actual=[proj(p) for p in pts[:j+1,arm]]
        if len(actual)>1:draw.line(actual,fill=COLORS[arm],width=3)
        for k in range(1,j+1,5):draw.line([proj(target[k-1,arm]),proj(target[k,arm])],fill='#b3b9c1',width=1)
        x,y=actual[-1];draw.ellipse((x-4,y-4,x+4,y+4),fill=COLORS[arm])
        a,b=proj(pts[0,arm]);draw.ellipse((a-3,b-3,a+3,b+3),outline='#73ff8d',width=2)
        draw.text((x0+10,y1-22),f'{"XYZ"[axes[0]]} [{lo[axes[0]]:.2f}, {hi[axes[0]]:.2f}] m · {"XYZ"[axes[1]]} [{lo[axes[1]]:.2f}, {hi[axes[1]]:.2f}] m',font=small,fill='#b5c2d0')
    with mujoco.Renderer(model,height=780,width=1100,max_geom=4000) as renderer:
        for view in ('overview','top'):
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=(xyz.min(0)+xyz.max(0))/2;camera.distance=max(2.,float(np.ptp(xyz,axis=0).max())*1.8)
            camera.azimuth=125 if view=='overview' else 90;camera.elevation=-25 if view=='overview' else -85
            option=mujoco.MjvOption();option.geomgroup[:]=1
            cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',f'{len(times)*2}/167','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(DEST/(view+'.mp4'))]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                for j,k in enumerate(ids):
                    if j%100==0:budget();print(json.dumps({'view':view,'frame':j,'total':len(ids)}),flush=True)
                    data.qpos[:]=arrays['qpos'][k];data.time=float(times[j]);mujoco.mj_kinematics(model,data);mujoco.mj_camlight(model,data)
                    renderer.update_scene(data,camera=camera,scene_option=option);scene=renderer.scene
                    for n in D['body_frames'][j]['nearest']:
                        near=n['distance_m']<.010;active=n['distance_m']<.050
                        color=[1.,.08,.2,1.] if near else [.95,.75,.20,1.]
                        if active:
                            for gi in range(scene.ngeom):
                                g=scene.geoms[gi]
                                if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==n['geom_id']:g.rgba[:]=color
                            a,b=n['connector'];connector(scene,a,b,color,.002,label=f"{n['obstacle']} {n['distance_m']*1000:.1f}mm")
                            bid=n['body_id'];connector(scene,data.xpos[model.body_parentid[bid]],data.xpos[bid],color,.005)
                    for arm in (0,1):
                        ii=np.unique(np.r_[trail[trail<=j],j]);color=[x/255 for x in COLORS[arm]]+[1.]
                        for a,b in zip(ii[:-1],ii[1:]):connector(scene,pts[a,arm],pts[b,arm],color,.004)
                        sphere(scene,pts[0,arm],.006,[.3,1.,.4,1.])
                        for is_goal,pose,name in ((False,arrays['ee'][k,arm],f'EE{arm}'),(True,arrays['desired'][k,arm],f'Goal{arm}')):
                            for ax,(a,b) in enumerate(frame_segments(pose,.09 if is_goal else .065)):connector(scene,a,b,AXIS_COLORS[ax],.003 if is_goal else .002,True,name if ax==2 else '')
                    for n,p in enumerate(D['waypoints']):sphere(scene,p,.005,[.4,.6,1.,1.],f'W{n+1}')
                    canvas=Image.new('RGB',(1600,900),'#0c1420');canvas.paste(Image.fromarray(renderer.render()),(0,60));draw=ImageDraw.Draw(canvas)
                    draw.text((18,12),f'真实记录 t={times[j]:.3f}s  | 末端累计轨迹：青色 连续体 / 橙色 刚性臂',font=font,fill='#e0edf8')
                    for arm in (0,1):
                        for n,plane in enumerate(('XY','XZ')):line_panel(draw,j,arm,plane,(1110,68+(arm*2+n)*190,1590,250+(arm*2+n)*190))
                    draw.text((18,844),'红色：障碍间隙 <10mm 的接近提示；黄色：10–50mm。物理验收仍为不穿透。',font=small,fill='#ffcd78')
                    draw.text((18,870),'APP_DEV：54.5–83.5s 原机器人动量门失败 | RGB=XYZ；侧窗实线=已走轨迹，灰色=完整实际轨迹/目标参照',font=small,fill='#c9d9e9')
                    proc.stdin.write(canvas.tobytes())
                    if j in (0,36,len(ids)-1):canvas.save(DEST/f'{view}_{j:04d}.png')
                proc.stdin.close();err=proc.stderr.read();rc=proc.wait(timeout=60)
                if rc:raise RuntimeError(err.decode(errors='replace'))
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            videos.append({'file':view+'.mp4','frames':len(times),'duration_s':83.5,'displayed_actual_times':True})
    # Keep the old delivery immutable. Borrow its unchanged curves by relative URL.
    page=re.sub(r'(src|href)="([^"#:]+)"',lambda m:m.group(1)+'="'+(m.group(2) if m.group(2) in ('overview.mp4','top.mp4') else '../delivery_001/'+m.group(2))+'"',page)
    explanation='<section><h2>颜色与末端轨迹</h2><p>红色只表示某臂段到障碍的距离小于10mm，不代表穿透，也不等于避障控制已经改变命令。当前显示序列首次红色在约2.40s：R1/后臂段9.71mm、C5/连续体第22段9.81mm。视频黄带10–50mm；10mm仅为显示阈值。</p><p>实际末端轨迹：连续体青色、刚性臂橙色；视频加粗空间轨迹并增加独立XY/XZ累计轨迹窗。起点绿圈，亮色走过轨迹，灰色全程实际轨迹仅为回放参照。EE/Goal轴继续以RGB标XYZ。</p><a href="full_actual_and_target_ee_xyz.csv">全部41751实际状态与目标XYZ（CSV）</a> · <a href="21_ee_trajectories.pdf">末端三维轨迹PDF</a><img src="21_ee_trajectories.png"></section>'
    page=page.replace('<section><h2>交互三维',explanation+'<section><h2>交互三维',1)
    page=page.replace("arm?'#ffbd53':'#40dfef',3)","arm?'#ffbd53':'#40dfef',5)")
    page=page.replace('<p id="info"></p>','<p id="info"></p><div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><canvas id="trail0" style="height:280px"></canvas><canvas id="trail1" style="height:280px"></canvas></div>')
    extra="""function drawTrails(){for(let arm=0;arm<2;arm++){let cv=document.getElementById('trail'+arm);cv.width=cv.clientWidth*devicePixelRatio;cv.height=280*devicePixelRatio;let q=cv.getContext('2d'),ps=D.ee.map(a=>a[arm]),xs=ps.map(p=>p[0]),zs=ps.map(p=>p[2]),x0=Math.min(...xs),z0=Math.min(...zs),dx=Math.max(.02,Math.max(...xs)-x0),dz=Math.max(.02,Math.max(...zs)-z0);q.clearRect(0,0,cv.width,cv.height);let xy=p=>[25*devicePixelRatio+(p[0]-x0)/dx*(cv.width-50*devicePixelRatio),cv.height-25*devicePixelRatio-(p[2]-z0)/dz*(cv.height-75*devicePixelRatio)];q.font=(13*devicePixelRatio)+'px system-ui';q.fillStyle=arm?'#ffbd53':'#40dfef';q.fillText((arm?'Rigid EE1':'Continuum EE0')+' | world XZ | accumulated actual path',12*devicePixelRatio,22*devicePixelRatio);for(let full of [true,false]){q.strokeStyle=full?'#46576a':arm?'#ffbd53':'#40dfef';q.lineWidth=(full?1:4)*devicePixelRatio;q.beginPath();for(let j=0;j<(full?ps.length:i+1);j++){let a=xy(ps[j]);if(j)q.lineTo(...a);else q.moveTo(...a)}q.stroke()}let a=xy(ps[i]);q.fillStyle=arm?'#ffbd53':'#40dfef';q.beginPath();q.arc(...a,5*devicePixelRatio,0,7);q.fill();}}
"""
    page=page.replace('function project(',extra+'function project(',1).replace('s.value=i}','s.value=i;drawTrails()}',1)
    (DEST/'index.html').write_text(page,encoding='utf-8')
    after={p.relative_to(ROOT).as_posix():digest(p) for p in sources};assert before==after
    manifest=dict(status='COMPLETE_READONLY_PRESENTATION',scientific_result='EXECUTION_FAILED_MOMENTUM_UNCHANGED',previous_delivery='delivery_001',new_native_steps=0,new_coupled_calls=0,new_geometry_queries=0,source_hashes=before,saved_pose_frames=len(arrays['time']),display_frames=len(times),videos=videos,source_files_unchanged=True,new_full_audit=False,new_resume=False,wall_s=time.time()-started,cap_bytes=64*1024**2,cap_wall_s=900,display_only_red_distance_m=.01,physical_penetration_gate_m=0,first_shown_red={'time_s':times[36],'R1_mm':9.708212601031496,'C5_mm':9.813799210875894},files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in DEST.iterdir() if p.is_file()})
    manifest['output_bytes']=sum(x['bytes'] for x in manifest['files'].values());save(DEST/'manifest.json',manifest);budget()
    print(json.dumps({k:manifest[k] for k in ('status','wall_s','output_bytes','source_files_unchanged')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    if '--figure-only' in sys.argv:
        a=np.loadtxt(DEST/'full_actual_and_target_ee_xyz.csv',delimiter=',',skiprows=1)
        page=(PREV/'index.html').read_text(encoding='utf-8')
        D=json.loads(page.split('<script>const D=',1)[1].split(';const c=',1)[0])
        trajectory_figure(a[:,1:7].reshape(-1,2,3),a[:,7:13].reshape(-1,2,3),D['waypoints'])
        print('PRESENTATION_FIGURE_ONLY_FROM_EXPORTED_CSV')
    else:main()
