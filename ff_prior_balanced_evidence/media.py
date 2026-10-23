"""Two actual-state videos and interactive whole-arm evidence, without mj_step."""
from pathlib import Path
import html
import json
import shutil
import subprocess
import time
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.coupled_predictor import pose_errors
from ff_prior.storage import file_digest
from .plots import BodySeries, generate_body_figures


AXIS_COLORS = [[1., .15, .15, 1.], [.15, .9, .2, 1.], [.2, .4, 1., 1.]]
PATH_COLORS = [[.15, .9, 1., .85], [1., .65, .15, .85]]


def frame_segments(pose, length=.075):
    pose = np.asarray(pose)
    return [(pose[:3, 3].copy(), pose[:3, 3] + length*pose[:3, j]) for j in range(3)]


def connector(scene, a, b, color, width=.002, arrow=False, label=''):
    if np.linalg.norm(np.asarray(a)-b) < 1e-12:
        return
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError('VISUAL_GEOMETRY_CAPACITY')
    geom = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, kind, width, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if label:
        geom.label = label
    scene.ngeom += 1


def sphere(scene, position, radius, color, label=''):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError('VISUAL_GEOMETRY_CAPACITY')
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
        np.asarray(position), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    geom.label = label
    scene.ngeom += 1


def add_saved_overlays(scene, ee, goals, index, trajectory_indices, waypoints, body_frame=None,
                       body_positions=None, parents=None):
    for arm in range(2):
        for target, pose, label in ((False, ee[index, arm], f'EE{arm}'), (True, goals[arm], f'Goal{arm}')):
            for axis, (a, b) in enumerate(frame_segments(pose, .09 if target else .065)):
                connector(scene, a, b, AXIS_COLORS[axis], .003 if target else .002, True, label if axis == 2 else '')
        ids = np.asarray(trajectory_indices)
        ids = ids[ids <= index]
        if not len(ids) or ids[-1] != index:
            ids = np.r_[ids, index]
        for a, b in zip(ids[:-1], ids[1:]):
            connector(scene, ee[int(a), arm, :3, 3], ee[int(b), arm, :3, 3], PATH_COLORS[arm], .0015)
    for i, point in enumerate(waypoints):
        sphere(scene, point, .006, [.35, .65, 1., .9], f'W{i+1}')
    if body_frame is not None:
        for item in body_frame['nearest']:
            near = item['distance_m'] < .010
            color = [1., .08, .2, 1.] if near else [.9, .8, .25, .8]
            a, b = item['connector']
            label = f"{item['obstacle']}/{item['body_name']} {item['distance_m']*1000:.1f}mm"
            for gi in range(scene.ngeom):
                drawn = scene.geoms[gi]
                if drawn.objtype == mujoco.mjtObj.mjOBJ_GEOM and drawn.objid == item['geom_id']:
                    drawn.rgba[:] = color
            connector(scene, a, b, color, .0025 if near else .001, label=label)
            sphere(scene, b, .005 if near else .003, color)
            if body_positions is not None and parents is not None:
                bid = item['body_id']
                connector(scene, body_positions[int(parents[bid])], body_positions[bid], color,
                    .006 if near else .003)
                # Fixed/axis body origins often coincide. The outgoing body
                # segment still shows the named link's physical extent.
                for child in np.flatnonzero(np.asarray(parents) == bid):
                    connector(scene, body_positions[bid], body_positions[child], color,
                        .006 if near else .003)


def render_video(out, robot, arrays, goals, indices, view, deadline, notify, waypoints, series):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFMPEG_NOT_FOUND')
    width, height, fps = 960, 720, 15
    model = robot.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    xyz = np.concatenate([arrays['ee'][:, :, :3, 3].reshape(-1, 3), arrays['base'][:, :3], waypoints])
    camera.lookat[:] = (xyz.min(axis=0)+xyz.max(axis=0))/2
    camera.distance = max(2., float(np.ptp(xyz, axis=0).max())*1.8)
    camera.azimuth = 125 if view == 'overview' else 90
    camera.elevation = -25 if view == 'overview' else -85
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    target = Path(out)/f'{view}.mp4'
    argv = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    proc = None
    paths = np.unique(np.linspace(0, len(arrays['time'])-1, min(400, len(arrays['time']))).round().astype(int))
    with mujoco.Renderer(model, height=height, width=width, max_geom=2500) as renderer:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for j, k in enumerate(indices):
                if time.time() >= deadline:
                    raise TimeoutError('VISUALIZATION_WALL_BUDGET')
                data.qpos[:] = arrays['qpos'][k]
                data.qvel[:] = arrays['qvel'][k]
                data.time = float(arrays['time'][k])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                add_saved_overlays(renderer.scene, arrays['ee'], goals[k], int(k), paths, waypoints,
                    series.frame(data.time), data.xpos, model.body_parentid)
                frame = renderer.render()
                proc.stdin.write(frame.tobytes())
                if j in (0, len(indices)//2, len(indices)-1):
                    plt.imsave(Path(out)/f'{view}_{j:03d}.png', frame)
                if notify and j % 50 == 0:
                    notify('render_saved_body_states', view=view, frame=j+1, total=len(indices))
            proc.stdin.close()
            proc.stdin = None
            _, error = proc.communicate(timeout=60)
            if proc.returncode:
                raise RuntimeError(error.decode('utf-8', errors='replace'))
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
    return dict(file=target.name, frames=len(indices), fps=fps,
        recorded_duration_s=float(arrays['time'][-1]-arrays['time'][0]), playback_duration_s=len(indices)/fps,
        new_native_steps=0)


def generate_episode_media(out, robot, arrays, goals, scene, waypoints, diagnostic_dirs,
                           deadline, notify, status, title, max_frames=420):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not len(arrays['time']):
        raise ValueError('NO_SAVED_ACTUAL_STATES_FOR_MEDIA')
    times = np.asarray(arrays['time'])
    goals = np.asarray(goals)
    if goals.shape == (2, 4, 4):
        goals = np.broadcast_to(goals, (len(times), 2, 4, 4))
    if goals.shape != (len(times), 2, 4, 4):
        raise ValueError('SAVED_TARGET_POSE_TIME_MISMATCH')
    if isinstance(waypoints, dict):
        waypoints = waypoints['waypoint_points_m']
    points = np.asarray(waypoints, dtype=float)
    if points.shape != (7, 3):
        raise ValueError('ORIGINAL_SEVEN_WAYPOINTS_REQUIRED')
    series = BodySeries(diagnostic_dirs)
    if time.time() >= deadline:
        raise TimeoutError('VISUALIZATION_WALL_BUDGET')
    body_figures = generate_body_figures(out, series)
    indices = np.unique(np.linspace(0, len(times)-1, min(max_frames, len(times))).round().astype(int))
    # Align images with recorded diagnostic ticks, not interpolated distances.
    if series.a:
        aligned = []
        for k in indices:
            j = series.index(times[k], .0101)
            if j is not None:
                candidate = int(np.argmin(abs(times-series.a['time'][j])))
                if abs(times[candidate]-series.a['time'][j]) < 1e-7:
                    k = candidate
            aligned.append(k)
        indices = np.unique(np.r_[0, aligned, len(times)-1])
    positions = []
    frames = []
    data = mujoco.MjData(robot.model)
    for k in indices:
        if time.time() >= deadline:
            raise TimeoutError('VISUALIZATION_WALL_BUDGET')
        data.qpos[:] = arrays['qpos'][k]
        data.time = float(times[k])
        mujoco.mj_kinematics(robot.model, data)
        positions.append(data.xpos.copy())
        frames.append(series.frame(times[k]))
    error, _ = pose_errors(np.asarray(arrays['ee'])[indices], goals[indices])
    clearance = np.asarray(arrays.get('clearance_m', arrays.get('clearance', np.full(len(times), np.nan))))
    if clearance.shape != (len(times),):
        clearance = np.full(len(times), np.nan)
    payload = dict(time=times[indices].tolist(), bodies=np.asarray(positions).tolist(),
        parents=robot.model.body_parentid.tolist(), ee=np.asarray(arrays['ee'])[indices, :, :3, 3].tolist(),
        ee_poses=np.asarray(arrays['ee'])[indices].tolist(), goal_poses=goals[indices].tolist(),
        obstacles=scene['objects'], waypoints=points.tolist(), body_frames=frames,
        position_mm=(1000*error).tolist(),
        clearance_mm=[float(v)*1000 if np.isfinite(v) else None for v in clearance[indices]],
        links=series.meta['links'] if series.meta else [], execution_level=status)
    videos, errors = [], []
    for view in ('overview', 'top'):
        try:
            videos.append(render_video(out, robot, arrays, goals, indices, view, deadline, notify, points, series))
        except Exception as exc:
            errors.append(dict(view=view, error=repr(exc)))
    gallery = ''.join(f'<section><h2>{html.escape(path.stem)}</h2><a href="{path.stem}.pdf">PDF</a> · <a href="{path.name}">PNG</a><img loading="lazy" src="{path.name}"></section>'
        for path in sorted(out.glob('*.png')) if path.stem[:2].isdigit() or path.stem.startswith('static_'))
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row[:5])+'</tr>' for row in body_figures.get('summary', []))
    page = HTML.replace('__DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    page = page.replace('__TITLE__', html.escape(title)).replace('__STATUS__', html.escape(status))
    page = page.replace('__VIDEOS__', ''.join(f'<section><h2>{v["file"]}</h2><video src="{v["file"]}" controls loop muted></video></section>' for v in videos))
    page = page.replace('__GALLERY__', gallery).replace('__BODY_TABLE__', rows)
    page = page.replace('__RENDER_ERRORS__', html.escape(json.dumps(errors, ensure_ascii=False)) if errors else '两视角视频均已生成')
    (out/'index.html').write_text(page, encoding='utf-8')
    return dict(status='COMPLETE' if len(videos)==2 and body_figures['status']=='COMPLETE' else 'PARTIAL_MEDIA',
        videos=videos, errors=errors, body_figures=body_figures, new_native_steps=0,
        overlays=dict(actual_ee_frames=True, time_varying_goal_frames=True, actual_cumulative_ee_paths=True,
            original_waypoints=True, whole_body_geometry=True, nearest_body_highlights=True,
            exact_distance_connectors=True, axis_colors='RGB = XYZ', display_band_m=.010, physical_gate_m=0.),
        media_frame_native_indices=indices.tolist(), media_frame_times=times[indices].tolist(),
        body_diagnostic_exact_frame_count=sum(x is not None for x in frames),
        source='saved actual native states and exact control-tick geometric observations only',
        files={p.name: dict(bytes=p.stat().st_size, sha256=file_digest(p)) for p in out.iterdir() if p.is_file()})


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{margin:0;background:#101722;color:#dce5ee;font:16px system-ui}main{max-width:1200px;margin:auto;padding:28px}section{background:#182333;border:1px solid #304255;border-radius:12px;padding:18px;margin:18px 0}h1{font-size:28px}h2{font-size:19px}a{color:#64cbe7}img,video{width:100%;border-radius:8px}canvas{width:100%;height:520px;background:#101923;touch-action:none}input{width:65%}button{padding:9px 18px}small{color:#a8b8c7}.badge{color:#ffcd78}table{width:100%;font-size:13px;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #384958;padding:6px}</style>
<main><h1>__TITLE__</h1><p class="badge">实际记录等级：__STATUS__</p><p>刚性侧2个、连续体侧5个障碍，原七不规则航点。媒体只使用保存的实际状态，失败只展示实际前缀。双臂EE及Goal轴为RGB=X/Y/Z，青色/橙色为累计实际末端轨迹。实体卫星没有注入速度；系统执行结果不等于网络自主避障。</p>
<section><h2>交互三维：末端、目标与臂身</h2><small>拖动旋转，滚轮缩放。连线为各障碍到最近实体臂段的精确距离，黄线标最近臂段，进入10mm显示带变红；验收不穿透门仍为0。骨架来自实际body位置，完整实体见视频。距离和形变是观测，不是反事实因果优势证明。</small><canvas id="c"></canvas><button id="play">播放/暂停</button> <input id="seek" type="range" min="0" value="0"><p id="info"></p><table><thead><tr><th>障碍</th><th>最近臂段 / 几何</th><th>间隙</th></tr></thead><tbody id="nearest"></tbody></table></section>
__VIDEOS__<p>__RENDER_ERRORS__</p><p><a href="timeseries.csv">执行指标CSV</a> · <a href="obstacle_nearest_body.csv">逐控制tick最近臂段CSV</a> · <a href="obstacle_body_minimum_summary.csv">各障碍最小距离表</a> · <a href="manifest.json">来源与文件hash</a></p>
<section><h2>各障碍最小实体间隙记录</h2><table><thead><tr><th>障碍</th><th>距离 m</th><th>时间 s</th><th>臂段</th><th>几何</th></tr></thead><tbody>__BODY_TABLE__</tbody></table></section>__GALLERY__</main>
<script>const D=__DATA__;const c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('seek');s.max=D.time.length-1;let i=0,playing=false,yaw=.7,pitch=.45,zoom=1,drag=null,last=0;const all=D.bodies.flat().concat(D.waypoints),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of all)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k])}const center=lo.map((x,k)=>(x+hi[k])/2),span=Math.max(...hi.map((x,k)=>x-lo[k]),1);
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);return[c.width/2+a*c.width*.55/span*zoom,c.height/2-(z*Math.cos(pitch)-b*Math.sin(pitch))*c.width*.55/span*zoom]}
function segment(a,b,color,width=2,dash=false){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.setLineDash(dash?[6*devicePixelRatio,4*devicePixelRatio]:[]);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([])}
function label(p,text,color='#eaf3ff'){let a=project(p);ctx.fillStyle=color;ctx.font=(11*devicePixelRatio)+'px system-ui';ctx.fillText(text,a[0]+4*devicePixelRatio,a[1]-4*devicePixelRatio)}
function dot(p,color,r=5){let a=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],r,0,7);ctx.fill()}
function frame(T,name,target){let origin=T.slice(0,3).map(r=>r[3]),colors=['#ff4545','#39df5e','#518aff'];for(let j=0;j<3;j++){let end=origin.map((v,k)=>v+(target ? .09 : .065)*T[k][j]);segment(origin,end,colors[j],target?3:2,target);label(end,name+' '+['X','Y','Z'][j],colors[j])}}
function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=520*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);let b=D.bodies[i];for(let j=1;j<b.length;j++){let p=D.parents[j];if(p>0)segment(b[p],b[j],'#73b8c2',2)}for(let o of D.obstacles){let a=project(o.position);ctx.fillStyle=o.object_id.startsWith('R')?'#ed8a5155':'#9e77ed55';ctx.strokeStyle=o.object_id.startsWith('R')?'#ed8a51':'#b995ff';ctx.beginPath();ctx.arc(...a,Math.max(2,o.size[0]*c.width*.55/span*zoom),0,7);ctx.fill();ctx.stroke();label(o.position,o.object_id)}for(let k=0;k<7;k++){dot(D.waypoints[k],'#719cff',3*devicePixelRatio);label(D.waypoints[k],'W'+(k+1),'#91baff');if(k)segment(D.waypoints[k-1],D.waypoints[k],'#4c78a54c',1,true)}for(let arm=0;arm<2;arm++){for(let k=1;k<=i;k++)segment(D.ee[k-1][arm],D.ee[k][arm],arm?'#ffbd53':'#40dfef',3);frame(D.ee_poses[i][arm],'EE'+arm,false);frame(D.goal_poses[i][arm],'Goal'+arm,true)}let f=D.body_frames[i],rows='';if(f){for(let n of f.nearest){let color=n.distance_m<.010?'#ff4056':'#e8d34c';segment(n.connector[0],n.connector[1],color,2);segment(b[D.parents[n.body_id]],b[n.body_id],color,n.distance_m<.010?6:4);for(let child=1;child<b.length;child++)if(D.parents[child]===n.body_id)segment(b[n.body_id],b[child],color,n.distance_m<.010?6:4);dot(n.connector[1],color,4*devicePixelRatio);label(n.connector[1],n.obstacle+'/'+n.body_name,color);rows+=`<tr><td>${n.obstacle}</td><td>${n.body_name} / ${n.geom_name}</td><td>${(n.distance_m*1000).toFixed(3)} mm</td></tr>`}}else rows='<tr><td colspan="3">此帧没有同步臂段诊断；不插值成精确距离</td></tr>';document.getElementById('nearest').innerHTML=rows;document.getElementById('info').textContent=`t=${D.time[i].toFixed(3)} s · 双臂位置误差 ${D.position_mm[i].map(x=>x.toFixed(3)).join(' / ')} mm · 全机间隙 ${D.clearance_mm[i]===null?'NA':D.clearance_mm[i].toFixed(3)} mm`;s.value=i}
s.oninput=()=>{i=+s.value;draw()};document.getElementById('play').onclick=()=>playing=!playing;c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;function tick(t){if(playing&&t-last>66){i=(i+1)%D.time.length;last=t;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);</script></html>'''
