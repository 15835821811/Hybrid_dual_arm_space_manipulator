"""Draft layout only: one fixed initial geometry check, no integration or search."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior.initial_state import InitialState,STATE
from ff_prior.geometry import CollisionBackend
from ff_prior.storage import file_digest,digest

OUT=ROOT/'ff_prior_balanced_scene/evidence';OUT.mkdir(parents=True,exist_ok=True)
public=ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json'
w=json.loads(public.read_text(encoding='utf-8'))['waypoints'];points=np.asarray(w['waypoint_points_m'])
objects=[Obstacle('R1','sphere',(.82,-.91,.22),(.035,0,0)),
         Obstacle('R2','sphere',(1.24,-.43,-.045),(.030,0,0))]
roles=['rigid upper/forearm passage','rigid forearm/wrist approach']
for k,seg in enumerate((0,2,4),1):
    objects.append(Obstacle(f'C{k}','sphere',tuple((points[seg]+points[seg+1])/2),(.025,0,0)))
    roles.append(f'continuum path segment W{seg+1}-W{seg+2}')
objects.extend([Obstacle('C4','sphere',(1.12,.750,.025),(.028,0,0)),
                Obstacle('C5','sphere',(1.50,.540,-.040),(.028,0,0))])
roles.extend(['continuum middle-body links 14-18 vicinity','continuum distal-body links 22-25 vicinity'])
scene=SceneSpec(tuple(objects),'balanced_2rigid_5continuum_proposal_v1')
a=np.load(ROOT/'artifacts/v7_1/horizon_restore_001/units/u00/execution/native_trace.npz')
robot=RobotAdapter(scene=scene)
assert mujoco.mj_stateSize(robot.model,STATE)==len(a['integration'][0])
# New scene initial identity; retain ALL numerical initial state entries. Not a handoff.
initial=InitialState(a['integration'][0].copy(),robot.identity()['scene_model_hash'])
data=initial.reset(robot);backend=CollisionBackend(robot);check=backend.scan(data)
readback=np.empty_like(initial.values);mujoco.mj_getState(robot.model,data,readback,STATE)
assert np.array_equal(readback,initial.values)
assert check['passed'],'INVALID_DRAFT_INITIAL_GEOMETRY: preserve evidence and revise openly'
objects_json=scene.to_dict()
checks=[]
for p in points:
    checks.append(min(float(np.linalg.norm(p-np.asarray(o.position))-o.size[0]) for o in objects))
rigid_sep=float(np.linalg.norm(np.asarray(objects[0].position)-objects[1].position)-objects[0].size[0]-objects[1].size[0])
source=ROOT/'artifacts/v7_1/horizon_restore_001/units/u00/visuals/index.html'
payload=json.loads(re.search(r'const D=(.*?);const c=',source.read_text(encoding='utf-8')).group(1))
draft=dict(status='LAYOUT_AND_COST_PROPOSAL_NOT_AN_EXECUTED_TRAJECTORY',scene=objects_json,
    roles=dict(zip([o.object_id for o in objects],roles)),waypoints=w,
    public_waypoints_sha256=digest(w['waypoint_points_m']),public_task_file_sha256=file_digest(public),
    original_full_numeric_initial_state_preserved=True,new_initial_state_id=initial.state_id,
    new_scene_model_hash=robot.identity()['scene_model_hash'],initial_geometry=check,
    fixed_state_geometry_evaluations=1,new_coupled_calls=0,new_native_steps=0,
    rigid_surface_separation_m=rigid_sep,waypoint_point_to_spheres_clearance_m=checks,
    caveat='Waypoint point clearance and one initial geometry check do not certify whole-body trajectories or dynamic reachability.',
    physical_trial_authorized=False,source_native_sha256=file_digest(ROOT/'artifacts/v7_1/horizon_restore_001/units/u00/execution/native_trace.npz'))
(OUT/'layout.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
body=data.xpos.copy();parent=robot.model.body_parentid
for ax,(x,y),title in zip(axes,[(0,1),(0,2),(1,2)],['俯视 XY','侧视 XZ','连续体航点平面 YZ']):
    for i in range(1,len(body)):
        if parent[i]>0:
            arm_color='#138eaa' if i<65 else '#c67819'
            ax.plot(body[[parent[i],i],x],body[[parent[i],i],y],color=arm_color,lw=2,alpha=.8)
    ax.plot(points[:,x],points[:,y],'--o',color='#176cba',ms=4,lw=1.5,label='原7航点连接示意（非执行轨迹）')
    for k,p in enumerate(points):ax.annotate(f'W{k+1}',(p[x],p[y]),xytext=(4,4),textcoords='offset points',fontsize=8,color='#145c98')
    for o in objects:
        color='#ce6943' if o.object_id.startswith('R') else '#7846b5' if o.object_id in ('C4','C5') else '#349b55'
        ax.add_patch(Circle((o.position[x],o.position[y]),o.size[0],fc=color,ec=color,alpha=.4))
        ax.annotate(o.object_id,(o.position[x],o.position[y]),xytext=(5,-12),textcoords='offset points',fontsize=9,color=color,weight='bold')
    ax.set(title=title,xlabel='XYZ'[x]+' / m',ylabel='XYZ'[y]+' / m');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
axes[0].set_xlim(.15,2.04);axes[0].set_ylim(-1.2,.88)
axes[1].set_xlim(.15,2.04);axes[1].set_ylim(-.28,.32)
axes[2].set_xlim(.30,.86);axes[2].set_ylim(-.25,.25)
fig.suptitle('新场景布局草案 · 刚性2 / 连续体5 · 骨架为完整初态\nC1–C3约束航点段；C4–C5面向臂身；尚未运行新场景',fontsize=15)
fig.savefig(OUT/'layout.png',dpi=160);fig.savefig(OUT/'layout.pdf');plt.close(fig)
# Saved geometry preview only; no dynamic playback and no fabricated trajectory.
preview=dict(bodies=body.tolist(),parents=parent.tolist(),points=points.tolist(),objects=objects_json['objects'],roles=draft['roles'])
(OUT/'preview_data.json').write_text(json.dumps(preview,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:draft[k] for k in ('status','rigid_surface_separation_m','waypoint_point_to_spheres_clearance_m','initial_geometry','new_native_steps')},ensure_ascii=False))
