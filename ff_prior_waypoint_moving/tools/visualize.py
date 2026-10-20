import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,numpy as np
from ff_prior.robot_adapter import RobotAdapter,physical_spec
from ff_prior.scenes import SceneSpec
from ff_prior.storage import atomic_json,file_digest
import ff_prior_waypoint_scene.visualize as view
ROOT=Path.cwd();OUT=ROOT/'artifacts/v7_1/waypoint_moving_gate_001';view.OUT=OUT
summary=json.loads((OUT/'summary.json').read_text());cards=[]
from scipy.interpolate import CubicHermiteSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spec=physical_spec()
for i,result in enumerate(summary['results']):
 d=OUT/f'train_{i:02d}';rep=json.loads((d/'representation.json').read_text());src=rep['source']['path']
 cards.append(f'<h2>TRAIN {i} — {result["status"]}</h2><p>{src}</p><p><a href="train_{i:02d}/result.json">实际验收结果</a> · <a href="train_{i:02d}/representation.json">表示检查</a></p>')
 with np.load(d/'reference.npz')as z:spline=CubicHermiteSpline(z['time'],z['q'],z['dq'])
 t=np.linspace(0,27,13501)
 fig,axes=plt.subplots(3,1,figsize=(11,7),layout='constrained')
 for ax,values,title in zip(axes,[np.max(abs(spline(t)[:,:10]),axis=1),np.max(abs(spline(t,1))/spec.planner_velocity_limits,axis=1),np.max(abs(spline(t,2))/spec.planner_acceleration_limits,axis=1)],['Continuum reference max |q| (rad)','Velocity / limit','Acceleration / limit']):
  ax.plot(t,values);ax.axhline(1,color='red',ls='--');ax.set_title(title);ax.set_xlabel('time (s)')
 fig.savefig(d/'representation.png',dpi=130);plt.close(fig)
 cards.append(f'<img src="train_{i:02d}/representation.png"><p>参考表示曲线，不是实际状态；准入使用各段解析极值。</p>')
 if (d/'native_trace.npz').exists() and result.get('steps',0)>0:
  with np.load(d/'native_trace.npz')as z:a={k:z[k]for k in z.files}
  robot=RobotAdapter(scene=SceneSpec.from_dict(json.loads((d/'identity.json').read_text())['scene']))
  view.plots(d,a,result);n=view.video(robot,d,a)
  atomic_json(d/'visualization_provenance.json',dict(video_frames=n,native_sha256=file_digest(d/'native_trace.npz'),video_sha256=file_digest(d/'actual_motion.mp4'),new_physics_steps=0))
  cards.append(f'<video controls preload="metadata" src="train_{i:02d}/actual_motion.mp4"></video><p>仅显示本轮实际执行段；提前停止不是完整27秒成功。</p><img src="train_{i:02d}/tracking.png"><img src="train_{i:02d}/world_paths.png"><img src="train_{i:02d}/joint_domain_and_torque.png"><p><a href="train_{i:02d}/tracking.csv">逐步CSV</a></p>')
 else:cards.append('<p>未进入力矩执行，不提供虚构运动回放。</p>')
 print('rendered',i,flush=True)
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>移动TRAIN表示与执行认证</title><style>body{max-width:1100px;margin:2em auto;padding:1em;font:16px sans-serif;line-height:1.6}video,img{max-width:100%}</style><h1>移动TRAIN表示与执行认证</h1><p>排除冲突轨迹。静止DEV为分布外诊断。本轮没有扩库，所有图像来自保存状态。</p>'+''.join(cards),encoding='utf-8')
