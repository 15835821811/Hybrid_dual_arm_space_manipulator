"""Read saved world-steer evidence only. No robot creation or physics calls."""
from pathlib import Path
import json,collections,csv,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/v7_1/world_steer_001'
OUT=ROOT/'ff_prior_world_steer/evidence'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
ident=read(RUN/'run.json')['identity']
goal=np.asarray(ident['selection']['entries'][0]['task']['condition']['goals'])
rows=[];attempt_modes=collections.Counter();trial_reasons=collections.Counter()
edge_reasons=collections.Counter();pairs=collections.Counter();modes_pass=collections.Counter()
world_patterns=collections.Counter();failed_world_patterns=collections.Counter()
edge_count=0;world_retained=0;total_nodes=0;numdiff=0;actual_edge_intervals=0
best=None;step_sizes=[];depth_list=[];maxelapsed=[];all_motion_reject=0
for unit in ident['selection']['units']:
 for k in range(5):
  d=RUN/'units'/unit['id']/'candidates'/f'{k:02d}'
  r=read(d/'tree/result.json');a=np.load(d/'tree/nodes.npz')
  pos=np.linalg.norm(a['ee'][:,:,:3,3]-goal[:,:3,3],axis=-1)
  rot=Rotation.from_matrix((np.swapaxes(goal[:,:3,:3],-1,-2)@a['ee'][:,:,:3,:3]).reshape(-1,3,3)).magnitude().reshape(-1,2)
  q=np.max(np.c_[pos/.005,rot/np.deg2rad(1)],axis=1)
  idx=int(np.argmin(q));depth=[]
  for i in range(len(a['parent'])):
   v=i;dep=0
   while a['parent'][v]>=0:dep+=1;v=int(a['parent'][v])
   depth.append(dep)
  total_nodes+=len(a['q'])-1;numdiff+=r['world_linearizations'];depth_list.extend(depth)
  maxelapsed.append(float(max(a['elapsed'])))
  attempts=read(d/'tree/attempts.json')
  counter=collections.Counter();accepted=collections.Counter()
  for attempt in attempts:
   mode=attempt['sampling'];attempt_modes[mode]+=1;counter[mode]+=1
   if attempt['accepted']:
    modes_pass[mode]+=1;accepted[mode]+=1
    p=int(attempt['parent']);v=int(attempt['node'])
    step_sizes.append(float(max(abs(a['q'][v]-a['q'][p]))))
   if mode=='WORLD_DESCENT':
    pattern=f"{unit['id']}/{k}/{attempt['parent']}"
    world_patterns[pattern]+=1
    if not attempt['accepted']:failed_world_patterns[pattern]+=1
   for trial in attempt['trials']:
    reason='ACCEPTED' if trial['accepted'] else trial.get('reason','UNKNOWN')
    if reason=='MOTION_OR_GEOMETRY_REJECTED' and not trial['edge_ids']:
     reason='MOTION_REJECTED_BEFORE_PROPAGATION';all_motion_reject+=1
    trial_reasons[reason]+=1
  for f in (d/'tree/edges').glob('*/report.json'):
   er=read(f);edge_count+=1;actual_edge_intervals+=er['actual_intervals']
   if not er['complete']:
    reason='PENETRATION'
    pairs[str(er['minimum_pair'])]+=1
   elif not er['numerical_ok']:reason='NUMERICAL'
   elif er['max_geometry_step_m']>.002:reason='REFINE_GEOMETRY_GRID'
   elif er['accepted']:reason='EDGE_SCREEN_PASS'
   else:reason='OTHER'
   edge_reasons[reason]+=1
  seed=read(d/'goal_seed/result.json')
  row={'unit':unit['id'],'candidate':k,'seed':unit['proposal_seeds'][k],
   'stop':r['stop'],'attempts':r['attempts'],'retained_nodes':len(a['q'])-1,
   'linearizations':r['world_linearizations'],'edges':r['propagations'],
   'seed_Q':seed['Q'],'best_Q':float(q[idx]),'best_node':idx,'best_elapsed_s':float(a['elapsed'][idx]),
   'best_remaining_s':27.-float(a['elapsed'][idx]),'best_depth':depth[idx],
   'max_depth':max(depth),'max_elapsed_s':float(max(a['elapsed'])),
   'best_position_mm':(pos[idx]*1000).tolist(),'best_rotation_deg':np.rad2deg(rot[idx]).tolist(),
   'world_attempts':counter['WORLD_DESCENT'],'world_accepted':accepted['WORLD_DESCENT'],
   'all_pose_pass_nodes':int(np.sum(q<=1)),
   'solution_node':r['solution_node'],'fitted':(d/'fit.npz').exists(),
   'repaired':(d/'repair').exists(),'validated':(d/'before_repair').exists()}
  rows.append(row)
  if best is None or row['best_Q']<best['best_Q']:best=row
root_q=float(q[0])
audit=read(RUN/'independent_verification.json')
summary={'run':'world_steer_001','elapsed_s':read(RUN/'status.json')['elapsed_s'],
 'source_files':len(ident['source_hashes']),'input_files':len(ident['binding']['files']),
 'raw_audit':audit['status'],'raw_checks':len(audit['checks']),
 'raw_failed_checks':[k for k,v in audit['checks'].items() if not v],
 'units':3,'candidates':15,'model_valid':0,'native_executed':0,'replays':0,
 'initial_Q':root_q,'best':best,'nodes_retained_excluding_roots':total_nodes,
 'all_pose_pass_nodes':sum(r['all_pose_pass_nodes'] for r in rows),
 'solution_count':sum(r['solution_node'] is not None for r in rows),
 'attempt_modes':dict(attempt_modes),'accepted_modes':dict(modes_pass),
 'trial_reasons':dict(trial_reasons),'edge_reports':edge_count,'edge_reasons':dict(edge_reasons),
 'penetrating_pairs':dict(pairs),'world_linearizations':numdiff,
 'world_parent_unique':len(world_patterns),'world_parent_repeated_attempts':sum(v-1 for v in world_patterns.values()),
 'failed_world_parent_unique':len(failed_world_patterns),
 'failed_world_repeated_attempts':sum(v-1 for v in failed_world_patterns.values()),
 'actual_edge_intervals_from_saved_reports':actual_edge_intervals,
 'stops':dict(collections.Counter(r['stop'] for r in rows)),
 'budget':read(RUN/'budget.json'),
 'all_new_steps_during_this_analysis':0,'qualification':'model tree evidence, not MODEL_VALID or execution'}
(OUT/'statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'candidate_statistics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (OUT/'candidate_statistics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# Only a diagnostic curve, never an actual execution visualization.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,2,figsize=(13,8))
labels=[r['unit']+'/'+str(r['candidate']) for r in rows]
axs[0,0].bar(labels,[r['best_Q'] for r in rows]);axs[0,0].axhline(1,color='red',ls='--',label='原位姿门Q=1');axs[0,0].legend();axs[0,0].set_title('各候选最优模型树节点：均未达完整目标')
axs[0,1].bar(labels,[r['best_elapsed_s'] for r in rows]);axs[0,1].axhline(27,color='red',ls='--');axs[0,1].set_title('最优节点已用时间（秒），原任务27秒')
axs[1,0].bar(labels,[r['world_accepted']/r['world_attempts'] for r in rows]);axs[1,0].set_title('世界方向短时扩展接受比例（非路径成功率）')
axs[1,1].bar(['穿透','需要加密','边筛选通过'],[edge_reasons['PENETRATION'],edge_reasons['REFINE_GEOMETRY_GRID'],edge_reasons['EDGE_SCREEN_PASS']]);axs[1,1].set_title('逐网格边报告；通过边不等于完整合格轨迹')
for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',rotation=65)
fig.suptitle('原五障碍D3：仅保存模型树诊断；无真实执行、无独立重放',fontsize=13)
fig.tight_layout()
folder=ROOT/'ff_prior_world_steer/docs/figures';folder.mkdir(exist_ok=True)
fig.savefig(folder/'world_steer_failure.png',dpi=170);fig.savefig(folder/'world_steer_failure.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='budget'},ensure_ascii=True,indent=2))

