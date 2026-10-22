"""Compact descriptive statistics and plots from saved evidence only."""
from pathlib import Path
import sys,json,csv
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
RUN=ROOT/'artifacts/v7_1/contact_steer_001';PACKAGE=ROOT/'ff_prior_contact_steer'
selection=read_json(PACKAGE/'evidence/selection.json');goals=selection['entries'][0]['task']['condition']['goals']

def dist(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(np.max(a))) if len(a) else dict(count=0)

rows=[];edgecounts=Counter();pairs=Counter();reasons=Counter();solvers=Counter();active=[];allactive=[]
half_distance=[];half_goal=[];cumulative=[];accepted_up=0;unique_action_count=0;geometry_steps=[];clearances=[]
for spec in selection['units']:
    d=RUN/'units'/spec['id'];result=read_json(d/'tree/result.json');attempts=read_json(d/'tree/attempts.json')
    n=np.load(d/'tree/nodes.npz');count=result['historical_prefix_nodes']
    pos,rot=pose_errors(n['ee'],goals);Q=np.max(np.c_[pos/.005,rot/np.deg2rad(1)],axis=1)
    best=int(np.argmin(Q));expanded={r['parent'] for r in attempts}
    unexpanded=[i for i in range(count-1,len(Q)) if i not in expanded]
    chains=[];i=best
    while i>=0:chains.append(i);i=int(n['parent'][i])
    chains.reverse()
    trials=[t for a in attempts for t in a['trials']]
    for a in attempts:
        for t in a['trials']:
            reasons[t['reason']]+=1
            if t['accepted'] and t['after_Q']>a['before_Q']:accepted_up+=1
        for s in a['solvers']:
            solvers['total']+=1;solvers['success']+=int(s['success']);solvers['linear_admissible']+=int(s['linear_admissible'])
    acts=[t['exact_action_key'] for t in trials if t.get('edge_ids')];unique_action_count+=len(set(acts))
    row=dict(unit=spec['id'],old_prefix_Q=spec['source_Q'],new_best_Q=float(Q[best]),
        reduction_percent=100*(1-float(Q[best])/spec['source_Q']),best_node=best,
        best_position_mm=(pos[best]*1000).tolist(),best_rotation_deg=np.rad2deg(rot[best]).tolist(),
        best_elapsed_s=float(n['elapsed'][best]),best_remaining_s=27.-float(n['elapsed'][best]),
        best_chain=chains,historical_prefix_nodes=count,new_nodes=result['new_nodes'],
        accepted_new_pose_pass_nodes=int(np.count_nonzero(Q[count:]<=1)),expanded_parents=len(expanded),
        unexpanded_nodes=len(unexpanded),unexpanded_time_feasible=sum(n['elapsed'][i]+1.35<=27. for i in unexpanded),
        minimum_remaining_s=float(27.-np.max(n['elapsed'])),stop=result['stop'],
        trials=len(trials),accepted_trials=sum(t['accepted'] for t in trials),
        repeated_executed_action_keys=len(acts)-len(set(acts)),unique_executed_actions=len(set(acts)),
        actual_coupled=read_json(d/'calls.json')['coupled'],fitted=(d/'fit.npz').exists(),
        repaired=(d/'repair/result.json').exists(),executed=(d/'execution/native_trace.npz').exists())
    rows.append(row);cumulative.append(np.minimum.accumulate(Q[count-1:]))
    for p in sorted((d/'tree/edges').glob('*/report.json')):
        r=read_json(p)
        reason='EDGE_SCREEN_PASS' if r['accepted'] else ('PENETRATION' if not r['complete'] else 'REFINE_GEOMETRY_GRID' if r['numerical_ok'] else 'NUMERICAL_FAIL')
        edgecounts[reason]+=1
        if reason=='PENETRATION':
            v=r['minimum_pair'];pairs[f"{v['body_a']} / {v['body_b']} ({v['geom_a']} / {v['geom_b']})"]+=1
        if r['accepted']:geometry_steps.append(r['max_geometry_step_m']);clearances.append(r['minimum_clearance_m'])
    for p in sorted((d/'tree/linear').glob('*/jacobian.npz')):
        a=np.load(p);r=read_json(p.with_suffix('.json'));active.append(len(r['active']));allactive.append(r['all_near_pairs'])
        k=int(a['half_coordinate']);h=float(a['epsilon'][k]);half=(a['half_distances']-a['distances'])/(h*.5)
        half_distance.append(float(np.max(abs(half-a['jd'][:,k]),initial=0)))
        hp,hr=pose_errors(a['half_ee'],a['nominal_ee']);half_goal.append(float(max(hp)*1000))
summary=read_json(RUN/'summary.json');budget=read_json(RUN/'budget.json');audit=read_json(RUN/'independent_verification.json')
statistics=dict(run='contact_steer_001',elapsed_s=read_json(RUN/'status.json')['elapsed_s'],
    complete_qualified=summary['qualified'],candidate_denominator=3,execution_denominator=3,
    native_and_replay_steps=budget['actual_native_and_replay_steps'],actual_coupled=budget['actual_coupled'],
    reserved_model_intervals=budget['integration_intervals'],new_nodes=sum(r['new_nodes'] for r in rows),
    historical_prefix_nodes=sum(r['historical_prefix_nodes'] for r in rows),unit_rows=rows,
    edge_reports=dict(edgecounts),penetration_pairs=dict(pairs.most_common()),trial_reasons=dict(reasons),
    solvers=dict(solvers),accepted_trials_with_Q_increase=accepted_up,unique_actions=unique_action_count,
    linearizations=len(active),active_observations_selected=sum(active),active_observations_available=sum(allactive),
    active_truncated_linearizations=sum(a<b for a,b in zip(active,allactive)),active_count=dist(active),
    half_step_distance_derivative_difference_m_per_rad=dist(half_distance),
    half_step_nominal_pose_shift_mm=dist(half_goal),accepted_edge_clearance_m=dist(clearances),
    accepted_edge_max_geometry_step_m=dist(geometry_steps),
    raw_audit=dict(status=audit['status'],checks=len(audit['checks']),failed=[k for k,v in audit['checks'].items() if not v]),
    no_new_physics=True,interpretation='chosen model-prefix continuation only; not executed path or advantage comparison')
atomic_json(PACKAGE/'evidence/statistics.json',statistics)
with (PACKAGE/'evidence/unit_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['unit','old_prefix_Q','new_best_Q','reduction_percent','best_elapsed_s','best_remaining_s','new_nodes','expanded_parents','stop']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(2,2,figsize=(12,8.2))
for r,q in zip(rows,cumulative):axs[0,0].plot(np.arange(len(q)),q,label=r['unit'])
axs[0,0].axhline(1,color='red',ls='--',label='原位姿门 Q=1');axs[0,0].set(xlabel='新保留模型节点序号（非时间）',ylabel='累计最优 Q',title='局部误差下降，但未到目标');axs[0,0].legend()
x=np.arange(3);axs[0,1].bar(x-.18,[r['best_position_mm'][0] for r in rows],.36,label='连续体');axs[0,1].bar(x+.18,[r['best_position_mm'][1] for r in rows],.36,label='刚性臂')
axs[0,1].axhline(5,color='red',ls='--');axs[0,1].set(xticks=x,xticklabels=[r['unit'] for r in rows],ylabel='mm',title='各单元最优节点的双臂位置误差');axs[0,1].legend()
axs[1,0].bar(['边筛选通过','穿透拒绝','需要细化'],[edgecounts[k] for k in ('EDGE_SCREEN_PASS','PENETRATION','REFINE_GEOMETRY_GRID')]);axs[1,0].set(ylabel='逐网格报告数',title='全机几何仍独立验收')
for i,r in enumerate(rows):
    axs[1,1].barh(i,r['best_elapsed_s'],color='#4795b8');axs[1,1].barh(i,r['best_remaining_s'],left=r['best_elapsed_s'],color='#d5e5ed')
axs[1,1].set(yticks=x,yticklabels=[r['unit'] for r in rows],xlabel='秒',title='最优模型节点已用时间 / 剩余原27秒')
for ax in axs.flat:ax.grid(alpha=.2)
fig.suptitle('contact_steer_001：保存模型证据；0/3完整路径，0真实执行，0重放',fontsize=14)
fig.tight_layout();folder=PACKAGE/'docs/figures';folder.mkdir(exist_ok=True)
for ext in ('png','pdf'):fig.savefig(folder/f'contact_steer_diagnosis.{ext}',dpi=170)
print(json.dumps(read_json(PACKAGE/'evidence/statistics.json'),ensure_ascii=True,indent=2))
