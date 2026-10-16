"""Automatic Chinese report; separate neural proposal and shared physical repair."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json
from ff_prior_coverage_train.storage import atomic_json
from ff_prior.contracts import all_pass,MODEL_GATES
from ff_prior_generalization.reporting import pre_norm
from .protocol import PROTOCOL as P,METHODS


def paired_quality(task_metrics):
    result={}
    for method in ('structured','learned_mean','frozen_diffusion'):
        pairs=[(v['expanded_diffusion']['pre_median_norm'],v[method]['pre_median_norm'])
               for v in task_metrics.values()
               if all(v[m]['completed']==15 and v[m]['pre_median_norm'] is not None
                      for m in ('expanded_diffusion',method))]
        if len(pairs)!=8:
            result[method]={'status':'INCOMPLETE','paired_tasks':len(pairs)};continue
        a=np.asarray(pairs);ratios=np.maximum(a[:,0],1e-12)/np.maximum(a[:,1],1e-12)
        rng=np.random.default_rng(P['bootstrap_seed'])
        boot=np.exp(np.log(ratios)[rng.integers(0,8,(P['bootstrap_samples'],8))].mean(axis=1))
        lo,hi=np.quantile(boot,[.025,.975])
        result[method]={'status':'DONE','paired_tasks':8,'median_task_ratio':float(np.median(ratios)),
            'geometric_mean_task_ratio':float(np.exp(np.log(ratios).mean())),
            'bootstrap95_geometric_ratio':[float(lo),float(hi)],
            'new_lower_tasks':int(sum(ratios<1)),'new_higher_tasks':int(sum(ratios>1)),
            'pre_registered_quality_pass':bool(np.median(ratios)<=.85 and hi<1.)}
    return result


def report(out,status):
    out=Path(out);fp=out/'frozen_tasks.json'
    entries=read_json(fp)['entries'] if fp.exists() else []
    metrics={};tasks={e['sample_id']:{} for e in entries}
    for method in METHODS:
        rs=[read_json(p) for p in (out/'candidates').glob(f'*/*/{method}/*.json')]
        selected=[(p,read_json(p)) for p in (out/'executions').glob(f'*/*/{method}/outcome.json')]
        norms=[pre_norm(r) for r in rs if 'sampling' in r]
        group_metrics={}
        for split in ('DEV','NEW_VAL'):
            ids={e['sample_id'] for e in entries if e['split']==split}
            group=[r for r in rs if r['split']==split];outcomes=[r for p,r in selected if p.parents[1].name in ids]
            ns=[pre_norm(r) for r in group if 'sampling' in r]
            group_metrics[split]={'completed':len(group),'eligible':sum(r['eligible'] for r in group),
                'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),
                'selected_outcomes':len(outcomes),'candidate_errors':sum(r['status']!='DONE' for r in group),
                'per_seed_eligible':{str(s):sum(r['eligible'] and r['seed_group']==s for r in group) for s in range(3)},
                'pre_pose_pass':sum(v<=1 for v in ns),'pre_pose_diagnostic_count':len(ns),
                'pre_median_norm':float(np.median(ns)) if ns else None}
        for e in entries:
            group=[r for r in rs if r['sample_id']==e['sample_id']];ns=[pre_norm(r) for r in group if 'sampling' in r]
            tasks[e['sample_id']][method]={'completed':len(group),'eligible':sum(r['eligible'] for r in group),
                'pre_median_norm':float(np.median(ns)) if ns else None,'pre_pose_pass':sum(v<=1 for v in ns)}
        diagnostics={}
        for stage in ('raw','smoothed_clipped'):
            dr=[read_json(p) for p in (out/'proposal_diagnostics').glob(f'*/{method}/{stage}.json')]
            diagnostics[stage]={'completed':len(dr),'eligible_with_frozen_goal':sum(r['eligible'] for r in dr),
                'physical_MODEL_gates_pass':sum(all_pass(r['validation']['gates'],MODEL_GATES) for r in dr if 'validation' in r),
                'errors':sum(r['status']!='DONE' for r in dr)}
        metrics[method]={'completed':len(rs),'eligible':sum(r['eligible'] for r in rs),
            'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for _,r in selected),
            'selected_outcomes':len(selected),'candidate_errors':sum(r['status']!='DONE' for r in rs),
            'pre_pose_pass':sum(v<=1 for v in norms),'pre_pose_diagnostic_count':len(norms),
            'groups':group_metrics,'proposal_diagnostics':diagnostics,
            'network_calls':sum(r['actual_network_calls'] for r in rs),
            'coupled_queries':sum(r['model_calls'].get('coupled',0) for r in rs),
            'wall_s':sum(r['wall_s'] for r in rs),
            'rejection_gates':dict(Counter(k for r in rs for k,v in r.get('validation',{}).get('gates',{}).items() if not v))}
    fresh={e['sample_id']:tasks[e['sample_id']] for e in entries if e['split']=='NEW_VAL'}
    quality=paired_quality(fresh)
    trpath=out/'learning/goal_loss/training.json';tr=read_json(trpath) if trpath.exists() else {}
    trained=(tr.get('status')=='PASS' and tr.get('updates')==6000 and tr.get('all_TRAIN_rows_seen') is True
             and len(tr.get('train_draw_counts',{}))==12688
             and tr.get('group_draw_counts')=={'old':192000,'coverage':192000})
    ap=out/'independent_verification.json';audited=ap.exists() and read_json(ap)['status']=='PASS'
    n=metrics['expanded_diffusion'];d=n['groups']['DEV'];v=n['groups']['NEW_VAL']
    stable=bool(status=='FINISHED' and trained and audited and n['completed']==180 and n['candidate_errors']==0
        and d['completed']==60 and v['completed']==120 and d['execution_verified']==12 and v['execution_verified']==24
        and all(x>=18 for x in d['per_seed_eligible'].values()) and all(x>=36 for x in v['per_seed_eligible'].values()))
    quality_pass=bool(stable and all(quality[m].get('pre_registered_quality_pass',False)
        and v['eligible']>=metrics[m]['groups']['NEW_VAL']['eligible']
        and v['execution_verified']>=metrics[m]['groups']['NEW_VAL']['execution_verified']
        for m in ('structured','learned_mean')))
    ledger=read_json(out/'ledger.json') if (out/'ledger.json').exists() else {}
    errors={k:r for k,r in ledger.get('completed',{}).items() if r['status'] in ('ERROR','BLOCKED','INTERRUPTED')}
    result={'status':status,'actual_training_passed':trained,'independent_audit_passed':audited,
        'origin_training_closure_passed':stable,'pre_registered_quality_claim_passed':quality_pass,
        'methods':metrics,'task_metrics':tasks,'NEW_VAL_paired_quality':quality,
        'errors':errors,'unresolved_calls':ledger.get('unresolved_call_accounting',[]),
        'relative_efficiency_required':False,'TEST_executed':False,'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# 新初态覆盖的实际训练与smoke分析','',
        f'状态：{status}；12688 TRAIN实际训练：{trained}；原始审计：{audited}。',
        f'新模型完整方法稳定性：{stable}；新验证目标质量主张：{quality_pass}。','',
        '## 实际训练','',
        f'更新 {tr.get("updates",0)}/6000；逐行消费 {len(tr.get("train_draw_counts",{}))}/12688；分组消费 {tr.get("group_draw_counts",{})}。',
        '544条新增覆盖按producer_run_id识别，每批32旧+32覆盖；原VAL48仅诊断，NEW_VAL8不参加训练、编码器或检查点选择。',
        '温启动12144模型，冻结原10k编码器，v预测+.05局部34方向耦合位姿变化损失；不是完整非线性目标零损失或数据量单因素实验。',
        '条件均值在全部12688 TRAIN以同样50/50组权重拟合，lambda10、时间基不变。','',
        '## 原DEV回归与新验证分别报告','',
        '| 方法 | DEV合格/60 | DEV执行重放/12 | NEW_VAL合格/120 | NEW_VAL执行重放/24 |',
        '|---|---:|---:|---:|---:|']
    for m,r in metrics.items():
        a,b=r['groups']['DEV'],r['groups']['NEW_VAL']
        lines.append(f'| {m} | {a["eligible"]} | {a["execution_verified"]} | {b["eligible"]} | {b["execution_verified"]} |')
    lines+=['','每任务3种子×5候选，预选一条真实67通道力矩执行及独立重放，失败不替补。稳定门每种子DEV≥18/20、新验证≥36/40、36/36选定执行重放、零候选数值异常。',
        '完整方法包括共同目标保持碰撞引导，不等于裸网络输出全部合格；引导位于最后一个干净去噪步，没有后续噪声更新。','',
        '## 提案与物理引导贡献','',
        '| 方法 | 共同预处理后pose-pass/180 | raw物理MODEL门/12 | raw指定目标合格/12 |',
        '|---|---:|---:|---:|']
    for m,r in metrics.items():
        raw=r['proposal_diagnostics']['raw']
        lines.append(f'| {m} | {r["pre_pose_pass"]}/{r["pre_pose_diagnostic_count"]} | {raw["physical_MODEL_gates_pass"]} | {raw["eligible_with_frozen_goal"]} |')
    lines+=['','pose-pass限位姿诊断，不等同MODEL_VALID。raw完整独立诊断预定第一种子候选0，比较结束后运行，不能影响选择。',
        '新验证质量按8个任务配对bootstrap；每任务15候选先取中位数，不把120候选当独立任务。对强样条和更新条件均值两者均需比值中位数≤.85、几何平均95%上界<1，且稳定、合格/执行不退化。','']
    for m,r in quality.items():lines.append(f'- 对照 `{m}`：`{r}`')
    lines+=['','## 边界与后续','',
        '软件、合格数据、实际训练、完整方法稳定性、神经收益分别判断。若主张未通过仍保留所有候选和负结果，不改任务或物理门。',
        '查询/时间仅辅助记录和预算，不设相对效率胜出门，不因此阻止有界合格扩库。没有NEW_VAL调参/检查点选择；如后续用于调参，承认其DEV角色。',
        '旧数据阶段ready=false/39ERROR及历史未知调用上界原样保留；新合并资格不追溯改旧门，未执行数据不升级。',
        '保留仅不穿透、17→67、完整初态、自由漂浮、全机几何、真实力矩和重放；未运行最终TEST、strict_multimodal或自动十万规模。',
        '若要声称数据本身因果收益，还需同更新次数旧数据续训对照。本阶段不作该主张。','',
        '```powershell','.\\ff_prior_origin_train\\job.ps1 status','.\\ff_prior_origin_train\\job.ps1 report','.\\ff_prior_origin_train\\job.ps1 resume','```']
    content='\n'.join(lines)+'\n'
    (out/'REPORT.md').write_text(content,encoding='utf-8')
    (out/'ANALYSIS.md').write_text(content,encoding='utf-8')
    return result
