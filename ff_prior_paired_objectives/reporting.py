from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json
from ff_prior_coverage_train.storage import atomic_json
from ff_prior.contracts import all_pass,MODEL_GATES
from ff_prior_generalization.reporting import pre_norm
from ff_prior_origin_learn_bounded.reporting import paired_quality
from .protocol import METHODS,PROTOCOL as P

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
    selection=read_json(out/'TRAIN_selection.json') if (out/'TRAIN_selection.json').exists() else {}
    single=read_json(out/'single_step_summary.json') if (out/'single_step_summary.json').exists() else {}
    training={}
    for arm in P['arms']:
        path=out/'learning'/arm/'training.json'
        if path.exists():
            t=read_json(path);training[arm]={k:t[k] for k in ('status','updates','world_gradient_updates','all_TRAIN_rows_seen','world_alignment','checkpoint_sha256')}
    audit=read_json(out/'independent_verification.json') if (out/'independent_verification.json').exists() else {}
    result={'status':status,'single_step_status':single.get('status','PENDING'),'training':training,
        'TRAIN_selection':selection,'methods':metrics,'task_metrics':tasks,'reused_DEV8_paired_quality':quality,
        'raw_audit':audit,'TEST_executed':False,'TEST_decision':'BLOCKED_BY_CURRENT_USER_SCOPE',
        'broad1024_started':False,'pilot_started':False,'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# TRAIN单步诊断与四组配对训练','',f'状态：{status}；单步诊断：{result["single_step_status"]}。',
        '数据复用上一轮13838 TRAIN/48 VAL（包含64条固定目标修正）；本轮新增数据和TRAIN执行次数均为0。',
        '四组共用13774行阶段的冻结初始权重、codec、6000次更新、批次和噪声序列；每50步C/D加入一次真实102维当前预测终点梯度。',
        'A仅v；B增加0.05局部位姿；C增加0.1真实终点；D两者兼用。该实验隔离损失项，仍仅一个训练种子。','',
        f'TRAIN选择：{selection}。选择在接触本轮DEV结果前完成；无合格C/D则DEV依赖BLOCKED。','',
        '|组|更新|真实终点更新|固定TRAIN探针中位损失比|','|---|---:|---:|---:|']
    for arm,t in training.items():lines.append(f'|{arm}|{t["updates"]}|{t["world_gradient_updates"]}|{t["world_alignment"]["median_paired_loss_ratio"]:.6f}|')
    lines+=['','## 冻结DEV复核','',
        '仅TRAIN门通过后运行：原DEV4和已反复使用的开发8任务，3采样种子，4方法，每任务5候选；共720候选、至多144执行重放。8任务不代表未见泛化。',
        '|方法|候选完成|合格|执行重放|修正前位姿通过|裸提案目标通过|','|---|---:|---:|---:|---:|---:|']
    for m,r in metrics.items():lines.append(f'|{m}|{r["completed"]}|{r["eligible"]}|{r["execution_verified"]}|{r["pre_pose_pass"]}|{r["proposal_diagnostics"]["raw"]["eligible_with_frozen_goal"]}|')
    lines+=['',f'原始审计：{audit}。',f'任务配对质量结果：{quality}。',
        '模型层合格、实际力矩执行、独立重放与Diffusion相对强样条/条件均值的收益分别判断。完整引导成功不能替代裸网络的目标命中证据。',
        '本轮禁止1024槽扩库、pilot和TEST；后续扩库及最终TEST须另有范围授权和冻结协议。长期目标尚未完成。',
        '进程结束后的独立结果审查和中文分析报告仍需完成；本文件是后台原始汇总。','',
        '```powershell',r'.\ff_prior_paired_objectives\job.ps1 status',r'.\ff_prior_paired_objectives\job.ps1 report',r'.\ff_prior_paired_objectives\job.ps1 resume','```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
