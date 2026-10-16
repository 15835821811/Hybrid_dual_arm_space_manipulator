"""Task-level paired evidence; saturated success never implies neural superiority."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import METHODS, PROTOCOL


def pre_norm(r):
    e=np.asarray(r['sampling']['guidance'][-1]['initial_goal_error'])
    return float(np.linalg.norm(e.reshape(4,3),axis=1).max())


def paired_quality(task_metrics):
    result={}
    for method in ('structured','learned_mean','frozen_diffusion'):
        pairs=[(v['expanded_diffusion']['pre_median_norm'],v[method]['pre_median_norm'])
               for v in task_metrics.values()
               if all(v[m]['pre_median_norm'] is not None and v[m]['completed']==15 for m in (method,'expanded_diffusion'))]
        if len(pairs)!=12:
            result[method]={'status':'INCOMPLETE','paired_tasks':len(pairs)};continue
        a=np.asarray(pairs);ratios=np.maximum(a[:,0],1e-12)/np.maximum(a[:,1],1e-12)
        rng=np.random.default_rng(PROTOCOL['bootstrap_seed'])
        bootstrap=np.exp(np.log(ratios)[rng.integers(0,12,(10000,12))].mean(axis=1))
        lo,hi=np.quantile(bootstrap,[.025,.975])
        result[method]={'status':'DONE','paired_tasks':12,'median_task_ratio':float(np.median(ratios)),
                        'geometric_mean_task_ratio':float(np.exp(np.log(ratios).mean())),
                        'bootstrap95_geometric_ratio':[float(lo),float(hi)],
                        'new_lower_tasks':int(sum(ratios<1)),'new_higher_tasks':int(sum(ratios>1)),
                        'pre_registered_quality_pass':bool(np.median(ratios)<=.85 and hi<1.)}
    return result


def report(out,status):
    out=Path(out);fp=out/'frozen_tasks.json';frozen=read_json(fp)['entries'] if fp.exists() else []
    tasks=[e for e in frozen if e['status']=='FROZEN'];metrics={};task_metrics={e['sample_id']:{} for e in tasks}
    for method in METHODS:
        records=[read_json(p) for p in (out/'candidates').glob(f'*/*/{method}/*.json')]
        outcomes=[read_json(p) for p in (out/'executions').glob(f'*/*/{method}/outcome.json')]
        diagnostic={stage:[read_json(p) for p in (out/'proposal_diagnostics').glob(f'*/{method}/{stage}.json')]
                    for stage in ('raw','smoothed_clipped')}
        norms=[pre_norm(r) for r in records if 'sampling' in r]
        for e in tasks:
            rs=[r for r in records if r['sample_id']==e['sample_id']]
            ps=[pre_norm(r) for r in rs if 'sampling' in r]
            task_metrics[e['sample_id']][method]={'completed':len(rs),'eligible':sum(r['eligible'] for r in rs),
                      'pre_median_norm':float(np.median(ps)) if ps else None,
                      'pre_pose_pass':sum(v<=1 for v in ps)}
        metrics[method]={'completed':len(records),'eligible':sum(r['eligible'] for r in records),
              'pre_pose_pass':sum(v<=1 for v in norms),'pre_pose_diagnostic_count':len(norms),
              'pre_median_norm':float(np.median(norms)) if norms else None,
              'selected_outcomes':len(outcomes),'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),
              'per_seed_eligible':{str(s):sum(r['eligible'] and r['seed_group']==s for r in records) for s in range(3)},
              'candidate_errors':sum(r['status']!='DONE' for r in records),
              'raw_independent_validation':{stage:{'completed':len(rs),'eligible':sum(r['eligible'] for r in rs),
                                                  'errors':sum(r['status']!='DONE' for r in rs)} for stage,rs in diagnostic.items()},
              'coupled_queries':sum(r['model_calls'].get('coupled',0) for r in records),
              'network_calls':sum(r['actual_network_calls'] for r in records),
              'wall_s':sum(r['wall_s'] for r in records),
              'rejection_gates':dict(Counter(k for r in records for k,v in r.get('validation',{}).get('gates',{}).items() if not v))}
    teachers=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    n=metrics['expanded_diffusion'];quality=paired_quality(task_metrics)
    audit_path=out/'independent_verification.json';audited=audit_path.exists() and read_json(audit_path)['status']=='PASS'
    teacher_pass=sum(r['status']=='EXECUTION_VERIFIED' for r in teachers)
    stable=bool(status=='FINISHED' and audited and len(tasks)==12 and teacher_pass==12 and n['completed']==180
                and all(v>=54 for v in n['per_seed_eligible'].values()) and n['execution_verified']==36 and n['candidate_errors']==0)
    quality_pass=bool(stable and all(quality[m].get('pre_registered_quality_pass',False) for m in ('structured','learned_mean'))
                      and all(n['eligible']>=metrics[m]['eligible'] and n['execution_verified']>=metrics[m]['execution_verified']
                              for m in ('structured','learned_mean')))
    result={'status':status,'methods':metrics,'task_metrics':task_metrics,'frozen_tasks':len(tasks),'task_slots':12,
            'teacher_execution_verified':teacher_pass,'independent_audit_passed':audited,
            'new_validation_stability_passed':stable,'pre_registered_quality_claim_passed':quality_pass,
            'paired_quality':quality,'training_updates':0,'TEST_executed':False,'full_goal_achieved':False,
            'relative_efficiency_required':False,'protocol':PROTOCOL}
    atomic_json(out/'summary.json',result)
    lines=['# 独立新世界目标验证分析','',f'状态：{status}；冻结任务：{len(tasks)}/12；教师执行重放：{teacher_pass}/12；原始审计：{audited}。',
           f'新Diffusion稳定性：{stable}；预注册目标质量主张：{quality_pass}。','',
           '| 方法 | 合格/180 | 修正前pose-pass | 选定执行重放/36 |','|---|---:|---:|---:|']
    for m,v in metrics.items():lines.append(f'| {m} | {v["eligible"]} | {v["pre_pose_pass"]}/{v["pre_pose_diagnostic_count"]} | {v["execution_verified"]} |')
    lines+=['','分母180为12任务×3种子×5提案；任务生成不足或执行依赖受阻仍保留固定槽位，不改写为较小满分。',
            'pose-pass是共同平滑及限位裁剪后、迭代修正前的位姿阈值诊断，不等同MODEL_VALID。独立新任务的教师与模型条件隔离，教师解不提供给规划器。',
            '原冻结DEV与48 VAL保留，本组是新的开发验证，不是最终TEST；若后续调参使用，本组必须降为DEV。无训练更新或检查点选择。','',
            '## 引导前完整独立验证（固定第一种子、候选0，事后诊断，不参与选择）','',
            '| 方法 | 原始提案MODEL_VALID/12 | 平滑限位后MODEL_VALID/12 |','|---|---:|---:|']
    for m,v in metrics.items():
        d=v['raw_independent_validation'];lines.append(f'| {m} | {d["raw"]["eligible"]} | {d["smoothed_clipped"]["eligible"]} |')
    lines+=['','## 按任务配对的修正前目标残差','',
            '每任务先取15个提案的误差中位数，再配对新模型/对照；以12个任务为bootstrap单位，不能把180个提案当作独立任务。',
            '预注册质量门槛：对样条及条件均值两者，任务比值中位数≤0.85、几何平均比值bootstrap95%上界<1，且新方法通过稳定性并无候选或执行成功率退化。','']
    for m,v in quality.items():lines.append(f'- 对照`{m}`：`{v}`')
    lines+=['','查询和耗时仅辅助核算，不要求相对胜出。真实失败、生成拒绝、数值异常与未完成诊断保留；不改变仅不穿透、世界目标、17→67映射、完整初态、自由漂浮、全机几何、真实力矩和独立重放合同。',
            '本报告不自动授权十万规模或最终TEST，也不将未执行数据升级为执行通过。','',
            '## 终端','', '```powershell','.\\ff_prior_generalization\\job.ps1 status','.\\ff_prior_generalization\\job.ps1 report','```']
    text='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text,encoding='utf-8')
    (out/'ANALYSIS.md').write_text(text,encoding='utf-8')
    return result
