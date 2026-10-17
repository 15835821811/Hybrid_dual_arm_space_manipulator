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
    trpath=out/'learning/goal_loss/training.json';tr=read_json(trpath) if trpath.exists() else {}
    data=read_json(out/'data_summary.json') if (out/'data_summary.json').exists() else {}
    trained=bool(tr.get('status')=='PASS' and tr.get('updates')==6000 and tr.get('world_gradient_updates')==120
                 and tr.get('all_TRAIN_rows_seen') and len(tr.get('train_draw_counts',{}))==data.get('total_TRAIN'))
    ap=out/'independent_verification.json';audited=ap.exists() and read_json(ap)['status']=='PASS'
    n=metrics['expanded_diffusion'];d=n['groups']['DEV'];v=n['groups']['NEW_VAL']
    stable=bool(status=='FINISHED' and trained and audited and data.get('status')=='PASS'
        and n['completed']==180 and n['candidate_errors']==0 and d['execution_verified']==12 and v['execution_verified']==24
        and all(x>=18 for x in d['per_seed_eligible'].values()) and all(x>=36 for x in v['per_seed_eligible'].values()))
    quality_pass=bool(stable and all(quality[m].get('pre_registered_quality_pass',False)
        and v['eligible']>=metrics[m]['groups']['NEW_VAL']['eligible']
        and v['execution_verified']>=metrics[m]['groups']['NEW_VAL']['execution_verified'] for m in ('structured','learned_mean')))
    quality_old=bool(stable and quality['frozen_diffusion'].get('pre_registered_quality_pass',False)
        and v['eligible']>=metrics['frozen_diffusion']['groups']['NEW_VAL']['eligible'])
    alignment=tr.get('world_alignment',{});raw=n['proposal_diagnostics']['raw']
    test_gates={'complete_stability_and_raw_audit':stable,
        'TRAIN_true_endpoint_median_ratio_le_08':alignment.get('pre_registered_improvement',False),
        'reused_DEV8_paired_quality_vs_frozen13774':quality_old,
        'fixed_raw_goal_at_least4of12':raw['eligible_with_frozen_goal']>=4,
        'fixed_raw_physics12of12':raw['physical_MODEL_gates_pass']==12}
    ledger=read_json(out/'ledger.json') if (out/'ledger.json').exists() else {}
    errors={k:r for k,r in ledger.get('completed',{}).items() if r['status'] in ('ERROR','BLOCKED','INTERRUPTED')}
    result={'status':status,'actual_training_passed':trained,'independent_audit_passed':audited,
        'world_endpoint_closure_passed':stable,'origin_training_closure_passed':stable,
        'pre_registered_quality_claim_passed':quality_pass,'quality_vs_frozen13774_passed':quality_old,
        'TRAIN_endpoint_alignment':alignment,'methods':metrics,'task_metrics':tasks,'reused_DEV8_paired_quality':quality,
        'data':data,'errors':errors,'unresolved_calls':ledger.get('unresolved_call_accounting',[]),
        'conditional_TEST_gates':test_gates,'eligible_for_TEST_protocol_review':all(test_gates.values()),
        'TEST_decision':'REVIEW_AND_FREEZE_UNTOUCHED_TEST_PROTOCOL' if all(test_gates.values()) else 'BLOCKED_BY_SMOKE_GATES',
        'relative_efficiency_required':False,'TEST_executed':False,'full_goal_achieved':False,'protocol':P}
    atomic_json(out/'summary.json',result)
    lines=['# 真实世界终点监督与固定TRAIN目标修正闭环','',
        f'状态：{status}；实际训练：{trained}；独立原始审计：{audited}；完整稳定门：{stable}。',
        f'新固定TRAIN修正 {data.get("new_TRAIN",0)}/64上限，累计 {data.get("total_TRAIN",13774)} TRAIN；原VAL48保持。',
        f'新TRAIN预选执行重放 {data.get("native_pass",0)}/16；真实梯度更新 {tr.get("world_gradient_updates",0)}/120；常规训练 {tr.get("updates",0)}/6000。','',
        '## 训练机制','',
        '对当前预测的原始轨迹执行完整自由漂浮耦合传播，以双臂位置/5mm与姿态/1°的平方误差之和除4作为真实计算的终点损失。每50步加入一次102维中央差分梯度，梯度在当前预测重新计算；保留半步、独立方向和256/512传播收敛检查。',
        '该前向是完整耦合模型计算，属于训练监督；不等同真实力矩执行，不替代全机几何或数据准入。正式独立验证和真实MuJoCo67力矩执行/完整初态重放仍按原门运行。',
        '继承已审计的13774行公共初始器和完整初态证明，逐一核对源数组；只为新修正行重建初始器。缓存复用不产生新物理调用，也不改变机器人或样本等级。',
        f'TRAIN固定32噪声探针结果：{alignment}。这是训练分布诊断，不能作为DEV或未见泛化证据。','',
        '## 同一冻结任务和候选预算','',
        '原DEV4及上一轮已使用的8个开发验证任务，原三种子、四方法、每任务5候选；8任务现在明确作REUSED_DEV，不能再称独立新验证。新旧网络同16步DDIM，候选15秒/113耦合/15872几何间隔上限，共同物理引导，不增加新网络独享采样精修。','',
        '| 方法 | 合格/180 | 执行重放/36 | 修正前pose-pass/180 | raw指定目标/12 |',
        '|---|---:|---:|---:|---:|']
    for m,r in metrics.items():
        lines.append(f'| {m} | {r["eligible"]} | {r["execution_verified"]} | {r["pre_pose_pass"]} | {r["proposal_diagnostics"]["raw"]["eligible_with_frozen_goal"]} |')
    lines+=['','共同预处理的pose-pass、裸提案完整门与最终引导后的合格各自报告。固定raw候选0诊断不用于选择。','',
        f'相对冻结13774模型的任务配对质量门：{quality_old}；相对强样条和更新均值的联合优势门：{quality_pass}。',
        '8开发任务先按每任务15候选汇总，再做任务配对bootstrap；一个训练种子、三个采样种子；不主张数据量或目标损失单一因果。']
    for m,r in quality.items():lines.append(f'- {m}: {r}')
    lines+=['','## 有条件的后续决策','',f'最终TEST前置门：{test_gates}。',
        f'决策：{result["TEST_decision"]}。本后台程序始终不读取或执行最终TEST。若门全过，先独立审查、判断扩库必要性并冻结一次未触碰TEST协议，才可按用户条件授权进入TEST；否则保留BLOCKED原因。',
        '1024槽前向扩库保留备用，本轮未启动。旧结果、旧失败、用户修改、main与旧分支保留；无强推、大数据上传或strict_multimodal。',
        '所有软件、数据、学习、完整执行和神经收益分开判断；查询/时间相对优势不是扩库或TEST门。','',
        '```powershell',r'.\ff_prior_world_endpoint\job.ps1 status',r'.\ff_prior_world_endpoint\job.ps1 report',r'.\ff_prior_world_endpoint\job.ps1 resume','```']
    text='\n'.join(lines)+'\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8');(out/'ANALYSIS.md').write_text(text,encoding='utf-8')
    return result
