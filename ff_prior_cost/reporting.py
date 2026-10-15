"""Reports measured success and cost separately; negative results remain negative."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json, atomic_json


def metrics(out,split,method,n):
    rows=[read_json(p) for p in sorted((Path(out)/'candidates'/split/method).glob('*.json'))]
    reports=[read_json(p) for p in sorted((Path(out)/'executions'/split/method).glob('*/report.json'))]
    eligible=[r for r in rows if r.get('eligible')]; diversity=[]
    for task in {r['task_hash'] for r in rows}:
        z=[np.array(r['Z']) for r in eligible if r['task_hash']==task]
        diversity += [float(np.sqrt(np.mean((a-b)**2))) for i,a in enumerate(z) for b in z[i+1:]]
    errors=[max(r['validation']['frozen_goal_position_error_m'])*1000 for r in rows
        if 'frozen_goal_position_error_m' in r.get('validation',{})]
    return {'candidate_denominator':5*n,'task_denominator':n,'candidate_records':len(rows),
        'model_valid':sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rows),
        'goal_valid':sum(r.get('frozen_goal_pass',False) for r in rows),'eligible':len(eligible),
        'execution_attempted':len(reports),'execution_verified':sum(r['level']=='EXECUTION_VERIFIED' for r in reports),
        'replay_pass':sum(r['replay']['passed'] for r in reports),
        'coupled_queries':sum(r.get('model_calls',{}).get('coupled',0) for r in rows),
        'repair_queries':sum(r.get('repair_coupled_calls',0) for r in rows),
        'validation_queries':sum(r.get('validation_coupled_calls',0) for r in rows),
        'max_candidate_queries':max([r.get('model_calls',{}).get('coupled',0) for r in rows],default=0),
        'candidate_wall_s':sum(r.get('wall_s',0.) for r in rows),
        'network_calls':sum(r.get('sampling',{}).get('actual_network_calls',0) for r in rows),
        'candidate_errors':sum(r['status']!='DONE' for r in rows),
        'goal_position_mm_range':[min(errors),max(errors)] if errors else None,
        'median_pair_control_rms_rad':float(np.median(diversity)) if diversity else None,
        'failed_gates':dict(Counter(k for r in rows for k,v in r.get('validation',{}).get('gates',{}).items() if not v))}


def better(a,b):
    return (a['candidate_records']==a['candidate_denominator'] and not a['candidate_errors']
        and not b['candidate_errors'] and a['eligible']>=b['eligible']
        and a['execution_verified']==a['task_denominator'] and a['execution_verified']>=b['execution_verified']
        and a['coupled_queries']<=.8*b['coupled_queries'] and a['candidate_wall_s']<=b['candidate_wall_s'])


def choose_on_val(out):
    from .experiment import METHODS
    values={m:metrics(out,'VAL',m,6) for m in METHODS}
    baseline=values[METHODS[0]]
    options=[m for m in METHODS[1:] if better(values[m],baseline)]
    return min(options,key=lambda m:values[m]['coupled_queries']) if options else METHODS[0]


def write_report(out,ledger,status):
    from .experiment import METHODS
    out=Path(out); groups=read_json(out/'frozen_tasks.json')
    sums={s:{m:metrics(out,s,m,len(t)) for m in METHODS} for s,t in groups.items()}
    complete=status=='FINISHED'
    learned=complete and all(better(sums[s][METHODS[3]],sums[s][METHODS[2]]) for s in sums)
    necessity=complete and any(all(better(sums[s][m],sums[s][METHODS[0]]) for s in sums) for m in METHODS[1:])
    teachers=[read_json(p) for p in (out/'teachers').glob('*/*.json')]
    train=read_json(out/'learning/training.json') if (out/'learning/training.json').exists() else {}
    diag=[]
    for p in (out/'diagnostics').glob('*/*.json'):
        r=read_json(p)
        if r['status']!='MEASURED': continue
        values={(v['iteration'],v['stage']):float(np.linalg.norm(v['normalized_pose_error'])) for v in r['probes']}
        diag.append({'split':r['split'],'task_hash':r['task_hash'],'initializer':values[(-1,'public_initializer')],
            'first_network':values[(0,'network_clean')], 'repaired_14':values[(14,'after_repair')],
            'network_15':values[(15,'network_clean')],'prior_15':values[(15,'after_temporal_prior')],
            'smooth_15':values[(15,'after_smoothing')],'repaired_15':values[(15,'after_repair')]})
    summary={'status':status,'methods':sums,'training_status':train.get('status','NOT_RUN'),
        'teacher_eligible':{s:sum(r.get('eligible',False) for r in teachers if r['split']==s) for s in ('TRAIN','VAL')},
        'learning_cost_gain':'SUPPORTED_WITHIN_THIS_SMOKE' if learned else 'NOT_ESTABLISHED',
        'diffusion_necessity':'SUPPORTED_WITHIN_THIS_SMOKE' if necessity else 'NOT_ESTABLISHED',
        'selected_method_on_VAL':choose_on_val(out),'diagnosis':diag,'attempts':ledger.data['attempts'],
        'calls':ledger.data['calls'],'wall_s':ledger.data['wall_s'],'cpu_s':ledger.data['cpu_s'],
        'pilot':'NOT_RUN','TEST':'NOT_RUN','hardware':'NOT_RUN','generalization':'NOT_ESTABLISHED',
        'unresolved_work':ledger.data.get('unresolved_call_accounting',[])}
    atomic_json(out/'summary.json',summary)
    lines=['# 同预算残差学习与采样 smoke 报告','',f'状态：{status}。',
        f"学习修正成本收益：{summary['learning_cost_gain']}；相对样条的Diffusion必要性：{summary['diffusion_necessity']}。",'',
        '| 分组 | 方法 | 模型有效 | 世界目标合格 | 同时合格 | 执行+重放 | 总耦合查询 | 候选秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s,methods in sums.items():
        for m,r in methods.items():
            n=r['candidate_denominator']; t=r['task_denominator']
            lines.append(f"| {s} | {m} | {r['model_valid']}/{n} | {r['goal_valid']}/{n} | {r['eligible']}/{n} | {r['execution_verified']}/{t} | {r['coupled_queries']} | {r['candidate_wall_s']:.3f} |")
    lines += ['',f"合格教师标签：{summary['teacher_eligible']}。TRAIN仅用于梯度，VAL仅用于固定损失检查及权重选择；标签仅MODEL_VALID，未批量升级执行等级。",
        f"微调状态：{summary['training_status']}。选择更新：{train.get('selected_update')}。小样本拟合：{train.get('sanity',{})}。",'',
        f"DEV开始前按VAL冻结推荐：{summary['selected_method_on_VAL']}。四种对照及种子预先固定，DEV结果不用于再次调整。",'',
        '## 误差来源','']
    for s in ('TRAIN','VAL'):
        a=[r for r in diag if r['split']==s]
        if a:
            med={k:float(np.median([r[k] for r in a])) for k in ('initializer','first_network','repaired_14','network_15','prior_15','smooth_15','repaired_15')}
            lines.append(f'- {s}归一化12维世界位姿误差范数中位数：{med}。')
    lines += ['', '## 判据与边界','',
        '所有方法5候选/任务、最多110次修正耦合查询+3次完整模型验证查询。两步方法为末步保留36次查询；没有达到预算上限的候选不补用剩余额度。',
        '共享17方向修正、全机几何及原物理门槛。仅不穿透>=0m；5mm/1度；2mm几何细化；17→67映射；完整初态、自由基座、真实67力矩mj_step、独立ctrl重放。',
        '收益须VAL与DEV都不降低合格数/执行成功，耦合查询至少少20%且候选wall不更高。额外训练成本另列，未算成免费收益。',
        '每任务每方法预选一条执行。候选模型通过不等于全部真实执行。背景诊断探测独立计账，不计入正式候选，也不隐瞒为零。',
        'world目标误差实际前向诊断；训练是关节残差监督，并未声称实现可微全动力学损失。控制点多样性不等于避障拓扑或strict_multimodal。',
        '本DEV已多轮开发，只支持smoke结论；不支持独立泛化、硬件或微重力是唯一失败原因。',
        f"阶段wall：{summary['wall_s']}；尝试数：{summary['attempts']}。所有失败保留，依赖失败为BLOCKED，未运行pilot/TEST。"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
