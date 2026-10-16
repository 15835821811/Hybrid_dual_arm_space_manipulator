"""Automatic terminal report, retaining empty and failed task slots."""
from collections import Counter
from ff_prior.storage import read_json, atomic_json


def analyze(out):
    from .experiment import METHODS,PROTOCOL
    frozen=read_json(out/'frozen_complex_tasks.json')['entries'] if (out/'frozen_complex_tasks.json').exists() else []
    rows=[]; aggregates={}; wins=[]; losses=[]
    for e in frozen:
        sid=e.get('sample_id'); item={'sample_id':sid,'split':e['split'],'kind':e['kind'],
            'generation':e['status'],'generation_reason':e.get('reason'),'methods':{}}
        wp=out/'witness_executions'/str(sid)/'outcome.json'
        item['witness']=read_json(wp)['status'] if wp.exists() else 'BLOCKED'
        for method in METHODS:
            candidates=[read_json(p) for p in sorted((out/'candidates'/str(sid)/method).glob('*.json'))]
            ep=out/'executions'/str(sid)/method/'outcome.json'
            status=read_json(ep)['status'] if ep.exists() else 'BLOCKED'
            failures=Counter()
            for c in candidates:
                if c.get('eligible'): continue
                if c['status']!='DONE': failures[c.get('reason','ERROR')]+=1
                elif c.get('validation',{}).get('validation_level')!='MODEL_VALID':
                    for gate,passed in c.get('validation',{}).get('gates',{}).items():
                        if not passed: failures['gate:'+gate]+=1
                elif not c.get('frozen_goal_pass'): failures['frozen_goal']+=1
            metric={'candidate_records':len(candidates),'candidate_denominator':5,
                'model_valid':sum(c.get('validation',{}).get('validation_level')=='MODEL_VALID' for c in candidates),
                'eligible':sum(bool(c.get('eligible')) for c in candidates),'execution':status,
                'candidate_errors':sum(c['status']!='DONE' for c in candidates),
                'coupled_queries':sum(c.get('model_calls',{}).get('coupled',0) for c in candidates),
                'distance_queries':sum(c.get('model_calls',{}).get('distance',0) for c in candidates),
                'network_calls':sum(c.get('sampling',{}).get('actual_network_calls',0) for c in candidates),
                'wall_s':sum(c.get('wall_s',0) for c in candidates),
                'cpu_s':sum(c.get('cpu_s',0) for c in candidates),'failures':dict(failures)}
            item['methods'][method]=metric
        valid=item['witness']=='EXECUTION_VERIFIED'
        diff=item['methods']['frozen_diffusion']['execution']=='EXECUTION_VERIFIED'
        baselines=[item['methods'][m]['execution']=='EXECUTION_VERIFIED' for m in METHODS[:2]]
        complete=all(v['candidate_records']==5 and not v['candidate_errors'] for v in item['methods'].values())
        if valid and complete and diff and not any(baselines): wins.append({'split':e['split'],'sample_id':sid,'kind':e['kind']})
        if valid and complete and not diff and any(baselines): losses.append({'split':e['split'],'sample_id':sid,'kind':e['kind']})
        item['comparison_complete']=valid and complete; rows.append(item)
    for split in ('TRAIN','VAL'):
        aggregates[split]={}
        for method in METHODS:
            subset=[r['methods'][method] for r in rows if r['split']==split]
            keys=('candidate_records','candidate_denominator','model_valid','eligible','candidate_errors',
                  'coupled_queries','distance_queries','network_calls','wall_s','cpu_s')
            aggregates[split][method]={k:sum(v[k] for v in subset) for k in keys}
            aggregates[split][method].update(task_denominator=6,execution_verified=sum(v['execution']=='EXECUTION_VERIFIED' for v in subset))
    dataset=[read_json(p) for p in (out/'records').glob('*.json')]
    return {'protocol':PROTOCOL,'tasks':rows,'aggregate':aggregates,'diffusion_unique_wins':wins,
        'diffusion_losses_to_any_spline':losses,'new_verified_records':dict(Counter(r['split'] for r in dataset)),
        'new_model_training_performed':False,'diffusion_benefit_observed_on_VAL':any(w['split']=='VAL' for w in wins),
        'generalization_benefit_established':False,'blocked_slots':sum(r['witness']!='EXECUTION_VERIFIED' for r in rows)}


def write_report(out,ledger,status):
    summary=analyze(out); summary.update(status=status,ledger_calls=ledger.data['calls'],ledger_wall_s=ledger.data['wall_s'])
    atomic_json(out/'summary.json',summary)
    lines=['# 复杂 TRAIN/VAL 小规模验证','',f'状态：{status}。仅 smoke；原冻结 DEV 与旧结果保留，未启动 pilot 或 TEST。','',
        '本轮增加复杂任务和已执行轨迹，冻结已有网络权重，不进行新训练。每方法每任务 5 个候选；每候选最多 110 次修正传播 + 3 次独立验证传播，墙钟上限 8 秒。实际开销另行报告。',
        '',f"新增已执行并独立重放合格样本：{summary['new_verified_records']}；受阻任务槽位：{summary['blocked_slots']}/12。",'',
        '| 分区 | 方法 | MODEL_VALID | 目标及模型合格 | 执行+重放 | 传播查询 | 候选秒 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split,methods in summary['aggregate'].items():
        for method,m in methods.items():
            lines.append(f"| {split} | {method} | {m['model_valid']}/30 | {m['eligible']}/30 | {m['execution_verified']}/6 | {m['coupled_queries']} | {m['wall_s']:.3f} |")
    lines+=['',f"Diffusion 独有成功（两个样条均失败）：{summary['diffusion_unique_wins']}",
        f"Diffusion 失败而至少一个样条成功：{summary['diffusion_losses_to_any_spline']}",'',
        '只有实际力矩积分与独立控制重放均通过，才计执行成功。参考见证轨迹用于证明可达性，未提供给规划器。未执行候选不升级验证等级。',
        '', '约束：17→67 映射、完整初态、自由漂浮耦合、全机几何、仅不穿透 ≥0 m、5 mm/1°目标、0.5 s保持、30 s截止及原力矩/速度/加速度/动量/重放门槛均保留。物理执行指 MuJoCo 力矩积分，不是实机。',
        '', '解释范围：两类任务是较大运动与静态绕障。共享修正器只优化末端目标，未优化碰撞代价；冻结网络不接收障碍位置。负结果不能归因于微重力本身，也不能证明带场景条件和碰撞引导的未来方法必然失败。',
        '', '本轮仅检验已有先验在新任务上的收益；新增 TRAIN 数据尚未被训练采用，VAL 仅作开发验证。原 DEV 不重跑，TEST 未读取样本。','',
        '| 任务 | 见证 | 窄样条 | 宽样条 | Diffusion |','|---|---|---|---|---|']
    for row in summary['tasks']:
        lines.append('| '+str(row['sample_id'])+' | '+row['witness']+' | '+' | '.join(v['execution'] for v in row['methods'].values())+' |')
    lines+=['','失败细分及各任务查询、候选耗时见 summary.json；完整轨迹和初态位于 tasks/，控制与重放证据位于 executions/ 和 witness_executions/。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
