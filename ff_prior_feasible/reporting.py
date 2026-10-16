"""Evidence-only reporting; a completed experiment is not the research objective."""
from collections import Counter
from pathlib import Path
from ff_prior.storage import read_json,atomic_json


def report(out,ledger,status):
    from .experiment import METHODS
    out=Path(out); targets=read_json(out/'frozen_tasks.json')['evaluation_entries']; metrics={}
    for method in METHODS:
        rows=[read_json(p) for p in (out/'candidates').glob(f'*/{method}/*.json')]
        executions=[read_json(p) for p in (out/'executions').glob(f'*/{method}/outcome.json')]
        reasons=Counter()
        for r in rows:
            if r['eligible']: continue
            if r['status']=='ERROR': reasons[r.get('reason','ERROR')]+=1
            else:
                for k,v in r['validation'].get('gates',{}).items():
                    if not v: reasons[k]+=1
                if not r.get('frozen_goal_pass'): reasons['world_goal']+=1
        metrics[method]={'completed':len(rows),'MODEL_VALID':sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rows),
            'eligible':sum(r['eligible'] for r in rows),'executed_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
            'coupled_queries':sum(r['model_calls']['coupled'] for r in rows),'distance_queries':sum(r['model_calls']['distance'] for r in rows),
            'integration_intervals':sum(r.get('integration_intervals') or 0 for r in rows),
            'network_calls':sum(r.get('actual_network_calls',0) for r in rows),
            'incomplete_interval_accounting':any(r.get('integration_intervals') is None for r in rows),
            'candidate_wall_s':sum(r['wall_s'] for r in rows),'failure_reasons':dict(reasons),
            'by_task':{e['sample_id']:sum(r['eligible'] for r in rows if r['sample_id']==e['sample_id']) for e in targets},
            'by_group':{g:sum(r['eligible'] for r in rows if r['group']==g) for g in ('old_complex','new_origin')}}
    new,base=metrics[METHODS[2]],metrics[METHODS[0]]
    stable=len(targets)==10 and new['by_group']['old_complex']>=27 and new['by_group']['new_origin']>=18 and all(n>=4 for n in new['by_task'].values()) and new['executed_verified']==10
    gain=stable and not any(m['incomplete_interval_accounting'] for m in metrics.values()) and new['eligible']>=base['eligible'] and new['executed_verified']>=base['executed_verified'] and (
        new['executed_verified']>base['executed_verified'] or (new['coupled_queries']<=.9*base['coupled_queries'] and new['candidate_wall_s']<=.9*base['candidate_wall_s']))
    summary={'status':status,'methods':metrics,'stability_gate_passed':bool(stable),
        'exploratory_advantage_gate_passed':bool(gain),'full_goal_achieved':False,'expansion':'NOT_RUN_THIS_ISOLATION_ROUND',
        'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s'],
        'software':'PASS','model_prior':'SEE_PER_CANDIDATE','physical':'SEE_SELECTED_EXECUTIONS',
        'scope':'same exploratory VAL tasks and checkpoints; original DEV unchanged, no TEST'}
    atomic_json(out/'summary.json',summary)
    lines=['# 保持世界目标的碰撞修正与自适应验证','',f'运行状态：{status}。完整优势目标仍未宣称完成。',
        '', '| 方法 | MODEL_VALID / 50 | 指定目标合格 / 50 | 力矩执行与重放 / 10 | 耦合查询 | 积分区间 | 候选秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,m in metrics.items():
        lines.append(f'| {method} | {m["MODEL_VALID"]} | {m["eligible"]} | {m["executed_verified"]} | {m["coupled_queries"]} | {m["integration_intervals"]} | {m["candidate_wall_s"]:.3f} |')
    lines+=['',f'候选稳定门槛：{stable}；探索性优势门槛：{gain}。',
        '', '只改变修正策略与验证分辨率；冻结任务、种子、旧/新两份权重。全方法共同上限：5候选、113耦合查询、15872积分区间、15秒。',
        '', '接受门槛仍为仅不穿透、位置5mm/姿态1°、几何相邻变化≤2mm及原动力学约束。1024网格需要额外积分与距离检查，全部计费。',
        '', '未训练新模型，未扩库，未执行原DEV、pilot、study或TEST。所有旧结果保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return summary
