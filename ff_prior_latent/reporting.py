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
    new,base=metrics[METHODS[3]],metrics[METHODS[0]]
    stable=len(targets)==10 and new['by_group']['old_complex']>=27 and new['by_group']['new_origin']>=18 and all(n>=4 for n in new['by_task'].values()) and new['executed_verified']==10
    comparisons={}
    for opponent in (METHODS[0],METHODS[2]):
        other=metrics[opponent]
        comparisons[opponent]=bool(stable and not new['incomplete_interval_accounting'] and not other['incomplete_interval_accounting']
            and new['eligible']>=other['eligible'] and new['executed_verified']>=other['executed_verified'] and (
            new['executed_verified']>other['executed_verified'] or (new['coupled_queries']<=.9*other['coupled_queries']
            and new['candidate_wall_s']<=.9*other['candidate_wall_s'])))
    gain=all(comparisons.values())
    summary={'status':status,'methods':metrics,'stability_gate_passed':bool(stable),
        'exploratory_advantage_gate_passed':bool(gain),'advantage_against_each_baseline':comparisons,
        'full_goal_achieved':False,'expansion':'NOT_RUN_THIS_ISOLATION_ROUND',
        'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s'],
        'software':'PASS','model_prior':'SEE_PER_CANDIDATE','physical':'SEE_SELECTED_EXECUTIONS',
        'scope':'same exploratory VAL tasks; one new checkpoint; original DEV unchanged, no TEST'}
    atomic_json(out/'summary.json',summary)
    lines=['# 条件均值与低维 Diffusion 对照','',f'运行状态：{status}。完整优势目标仍未宣称完成。',
        '', '| 方法 | MODEL_VALID / 50 | 指定目标合格 / 50 | 力矩执行与重放 / 10 | 耦合查询 | 积分区间 | 候选秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,m in metrics.items():
        lines.append(f'| {method} | {m["MODEL_VALID"]} | {m["eligible"]} | {m["executed_verified"]} | {m["coupled_queries"]} | {m["integration_intervals"]} | {m["candidate_wall_s"]:.3f} |')
    lines+=['',f'候选稳定门槛：{stable}；探索性优势门槛：{gain}。',
        '', '新模型固定训练1800步；保留强样条、600步冻结模型和学习均值样条消融。全方法共同上限：5候选、113耦合查询、15872积分区间、15秒。',
        '', '接受门槛仍为仅不穿透、位置5mm/姿态1°、几何相邻变化≤2mm及原动力学约束。1024网格需要额外积分与距离检查，全部计费。',
        '', '实际训练新低维模型；未扩库、未执行原DEV、pilot、study或TEST。保留旧结果。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return summary
