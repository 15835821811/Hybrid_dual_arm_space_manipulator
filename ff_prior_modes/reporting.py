"""Separate dataset, training, model, execution and advantage claims."""
from pathlib import Path
from collections import Counter
from ff_prior.storage import read_json,atomic_json


def report(out,ledger,status):
    from .experiment import METHODS
    out=Path(out);frozen=read_json(out/'frozen_tasks.json');targets=frozen['evaluation_entries'];metrics={}
    for method in METHODS:
        rows=[read_json(p) for p in (out/'candidates').glob(f'*/{method}/*.json')]
        executions=[read_json(p) for p in (out/'executions').glob(f'*/{method}/outcome.json')];reasons=Counter()
        for r in rows:
            if r['eligible']:continue
            if r['status'] in ('ERROR','BLOCKED'):reasons[r.get('reason',r['status'])]+=1
            else:
                for k,v in r.get('validation',{}).get('gates',{}).items():
                    if not v:reasons[k]+=1
                if not r.get('frozen_goal_pass'):reasons['world_goal']+=1
        metrics[method]={'completed':len(rows),'MODEL_VALID':sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rows),
            'eligible':sum(r['eligible'] for r in rows),'executed_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
            'coupled_queries':sum(r['model_calls']['coupled'] for r in rows),'distance_queries':sum(r['model_calls']['distance'] for r in rows),
            'integration_intervals':sum(r.get('integration_intervals') or 0 for r in rows),'network_calls':sum(r['actual_network_calls'] for r in rows),
            'candidate_wall_s':sum(r['wall_s'] for r in rows),'failure_reasons':dict(reasons),
            'incomplete_accounting':any(r.get('integration_intervals') is None or r['status'] in ('ERROR','BLOCKED') for r in rows),
            'by_task':{e['sample_id']:sum(r['eligible'] for r in rows if r['sample_id']==e['sample_id']) for e in targets},
            'by_group':{g:sum(r['eligible'] for r in rows if r['group']==g) for g in ('old_complex','new_origin')}}
    new=metrics[METHODS[-1]]
    stable=len(targets)==10 and new['by_group']['old_complex']>=27 and new['by_group']['new_origin']>=18 and all(n>=4 for n in new['by_task'].values()) and new['executed_verified']==10
    comparisons={}
    for m in (METHODS[0],METHODS[2]):
        b=metrics[m]
        comparisons[m]=bool(stable and not new['incomplete_accounting'] and not b['incomplete_accounting']
            and new['eligible']>=b['eligible'] and new['executed_verified']>=b['executed_verified'] and (
            new['executed_verified']>b['executed_verified'] or (new['coupled_queries']<=.9*b['coupled_queries'] and new['candidate_wall_s']<=.9*b['candidate_wall_s'])))
    generated=[read_json(p) for p in (out/'generation').glob('*.json')]
    records=[read_json(p) for p in (out/'records').glob('*.json')]
    witnesses=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    by_task={e['sample_id']:sum(r['sample_id'].startswith(e['sample_id']+'_') and r['execution_status']!='FAIL' for r in records) for e in frozen.get('generation_entries',[])}
    accepted=sum(r['execution_status']!='FAIL' for r in records)
    data_gate=len(generated)==48 and accepted>=39 and len(by_task)==8 and all(by_task[f'mode_train_{i:02d}']>=2 for i in range(3)) and sum(w['status']=='EXECUTION_VERIFIED' for w in witnesses)==4
    summary={'status':status,'methods':metrics,'dataset':{'attempts':len(generated),'initial_admissions':sum(r['admitted'] for r in generated),
        'new_training_eligible':accepted,'known_execution_failures':sum(r['execution_status']=='FAIL' for r in records),
        'preselected_execution_verified':sum(w['status']=='EXECUTION_VERIFIED' for w in witnesses),'by_task':by_task,'quality_gate_passed':bool(data_gate)},
        'stability_gate_passed':bool(stable),'exploratory_advantage_gate_passed':all(comparisons.values()),'advantage_against_each_baseline':comparisons,
        'full_goal_achieved':False,'production_expansion':False,'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s'],
        'scope':'small qualified TRAIN enrichment and frozen exploratory VAL; original DEV/TEST not executed'}
    atomic_json(out/'summary.json',summary)
    lines=['# 固定TRAIN目标多轨迹与耦合目标损失','',f'状态：{status}；完整优势和扩库目标未宣称完成。',
        '',f'新增训练可用轨迹：{accepted}/48；预选真实执行与重放：{summary["dataset"]["preselected_execution_verified"]}/4。',
        '', '| 方法 | MODEL_VALID/50 | 目标合格/50 | 执行重放/10 | 耦合查询 | 积分区间 | 候选秒 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,v in metrics.items():lines.append(f'| {m} | {v["MODEL_VALID"]} | {v["eligible"]} | {v["executed_verified"]} | {v["coupled_queries"]} | {v["integration_intervals"]} | {v["candidate_wall_s"]:.3f} |')
    lines+=['',f'数据质量门槛：{data_gate}；目标损失模型稳定门槛：{stable}；Diffusion探索性优势门槛：{summary["exploratory_advantage_gate_passed"]}。',
        '', '离线TRAIN敏感度预计算、训练和候选规划分账；局部34方向代理不是物理验证器。仅不穿透、5mm/1°、2mm几何门槛和原动力学约束不变。',
        '', '保留全部失败，不重标冻结目标，不将未执行样本批量升级。不启动pilot、study、TEST或生产扩库。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary
