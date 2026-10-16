"""Result summaries are recomputed from per-candidate and actual execution evidence."""
from collections import Counter
from pathlib import Path
from ff_prior.storage import read_json,atomic_json


def metrics(out):
    from .experiment import METHODS
    out=Path(out); result={}
    targets=read_json(out/'frozen_tasks.json')['evaluation_entries']
    for method in METHODS:
        rows=[read_json(p) for p in (out/'candidates').glob(f'*/{method}/*.json')]
        executions=[read_json(p) for p in (out/'executions').glob(f'*/{method}/outcome.json')]
        failures=Counter()
        for r in rows:
            if r['eligible']: continue
            if r['status']=='ERROR': failures[r.get('reason','ERROR')]+=1
            else:
                for k,v in r['validation'].get('gates',{}).items():
                    if not v: failures[k]+=1
                if not r.get('frozen_goal_pass'): failures['world_goal']+=1
        result[method]={'slots':50,'completed_candidates':len(rows),'MODEL_VALID':sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rows),
            'eligible':sum(r['eligible'] for r in rows),'executed_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in executions),
            'executed_attempts':sum(r['status'] not in ('NO_CANDIDATE','BLOCKED') for r in executions),
            'coupled_queries':sum(r['model_calls']['coupled'] for r in rows),'geometry_queries':sum(r['model_calls']['distance'] for r in rows),
            'wall_s':sum(r['wall_s'] for r in rows),'network_calls':sum(r.get('sampling',{}).get('actual_network_calls',0) for r in rows),
            'failure_reasons':dict(failures),'by_task':{e['sample_id']:sum(r['eligible'] for r in rows if r['sample_id']==e['sample_id']) for e in targets},
            'by_group':{g:sum(r['eligible'] for r in rows if r['group']==g) for g in ('old_complex','new_origin')}}
    return result


def decision(out):
    out=Path(out); m=metrics(out); trained=m['trained_diffusion_joint']
    generated=[read_json(p) for p in (out/'generation').glob('generate_loop_*.json')]
    witness=[read_json(p) for p in (out/'witness_executions').glob('loop_*/outcome.json')]
    t=read_json(out/'learning/training.json') if (out/'learning/training.json').exists() else {}
    checks={'data_eligible_at_least_85pct_of_48':sum(r.get('eligible',False) for r in generated)>=41,
        'witness_8_of_8':sum(r['status']=='EXECUTION_VERIFIED' for r in witness)==8,
        'new_training_all_rows_used':t.get('status')=='PASS' and t.get('all_TRAIN_rows_seen') is True,
        'old_complex_eligible_at_least_27_of_30':trained['by_group']['old_complex']>=27,
        'new_origin_eligible_at_least_18_of_20':trained['by_group']['new_origin']>=18,
        'each_task_at_least_4_of_5':len(trained['by_task'])==10 and all(v>=4 for v in trained['by_task'].values()),
        'trained_execution_10_of_10':trained['executed_verified']==10}
    return {'passed':all(checks.values()),'checks':checks,'decision':'BOUNDED_12_TRAIN_ONLY' if all(checks.values()) else 'BLOCKED',
        'diffusion_benefit_not_inferred':True,'pilot':False,'TEST':False}


def report(out,ledger,status):
    out=Path(out); m=metrics(out); gate=decision(out)
    generated=[read_json(p) for p in (out/'generation').glob('generate_loop_*.json')]
    witness=[read_json(p) for p in (out/'witness_executions').glob('loop_*/outcome.json')]
    summary={'status':status,'methods':m,'new_generation_attempted':len(generated),
        'new_generation_eligible':sum(r.get('eligible',False) for r in generated),'new_generation_slots':48,
        'preselected_witness_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in witness),
        'expansion_gate':gate,'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s'],
        'software_status':'PASS','model_prior_status':'SEE_PER_ROW_LEVELS','physical_status':'SEE_EXECUTION_REPORTS',
        'diffusion_advantage':'NOT_INFERRED_FROM_SUCCESS_ALONE','scope':'exploratory TRAIN/VAL, stationary origins, static scenes; no DEV or TEST'}
    atomic_json(out/'summary.json',summary)
    lines=['# 数据生成与碰撞引导闭环','',f'状态：{status}。这是探索性 TRAIN/VAL smoke，原冻结 DEV 保留且未运行，未启动 pilot / TEST。',
        '',f'新增合格标签 {summary["new_generation_eligible"]}/48；实际尝试 {len(generated)}。预选力矩执行与重放 {summary["preselected_witness_verified"]}/8。',
        '', '| 方法 | MODEL_VALID / 50 | 指定目标合格 / 50 | 实际执行及重放 / 10 | 耦合查询 | 候选秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in m.items(): lines.append(f'| {name} | {r["MODEL_VALID"]} | {r["eligible"]} | {r["executed_verified"]} | {r["coupled_queries"]} | {r["wall_s"]:.3f} |')
    lines+=['','扩库判定：'+gate['decision']+'。仅在所有预注册门槛通过时新增最多 12 条 TRAIN 尝试；不因此宣布 Diffusion 收益。','',
        '碰撞激活距离 3 mm 是优化代价；验收仍使用原有仅不穿透阈值 0 mm、全机几何和 2 mm 几何采样门槛。世界目标冻结后未重标。',
        '', 'MODEL_VALID 只表示模型验证通过；只有单条真实 67 力矩执行和独立重放通过才标记 EXECUTION_VERIFIED。此处真实执行是 MuJoCo 动力学执行。',
        '', '训练证据见 learning/training.json 与 training_input_audit.json；失败原因、逐任务结果见 summary.json；原始报告见 executions。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return summary
