"""Evidence-level summaries: pipeline feasibility is separate from advantage."""
from pathlib import Path
from collections import Counter
from ff_prior.storage import read_json,atomic_json


def report(out,ledger,status):
    from .experiment import METHODS,PROTOCOL
    out=Path(out);attempts=[read_json(p) for p in sorted((out/'generation').glob('*.json'))]
    rows=[read_json(p) for p in sorted((out/'records').glob('*.json'))]
    admitted=[r for r in rows if r['execution_status']!='FAIL']
    witnesses=[read_json(p) for p in (out/'witness_executions').glob('*/outcome.json')]
    metrics={}
    for method in METHODS:
        candidates=[read_json(p) for p in sorted((out/'candidates').glob(f'*/*/{method}/*.json'))]
        outcomes=[read_json(p) for p in (out/'executions').glob(f'*/*/{method}/outcome.json')]
        per_seed={}
        for seed in range(3):
            current=[r for r in candidates if r['seed_group']==seed]
            per_seed[str(seed)]={'completed':len(current),'eligible':sum(r['eligible'] for r in current),
                'tasks_reached':sum(any(r['eligible'] and r['sample_id']==f'DEV_{ti:02d}' for r in current) for ti in range(4))}
        complete=len(candidates)==60 and len(outcomes)==12
        stable=complete and all(s['eligible']>=18 and s['tasks_reached']==4 for s in per_seed.values()) and all(r['status']=='EXECUTION_VERIFIED' for r in outcomes) and all(r['status']=='DONE' for r in candidates)
        metrics[method]={'completed':len(candidates),'eligible':sum(r['eligible'] for r in candidates),
            'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),'per_seed':per_seed,
            'stable':bool(stable),'coupled_queries':sum(r['model_calls'].get('coupled',0) for r in candidates),
            'wall_s':sum(r.get('wall_s',0) for r in candidates),'network_calls':sum(r['actual_network_calls'] for r in candidates),
            'failures':dict(Counter(r.get('reason','PHYSICAL_OR_WORLD_GOAL_REJECTED') for r in candidates if not r['eligible']))}
    train_file=out/'learning/goal_loss/training.json'
    training=read_json(train_file) if train_file.exists() else {}
    trained=training.get('status')=='PASS' and training.get('all_TRAIN_rows_seen') is True
    families={r['family_id'] for r in admitted if r['family_id'].startswith('expand_family_')}
    data_pass=len(admitted)+256>=1000 and len(families)>=150 and len(witnesses)==16 and all(r['status']=='EXECUTION_VERIFIED' for r in witnesses)
    audited=(out/'independent_verification.json').exists() and read_json(out/'independent_verification.json')['status']=='PASS'
    no_unknown=not ledger.data.get('unresolved_call_accounting') and not any(v['status'] in ('ERROR','INTERRUPTED','BLOCKED') for v in ledger.data['completed'].values())
    pilot=bool(status=='FINISHED' and data_pass and trained and metrics['expanded_diffusion']['stable'] and audited and no_unknown)
    mechanism=[]
    for ti in range(3):
        for sj in range(3):
            paths=[out/'mechanism'/f'mechanism_{ti}_{sj}_{m}.json' for m in ('frozen_diffusion','diffusion_goal_only')]
            if not all(p.exists() for p in paths):continue
            a,b=map(read_json,paths)
            mechanism.append({'task_slot':ti,'seed_group':sj,'same_raw_sample':a.get('sampling',{}).get('raw_proposal_hash')==b.get('sampling',{}).get('raw_proposal_hash'),
                'collision_guided_eligible':a['eligible'],'goal_only_eligible':b['eligible'],
                'collision_guided_clearance_m':a.get('validation',{}).get('geometry',{}).get('minimum_clearance_m'),
                'goal_only_clearance_m':b.get('validation',{}).get('geometry',{}).get('minimum_clearance_m')})
    result={'status':status,'protocol':PROTOCOL,'generation_attempts':len(attempts),
        'generation_admitted':sum(r['admitted'] for r in attempts),'new_TRAIN_eligible':len(admitted),'total_TRAIN':256+len(admitted),'VAL':48,
        'new_empty_scene_families':len(families),'new_obstacle_rows':sum(bool(r['scene']['objects']) for r in admitted),
        'generation_failures':dict(Counter(r.get('reason','UNKNOWN') for r in attempts if not r['admitted'])),
        'witness_executions_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in witnesses),
        'new_MODEL_VALID_only':sum(r['execution_status']=='NOT_RUN' for r in admitted),
        'methods':metrics,'mechanism_pairs':mechanism,'data_gate_passed':bool(data_pass),
        'actual_training_passed':bool(trained),'independent_audit_passed':bool(audited),
        'PILOT_ELIGIBLE':pilot,'next_action':'START_AUTHORIZED_PILOT' if pilot else 'REPAIR_IDENTIFIED_FAILURES_BEFORE_PILOT',
        'efficiency_superiority_required':False,'diffusion_superiority':'NOT_ESTABLISHED_BY_THIS_FEASIBILITY_RUN',
        'full_goal_achieved':False,'TEST_executed':False,'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s']}
    atomic_json(out/'summary.json',result)
    lines=['# 有界扩库与Diffusion完整闭环报告','',f'状态：{status}。本轮真实新增合格TRAIN {len(admitted)} 条，累计 {256+len(admitted)}；VAL保持48。',
        f'生成尝试 {len(attempts)}/1200；新空场景家族 {len(families)}；新增障碍场景条目 {result["new_obstacle_rows"]}。',
        f'预选执行及独立重放通过 {result["witness_executions_verified"]}/16。未执行的 {result["new_MODEL_VALID_only"]} 条只标MODEL_VALID。',
        '', '| 方法 | 合格/60 | 选中执行重放/12 | 三种子稳定门槛 |','|---|---:|---:|---|']
    for m,r in metrics.items():lines.append(f'| {m} | {r["eligible"]} | {r["execution_verified"]} | {r["stable"]} |')
    lines+=['',f'实际训练加载/更新通过：{trained}；独立原始证据审计通过：{audited}；pilot资格：{pilot}。',
        '效率仅辅助记录，不参与扩库或机制可行性判断。候选成功包含物理引导修正，不表示裸网络全部合格；优势仍按对照结果另行判断。',
        'TRAIN碰撞引导使用同网络同种子成对诊断，详见mechanism_pairs；原DEV任务保持、只运行冻结smoke，最终TEST未执行。',
        '物理门槛、自由漂浮、完整初态、17→67映射、全机几何及真实力矩重放保持；失败不重标目标、不替换执行。',
        '预留TEST只读取目标元数据用于分区隔离；未读取其轨迹数组、训练或评测。',
        '',f'生成拒绝/异常：{result["generation_failures"]}。',
        '下一阶段：全部验收与审计通过后按用户授权启动pilot；否则保存具体失败并继续修复，不自动运行最终TEST。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
