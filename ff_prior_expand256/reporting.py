"""Only observed qualification passes; unknown attempts retain their denominator."""
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import PROTOCOL


def gate(qualified,targets,verified,execution_units,slots,attempts,errors,audit):
    checks={'qualified_minimum':qualified>=128,'target_minimum':targets>=32,
        'execution_minimum':verified>=26,'all_execution_units_accounted':execution_units==32,
        'all_slots_accounted':slots==256,'qualification_per_actual_attempt':attempts>0 and qualified/attempts>=.5,
        'attempt_cap':attempts<=512,'no_unresolved_errors':not errors,'raw_audit_passed':audit=='PASS'}
    return {'status':'PASS' if all(checks.values()) else 'PARTIAL_QUALIFIED','checks':checks}


def report(out,state,ledger):
    attempts=[read_json(p) for p in sorted((out/'generation').glob('*.json'))]
    rows=[read_json(p) for p in sorted((out/'records').glob('*.json')) if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']
    qualified=[r for r in rows if r['training_eligible'] and r['execution_status'] in ('NOT_RUN','PASS')]
    outcomes=[read_json(p) for p in (out/'executions').glob('*/outcome.json')]
    slots=list((out/'slots').glob('*.json')); consumed=ledger.data['attempts'].get('generation',0)
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED')]
    errors+=ledger.data.get('unresolved_call_accounting',[])
    audit=read_json(out/'independent_verification.json')['status'] if (out/'independent_verification.json').exists() else 'PENDING'
    verified=sum(o['status']=='EXECUTION_VERIFIED' for o in outcomes)
    readiness=gate(len(qualified),len({r['target_id'] for r in qualified}),verified,len(outcomes),len(slots),consumed,errors,audit)
    result={'status':state if state=='RUNNING' or readiness['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS',
        'stage':'E256','data_status':readiness['status'],'engineering_gate':readiness,'attempts':consumed,
        'actual_generation_records':len(attempts),'unknown_or_pending_attempts':consumed-len(attempts),
        'slots_accounted':len(slots),'model_admissions':len(rows),'qualified_increment':len(qualified),
        'fixed_targets_covered':len({r['target_id'] for r in qualified}),'execution_units':len(outcomes),'execution_verified':verified,
        'increment_MODEL_VALID':sum(r['validation_level']=='MODEL_VALID' for r in qualified),
        'increment_EXECUTION_VERIFIED':sum(r['validation_level']=='EXECUTION_VERIFIED' for r in qualified),
        'total_TRAIN':13966+len(qualified),'VAL':48,'errors':errors,'audit':audit,
        'admission_rate_actual_attempts':len(qualified)/consumed if consumed else 0.,
        'training_updates':0,'diffusion_advantage':'NOT_ESTABLISHED','TEST':'NOT_RUN',
        'next_stage_ready':readiness['status']=='PASS','protocol':PROTOCOL}
    atomic_json(out/'summary.json',result)
    atomic_json(out/'failures.json',{'generation':[{'sample_id':r['sample_id'],'status':r['status'],'reason':r.get('reason'),
        'validation':r.get('validation_result',{}).get('validation')} for r in attempts if not r['admitted']],
        'execution':[dict(target_id=p.parent.name,**read_json(p)) for p in (out/'executions').glob('*/outcome.json') if read_json(p)['status']!='EXECUTION_VERIFIED'],
        'unresolved':errors})
    text=f'''# 256槽固定TRAIN世界目标扩库

状态：{result['status']}；数据：{result['data_status']}；原始审计：{audit}。

- 消耗尝试 {consumed}/512，实际结果记录 {len(attempts)}，未知/待结算 {consumed-len(attempts)}；槽 {len(slots)}/256。
- 模型准入 {len(rows)}；排除执行失败/未知后合格增量 {len(qualified)}，覆盖 {result['fixed_targets_covered']} 个固定TRAIN目标。
- 预选真实执行及独立重放 {verified}/32，无替补；MODEL_VALID {result['increment_MODEL_VALID']}，EXECUTION_VERIFIED {result['increment_EXECUTION_VERIFIED']}。
- 合并TRAIN {result['total_TRAIN']}，VAL48不变；实际训练更新0，模型没有更新。
- 下一阶段准备门：{result['next_stage_ready']}。资格率分母含未知保留尝试；预算不足保留部分，不循环填满。

保持历史仅不穿透、双世界末端5mm/1°、2mm几何步长及其余原物理门，完整初态、17→67、自由漂浮、全机几何、原生力矩执行和独立重放。不得失败重标目标或批量升级未执行样本。四槽不等于四模式，主库不要求同目标五条。

收益NOT_ESTABLISHED，未测新优势。旧1024、pilot、study、strict_multimodal、最终TEST及自动递归扩库均未运行。结束后以独立结果审查和分析报告判断下一阶段。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    return result
