"""Partial qualification is retained; only actual successful evidence passes."""
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import PROTOCOL


def gate(qualified,targets,verified,execution_units,slots,errors,audit):
    checks={'qualified_minimum':qualified>=64,'target_minimum':targets>=16,
            'execution_minimum':verified>=13,'all_execution_units_accounted':execution_units==16,
            'all_slots_accounted':slots==128,'no_unresolved_errors':errors==[], 'raw_audit_passed':audit=='PASS'}
    return {'status':'PASS' if all(checks.values()) else 'PARTIAL_QUALIFIED','checks':checks}


def report(out,state,ledger):
    attempts=[read_json(p) for p in sorted((out/'generation').glob('*.json'))]
    rows=[read_json(p) for p in sorted((out/'records').glob('*.json'))
          if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']
    qualified=[r for r in rows if r['training_eligible'] and r['execution_status'] in ('NOT_RUN','PASS')]
    outcomes=[read_json(p) for p in (out/'executions').glob('*/outcome.json')]
    slots=list((out/'slots').glob('*.json'))
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','INTERRUPTED')]
    errors+=ledger.data.get('unresolved_call_accounting',[])
    audit=read_json(out/'independent_verification.json')['status'] if (out/'independent_verification.json').exists() else 'PENDING'
    verified=sum(o['status']=='EXECUTION_VERIFIED' for o in outcomes)
    readiness=gate(len(qualified),len({r['target_id'] for r in qualified}),verified,len(outcomes),len(slots),errors,audit)
    result={'status':state if state=='RUNNING' or readiness['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS',
            'stage':'M2','data_status':readiness['status'],'engineering_gate':readiness,
            'attempts':len(attempts),'slots_accounted':len(slots),'model_admissions':len(rows),
            'qualified_increment':len(qualified),'fixed_targets_covered':len({r['target_id'] for r in qualified}),
            'execution_units':len(outcomes),'execution_verified':verified,
            'increment_MODEL_VALID':sum(r['validation_level']=='MODEL_VALID' for r in qualified),
            'increment_EXECUTION_VERIFIED':sum(r['validation_level']=='EXECUTION_VERIFIED' for r in qualified),
            'total_TRAIN':13838+len(qualified),'VAL':48,'errors':errors,'audit':audit,
            'admission_rate_actual_attempts':len(qualified)/len(attempts) if attempts else 0.,
            'training_updates':0,'diffusion_advantage':'NOT_ESTABLISHED','TEST':'NOT_RUN',
            'M3_ready':readiness['status']=='PASS','protocol':PROTOCOL}
    atomic_json(out/'summary.json',result)
    atomic_json(out/'failures.json',{'generation':[{'sample_id':r['sample_id'],'status':r['status'],'reason':r.get('reason'),
        'validation':r.get('validation_result',{}).get('validation')} for r in attempts if not r['admitted']],
        'execution':[dict(target_id=p.parent.name,**read_json(p)) for p in (out/'executions').glob('*/outcome.json') if read_json(p)['status']!='EXECUTION_VERIFIED'],
        'unresolved':errors})
    text=f'''# M2 固定 TRAIN 目标有界补库

状态：{result['status']}；数据：{result['data_status']}；原始审计：{audit}。

- 实际尝试 {len(attempts)}/256，已记录槽 {len(slots)}/128。
- 模型准入 {len(rows)}；排除执行失败/未知后合格增量 {len(qualified)}，覆盖 {result['fixed_targets_covered']} 个固定世界目标。
- 预选执行重放 {verified}/16；未执行样本仍仅 MODEL_VALID。
- 增量等级：MODEL_VALID {result['increment_MODEL_VALID']}，EXECUTION_VERIFIED {result['increment_EXECUTION_VERIFIED']}。
- 合并 TRAIN {result['total_TRAIN']}，VAL 48 保持；本 worker 训练更新 0。
- M3 准备就绪：{result['M3_ready']}。Diffusion 优势 NOT_ESTABLISHED。

完整初态、17→67、自由漂浮、全机几何、原生力矩和独立重放保留。物理门未改变；失败不重标、不替补执行、不升级未执行数据。4 槽不代表 4 模式，轨迹差异见 diversity.json。

旧 1024、pilot、study、strict_multimodal、TEST 未运行。主库不要求同目标五条准入。预算耗尽保留部分合格数据，不循环填满。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    return result
