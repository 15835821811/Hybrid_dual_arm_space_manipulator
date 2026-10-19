from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_delivery.gates import engineering_gate
from .protocol import METHODS


def report(out,state):
    audit=read_json(out/'independent_verification.json') if (out/'independent_verification.json').exists() else {'status':'PENDING'}
    training=read_json(out/'learning/goal_loss/training.json') if (out/'learning/goal_loss/training.json').exists() else {'status':'BLOCKED'}
    metrics={};gates={};failures=[]
    for method in METHODS:
        rows=[read_json(p) for p in (out/'candidates').glob(f'*/*/{method}/*.json')]
        outcomes=[read_json(p) for p in (out/'executions').glob(f'*/*/{method}/outcome.json')]
        metrics[method]={'completed':len(rows),'eligible':sum(r['eligible'] for r in rows),
            'selected_outcomes':len(outcomes),'execution_verified':sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes),
            'candidate_errors':sum(r['status']=='ERROR' for r in rows),'candidate_blocked':sum(r['status']=='BLOCKED' for r in rows),
            'execution_errors':sum(r['status']=='ERROR' for r in outcomes),
            'candidate_queries':sum(r['model_calls'].get('coupled',0) for r in rows),'candidate_wall_s':sum(r['wall_s'] for r in rows)}
        m=metrics[method]
        gates[method]=engineering_gate(m,180,36,audit['status']=='PASS',[] if not m['execution_errors'] else ['execution exception'])
        failures.extend({'method':method,'sample_id':r['sample_id'],'seed_group':r['seed_group'],'candidate_id':r['candidate_id'],
                         'status':r['status'],'reason':r.get('reason'),'validation':r.get('validation')} for r in rows if not r['eligible'])
    chosen='expanded_diffusion' if training['status']=='PASS' and gates['expanded_diffusion']['status']=='PASS' else 'frozen_diffusion' if gates['frozen_diffusion']['status']=='PASS' else None
    result={'status':state if state=='RUNNING' or chosen else 'FINISHED_WITH_BLOCKED_ITEMS','stage':'E256T',
            'metrics':metrics,'engineering_gates':gates,'training_status':training['status'],
            'actual_updates':training.get('updates'),'actual_TRAIN_draws':training.get('actual_TRAIN_draws'),
            'TRAIN_rows_seen':sum(n>0 for n in training.get('train_draw_counts',{}).values()),
            'audit':audit['status'],'selected_method':chosen,'selected_sampler':'late4_diffusion' if chosen else None,
            'report_ready':chosen is not None,'historical_final_step_fallback':'requires separate identity-bound M0 evidence if needed',
            'diffusion_advantage':'NOT_ESTABLISHED','new_data_this_worker':0,'TEST':'NOT_RUN'}
    atomic_json(out/'summary.json',result);atomic_json(out/'failures.json',failures)
    lines=['# 扩库后一次训练与冻结DEV对照','',f'状态：{result["status"]}；原始审计：{audit["status"]}。',
           f'训练：{training["status"]}；实际更新 {result["actual_updates"]}；TRAIN 实际消费 {result["TRAIN_rows_seen"]}/本轮冻结全部TRAIN，VAL不入梯度。','',
           '|方法|候选合格/180|执行重放/36|工程门|','|---|---:|---:|---|']
    for method,m in metrics.items():lines.append(f'|{method}|{m["eligible"]}/180|{m["execution_verified"]}/36|{gates[method]["status"]}|')
    lines+=['',f'交付候选：{chosen or "尚未满足"}。新旧 Diffusion 都使用同一个后四步引导，不能把旧末步历史结果冒充本轮。',
        '原DEV4＋已用8开发任务是开发回归；不是最终未见泛化。完整候选成功依赖物理引导，Diffusion稳定优势仍需独立证据。',
        '仅不穿透、5mm/1°及其余全部物理门未改变；失败保留分母，执行无替补。不因结果不佳调参重测。',
        '旧1024、pilot、study、strict_multimodal及TEST未运行；结束后交付新旧模型对比和一次独立结果审查，不再扩库或启动TEST。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
