"""Mechanism metrics, retaining failure counts and the TRAIN-only claim scope."""
from pathlib import Path
from collections import Counter
import numpy as np
from ff_prior.storage import read_json,atomic_json


def diversity(matrices):
    if not matrices:return {'distinct':0,'pairwise_rms_rad':0.}
    values=[np.asarray(z) for z in matrices];kept=[]
    for z in values:
        if all(np.sqrt(np.mean((z-q)**2))>=.005 for q in kept):kept.append(z)
    pairs=[float(np.sqrt(np.mean((values[i]-values[j])**2))) for i in range(len(values)) for j in range(i)]
    return {'distinct':len(kept),'pairwise_rms_rad':float(np.mean(pairs)) if pairs else 0.}


def report(out,ledger,status):
    from .experiment import METHODS,PROTOCOL
    out=Path(out);entries=read_json(out/'frozen_tasks.json')['evaluation_entries'];metrics={}
    for method in METHODS:
        rows=[read_json(p) for p in sorted((out/'candidates').glob(f'*/{method}/*.json'))]
        outcomes=[read_json(p) for p in (out/'executions').glob(f'*/{method}/outcome.json')]
        reasons=Counter();by_task={};ds=[];raw=[]
        for r in rows:
            if not r['eligible']:
                if r['status']!='DONE':reasons[r.get('reason',r['status'])]+=1
                else:
                    for k,v in r['validation']['gates'].items():
                        if not v:reasons[k]+=1
                    if not r['frozen_goal_pass']:reasons['world_goal']+=1
            if r.get('sampling'):
                raw.append(np.linalg.norm(np.asarray(r['sampling']['guidance'][-1]['initial_goal_error']).reshape(4,3),axis=1))
        for e in entries:
            here=[r for r in rows if r['sample_id']==e['sample_id']]
            accepted=[r for r in here if r['eligible']];d=diversity([r['Z'] for r in accepted]);ds.append(d)
            by_task[e['sample_id']]={'completed':len(here),'eligible':len(accepted),'coupled_queries':sum(r['model_calls']['coupled'] for r in here),
                'candidate_wall_s':sum(r['wall_s'] for r in here),**d}
        eligible=sum(r['eligible'] for r in rows);executed=sum(r['status']=='EXECUTION_VERIFIED' for r in outcomes)
        model=sum(r.get('validation',{}).get('validation_level')=='MODEL_VALID' for r in rows)
        stable=len(rows)==40 and eligible>=36 and all(t['eligible']>=4 for t in by_task.values()) and executed==2
        norm=np.asarray(raw)
        metrics[method]={'completed':len(rows),'MODEL_VALID':model,'eligible':eligible,'execution_verified':executed,
            'coupled_queries':sum(r['model_calls']['coupled'] for r in rows),'repair_queries':sum(r['repair_coupled_calls'] for r in rows),
            'integration_intervals':sum(r.get('integration_intervals') or 0 for r in rows),'network_calls':sum(r['actual_network_calls'] for r in rows),
            'candidate_wall_s':sum(r['wall_s'] for r in rows),'by_task':by_task,'stable':stable,'failure_reasons':dict(reasons),
            'incomplete_accounting':any(r.get('integration_intervals') is None for r in rows),
            'distinct_eligible_total':sum(d['distinct'] for d in ds),'mean_pairwise_eligible_rms_rad':float(np.mean([d['pairwise_rms_rad'] for d in ds])) if ds else 0.,
            'pre_repair_position_median_mm':float(np.median(norm[:,:2].max(1))*5) if len(raw) else None,
            'pre_repair_rotation_median_deg':float(np.median(norm[:,2:].max(1))) if len(raw) else None,
            'pre_repair_pose_pass':int(sum(norm.max(1)<=1)) if len(raw) else 0}
    signals={}
    for prefix in ('v','g'):
        full=metrics[prefix+'100']
        for suffix in ('025','050'):
            m=prefix+suffix;v=metrics[m]
            signals[m]=bool(v['stable'] and full['stable'] and not v['incomplete_accounting'] and not full['incomplete_accounting']
                and v['eligible']>=full['eligible'] and v['coupled_queries']<=.9*full['coupled_queries'] and v['candidate_wall_s']<=.9*full['candidate_wall_s']
                and v['mean_pairwise_eligible_rms_rad']>=.8*full['mean_pairwise_eligible_rms_rad'] and v['distinct_eligible_total']>=.8*full['distinct_eligible_total'])
    summary={'status':status,'protocol':PROTOCOL,'methods':metrics,'attenuation_signal':signals,
        'scope':'TRAIN-only diagnostic; no independent generalization or Diffusion-superiority claim',
        'full_goal_achieved':False,'production_expansion':False,'ledger_calls':ledger.data['calls'],'ledger_wall_s':ledger.data['wall_s']}
    atomic_json(out/'summary.json',summary)
    lines=['# TRAIN残差幅度机制诊断','',f'状态：{status}。仅TRAIN机制证据，不构成泛化或完整优势证明。','',
        '| 方法 | 合格/40 | 预选执行/2 | 查询 | 候选秒 | 修正前位置中位mm | 合格不同轨迹数 | 合格两两RMS均值rad |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in metrics.items():
        pos=v['pre_repair_position_median_mm'];ps='—' if pos is None else f'{pos:.3f}'
        lines.append(f'| {m} | {v["eligible"]} | {v["execution_verified"]} | {v["coupled_queries"]} | {v["candidate_wall_s"]:.3f} | {ps} | {v["distinct_eligible_total"]} | {v["mean_pairwise_eligible_rms_rad"]:.6f} |')
    lines+=['',f'预注册衰减保留多样性且降成本信号：{signals}。','',
        '失败、未完成和未执行保留各自等级；仅预选2个TRAIN任务/方法实际执行。原DEV、VAL和TEST不运行；不新增训练或扩库。',
        '固定基座、目标重标和物理门槛放宽均未采用；所有模型调用和共同修正成本按候选计费。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
