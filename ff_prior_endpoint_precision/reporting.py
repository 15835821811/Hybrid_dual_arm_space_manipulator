import numpy as np
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import CONTROL

def report(out):
    d=read_json(out/'single_step_summary.json');a=read_json(out/'independent_verification.json')
    train=read_json(out/'learning/training.json') if (out/'learning/training.json').exists() else {'status':'BLOCKED_M1'}
    candidates=[read_json(p) for p in out.glob('candidates/*/*/*.json')];executions=[read_json(p) for p in out.glob('executions/*/*/outcome.json')]
    qualified=sum(x['eligible'] for x in candidates);passed=sum(x['status']=='EXECUTION_VERIFIED' for x in executions)
    engineering=a['status']=='PASS' and len(candidates)==180 and len(executions)==36 and qualified>=144 and passed>=29 and not any(x['status']=='ERROR' for x in candidates+executions)
    result={'status':'FINISHED' if engineering else 'FINISHED_WITH_BLOCKED_ITEMS','M1':d['status'],'M1_accepted':d['accepted'],
     'M1_per_noise':d['accepted_per_noise'],'training_status':train['status'],'updates':train.get('updates',0),
     'world_events':train.get('world_events',0),'world_accepted':train.get('world_accepted',0),'audit':a['status'],
     'new_candidate_records':len(candidates),'new_qualified':qualified,'new_execution_units':len(executions),'new_execution_verified':passed,
     'reused_candidate_records':540,'reused_execution_units':108,'engineering_gate':'PASS' if engineering else 'BLOCKED',
     'Diffusion_benefit':'REQUIRES_INDEPENDENT_REVIEW' if engineering else 'NOT_ESTABLISHED','TEST':'NOT_RUN','new_training_data':0}
    if d['status']=='PASS':
        from .statistics import summarize
        summarize(out)
    atomic_json(out/'summary.json',result)
    text=['# 绝对世界终点有界验证','',f'状态 {result["status"]}；原始审计 {a["status"]}。',
      f'M1接受 {d["accepted"]}/24，各噪声 {d["accepted_per_noise"]}，依赖门 {d["status"]}。',
      f'新训练 {train["status"]}；实际更新 {result["updates"]}；世界事件 {result["world_events"]}，接受 {result["world_accepted"]}。',
      f'本轮新DEV候选 {qualified}/{len(candidates)}（计划180）；新执行重放 {passed}/{len(executions)}（计划36）。',
      '旧对照540候选/108执行仅身份核对复用，不计本轮新执行；M1失败时全部下游BLOCKED，不是0成功率。',
      '物理门未改变，未扩库、未启动最终TEST。科学收益须结束独立审查，不把软件完成当收益成立。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');return result
