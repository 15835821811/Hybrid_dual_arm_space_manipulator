"""Persist the user's explicit budget extension once, without starting physics."""
from pathlib import Path
import json,time,datetime
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'ff_prior_guided_chain'
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=PACKAGE/'evidence/authorization.json'
    if p.exists(): raise RuntimeError('AUTHORIZATION_ALREADY_RECORDED')
    b=json.loads((ROOT/'ff_prior_guided_chain_plan/evidence/budget_derivation.json').read_text(encoding='utf-8-sig'))
    a=dict(created_at=datetime.datetime.now().astimezone().isoformat(),started_at=time.time(),
        user_request='授权追加30000次耦合求解、50000个原生步、11小时和7.5GiB，按方案实现、核算并冻结后单worker运行。冻结现有权重，把Diffusion接入整轨迹生成和全机约束引导，从0秒完整初态重建83.5秒连续链并修复动量问题。保持全部物理门，不注速、不重置、不扣初始动量；完整重放通过后交付全部可视化，不训练、不扩库、不做优势对照或TEST。',
        approved_extension=b['rounded_extension_proposal'],unit_caps=b['proposed_total_unit_caps'],
        global_caps=b['proposed_global_caps_if_approved'],D3_caps=b['proposed_D3_caps_if_approved'],
        reservation_after=b['after_extension_and_full_reservation'],physical_trial_authorized=True,
        candidates=1,retries=0,source_plan='ff_prior_guided_chain_plan/docs/EXPERIMENT_PLAN.md',
        production_frozen=False,worker_started=False,training=False,TEST=False)
    write(p,a)
    write(PACKAGE/'evidence/budget.json',dict(b,status='AUTHORIZED_CAPS_APPLIED_NEW_UNIT_RESERVED',
        approved=True,old_reservations_reclaimed=False,original_caps_changed=True,new_worker_started=False))
    cp=ROOT/'ff_prior_fullchain/control.json';c=json.loads(cp.read_text(encoding='utf-8-sig'))
    write(PACKAGE/'evidence/control_before.json',c)
    c.update(current_phase='D3_guided_diffusion_chain_implementation',
        current_user_steering='ff_prior_guided_chain/evidence/authorization.json',
        authorization_evidence='ff_prior_guided_chain/evidence/authorization.json',
        next_phase='D3_GUIDED_CHAIN_AUTHORIZED_IMPLEMENT_COST_AND_FREEZE',
        next_phase_action='Implement all phases, verify costs/software and freeze; then one worker. Additional budget explicitly approved.',
        latest_planning_budget='ff_prior_guided_chain/evidence/budget.json',
        autonomous_scope='Latest explicit user approval includes extra resources and one integrated guided chain.',
        budget_extension=a['approved_extension'],effective_global_caps=a['global_caps'],effective_D3_caps=a['D3_caps'])
    c['phases'][-1]['status']='AUTHORIZED_IMPLEMENTATION_NOT_FROZEN'
    write(cp,c)
    prefix='# 2026-10-26 新增额度已授权，进入整轨迹引导实现\n\n用户已明确追加30000耦合、50000原生、11小时和7.5GiB。按已交付方案实现、核算并冻结一个连续单元，再单worker运行。证据ff_prior_guided_chain/evidence/authorization.json；预算ff_prior_guided_chain/evidence/budget.json。原门、权重、完整初态、七点及卫星公开推力不变，无注速/重置/扣初值/训练/扩库/优势对照/TEST。以下“追加预算未决”为历史，不覆盖新授权。当前尚未冻结或运行，旧run仍封存。\n\n'
    for rel in ['DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md']:
        f=ROOT/rel;f.write_text(prefix+f.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(dict(status='AUTHORIZED_RESERVED',caps=a['unit_caps'],worker_started=False)))
if __name__=='__main__':main()
