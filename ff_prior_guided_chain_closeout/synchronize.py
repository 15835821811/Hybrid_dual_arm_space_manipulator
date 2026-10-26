"""Reconcile the closed new unit without changing frozen source or old evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
control_path=ROOT/'ff_prior_fullchain/control.json'
c=json.loads(control_path.read_text(encoding='utf-8'))
c.update(active_run_role='most_recent_closed_run',active_worker=False,
 current_phase='D3_guided_diffusion_chain_closed_derivative_failure',
 next_phase='READ_ONLY_DERIVATIVE_REPAIR_DECISION_NO_AUTOMATIC_RUN',
 next_phase_action='One frozen chain ended at first guidance derivative check. Preserve gates; diagnose geometric derivative and abort archival before any separately frozen new unit. No automatic retry.',
 goal_status='LONG_TERM_INCOMPLETE_CURRENT_UNIT_CLOSED_NEGATIVE',
 new_guidance_failure_impasse_turns=1,
 latest_actual_visualization=c.get('latest_visualization'),
 latest_visualization='artifacts/v7_1/guided_diffusion_chain_001/delivery/index.html',
 latest_visualization_kind='MODEL_ONLY_NO_NEW_NATIVE',
 decision_document='ff_prior_guided_chain/docs/ANALYSIS.md',
 latest_closeout_budget='ff_prior_guided_chain_closeout/evidence/final_budget.json',
 latest_result_review='paper/review-traces/experiment-result-to-claim/20261026_guided_chain/verdict.json')
for p in c['phases']:
 if p['id']=='D3_guided_diffusion_chain':p.update(status='CLOSED_GUIDANCE_DERIVATIVE_FAILED_NO_MODEL_CERTIFICATION_NO_NATIVE',neural_forward_calls=13,completed_ddim_steps=12,coupled=342,native=0,replay=0)
write(control_path,c)
prefix='''# 2026-10-26 当前：新 Diffusion 整轨迹单元已冻结运行，导数检查负面收尾

追加30000耦合、50000原生、11小时和7.5GiB已获授权并落实。唯一 guided_diffusion_chain_001 已实现并冻结（源码4474adeb、冻结4e4fde56；218实际源码/89输入），37纯软件检查通过，单worker真实启动后于88.093秒停止。A段第13次冻结网络前向、DDIM零基12的首个全机引导导数门失败；12完整DDIM步骤留存，323列Jacobian及四组方向探针完成，0接受的全机修正，0最终完成采样，未进入完整MODEL_VALID认证，0原生/0重放，B/C依赖BLOCKED。RK4动量修复尚未原生验证，不能声称83.5秒链已完成。

失败涉及R1球障碍与back_arm_link网格的5个时刻、12个探针行组合；最近特征切换尚非已证实唯一根因。1094保存算术/身份核对一致，不等于物理通过。完成resume已经且仅一次PASS/NO_WORK_ALREADY_COMPLETE，34文件7058866bytes清单/hash/大小不变；原始stdout523/stderr0bytes保留，禁止再resume。独立审查trace：paper/review-traces/experiment-result-to-claim/20261026_guided_chain。报告ff_prior_guided_chain/docs/ANALYSIS.md；完整收尾预算与收据见control。

当前无worker。原全部门、初态、权重、七点和2秒公开推力合同不改；无训练/扩库/优势对照/TEST。下一步是距离导数与异常证据落盘的具体修复决策；本次零重试，剩余额度不自动授权第二候选或新物理。新模型诊断页artifacts/v7_1/guided_diffusion_chain_001/delivery/index.html明确MODEL ONLY，没有新真实视频/三维；旧实际媒体保留。长期目标未完成，旧定时任务保持删除，不伪造App goal complete/paused。以下旧“尚未授权/冻结/运行”只作历史，不覆盖本节和control。

'''
for rel in ['DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md']:
 p=ROOT/rel;old=p.read_text(encoding='utf-8-sig')
 assert not old.startswith(prefix.splitlines()[0]),'ALREADY_SYNCHRONIZED'
 p.write_text(prefix+old,encoding='utf-8')
print('closed state synchronized; frozen run unchanged')
