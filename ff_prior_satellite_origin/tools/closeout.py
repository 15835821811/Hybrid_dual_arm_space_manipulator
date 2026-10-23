"""Seal reported metadata from completed saved evidence, without re-execution."""
from pathlib import Path
import hashlib, json, time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'ff_prior_satellite_origin'
RUN=ROOT/'artifacts/v7_1/moving_satellite_001'
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261023_satellite_origin'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    audit=read(TRACE/'audit_results.json');verdict=read(TRACE/'verdict.json')
    result=read(RUN/'execution/result.json');status=read(RUN/'status.json')
    resume=read(PKG/'evidence/completed_resume.json');media=read(PKG/'delivery_001/manifest.json')
    assert audit['all_checks_pass'] and not audit['gate_disagreements']
    assert [k for k,v in result['gates'].items() if not v]==['momentum']
    assert verdict['C1']['claim_supported']=='no' and verdict['Caux']['claim_supported']=='yes'
    assert resume['completed_invocations']==1 and resume['identical_before_after']
    # The JSON audit/verdict correctly contain 24 groups. Correct only prose's
    # clerical 25 count; do not rerun the finished review or change its judgments.
    corrected=[]
    for name in ('response.md','handoff.md'):
        p=TRACE/name;s=p.read_text(encoding='utf-8');t=s.replace('25 named checks','24 named checks').replace('all 25 grouped','all 24 grouped')
        if t!=s:p.write_text(t,encoding='utf-8');corrected.append(name)
    write(PKG/'evidence/review_metadata_correction.json',dict(kind='clerical_prose_count_only',audit_checks=len(audit['checks']),verdict_checks=verdict['grouped_checks_total'],corrected_files=corrected,prior_prose_count=25,correct_count=24,new_audit_invocations=0))
    current='''# 当前阶段：实体卫星起源与29秒接续已执行，原机器人动量门失败

最新明确授权已履行：moving_satellite_001从54.5秒真实完整836末态继续2秒公开卫星COM推力，再关机追踪27秒，到83.5秒。一次执行14500步和一次独立完整重放14500步，14501×836差0；29门中28通过，原机器人动量门失败。卫星最终2mm/s、位移56.002mm，未注入速度、裁剪或重置实际状态。不得将执行完成或重放相同写成全门合格。

独立审查C1=no/high、Caux=yes/high（限定披露动量失败），24组核对一致。继承54.5秒状态已超过本段原交换量比例阈值，所有新状态超限；原公式未改，不能扣除初值追认。一次完成resume PASS，28核心文件489816732bytes不变；禁止再次resume/审查/物理。本run报告ff_prior_satellite_origin/docs/ANALYSIS.md，收据ff_prior_fullchain/evidence/moving_satellite_closeout.json。全部实际0–83.5秒视频、三维、坐标系、轨迹、臂身与推力账本在ff_prior_satellite_origin/delivery_001/index.html；新服务被自动审批拒绝，使用本地页面。

本次唯一工程单元已负面收尾；长期目标未完成，当前无重worker，无自动第二候选/修复/训练/扩库/优势对照/TEST。全部物理门保持。下一步问题是连续交接的动量约束一致性，不是起源受力尚未授权。旧授权阻塞计数已归零，新失败只记首次；App旧blocked卡缺少active更新接口，不能伪造complete/paused或替换goal。旧heartbeat按用户要求保持删除。后文均为历史记录，不覆盖本节及control。

'''
    for rel in ('DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md','ff_prior_moving_origin/docs/ORIGIN_AND_BUDGET.md'):
        p=ROOT/rel;s=p.read_text(encoding='utf-8-sig');next_header=s.find('\n# ',2)
        if s.startswith('# 2026-10-23 当前有效授权：') and next_header>=0:s=s[next_header+1:]
        p.write_text(current+s,encoding='utf-8')
    (ROOT/'ff_prior_moving_origin/docs/TRACKER.md').write_text(current+'''|阶段|状态|
|---|---|
|G0 起源与余额方案|完成|
|G1 起源范围|用户明确授权完成|
|G2 实现、成本、冻结|完成；9e4fb4fb，185源码/274输入|
|G3 实际执行与完整重放|完成执行与零差重放，原动量门失败|
|G4 新run审查、媒体、收尾|完成；C1 no，Caux限定yes；负面收尾|

保守余额见ff_prior_satellite_origin/evidence/closeout_budget.json。余额不是下一轮许可。
''',encoding='utf-8')
    c=read(ROOT/'ff_prior_fullchain/control.json')
    c.update(active_run_role='most_recent_closed_run',active_worker=False,current_phase='D3_physical_satellite_origin_closed_momentum_failure',goal_status='INCOMPLETE_ORIGINAL_MOMENTUM_GATE_FAILED',next_phase='D3_SATELLITE_ORIGIN_CLOSED_READONLY_MOMENTUM_DECISION',next_phase_action='Physical origin and coast evidence delivered; original robot momentum failed from inherited initial state. No automatic second unit, rerun, resume, relaxed gate or old audit.',decision_document='ff_prior_satellite_origin/docs/ANALYSIS.md',implementation_handoff='ff_prior_satellite_origin/docs/ANALYSIS.md',origin_scope_unresolved_turns=0,new_momentum_failure_impasse_turns=1)
    for phase in c['phases']:
        if phase['id']=='D3_physical_satellite_origin':phase['status']='CLOSED_EXECUTION_FAILED_MOMENTUM_REPLAY_EXACT_MEDIA_DELIVERED'
    write(ROOT/'ff_prior_fullchain/control.json',c)
    usage=[]
    for p in (RUN,PKG,TRACE):
        files=[f for f in p.rglob('*') if f.is_file()]
        usage.append(dict(path=p.relative_to(ROOT).as_posix(),files=len(files),bytes=sum(f.stat().st_size for f in files)))
    frozen=read(PKG/'evidence/budget_derivation.json')
    elapsed=time.time()-frozen['implementation_started_at']
    budget=dict(status='CLOSED_WITHIN_FROZEN_CAPS',recorded_new_worker_wall_s=status['elapsed_s'],measured_unit_elapsed_to_closeout_s=elapsed,elapsed_scope='calendar wall since this unit implementation start, including review/render/coordination; not historical unmetered time',new_coupled_calls=0,new_model_propagation_calls=0,new_neural_samples=0,execution_steps=14500,independent_replay_steps=14500,new_native_including_replay=29000,core_files=28,core_bytes=resume['core_bytes'],new_file_inventory=usage,inventory_bytes=sum(x['bytes'] for x in usage),unlisted_metadata_and_git_overhead_reserve_bytes=16*1024**2,unit_caps=frozen['unit_caps'],conservative_remaining_after_entire_cap=frozen['remaining_after_caps'],D3_cumulative=dict(coupled=47127,native_including_replay=130660,worker_wall_s=3419.359+status['elapsed_s'],core_bytes=2843483028+resume['core_bytes']),new_run_retry_allowance=0)
    assert elapsed<6000 and budget['inventory_bytes']+16*1024**2<1073741824
    write(PKG/'evidence/closeout_budget.json',budget)
    receipt=dict(version=1,closed_at_utc=datetime.now(timezone.utc).isoformat(),run='artifacts/v7_1/moving_satellite_001',status='CLOSED_EXECUTION_FAILED_MOMENTUM',worker_status=status['status'],producer_head=read(RUN/'run.json')['producer_head'],source_count=185,bound_input_count=274,physical_gate_count=29,physical_gates_passed=28,failed_gates=['momentum'],native_steps=14500,replay_steps=14500,replay_shape=[14501,836],replay_max_abs=0.0,source_and_state_identity=audit['identity'],momentum=audit['momentum'],satellite_ledger=result['satellite_ledger'],new_coupled_calls=0,new_neural_samples=0,physical_retry_count=0,worker_wall_s=status['elapsed_s'],core_file_count=28,core_bytes=resume['core_bytes'],review='paper/review-traces/experiment-result-to-claim/20261023_satellite_origin',review_verdict=verdict,completed_resume_receipt='ff_prior_satellite_origin/evidence/completed_resume.json',completed_resume_receipt_sha256=sha(PKG/'evidence/completed_resume.json'),completed_resume_calls=1,media_index='ff_prior_satellite_origin/delivery_001/index.html',media_manifest_sha256=sha(PKG/'delivery_001/manifest.json'),media_output_bytes=media['output_bytes'],media_duration_s=media['actual_duration_s'],media_physical_status=media['physical_result'],report='ff_prior_satellite_origin/docs/ANALYSIS.md',report_sha256=sha(PKG/'docs/ANALYSIS.md'),budget='ff_prior_satellite_origin/evidence/closeout_budget.json',long_term_goal_complete=False,further_physics_authorized_by_this_receipt=False,automation='deleted by earlier explicit user request; not recreated',server_start='automatic approval rejected: blocked by policy; local file delivery')
    write(ROOT/'ff_prior_fullchain/evidence/moving_satellite_closeout.json',receipt)
    print(json.dumps(dict(status=receipt['status'],new_bytes_with_metadata_reserve=budget['inventory_bytes']+16*1024**2,elapsed_s=elapsed,remaining=frozen['remaining_after_caps']),ensure_ascii=False))

if __name__=='__main__':main()
