"""Write reviewer trace records; no experiment execution."""
from pathlib import Path
import json,hashlib,datetime
TRACE=Path(__file__).resolve().parent;ROOT=TRACE.parents[3]
RUN=ROOT/'artifacts/v7_1/temporal_learning_001'
raw=json.loads((TRACE/'raw_audit.json').read_text(encoding='utf-8'))
inputs=json.loads((TRACE/'inputs.json').read_text(encoding='utf-8'))
def read(rel):
    p=ROOT/rel;b=p.read_bytes();inputs['inputs'][rel]={'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b),'mtime_ns':p.stat().st_mtime_ns};return json.loads(b)
receipt=read('ff_prior_temporal_learning/evidence/completed_resume.json')
attempt=read('ff_prior_temporal_learning/evidence/completed_resume_attempt.json')
manifest=read('ff_prior_temporal_learning/evidence/core_manifest.json')
statistics=read('ff_prior_temporal_learning/evidence/completed_statistics.json')
actual={p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()}
mismatches=[]
for rel,item in manifest.items():
    p=RUN/rel;h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=item['sha256'] or p.stat().st_size!=item['bytes']:mismatches.append(rel)
assert len(manifest)==278 and actual==set(manifest) and not mismatches
assert receipt['only_attempt'] and receipt['returncode']==0 and receipt['core_files_unchanged'] and receipt['before_files']==receipt['after_files']==278 and 'NO_WORK_ALREADY_COMPLETE' in receipt['stdout']
assert statistics['counts']==raw['counts'] and statistics['actual_native_steps']==raw['budget']['native_steps_actual'] and statistics['actual_replay_steps']==raw['budget']['replay_steps_actual']
raw['executor_artifacts_read_only']={'resume_receipt':receipt,'resume_attempt':attempt,'core_manifest_files_verified':278,'core_manifest_mismatches':mismatches,'statistics_path':'ff_prior_temporal_learning/evidence/completed_statistics.json','paired_statistics_producer':'parent executor; reviewer does not duplicate paired-statistics computation','counts_steps_crosscheck':True,'reviewer_resume_calls':0}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
verdict={
 'audit_skill':'experiment-result-to-claim','review_role':'fresh independent paper architect/reviewer','review_trace':'20261022_temporal_learning_run01','generated_at':now,
 'run':'artifacts/v7_1/temporal_learning_001','frozen_source_commit':'681965a0889fa1f4625e66d356a34b838f77c837',
 'claim_supported':'partial','verdict':'partial','confidence':'high','provisional':True,'provisional_reason':'provisional — no paper claim audit available','paper_claim_audit':'unavailable','pending_external_review':False,
 'C1':{'claim':'固定任务上的动态时序训练、完整采样、prepare、共同世界反馈、原生执行及完整状态重放工程链闭合','claim_supported':'yes','confidence':'high','scope':'两个冻结DEV任务；每场景每方法至少一条完整合格执行；不赋裸网络MODEL_VALID或长期总目标通过'},
 'C2':{'claim':'Diffusion相对同数据确定性序列模型在跨场景上存在统一科学优势','claim_supported':'no','confidence':'high','scientific_advantage':'NOT_ESTABLISHED_MIXED_DESCRIPTIVE_OUTCOMES','scope':'漂移DEV部分指标改善；静止位置退化；VAL两法均失败；不等于证明所有未来Diffusion方法无效'},
 'what_results_support':[
   '178冻结源码、369绑定输入与执行zip一致；8TRAIN/4VAL条件、标签、codec重构一致，无DEV/VAL梯度污染证据。',
   '两网络各3000真实AdamW更新、32临时诊断更新后恢复、所有3000摘要与重载哈希通过；10个Diffusion提案均完整16DDIM并精确复现。',
   '20逻辑记录、16实际执行中8通过8失败；DEV确定性2次实际执行+4复用、Diffusion6次实际执行，两法6/6逻辑通过。',
   '16条全部836维原生与保存独立重放完全相同；保存物理门、URDF位姿、时间、几何步长、预算核对一致。',
   '共同世界反馈修正实质参与实际控制；工程协议允许继续P1/P4。'],
 'what_results_dont_support':[
   '无引导网络可用性或MODEL_VALID：VAL两法各0/4；确定性4条碰撞短轨迹，Diffusion4条完整但世界误差/导数失门。',
   '跨场景统一Diffusion优势、独立统计显著性、总体泛化、实时运行或硬件/连续时间安全证书。',
   '把共同反馈后目标精度归为裸网络，或把6逻辑确定性DEV记为6独立执行。',
   '全五障碍、连续组合或长期总目标完成。'],
 'missing_evidence':[
   'P1五障碍机制与P4连续组合完整合格执行。','无引导VAL成功及prepare平滑性/实际响应余量修复证据。',
   '均值参考与共同控制器贡献消融。','更多独立任务/训练种子的公平科学比较。','paper/PAPER_CLAIM_AUDIT.json。'],
 'suggested_claim_revision':'已验证固定动态任务上的时序学习与共同反馈执行闭合；Diffusion相对收益尚未建立，VAL仍失败。',
 'next_experiments_needed':[
   '按已有授权在独立冻结身份推进P1/P4，保留本run全部正负结果；本审查不新增任何物理实验。',
   '未来候选将full integration replay误差纳入接受门，研究参考平滑性和实际跟踪余量；不能同身份修复、改门或升级旧失败。',
   '只有要增强C2时再规划均值/共同反馈消融及独立任务训练重复；C2科学优势不是当前工程推进前置条件。'],
 'implementation_findings':[
   {'severity':'future_gate_robustness','file':'ff_prior_dynamic_reference/execution.py','line':67,'issue':'fullerr已计算但independent_replay门仅使用qpos/qvel err；本run全部fullerr=0，非本次接受失败。'},
   {'severity':'failed_candidate_behavior','file':'ff_prior_temporal_learning/representation.py','line':23,'issue':'prepare参考加速度逼近1-1e-7倍限值；VAL Diffusion各出现1个实际步小幅超限。新身份研究响应余量，不能放宽本run物理门。'},
   {'severity':'reporting_scope','file':'ff_prior_temporal_learning/learning.py','line':82,'issue':'consumed只计base minibatch，每TRAIN=3000不包括Diffusion额外187/188条完整采样监督；总正式目标暴露25500对24000。'}],
 'routing_action':'confirm_C1_narrow_C2_continue_authorized_P1_P4','can_proceed_P1_P4':True,'scientific_advantage_required_to_proceed':False,'long_term_goal_complete':False,
 'validation':{'saved_evidence_checks':raw['checks_total'],'saved_evidence_failures':raw['checks_failed'],'additional_core_manifest_files_verified':278,'new_physics_steps':0,'new_geometry_queries':0,'optimizer_steps_by_reviewer':0,'resume_calls_by_reviewer':0},
 'trace_path':'paper/review-traces/experiment-result-to-claim/20261022_temporal_learning_run01/'
}
assert verdict['claim_supported'] in ['yes','partial','no'] and raw['checks_failed']==0
for filename,value in [('verdict.json',verdict),('raw_audit.json',raw),('inputs.json',inputs)]:
    (TRACE/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
review=(TRACE/'review.md').read_text(encoding='utf-8');(TRACE/'response.md').write_text(review,encoding='utf-8')
(TRACE/'routing.md').write_text('结论：partial / high；C1 yes，C2统一优势 no（尚未建立）。论文结论 provisional。\n\n路由：确认固定DEV工程闭合、收窄科学主张，按既有授权继续P1/P4；不得宣称长期目标完成。后续修补完整重放门、prepare实际响应余量时使用新冻结身份，本run不可重写或追加追认。\n',encoding='utf-8')
(TRACE/'findings.md').write_text('本地 verdict 记录（按委托不编辑主 findings）：temporal_learning_001，partial/high，C1 yes，C2统一优势未建立。\n\n关键负结果：VAL各0/4；确定性碰撞截断4条，Diffusion完整4条但世界误差与实际导数失败。关键正结果：固定DEV八条独特执行全部合格、16全状态重放一致。路由为继续授权P1/P4；五障碍/连续组合未完成。详见review.md、verdict.json。\n',encoding='utf-8')
(TRACE/'handoff.md').write_text('独立 reviewer 已完成新run审查，保存 prompt、原始 review/response、结构化 verdict、raw_audit、全部 checks、输入hash/timestamps和辅助脚本。\n\n1,334项保存证据核验0不一致；另核278文件core manifest与主代理一次completed resume收据，不重复resume。未改冻结源码、主报告或主findings，未提交。历史物理执行只核身份/哈希；只读12条合格标签和原DEV身份。没有新物理步、几何查询、optimizer.step或实验启动。\n\n剩余限制：C2科学优势未建立；VAL均失败；完整五障碍/连续组合未完成；论文claim audit缺失。未来版本应将fullerr纳入门，并研究prepare运动余量；本次fullerr均为0。主代理负责统一报告、tracker及本地提交。\n',encoding='utf-8')
print(json.dumps({'verdict':verdict['verdict'],'confidence':verdict['confidence'],'C1':verdict['C1']['claim_supported'],'C2':verdict['C2']['claim_supported'],'checks':raw['checks_total'],'failures':raw['checks_failed'],'manifest_files':278,'files':sorted(p.name for p in TRACE.iterdir())},ensure_ascii=False))
