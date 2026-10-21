"""Serialize the new independent reviewer judgment; no experiment invocation."""
import json
from pathlib import Path
from ff_prior.storage import read_json,file_digest
from ff_prior_control_transfer.protocol import ROOT,OUT
dest=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01'
checks=read_json(dest/'checks.json');assert checks['status']=='PASS'
assert read_json(dest/'native_math.json')['status']=='PASS'
assert read_json(dest/'regression_provenance.json')['status']=='PASS'
prompt='''# Fresh independent paper architect/reviewer task

你是全新的独立 paper architect/reviewer，按 experiment-result-to-claim 技能审查 ff_prior_control_transfer 本轮，不能复用前轮 review。唯一 run 为 artifacts/v7_1/control_transfer_bounded_001，冻结协议 ff_prior_control_transfer/docs/EXPERIMENT_PLAN.md；源码 df13c4e1，启动记录 7d32e4e2。

只读原始数据，不改冻结 root 源码、运行目录或旧文件；严禁额外训练、optimizer.step、物理仿真、旧审计或 resume。允许新包 tools 内写只读审查脚本，新 trace 写 paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/。已有 summary 只能作为待验证线索。

独立核 170 源码/15412 输入绑定、旧低102权重/codec/随机流与221细节 clean 零初始化、完整基和 TRAIN 有界支持、TRAIN/VAL/DEV 隔离。核6个实际float32重建及3个native67/replay数组；16迁移原始控制与重新神经推理的完整旧/新无引导采样等价。核固定8TRAIN×2种子、临时256AdamW、首末实际before/after与gradients，用全16rollout VJP及手写AdamW核首末，不得 optimizer.step。核前后16保存物理预测、prepare、全部门、归一化Q和世界位置/姿态，原已合格样本退化；临时状态不转候选、初态可恢复，M2/M3/M4阻断与账本无未知工作。教师执行不得冒充网络执行。

特别追溯 coverage_train_00288 seed20270213 的源教师、缓存Z、冻结目标、完整初态和原资格，排除已知标签/缓存错配后再解释控制损失与世界误差不一致。检查主执行的中文 ANALYSIS 数字与主张。只给下一阶段建议，不追加。

判断 C1 功能保持/有限 TRAIN 学习与 C2 相对收益；给出 partial/no 与 not_evaluated 的区别，证据边界、missing evidence、claim revision、next experiments 和 confidence。paper claim audit 缺失则 unavailable/provisional。输出 prompt.md、response.md、verdict.json、inputs.json、checks/handoff、ff_prior_control_transfer/docs/CLAIM_VERDICT.md。主执行负责根 findings/tracker、一次完成resume、统计、提交及暂停接续。

Input list: frozen plan and sources/archive; run identity/binding/selection/codec; original aligned model/codec; expand256 cached arrays and every bound record; six reconstruction arrays/predictions; TRAIN_diagnostic before/after checkpoints, first/last updates and gradients, migration NPZ, all 32 controls/predictions/grids/gates; three native/replay/geometry/initial-state packages; ledger/budget/status; original coverage_train_00288 record, arrays, generation proof and complete initial state; final ANALYSIS.md. Exact reviewed paths, SHA256, size and timestamps are in inputs.json.
'''
response='''# 独立结果到主张审查

本轮判定 **C1 = partial，C2 = no（证据状态 not_evaluated）**。正式6000步候选训练、VAL和DEV没有运行，不能把C2写成新模型DEV失败率为零、已失败的相对实验，或已取得泛化收益。原始只读一致性审查 PASS 不改变学习门 BLOCKED。

## 新核查结果

新审查从原文件重新计算，未调用旧审计主程序，未运行模拟、优化器更新或resume。170个冻结源码与executed_source.zip和当前文件一致，15412个绑定输入全部hash一致。TRAIN14222/VAL48家族互斥，TRAIN与冻结DEV/APPLICATION_DEV任务互斥，8个诊断目标与冻结选择规则一致。codec仅重算TRAIN统计；19列满秩，前6列、低102条件中心/尺度及旧低频权重精确保留。新增221维clean头为零；旧102噪声流保留。所有现存TRAIN细节落在有界tanh支持内部；这不保证任意未见绕行或07可达。

6/6实际float32教师重建和保存的世界目标/模型门一致；3/3预选static_detour原生67力矩执行及独立重放通过，原生+重放合计46500步。另重算完整初态摘要、17→67映射、参考位置/速度/加速度、跟踪/流形残差、力矩裁剪、动量峰值及几何扫描清单。它们是既有教师重建的执行证据，网络生成轨迹执行数为0。

重新逐个运行16条旧/新神经无引导完整16步采样，与保存NPZ逐元素相符，新旧raw最大差1.1102230246251565e-16 rad。此数值功能保持仅涵盖固定条件/种子和raw初态；不包含带引导方法，也不声称任意输入数值证明。迁移结构本身解释了零细节下为何保持低频函数。

固定16个条件/种子与检查点索引一致，首末真实更新的独立16步展开VJP和保存裁剪梯度最大误差均0；手写AdamW的权重及一二阶矩最大误差均0。最终所有优化器状态步数256、日志32至256步齐全。首末检查证明保留的两次更新可重现，未重算中间254次更新，不能把端点抽查说成全部256次逐步重新执行。初态seed重建及恢复均与保存hash一致，临时权重未成为正式候选。

前后32个保存预测末端重新提取位置/姿态，逐条核全门、细化网格、几何扫描和prepared控制；raw=prepare全部逐元素相同，无缺失世界误差。合格1/16→6/16，模型门13/16→16/16，位置门2/16→6/16，姿态门3/16→15/16。Q中位9.1097271173→1.8553342844；完整教师损失1.5174149249→0.003363473812；控制二阶差分RMS0.001598701299→0.001372278438 rad。更新后位置P95为13.985222424 mm、最大15.883037661 mm，姿态P95为0.902205102°、最大1.118511770°。剩余10条失败均含位置越过5mm，其中1条同时越过1°。

两道预冻结门明确失败：至少12/16合格（实际6），原已合格逐条保护（唯一原合格样本退化）。总合格数增加不能替代逐条非退化门。

关键样本coverage_train_00288/20270213，从3.188319136 mm、0.724483528°退化至12.369051094 mm、1.118511770°，尽管教师控制RMSE从0.0302999150降至0.00193637553 rad。新溯源核实缓存Z精确等于绑定源NPZ，task/condition/goals和完整初态一致；原教师为MODEL_VALID，execution NOT_RUN，目标来自历史coupled_forward_256_before_independent_validation，原位置误差3.4237e-6 mm。未发现教师/目标缓存错配。该现象说明当前控制误差目标下降不能保证每条非线性世界终点误差下降；没有证据单独归因物理模型错误、方向敏感性或共享多样本梯度。

账本42个完成单元，无inflight/未知工作；实际耦合76、预留266/33000，256次临时更新完整计入；正式learning、unguided、comparison目录不存在。M2/M3/M4均因M1门阻断。已阅读并核对中文ANALYSIS关键数字与边界，hash列入inputs。主执行的一次完成resume不由审查员重复运行。

## 主张判断与路线

**C1 partial，高置信度（限本次证据）**：支持迁移初态在固定TRAIN样本保持旧raw功能、完整表示能保留已合格教师、固定8目标×2种子的有限TRAIN拟合和时间规律改善。C1中“满足预定可学性放行、逐条保持目标资格”不支持。建议表述为“功能保持迁移与有限TRAIN可优化性得到验证，但256步临时诊断未通过预冻结全门资格和原合格保护条件”。

**C2 no，高置信度地判断证据缺失，evaluation_status=not_evaluated**：没有正式候选，没有新VAL96、DEV180/36或APPLICATION_DEV10/2结果。旧aligned/样条/均值绑定备用不构成新B相对试验。不能给新B成功率、相对优势/劣势、泛化收益、Diffusion优越性、硬件或连续时间安全结论。

下一步仅建议：保持本轮迁移和幅度边界，预先冻结一次TRAIN内有界验证，把完整16步最终输出的世界位置/姿态越门误差和原合格样本保护加入明确目标/约束；先验证导数方向与逐条结果，预算、样本、种子和停止门事先写定。不得在本轮追加、增步救场、筛种子或放宽5mm/1°及物理门。只有新方案满足TRAIN放行后，才可按新授权训练唯一预定候选并完成VAL及原DEV同预算对照。当前不需要重复证明323重建或初态迁移，可按身份复用已有证据。

审查限制：只读保存数组和源代码，未重新做碰撞、质量矩阵、运动学或动力学查询；几何距离、部分数值/身份/标签门仍依赖已保存执行与源码证据，不是独立物理实现。没有外部论文主张审计文件，paper_claim_audit=unavailable，任何下游写作判定保持provisional。独立review已完成，不是pending external review。当前路线为记录负门控结果、收尾并暂停接续，补充方向只建议未执行。
'''
verdict={
 'claim_supported':'partial','confidence':'high','run':'control_transfer_bounded_001','review_role':'fresh independent paper architect/reviewer','review_status':'completed','pending_external_review':False,
 'paper_claim_audit':'unavailable','provisional':True,'provisional_reason':'No paper/PAPER_CLAIM_AUDIT.json available',
 'claims':{'C1':{'claim_supported':'partial','evaluation_status':'evaluated_on_fixed_TRAIN_only','confidence':'high','qualification_gate':'BLOCKED','failed_gates':['qualified_at_least_12_of_16','old_qualified_preserved']},'C2':{'claim_supported':'no','evaluation_status':'not_evaluated','confidence':'high','reason':'M2 candidate training, M3 VAL and M4 DEV blocked by M1; no new B comparison exists','new_B_DEV_success_rate':None}},
 'what_results_support':['Fixed 16 raw full-DDIM samples preserve old aligned function at migration initialization','6 float32 teacher reconstructions and 3 preselected teacher native/replay executions pass','Fixed TRAIN diagnostic reduces teacher loss, median Q and temporal roughness after 256 updates','All 16 post-diagnostic outputs meet saved model gates, six meet both model and target gates'],
 'what_results_dont_support':['Frozen learnability qualification and original-qualified preservation','Formal candidate viability or network-generated native execution','New B VAL/DEV success rate, relative benefit or generalization','Arbitrary unseen detour reachability or full guided-method equivalence','Independent physical rerun or continuous-time/hardware safety'],
 'missing_evidence':['A TRAIN gate-passing update rule that preserves originally qualified samples','One formal candidate from restored migration initialization','New full unguided VAL and paired original-budget DEV/APPLICATION_DEV comparison','Paper-to-evidence claim audit'],
 'suggested_claim_revision':'功能保持迁移和有限TRAIN可优化性成立；预冻结可学性门失败，正式训练与泛化对照未评估。',
 'next_experiments_needed':['Recommend only: freeze a bounded TRAIN test of full16 final world-error supervision/constraints and original-qualified protection, validate derivatives and individual outcomes, then conditionally one candidate and paired evaluations under a new protocol.'],
 'routing_action':'record partial C1 and unevaluated C2; close current run and pause continuation; recommendations only, no additional experiment',
 'metrics':checks['checks']['diagnostic_32_raw_physics_records']['aggregates'],
 'evidence_checks':['checks.json','native_math.json','regression_provenance.json'],
 'limitations':['Fresh neural VJP and AdamW checks cover retained first/last updates only','Saved physics arrays and source checked; no new physical queries or simulator execution','Teacher execution must not be relabelled network-generated execution']}
assert not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists()
(dest/'prompt.md').write_text(prompt,encoding='utf8')
(dest/'response.md').write_text(response,encoding='utf8')
(dest/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2),encoding='utf8')
(dest/'handoff.md').write_text('''# Reviewer handoff

New independent review completed. Overall partial; C1 partial; C2 no with evaluation_status not_evaluated. Both frozen failures retained. No source/run/old artifact edits, no optimizer.step, physical simulation, old audit or resume invocation. New scripts only in ff_prior_control_transfer/tools; reports only in this new trace and package docs/CLAIM_VERDICT.md.

Executed fresh_reviewer_check.py (PASS), fresh_reviewer_native_math.py (PASS), fresh_reviewer_provenance.py (PASS). The first attempted CUDA virtualenv import failed due to a torch DLL before any audit; switched to the actual producer .venv CPU runtime. No experimental work occurred from this environment issue. Native-math implementation adapted from prior tool source, but evaluated exclusively new-run arrays; no prior result or verdict reused.

Parent executor owns root findings/EXPERIMENT_RESULT_TO_CLAIM/tracker, final ANALYSIS, completed-resume evidence, commit and pause. Append the exact C1/C2 distinctions and unavailable paper audit to those notes. No additional test or experiment is requested for this current run. If ANALYSIS changes after review, rebind its final hash after checking the wording.

Remaining risk: source-traced saved physics is not an independent simulator; no formal candidate or generalization evidence. Detailed numeric limitations and recommendations are in response.md. Preserve original run summary's pending-review snapshot; link this new trace rather than rewriting raw results.
''',encoding='utf8')
compact='''# 本轮独立主张判定

正式6000步候选训练、VAL和DEV均未运行。**C1 = partial；C2 = no，evaluation_status = not_evaluated**。证据一致性PASS不等于可学性门PASS。独立review已完成；论文claim audit不可用，写作结论provisional。

- 支持：16条迁移初态full16 raw与旧aligned差≤1.11e-16 rad；6/6实际float32教师重建、3/3教师native67与独立重放；固定8TRAIN×2种子的有限拟合改善。
- 256步临时诊断：loss 1.517415→0.00336347，Q中位9.109727→1.855334，粗糙度0.00159870→0.00137228 rad，模型门13/16→16/16，全门1/16→6/16。
- 两门失败：少于12/16，且唯一原已合格coverage_train_00288/20270213退化（位置3.188→12.369 mm，姿态0.724→1.119°）。已核教师缓存、原始Z、目标和完整初态身份一致；不能用控制RMSE降低代替逐条世界目标资格。
- 不支持：正式候选可用、新B成功率、相对收益、泛化收益或网络生成执行。教师3次执行不得代替网络执行；后者为0。

首末完整16步VJP与手写AdamW权重/矩误差均0；170源码/15412输入绑定、新32预测及3native/replay证据独立核查通过。仅检查保存物理数组和代码，未新做仿真或优化器更新。

下一步仅建议冻结一次TRAIN内世界位置/姿态越门监督及原合格保护方向验证；本轮不追加。新TRAIN门通过之后才考虑唯一正式候选与VAL/DEV同预算对照。当前收尾并暂停接续。

详细证据：paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/ 下 response.md、verdict.json、checks.json、native_math.json、regression_provenance.json、inputs.json。本判定 confidence=high（限现有证据），paper_claim_audit=unavailable；不代表长期科学目标已经完成。
'''
(ROOT/'ff_prior_control_transfer/docs/CLAIM_VERDICT.md').write_text(compact,encoding='utf8')
inputs=read_json(dest/'inputs.json')
for p in (Path(__file__).resolve(),ROOT/'ff_prior_control_transfer/docs/CLAIM_VERDICT.md'):
    inputs[p.relative_to(ROOT).as_posix()]={'sha256':file_digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
(dest/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
print('New reviewer trace and CLAIM_VERDICT written.')
