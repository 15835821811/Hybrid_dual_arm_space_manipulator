"""Create the factual closeout after independent review and no-op resume pass."""
from pathlib import Path
from datetime import datetime
import json
from ff_prior.storage import read_json, file_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
EVIDENCE = ROOT/'ff_prior_world_endpoint/evidence/completed_001'
TRACE = ROOT/'paper/review-traces/experiment-result-to-claim/20261017_world_endpoint_run01'


def main():
    s = read_json(OUT/'summary.json')
    d = read_json(EVIDENCE/'posthoc_diagnostics.json')
    review = read_json(TRACE/'verdict.json')
    resume = read_json(EVIDENCE/'resume_verification.json')
    assert s['status'] == 'FINISHED' and s['independent_audit_passed']
    assert resume['status'] == 'PASS' and resume['before'] == resume['after']
    assert review['claim_supported'] == 'partial'
    assert not s['eligible_for_TEST_protocol_review'] and not s['TEST_executed']
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    text = r'''# 真实世界终点监督闭环：实现和执行通过，学习收益退化

2026-10-17，world_endpoint_smoke_001。正式作业已FINISHED并退出，耗时4678.922秒（77.98分钟）；其中训练循环含真实终点梯度约1042.000秒（17.37分钟）。源码7bac71d5，启动记录551f13d6；上一轮报告和提交收尾b5921b54保持。本报告统计当前实际运行，没有使用旧结果冒充新实验。

**本轮完成了真实非线性世界终点监督的代码接入、少量固定TRAIN补样和同预算smoke，但没有改善指定目标生成能力，反而出现学习退化。** 新权重只保留为实验结果，不推荐替代冻结13774模型。64条新增修正轨迹通过原准入门，可以保留；数据合格不等于训练策略有效。原1024槽扩库继续未启动，最终TEST因三项预注册前置门未满足而BLOCKED。

## 四类结论分别判定

|层次|本轮实际结论|
|---|---|
|软件与训练实现|8/8真实测试；6000实际更新、120次当前预测处102维非线性终点梯度、权重改变及重载校验通过|
|数据资格|16固定TRAIN目标新增64/64修正，覆盖8空场景和8球场景；所有原模型和目标门通过|
|完整执行流程|四方法各180/180合格、36/36真实MuJoCo67通道力矩执行和完整初态独立重放；C1通过|
|学习目标改善及Diffusion收益|TRAIN独立噪声探针恶化、开发提案精度恶化、裸提案仍0/12；C2未通过|

这里的真实执行指MuJoCo力矩驱动动力学执行，不是实体硬件实验。训练中的耦合前向仍是模型计算，不升级数据执行等级。

## 数据和实际训练

按运行前hash规则从已有TRAIN冻结16个世界目标，初态、目标、场景均不重标。64次修正全部准入，最大双臂世界位置误差1.772588mm，最大姿态误差0.059377°；最小控制点RMS差0.006483rad，高于原0.005rad变体门。每目标预选1条执行、失败不替换，16/16通过，共168000物理步（含独立重放）。

合并13838TRAIN，原VAL48保持。新增48条MODEL_VALID、16条EXECUTION_VERIFIED；全TRAIN为13615 MODEL_VALID和223 EXECUTION_VERIFIED。其余未执行行没有被抽查或训练批量升级。后处理逐条重建了64条全部样条数组（含位置、速度、加速度视图），并核对836维完整初态hash，全部一致；这一步没有新增物理调用。

全部13838行参与6000×64=384000次常规样本消费；新64行各37–39次。均匀行流192000次、均匀初态流192000次，1578个初态组各121–122次。VAL梯度0、无VAL或DEV检查点选择。固定最终模型，未从中间检查点挑选最佳DEV。training_index.json中actual_training_consumed=false是生成阶段的前置快照，实际消费由training.json、最终优化器、逐行计数和抽样重建证明；本报告不改写冻结快照。

新增世界终点项每50次更新计算1个当前预测，共120次；每次完整102维两侧差分加半步、混合方向及256/512收敛检查，合计25440次耦合传播。全部120次检查通过，最大半步相对误差9.70e-7，混合方向误差3.57e-5，256/512世界位置差1.67e-8m。每次扰动终点矩阵、损失、梯度、输入权重hash均保存。

权重hash从c4d599a37296ad27a267c3f38eb9a18113370ca689bb45cd44942846f636928e变为83f43aedade7a8ffb41ef02146322440044fa9088eced6bac9ce184115247154；最终检查点SHA256为1870205384f37718bd371574facef2854fff4f1e6b8667d6c357343e17dc4473。

## 原冻结任务、相同候选预算结果

原DEV4＋上一轮已用8开发任务，原三采样种子、四方法、每任务5候选，共720条候选。八任务在本轮明确为REUSED_DEV；原路径保留NEW_VAL字样不表示它们仍是未见验证。四方法相同候选资源上限，共同目标保持碰撞修正、同一选择规则，无失败替补。

|方法|完整合格/180|执行重放/36|共同预处理后、物理修正前位姿门/180|固定裸提案指定目标/12|固定裸提案物理门/12|
|---|---:|---:|---:|---:|---:|
|强样条|180|36|45|3|12|
|冻结13774 Diffusion|180|36|15|0|12|
|同数据更新条件均值|180|36|22|1|12|
|新世界终点Diffusion|180|36|5|0|12|

原DEV各60/60、12/12执行重放；已用8开发任务各120/120、24/24。全部144比较执行及96固定raw/预处理诊断完成，候选ERROR为0、无未决调用。比较物理步1512000，加16TRAIN见证合计1680000；全部160轨迹的最小已存状态间隙2.276000mm、最大hold位置误差3.591158mm，重放报告误差0。该间隙统计不宣称连续时间严格几何证明。

新模型固定裸提案最大臂位置误差中位12.695mm，旧模型8.662mm；新范围5.835–36.961mm。新位置门0/12、姿态门3/12、全部目标门0/12。旧模型位置门1/12、姿态门3/12、全部目标门0/12。完整180/180由共同物理修正后获得，不能归为神经网络独立命中目标。

REUSED_DEV8按每任务15候选修正前标准化残差中位数配对，比例大于1表示新模型较差：

|参照|任务比值中位数|几何均值比95% bootstrap区间|新模型更好任务|
|---|---:|---|---:|
|冻结13774模型|1.239667|[1.124355, 1.266512]|0/8|
|更新条件均值|1.120066|[1.079481, 1.689070]|1/8|
|强样条|1.197384|[0.990510, 3.201515]|2/8|

新模型相对旧模型8/8任务变差，任务比值中位约增加23.97%。这是固定8开发任务、一个训练种子内的结果，三采样种子不能替代多训练种子；不外推为所有任务上Diffusion无效。查询与时间只保留预算核算，不作为本轮收益或扩库门。

## TRAIN终点监督为什么没有带来改善

固定16TRAIN目标×t500/999，共32例独立噪声探针，没有梯度或检查点选择：

|噪声档|改善案例|训练后/前真实终点损失配对中位比|最大臂位置误差中位：前→后|
|---|---:|---:|---|
|t500|2/16|4.1263|9.713→18.104mm|
|t999|3/16|5.2595|8.799→17.143mm|
|合计|5/32|4.6086|9.713→17.654mm|

整体位姿门5/32→1/32。4.6086是逐案例比值的中位数，不是两组损失中位数相除，也不是DEV指标。因此当前问题已经出现在训练任务的独立噪声诊断中，不能只解释成未见任务泛化困难。

已确认的训练设计变化与限制：

1. **新实现更换了目标监督方式，同时降低了覆盖密度。** 旧训练每批64例都带0.05权重的局部G目标项；本轮移除了该项，5880步只有v去噪损失，仅120步增加一个真实终点样本。真实计算更准确，并不自动保证这种稀疏替换比原训练有效。没有做相同数据、种子和更新数的局部G保留/移除对照，不能把退化单独归因于真实终点项。
2. **世界监督的覆盖较窄，而且变体与噪声档绑定。** 只有16目标各7–8次真实终点事件；64变体中v0/v2只用t500、v1/v3只用t999，没有单条变体同时覆盖两个档位。所有新行确实参与常规去噪训练，但不代表它们得到了密集、跨噪声的目标监督。
3. **120次世界事件均触发混合参数梯度裁剪。** 裁剪前范数中位52.649、最大1267.304，阈值1。它提示需要检查尺度与优化步，但当前日志只有vMSE＋世界项的总范数，未记录两项独立参数梯度、夹角或逐步真实损失变化，不能直接宣称某一项主导、梯度冲突或裁剪就是根因。
4. **训练探针与完整DDIM采样仍有差异。** 世界损失作用于教师轨迹加噪后的单次清洁预测，主要在t500/999；实际生成从随机噪声走16步DDIM，在最后一步接共同修正。直接监督真实终点解决了前向损失定义的问题，但尚未监督完整采样分布和多步误差积累。

第1、2项是源码和事件记录可直接确认的设计限制；第3、4项对应的退化机制还需对照实验证明。新增数据、额外6000更新、新随机种子、移除局部项和新增世界项同时变化，本轮不是单因素因果实验。

本轮DDIM分解恒等式残差最大4.72e-15；120次梯度数值检查和所有最终域/范围门通过，没有证据把这次学习退化归因于DDIM公式错误、已接受候选的浮点越界或微重力传播失真。stderr确实保留了两条SciPy优化器内部试探点越界并裁剪的RuntimeWarning，不能声称全流程从未有临时越界；这些警告没有使最终已接受样本越过原物理门。当前证据也不能证明模型在所有状态都完全准确。

## 扩库和最终TEST决定

|预注册前置门|实测|决定|
|---|---|---|
|完整稳定、数据、原始审计|通过|PASS|
|TRAIN真实终点损失配对中位比≤0.8|4.6086|BLOCKED|
|相对冻结13774模型的开发质量门|中位1.2397、8/8更差|BLOCKED|
|固定裸提案≥4/12目标合格|0/12|BLOCKED|
|固定裸提案12/12物理门|12/12|PASS|

本次不启动原1024槽扩库、pilot或最终TEST。64合格样本保留，旧13774模型继续作为开发参照，新模型不提升为推荐替代。这里不自动回退Git或删除新权重。最终TEST不能用于补救开发失败或挑选模型。

下一步优先做一个TRAIN-only短程优化诊断，而非直接重复长训练或扩库：从同一旧模型、同数据/同噪声/同更新数，比较保留原局部G项的控制组与“保留局部项＋真实终点监督”的组；对齐每个目标、变体和噪声档，逐项记录参数梯度范数与夹角、裁剪前后及优化步前后真实终点变化。先证明在固定TRAIN诊断上可靠下降，再冻结同预算DEV复核；DEV选定后还需要新的独立验证来支持泛化。该建议尚未启动，具体预算和准入门应另行预注册。原有物理门全部保持。

## 审查、恢复和证据

experiment-result-to-claim独立审查：整体partial，C1 yes，C2 no，TRAIN改善no，最终TEST不就绪。审查重算120×102维梯度对应24480个扰动损失，核对720候选、96诊断、全部160条已存执行轨迹的控制哈希、形状、末端保持误差和间隙；217源码及执行zip一致。正式worker原始审计还核对了继承来源、模型、用户修改和旧分支。独立评审没有重新跑训练、网络推理或动力学。

审查限制：未保留每个梯度事件独立的完整中间权重，因此当前预测绑定依靠冻结源码、输入hash和事件记录，未逐事件重演网络前向；独立审查也没有再次重跑全部动力学或所有几何对。无paper/PAPER_CLAIM_AUDIT.json，论文写作层结论标为provisional，不影响这里的实际统计。多轮partial后，当前可支持的主张应收窄为“合格数据生成与共同物理修正后的混合流程稳定”，不能宣称神经网络目标生成优越性。

完成后实际执行resume返回NO_WORK_ALREADY_COMPLETE，status/summary/ledger/launcher SHA全部未变、没有新worker或重复训练/物理执行。首次收尾观察脚本的Windows编码解码失败已修复并重试核验，属于观察脚本问题，不是正式实验失败。实际记录见resume_verification.json。缓存记录边界峰值5.639GiB，不声称连续内存峰值。

完整初态、17→67映射、自由漂浮、全机几何、仅不穿透及其余目标/保持/跟踪/动量/数值/独立重放门保持。旧数据、旧结果、用户修改、main和旧分支保留，无强推或大数据上传。长期研究目标尚未完成。

主要证据：

- 独立输出：artifacts/v7_1/world_endpoint_smoke_001（run、ledger、training、world_gradients、candidates、executions、proposal_diagnostics）。
- 版本化小型证据：ff_prior_world_endpoint/evidence/completed_001。
- 只读后处理：ff_prior_world_endpoint_tools/summarize_completed.py；没有新物理/网络调用。
- 独立评审：paper/review-traces/experiment-result-to-claim/20261017_world_endpoint_run01/response.md、verdict.json、verification_evidence.json。

```powershell
.\ff_prior_world_endpoint\job.ps1 status
.\ff_prior_world_endpoint\job.ps1 report
.\ff_prior_world_endpoint\job.ps1 resume
```
'''
    (ROOT/'ff_prior_world_endpoint/docs/ANALYSIS.md').write_text(text, encoding='utf-8')
    decision = {'run_id':OUT.name, 'data':'QUALIFIED_KEEP_64', 'new_model':'RETAIN_EXPERIMENTAL_NOT_RECOMMENDED',
                'development_reference':'qualified_origin_learn_002', 'C1':True, 'C2':False,
                'TRAIN_endpoint_improved':False, 'broad1024_expansion':'DEFERRED_NOT_LAUNCHED',
                'final_TEST':'BLOCKED_BY_PRE_REGISTERED_SMOKE_GATES', 'failed_TEST_gates':[k for k,v in s['conditional_TEST_gates'].items() if not v],
                'independent_review':review['claim_supported'], 'review_sha256':file_digest(TRACE/'verdict.json'),
                'existing_models_or_data_modified':False, 'full_goal_complete':False}
    (EVIDENCE/'release_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    entry='''## 2026-10-17 最新闭环结论：真实世界终点监督实现通过，学习收益退化

world_endpoint_smoke_001已FINISHED并退出，4678.922秒（77.98分钟）。16固定TRAIN世界目标新增64/64合格修正、16/16预选真实67力矩执行及独立重放；累计13838TRAIN、原VAL48。全部TRAIN实际参加6000更新/384000消费，120次当前预测处全102维真实终点梯度，25440耦合传播，权重重载和原始审计PASS。原DEV4+已用8开发任务，原三种子/四方法/同候选预算；各180/180合格、36/36执行重放，720候选/144比较轨迹/96诊断完成。

实现和完整物理流程C1通过；TRAIN终点改善和神经质量C2不通过。32固定TRAIN独立噪声探针仅5改善，配对损失中位比4.608628；新/旧修正前位姿门5/180对15/180，固定raw均0/12，新最大臂位置误差中位12.695mm对旧8.662mm。已用8开发任务的新/旧任务残差比中位1.239667、geo95[1.124355,1.266512]，8/8更差。完整成功依赖共同物理修正，不可归为网络独立目标命中。

新方案移除了原每batch局部G监督，改为每50步一个真实终点事件；64变体各只绑定一个噪声档，120混合梯度均裁剪。它们是已确认的设计限制及待验证退化机制，尚无单因素因果证据。差分、传播收敛、最终数值/物理门通过；保留两条SciPy内部试探点clipping警告，不归因为微重力或DDIM错误。

独立experiment-result-to-claim审查partial（C1 yes/C2 no）；完成resume已验证NO_WORK_ALREADY_COMPLETE、四项核心文件SHA不变。新权重保留为实验候选，不推荐替代旧13774开发基准；64合格数据保留。原1024槽扩库未启动，最终TEST三项预注册门未过，明确BLOCKED；不运行pilot，不放宽物理门。最高待研究项改为TRAIN-only短程目标项保留/叠加与优化尺度诊断，先证明真实终点可靠下降，再另冻DEV复核；尚未启动下一实验。长期目标未完成。报告ff_prior_world_endpoint/docs/ANALYSIS.md；终端`.\\ff_prior_world_endpoint\\job.ps1 status|report|resume`。以下为历史记录。

'''
    for name in ('DIFFUSION_GOAL_CURRENT.md','ff_prior_feasible/docs/CORE_GOAL.md'):
        p=ROOT/name; old=p.read_text(encoding='utf-8'); head,rest=old.split('\n',1)
        p.write_text(head+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8')
    tracker='''# 世界终点监督闭环已完成，收益未成立

|阶段|状态|实际证据和边界|
|---|---|---|
|上一轮收尾|FINISHED b5921b54|旧13774模型和结果保持|
|软件/来源预检|PASS|8/8真实测试；176479来源/217Python绑定|
|固定TRAIN修正|PASS|16目标64/64准入；16/16预选真实执行重放；合并13838TRAIN/原VAL48|
|真实世界终点训练|实现PASS/改善FAIL|6000更新/全部TRAIN参与、120全102梯度；TRAIN独立噪声损失中位比4.608628|
|冻结smoke完整流程|C1 PASS|四方法各180/180和36/36；720候选/144比较轨迹/96诊断|
|神经质量|C2 FAIL|已用8开发任务8/8比旧模型更差，任务中位比1.239667；raw0/12|
|原始审计/独立评审|PASS/partial|完整保存证据一致；不支持目标质量提升和神经优越性|
|完成恢复|PASS|NO_WORK_ALREADY_COMPLETE，核心四文件SHA不变，无重复工作|
|原1024槽扩库|PREPARED_NOT_LAUNCHED|按最新条件不启动；64合格数据保留|
|最终TEST|BLOCKED|TRAIN改善、对旧模型质量、raw≥4/12三门未通过|
|下一实验|NOT_STARTED|建议TRAIN-only目标项保留/叠加、梯度尺度和跨噪声配对诊断；另冻预算后实施|

正式实验耗时77.98分钟；主报告ff_prior_world_endpoint/docs/ANALYSIS.md。新模型只作实验候选，保留旧开发基准，长期目标未完成。仅不穿透及其他物理门、旧结果/分支/用户修改保持；无pilot或最终TEST。
'''
    for name in (f'paper/EXPERIMENT_TRACKER_{stamp}.md','paper/EXPERIMENT_TRACKER.md'):
        (ROOT/name).write_text(tracker,encoding='utf-8')
    for name in ('findings.md','EXPERIMENT_RESULT_TO_CLAIM.md','MANIFEST.md'):
        with (ROOT/name).open('a',encoding='utf-8') as f:
            f.write(f'\n## {stamp} world_endpoint_smoke_001 closeout\n\nClaim verdict partial: C1 stable implementation/data/native flow; C2 and TRAIN endpoint improvement no. 64 qualified corrections; 13838 TRAIN/6000 updates/120 true gradients. TRAIN paired ratio4.608628, reusedDEV8 new/old1.239667, raw0/12. New model experimental only; no1024 expansion orTEST. Review trace20261017_world_endpoint_run01; reportff_prior_world_endpoint/docs/ANALYSIS.md. Missing causal evidence: old local G removed, sparse world coverage, all world mixed gradients clipped, no separate-gradient logs, one training seed. Route to TRAIN-only optimization diagnosis, not blind expansion. Root append-only notes not staged wholesale.\n')
    handoff=f'''# Executor closeout

{stamp}: Independent reviewer partial; retained C1 implementation/data/full-flow claim, rejected TRAIN endpoint improvement and C2 neural superiority. Raw outputs untouched; final report and latest goal/tracker updated. Posthoc statistics and64 spline/full-state checks add no network or physics. Actual completed resume PASS with unchanged status/summary/ledger/launcher hashes. First observer UTF-8 decoding failed, corrected observer retry verified no-op; no experiment rerun.

No1024 expansion, pilot orTEST. ConditionalTEST fails three frozen gates. Keep64 qualified data, keep new weights experimental, retain old13774 development reference; no file rollback. Next short TRAIN-only objective/optimization diagnostic proposed but not started. No paper claim audit, downstream writing provisional. User edits and unrelated files not staged. Automation is to be paused after this closeout delivery; full research goal remains incomplete.

Evidence: ff_prior_world_endpoint/evidence/completed_001 and reviewer verification_evidence.json. No raw data uploaded. Independent review limitations and SciPy internal clipping warnings preserved in final report.
'''
    (TRACE/'handoff.md').write_text(handoff,encoding='utf-8')
    print(json.dumps({'report':'ff_prior_world_endpoint/docs/ANALYSIS.md','tracker_stamp':stamp,'release':decision},ensure_ascii=False))


if __name__=='__main__':
    main()
