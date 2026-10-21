"""Record bounded experiment outcome; never changes frozen runtime evidence."""
from datetime import datetime
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_full_control.protocol import ROOT, PACKAGE


def main():
    trace=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_full_control_run01'
    assert (trace/'response.md').exists() and (trace/'verdict.json').exists()
    resume=read_json(PACKAGE/'evidence/completed/completed_resume.json')
    assert resume['core_files_unchanged'] and resume['frozen_identity_unchanged'] and all(resume['user_files_unchanged'].values())
    stamp='20261021_162903'
    tracker='''# 完整323控制有界实验收尾

run full_control_bounded_001；源码f788a689，启动aa3a0d2e。报告ff_prior_full_control/docs/ANALYSIS.md。

|阶段|实测状态|证据与边界|
|---|---|---|
|M0|完成|7软件测试；165源码、15411输入绑定；14222 TRAIN/48 VAL；DEV不入TRAIN|
|M1|PASS|6/6实际float32目标重建；3/3预选原生力矩执行和独立重放；仅该范围|
|M2|实际完成|6000更新、384000抽样、全部14222行、1200完整16步教师批次；不是世界终点梯度|
|M3|完成，学习收益NO|新0/96 vs旧24/96；位置P95 140.414 vs46.225mm|
|M4 原DEV|完成，工程门未通过|新15/180；13/36执行单元成功，23无候选；旧各180/180和36/36为复用|
|M4 三/五障碍|完成，未通过|新各0/5、0/1执行；检测穿透分别4/5、5/5，均密度不足|
|M5|完成|原始审计PASS；新独立review C1限定YES/C2 NO/总体PARTIAL；一次resume 596文件hash不变；报告和提交收尾|

本轮新执行16/16实际通过＝3 TRAIN＋13 DEV；原生91500步＋独立重放91500步。41个新预选单元另25个NO_CANDIDATE，不藏分母。旧114个执行单元中110实际通过，仅历史复用。

长期静态避障抓捕→动态航点目标尚未达成。保留旧aligned基线；建议下一轮先验证函数保持迁移与全维时间规律，TRAIN先行，后续一次候选和同预算DEV；尚未启动。无扩库、pilot或TEST，无物理放宽。未证明Diffusion相对优势。
'''
    (ROOT/'paper'/f'EXPERIMENT_TRACKER_{stamp}.md').write_text(tracker,encoding='utf-8')
    (ROOT/'paper/EXPERIMENT_TRACKER.md').write_text(tracker,encoding='utf-8')
    goal='''# 当前执行目标：完整19×17控制有界闭环已收尾，下一实验未启动

用户授权的full_control_bounded_001已完成独立TRAIN重建、一次训练、完整无引导与原冻结DEV同预算对照、原始证据核对、全新独立审查和中文报告。软件/实验流程完成不等于新模型性能通过，长期研究目标仍未达成。

- M1：独立TRAIN实际float32目标重建6/6、3/3原生67力矩执行与独立重放通过；未用07 DEV教师。
- M2：14222 TRAIN/48 VAL；一个323维MLP从头初始化，6000实际更新、1200完整16步教师批次。全部TRAIN参与，VAL不进梯度或选模型。
- M3：新完整无引导0/96，旧24/96；位置误差尾部变差。
- M4：原DEV新15/180、13/36执行单元；165条候选全部仅因2mm几何密度不足，180条均通过目标门。新三/五障碍均0/5，存在真实检测穿透，未执行。工程门未过，科学收益NO。
- M5：原始审计PASS；新的独立审查C1限定YES、C2 NO、总体PARTIAL；完成resume仅核一次，596文件hash不变。报告ff_prior_full_control/docs/ANALYSIS.md，版本tracker paper/EXPERIMENT_TRACKER_20261021_162903.md。

当前保留旧aligned基线，不把full323替换上线。下一步仅建议：保持完整323表达能力，先验证旧低频功能保持迁移＋受控全维细节/可逆多尺度时间表示，以及TRAIN教师的完整采样可学性；合格后一个有界候选再做同预算DEV。需新协议，未自动运行。

长期目标仍为刚性臂静态避障抵达抓捕接近目标、连续体保持指定世界目标，再接动态不规则航点。现有实验不是硬件抓取或接触闭锁证明。

所有旧模型/数据/结果和用户修改保留。DEV不入TRAIN，不放宽仅不穿透、5mm/1度、2mm几何、完整初态、自由漂浮、全机几何、17→67、运动/力矩/重放门。不新增数据、不扩库、不pilot、不TEST。本轮后台接续收尾后暂停，不因负结果重训重测。

终端：.\\ff_prior_full_control\\job.ps1 status|report|resume。源f788a689，启动aa3a0d2e；后续收尾提交以git日志为准。
'''
    (ROOT/'DIFFUSION_GOAL_CURRENT.md').write_text(goal,encoding='utf-8')
    note='''
## 2026-10-21 full_control_bounded_001 完整323控制闭环

意图：独立TRAIN资格保真后，检验完整控制学习与冻结DEV相对收益。新的独立experiment-result-to-claim：C1 yes（仅6例重建/3例新执行）、C2 no、总体partial；confidence high（限定证据），paper claim audit unavailable，论文写作结论provisional。6/6重建及3/3执行支持完整表示/prepare在该范围可用；新无引导0/96 vs旧24/96，DEV15/180及13/36 vs旧180/180及36/36，不支持学习或Diffusion优势。DEV165拒绝仅密度不足；三/五障碍实际穿透仍存在。6000真实训练和四保存更新原始重建通过，不能把失败归为未训练。输出/codec/初始化/损失/prepare混合变化，缺少因果隔离、硬件和未触碰TEST证据。下一步pivot到函数保持迁移和全维时间规律，先TRAIN可学性；仅建议，用户本轮禁止追加实验。保留旧基线，不扩库、不放宽门、不TEST。详见ff_prior_full_control/docs/ANALYSIS.md与paper/review-traces/experiment-result-to-claim/20261021_full_control_run01/。
'''
    for name in ['findings.md','EXPERIMENT_RESULT_TO_CLAIM.md']:
        p=ROOT/name;old=p.read_text(encoding='utf-8') if p.exists() else ''
        if '## 2026-10-21 full_control_bounded_001 完整323控制闭环' not in old:p.write_text(old+'\n'+note,encoding='utf-8')
    (PACKAGE/'docs/CLAIM_NOTES.md').write_text(note,encoding='utf-8')
    manifest=ROOT/'MANIFEST.md';old=manifest.read_text(encoding='utf-8') if manifest.exists() else ''
    line='\n- full_control_bounded_001：ff_prior_full_control/docs/ANALYSIS.md；review-traces/experiment-result-to-claim/20261021_full_control_run01；C1限定YES/C2 NO，完整流程完成但工程/收益未通过。\n'
    if line.strip() not in old:manifest.write_text(old+line,encoding='utf-8')
    atomic_json(PACKAGE/'evidence/completed/closeout_requirements.json',dict(run='full_control_bounded_001',
        scope='bounded authorized experiment only, not long-term scientific success',
        requirements={
            'independent_TRAIN_fidelity':'reconstruction/6 saved validation+representation and 3 execution/replay; gate PASS',
            'one_actual_training':'6000/384000 draws/14222 seen/1200 full16 batches; four VJP/AdamW updates independently checked',
            'unguided_and_same_budget_DEV':'96 new unguided,180 DEV+10 application new candidates, exact bound historical reuse',
            'physical_constraints':'original acceptance/identity and all gates unchanged; 165 density failures retained',
            'data_levels':'new 16 actual executions verified only, 25 no-candidate slots retained, no new TRAIN rows',
            'preservation':'completed_resume.json matches 596 raw files, identity and 4 modified user files',
            'fresh_review':'paper/review-traces/experiment-result-to-claim/20261021_full_control_run01',
            'report_goal_tracker':'ANALYSIS.md, DIFFUSION_GOAL_CURRENT.md, versioned tracker',
            'not_run':'expansion/pilot/study/strict_multimodal/TEST',
            'submission':'only task files staged; local commit, no push or data upload'},
        pending_external_actions=['pause completed heartbeat','commit task files','final delivery']))
    print('notes and versioned tracker written')


if __name__=='__main__':main()
