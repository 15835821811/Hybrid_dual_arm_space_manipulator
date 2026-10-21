"""Versioned local closeout notes; no experiment is executed."""
from ff_prior_control_transfer.protocol import ROOT, PACKAGE

notes = '''# control_transfer_bounded_001：结果与主张

本轮源码df13c4e1，后台135.03秒；保留完整323控制、旧低102功能、221受控细节及资格保持prepare。
独立review：paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/。
结论partial：16/16迁移等价、6/6教师重建、3/3本轮原生执行与独立重放；256次临时TRAIN更新将全门合格从1/16升至6/16，Q中位下降79.63%，全部16条通过模型物理门。
差距：未达到12/16，原唯一合格样本退化；正式6000步候选、VAL与DEV均BLOCKED且未运行。C2相对收益未评价，不可称新候选无效。临时权重不进入候选，诊断6条未执行，不能升级执行等级。
退化样本原教师/目标/初态与缓存身份一致，控制RMSE下降但世界误差增大。当前证据支持控制MSE与逐条世界终点验收不一致，未证明微重力建模错误。
下一建议：在完整16步采样输出上加入真实世界终点方向监督/约束与原合格样本保资格约束；先冻结导数和预算进行有界TRAIN诊断。此处只建议，不自动追加试验。
保留旧结果、用户修改，DEV未入TRAIN；不扩库、不放宽物理门、不TEST。论文级结论provisional，参见独立verdict的paper claim audit状态。长期目标未完成。
'''
(PACKAGE/'docs/CLAIM_NOTES.md').write_text(notes,encoding='utf-8')
for name in ('findings.md','EXPERIMENT_RESULT_TO_CLAIM.md'):
    p=ROOT/name
    text=p.read_text(encoding='utf-8') if p.exists() else ''
    if '# control_transfer_bounded_001：结果与主张' not in text:
        p.write_text(text+'\n\n'+notes,encoding='utf-8')
goal='''# 当前执行目标：完整323迁移与TRAIN可学性——本轮完成收尾

用户2026-10-21授权的有界实验已执行，run artifacts/v7_1/control_transfer_bounded_001，源码df13c4e1，启动7d32e4e2。完整协议保留在ff_prior_control_transfer/docs/EXPERIMENT_PLAN.md，分析见同目录ANALYSIS.md。

- 完整323输出与资格保持prepare已实现；16条旧模型完整采样迁移等价，6/6独立TRAIN重建、3/3原生67力矩执行与独立重放通过。
- 256实际临时更新完成；全门1/16→6/16，Q中位下降79.63%，时间细节受控；未达12/16且原合格样本退化，TRAIN综合门未过。
- 正式候选6000更新、VAL无引导、原冻结DEV及应用对照均BLOCKED，未启动。不能把未训练候选写成收益失败或成功。
- 原始审计PASS；全新独立审查、中文报告、版本tracker和一次完成resume收尾。452原始文件清单/hash不变。接续在交付后暂停。

本轮条件目标结束，长期静态避障抓捕→动态航点与Diffusion相对收益仍未完成。下一协议尚未启动：保留本轮迁移和受控细节，在完整采样末端增加世界终点监督与原合格保护，先做一次有界TRAIN方向验证，再决定正式候选。已完成迁移与重建证据可按身份复用，不反复重跑。

所有旧模型、数据、结果及用户修改保留。DEV不混TRAIN，不放宽仅不穿透、5mm/1度、2mm、完整初态、自由漂浮、全机几何、17→67、运动/力矩/重放门。不扩库、pilot或TEST，不自动追加实验。

终端：.\\ff_prior_control_transfer\\job.ps1 status|report|resume。resume已核一次不会重复工作，正常无需再调用。
'''
(ROOT/'DIFFUSION_GOAL_CURRENT.md').write_text(goal,encoding='utf-8')
tracker='''# 完整323功能迁移有界实验：终态tracker

2026-10-21；源码df13c4e1；run control_transfer_bounded_001；135.03秒。

|阶段|状态|实测证据与边界|
|---|---|---|
|M0 软件/输入冻结|DONE|8/8软件测试；170源码/15412输入绑定；DEV不入TRAIN|
|M1 功能与教师重建|PASS|16/16旧模型等价；6/6完整门重建；3/3新执行与重放|
|M1 临时TRAIN可学性|DONE / GATE BLOCKED|256更新；1/16→6/16<12/16，原唯一合格样本退化；Q中位下降79.63%，模型16/16|
|M2 唯一正式候选|BLOCKED / NOT_RUN|依赖M1；无6000更新、无新候选|
|M3 VAL无引导|BLOCKED / NOT_RUN|不能给新候选成功率|
|M4 冻结DEV及应用|BLOCKED / NOT_RUN|旧对照仅绑定，不冒充本轮成果|
|M5 收尾|DONE|原始审计PASS、新独立review partial、中文报告、完成resume452文件不变、目标更新与本地提交|

报告ff_prior_control_transfer/docs/ANALYSIS.md；独立review paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/。C2相对收益not_evaluated。论文级结论provisional，具体见verdict。下一阶段仅提出完整采样世界终点监督和原合格保护，不自动启动。
本轮无新增数据、不扩库、不放宽物理门、不TEST；长期科学目标未完成。
'''
for name in ('EXPERIMENT_TRACKER_20261021_165617.md','EXPERIMENT_TRACKER.md'):
    (ROOT/'paper'/name).write_text(tracker,encoding='utf-8')
print('Closeout notes saved; no experiments run')
