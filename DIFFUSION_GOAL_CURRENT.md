# 当前执行目标：完整323功能保持迁移与受控时间细节

用户2026-10-21授权；原full_control_bounded_001已fbf9e882收尾，不重跑。长期静态避障抓捕→动态航点不变。

唯一run artifacts/v7_1/control_transfer_bounded_001；协议ff_prior_control_transfer/docs/EXPERIMENT_PLAN.md；源码包ff_prior_control_transfer。

1. 固定旧aligned低102网络/codec，正交补全19时间维、221维细节clean头零输出初始化；完整323能力保留。旧102噪声流保持，细节独立流。prepare资格保持，不强制平滑。
2. 6条独立TRAIN实际float32重建与3预选真实力矩执行重放；8固定TRAIN×2种子完整16步无引导功能保持≤1e-12rad。
3. 仅256步临时TRAIN诊断；固定教师损失降10%、至少12/16全门合格且原合格不丢失、Q中位降10%、时间差分RMS受控，全部门过才M2。临时权重不进入候选。
4. 唯一候选从相同迁移初态开始，14222TRAIN/48VAL、6000实际更新、1200完整16步教师批次；无世界非线性梯度，VAL不拟合/选模型。
5. 96无引导；原DEV180新候选/36执行单元＋原三/五障碍10/2；同预算，合格旧对照只身份复用。失败保留分母；门失败下游BLOCKED但收尾，不重训追加。
6. 新独立experiment-result-to-claim审查、中文报告、一次完成resume、目标/版本tracker与任务提交；收尾暂停接续。

保留旧模型/数据/结果/用户修改和作业。DEV不混TRAIN，不放宽仅不穿透、5mm/1度、2mm、完整初态、自由漂浮、全机几何、17→67、运动/力矩/重放门；不扩库、pilot或TEST。

2数学库线程/6h/20GiB/33000耦合；本轮只做预冻结一次临时诊断＋最多一次候选。科学收益未预设，工程完成和收益分别报告。

终端：.\ff_prior_control_transfer\job.ps1 status|report|resume。当前预检/源码冻结与后台启动记录见本包evidence。

源码df13c4e1；worker45928后台已启动，heartbeat正常；接续automation ACTIVE。仅读取状态到终态，再做结果审查收尾。
