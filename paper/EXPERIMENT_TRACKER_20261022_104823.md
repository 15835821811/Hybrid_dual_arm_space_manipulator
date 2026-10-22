# Experiment Tracker：全链路持续目标

更新时间2026-10-22T10:48:23。没有新物理结果；P0状态从真实status读取，不凭应用目标卡判结束。

|Run ID|Milestone|Purpose|System|Split|Priority|Status|Notes|
|---|---|---|---|---|---|---|---|
|preserve323_bounded_001|P0|功能保持、分层保护与独立检查|原冻结323迁移|TRAIN/VAL/原DEV|MUST|RUNNING|源d5e1b7c5；原协议不变，收尾后继续长期目标|
|static_fullchain_001（拟）|P1|任务输入到三/五障碍执行|Diffusion+共同修正/结构规划|原APPLICATION_DEV|MUST|QUEUED_IMPLEMENTATION|先冻结准确协议与身份；最多60候选/12执行|
|动态受约束表示及交接（待冻结）|P2|解决1.242倍加速度和状态连续|受约束参考+已有合格控制器|固定4TRAIN/原动态DEV|MUST|QUEUED|旧3/4及2/2仅历史证据|
|时序学习（待冻结）|P3|完整任务条件到实际路径|一个Diffusion/确定性序列|既有合格TRAIN/VAL|MUST|QUEUED_DEPENDENCY|不扩库、无DEV梯度、负收益仍评价|
|开发整链验收（待冻结）|P4|连续避障接近→目标追踪|冻结最终系统|开发套件|MUST|QUEUED_DEPENDENCY|最多30执行/重放，最终TEST未启动|

暂停卡片的评估：实际状态blocked，内容局部过时；不删除未收尾目标或提前标complete。现有ACTIVE heartbeat保留并更新为持续路线，旧配置已归档。工程完成、数据等级、真实执行和科学收益分别报告。
