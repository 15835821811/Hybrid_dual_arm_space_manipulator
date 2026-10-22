# Experiment Tracker：持续工程全链路

更新2026-10-22T10:59:51。P0已完成负结果收尾，P1代码、准确任务/种子/预算已冻结；当前等待首次后台启动。

|Run ID|Milestone|Purpose|System|Split|Status|Notes|
|---|---|---|---|---|---|---|
|preserve323_bounded_001|P0|旧功能保持与独立检查|冻结低102/细节221|固定TRAIN|CLOSED_NEGATIVE|902.969秒；32J通过、64临时更新、0接受、fit/check各8/32；新review partial/high；2022文件完成resume不变|
|preserve323后续|P0|6000候选/无引导/DEV|未训练|VAL/DEV|BLOCKED_NOT_RUN|保持原冻结门；无新力矩执行|
|static_fullchain_001|P1|公开任务到静态避障执行|先验+几何规划/结构对照|原APPLICATION_DEV|FROZEN_PENDING_LAUNCH|11软件检查通过；原任务60提案/12执行，前2各16轮完整323修正；准确协议ff_prior_static_fullchain/docs/EXPERIMENT_PLAN.md|
|动态表示及交接（待冻结）|P2|加速度保真、连续状态跟踪|受约束参考+合法反馈|既有TRAIN/动态DEV|QUEUED|不依赖Diffusion相对优势|
|时序学习（待冻结）|P3|任务条件完整采样|一候选/确定性序列|既有合格TRAIN/VAL|QUEUED_DEPENDENCY|无扩库、无DEV训练|
|开发整链验收（待冻结）|P4|避障接近到移动追踪|最终冻结系统|开发套件|QUEUED_DEPENDENCY|最终TEST未启动|

科学判决、软件完成、模型数据资格和真实执行分开。仅一个ACTIVE持续heartbeat；旧局部目标完成后替换为长期工程目标，不能把此处局部完成当作科学优势成立。


目标与调度交接：P0收尾提交3e659ce4；旧局部卡已完成、新长期卡active；同一automation保持ACTIVE每15分钟，P1实施准备，未启动新重worker。终端next指向IMPLEMENT_AND_FREEZE_NEXT_PHASE。

P1本轮输入冻结hash：516b5d1373a003dbce90739f621203bc1fe50032d8a2bc5be7c4c4fad2675079。没有读取旧00/07解作为规划输入；未运行新物理时不记通过。
