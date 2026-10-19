# delivery_v1 当前进展

当前执行依据为 DIFFUSION_GOAL_CURRENT.md 与 paper/EXPERIMENT_PLAN.md，独立分支 v7.1-lightweight-free-floating-prior。用户删除旧暂停目标后已创建新 active 目标。工程交付与研究优势分别判断。

## 实际阶段

|阶段|当前证据|状态|
|---|---|---|
|M0|稳定13774模型原DEV4＋已用8任务，各180/180合格、36/36执行重放；旧索引13838 TRAIN/48 VAL|PASS_REUSED，历史证据|
|M1|本轮四方法各80/80合格、16/16原生执行重放；原始审计PASS，耗时26.4分钟|PASS，已完成|
|M2|128/128新增合格，32目标，16/16执行重放；TRAIN13966/VAL48|PASS，源码4aeced8a|
|M3|至多6000更新、原冻结DEV同720候选/144执行一次对照|M2已通过，M3实施中|
|M4|数据/模型manifest、失败账本、中文报告、新独立审查和TEST准备度|待前序结果|

M1源码 a5e894df，收尾提交 0c7351ea。8/8教师与8/8编码往返均通过世界目标和模型门；后四步的最大latent反馈误差2.22e-16 rad，DDIM分解残差1.78e-15。新采样按预冻结“合格/执行计数不低于旧方法”规则采用，并不代表优势成立。位置误差中位数旧末步0.374mm、新后四步1.149mm；最大误差旧3.832mm、新2.512mm。四方法成功率相同，Diffusion稳定优势仍NOT_ESTABLISHED。

M1完成恢复返回NO_WORK_ALREADY_COMPLETE，680输出文件清单未改变。详细报告 ff_prior_delivery_guidance/docs/ANALYSIS.md，原始结果 artifacts/v7_1/delivery_guidance_smoke_001。

## 下一阶段的继续门

M2按冻结32目标×4槽、每槽最多2次、最多256尝试；至少64合格、16目标、16预选单元中13执行重放，通过后立即进入M3。执行失败/未知排除新增训练清单，其余未执行仍MODEL_VALID。资格不足保留PARTIAL_QUALIFIED，不循环补满。主库无需同目标五条；不把4槽当4模式。

M3只训练和复核一次；交付配置候选至少144/180、执行重放至少29/36且无实现/来源错误即可工程通过。不要求裸提案全命中、20%损失改善、强基线优越性、查询或耗时胜出。新模型不足可以交付身份一致且有证据的旧模型，不能赋予新模型旧结果。

单条物理门保持仅不穿透、5mm/1°、2mm几何步长与其余原运动/传播/力矩门；完整初态、17→67、自由漂浮、全机几何、真实力矩和独立重放保留。旧世界监督D=0.914089与其DEV BLOCKED不改写，C/D不晋级。

## 交付边界与入口

本执行轮以M4实际工程交付结束，优势可以NOT_ESTABLISHED。最终TEST当前NOT_READY且本轮不运行；下一批256只制定覆盖协议。旧1024、pilot、study、strict_multimodal和生产规模未启动。

统一：`.\ff_prior_delivery\job.ps1 status|plan|report`。
M1：`.\ff_prior_delivery_guidance\job.ps1 status|report|resume`。
M2：`.\ff_prior_delivery_dataset\job.ps1 start|status|report|resume`。

运行状态以各独立目录status/PID/心跳为准；正常运行中只作轻量存活检查。旧模型、数据、结果、分支、用户修改和旧作业全部保留。原始审计与独立研究审查区分；M4将做一次新的独立结果审查。
