# 当前实验进度：世界终点优化的配对诊断

|Run ID|阶段|目的/方法|分区|主要门|状态|说明|
|---|---|---|---|---|---|---|
|world_endpoint_smoke_001|上一轮|真实终点监督闭环|TRAIN/冻结DEV|实现和物理审计|FINISHED dade7190|完整通过；TRAIN损失比4.608628、神经收益FAIL；保留64合格修正和失败模型|
|paired_objectives_smoke_001|M0|来源及代码验证|既有TRAIN/VAL|源码和来源hash、真实物理回归测试|IMPLEMENTED_TESTING|独立ff_prior_paired_objectives；尚不声称正式通过|
|paired_objectives_smoke_001|M1|8例TRAIN参数单步|TRAIN|完整102梯度+参数方向+真实下降|PENDING|门过才开始四组|
|paired_objectives_smoke_001|M2|v/local/world/both四组|相同13838 TRAIN|6000更新、同初始化/批次/噪声、240真梯度|PENDING|不增加数据、不换强基准|
|paired_objectives_smoke_001|M3|TRAIN选组后同预算smoke|原DEV4+已用8开发|TRAIN比≤0.8且不劣B，720候选/144执行上限|DEPENDENT|未通过不启动DEV；无TEST|
|paired_objectives_smoke_001|M4|原始审计/独立审查/恢复/报告|本轮原始结果|可重建证据、NO_WORK恢复、仅任务提交|PENDING|后台结束后收尾|
|qualified_origin_batch_next|长期后续|1024槽扩库|TRAIN|另冻范围和资格门|PREPARED_NOT_LAUNCHED|用户最新要求保持暂停|
|pilot/TEST|长期后续|最终覆盖与一次性检验|未触碰TEST|未来模型/协议冻结与范围授权|BLOCKED_BY_CURRENT_SCOPE|本轮不读取、不运行|

用户已恢复本轮实际执行；旧应用目标卡片不可通过现有goal工具原地改写/恢复。统一目标以DIFFUSION_GOAL_CURRENT.md和本轮代码协议为准。历史目标、计划及tracker均保存在ff_prior_paired_objectives/docs/history/。长期目标未完成。
