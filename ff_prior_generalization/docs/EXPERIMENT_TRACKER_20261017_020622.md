# 独立新验证执行跟踪

| Run ID | 阶段 | 目的 | 分区 | 状态 | 说明 |
|---|---|---|---|---|---|
| software_001 | M0 | 实际物理任务、保存模型采样等价、准入篡改拒绝、统计/存活状态 | 软件测试 | TODO | 不作为正式实验结果 |
| independent_new_validation_001 | M1–M4 | 12独立家族、720候选、教师及选定真实执行/重放、引导前完整验证、原始审计 | NEW_VAL | TODO | 原DEV/VAL保留；无训练更新或最终TEST |

终端：`.\ff_prior_generalization\job.ps1 status`及`report`。优先检查完整机制和数据覆盖，不设查询/耗时相对优胜门槛。
