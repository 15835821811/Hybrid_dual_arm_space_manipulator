# 完整采样世界监督与合格保护tracker

|run|阶段|状态|门/证据|
|---|---|---|---|
|rollout_world_bounded_001|M0|预检通过后冻结启动|完整323迁移；旧6重建/3执行身份复用；7项软件测试|
|rollout_world_bounded_001|M1|待后台|同8TRAIN×2种子，当前完整采样323 Jacobian，最多4轮×64提议更新，真实原合格保护；≥12/16且Q中位改善10%|
|rollout_world_bounded_001|M2|依赖M1|一个候选；14222TRAIN，6000普通更新，再最多4轮世界细化；M1权重不入候选|
|rollout_world_bounded_001|M3|依赖M2|新96完整无引导；旧96复用|
|rollout_world_bounded_001|M4|依赖M2|原DEV180/36＋应用10/2；相同任务、种子、候选及物理预算|
|rollout_world_bounded_001|M5|终态后|只读原始审计、全新独立审查、一次resume/保留核验、报告和提交|

只按真实结果放行。无扩库/TEST，旧结果保留，DEV不进TRAIN，不放宽物理门。M1失败则下游BLOCKED仍收尾，不自动追加试验。
