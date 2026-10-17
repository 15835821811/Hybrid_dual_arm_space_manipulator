# 有界扩库进程

| Run ID | 阶段 | 目的 | 数据 | 状态 | 说明 |
|---|---|---|---|---|---|
| qualified_origin_train_001 | 完整机制/训练 | 12688行真实学习及同预算smoke | 原DEV4+已用验证8 | FINISHED | C1PASS/C2FAIL；原报告及c514407e保留 |
| origin_batch_software | 新批实现 | 生成、隔离、真实执行、工作保留 | 小测试夹具 | PASS 8/8 | 真实前向/固定目标/穿透拒绝/67力矩重放；不计正式扩库 |
| qualified_origin_batch_001 | 有界扩库 | 1024槽+16固定目标修正+56预选执行 | TRAIN+新冻结验证8 | FINISHED / 数据门PASS | 1023前向+63模式=1086新TRAIN；合并13774/1578初态；56/56真实执行重放、原始审计PASS |
| qualified_origin_learn_001 | 接续学习 | 条件等价重标定、行/初态混合抽样、13495锚点 | 全13774 TRAIN+原VAL48；原DEV4+新8验证 | FROZEN / 8测试PASS / PREFLIGHT_PASS | 6000步、720候选、144预选执行；正式结果未运行 |
