# 实验执行追踪

| 阶段 | 范围 | 状态 | 证据 |
|---|---|---|---|
| M0 | 当前HEAD、运行作业与保护边界 | 已审计 | 7e939f03；四项用户修改保留；无Python作业 |
| M1 | 保持目标的碰撞修正与自适应验证 | 已实现 | guidance.py / validation.py |
| M2 | TRAIN校准 | 12/12模型与世界目标通过 | evidence/train_calibration_001；未执行力矩 |
| M3 | 软件测试 | 新增8/8通过；完整回归待完成 | evidence/new_tests_initial.xml |
| M4 | 冻结VAL smoke与执行重放 | 待运行 | artifacts/v7_1/goal_preserving_smoke_001 |
| M5 | 独立证据复核与下一步决策 | 待运行 | 不因完成运行而宣布完整优势目标达成 |
