# 实验跟踪

| Run ID | 阶段 | 实际结果 | 状态 |
|---|---|---|---|
| equal_budget_residual_001 | M0 | 冻结旧状态，72/72回归，源码提交3116e9e7 | COMPLETE |
| equal_budget_residual_001 | M1 | 12个逐步诊断，30 TRAIN+6 VAL教师合格 | COMPLETE |
| equal_budget_residual_001 | M2 | sanity通过，240次微调；VAL选回原权重，收益未成立 | COMPLETE_NEGATIVE_RESULT |
| equal_budget_residual_001 | M3 | 先VAL冻结选择后原DEV；200候选与预选执行 | COMPLETE |
| equal_budget_residual_001 | M4 | 原始证据审计、恢复不重复、报告 | COMPLETE |
