# 执行跟踪

| Run | 阶段 | 证据 | 状态 |
|---|---|---|---|
| goal_projection_001 | M0 | 64/64软件测试；16/16固定TRAIN标签重验 | PASS |
| goal_projection_001 | M1 | 投影Diffusion VAL30/30候选合格；6/6执行重放 | PASS |
| goal_projection_001 | M2 | 原DEV20/20候选合格；4/4执行重放 | PASS |
| goal_projection_001 | M3 | 样条与Diffusion共20次执行和20次独立重放 | PASS |
| goal_projection_001 | M4 | 只读审计PASS；完成后恢复270文件字节不变 | PASS |

C1仅在本次smoke范围SUPPORTED；C2 Diffusion额外收益NOT_ESTABLISHED。没有pilot/TEST。
