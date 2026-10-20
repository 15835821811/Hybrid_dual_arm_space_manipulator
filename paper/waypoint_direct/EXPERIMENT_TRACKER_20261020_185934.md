# 完整时序直接轨迹生成进度

状态：PLANNED；本次仅制定目标，未运行新训练/仿真。

| 块 | 状态 | 退出依据 |
|---|---|---|
| B0 数据与表示 | PENDING | 现有动态TRAIN/VAL来源清单、表示与同伺服重建验收；数据不足则BLOCKED |
| B1 一次训练 | PENDING | 两模型各3000实际更新、最终权重与消费记录 |
| B2 无引导诊断 | PENDING | 全程误差与条件使用，不用优势作准入门 |
| B3 原DEV | PENDING | 6个Diffusion、2个确定性候选；最多8新执行与重放 |
| B4 报告 | PENDING | 资格/执行/科学优势分开、回放和失败分母完整 |

旧结果：ff_prior_waypoint_domain/docs/ANALYSIS.md；不改动。协议：EXPERIMENT_PLAN.md。
