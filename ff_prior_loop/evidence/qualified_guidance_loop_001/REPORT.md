# 数据生成与碰撞引导闭环

状态：FINISHED_WITH_BLOCKED_ITEMS。这是探索性 TRAIN/VAL smoke，原冻结 DEV 保留且未运行，未启动 pilot / TEST。

新增合格标签 48/48；实际尝试 48。预选力矩执行与重放 8/8。

| 方法 | MODEL_VALID / 50 | 指定目标合格 / 50 | 实际执行及重放 / 10 | 耦合查询 | 候选秒 |
|---|---:|---:|---:|---:|---:|
| structured_joint | 45 | 44 | 9 | 3117 | 89.564 |
| frozen_diffusion_joint | 44 | 41 | 10 | 3561 | 103.786 |
| trained_diffusion_joint | 43 | 40 | 10 | 3468 | 100.051 |
| trained_diffusion_goal_only | 40 | 40 | 9 | 2971 | 56.833 |

扩库判定：BLOCKED。仅在所有预注册门槛通过时新增最多 12 条 TRAIN 尝试；不因此宣布 Diffusion 收益。

碰撞激活距离 3 mm 是优化代价；验收仍使用原有仅不穿透阈值 0 mm、全机几何和 2 mm 几何采样门槛。世界目标冻结后未重标。

MODEL_VALID 只表示模型验证通过；只有单条真实 67 力矩执行和独立重放通过才标记 EXECUTION_VERIFIED。此处真实执行是 MuJoCo 动力学执行。

训练证据见 learning/training.json 与 training_input_audit.json；失败原因、逐任务结果见 summary.json；原始报告见 executions。
