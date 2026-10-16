# 复杂 TRAIN/VAL 小规模验证

状态：FINISHED。仅 smoke；原冻结 DEV 与旧结果保留，未启动 pilot 或 TEST。

本轮增加复杂任务和已执行轨迹，冻结已有网络权重，不进行新训练。每方法每任务 5 个候选；每候选最多 110 次修正传播 + 3 次独立验证传播，墙钟上限 8 秒。实际开销另行报告。

新增已执行并独立重放合格样本：{'TRAIN': 6, 'VAL': 6}；受阻任务槽位：0/12。

| 分区 | 方法 | MODEL_VALID | 目标及模型合格 | 执行+重放 | 传播查询 | 候选秒 |
|---|---|---:|---:|---:|---:|---:|
| TRAIN | structured_narrow | 18/30 | 18/30 | 4/6 | 2550 | 43.794 |
| TRAIN | structured_wide | 4/30 | 4/30 | 2/6 | 2640 | 47.586 |
| TRAIN | frozen_diffusion | 22/30 | 22/30 | 6/6 | 2559 | 45.475 |
| VAL | structured_narrow | 22/30 | 12/30 | 3/6 | 2226 | 37.454 |
| VAL | structured_wide | 7/30 | 2/30 | 2/6 | 2395 | 43.618 |
| VAL | frozen_diffusion | 20/30 | 15/30 | 4/6 | 2798 | 48.108 |

Diffusion 独有成功（两个样条均失败）：[{'split': 'TRAIN', 'sample_id': 'complex_train_00_static_detour', 'kind': 'static_detour'}, {'split': 'TRAIN', 'sample_id': 'complex_train_01_static_detour', 'kind': 'static_detour'}, {'split': 'VAL', 'sample_id': 'complex_val_03_larger_motion', 'kind': 'larger_motion'}, {'split': 'VAL', 'sample_id': 'complex_val_03_static_detour', 'kind': 'static_detour'}]
Diffusion 失败而至少一个样条成功：[{'split': 'VAL', 'sample_id': 'complex_val_04_static_detour', 'kind': 'static_detour'}, {'split': 'VAL', 'sample_id': 'complex_val_05_static_detour', 'kind': 'static_detour'}]

只有实际力矩积分与独立控制重放均通过，才计执行成功。参考见证轨迹用于证明可达性，未提供给规划器。未执行候选不升级验证等级。

约束：17→67 映射、完整初态、自由漂浮耦合、全机几何、仅不穿透 ≥0 m、5 mm/1°目标、0.5 s保持、30 s截止及原力矩/速度/加速度/动量/重放门槛均保留。物理执行指 MuJoCo 力矩积分，不是实机。

解释范围：两类任务是较大运动与静态绕障。共享修正器只优化末端目标，未优化碰撞代价；冻结网络不接收障碍位置。负结果不能归因于微重力本身，也不能证明带场景条件和碰撞引导的未来方法必然失败。

本轮仅检验已有先验在新任务上的收益；新增 TRAIN 数据尚未被训练采用，VAL 仅作开发验证。原 DEV 不重跑，TEST 未读取样本。

| 任务 | 见证 | 窄样条 | 宽样条 | Diffusion |
|---|---|---|---|---|
| complex_train_00_larger_motion | EXECUTION_VERIFIED | EXECUTION_VERIFIED | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_train_00_static_detour | EXECUTION_VERIFIED | NO_CANDIDATE | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_train_01_larger_motion | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED |
| complex_train_01_static_detour | EXECUTION_VERIFIED | NO_CANDIDATE | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_train_02_larger_motion | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED |
| complex_train_02_static_detour | EXECUTION_VERIFIED | EXECUTION_VERIFIED | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_val_03_larger_motion | EXECUTION_VERIFIED | NO_CANDIDATE | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_val_03_static_detour | EXECUTION_VERIFIED | NO_CANDIDATE | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_val_04_larger_motion | EXECUTION_VERIFIED | EXECUTION_VERIFIED | NO_CANDIDATE | EXECUTION_VERIFIED |
| complex_val_04_static_detour | EXECUTION_VERIFIED | NO_CANDIDATE | EXECUTION_VERIFIED | NO_CANDIDATE |
| complex_val_05_larger_motion | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED | EXECUTION_VERIFIED |
| complex_val_05_static_detour | EXECUTION_VERIFIED | EXECUTION_VERIFIED | NO_CANDIDATE | NO_CANDIDATE |

失败细分及各任务查询、候选耗时见 summary.json；完整轨迹和初态位于 tasks/，控制与重放证据位于 executions/ 和 witness_executions/。
