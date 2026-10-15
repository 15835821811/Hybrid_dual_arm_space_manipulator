# V7.1 固定 TRAIN 目标修正与预测—执行一致性报告

状态：FINISHED。Diffusion收益：NOT_ESTABLISHED；模型一致性：CONSISTENT_IN_CHECKED_CASES。

## 固定目标数据补充

冻结8个TRAIN family目标，最多16次修正；模型门槛与原世界目标同时合格 16/16。
诊断中已知执行失败的样本排除后，新增训练样本 16 条；TRAIN 166，VAL 30。
本轮排除的已知执行失败标签：[]。原始文件不回写。
只改变平滑轨迹参数，不改变冻结目标、时长、初态或物理合同；失败尝试不补样。主先验没有新增每目标五条的配额。

## 耦合预测与真实力矩执行

| 样本 | 执行 | 参考预测EE最大偏差 mm | 按实际67关节重建EE最大偏差 mm | 重建基座最大偏差 mm |
|---|---|---:|---:|---:|
| trainfix_000_0 | PASS | 0.001556 | 0.003552 | 0.000299 |
| main_000044 | PASS | 0.002325 | 0.005897 | 0.000340 |
| trainfix_001_0 | PASS | 0.002599 | 0.005434 | 0.000358 |
| main_000017 | PASS | 0.002384 | 0.006922 | 0.000414 |

参考预测使用规定关节路径；实际67关节重建使用记录的全部原生关节位置和速度，从同一初始基座状态积分，绝不逐帧复制实际基座，也不用B+投影代替原生几何。
两者的差别用于区分参考跟踪误差与耦合传播残差。诊断只给一致性证据，不授予未执行样本EXECUTION_VERIFIED，也不能证明真实硬件模型准确。
比较覆盖整个运动和hold；原目标验收仍5 mm/1度，非穿透>=0 m。1 mm/0.2度重建标志仅为预登记诊断，不改变执行门槛。

## 原冻结DEV smoke

| 方法 | MODEL_VALID / 20 | 原世界目标合格 / 20 | 执行与重放通过 / 4 |
|---|---:|---:|---:|
| structured_public_seed | 20 | 2 | 1 |
| residual_diffusion_guided | 14 | 0 | 0 |
| residual_diffusion_unguided | 14 | 0 | 0 |
- structured_public_seed：较差一臂位置误差范围mm [4.2299672654340865, 34.926864918897785]；失败模型门槛 {}；候选错误 0。
- residual_diffusion_guided：较差一臂位置误差范围mm [10.424762889628186, 36.81062236381695]；失败模型门槛 {'geometry_refinement': 6}；候选错误 0。
- residual_diffusion_unguided：较差一臂位置误差范围mm [14.259091951314218, 73.66139905718452]；失败模型门槛 {'geometry_refinement': 6}；候选错误 0。

所有方法仍共享公开目标初始化。三方法每任务五候选；Diffusion每候选16次网络调用，guided与结构基线各两次八方向全路径引导。
TRAIN修正的额外优化和四个选定诊断均单独计费，不作为DEV额外候选；冻结DEV只在模型训练完成后用于本轮比较，不参与选数据或权重。
这是反复查看同四个DEV任务的开发验证，任何收益仅为探索性信号。若无改善，不自动扩数据、训练或调参。

## 原因分析

四个检查案例的规定参考预测及实际原生路径重建均满足诊断标志，当前证据支持这些案例内的耦合传播一致性；若DEV目标仍失败，应优先分析条件学习与有限引导求解。不能由四例外推全部机器人状态。
训练 PASS，200 更新，重载误差 0.0；VAL零残差RMSE 0.038419 rad。
VAL噪声分段诊断：[{'shuffled_condition_rmse_rad': 0.00034133411659937945, 't': 0, 'val_rmse_rad': 0.00035222500310097966}, {'shuffled_condition_rmse_rad': 0.03270915133602957, 't': 500, 'val_rmse_rad': 0.029257377039330897}, {'shuffled_condition_rmse_rad': 0.03913294398108421, 't': 999, 'val_rmse_rad': 0.03469509443872963}]。这些控制点去噪指标不能替代目标与真实执行成功。

## 开销与保存

新增真实执行：诊断4次、DEV 1次；独立重放通过5次。
分阶段wall秒：{'correction': 14.908996698446572, 'consistency': 59.837788600008935, 'candidates': 108.5212800996378, 'execution': 10.977063999976963}；CPU秒：{'correction': 14.140625, 'consistency': 58.796875, 'candidates': 106.046875, 'execution': 10.703125}。训练耗时另见training.json，进程总时长见status.json。
实际模型与物理调用：{'correction.mass_matrix': 215156, 'correction.fk': 219284, 'correction.forward': 898, 'correction.distance': 190492, 'correction.coupled': 818, 'correction.physics_steps': 0, 'correction.rrt': 0, 'correction.sqp': 0, 'correction.torque_rollout': 0, 'consistency.mass_matrix': 117024, 'consistency.fk': 58220, 'consistency.forward': 105036, 'consistency.distance': 1950408, 'consistency.coupled': 8, 'consistency.physics_steps': 42000, 'consistency.rrt': 0, 'consistency.sqp': 0, 'consistency.torque_rollout': 4, 'candidates.mass_matrix': 529326, 'candidates.fk': 776543, 'candidates.forward': 3878, 'candidates.distance': 11369089, 'candidates.coupled': 1879, 'candidates.physics_steps': 0, 'candidates.rrt': 0, 'candidates.sqp': 0, 'candidates.torque_rollout': 0, 'execution.mass_matrix': 15752, 'execution.fk': 0, 'execution.forward': 26256, 'execution.distance': 483046, 'execution.coupled': 0, 'execution.physics_steps': 10500, 'execution.rrt': 0, 'execution.sqp': 0, 'execution.torque_rollout': 1}。未决中断工作：[]。未知调用不作零。
全部16次TRAIN修正、固定目标、诊断预选、逐候选、真实执行/重放、checkpoint与代码快照保存在本运行目录。
旧结果、原未提交修改及旧作业保留；没有pilot或TEST，没有上传权重或推送。
