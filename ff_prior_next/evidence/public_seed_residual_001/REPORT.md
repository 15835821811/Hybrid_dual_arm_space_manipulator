# V7.1 公开目标初始化与残差 Diffusion 验证报告

运行状态：FINISHED；收益判定：NOT_ESTABLISHED。本轮为原四个 DEV 任务的适应性调试，不能视为独立 TEST。

| 方法 | MODEL_VALID / 20 | 冻结目标通过 / 20 | 真实执行通过 / 4 |
|---|---:|---:|---:|
| structured_public_seed | 20 | 2 | 1 |
| residual_diffusion_guided | 19 | 0 | 0 |
| residual_diffusion_unguided | 19 | 0 | 0 |

三个方法使用同一公开目标广义雅可比初始化，基线加平滑随机扰动；残差网络学习 Z 减该初始化，所有统计仅拟合 TRAIN。
每方法每任务 5 候选；两种 Diffusion 每候选 16 网络调用；基线与 guided 均仅 2 次、每次 8 方向全路径引导。
真实执行必须先通过全部模型门槛和冻结双末端目标，再预选一个候选；未执行不代表执行通过。
非穿透 >=0 m、目标 5 mm/1 度、几何细化 2 mm、17→67、自由基座、真实力矩和独立重放保持原合同。

## 诊断与限制

- structured_public_seed：较差一臂位置误差范围 mm = [4.2299672654340865, 34.926864918897785]；失败门槛 = {}；候选错误 = 0。
- residual_diffusion_guided：较差一臂位置误差范围 mm = [8.93291645441437, 29.91460673812113]；失败门槛 = {'geometry_refinement': 1}；候选错误 = 0。
- residual_diffusion_unguided：较差一臂位置误差范围 mm = [12.280039813991447, 71.16279370763196]；失败门槛 = {'geometry_refinement': 1}；候选错误 = 0。
- 训练状态 PASS，更新 200 次；零残差 VAL RMSE = 0.038419 rad。
- 各噪声级别去噪诊断：[{'shuffled_condition_rmse_rad': 0.0003533567307546528, 't': 0, 'val_rmse_rad': 0.00036705115304354005}, {'shuffled_condition_rmse_rad': 0.034168236576744004, 't': 500, 'val_rmse_rad': 0.02996612278525045}, {'shuffled_condition_rmse_rad': 0.04074591209444992, 't': 999, 'val_rmse_rad': 0.0351321709180414}]。训练误差改善不能替代规划执行成功。
- 引导方向局部秩 8/12；无约束线性化后剩余误差比例 0.2215。这是乐观局部诊断，不能证明全局可达或不可达。
- 引导方向局部秩 8/12；无约束线性化后剩余误差比例 0.3127。这是乐观局部诊断，不能证明全局可达或不可达。
- 引导方向局部秩 8/12；无约束线性化后剩余误差比例 0.4817。这是乐观局部诊断，不能证明全局可达或不可达。
- 引导方向局部秩 8/12；无约束线性化后剩余误差比例 0.1385。这是乐观局部诊断，不能证明全局可达或不可达。

几何细化失败表示未满足验证分辨率；是否发生穿透要单独读取 geometry 门槛。只有真实执行和独立重放全部通过才计入执行成功。
若残差方案没有超过基线，本轮不支持 Diffusion 贡献；不自动扩数据、扩训练、调参或启动 pilot/TEST。
若出现收益，也仅记为探索性 DEV 信号，仍需以后另行冻结独立验证协议。

## 资源和复现

分阶段 wall 秒：{'direction_diagnostics': 1.0559220002032816, 'candidates': 106.39800800103694, 'execution': 10.967469200491905}；CPU 秒：{'direction_diagnostics': 1.015625, 'candidates': 104.53125, 'execution': 10.65625}。
物理与模型调用：{'direction_diagnostics.mass_matrix': 17544, 'direction_diagnostics.fk': 17544, 'direction_diagnostics.forward': 68, 'direction_diagnostics.distance': 0, 'direction_diagnostics.coupled': 68, 'direction_diagnostics.physics_steps': 0, 'direction_diagnostics.rrt': 0, 'direction_diagnostics.sqp': 0, 'direction_diagnostics.torque_rollout': 0, 'candidates.mass_matrix': 515220, 'candidates.fk': 758982, 'candidates.forward': 3852, 'candidates.distance': 11209912, 'candidates.coupled': 1866, 'candidates.physics_steps': 0, 'candidates.rrt': 0, 'candidates.sqp': 0, 'candidates.torque_rollout': 0, 'execution.mass_matrix': 15752, 'execution.fk': 0, 'execution.forward': 26256, 'execution.distance': 483046, 'execution.coupled': 0, 'execution.physics_steps': 10500, 'execution.rrt': 0, 'execution.sqp': 0, 'execution.torque_rollout': 1}。训练调用和耗时另见 training.json；进程总耗时见 status.json。
中断导致未知调用的工作项：[]。未知不计作零。
准备/公开初始化调用及耗时：{'calls': {'coupled': 0, 'distance': 46, 'fk': 184, 'forward': 186, 'mass_matrix': 185, 'physics_steps': 0, 'rrt': 0, 'sqp': 0, 'torque_rollout': 0}, 'cpu_s': 0.5, 'wall_s': 0.5004491996951401}。进程中断记录：None。
所有逐候选结果、执行与重放报告、冻结源哈希、源码快照、事件日志均保留在本运行目录。
