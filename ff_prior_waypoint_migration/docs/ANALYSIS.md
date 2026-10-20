# 旧动态TRAIN单案例迁移核对与静止数据补查

2026-10-20；独立后台运行`artifacts/v7_1/waypoint_migration_001`。诊断实际耗时约18.30秒，终态BLOCKED，worker已退出。旧审计与旧实验均保留。

## 核心结果

**已完成用户要求的单案例来源/初态/合同核对和已有静止数据补查；未满足继续条件，表示执行、训练和DEV尚未启动。** 不是新模型失败，也不是证明全部旧数据不可用。本轮没有生成新训练轨迹，没有推进物理时间，没有扩库或TEST。

选定seed20261012冲突TRAIN：`tracking_output/local10mm_v1/teacher_a01/pilot_00`与`tracking_output/ui_pilot/data/episodes/train_20261012`。

## 1. 初态重建有了明确证据，但不能冒充历史原件

按保存任务参数和已绑定的当前初始化源码构造旧TrackingEnv，仅初始化与mj_forward，没有mj_step。

| 检查 | 实测 |
|---|---:|
| 重建qpos与旧首帧最大差 | 0 |
| 重建qvel与旧首帧最大差 | 0 |
| 17维q / dq首帧最大差 | 0 / 0 |
| 重建完整integration state维数 | 836 |
| mj_getState/setState往返误差 | 0 |
| 初始ctrl / 外施力最大值 | 0 / 0 |

重建状态保存在`reconstructed_initial_candidate.npz`，明确标为**确定性重建候选**。qacc_warmstart约4.17e-21，不能把未归档的历史隐藏状态说成已逐位恢复。对44个非TEST源快照计算旧四文件组合hash，未找到与本案例原`code_version.json`相符的版本。当前源码已冻结，但其初始化匹配不证明整个历史执行程序一致。

因此，原来“完整初态缺失”已缩小为：可见物理状态精确恢复；隐藏积分状态和历史代码绑定仍不能从现有记录完整证明。以后可以在明确重建初态下做新认证，但不能声称旧执行已被独立重放。

## 2. 机器人本体与映射一致，运行合同并不完全一致

| 项目 | 旧数据 / 当前 |
|---|---|
| runtime物理spec hash | 相同：ec36bd1e… |
| 17→67映射矩阵 | 67×17，最大差0 |
| 原生力矩限值 | 最大差0 |
| 连续体工作域 | 旧±π；当前±1rad |
| 启用接触的geometry数量 | 旧0；当前裸机器人78 |
| 目标局部接近点y | 旧-0.205m；当前DEV -0.235m |

同一个底层spec hash不覆盖所有上层工作域、接触开关、碰撞政策与任务差别。旧pair数2927、当前裸机器人2860；旧场景另有两个障碍物，这不是同场景逐pair比较，不能据计数判断谁更严格。

当前DEV接近点和旧TRAIN接近点的差异是任务分布差异，不单独构成旧轨迹物理失败。工作域不同也不意味着旧轨迹一定越界；上一轮保存样本没有观察到±1rad越界。

## 3. 标签冲突：隔离两份拷贝，不能选择较宽松的通过标签

两份trace哈希完全相同，末端误差统计也相同。旧“通过”result没有actual acceleration/velocity和joint_limit_violation字段；“不通过”result记录：

- maximum_actual_acceleration_ratio = **1.0602421738**；
- joint_limit_violation_steps = **18**；
- accepted_for_teacher_dataset = false。

当前可见准入逻辑要求运动违规计数为0。这里的joint_limit_violation_steps是位置/速度/加速度违规的联合计数，并非18次位置越界；结合加速度比支持“存在运动约束失败记录”。没有完整500Hz力矩/状态历史，**1.06024与18是历史结果证据，不是本轮重新仿真测出的数字**。不能据旧宽松正标签覆盖负证据。

本轮处置是对该轨迹hash的两份来源都标记迁移隔离，保留旧文件；没有删样本、改标签、改目标或覆盖旧数据。其他旧任务不因这一条冲突被判失败。

## 4. 静止数据补查

在已编译源模型中按名称解析`target_world_joint`：qpos地址74:81，qvel地址73:79。读取581条已有显式TRAIN/VAL记录的初始六维速度，线速度与角速度一并检查。结果：**581条均为移动，静止0条**。这包括此前4个参数缺失的unknown；按原生地址核对后不再保留unknown。记录含版本/重复，不能当581个独立任务。

另外在tracking_output与artifacts/v7_1中以文件名限定搜索manifest/split/dataset_report/pilot_manifest元数据，排除TEST及diagnostic目录，找到5个来源清单，均为既有盘点来源，未发现额外有TRAIN/VAL身份的静止数据。没有把未分割baseline、两个DEV、静态终点先验或TEST重新标成动态训练样本。

搜索范围是这两个输出根下的指定元数据约定，不是证明其他盘符或其他文件命名方式绝无数据。完整路径和排除范围记录在stationary_search.json。

## 5. 为什么暂不继续训练

单案例有未解决运动违规记录；历史完整状态/源码尚未闭合；最关键的独立阻塞是原协议要求TRAIN和VAL都覆盖静止/移动，而当前来源没有静止样本。即便换一个无标签冲突的移动TRAIN，覆盖门仍未满足。

因此表示执行0次、训练0更新、无引导诊断0、DEV新候选0、力矩执行及重放0。未为继续而放宽物理门，也没有拿DEV补TRAIN。当前交付的后台程序执行来源诊断；没有宣称已实现或训练全时序生成器。

## 下一步

不用再重复这581条的相同搜索。优先获取一个**已有明确分割与来源的静止动态任务目录**，或明确修改研究范围：移动TRAIN训练，静止DEV只做分布外诊断。这是覆盖范围调整，不是降低物理门；当前未自动修改。

对于已有移动数据，迁移时优先选择没有负证据的旧TRAIN，将完整可重建初态、准确任务和当前合同冻结后，做原最多4个TRAIN的表示/执行认证。认证结果是新证据，不冒充补回旧力矩历史；不因单条执行通过升级全库。当前冲突轨迹继续隔离，除非新的独立认证实际满足原约束。

## 交付与复查

终端：

```powershell
.\ff_prior_waypoint_migration\job.ps1 status
.\ff_prior_waypoint_migration\job.ps1 report
.\ff_prior_waypoint_migration\job.ps1 resume
```

新证据入口`artifacts/v7_1/waypoint_migration_001/index.html`。没有新机器人动画，因为没有实际新轨迹执行；入口提供历史回放链接且标为历史。resume核验和源文件保留清单见对应JSON。软件诊断完成、资格迁移未完成、Diffusion直接生成可行性与收益未检验，分别报告。
`n独立审查补充：freeze保留核对覆盖802个列明文件，不是全部传递依赖。robot_adapter/geometry等遗漏依赖已另存运行后hash，不能追认为启动前冻结。DEV接近点最初在诊断程序写作常量，收尾时已在ff_prior_waypoint_scene/experiment.py第47行核对；此核对是源码证据，不是新DEV执行。
