# 复杂 TRAIN/VAL：等预算 Diffusion 与样条验证分析

后台状态 **FINISHED**，实测耗时 **888.2 秒**。证据复核 **PASS**，软件回归 **81/81**。

新增可达性见证：**12/12** 条通过真实力矩执行与独立重放。新 TRAIN/VAL 条数：**{'TRAIN': 6, 'VAL': 6}**。引用式组合索引包含 **TRAIN 172、VAL 36**，原记录和旧数据均未改写。

Diffusion 在两个样条都失败时独有成功的任务：**TRAIN 2、VAL 2**。VAL 独有成功覆盖 **1 个父家族**。Diffusion 失败、至少一个样条成功的任务：**2**。

本轮在新 VAL 上观察到现有 Diffusion 的局部实测收益。该收益来自冻结网络提供的候选形状，不能写成新增数据训练的收益；本轮没有更新网络。

## 1. 冻结的目标与公平性

执行前冻结 3 个 TRAIN 家族和 3 个 VAL 家族，每家族构造较大运动、静态绕障两种任务。控制偏移放大 1.5 倍，绕障见证加预先指定的平滑弓形；运动时间 15 秒，保持 0.5 秒。世界目标由见证自由漂浮传播派生一次后冻结，所有 12 槽位在对照采样前完成冻结，失败不换题。

静态障碍是半径 12 mm 的球。其位置依据公共初始轨迹与独立见证选定，禁止读取任何竞争方法结果。障碍必须使公共轨迹产生测得的负间隙，且见证先过筛选、完整几何检查、真实执行与独立重放。因此本轮是在刻意构造的绕障压力集上比较，不代表随机现实场景分布。

三方法每任务均有 5 个固定种子候选；相同最终目标修正、每候选最多 110 次修正传播+3 次完整验证传播、8 秒墙钟上限，每方法每任务总上限 565 次/40 秒。Diffusion 固定 16 步 DDIM。任务轮换方法执行次序。表中是实际消耗，上限相同不意味着强行耗尽或实际耗时相同。

窄样条幅度 0.015 rad；宽样条采用 0.10/0.05 rad 的一、二阶正弦扰动。宽样条用于排除简单扩大扰动即可获得同样结果的解释；它不是经过调参的最优样条规划器。本轮未改动共享修正器，所有方法都只有目标误差修正，没有碰撞梯度修正。

## 2. 全分母结果

| 分区 | 方法 | MODEL_VALID | 目标+模型合格 | 执行及重放 | 传播查询 | 距离查询 | 网络调用 | 候选秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 窄扰动样条 | 18/30 | 18/30 | 4/6 | 2550 | 475754 | 0 | 43.794 |
| TRAIN | 宽扰动样条 | 4/30 | 4/30 | 2/6 | 2640 | 715755 | 0 | 47.586 |
| TRAIN | 冻结 Diffusion | 22/30 | 22/30 | 6/6 | 2559 | 582072 | 480 | 45.475 |
| VAL | 窄扰动样条 | 22/30 | 12/30 | 3/6 | 2226 | 373350 | 0 | 37.454 |
| VAL | 宽扰动样条 | 7/30 | 2/30 | 2/6 | 2395 | 719351 | 0 | 43.618 |
| VAL | 冻结 Diffusion | 20/30 | 15/30 | 4/6 | 2798 | 516656 | 480 | 48.108 |

VAL 相比窄样条，Diffusion 实际传播查询变化 **+25.70%**，候选墙钟变化 **+28.45%**；执行成功为 **4/6 对 3/6**。时间包含采样、修正和候选验证，不包含选定候选的物理审计；这是一次运行测量，未重复测量时间方差。

## 3. 每个任务的配对结果

| 任务 | 见证执行 | 窄样条（合格/5；执行） | 宽样条（合格/5；执行） | Diffusion（合格/5；执行） |
|---|---|---|---|---|
| complex_train_00_larger_motion | EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 5/5；EXECUTION_VERIFIED |
| complex_train_00_static_detour | EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 0/5；NO_CANDIDATE | 3/5；EXECUTION_VERIFIED |
| complex_train_01_larger_motion | EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 2/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| complex_train_01_static_detour | EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 0/5；NO_CANDIDATE | 2/5；EXECUTION_VERIFIED |
| complex_train_02_larger_motion | EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 2/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| complex_train_02_static_detour | EXECUTION_VERIFIED | 3/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 2/5；EXECUTION_VERIFIED |
| complex_val_03_larger_motion | EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 0/5；NO_CANDIDATE | 2/5；EXECUTION_VERIFIED |
| complex_val_03_static_detour | EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 0/5；NO_CANDIDATE | 3/5；EXECUTION_VERIFIED |
| complex_val_04_larger_motion | EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 5/5；EXECUTION_VERIFIED |
| complex_val_04_static_detour | EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 1/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE |
| complex_val_05_larger_motion | EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 1/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| complex_val_05_static_detour | EXECUTION_VERIFIED | 2/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE | 0/5；NO_CANDIDATE |

独有成功仅计算候选记录完整、无运行错误、见证执行通过且 Diffusion 选定轨迹执行和重放均通过的任务。

VAL 独有成功任务的失败原因与实际开销：

| 任务 | 方法 | 合格数 | 传播查询 | 候选秒 | 失败门槛 |
|---|---|---:|---:|---:|---|
| complex_val_03_larger_motion | 窄扰动样条 | 0/5 | 541 | 9.029 | {'frozen_goal': 5} |
| complex_val_03_larger_motion | 宽扰动样条 | 0/5 | 545 | 9.614 | {'frozen_goal': 2, 'gate:geometry_refinement': 3} |
| complex_val_03_larger_motion | 冻结 Diffusion | 2/5 | 545 | 9.689 | {'frozen_goal': 3} |
| complex_val_03_static_detour | 窄扰动样条 | 0/5 | 540 | 8.903 | {'frozen_goal': 5} |
| complex_val_03_static_detour | 宽扰动样条 | 0/5 | 545 | 9.571 | {'frozen_goal': 3, 'gate:geometry': 1, 'gate:geometry_refinement': 2} |
| complex_val_03_static_detour | 冻结 Diffusion | 3/5 | 545 | 9.689 | {'frozen_goal': 2} |

## 4. 失败在哪里

| 方法 | 所有未通过门槛计数（可重叠） |
|---|---|
| 窄扰动样条 | {'geometry': 20, 'frozen_goal': 10} |
| 宽扰动样条 | {'geometry_refinement': 48, 'geometry': 10, 'frozen_goal': 10} |
| 冻结 Diffusion | {'geometry': 18, 'frozen_goal': 5} |

本轮实测：**2 个 VAL 独有成功任务中，窄样条均是 5/5 候选未达到冻结目标**；主要证据是学习提案改善了有限预算下的目标修正，而不是单独证明避障更强。另有 **2 个 VAL 任务，Diffusion 的 5/5 候选全部因碰撞被拒绝，但某种样条成功**。因此避障覆盖仍存在明确短板。

`geometry` 表示全机几何出现负间隙；`geometry_refinement` 表示 512 区间下仍不满足原几何离散检查要求，不能将其叫作已证明的碰撞或物理失败；`frozen_goal` 表示未满足固定世界目标。一个候选可触发多项，计数不应相加当作样本数。

若宽扰动样条大量触发 geometry_refinement，说明简单增大随机扰动并不构成有效的强样条对照。这会限制“Diffusion 优于所有样条方法”的结论。本轮最多证明它优于这两个冻结的样条候选策略，后续应考虑在同预算内构造更平滑且几何离散有效的样条对照；不得通过放宽 2 mm 几何门槛让宽样条通过。

两种收益机制需要区分：较大运动中，学习提案可能提供更好的目标修正起点，使有限查询内的局部求解成功；绕障中，先验提供与公共直达曲线不同的内部形状，而末端由共享修正器收敛。前者应看 frozen_goal 失败，后者看 geometry 失败，不能都归为“避障提升”。目标修正器只约束末端，能保留部分中途形状。网络没有障碍位置输入，且没有碰撞代价引导，绕障依赖有限候选覆盖可行路径，不能称为稳定的主动避障规划。

## 5. 物理证据与数据可用性

实际执行总数（含见证）**33**，执行及重放累计 **511500** 个积分步；正式候选 **180** 条，网络前向 **960** 次。单候选最大传播数 **109**，最大墙钟 **1.961 秒**。

执行中最小实测几何间隙 **1.589239 mm**，最大保持期位置误差 **1.351464 mm**，最大姿态误差 **0.072407°**。全部以原始轨迹重新计算并核对报告，重放误差按原 ≤1e-10 门槛检查。

静态场景迁移验证了原机器人惯性、映射、关节、驱动器、完整初态数值、原几何及网格均不变，仅新增世界静态碰撞球；新的模型哈希、初态 ID 和碰撞对合同独立冻结。全机几何、真实 67 路力矩、被动自由基座与独立控制重放保留，没有隐藏状态写入或外力注入。物理执行指 MuJoCo 力矩积分，不是硬件实机。

新增记录的 EXECUTION_VERIFIED 只对应各自见证轨迹，不批量升级其他模型候选。TRAIN 记录可用于后续训练，VAL 记录不能进入训练；家族分区未交叉。旧数据维持原验证等级。`combined_train_val_index.json` 用文件哈希引用原数据与新增数据，形成可实际读取的组合库。新场景的网络条件与未来碰撞引导仍需明确设计，不能忽略场景身份直接混入原训练器。

## 6. 当前结论与下一阶段触发条件

软件实现与测试完成；新增复杂轨迹是否合格看各自模型及执行证据；Diffusion 收益看上面的 VAL 配对结果。四者分别报告。所有任务都由有限家族派生，6 个 VAL 任务还包含同家族的两种任务，不能当作 6 个独立分布样本推导显著性。没有进行 TEST，不能推断最终泛化收益。

当前最值得补强的机制是场景条件与碰撞约束下的候选覆盖，以及保持可验证平滑性的样条对照。如果本轮发现局部 Diffusion 收益，可先保存这些失败/成功配对；下一轮仍应小规模、预先冻结，检验收益能否在更强样条对照下保留。若本轮没有 VAL 收益，应优先诊断这些环节，不能仅凭继续扩数据或继续训练承诺改善。本次没有自动启动任何后续训练或新实验。

## 7. 文件、后台与保护检查

- `ff_prior_complex/docs/EXPERIMENT_PLAN.md`：执行前详细目标与预算。
- `artifacts/v7_1/complex_train_val_001/frozen_complex_tasks.json`：全部冻结槽位、场景、见证和来源。
- `artifacts/v7_1/complex_train_val_001/dataset.json`：新增已验证记录。
- `artifacts/v7_1/complex_train_val_001/combined_train_val_index.json`：原库与新增库的引用式组合。
- `artifacts/v7_1/complex_train_val_001/{witness_executions,executions}/`：原始控制、状态及独立重放证据。
- `ff_prior_complex/evidence/complex_train_val_001/independent_verification.json`：原始证据复核；复核额外物理/网络调用均为 0。

```powershell
.\ff_prior_complex\job.ps1 status
.\ff_prior_complex\job.ps1 report
.\ff_prior_complex\job.ps1 resume
```

旧冻结 DEV、原结果、用户未提交修改、main 与旧分支均通过哈希或提交身份检查；未停止旧作业，未运行 pilot、study、严格多模态或 TEST，未上传大数据。后台完成后 resume 的不重复运行检查见 evidence/resume_verification.json。

测试记录：最初的“球位于网格原点”测试不能构成明确穿透，已改为实际网格表面的相交测试，最终全套 81 项通过；早期失败记录保留为 unit_tests.xml。正式数据生成只接受实际负间隙的障碍位置。
