# 合格数据、训练与碰撞引导闭环分析

后台实测 1373.3 秒，状态 **FINISHED_WITH_BLOCKED_ITEMS**。软件 **90/90**，原始证据复核 **PASS**。

新标签 **48/48** 通过指定世界目标和 MODEL_VALID。预选 **8/8** 通过真实 MuJoCo 67 力矩执行与独立重放。

扩库结论：**BLOCKED**。尚未达到预先冻结的稳定采样条件，未启动额外12条扩库，更未启动pilot或TEST。

## 1. 本轮实际改动

把上一轮只有索引引用的复杂数据接入实际数组加载、TRAIN抽样和新网络训练。增加16个静止零动量初始构型，每个完整保存初态；物理home、机器人编译模型、惯量、自由基座和17→67映射不变。每个世界目标只由首条完整耦合传播派生一次，另外两条轨迹向同一目标修正。失败不重标。

新增全路径碰撞代价，使用128区间的自由漂浮传播与33个几何引导状态。24个低频方向在同一求解器中修正末端和绕行路径；每方向限步避免一个病态大方向压小所有有效方向。增加沿途几何变化惩罚，避免绕行过猛触发原有2mm采样门槛。引导栅格只用于提案，随后独立做完整几何、动力学执行和重放验收。

共享修正器只在第16步DDIM的干净轨迹上作用；v预测、float64状态、投影后重建epsilon以及最后控制点精确返回继续保留。

## 2. 数据能否用于学习

实际训练索引 {'TRAIN': 208, 'VAL': 48}；最终可用索引 {'TRAIN': 208, 'VAL': 48}。TRAIN抽样最少/最多次数为 46/47，所有TRAIN行均参与训练，VAL梯度更新为0。

训练600次更新，VAL选中第 **600** 次权重；权重确实变化，加载后输出最大差 0.0。TRAIN-only temporal prior 秩为 7。fit_population_token是训练初态集合的身份标识，不冒充任意单一初态。

合格数据已可用于当前范围的残差运动先验训练。MODEL_VALID样本只承诺通过模型门槛，未执行样本不会被抽查批量升级。数据可用与网络能稳定规划是两件事；后者看下一节逐任务结果。

原TEST文件仅参与保全哈希核验，没有加载TEST数组用于训练、调参或最终评估。

## 3. 同预算对照

10个VAL任务：原6复杂任务加4个新初态任务。每方法每任务5个相同种子候选；每候选110次修正+3次独立验证查询、15秒上限，Diffusion16次网络调用。方法次序按任务轮换。原DEV只核验哈希，不重跑。

| 方法 | MODEL_VALID/50 | 指定目标合格/50 | 执行重放/10 | 耦合查询 | 距离查询 | 候选耗时秒 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Diffusion＋联合修正 | 44 | 41 | 10 | 3561 | 6291231 | 103.79 |
| 平滑样条＋联合修正 | 45 | 44 | 9 | 3117 | 5481081 | 89.56 |
| 新 Diffusion＋仅目标修正 | 40 | 40 | 9 | 2971 | 801784 | 56.83 |
| 新 Diffusion＋联合修正 | 43 | 40 | 10 | 3468 | 6025920 | 100.05 |

| 任务 | 样条 | 旧Diffusion | 新Diffusion联合 | 新Diffusion仅目标 |
|---|---|---|---|---|
| complex_val_03_larger_motion | 5/5；EXECUTION_VERIFIED | 4/5；EXECUTION_VERIFIED | 3/5；EXECUTION_VERIFIED | 3/5；EXECUTION_VERIFIED |
| complex_val_03_static_detour | 0/5；NO_CANDIDATE | 3/5；EXECUTION_VERIFIED | 1/5；EXECUTION_VERIFIED | 4/5；EXECUTION_VERIFIED |
| complex_val_04_larger_motion | 5/5；EXECUTION_VERIFIED | 4/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| complex_val_04_static_detour | 4/5；EXECUTION_VERIFIED | 2/5；EXECUTION_VERIFIED | 4/5；EXECUTION_VERIFIED | 3/5；EXECUTION_VERIFIED |
| complex_val_05_larger_motion | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| complex_val_05_static_detour | 5/5；EXECUTION_VERIFIED | 3/5；EXECUTION_VERIFIED | 2/5；EXECUTION_VERIFIED | 0/5；NO_CANDIDATE |
| loop_val_12 | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| loop_val_13 | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| loop_val_14 | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |
| loop_val_15 | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED | 5/5；EXECUTION_VERIFIED |

新Diffusion对样条的任务级独赢：['complex_val_03_static_detour']；反向失败：[]。新Diffusion实际查询相对样条 +11.26%，候选耗时 +11.71%。

同一新权重、同一噪声种子的碰撞引导消融：联合修正独有合格 **3** 个候选，取消碰撞代价反而独有合格 **3** 个。必须同时看这两个方向，不能把更多碰撞计算自动写成普遍收益。

## 4. 失败归因与噪声诊断

| 方法 | 未合格原因计数（同一候选可多项） |
|---|---|
| 旧 Diffusion＋联合修正 | {'geometry': 1, 'geometry_refinement': 6, 'world_goal': 5} |
| 平滑样条＋联合修正 | {'geometry': 2, 'geometry_refinement': 5, 'world_goal': 4} |
| 新 Diffusion＋仅目标修正 | {'geometry': 8, 'geometry_refinement': 2} |
| 新 Diffusion＋联合修正 | {'geometry': 1, 'geometry_refinement': 7, 'world_goal': 7} |

新训练与旧冻结网络的合格数分别是 40/50 与 41/50，选中执行分别为 10/10 与 10/10。新增数据训练的收益没有据此成立。

本轮最具体的机制问题是联合修正仍采用加权软代价。在同权重同噪声的消融中，加入碰撞项使碰撞失败从8降至1，但世界目标失败从0增至7，几何细化失败从2增至7。这些计数有重叠。50组消融在最后去噪修正前的输入哈希完全相同，因此这次变化来自联合修正过程；不能归咎于重新抽样。

3mm碰撞激活裕量高于实际0mm验收要求，修正器可能为增大间隙牺牲终端误差或增加路径弯曲。现有加权和与有限迭代不能保证所有约束同时满足。应优先验证保留目标可行性的分层/带约束修正，以及在已不穿透时是否仍需要额外推离；物理验收仍保持0mm、5mm/1°等原门槛。不能靠继续堆数据来替代这一修正。

8个不同初态的见证执行全部通过，最大位置误差约0.00127mm。这个范围内没有证据把失败归因为微重力耦合模型错误；更直接的证据指向局部修正的约束权衡和几何验证分辨率上限。该判断仅覆盖本轮案例。

| 噪声时刻 | VAL残差重建RMSE rad | 末端控制点RMSE rad | 打乱条件RMSE rad |
|---|---:|---:|---:|
| 0 | 0.000365953 | 0.000388166 | 0.000361776 |
| 500 | 0.0377958 | 0.0542133 | 0.0449136 |
| 999 | 0.0443904 | 0.0639688 | 0.0535795 |

零残差基线VAL RMSE：0.0520903 rad。噪声重建是诊断指标，不是世界目标到达率；低训练损失不能取代实际规划/执行。

碰撞、目标误差与几何离散误差分开计数。局部引导不保证全局可达：固定查询预算内可能停在局部形状，修正目标又改变沿途路径，或为避障增加曲率而触发严格几何细化门槛。本轮未以固定基座、放宽门槛或重标目标规避失败。

## 5. 真实物理证据与保持项

本轮实际执行（含见证） **46** 次，含独立重放 **713000** 个物理步。最小实测间隙 0.0029918206 m；最大保持位置误差 0.00288838655 m；最大保持姿态误差 0.00419780412 rad。

逐候选最大查询 109，最大候选秒 3.1453。原始控制数组哈希、81维qpos、67维力矩、完整时间、保持区间末端误差与独立重放逐份复核。

已核验 2224 个旧运行文件未变；main、旧分支、用户原有四个未提交文件和冻结DEV保留。旧网络不覆盖，新checkpoint与原始NPZ均仅保存在本地。

## 6. 扩库门槛与下一步

| 预注册检查 | 结果 |
|---|---|
| data_eligible_at_least_85pct_of_48 | PASS |
| each_task_at_least_4_of_5 | BLOCKED |
| new_origin_eligible_at_least_18_of_20 | PASS |
| new_training_all_rows_used | PASS |
| old_complex_eligible_at_least_27_of_30 | BLOCKED |
| trained_execution_10_of_10 | PASS |
| witness_8_of_8 | PASS |

### 补充：几何分辨率诊断（不改变正式结果）

按照预先写入诊断脚本的规则，取冻结任务顺序中最先出现的最多3个“目标通过、仅geometry_refinement失败”的新联合Diffusion候选。保持原路径、初态、世界目标、0mm不穿透与2mm几何步长门槛，另做512→1024区间传播和全机几何检查。没有重新采样或执行，没有替换正式候选。

| 候选 | 原最大几何步长mm | 1024区间步长mm | 加密后模型+目标 |
|---|---:|---:|---|
| artifacts\v7_1\qualified_guidance_loop_001\candidates\complex_val_03_larger_motion\trained_diffusion_joint\01.json | 2.1908 | 1.0954 | True |
| artifacts\v7_1\qualified_guidance_loop_001\candidates\complex_val_03_larger_motion\trained_diffusion_joint\03.json | 2.0896 | 1.0448 | True |
| artifacts\v7_1\qualified_guidance_loop_001\candidates\complex_val_04_static_detour\trained_diffusion_joint\01.json | 2.0317 | 1.0159 | True |

额外耦合查询 6 次，单列为事后诊断，不归入正式等预算表。正式候选和扩库门控哈希未变。加密后通过只支持这些案例原先受分辨率上限阻断，不能记成新增EXECUTION_VERIFIED或将正式结果改成通过。

因此下轮可把“几何验证上限是否应自适应”作为独立软件改造：保持2mm门槛和全部物理约束，提前冻结各方法相同的计算预算，并重新计费完整验证成本。不能把更密集检查隐藏在某一个方法之外。

本轮是小规模工程与探索性VAL证据。一个训练种子、五个采样种子及复用开发任务不足以证明独立测试泛化；新旧模型还同时改变了数据、训练步数、归一化和选择损失，所以不能单独归因为新增数据。

下一步应由失败项决定：若新初态通过而旧复杂任务未通过，优先在TRAIN补充受约束绕障和路径形状修正，验证局部修正是否能同时满足目标与几何细化；保持原VAL冻结。仅当稳定门槛通过后再扩大覆盖，不以数量替代每条标签与采样质量。

## 7. 终端与证据

```powershell
 .\ff_prior_loop\job.ps1 status
 .\ff_prior_loop\job.ps1 report
 .\ff_prior_loop\job.ps1 resume
```

后台报告：artifacts/v7_1/qualified_guidance_loop_001/REPORT.md。详细原始证据包括 generation、learning、candidates、executions、witness_executions、independent_verification.json。紧凑证据归档在 ff_prior_loop/evidence/qualified_guidance_loop_001；不包含大规模权重或轨迹数组。

完成后的resume实测返回NO_WORK_ALREADY_COMPLETE，653个既有输出文件字节未变，后台进程已退出。
