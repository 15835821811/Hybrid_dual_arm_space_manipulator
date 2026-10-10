# C4-A Diffusion 架构与数值审计

## Material Passport

任务：V6.4-C4-A；阶段 A；2026-10-10；数据为 C.3 冻结产物；无训练更新、无 TEST 采样、无物理积分。数值性质经测试 VERIFIED，适配性为有限数据上的 ANALYZED。

结论：**ARCHITECTURE_NOT_JUSTIFIED**（当前数据、预算和用途下，尚未证明选择 Diffusion 的必要性；不是证明 Diffusion 理论上不适用）。算法的低维条件生成形式可用，但不能以数学实现正确替代安全闭环价值。当前应检验初值接口，而不扩大模型。

## 基点和证据

本地与 `git ls-remote origin refs/heads/v6.4-c3-search-aware-closed-loop-val` 均为 `18f6329e3960c76dce8b787749b269695888f9c9`，未发生基点漂移。C.3 物理 producer 为 `9f39b42775283432eb933f63a9047c488ba22070`。基点仅包含其后已记录的 teacher 空身份维护，不冒充原 producer。

新 worktree `E:/v64c4a`，分支 `v6.4-c4a-diffusion-architecture-audit`。原 `Hybrid_dual_arm_space_manipulator` 的未提交改动未移动或覆盖。未 push、未合并 main。

机器证据：`v6_4/c4a_evidence/audit_01/`；完整冻结输入 SHA：`input_sha256.json`；`PortableResolver.verify_all()` 验证发布包已有字节。所有计算从发布包快照读取，不重跑历史 TEST。

## 数学与实现

`residual_diffusion.py` 的模型实际预测 v，配置名 `epsilon_residual` 是继承的外部表示名，不能据此认定内部训练 epsilon。对累计信号系数 a=alpha_bar[t]：

```
x_t = sqrt(a) x_0 + sqrt(1-a) epsilon
v   = sqrt(a) epsilon - sqrt(1-a) x_0
x_0 = sqrt(a) x_t - sqrt(1-a) v
eps = sqrt(1-a) x_t + sqrt(a) v
```

这是正交的二维线性变换，训练目标和两项反推一致。loss 是每个样本在活动维度上的均方误差，然后按批求均值；掩码不能改变为全 12 维平均。

cosine 的 f(u)=cos²((u+0.008)/1.008*pi/2)，beta[t]=min(1-f((t+1)/100)/f(t/100),0.999)，累计乘积以 float64 计算后保存 float32。末端 alpha_bar 非严格零，但小于 1e-6；从标准高斯开始是该已冻结离散调度的近似，不是索引错误。DDIM eta=0，调用时间索引为 `[99,94,89,83,78,73,68,63,57,52,47,42,36,31,26,21,16,10,5,0]`。最后一步 previous alpha=1，返回预测 clean。

12D 是六个区间各两个横向系数；`active_intervals` 仅取关键区间 2 和最近的可用前驱，最多 4D。非活动坐标精确为零。每区间二维圆盘半径 0.020 m，不是各坐标独立方框，也不是全 12D 总范数限制。两个活动区间均可达到 20 mm，因此总范数可大于 20 mm。

|检查|状态|最小复现实例与证据|
|---|---|---|
|12D/至多4D映射|PASS|`test_effective_space_and_interval_disks`，逐冻结 Task 映射、逆取、禁用坐标与圆盘测试|
|v目标/反推|PASS|`test_v_forward_inverse_and_target`，t=0/1/49/98/99，oracle v 重建 clean/eps，误差容限3e-7|
|Cosine100/DDIM20|PASS|独立 NumPy 公式、严格递减调度、全20次 oracle 去噪、末端 clean 测试|
|训练/加噪/去噪mask|PASS|污染非活动 noise 不改变结果；非活动梯度为零；非法 clean 拒绝；逐 DDIM 状态检查|
|inverse scaler合同|PASS|所有55条标签与原 inverse 精确相同；归一化往返误差≤1e-18；非法 inactive/NaN/Inf 不被擦除|
|同Task/权重/噪声/配置复现|PASS（同运行时）|16个条件重复采样逐元素相等，另有单测；不外推跨硬件逐位相同|
|raw有限/零inactive/每区间限幅|PASS（验证器）；生成器不保证全合法|128次离线诊断118合法、10幅值越界；全部保留。端到端全合法生成断言 FAIL，不能声称已解决输出质量|
|非法raw不修复/替换/补抽|PASS|NaN/Inf/非零inactive/越界分别测试；`raw_seed_plan` 原样拒绝，原优化器拒绝消耗槽|
|仅TRAIN scaler|PASS|从17个独立 Task/偏好/族条件精确重拟合原 scaler；尝试 VAL 拟合被拒绝|
|权限范围|PASS（源码边界）|sampler只返回系数与元数据，`physics_steps=0`；参数合同、名义评价、独立Actual五门禁在外层，网络无控制器写入权限|

15 项新增核心数值测试通过；原相关回归72项通过。第一次回归测试收集遇到 namespace 包的相对导入问题，未执行测试；使用 pytest `--import-mode=importlib` 后通过，无旧源码修改。失败和修正命令分别保留在 `docs/audit_receipts/c4a/baseline_regression_01`、`baseline_regression_02`，新增数值回执在 `numerics_01`。

注意：raw生成全合法不是当前模型保证的性质。阶段 C 的核心安全前提是“非法数据被严格拒绝并消耗预算”，不是将越界数据修好后算作通过。阶段 A 没有发现使后续执行不安全的数学、mask、验证器或数据隔离缺陷。

## 条件、数据与容量

896维构成的逐字段清单见 `condition_features.csv`，主要包括：初始planner位置/速度各17；基座/目标姿态各7；两种twist各6；抓取/末端姿态21；base path 110；时长3；场景family3；free_path1；Task要求384；障碍192；区间54；保护时间48；cutoff/系数界各1；新增偏好/参考族/净空/搜索mask/open_slot共18。

在55条监督标签的17个去重条件上，133维发生变化，763维常量，375维始终缺省，1维部分缺省，非有限字段为0。归一化中心矩阵以1e-6阈值计算秩为7；这是此数据样本的秩，不能推断所有合法Task的内在维度。|r|≥0.99的变量对979组（`feature_correlations.csv`），样本少且共享母场景使这些相关性不适合作为泛化结论。

母身份、母场景SHA、seed、Task SHA的 TRAIN/VAL/TEST 交集均为空，分别3/1/2个母场景、6/2/4个Task。历史Task内部的split字段可能仍写旧用途；本轮外部母身份split manifest才是学习分组合同。完整交集在 `data_audit.json`。

55唯一标签身份是 Task SHA+preference+family+精确z，不能称55独立场景：忽略preference后48条，单看z字节31条。52条有route-quality来源、8条有initializer-effect来源，其中5条兼有；因此效果监督只新添3条标签。A/v1=20、A/v2=12、B/v1=11、B/v2=12。各母场景、条件的覆盖很不均匀，不能从总数掩盖无B标签的Task。

精确去重也不等同数值独立：部分B/v2桶中两个z相差约6.7e-18 m，数值秩为0。原去重合同未更改，该事实降低有效监督多样性。`label_buckets.csv` 保留每桶数量、秩与最大距离；没有做以测试结果为导向的容差重去重或重训练。

D有152332参数、S有134412参数，D多17920（13.33%）；两者均两层128 SiLU，D额外输入12维噪声和128维时间嵌入。D约50777参数/独立训练母场景。参数数目不能单独证明过拟合，但独立场景仅3个使扩大网络缺乏依据。

C.3 D/S共用同一监督池、scaler、4000次更新、batch32、同一128000次标签抽取序列及AdamW预算。它是数据/更新配对比较，不是参数量、浮点运算或推理次数完全相等的比较。D每提案20次网络forward，S每提案1次。C.3训练时间和本次单线程CPU推理时间见 `c3_cost_accounting.json` 与 `sampling_audit.json`；小样本计时只描述此环境。

## 离线诊断边界

预定seed=6441001，8组12D噪声，对全部6 TRAIN+2既有VAL的A/v1和B/v2各采8次，共128输出；16次首样本复现检查另计，D总调用144，S比较forward16。所有结果、原始z、幅值、最近匹配TRAIN标签距离、同条件样本间距离及同噪声跨Task敏感性全部保留，无按合法性过滤、无TEST调用、无模型选择。

`offline_sampling_summary.csv` 表明模型输出有数值多样性及条件响应，但这既不证明不同样本可执行，也不证明存在多个分离的合格路线模式。监督来自有限近优路线和有限初值效果，缺少连通性/不可达谷值/不同同伦类证据。固定单次提案只能展示该噪声的一个样本，不能评价整个分布的覆盖率；本次多次离线采样也不产生新的物理合格标签。

进入最小接入诊断的依据：核心数值与数据合同通过；法证指向规则覆盖被替换和轮询预算不足。扩大数据/模型的依据仍未建立。
