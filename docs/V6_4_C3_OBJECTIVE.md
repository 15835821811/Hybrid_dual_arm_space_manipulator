# Codex 执行任务书
# V6.4-C.3：搜索导向初值学习、闭环 VAL 与简单回归对照

## 0. 任务性质：从已完成 C.2 继续，不重新打开旧目标

本文件是建议的新实验协议，不是已经执行的结果，也不保证 Diffusion 获胜。

原 C.2 保持：研究完成、学习初始化已运行、总体收益未建立、默认保留 C.1 规则、部署 NOT_MET。B.2、B.3、B.3.1、C.1、C.2 的报告、失败、权重和封存记录均不可覆盖。

本轮只研究一个问题：在完全相同的高层搜索后端和物理安全执行链下，用面向搜索效果的数据训练的两个 Diffusion 初值，经过闭环 VAL 选权重后，能否在新的母场景上优于规则、TRAIN-only 检索和简单条件回归。

“优于”是待检验假设，不是研究完成的前置条件。交付必须包含真实教师搜索、真实模型训练、闭环 VAL、冻结 TEST 和原独立验收；不能以新增脚本、mock 测试或媒体数量代替。

## 1. 开发基点与操作边界

- Repository：15835821811/Hybrid_dual_arm_space_manipulator
- 固定基点：1758e13b01735b80c5a512b81cbc1d5e47a07ec9
- 新分支：v6.4-c3-search-aware-closed-loop-val
- 算法基线：C.2 中继承的 C.1 搜索、C.2 初值接口及原执行/验收链。

在实际仓库根目录中，确认工作区与未提交文件后执行等效操作：

```bash
git fetch origin
git switch -c v6.4-c3-search-aware-closed-loop-val 1758e13b01735b80c5a512b81cbc1d5e47a07ec9
```

已有同名分支时核查身份，不强制覆盖。脏工作区使用新的 worktree 或独立工作副本；不得 reset --hard、git clean 或覆盖未提交用户工作。

分别记录基点、实现提交、实验 producer、训练 checkpoint、数据和配置身份。发布提交不冒充早期实验 producer。未经明确要求，不推送远端、不合并 main、不改写历史。

非代码展示产物限定为 Markdown 报告、JSON/CSV 和最多两张核心结果图；真实模型权重、必要原始 trace/NPZ、命令及验证记录正常保留。无须批量刷新历史视频/PDF，无须为源码不变的历史数据重新运行物理审计。

## 2. 核心职责与记号

### 2.1 冻结的主干

冻结现有：

- 浮动基座双臂名义物理模型、机器人映射及初态定义；
- 原 17 维速度 QP、67 路力矩伺服、制动约束、权重、速度/加速度/力矩限制；
- PCC/胶囊/MuJoCo 几何、原碰撞对及豁免、安全距离、工作域、数值容差；
- 原十步斜坡、微状态预演、动作守卫和失败处置；
- 原任务锚点、姿态、窗口、27 s 时限；
- 20 ms 规划与 2 ms 物理步长，research_simulation；
- v1/v2 参考、每开放区间二维圆盘 20 mm 范数上限、精确 23.98 s 保护截止；
- C.2 初值之后的连续轮询顺序、步长、A/B 共享候选池、排名、工程 tie、缓存与停止规则。

墙钟 20 ms 只作为性能记录，不阻塞研究。deployment=NOT_MET；不宣称真实延迟下执行有效、连续时间安全、模型误差鲁棒性或硬件安全。

### 2.2 方法名称

| 名称 | 含义 |
|---|---|
| R | 原规则初值（Rule）；不是回归 |
| N | TRAIN-only 最近邻检索（Nearest Neighbor） |
| S | 简单条件回归（Simple regression）；它也是学习方法，但不是 diffusion |
| D | 条件 Diffusion |
| 8 / 12 | 每个规划请求的共享候选评价预算，不是 DDIM 次数 |
| A | 低 I_support，工程并列时按原规则选择 |
| B | 原任务和执行资格满足，并达到相关局部净空 30 mm 后尽量短 |

A/B 不是两条机械臂；共用一次搜索及其成本。B 的 30 mm 是额外质量偏好，不替换原硬安全门槛。

### 2.3 学习层的唯一输出权限

每个 Task 的一次高层规划请求只产生两项：

- slot 1：A/v1 路线初值；
- slot 3：B/v2 路线初值。

slot 0（v1 零残差）与 slot 2（v2 几何远离侧 12 mm）对各方法完全相同。

输出沿用 12 维 z 容器，只允许 C.1 开放的前驱/关键区间最多四维非零。family、偏好和 mask 是声明条件，不由模型擅自更改。每段不是每坐标，分别满足二维范数 20 mm。

网络不在每个 20 ms 周期重新采样，不生成力矩，不修改控制器权重，不判断或签发安全证书。

本轮不同时改成“检索+神经残差”新结构，不增大网络或残差幅值，不加 learned critic，不做 RL、Flow Matching、可微 QP/MuJoCo。

## 3. 本轮三个实质变化

1. 教师不仅记录候选自身质量，还对有限的“两初值组合”真实运行固定搜索，取得初始化效果证据。
2. D 与 S 各真实训练一次；两者用相同任务信息、监督池、归一化、训练曝光预算。
3. 预声明两个 checkpoint 通过完整“生成→搜索→最终执行→独立验收”的闭环 VAL 选权重，而不是只按去噪损失或轨迹距离。

旧 C.2 结果不能成为 C.3 新 TEST。以上三项组成新方法包，若有改善，不能仅通过跨轮结果声称其中任一单项具有独立因果贡献。

## 4. 数据来源与分组

### 4.1 重用范围

只复用 C.2 learning split manifest 中原 TRAIN 的三个母场景、六个 Task 的候选事实和近优参考。

- 先按 task/model/config/family/plan SHA 去重物理候选；
- A/B 视图不是两条独立轨迹；actual alias 不是新的物理样本；
- 保留 VALIDATED_EXECUTION、PREDICTED_COMPLETE、FAILED_OR_INCOMPLETE、MISSING_OR_UNBOUND；
- 实际成功并不表示两个偏好都满足；B 标签另检查 30 mm 和原质量资格；
- 原 C.2 VAL/TEST 保持历史开发资料，本轮不加入正式 TRAIN，也不重新充当独立 TEST；
- 不修改历史 TaskSpec 中的 split 字段，通过新的外部分组 manifest 表达本轮角色。

若 portable release 只保留摘要或绝对路径，不得将路径存在于 JSON 当成文件实际存在。优先读取本地可用归档并核对哈希；缺失项如实标记。不得编造完整状态、教师执行或实验身份。

### 4.2 冻结新任务

新生成三个此前未用于任何训练、VAL、TEST 或手工调试的母场景：

| 划分 | 母场景 | Task |
|---|---:|---:|
| TRAIN | 复用原 C.2 的 3 个 | 6 |
| VAL | 新母场景 1 个 | 2（成对） |
| TEST | 新母场景 2 个 | 4（成对） |

仅为来源唯一性选择 seed；沿用 C.2/C.1 已有场景生成规则、55 mm 路线球偏移、原球半径和任务定义。不按成功、失败或模型输出寻找种子。

左右镜像、微扰版本、同源候选整体属于同一个 mother group。新任务在教师和训练前一次性冻结；初态/锚点/合法区间预检失败也保留，不重抽，不移动障碍。

这仍只有三个 TRAIN 母场景、两个 TEST 母场景，应保持 DATA_LIMITED/先导范围说明。不能把数据曝光或候选数量当成独立母场景数。

## 5. 搜索导向教师：只做有限配对实验

### 5.1 两种教师概念必须区分

- route_quality_example：该参数本身在预测或 actual 中有较好质量。
- initializer_effect_example：把两个参数放到指定初值槽后，固定搜索后端在有限预算中取得的结果。

一条最终近优路线不自动等于有用的搜索起点；旧搜索后代的存在也不自动证明它的初值更好。不得将没有运行过的 initializer 配置标成“搜索效果已验证”。

### 5.2 每个 TRAIN Task 构造两个初值组合

共六个 TRAIN Task，每个两个组合。所有组合同时覆盖 A/v1、B/v2 两个槽，共同规则 slot0/2 保持。

**T_local：本任务教师组合。**

仅用该 TRAIN Task 已封存的候选事实：

- A/v1 采用该 family 完整合格近优集合中按 C.2 工程选择规则产生的一个参数；
- B/v2 优先采用满足 30 mm 的短路径参数；
- 某桶没有对应合格候选时，构造阶段使用该槽原规则初值，标记 RULE_CONSTRUCTION_NO_MATCHING_TEACHER，不冒称偏好合格标签；
- 用同 Task 优化结果构造训练 teacher 是允许的离线监督；该 Task 不能作为泛化测试。

**T_transfer：留一母场景检索组合。**

按原 C.2 检索距离和 tie 规则，从其余 TRAIN 母场景的初始近优池取 A/v1、B/v2；排除本母场景所有候选。

桶缺失时按初始化接口返回 UNSUPPORTED_TRAINING_CONDITION，消耗对应槽；不临时增加检索库或代入某个成功答案。除共同 slot0/2 外不做隐式 fallback。

两个组合在任何新教师搜索前生成并封存。不随着 T_local 的搜索结果动态修改 T_transfer。

### 5.3 真正运行固定搜索

每个组合调用原搜索器，预算为八槽，两偏好共享：

6 Task × 2 组合 × 8 槽 = 最多 96 个新的名义候选评价槽。

每次候选评价使用原物理模型、控制器和全部在线守卫，从相同初态及控制器历史重置开始。不能将先前组合的预测终态、对偶、参考积分或热启动历史带入另一个组合。

所有结果冻结：初值内容、raw 合法性、逐候选结果、父子关系、首个合格位置、A/B 最终选中、I_support/L/d、失败与成本。

该阶段不为每个 teacher 再追加 actual；其结束输出仍是名义搜索效果，formal_actual_validation=NOT_RUN_TEACHER_SEARCH。

可只读复用历史规则搜索的八槽前缀作参考，但必须身份一致、来源明示。不从旧十二槽后四槽向“八槽结果”倒灌候选，也不把历史比较当本轮新实际对照。

### 5.4 如何选择搜索效果标签

预先从每个 TRAIN Task 的历史候选池按原规则计算 A/B 质量参考；它只用于离线 TRAIN 标签，不进入未来 VAL/TEST 推理。

对 T_local/T_transfer 采用预声明的分层比较：

1. 八槽后预测完整任务/在线守卫合格的 A/B 端点数量；B 的对应端点还必须达到 30 mm。
2. 对历史有合格参考的偏好，满足 C.2 近质量带的端点数量：A 为 I 不超过参考+0.001 rad/s 且 L 不超过参考+0.005 m；B 为 d≥0.030 m 且 L 不超过参考+0.005 m。
3. 在相同已达标偏好集合上，首次找到相应合格/近质量候选的槽位置越早越好；未命中保留右删失，不能把未知当零。
4. 仍并列时保留两个组合；不要依据 TEST 或临时改 cost 权重挑一个。

分层比较是有限的工程教师规则，不声称全局最优或普遍因果价值。

新增正监督必须满足：对应槽 raw 合法、对应偏好有合格搜索结果，并具有直接种子资格或种子/后代实际参与的证据。若最终只由共同规则种子解决，保存 RULE_ONLY 结果，不自动给未发挥作用的初值“学习收益”标签。搜索后代关系只表示谱系，不自动说明单个初值的因果必要性。

initializer_effect_example 的监督 z 必须是该已评价组合原来送入 slot1/slot3 的初值，不是搜索结束后找到的最优 z。最终解可以另外归入 route_quality_example，但未经重新作为初值评价，不得冒称它已有初始化效果证据。

两个初值在共享池中有相互作用。保留 seed_pair_id、两个原始初值、固定伙伴、共同种子和搜索轨迹；不得将同一次共享搜索拆成两个独立的反事实试验。

B 搜索 NO_PLAN 时不生成 B 成功监督；即使它对应的 A 成功，也不能改写 B 标签。

### 5.5 建立统一 TRAIN 池

组成两个有明确来源的监督子池：

- 原近优 route_quality_example；
- 新有限 initializer_effect_example。

同一 Task/family/偏好/z 去重，保留全部来源引用。建议在有两类样本的桶中以 1:1 选择来源，再按 mother→Task→(偏好,family)→唯一参数均衡抽样；只有一类时使用该类并记录。

D/S/N 共用同一最终 TRAIN 标签池。不要强制剔除零残差，也不平均多个多模态标签成一条人工参考。

若新教师未产生有用初始化证据，报告 SEARCH_EFFECT_SUPERVISION_NOT_ESTABLISHED。只要原合法 TRAIN 数据仍足够运行，可以继续预声明的闭环 VAL 研究，但不能声称已验证搜索导向教师的收益。没有任何合法 TRAIN 监督时停止训练，交付实现和明确数据阻塞，不造假 checkpoint。

## 6. 两个模型：保持简单、共享信息

### 6.1 D：Diffusion

复用已核实的 C.2 数值机制：

- 12 维容器，C.1 search mask 外严格为零；
- 两层 128 MLP，SiLU；
- 原任务条件＋偏好＋family＋search mask；
- cosine100、内部 v-prediction MSE、DDIM20；
- 全新初始化，本轮不在旧 checkpoint 上隐式续训；
- 反归一化后不 clamp、不 project、不补采样。

扩散 v 不是机器人速度。网络内部 mask 是声明生成空间，不能拿 mask 修复模型在合法空间外擅自输出的值。

### 6.2 S：简单条件回归

新增两层 128 MLP 直接预测相同归一化 z，条件、输出语义、mask、scaler与 D 一致；不需要扩散步或随机噪声输入。

在完全相同的逐参考监督池上最小化合法维度 MSE。每次读取一条原始有效标签，不事先构造参考均值，不为 D 专门选择更好标签。

S 可能产生均值化输出，这属于确定性回归的待测行为，不预设它一定失败。报告有效参数量；不故意缩小其信息输入或训练预算。

### 6.3 训练预算

两个模型各一次训练：

- 4000 optimizer updates；
- batch32；
- AdamW，lr=1e-4，weight_decay=0.01，gradient_norm_clip=1；
- 各一个固定初始化 seed，任务/参考抽样索引配对一致；
- 所有归一化只拟合新的外部 TRAIN manifest；
- 保留 update250 与 update4000 两个固定候选 checkpoint。

两组曝光各128000不是独立参考数量。其他训练日志可保留，但不得把额外 checkpoint 送入闭环 VAL，或看到 TEST 后重新选旧250权重。

本轮没有 RL、reward backprop、可微QP、在线自更新；所谓“搜索导向”来自教师和模型选择，不是端到端优化器求导。

## 7. 闭环 VAL：只比较两个预声明 checkpoint

### 7.1 与正式用途相同

对 D250、D4000、S250、S4000，分别在两个新 VAL Task 上执行：

生成两初值 → 原八槽搜索 → 封存A/B选择 → 从初态实际执行选中计划 → 原五项独立验收。

每个 D checkpoint 对每个 Task 恰好生成 A/v1、B/v2 两个初值；同 VAL Task 在不同 checkpoint 上采用同一预冻结噪声。不能生成额外候选后挑最好的两条。S 同样每条件一个确定性输出。

raw 非法仍占槽；共同规则种子可按原规则解决任务，但来源要单列。无计划计入 VAL 分母。

### 7.2 VAL 参考组

两个 VAL Task 各运行一次规则 R12，保留并独立执行其 A/B 输出，作为比较质量的参考。

R12 的结果不得输入 D/S、用于替换生成结果或调整搜索。只在所有 VAL 搜索选择已封存后用于 checkpoint 评分。R12 自身没有计划或 actual 未通过时，对应相对质量记 N/A，不能记通过或硬要求该任务必须本质可行。

### 7.3 精确的选权重顺序

对每个模型分别按以下固定字典序选一个 checkpoint：

1. 最大化最终完整27s且原五门禁通过的逻辑端点数量；
2. 最大化B端点实际达到30mm数量；
3. 最大化在R12有合格实际参考的端点上保持原近质量带的数量；
4. 前面相同，优先raw非法初值较少者；
5. 再比较相同参考资格集合中的首次近质量命中位置，采用有界预算内指标，未命中显示右删失；若需固定排序分数，以8+1作明确的“预算内未命中编码”，不得当作实际第9槽命中；
6. 再比较实测候选物理/预演/几何工作量及端到端规划时间；
7. 完全相同，选更早update。

A近质量：I_support≤R12-A+0.001 rad/s，且L_full≤R12-A+0.005m。
B近质量：完整Task/五门禁、d_support≥0.030m，并且L_full≤R12-B+0.005m。

任务能力和质量优先于成本；非法提案少跑物理或提前失败不能靠耗时小获胜。A/B各端点同时报告，不用总分掩盖某偏好退化。

VAL只有一个新母场景，是有限模型选择，不构成泛化证明。不得按VAL结果修改网络、训练、教师、任务、选择优先级或增加checkpoint。

### 7.4 VAL 预算

- 规则参考：2 Task×12=24候选槽；最多4个actual逻辑槽；
- 模型：2模型×2checkpoint×2Task×8=64候选槽；最多16个actual逻辑槽；
- 合计88候选槽、20actual逻辑槽；
- D正式VAL生成2checkpoint×2Task×2初值=8个DDIM样本。

checkpoint间精确同参考可以严格alias实际证据，但必须满足同Task、同模型/控制配置、同初始历史和同plan身份；不能减少统计上的缺失或增加独立样本数。

## 8. 冻结模型与检索后，运行新的独立 TEST

### 8.1 冻结顺序

完成教师、训练、闭环VAL后，封存：

- D与S各一个选中checkpoint；
- 数据/条件/残差scaler、mask和family schema；
- N的TRAIN库、原条件距离及tie规则；
- 两个模型的初始化种子、TEST噪声派生规则；
- 全部搜索、质量、成本和失败处理协议。

之后才允许启动 TEST 搜索。TEST既不用于选checkpoint，也不进入检索库、教师、训练和搜索停止条件。

### 8.2 初值表

| 槽 | R规则 | N检索 | S回归 | D扩散 |
|---:|---|---|---|---|
|0|v1零残差|相同|相同|相同|
|1|v1远离12mm|A/v1 TRAIN-only检索|A/v1直接回归|A/v1一次DDIM|
|2|v2远离12mm|相同|相同|相同|
|3|v2远离20mm|B/v2 TRAIN-only检索|B/v2直接回归|B/v2一次DDIM|

各方法在搜索前冻结两初值；不随中间搜索结果再生成。N/D/S均不得读取候选质量后决定第2个初值。

N与模型共享TRAIN监督池、条件维度和family/mask要求。沿用C.2的完整标准化条件距离及tie，除非本协议在任何实验前单独明确新的检索版本；不得为帮助D故意弱化检索。

### 8.3 搜索流与预算前缀

每个 TEST Task：

- R12：12槽，保存R4/R8/R12；
- N8：8槽，保存N4/N8；
- S8：8槽，保存S4/S8；
- D8：8槽，保存D4/D8。

共4 Task×(12+8+8+8)=144候选槽。

R8必须在第8槽时封存，不读取R9—R12；后三组真实在8槽停止。A/B共享同一流，不能再乘二。

候选间、方法间物理初态和控制历史严格隔离。研究主计时不跨方法共享已计算候选池；只能使用每条搜索流原有的精确缓存。恢复已完成同一流的缓存不计新的实验，但保留原逻辑和资源成本。

### 8.4 最终 actual

先封存全部TEST流的选择，再执行：

R8-A/B、R12-A/B、N8-A/B、S8-A/B、D8-A/B。

最多4Task×5端点×2偏好=40actual逻辑槽。

NO_PLAN保留0步；一条方法失败不换候选。最终actual必须从Task初态重新执行反馈ctrl+mj_step，不播放预测力矩，不从中途拼接，不把私有预测记为actual。

严格相同plan的actual可alias，分别报告logical slots、unique actual和alias数。跨Task或初始控制历史不同不得alias。

## 9. 效果指标与结论规则

### 9.1 主表

每个端点分别报告A、B：

- 完整27s与原五门禁 /4；
- B实际30mm /4；
- NO_PLAN /4、raw拒绝、actual前拒绝、actual失败；
- actual独立次数与alias；
- 同任务完整配对的I_support、L_full、d_support及基座扰动；
- 预测→actual质量差异；
- 相对于R12的能力保持和近质量状态。

N/A不计通过。基线NO_PLAN不代表任务不可行，方法自己NO_PLAN也不代表碰撞。失败前缀质量单列，不能用短路径或低干预赢过完整运行。

### 9.2 首次命中与效果—成本曲线

记录三种计数：参数构造、消耗评价槽、真实预测rollout。

分别报告首次完整可接受、首次B30mm、首次匹配R12近质量的位置。没有命中是右删失，不填0，不只统计成功样本。

R12基准只用于所有选择封存后的评价，绝不作为TEST在线终止阈值。

从同一条流的4/8/12前缀给出预测曲线；只有实际执行了对应输出的端点可以给actual成功率。不能把4槽预测自动写成4槽实际效果。

### 9.3 两条比较线

1. 同预算：R8/N8/S8/D8，检验生成模型相对于经验检索和便宜回归的实际价值。
2. 缩预算：D8（及N8/S8）对R12，检验任务和近质量保持后的节省。

“8比12少33.3%配额”只是设置，不是有效加速。必须同时保持相关实际能力/质量并实测总成本。

报告较少候选何时来自更好初值、精确重复、非法提案或提前拒绝。不能只展示节省最大的任务。

### 9.4 不将小样本计数称作统计非劣

两TEST母场景、单训练seed只能支持冻结先导。

预声明的候选采用建议：

- 不少于R12已观察实际能力；
- R12有完整参考的端点上保持既定近质量；
- 同8槽不劣于N/S的已观察覆盖，并有预声明的实质性质量或成本改善；
- 默认只在符合上述条件且证据完整时，将D列为下一轮候选，仍不自动用于硬件。

可以将“相近实际质量下，相对N8和S8至少减少10%的规划成本/有效预测工作”设为本轮工程目标；这是新目标，不是安全阈值或统计保证。若质量不保持，该成本比较不构成学习净收益。

D等于N或S但更贵，则保留简单方法。D仅优于R8、没有优于N/S，只能报告相对规则的局部改善。

## 10. 来源、非法提案与搜索后代的归因

记录：

- raw初值内容、生成条件、checkpoint、噪声、生成成本；
- 初值是否合法，非法原因；
- 是否直接选中学习初值；
- 是否选择其搜索后代；
- 是否选中共同规则及其后代；
- N/S来源、候选parent、所有预算消费；
- NO_PLAN、前缀失败、最终独立验收和alias。

新增 source="regression" 必须通过和 diffusion/retrieval完全相同的raw资格检查；不能为了兼容而冒充 source="diffusion"。

C.2的规则初值、合法重复和搜索投影语义保持。神经/回归raw不合法不能调用搜索的圆盘投影函数进行修复；拒绝占槽且不补生成。

谱系说明来源，不等同于因果必要性。每个最终成功归因于“该初始化方式+同一搜索器+原控制器”，不称raw端到端策略成功。

## 11. 时序、资源与总预算

### 11.1 总预算表

| 阶段 | 新候选评价槽上限 | actual逻辑槽上限 |
|---|---:|---:|
| TRAIN教师：6Task×2组合×8|96|0|
| VAL规则R12：2Task×12|24|4|
| VAL模型：2模型×2checkpoint×2Task×8|64|16|
| TEST：4Task×(12+8+8+8)|144|40|
|总计|328|60|

所有上限均包括失败、预检拒绝和无资格槽，不承诺全部需要物理积分。旧数据只读导入不计新增运行。

对应主预测物理步理论上限：328×13500=4,428,000。
最终actual主物理步理论上限：60×13500=810,000，alias和NO_PLAN可降低。

原微状态预演、独立力矩重放和几何查询是额外工作，必须分账，不得从上表推断它们已经包含。

- D和S：各一次4000更新；
- 正式D闭环VAL生成：8个；
- 正式D TEST生成：8个；
- 可选冻结错配条件只读诊断：最多8个，无搜索/actual；
- 额外生成smoke：最多8个；
- DDIM生成样本总上限：32个，每个20步；
- 网络loss前向不是DDIM采样，不混计。

本轮闭环VAL增加了成本，但只对两个固定checkpoint执行，不扩展成16轮模型搜索。预算不因失败或看到潜在胜率而扩展。

### 11.2 成本测量

主计时从规划请求开始、含输入检查/条件编码、检索或模型冷加载、推理、raw检查、搜索、候选预测和selection封存；外层进程起止另存，以免遗漏启动。

正式VAL/TEST方法计时默认顺序执行，同机器同负载约束；不同时安排其他重型训练/视频编码。方法顺序采用预声明轮换，避免始终让某方法占据首轮冷启动。若启用并行作业，保留分组/共享负载说明，不能据此声称稳定单次时延优势。

冷加载与驻留模式分开。只实际测过cold就不推算warm实验通过。A/B共享一次搜索时间不能重复相加。

教师、两个模型训练、全部闭环VAL、数据整理和测试评价的离线成本都单列；发生在VAL的actual不是生产推理成本。

不在质量保持未建立时给“若干任务即可摊销”的肯定结论。不要把累计worker服务时间和阶段makespan混用。

## 12. 必要代码修改与复用

优先复用：

- v6_4/preference_teacher_dataset.py
- v6_4/preference_diffusion_warmstart.py
- v6_4/route_initializers.py
- v6_4/continuous_route_optimizer.py
- v6_4/evaluate_preference_warmstart.py
- 原route candidate evaluator、residual_execution及独立验收。

以实际文件与签名为准，不在任务书中假定所有新入口已经存在。

建议新增最多四个核心模块：

1. search_effect_teacher.py：教师组合、八槽试验和证据化标签。
2. simple_warmstart_regression.py：小型S模型、训练与immutable推理。
3. closed_loop_warmstart_validation.py：两个checkpoint闭环VAL和冻结选择。
4. search_aware_warmstart_experiment.py：协议、调度、TEST、报告。

D新增训练协议应是C.3 wrapper/config，保留C.2冻结默认。仅为S新增合法source和透明日志字段；不得改低层控制数值逻辑。

原C.2初始化器省略S时，必须保持原结果语义。使用mock/保存输入验证，不为证明历史结果再跑全部旧实际。

## 13. 必需测试

软件测试尽量使用mock evaluator和小数组；物理测试只在预声明budget中运行。

- 物理候选去重、A/B视图与actual alias分离；
- 新外部分组不改历史TaskSpec，镜像母场景不跨组；
- 留一母场景检索确实排除同母场景；
- 教师搜索效果与候选本身质量分层，rule-only/NO_PLAN不被伪造为神经成功；
- B标签不把仅有A成功或不足30mm的结果视作B已满足；
- 两个模型相同条件/schema、TRAIN-only scaler、相同监督范围；
- S source通过原raw检查，不能绕过inactive/范数/锚点/速度；
- 12维容器与最多4维搜索mask一致；
- D采样次数固定，非法raw不修复、不补抽；
- 两个checkpoint白名单与闭环VAL排序固定；
- VAL选择不读取TEST文件或字段；
- R8封存后不受R9—R12影响；
- 状态/控制器缓存隔离、同一候选内部热启动保留；
- 工具/管线错误、真实无方案、预演拒绝、actual失败分别记录；
- 消费非有限命令必须失败，未消费拒绝行不污染已有前缀；
- 当前安全几何范围不升级，缺测NOT_RUN/null；
- 别名条件、预算恢复与完整日志终态；
- 成本包含非法槽与NO_PLAN而不奖励提前失败。

不要一键更新历史hash或容差取得全绿。当前测试与历史冻结测试分开入口与身份。

## 14. 执行阶段、命令与交付

建议按三个提交推进：

### 提交一：协议与接口

实现新的外部split、教师双组合、S source与回归、闭环VAL评分器、预算和mock测试。此时不改变默认C.2后端行为。

### 提交二：教师搜索、训练与闭环VAL

96槽以内教师实验；D/S各一次真实训练；两个checkpoint闭环VAL；保存并封存模型、scaler、N库和正式配置。

### 提交三：独立TEST与结果

完成R/N/S/D搜索和实际结果、原独立验证、源与成本记录。无论正负，完成固定程序后封存，保持默认C.1直至有足够证据改变。

新CLI建议包括：

```text
prepare
import-history
teacher-search
build-dataset
train
closed-loop-val
freeze-models
test-search
execute-test
validate
report
run-all
```

以下只是实现后的预期调用，未声称仓库已有此命令：

```bash
python -B -X utf8 -m v6_4.search_aware_warmstart_experiment prepare \
  --output v6_4/output/search_aware_warmstart_<unique_run_id>

python -B -X utf8 -m v6_4.search_aware_warmstart_experiment run-all \
  --run v6_4/output/search_aware_warmstart_<unique_run_id>
```

`run-all`已完成阶段仅核验后跳过；失败候选不因恢复运行被补跑。中断且证据不完整单列技术状态，不伪装科学失败；需要新的技术诊断时另计，不消失于日志。

建议产物：

```text
plan.json
source_identity.json
split_manifest.json
teacher_search/
dataset/
models/D/
models/S/
closed_loop_val/
model_selection.json
frozen_test/
test_search/
actual/
validation/
result_tables/
REPORT.md
manifest.json
```

REPORT至少包含：历史边界、实际数据量与证据等级、教师搜索新贡献、两个模型训练及闭环VAL选择、独立TEST各端点表、逐任务质量/成本、来源谱系、所有失败及缺测、最终默认决策。

新增最多两图：预算—覆盖/近质量曲线；actual质量—总规划成本图。不是媒体刷新任务。

## 15. 最终状态必须拆开

```text
research_execution_completed
implementation_completed
search_effect_teacher_evidence_available
training_executed_D
training_executed_S
selected_checkpoint_D
selected_checkpoint_S
closed_loop_val_completed
independent_test_completed
full_actual_task_by_method_and_preference
near_quality_preserved_by_method
learning_benefit_over_rules
learning_benefit_over_retrieval
learning_benefit_over_simple_regression
default_initializer_decision
deployment = NOT_MET
continuous_time_safety = NOT_ESTABLISHED
hardware_safety = NOT_ESTABLISHED
```

技术未完成就如实标记partial/blocked，不能只因有报告就写completed。全部有限研究程序完成但没有优势，可以research_execution_completed=true、learning_benefit=false/not_established。

不得：扩大预算追求正结果；根据TEST重训或重选250；减小安全裕度；去掉原守卫；偷换K1；把rule-only或搜索后代称raw模型成功；把预测样本当独立actual；把别名当独立试验。

## 16. 下发给 Codex 的一句话

冻结1758e13的物理模型、路线表示、高层搜索和安全执行后端，仅让Diffusion生成两个高层初值；用有限教师组合的实际搜索效果补充监督，与同信息、同数据、同训练预算的简单回归共同通过两个固定checkpoint的闭环VAL，再在新母场景上与规则和检索公平对照。主目标是保留真实任务与质量后更少搜索，而不是提高神经网络控制权。按328候选评价槽、60最终actual逻辑槽上限完成并保留正负结果，所有来源与失败分开记账。

---

## 来源依据与本设计的关系

本任务基于已发布固定提交：

- C.2报告：`v6_4/releases/preference_warmstart_20261008_01/snapshot/REPORT.md`
- C.2模型选择：`v6_4/preference_diffusion_warmstart.py`
- 两初值接口：`v6_4/route_initializers.py`
- 搜索与参数协议：`v6_4/continuous_route_optimizer.py`、`v6_4/route_optimizer_protocol.py`

核对基点均为`1758e13b01735b80c5a512b81cbc1d5e47a07ec9`。

已证实的历史信息包括：C.2 D8-A与N8-A都为4/4，R8-A为2/4；D8有一项A近质量未保持；旧VAL主要看参考合法性、标签距离与v-MSE；override仅在slot1/3；没有学习优势声明。

本文件中的C.3教师组合、两checkpoint闭环VAL、简单回归、分组及预算是新建议，不是历史源码已有能力或已执行证据。研究中需要新增实现并据实验收。

结束后把当前版本作为单独分支上传github，并且把该分支下的所有可视化刷新。把五视角、连续体测视角视频，末端轨迹跟踪误差等这些可视化也同步更新