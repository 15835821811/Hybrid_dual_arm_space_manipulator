# 独立结果到主张审查：rep_fidelity_001

日期：2026-10-21。角色：新委派的 paper architect/reviewer，独立只读核查。审查对象：冻结计划 `ff_prior_rep_fidelity/docs/EXPERIMENT_PLAN.md` 和 `artifacts/v7_1/rep_fidelity_001` 的原始保存证据；实现源码 commit `ec59aa1f764f1365de8fd4671ae097875e7cb3c0`。

**claim_supported: partial；confidence: high（仅限本次冻结单案例判决）。C1: no。C2: partial，严格限定到 rank19 的单案例配对观察时为 yes。工程有界运行和收尾可以判为完成，无须为了将科学判决变成 yes 自动补做实验。**

`paper/PAPER_CLAIM_AUDIT.json` 不存在，`paper_claim_audit: unavailable`，此判决标为 **provisional — no paper claim audit available**。这是已实际完成的新 reviewer 审查，不是 `[pending external review]`；provisional 指缺少论文到原始证据审计，不表示本次审查未执行。

## 实际核查，而非自动 PASS 转述

仅本 trace 新增文件。未调用任何项目物理/优化/旧审计/resume 入口；独立脚本只 import Python 标准库、NumPy 和 SciPy，读取保存数组、JSON 和源码。`read_only_check.py` 生成343项检查，`closure_check.py` 追加14项必要收尾与执行门核查，合计357项，无失败。详细可复查记录分别在 `independent_saved_checks.json` 和 `closure_checks.json`。这个计数是确定性证据一致性检查数，不是357次实验或独立统计样本。

- 127个冻结源码当前 SHA256、执行 zip 内容和静态本地 import 闭包一致；旧绑定 files 与 previous_source_hashes 逐文件重算匹配。继承 canonical physical identity、acceptance、initial state id 与旧 physical contract 精确一致。场景任务 initial id 与 canonical id 是不同层次；新任务完整初态 id 正确匹配 task contract。
- 旧 rank6 基与冻结 codec 数组精确相同。codec 记录 `fit_split=TRAIN`、10000个 fit ids，当前任务 hash 不在其中。新增 rank10/14 基从该固定基、终端单位向量和固定 DCT 顺序双遍正交化重建；基正交性、所有12个 representation.npz 中已保存的 teacher/base/basis/raw/input/smoothed/clipped/projected/output 数组均复核；未发现 DEV 拟合代码或新训练。旧10k条数据的历史 TRAIN 分割归属未在本轮逐条重新审计。
- base 的公开初始化时间曲线形式复核；其 Jacobian/质量矩阵产生的末端值依赖冻结实现及旧输入，本轮没有运行物理重新生成。所有候选 reference hash、控制数组、固定12项 manifest digest、基内投影、原运动快门全部从代数重算。
- raw 与 preserve 每个 rank 精确相同，rank19 raw 与教师精确相同。历史模型复用仅 r06_raw/pure_codec 和 r19_raw/teacher，验证 JSON、Z、任务身份精确一致；其余4个 preserve 均准确指向本轮 owner。没有旧 prepare 复用。
- 从保存 `prediction.npz` 的双臂世界末端矩阵重算位置/姿态误差；每级保存 grid 的终点与全几何范围收敛差重算；每级几何扫描 qpos 与该 grid 的偶点+奇点顺序精确对应，minima 与逐扫描 JSON 一致；最终密度和最小间隙与报告一致。所有候选 MODEL/原目标合并资格判定一致。自动审计 PASS 未作为判决前提。
- 12逻辑=8唯一；6新模型验证+2旧精确模型复用+4同轮重复。唯一模型合格选择严格等于 `['r19_raw']`；2/12逻辑合格实际只有1/8唯一合格。其余7条唯一路径 outcome 为未选中，无依据称其执行失败。
- 新初态保存836个 integration 数值、mask8191；独立重算 state_id 为 `3ffd4a33b1dc9c45632b2442077dac4bf45c2f13a7af222d183bc3746f96e7db`，run与execution保存初态精确相同，native首帧时间/qpos/qvel匹配初态。
- 原生与重放两套轨迹均实际存在；各13751状态、13750×67控制。qpos/qvel/act/time/warmstart/ctrl保存数组逐值相同，五项报告误差0得到核对；27.5秒按0.002秒步长完整。执行几何记录含13751原生+13750中点=27501；原生状态逐值一致，中点按两个自由体四元数测地线中点及其他坐标线性插值独立重建，误差在2e-12内。
- 从原生保存数组重算世界目标保持段、17↔67映射、关节范围、实际导数、参考q/qd/qdd控制时序、跟踪、流形、原始力矩限值和无裁剪。raw torque最大限值占比0.038538，动量峰值由保存的base_generalized_momentum独立汇总，与报告一致。
- 对照 `ff_prior/physical_audit.py` 与新版逐行 diff，变化为初态、重放、几何记录及 output 参数传递。原 `track`、步进、完整重放重置方式、几何查询、门与阈值未被替换；replay只在开始reset，未逐步复制native状态。新轨迹与旧教师native文件hash相同是确定性同输入的预期现象；仅凭相同字节不能证明发生了新执行，新执行证据还依赖冻结代码、此次ledger和运行记录，不能把字节相同当成独立随机复现。
- 从6条新验证的grid文件直接计出34次耦合、96768积分区间；双执行数组计出27500物理步和1次torque rollout；账本16个owner工作单元完成、inflight为空、无未知调用标记、无run lock。137.343秒、约1.294GB在预算内。两个数学库线程由launcher环境设置，不等于CPU硬隔离。
- 唯一 completed resume回执为NO_WORK_ALREADY_COMPLETE；before/after均235项且相同，并与当前235文件逐项重算hash一致。本审查没有再次resume。

## per_claim 与直接结果

下表位置/姿态为双臂最大值，间隙负值表示穿透；完整双臂值在机器可读核查结果内。raw/preserve是相同数组，不是两个样本。

|维数|分支|位置mm|姿态deg|最小间隙mm|世界目标|完整资格|
|---:|---|---:|---:|---:|---|---|
|102|raw/preserve|208.473923|16.658626|-48.154448|失败|失败|
|102|legacy|206.923009|16.301084|-47.386088|失败|失败|
|170|raw/preserve|10.611620|0.392581|-36.188479|失败|失败|
|170|legacy|15.609902|0.485298|-56.802091|失败|失败|
|238|raw/preserve|3.617438|0.145242|-17.903914|通过|失败|
|238|legacy|15.556743|0.483852|-56.044861|失败|失败|
|323|raw/preserve|1.972966|0.035780|+0.036019|通过|通过|
|323|legacy|15.555535|0.483811|-56.074517|失败|失败|

**C1: no。** 预注册成功标准是至少一个170/238维候选全部合格并通过真实执行。两个维度均无合格候选，未执行；238维终点通过不能弥补几何穿透。只支持“此固定 enrichment 降低表示/终点失真，但仍未保存完整资格”，不支持170/238维可用，更不支持所有紧凑表示不可能。

**C2: partial。** rank19同一raw输入上，preserve精确保留已合格路径且新执行/重放通过，legacy失去世界目标和避障资格。这支持“在这个完整表示正对照上，无条件平滑会破坏资格；保留策略避免该破坏”。rank6/10/14两分支完整资格都失败，按计划不能称这些rank上取得资格收益。四个preserve都走已过快门的直接复制分支，未测试需要缩放/修复的preserve分支。因此面向整个预定rank集合的C2只能partial；若明确将C2改写成rank19这一个配对事实，可记yes。

rank10/14末端控制点与教师差仅4.4408921e-16rad，但世界终点有差异。保存结果与自由漂浮的路径依赖解释相容；本轮没有实验隔离所有动力学因果，也不构成微重力建模错误的证据。各分支最终网格不同：r06 raw8192，r10/r14/r19 raw16384，legacy4096；均按同验收规则停止，不能把表写成统一网格下的精细数值效应比较。

唯一执行的保持段最大位置1.965991mm、最大姿态约0.035047°；最小保存几何间隙+0.02696885mm。原生67通道与独立重放证据闭合到当前冻结仿真门；这些小间隙不提供扰动安全余量。

## what_results_support / what_results_dont_support

支持：本单例在完整323维表示和保留策略下贯通模型、原生力矩执行和独立重放；预定170/238维失败有直接几何证据；rank19平滑分支破坏合格性；本次有界工程流程和可恢复收尾完成。

不支持：学习、Diffusion相对优势、自主规划、TRAIN闭环成功、跨任务/随机种子泛化、硬件有效性、连续时间碰撞证书、扰动鲁棒性、CPU硬隔离、7条未执行路径的执行失败、把重复项算作独立成功，以及跨旧prepare/新prepare输入顺序比较的单因素因果结论。

## missing_evidence 与实际限制

1. 不存在任何合格170/238维输出；这是预定实验的负结果，不是收尾遗漏。不得通过重标、额外搜索或自动补实验绕过。
2. 本轮未重新计算每个扫描点的MuJoCo距离、FK或质量矩阵。几何距离和模型标签的物理值依赖已保存原计算及冻结代码；本审查核的是它们与状态/扫描顺序/阈值的一致性。记录的ee不能单独证明重新FK的正确性。
3. 保存了动量本体峰值，但未保存每时刻质量矩阵或momentum_exchange数组；exchange scale/阈值及无外力、无隐藏重置依赖已审阅源代码和原执行报告，不能宣称本轮纯数组审查独立重建了全部物理门。
4. 旧rank6 TRAIN分割、公开base Jacobian来源仅核绑定和实现，没有重跑旧训练/初始化物理。本次新增基不依赖DEV拟合可由明确数学构造直接核验。
5. 缺论文claim audit；泛化、多任务、学习、扰动测试均未设计为本次交付要求，不能把缺这些证据误当作当前工程未完成。

## suggested_claim_revision / next_experiments_needed / routing

建议主张：“在固定07五障碍APPLICATION_DEV诊断上，预定170/238维时间基未能保留既有合格路径；238维虽保留世界终点，仍出现穿透。完整323维身份对照经保留处理通过新原生力矩执行及独立重放，而同输入无条件平滑使其失去资格。”

已检查 `ff_prior_rep_fidelity/docs/ANALYSIS.md` 草稿；计数、阈值、非统一网格、路径依赖措辞、窄间隙和非学习边界与证据相符。建议最终审查段采用上述 C1 no / C2 partial（rank19限定yes）并保留provisional标记。其“下一阶段宜…”属于工程建议，不是由一条路径证明323维训练更优。

本轮完成无需新增实验。若用户以后选择继续：先在独立TRAIN任务上固定323维基线和保持策略，做有界重建→采样→同门验证闭环；若仍以紧凑表示为目标，应在TRAIN上设计替代表示后另行冻结评估，不能继续在07上搜索并称为泛化。任何下一轮均只建议、不启动。

路由：记录负结果与限定正结果，完成本轮工程交付；不进入“紧凑表示已证实”或“学习收益已证实”的写作路径。无未决物理工作；保留以上证据局限。
