总体判定：claim_supported = partial；confidence = high；blocking_issues = []。本轮有限研究可以作为诚实的部分结果交付。已建立的是固定开发网格上的可执行参考差异、少量任务相关质量收益与回顾性条件选择潜力；未建立 v2 普遍优越、学习策略有效或 Diffusion 必要性。本判定不把研究执行完成等同于 GitHub 发布完成。

审阅身份与输入边界：我没有实现 v2、reducer 或绘图模块；曾完成 T0 保存数组诊断及本轮三个增量 raw 审计，正式审阅明确复用这些已认证证据。另一个只读 reviewer 独立从 28 行 CSV 重算排序、均值、门槛和成本，结果一致。正式输入为 run03/prompt.md（SHA256 889f7c568f69a2f3d342230605ddf020b24373f3359bfd48624866a16da21e9f）与 inputs.json（SHA256 6395fc3f99faa927eb855e9eafc9283a3d1cd6ee12b6314e55bd694b12f18982），run_id 为 execution_aware_route_teacher_20261007_01，实际算法 producer 为 d4464c8ae2aa7913a730ebbd775917e7a3b1af71。审前与审后核对 26 项当前原件的 SHA/bytes，并核对其中 22 项精确快照；全部一致。postprocessing producer 与 actual producer 的身份分开，未以发布 commit 冒充实际运行来源。

复用的 raw 审计覆盖全部 28 槽，分别是 43mm 审计 701aaa7dad6dd727ea992b162ef442099959424218e2960dfd186da730406afb、55mm early 审计 da383de04ce68d9ec2c908abf70c5e4daba4b61a962be184a38c15b5248fb179、55mm remaining 审计 c718079e7ab2eaccd96a246702bac8e0f7f1552b0d04e396caae367bcb5785d4。它们此前已把终态记录绑定到原 TaskSpec、plan、attempt、独立 evaluation、manifest、trace、fresh replay、保存净空序列与成本账本，并从原 NPZ 纯数值复核任务点/最终状态、干预量、路径、基座漂移、reference 消费及响应。本次将 quality_matrix 的 28 行逐项对照这些独立重算结果，CSV 的每个字段对照 matrix，22 条新向量的 17D/10D/7D、28 条 full/route 世界 3D 响应，以及 plot_manifest 的横向 RMS 再作交叉核对。没有重新运行物理、几何、QP、模型、reducer 或绘图。

1. 对“补充研究是否按协议完成”的判断：支持。一个母场景、四个开发任务、每任务七个固定模式构成完整 4×7 网格；六条严格旧证据复用与 22 条新 actual 的来源、槽位和费用一致。28/28 原完整 Task 与五项独立 gate 通过，28/28 质量可比，24 条非零参考全部完整。固定槽、终态清单、源绑定和账本没有显示候选替换、actual 重试、训练、模型采样或 seed 搜索。本次没有失败、拒绝或质量缺测，因而所有质量均值都使用四个任务；成功率分母仍为预声明槽数。保留槽规则不意味着失败后能填完整质量值，安全 actual 后质量缺测也不能获得 A 类“完整成功对物理失败”的优势证据。research_delivery_complete=false 与审查状态 pending 是审阅/发布阶段 metadata，不是科学失败。

2. 对“v2实际消费并保持任务锚点”的判断：支持所报告的有限运行。16 条 v2 的 consumed version、definition、plan SHA 与 z 均与冻结候选一致；独立 raw 审计中解析 p/v 与保存消费字段的最大差为 4.44e-16。原 Task 点同时刻位置/姿态及最终时刻终态检查都通过；必要参考测试收据支持 C² 接合、23.98s 保护以及 v1/零残差原路径保留。新记录的向量使 17D 干预量可以拆为连续体 10D 与刚体 7D；其平方 RMS 恒等式成立，17D 与原标量最大差为 5.55e-17。该标量是 box-clipped 原名义速度与实际 selected command 的差，不能改成 raw unconstrained 差，不能称为单个路线球贡献。旧六槽没有这些向量及球行日志，维持 NOT_MEASURED_OLD_EVIDENCE 正确。

3. 对“非零/20mm参考产生实际路线差异”的判断：支持描述性结论。四任务预声明远离侧 20mm 模式的 route actual−同任务 z0 actual 世界 3D RMS 分别为 13.706981、13.698688、14.808427、14.806674 mm；全 27s RMS 分别为 4.946163、4.943171、5.343620、5.342987 mm。每条 route 指标使用固定 [8.461298845015024, 11.977496008196567]s 窗口：175 个已消费 planning ticks、1758 个保存 native states（8.462 至 11.976s）。因此结论来自整个预声明窗口的统计，不依靠中点切片。reference−base、actual−base、actual−reference、actual−z0 的含义不同；RMS 差异不等于每一时刻跟踪增益、最小必要避让量或单障碍因果归因。selected task velocity 也不是物理 actual velocity。

4. 对“v2降低原 I_route 且不存在其他代价”的判断：只部分支持前半句，不支持普遍改进或无代价。八组同任务、同 z、同 12mm 幅值的 v2/v1 对照，只有三组 v2 的 I_route 降低，五组升高，八组全程路径全部增加。最大降低是 0.001013347141710419 rad/s，即相对 v1 的 0.940055%；不能由 v2_20 对 v1_12 的跨幅值比较单独归因参考形状。四任务严格 teacher 相对各自 zero 的改善为 2.791079%、7.928977%、0%、3.658899%，没有任务达到相对 zero 的 10% B 门槛。43mm 负侧和 55mm 负侧严格最低选择相对 zero 的全程路径分别增长约 5.97mm 和 13.88mm。净空、路径和基座漂移存在不同方向的变化，未建立 Pareto 支配；原安全/Task gate 通过不等于所有代价更优。

严格最低与工程 tie-break 的独立重算如下：

|任务|严格最低|严格 I_route rad/s|tie-selected|tie-selected I_route rad/s|
|---|---|---:|---|---:|
|43mm c+|EA_02 / v1_minus12|0.1055704751670103|EA_02|0.1055704751670103|
|43mm c−|EA_12 / v2_plus20|0.10318376993273673|EA_12|0.10318376993273673|
|55mm c+|EA_14 / z0|0.10528926410279534|EA_16 / v1_minus12|0.1053687726948893|
|55mm c−|EA_26 / v2_plus20|0.10333562292873825|EA_22 / v1_plus12|0.10390287236664804|

分组按组最低值锚定 0.001rad/s 范围，不作传递式连组；组内按全程路径、系数范数与 mode candidate_ID 排序，均符合冻结规则。55mm c+ 的严格最佳为 zero，不能强制非零监督标签。tie-band 是逐任务工程分辨率，既不是统计等价，也不是置信区间。

5. 对“有限条件选择具有超越固定模式和几何规则的潜力”的判断：支持有限回顾性潜力，未支持已实现 policy。七常量模式、zero、几何规则、strict oracle 与 tie-selected 全都在相同四任务上比较，完整矩阵与共同完整子集一致，没有失败成本填补或跨分母均值。独立重算 strict oracle 均值 0.10434478303282016rad/s，最佳常量 v2_plus12 为 0.10786733022218851，zero 为 0.10830520387170461，冻结几何规则为 0.10542694957725102。因此 V_cond = 0.003522547189368355rad/s，strict oracle 相对最佳常量、zero、几何规则分别降低 3.265629%、3.656723%、1.026461%。这不是用最差固定方向作分母。

tie-selected 均值为 0.1045064725403211rad/s，不能与 strict oracle 混用；它相对几何规则仅低约 0.000920477rad/s（0.873095%）。strict oracle 相对几何的差也仅 0.001082167rad/s。不能拿逐任务 tie-band 对这些平均差作统计检验，但这些数值说明可用优势较小。V_cond 用同一四任务结果同时选常量和逐任务最小，是有限集合上的事后优化量；没有训练、在线选择程序、独立 TEST、置信区间或独立母场景，因此不能外推实际 conditional policy 的平均收益。

6. 对“原 B.3 门槛与旧负结论”的判断：当前七问表述正确。原协议要求分别报告 zero 和 opposite 比较，以 A 或 B against zero OR opposite 资格，再要求两侧合格优选方向反转；旧字段 require_comparison_with_zero_and_opposite 的含义已在 actual 前 canonical clarification 中明确，不能自行改为 AND。A 在本轮没有贡献，因为所有 actual 完整成功。唯一通过原 OR+反转的是新增 43mm/v2_20 组：c+ 的 v2_minus20 相对 opposite 降低 14.923696%，相对 zero 仅 1.877804%；c− 的 v2_plus20 相对 opposite 降低 13.886486%，相对 zero 7.928977%。其余五个同幅值/距离/版本组不通过，所有另报 supplementary AND 均不通过。

这不能写成“所有补充组都不通过”，也不能写成“相对 zero 超过 10%”。同时它不重开或改写旧 43mm/v1 六槽结论 ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION。新增参考族满足有限原谓词与旧已封存六槽的负结果属于不同实验输入，现有报告分开处理正确。

7. 对“是否另立学习试验”的判断：支持本轮结束且暂不训练的决策。相对简单几何规则的有限 strict oracle 优势约 1.03%，tie-selected 更小，加上一个母场景、事后选择和多代价冲突，尚不足以证明复杂生成模型必要性或同预算收益。现有数据适合保留完整质量向量、zero 和近优组、简单几何/检索基线。若将来确有独立工程需求，可另立按母任务分组的独立 TEST 协议，预声明最小有用效应，包含 zero、冻结几何、检索、幅值匹配随机、正确/错配条件及固定 K1；K4 应单列全部训练、sampling、QP、preview、replay 和几何成本。这是未来方案，不是本轮待执行事项，也不是本轮交付 blocker。

成本重算：22 条新增 actual、private preview 与独立保存力矩 replay 各 297000 步；原 QP 与 preview 各 29700 次。执行 native geometry 为 737681587（actual 361052677，replay 376628910），附加 route-quality 为 18415364，初始/锚点声明为 5978，总数 756102929。两个 ledger 的 phase 计数独立，started = returned + raised，所有 raised 为 0，没有把 route-quality 重复加进执行计数；六条旧证据复用新增成本为 0。必要日志检查的四次原约束 QP 与四次既有无约束名义计算另列，不能加成新 actual 或声称在线增加第二次 solve。两张 PNG 和本次审阅的物理、几何、QP、模型调用为 0；未生成新 PDF/video。

两图已逐张目视核对。图1明确是固定世界第一横向投影，分别展示 reference−base、actual−base、actual−同任务 zero；图2保留全部七候选、原 17D I_route、相关连续体—球净空、完整路径与单轴窗口 RMS，柱与右轴路径符号有明确说明。plot_manifest 中 28 槽的 I、净空、路径及 actual−zero 横向 RMS 与 raw 审计一致。报告的世界 3D RMS 表也与 raw 审计一致；两者不能混称。图形显示实际路线差异，并未把参考曲线或 selected command 冒充实际轨迹。

证据缺口是实质性的结论边界，但不是伪造测量：路线 minimum 的保存状态、时刻、距离序列已提供，局部 minimum 对应的 pair witness 没有保存，28 条均明确 NOT_SAVED_FOR_ROUTE_WINDOW_MINIMUM，且没有拿全程 global witness 替代。本轮不得新增几何查询去补它。旧六槽的 10D/7D 与相关球活动日志同样不能从标量反算。所有相关球行活动/瞬时绑定/lookahead 绑定只能按已有 source 和 tick 并集描述，不能将 17D 总干预归因单球。本次部署维持 NOT_MET，20ms 规划与 2ms 物理不是硬实时认证；独立保存几何/区间检查也不构成连续时间安全证明。

建议可用 claim：“在一个母场景衍生的四个固定开发任务及七种预声明参考上，控制器感知参考可产生可比的实际路线差异。提前平台在同幅值对照中不一致优于原参考；有限回顾性 teacher 相对最佳常量与冻结几何规则分别呈现约 3.27% 与 1.03% 的 I_route 潜力，同时存在路径及其他代价冲突。结果尚不支持已学习条件策略、Diffusion 优势、生成模型必要性或泛化。”

未发现需要修改本次科学输入的具体 blocking issue。final_conclusions.json 的 independent_judgment=PENDING_SEPARATE_REVIEW_OF_THIS_EXACT_SCIENTIFIC_PAYLOAD 是不可变审前输入的创建阶段标记；由本 verdict 与 report.json 允许变化的交付/审查 metadata 记录最终状态，不能把该旧标记当成未审科学结果或修改它。GitHub handoff、封签与远端验证由 executor 后续收据处理，本审阅不预先认证发布完成。

paper/PAPER_CLAIM_AUDIT.json 不存在，因此 paper_claim_audit=unavailable，downstream_writing_verdict=provisional — no paper claim audit available。缺失的是论文到证据的后续审计，不阻止本轮有限研究的诚实交付；不得把本 partial verdict 舍入为完整方法或学习贡献已获证实。本轮不需要追加实验来“追正结果”，也不触发训练或 ablation。
