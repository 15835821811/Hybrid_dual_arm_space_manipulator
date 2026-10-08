V6.4-A.1结果判定：主里程碑yes；四条全部成功partial；confidence high。

本轮的有限能力里程碑得到支持，claim_supported = yes，confidence = high。把结论扩大为四条全部成功时，只能判为partial；完整成功数是2/4。

R1在1,351个相同反馈快照上，十项参考差异严格为零，并以13,500个真实控制步完成原27s任务。D0、D1使用相同最终源码9fe36eba65b383741d2a58dc94c72d7f82e7e230、相同QP配置及相同进度修复，均通过原任务、执行合同、区间及声明范围的几何验收。独立重算八项任务要求、17/10/7维范数RMS和17个坐标RMSE，与保存结果一致。刚性终态误差分别40.890034µm和61.080189µm，均低于原100µm门槛。

D2在23.28s因RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN停止，仍是任务失败。其原部分trace漏存最后一个20ms边界；单独证据视图只补入已保存actual_full_qpos[-1]，此前全部边界逐值相同，其他数组全部不变。保存力矩重放、参考消费、合同、区间及声明几何范围通过，只支持该已执行段。D3的额外名义几何门禁得到3.475837mm最小间距，低于原5mm门槛，26次违规；它有0个实际控制步。

四个旧终止QP的完整矩阵包含原先隐含的十条PCC端点行。独立LP复核：前两个周期均可行，终止周期即使按原1e-4行容差仍不可行。不能把这种冻结状态不可行解释成任务从初态不可行，也不能把冷启动诊断解释为恢复了未保存的历史dual或积分热启动。

结果归因于原工程repair、名义制动修复及披露的末段进度修复。原权重、scaler和四条控制点不变；原raw负结果、旧33,540步和A/B/C停止决定保持。旧TEST提案已参与调试，因此本轮是开发回归，不支持未经调试的泛化或Diffusion优势。whole-body范围为50Hz边界与配置空间加密4，robot-target覆盖原生500Hz状态；均不是连续时间安全证明。墙钟20ms不作研究门禁，部署仍NOT_MET，硬件仍NOT_ESTABLISHED。

最终交付inventory已确认65个冻结原件/副本、86个来源记录、29个历史manifest条目及50个原工作区源码文件未变。当前目标无需追加实验。建议root完成最终报告状态收尾和sealed manifest哈希验证后，将补充目标标为完成。根本问题、能力里程碑与剩余两条失败必须同时保留。

若以后另行继续，D2支持调查实测状态和参考滞后下的PCC制动余量；这只是后续研究方向，不是本轮已验证的修复。D3则须先解决修复后的几何间距。不得把这两个方向包装成当前已有的全路径能力。

## V6.4-B.1：architecture_pilot_20261006_01

独立结果判定：`claim_supported=no`，`confidence=high`（限于本轮有限计数与身份），研究实施完成。原期望是同信息/预算下，M1提高raw综合候选通过数且冻结工程链完整任务结果不下降。

两模型各完成6000更新、192000参考曝光，配对抽样SHA与曝光完全一致；真实权重按固定VAL最小共有损失选择M0 update250/M1 update500。6新TEST×4配对latent×2模型的48个计分提案均已保存，每模型24独特提案，无raw裁剪/repair/fallback。raw综合通过均0/24，Task要求通过均0/24，K1与any-of-K4均0/6。原raw门禁未进入几何检查的48槽全部NOT_RUN、passed=null。

组件收益：M1速度通过24/24，M0为0/24；M1终端及强制点位置误差描述统计下降。条件消费与障碍置换诊断通过，但3个不计分影子几何均不通过，不能称学会绕障。M1参数、同步DDIM20耗时及训练耗时更多，不能称计算加速；比较的是整个条件化方案，未分离各模块因果作用。

K1工程链保持v2 repair、A.1末段修复与原控制器。M0的6尝试均repair拒绝；M1为3拒绝、3进入actual，完整27s成功均0/6。三M1 consumed prefixes分别11670/9610/9900步，合计31180步；在23.340s/19.220s/19.800s发生私有预演域外拒绝，下一servo步未执行。冻结runner未匹配interval_partial_trace文件名，自动独立prefix评价仍NOT_RUN，evaluation_metrics=null；保存数组描述只用于有限prefix，不补计Task/执行/几何通过或完整成功。

证据边界：13成功消费参考只覆盖6源任务，3TRAIN/7参考、3VAL/6参考，9失败/未执行proposal-only排除；每模型1训练seed、6新TEST，泛化为DATA_LIMITED。TEST可行性未经本轮成功teacher/闭环正例预先证明。M0 CPU/CUDA VAL差8.46386e-6的临时数值诊断miss保留，真实CUDA最小VAL与选权重无误，未重训/重选。两个0/6只表示本轮计分相同，不证明总体非劣、广泛架构无效或M0任务能力。

路由：默认保留M0，保存M1代码、权重和负结果，结束本轮架构假设；`next_experiments_needed=[]`，不追加训练、候选、任务、控制器修补或物理预算。有限研究目标完成与主效果不成立分别标记。旧A.1的2/4及其负结果未变。

本轮research_simulation的20ms控制周期/2ms物理步长保持，墙钟20ms不作研究前置门槛，部署仍NOT_MET。历史C.1补充不替换当前目标或要求重复开发。本轮三次actual失败不是墙钟截止时间拒绝。

完整判定与review trace见 `v6_4/output/architecture_pilot_20261006_01/result_to_claim_review.json` 和 `paper/review-traces/experiment-result-to-claim/2026-10-06_run01/`。Reviewer参与过数据/训练模块，现为评价链交叉复核；这种有限独立性已披露，未冒称zero-context。无paper claim audit，后续论文措辞为provisional，不阻碍此有限负结果交付。数值、失败前缀与可复现命令见REPORT/report.json，最终字节验证见manifest/verification。


## V6.4-C.2 — preference_warmstart_20261008_01

独立result-to-claim判定 partial、confidence medium；有限工程交付无阻塞。真实偏好/family条件Diffusion完成4000更新，VAL选中250；TEST生成8个初值、7个raw合法、1个幅值原样拒绝。历史48候选复用，新教师48槽、TEST112槽，DDIM共136样本，未追加seed、候选、训练或actual。32逻辑actual中22完整并通过原五门禁、10NO_PLAN；12唯一执行、10严格alias，零actual失败、零工具错误。

同八槽A actual：R8=2/4，N8=D8=4/4；B actual且30mm均2/4。D8保留R12完整任务覆盖，但A近质量3/4，test0_minus ΔI=+0.003586139rad/s超0.001；B两项通过、两项N/A。N8 A4/4，B一项通过、一项失败、两项N/A，test0_plus ΔL=+5.268784mm超5mm。D8相对N8两项B路径短3.447899/1.165267mm，净空各少约6.2mm仍大于30mm；这是局部取舍，无总体优越结论。

四Task含负值的实测cold节省下界均值：D8/R12 +137.856530s，D8/N8 +25.047645s，D8/R8 −129.873861s。内外计时夹逼、四worker共享负载与warm分解估计均披露；最大D/N节省含非法初值少跑一次物理，不能单独归因学习。两种八槽方法各失一项R12近质量，质量保持摊销NOT_ESTABLISHED；阶段成本仅记录新教师与训练，不代表完整生命周期。

learning_benefit_established_in_pilot=NOT_ESTABLISHED，default_initializer_decision=retain_C1_rule。研究停止，next_experiments_needed=[]；不扩大模型或预算。deployment=NOT_MET，连续时间/硬件安全NOT_ESTABLISHED。四Task来自两个新母场景、单seed，无总体非劣或广泛泛化结论。无paper claim audit，下游论文措辞provisional；不阻塞本有限研发交付。

完整独立审阅：paper/review-traces/experiment-result-to-claim/2026-10-08_run01；最终报告、对照表及heldout teacher更新位于v6_4/releases/preference_warmstart_20261008_01/snapshot。TEST更新禁止回流本轮训练。
