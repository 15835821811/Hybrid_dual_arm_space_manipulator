继续本线程用户2026-10-22授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator；有效目标DIFFUSION_GOAL_CURRENT.md、路线ff_prior_fullchain/docs/ROADMAP.md、真实阶段ff_prior_fullchain/control.json。长期goal保持active；旧卡片初始P1文字过时，以control为准，不能为改文字提前complete。

当前独立后台artifacts/v7_1/temporal_learning_001已实际启动，源码681965a0，协议ff_prior_temporal_learning/docs/EXPERIMENT_PLAN.md，入口.\ff_prior_temporal_learning\job.ps1 status|report|resume。每次先只运行.\ff_prior_fullchain\job.ps1 status，核PID创建时间与心跳。RUNNING且健康则安静结束，不读中间结果、不分析/重审、不重复start/resume、不另开重worker。状态不变不通知；完成、失败、真实阶段切换或需要处理才通知。异常读取当前冻结协议和日志，不杀旧作业、不重放未知工作；真正实现bug须独立版本/输出并扣已用预算，不能覆盖冻结源码冒用原身份。

历史已真实收尾，禁止重复resume/物理审计：P0 preserve323；static_fullchain_001部署失败；static_fullchain_002（19a7c744）；dynamic_reference_001（95735ab0）；dynamic_margin_001（a10cb8dd）；dynamic_handoff_001（d88b5f39）；temporal_data_001（66c54fb4）。closeout receipts在ff_prior_fullchain/evidence。P1三障碍通过、五障碍0合格，60提案/23完整认证/1预算未决/36初筛/12合格/6实际执行重放；P1仍留一次有证据的五障碍机制修复，限原剩余64566耦合/14021.469秒/297980076积分及输出余额，禁止加载已知07答案或无界扫参。原static002完成resume仅一次2476文件不变。P2a固定rho=.9695632054405793修复新1+旧3复用，组合4/4、历史真实4/5；reference/margin完成resume仅一次64/46文件不变。P2b8独特执行6通过，12逻辑10通过（4共享基线），基线2/2，终点先验4/6；漂移两失败最坏6.753/7.626mm，完整836重放0。C2不支持，局部工程PASS；完成resume仅一次135文件不变，P2累计351000≤432000步。都不能再为补文本或改结论重跑。

P3a temporal_data_001已独立审查partial/high、局部资格PASS，可推进时序学习。固定原32TRAIN/8VAL来源前8TRAIN/前4VAL，12合格=4准确历史复用+8新执行，剩余28保持LEGACY_NOT_REQUALIFIED。新最大位置2.112068mm、姿态.021865度，完整836重放逐元素一致；170源码/214输入，1913检查通过。377.453秒/216000步/1152681165字节，完成resume仅一次146文件hash不变。无新任务/主库写入、无训练、无DEV。报告ff_prior_temporal_data/docs/ANALYSIS.md；qualified_view是现有来源资格索引，不升级全40。

当前P3b具体：只用上述8TRAIN/4VAL，DEV/07不入TRAIN，旧14222静态标签不混入。新的公共条件1014维（836完整初态、q0/v0、七世界航点/时间、抓捕局部位姿/初始位姿、卫星运动、最多5球障碍mask/几何等）+271×36每系数时间两端公开期望位姿/速度。未来条件仅由公开初始目标状态恒速/局部角速度外推，不读取未来实际状态；不声称等于所有真实自由转动。完整273×17 C2/277结点/.1秒/27秒，前2系数固定q0/dq0，271×17=4607自由输出；静态323原流程另保留。codec仅8TRAIN均值/尺度，teacher均值贡献披露。

两方法同175361参数时序卷积SequenceNetwork：48通道、4残差核5/膨胀1,2,4,8/GroupNorm/FiLM，完整局部时间条件；确定性输入噪声/t为0。相同初始化seed20261022，旧模型不改不转移。Diffusion v预测、原linear beta 1e-4→.02/1000，16DDIM，float64状态/float32网络，正交clean/noise，逐步分解残差<=1e-10且有限，末alpha1；无物理引导进入采样。prepare合格参考逐元素不变，其他参考向常值q0作一次凸包约束缩放，当前只支持零planner初速（卫星可非零），不改实际状态、不声称世界/碰撞资格。40软件检查通过，零原生步/零优化器，仅软件非正式学习证据。369输入绑定与根源码已冻结。

先两方法各32次临时TRAIN完整采样更新，8固定probe噪声20264200…20264207；数值/梯度全部有限、固定probe损失严格下降、权重完全回退才正式训练。不是旧1%门，也不能改门追认旧负结果。失败则M2/VAL/DEV均BLOCKED，仍做负结果收尾，转独立P1修复；不调lr/权重重复诊断。正式两方法各3000实际AdamW更新、batch8/lr2e-4/weight_decay.01/clip1/seed20261022，全8TRAIN每步消费一次。Diffusion普通v，每4步两条固定轮转TRAIN完整16DDIM输出监督并入同一次更新，共750次/1500完整样本；确定性每步完整输出监督。损失是归一化合格系数重建+.1相邻差分细节，绝不能称完整当前非线性世界终点梯度。原合格保护是标签重建和prepare不变性，不是逐更新真实物理资格。VAL无梯度、不选checkpoint，固定最终3000步；核实际optimizer.step=3000、权重变化/精确重载、全部消费和50步摘要。最多一个候选，不因效果差重训。

训练通过才VAL：4原VAL×两方法×1候选（Diffusion seed20266000…20266003），最多8新执行。完整采样保存raw/prepare，prepare运动门后用DirectEnvironment原67力矩逐.002秒执行/独立重放，各13500步；无世界反馈/避障修正，仍有必要关节参考伺服。全程5mm/1度，不仅末端终点。失败保持分母，未执行/raw不赋MODEL_VALID。VAL结果只评价，不回流改训练。

再原DEV两个任务精确初态/七航点/目标运动（task seed20261020/20261021），静止采样20261020/20262020/20263020，漂移20261021/20262021/20263021，三组两方法每组一提案，12逻辑/最多8新独特执行（2确定性+6Diffusion），另4基线逻辑精确共享包括失败，不作独立重复。两模型同世界反馈：preferred=qdot_sequence+.5(q_sequence−actual_q)，加权J下H=JwᵀJw+1e-4I最小世界修正，再原velocity_filter/advance_native。保存sequence/world correction/constraint correction贡献；共同控制成功不能归给裸网络。旧P2反馈基线仅历史背景，不能冒充本轮同控制纯归因。原DEV与小型移动TRAIN存在域差异，须披露。不新增候选、不回退旧答案、不失败重标。

硬预算单线程<=2CPU/18000秒/16GiB/432000新原生重放步=8VAL+8DEV共16×27000，正式训练阶段最多10800秒。与P3a实际合计低于原P3总6h/20GiB，余量不授权追加实验。reserve计数保守，真实步以trace。保持所有旧模型/数据/结果/用户修改/旧作业，单重worker。当前不扩库、不1024/30000、不pilot/study/strict_multimodal、TEST、硬件或接触捕获。仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何和所有原运动/力矩/动量/重放门不变，不裁剪/重置实际状态、不mock、不抽查升级。

终态一次收尾：读冻结源/绑定、公共条件来源、8/4数据消费、initial/diagnostic_after及回退、3000更新和optimizer/权重重载、16DDIM/raw/prepare、全部VAL与DEV native/replay/geometry/transitions/filter/feedback、共享关系和失败/缺失分母。实际未运行不写通过。按任务/种子位置姿态均值/P95/最大及全程最坏作配对统计；模型无引导/prepare/反馈/实际执行等级分开，查询时间只记录不要求胜出。工程局部门每场景×方法至少一合格+原始审计仍需保留所有失败；科学优势另外判。不把学习/数据局部PASS当完整工程完成。

依experiment-result-to-claim委派全新独立review，只保存数组/神经/代数和源码，不追加物理或optimizer追认、不重审旧实验。完成resume仅一次、UTF8父子、保存核心文件清单/hash不变。报告脚本放tools/evidence，冻结root源码不改。写中文ff_prior_temporal_learning/docs/ANALYSIS.md、紧凑证据/新review-traces、统一目标/版本paper tracker、正确schema closeout receipt及本地任务提交，不推送大数据。

收尾后继续原P1五障碍一次机制修复和P4固定开发整链验收；先读原负结果提出有证据机制，准确任务/种子/源码/预算冻结再运行，禁止加载07答案。P4须真实连续避障接近→移动追踪，不能拼接独立视频。每阶段同瓶颈最多一次有证据修复，不无限lr/权重/rank试错。局部失败转不依赖工作；确无可推进工作按goal规则报告真实阻塞。长期goal/唯一接续保持active，不能因单次收尾自动暂停。只有公开任务到生成路径、避障接近、实际执行、连续交接、动态追踪及学习评价均有真实证据和复现交付才完成工程目标，科学优势另判；TEST另列授权与就绪。终端.\ff_prior_fullchain\job.ps1 status|next|report只读。每阶段交付中文报告链接、实测正负结果、提交、命令和下一步。
