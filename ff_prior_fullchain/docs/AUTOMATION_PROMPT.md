继续本线程用户2026-10-22明确授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段ff_prior_fullchain/control.json。新应用长期目标卡已active，旧preserve323有界目标已真实收尾complete，不得把旧complete当作长期工程/科学目标完成。

目标：从公开初态、障碍、世界目标和时间条件产生路径，经自由漂浮全机避障修正、原生67力矩执行与独立重放，完成刚性臂非接触预抓捕接近，再接移动卫星与连续体七不规则航点追踪；完整训练/采样/引导/对照及报告。工程可行与Diffusion收益分别裁决，优势不再是工程推进门；回退规划器、共同修正器和反馈控制器的贡献不能归给网络。当前65维网络输入没有障碍/完整时序，不能称自主动态避障网络。

每次先运行 .\ff_prior_fullchain\job.ps1 status，读取control所指run status并核PID创建时间。RUNNING且身份和心跳正常就安静结束，不读中间结果、不分析、不重审、不重复start/resume，不新增第二重worker。状态不变不通知，只在完成、失败、真正阶段切换或需要用户处理时通知。状态异常先读该run冻结协议和日志诊断，不杀旧作业、不重放未知工作。新bug修复独立版本输出，不能覆盖冻结身份。

交接基点P0 preserve323_bounded_001已3e659ce4收尾：32梯度通过、64临时更新，三个参数比例均未达原1%学习门，fit和check各8/32不增，权重完全回退；M2正式训练、VAL、DEV、新执行全部未运行。独立审查partial/high，原审计PASS；完成resume已仅一次核2022文件/hash不变。ff_prior_fullchain/evidence/preserve323_closeout.json是收尾回执。禁止再次执行该run的resume/审计/训练，也不重新审查历史对照。旧模型继续保留。本轮科学负结果不改门追认。

当前P1实际运行是独立修复static_fullchain_002，源码8d3d562a，包ff_prior_static_fullchain_v2，准确协议该包docs/EXPERIMENT_PLAN.md，入口 .\ff_prior_static_fullchain_v2\job.ps1 status|report|resume。12项软件检查通过，最长原子输出路径221字符真实写读通过，不代表物理通过。旧static_fullchain_001在34.844秒首次修正写原子文件因Windows长路径失败；源码25b98dd6/58文件均保留，ledger有inflight且精确调用未知，禁止resume或追认。已保守扣旧单元13000耦合/32000000积分/40秒/128MiB预留，这些不是测量值。新上限2线程/21560秒/(20GiB-128MiB)/147000耦合/352000000积分；每单元12000/29000000/1650秒、2条修正各5800/14000000/600秒，执行留360秒。新旧合计不突破原总预算。修复只缩短单元ID为u00至u11，完整任务/种子/算法/原物理门保持。不要再次start。

原三/五障碍×3种子×两方法各5提案，共60。公共初筛前2各最多16轮完整323修正，每单元执行前选至多1条，共12执行重放。两方法同新预算；未选中、未完成、超预算保留分母，不能冒充已验失败或合格。原始网络输出256区间和prepare后512初筛仅诊断/排序，不授MODEL_VALID。网络为现有aligned低102加零221细节的323真实16步采样，0新训练；65条件无障碍。通用几何弓形、共同修正与实际控制贡献单列，不加载旧00/07解，不做旧late4/113查询纯归因。

结束后读取该run来源、全部60提案分层记录、最多24条完整修正/认证、12执行单元选择、完整初态/原生与重放trace、跨模型budget及ledger。自动原始审计只核部分身份/位姿/分母/控制/调用；新的独立review须补查真实solver差分、可行恢复、搜索认证与物理门，不能只信summary或PASS。旧001部署失败独立review目录paper/review-traces/experiment-result-to-claim/20261022_static_fs_failure_run01，不可复用它作为002性能审查。只验证一次002完成resume并核清单/hash；禁止旧001未知工作resume。写ff_prior_static_fullchain_v2/docs/ANALYSIS.md、fullchain/evidence/static_fullchain_v2_closeout.json和版本化tracker并本地提交。两个场景各至少1条完整合格执行支持静态工程可行，不等于鲁棒性或网络优势。失败也收尾，继续不依赖的P2；同瓶颈最多一次有具体机制证据的规划修复，禁止无限微调。

P1后按ROADMAP持续执行P2动态受约束表示与连续状态交接、P3有时序/环境条件的一个学习候选和确定性序列对照、P4固定开发套件整链验收。各阶段先冻结具体代码/任务/种子/预算再实际运行，并更新control.json指向真实当前run，不显示已结束run为正在运行。修复既有4TRAIN的Hermite加速度问题时保持原27秒和参考目标，不改时间或放宽门。已有合法控制器可推进工程，不等待网络相对胜出。P3只能用既有已认证TRAIN/VAL，DEV及07不入训练；数据不足则标该分支BLOCKED并继续独立工程工作，不擅自生成大库。一个训练候选，不因效果差重复训练。每阶段最多一次有证据的同瓶颈机制修复，禁止无限局部lr/权重/rank试错。

各run终态完成一次原始结果收尾：记录全部候选/执行失败和缺失分母，来源/数据消费/模型重载、无引导/prepare/引导/回退/执行的逐层统计。完成的新实验按experiment-result-to-claim委派全新独立review，只读保存证据，不重复历史审计，不追加物理来追认。完成resume只验证一次且核心清单/hash不变；报告辅助脚本放tools/evidence，冻结源不改。保存中文ANALYSIS、紧凑证据、review-traces、版本paper tracker、本地提交；新/旧执行与MODEL_VALID/EXECUTION_VERIFIED分别报告。收尾后自动推进下一已授权阶段，不因单个局部实验结束就暂停接续。

始终保持仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及全部原运动/力矩/独立重放门。实际状态不可裁剪/重置，不失败重标，不mock，不抽查升级全库；通过模型不等于执行通过。原物理门不能为追求优势放宽。保留main/旧分支/旧模型数据结果/用户修改/旧作业，不强推、不上传大数据、不自动停旧任务。

当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获；既有数据逐条重新认证与开发场景不自动写入TRAIN。查询和耗时记录但不要求胜出。不惜代价表示持续机制改进，不是无限算力、放松物理或保证科学正结论。最终TEST就绪及授权单列，禁止读取TEST调参。

终端 .\ff_prior_fullchain\job.ps1 status|next|report 为只读统一入口。每阶段交付自包含报告链接、实测正负结果、提交、命令及下一阶段。只有公开任务到避障接近、实际执行、连续交接、动态追踪与学习评价均有真实证据及可复现交付才完成长期工程目标；科学优势另判。确无可推进工作时按goal工具规则报告具体阻塞和需要的输入，避免无意义轮询。不要因单个阶段完成自动暂停持续接续；用户要求停止或本目标工程交付实际完成才收尾管理接续。
