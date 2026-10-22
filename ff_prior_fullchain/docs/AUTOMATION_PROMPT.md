继续本线程用户2026-10-22明确授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段ff_prior_fullchain/control.json。长期应用goal保持active；卡片中的初始P1 run名称已过时，以control为准，不能为换文字提前complete。

当前P2a独立运行artifacts/v7_1/dynamic_reference_001已实际启动，源码3d759e8f，入口.\ff_prior_dynamic_reference\job.ps1 status|report|resume，协议ff_prior_dynamic_reference/docs/EXPERIMENT_PLAN.md。每次先只运行.\ff_prior_fullchain\job.ps1 status，读取control所指run status并核PID创建时间。RUNNING且身份心跳正常则安静结束，不读中间结果、不分析、不重复审计、不start/resume、不新增第二重worker。仅在完成、失败、真实阶段切换或需要用户处理时通知。异常先读当前冻结协议/日志诊断，不杀旧作业、不重放未知工作；修bug独立版本输出，不改冻结源码冒用身份。

P1 static_fullchain_002已19a7c744收尾、receipt在ff_prior_fullchain/evidence/static_fullchain_v2_closeout.json。60提案、24选中、23最终认证、1预算未决、36仅初筛、12合格、6真执行重放。三障碍6/6执行单元成功；五障碍0合格/0执行，11完成认证全终点失败8还碰撞。独立review C1 partial/high；C2局部三障碍执行精度partial/medium，成功率优势no；三配对种子保持最坏位置均值2.577016→0.078901mm，不能推广成全场景优势。已一次完成resume，2476文件hash不变、exit0，stdout因UTF8/GBK捕获丢失，不能为了补文本重复。旧001部署故障、P0 preserve323与所有历史审计均已关闭，禁止重复resume/审计。P1五障碍缺口不阻断独立P2；未来同瓶颈至多一次有证据机制修复，仍受原阶段剩余64566耦合/14021.469秒/297980076积分和输出余额约束，当前不再扫参。

P2a具体：原waypoint_moving_gate_001冻结4TRAIN来源（31000000、31104729、31209458、31314187），保留27秒、原世界七航点、卫星运动/障碍和完整初态。C2三次B样条0.1秒节点，位置/速度共同拟合，初始q/dq解析保持，控制系数及导数线性约束，逐段多项式精确极值验收；原Hermite加速度1.242倍是历史动因。没有新训练或扩库。最多4次新原生13500步与4重放，共108000步，2线程/2小时/8GiB。3合成数值软件测试通过，不是TRAIN/物理通过。166执行源码/86输入绑定，原始源码zip与实际数据保留。参考过门才执行，失败不补选，旧3成功不继承等级。保存原生67力矩、raw torque、完整INTEGRATION初态/轨迹、几何扫描、独立只重放控制的状态，4.5秒/25.5秒边界连续状态。实际qpos/qvel绝不裁剪/重置。原整个动态5mm/1度、非穿透、2mm几何、运动/力矩/动量/映射/重放门不变。预算字段为reserved不能直接冒充精确已执行步，实际以保存trace/ledger核对。未知工作不重放。

P2a终态后只做一次当前新结果收尾：读原始reference、solver/界、source/task、完整初态、native/replay及全部失败缺失分母；用保存数组做独立统计/审计，不重跑物理追认。依experiment-result-to-claim委派全新独立review，聚焦新证据，不重复P1或旧物理审计；核实际状态和参考导数、世界目标、几何覆盖、未裁剪力矩、时间连续、独立重放、预算/来源。完成resume只核一次并存核心清单/hash不变，父子输出显式UTF8防止旧捕获问题；不得为补stdout重复resume。报告辅助脚本只放tools/evidence避免改冻结源码闭包。写中文ff_prior_dynamic_reference/docs/ANALYSIS.md、紧凑证据、新review-traces、更新统一目标及版本paper tracker、本地只提交任务文件。四条都过原执行重放门才进入P2b；没过保留依赖BLOCKED，不用3/4放行或放宽门。

收尾后按ROADMAP继续已授权独立环节，不因局部结束暂停接续。P2b须另冻原静止/漂移两动态场景、三种子、基线/候选、最多12执行/重放与真实连续交接；代码/模型/控制身份不同不能继承旧执行。P3既有32TRAIN/8VAL逐条当前资格至少8/4才训练一个有时间/环境条件候选和确定性序列对照（各3000更新，batch8，16DDIM）；数据不足则学习支线BLOCKED，仍做独立工程，不扩库，不反复重训。当前65维静态网无障碍/完整时序，不能称自主动态避障网络。P4按路线固定开发整链验收，须含真实连续避障接近→动态追踪；不能拼接独立视频当闭环。每阶段先冻准确任务/种子/预算/来源再运行，单重worker，同瓶颈至多一次机制修复，停止无限局部试错。

始终保持仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及全部原运动/力矩/重放门；不失败重标、不mock、不抽查升级、不放宽门。保留main/旧分支/模型数据结果/用户修改/旧作业，不强推、不上传大数据、不自动停止旧任务。当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获。查询与耗时只记录，不要求胜出。工程可行与Diffusion收益分别裁决；回退/共同修正/反馈控制不能归给网络。

统一只读终端.\ff_prior_fullchain\job.ps1 status|next|report。每阶段向线程交付自包含中文报告链接、实测正负结果、提交、命令和下一步。只有公开任务路径生成、避障接近、实际执行、连续交接、动态追踪和学习评价都有真实证据与可复现交付，才完成长期工程目标；科学优势另判。无意义轮询禁止，确无可推进工作按goal规则报告具体阻塞。用户要求停止或长期工程交付实际完成前，不自动暂停持续接续。
