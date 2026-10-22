继续本线程用户2026-10-22明确授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段ff_prior_fullchain/control.json。长期应用goal保持active；卡片中的初始P1 run名称已过时，以control为准，不能为换文字提前complete。

当前P2b独立运行artifacts/v7_1/dynamic_handoff_001已实际启动，源码549c4a2e，入口.\ff_prior_dynamic_handoff\job.ps1 status|report|resume，协议ff_prior_dynamic_handoff/docs/EXPERIMENT_PLAN.md。每次先只运行.\ff_prior_fullchain\job.ps1 status，读取control所指run status并核PID创建时间。RUNNING且身份心跳正常则安静结束，不读中间结果、不分析、不重复审计、不start/resume、不新增第二重worker。仅在完成、失败、真实阶段切换或需要用户处理时通知。异常先读当前冻结协议/日志诊断，不杀旧作业、不重放未知工作；修bug独立版本输出，不改冻结源码冒用身份。

P1 static_fullchain_002已19a7c744收尾、receipt在ff_prior_fullchain/evidence/static_fullchain_v2_closeout.json。60提案、24选中、23最终认证、1预算未决、36仅初筛、12合格、6真执行重放。三障碍6/6执行单元成功；五障碍0合格/0执行，11完成认证全终点失败8还碰撞。独立review C1 partial/high；C2局部三障碍执行精度partial/medium，成功率优势no；三配对种子保持最坏位置均值2.577016→0.078901mm，不能推广成全场景优势。已一次完成resume，2476文件hash不变、exit0，stdout因UTF8/GBK捕获丢失，不能为了补文本重复。旧001部署故障、P0 preserve323与所有历史审计均已关闭，禁止重复resume/审计。P1五障碍缺口不阻断独立P2；未来同瓶颈至多一次有证据机制修复，仍受原阶段剩余64566耦合/14021.469秒/297980076积分和输出余额约束，当前不再扫参。

P2a两轮已真实收尾，禁止再resume/审计/调rho：dynamic_reference_001提交95735ab0，4参考合格、3/4真实执行、108000步、一次完成resume64文件不变；dynamic_margin_001提交a10cb8dd，固定内部rho=.9695632054405793，新1执行通过+3精确历史复用，组合4/4、实际加速度最大.970613倍原限，新27000步，一次完成resume46文件不变。旧失败保持，累计真实4/5，不能称4次新成功。新独立审查局部P2a PASS、整体C1 partial/high、C2未评估、论文provisional。两个closeout receipt在ff_prior_fullchain/evidence。原四TRAIN和27秒/世界目标/物理门未改，不扩库。

P2b具体：保留waypoint_domain_002两个原冻结任务/完整初态，静止task seed20261020，漂移20261021，原七航点与卫星运动、27秒4.5接近/21追踪/1.5保持。三种子是采样噪声而非新任务：静止20261020/20262020/20263020，漂移20261021/20262021/20263021。task_controller与diffusion_null_prior各6逻辑单位，共12、每单位1提案。确定性基线无噪声输入，每任务仅新执行一条，其余两个逻辑单位精确共享本run先前已完成基线（保留其失败亦共享），不能称六次独立基线重复；总计划2基线+6先验=8独特执行/重放216000步，4共享单位。历史原基线没有中间完整INTEGRATION/全部几何观测，这次补一次全状态观测有具体证据缺口，实际velocity/advance_native完全继承原控制器，仅增只读记录。新旧任务identity和836维初态、初帧全部逐位对齐，4原域约束软件测试通过，不等于新物理通过。

精确加载旧endpoint_jacobian_bounded_001/learning权重codec，6条先验各完整16DDIM=96新网络调用，无训练/选checkpoint/结果后补提案。原终点条件网络未包含完整时序/障碍，通过原Jacobian零空间0.05rad/s适配，反馈控制器负责动态任务；不得称完整323时序网络直接生成或网络自主动态避障。raw Z未动态认证不能标MODEL_VALID。记录原提案、公共控制滤波、实际执行/独立重放与失败缺失，贡献分别报告。

168源码/86输入绑定；运行前精确冻结，源码zip含工作区原用户修改。每新单位13500原生67力矩步+同初态控制独立重放，记录完整INTEGRATION、原始力矩、全机几何中间态、actual/desired和4.5/25.5连续状态，不裁剪/重置实际状态。工程局部门：两场景×两方法各至少一条合格连续执行，完整身份/原始审计通过，仍报告各方法6逻辑分母及独特执行数；共享基线不能作为独立样本统计。不因Diffusion未胜过基线停止工程推进。P1五障碍与P4避障→移动追踪组合仍缺。

硬预算1数值线程（<=2CPU）、4h、12GiB、297000新步，计划216000；P2原总432000，已用108000+27000，计划总351000。余量不授权新增单元。预算native字段reserved可能保守，实际步骤以trace核。异常保留unknown/失败，不自动重放。旧模型/数据/结果/用户修改保持；不扩库、不pilot/study/strict_multimodal/TEST/硬件/接触捕获。

当前新run终态一次收尾：读取原始run/source/selection/initial/model_loaded/prior(全16DDIM)/control_filter/native/replay/geometry/transitions/paired_statistics及全部失败。核任务目标未变、完整状态连续、仅观测wrapper没改变控制、共享身份/控制hash精确，统计按任务/采样种子配对位置姿态均值/P95/最大与最坏，独特执行/共享清楚区分，不用帧当独立样本。软件/原始提案资格/实际执行/科学收益分开。依experiment-result-to-claim委派全新独立review，仅新证据及必要身份，不重审旧物理、不追加仿真追认。完成resume仅一次，父子UTF8，存核心清单/hash不变。辅助放tools/evidence，不改冻结root源码。写中文ff_prior_dynamic_handoff/docs/ANALYSIS.md、新review-traces、紧凑证据、统一目标/版本tracker、closeout receipt及本地任务提交；完成即自动推进P3，不因局部收尾暂停接续。

收尾后按ROADMAP继续P3既有32TRAIN/8VAL逐条当前资格，至少8/4才训练一个有时间/环境条件候选和确定性序列对照（各3000更新、batch8、16DDIM）；先固定动态表示/完整初态时序/障碍条件及连续性合同，不把65维终点先验改名当时序网络。数据不足则学习支线BLOCKED，继续独立工程，不扩库不反复重训。P4固定开发整链验收，含真实连续避障接近→移动追踪，不能拼接独立视频。每阶段先冻结准确任务/种子/预算/来源再运行，单重worker，同瓶颈至多一次有证据机制修复，停止无限局部试错。P1五障碍还有原预算内一次机制修复机会，读其真实负结果再设计，不加载07答案。

始终保持仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及全部原运动/力矩/重放门；不失败重标、不mock、不抽查升级、不放宽门。保留main/旧分支/模型数据结果/用户修改/旧作业，不强推、不上传大数据、不自动停止旧任务。当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获。查询与耗时只记录，不要求胜出。工程可行与Diffusion收益分别裁决；回退/共同修正/反馈控制不能归给网络。

统一只读终端.\ff_prior_fullchain\job.ps1 status|next|report。每阶段向线程交付自包含中文报告链接、实测正负结果、提交、命令和下一步。只有公开任务路径生成、避障接近、实际执行、连续交接、动态追踪和学习评价都有真实证据与可复现交付，才完成长期工程目标；科学优势另判。无意义轮询禁止，确无可推进工作按goal规则报告具体阻塞。用户要求停止或长期工程交付实际完成前，不自动暂停持续接续。
