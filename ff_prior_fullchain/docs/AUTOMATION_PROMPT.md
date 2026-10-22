继续本线程用户2026-10-22授权的持续工程全链路。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段control.json；goal保持active，旧卡片P1文字不覆盖当前control。

先运行.\ff_prior_fullchain\job.ps1 status，核当前PID创建时间和心跳。健康RUNNING就安静结束，不读中间结果、不重复审计/start/resume、不另开worker。异常仅诊断当前冻结协议和日志，不杀旧任务、不重放未知工作。终态先收尾再串行推进；通知只在完成/失败/真实阶段切换或需要处理时。

static_goal_seed_001已9b894469收尾，收据ff_prior_fullchain/evidence/static_goal_seed_closeout.json。30/30种子位姿达门、6完整验证均穿透16.7–35.4mm、24仅初筛、0执行。原末尾JSON审计错误保留FAILED；独立辅助只读审计141/141 PASS。新独立review整体/C1/C2 no/high，初始化子结论yes，5326/5327谓词为真，唯一额外Git树一致性为否，实际176源码归档/86输入和数值全一致。用户hierarchical_qp旧修改已归档未覆盖，复现要executed_source.zip+manifest而非只git。失败resume拒绝已仅一次核1254文件不变，禁止再执行或再审历史。唯一同瓶颈修复机会已用，不追加局部步长/rank或新训练。

下一步P4实施/冻结，按ff_prior_fullchain/docs/P4_HANDOFF.md及control真实状态。包ff_prior_fullchain_acceptance，拟独立fullchain_acceptance_001，准确代码/任务/种子/预算/复用矩阵冻结后才后台启动。保留五场景×三种子×两系统配置30逻辑机会：原静态三/五障碍、原静止/漂移七航点、五障碍接近到漂移组合。三障碍使用static_fullchain_002原预选6公开生成参考，最终新执行接口加完整INTEGRATION门后6新执行/重放，不冒充6次新规划。动态精确复用temporal_learning_001的12逻辑/8独特记录；源码/模型/任务/初态/参考/控制身份一致并且保存完整状态达门才复用，4确定性共享不算独立实验。不重训/重审/旧resume。五障碍6和连续组合6保留BLOCKED及分母，缺合格路径不执行、不用三障碍组合替换、不伪造组合已冻结。实现同live model/data无状态重置交接接口并软件验证；未有实际组合轨迹不称物理闭合。P4上限2线程/6h/20GiB/最多6新执行重放165000步，不增加搜索或训练。

每新run终态只收尾一次：失败/缺失分母，来源/复用/实际新执行分开，MODEL_VALID和EXECUTION_VERIFIED逐条。按experiment-result-to-claim委派新独立review，只读新保存证据，历史只核复用身份不重新物理审计；完成resume仅一次并核核心清单/hash不变。辅助脚本在tools/evidence，冻root源不能改。中文ANALYSIS/曲线CSV/紧凑证据/review-traces/版本paper tracker/统一目标/正确schema收据/只本任务本地提交，不上传大数据。旧P0/P1/P2/P3收尾均不重复。

工程目标需要原三/五障碍和动态跟踪及五障碍连续组合都真实通过、完整训练采样引导评价与可复现交付；优势不是工程门，但回退与共同修正/反馈贡献不能归网络。P4有缺口则不complete，继续尚有的独立工作；确无可推进工作按goal规则三轮同阻塞后blocked，不无限局部试错。没有用户停止或整体真实完成不自动暂停接续。

保持仅不穿透、世界5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何、运动/原生力矩/动量/重放门；不状态裁剪/重置，不失败重标、mock、抽查升级。保留全部旧模型数据结果、用户修改、main旧分支和旧作业。DEV及07不入TRAIN。当前不扩库、不1024/30000、不pilot/study/strict_multimodal、TEST、硬件/接触捕获。终端.\ff_prior_fullchain\job.ps1 status|next|report只读。
