继续本线程用户2026-10-22明确授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段ff_prior_fullchain/control.json。长期应用goal保持active；卡片中的初始P1 run名称已过时，以control为准，不能为换文字提前complete。

当前P2a唯一响应余量修复artifacts/v7_1/dynamic_margin_001已实际启动，源码d5ba2106，入口.\ff_prior_dynamic_margin\job.ps1 status|report|resume，协议ff_prior_dynamic_margin/docs/EXPERIMENT_PLAN.md。每次先只运行.\ff_prior_fullchain\job.ps1 status，读取control所指run status并核PID创建时间。RUNNING且身份心跳正常则安静结束，不读中间结果、不分析、不重复审计、不start/resume、不新增第二重worker。仅在完成、失败、真实阶段切换或需要用户处理时通知。异常先读当前冻结协议/日志诊断，不杀旧作业、不重放未知工作；修bug独立版本输出，不改冻结源码冒用身份。

P1 static_fullchain_002已19a7c744收尾、receipt在ff_prior_fullchain/evidence/static_fullchain_v2_closeout.json。60提案、24选中、23最终认证、1预算未决、36仅初筛、12合格、6真执行重放。三障碍6/6执行单元成功；五障碍0合格/0执行，11完成认证全终点失败8还碰撞。独立review C1 partial/high；C2局部三障碍执行精度partial/medium，成功率优势no；三配对种子保持最坏位置均值2.577016→0.078901mm，不能推广成全场景优势。已一次完成resume，2476文件hash不变、exit0，stdout因UTF8/GBK捕获丢失，不能为了补文本重复。旧001部署故障、P0 preserve323与所有历史审计均已关闭，禁止重复resume/审计。P1五障碍缺口不阻断独立P2；未来同瓶颈至多一次有证据机制修复，仍受原阶段剩余64566耦合/14021.469秒/297980076积分和输出余额约束，当前不再扫参。

P2a原dynamic_reference_001已95735ab0收尾，receipt为ff_prior_fullchain/evidence/dynamic_reference_closeout.json。4/4 C2参考合格，四次原生13500步与重放13500步完整，共108000步；只有3/4实际执行合格。t00实际加速度约1.00115倍原限，其他原门通过；完整重放状态差均0。新review partial/high，C2未评估，论文provisional。完成resume已仅一次，64文件/hash全不变、UTF8完整输出。禁止再次resume/审计/重跑旧001，禁止把它追认为4/4。

当前修复仍原4TRAIN来源31000000/31104729/31209458/31314187，原27秒/七世界航点/目标卫星运动/障碍/完整初态/控制器和全部物理门保持。由保存TRAIN实际加速度减参考的最大归一化残差0.015218397279710373，唯一固定内部参考加速度比例rho=1-2残差=0.9695632054405793。只收紧样条参考加速度凸包，不扫描rho/节点/种子；3软件测试是合成数值，不是实际物理成功。168源码/151输入绑定，包含旧run完整64文件。源码zip包含原有用户hierarchical_qp修改，不能称纯clean HEAD。

仅t00是新执行/重放单元，最多13500+13500=27000步；t01–03只重建参考并核新旧节点/系数逐元素相同、旧166源码hash全同、完整初态数组与identity（机器人/场景/task/source/waypoints）相同、原native控制证据hash/原通过门相同后历史复用。身份不符REUSE_IDENTITY_BLOCKED，不自动补执行；新旧成功和actual_new_steps分开。最多1新执行、3历史复用，零训练/扩库/DEV/TEST。2线程/600秒/1GiB，计P2a原2h/8GiB余额。27000从未运行P2b预算转移，P2总432000步不增；P2b最多297000新步=11新单元，但原2任务×3种子×2方法共12逻辑单元不变，至少1准确历史复用。不能改变原任务/种子/分割凑复用，缺失则BLOCKED。

终态后仅一次新结果收尾：实际拟合/导数极值、t00真实native/replay、完整初态/INTEGRATION、全机几何、原动态目标、原始力矩、4.5/25.5连续边界和各门；三复用逐条精确身份/控制证据，失败和累计旧4+新至多1的尝试分母保留。依experiment-result-to-claim委派全新独立review，审当前新证据与必要复用身份，不重复旧物理审计、不追加物理追认。完成新run resume仅一次，父子UTF8并存核心清单/hash不变，不为补输出重复。报告辅助放tools/evidence，冻结root源码不改。写中文ff_prior_dynamic_margin/docs/ANALYSIS.md、新review-traces、紧凑证据，更新目标/tracker/控制/本地任务提交。只有新t00合格且3精确复用成立，组合4任务证据才满足P2b门；旧001仍3/4。若失败同机制不再追加，P2b BLOCKED，继续独立P3资格/时序接口工程，不放行依赖训练。

收尾后按ROADMAP继续已授权独立环节，不因局部结束暂停接续。P2b须另冻原静止/漂移两动态场景、三种子、基线/候选、12逻辑执行/重放单元与真实连续交接（新步上限297000，至少1严格历史复用）；代码/模型/控制身份不同不能继承旧执行。P3既有32TRAIN/8VAL逐条当前资格至少8/4才训练一个有时间/环境条件候选和确定性序列对照（各3000更新，batch8，16DDIM）；数据不足则学习支线BLOCKED，仍做独立工程，不扩库，不反复重训。当前65维静态网无障碍/完整时序，不能称自主动态避障网络。P4按路线固定开发整链验收，须含真实连续避障接近→动态追踪；不能拼接独立视频当闭环。每阶段先冻准确任务/种子/预算/来源再运行，单重worker，同瓶颈至多一次机制修复，停止无限局部试错。

始终保持仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及全部原运动/力矩/重放门；不失败重标、不mock、不抽查升级、不放宽门。保留main/旧分支/模型数据结果/用户修改/旧作业，不强推、不上传大数据、不自动停止旧任务。当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获。查询与耗时只记录，不要求胜出。工程可行与Diffusion收益分别裁决；回退/共同修正/反馈控制不能归给网络。

统一只读终端.\ff_prior_fullchain\job.ps1 status|next|report。每阶段向线程交付自包含中文报告链接、实测正负结果、提交、命令和下一步。只有公开任务路径生成、避障接近、实际执行、连续交接、动态追踪和学习评价都有真实证据与可复现交付，才完成长期工程目标；科学优势另判。无意义轮询禁止，确无可推进工作按goal规则报告具体阻塞。用户要求停止或长期工程交付实际完成前，不自动暂停持续接续。
