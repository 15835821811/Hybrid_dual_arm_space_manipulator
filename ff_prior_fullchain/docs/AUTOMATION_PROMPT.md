# 唯一接续：完整Diffusion工程交付与可视化

继续用户2026-10-22最新授权的持续工程交付，优先“Diffusion能工作就跑一个完整的、输出所有可视化，不要一直验证优势”。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md；新路线ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md；真实阶段control.json。新授权已解除此前等待范围，不得重复询问同一范围。旧应用goal卡blocked/旧P1文字不覆盖新授权，不能伪造complete换卡。

当前D1包ff_prior_diffusion_demo，源码8456203e1c0f957f02b6932bc54480897a418146，独立run artifacts/v7_1/diffusion_demo_001已后台启动。协议docs/EXPERIMENT_PLAN.md，176源码/87输入绑定，8软件检查＋渲染探针＋JS语法通过（不是物理通过）。统一 .\ff_prior_fullchain\job.ps1 status|next|report，单run .\ff_prior_diffusion_demo\job.ps1 status|report|resume。每次先只读统一status核PID创建时间/心跳，RUNNING健康则安静结束，不看中间结果、不分析、不重审、不start/resume第二worker。状态不变不通知。终态/异常才读本run协议、原始证据和日志收尾或诊断。

D1固定原三障碍/seed20264611/Diffusion一个单元：5个新16步提案，原公开初始化与完整323修正最多选2条，原资格后选1条真实原生67执行及完整836独立重放；27秒+0.5秒保持，共最多27500新步。没有基线/优势对照、新训练或数据。2线程/2h/8GiB/12000耦合/29000000积分。保留原完整初态/世界目标/障碍；三障碍不能冒充五障碍完成。静态验收通过后不追加本轮候选、补执行或重复旧run。

worker会从保存实际状态生成visuals/index.html、双视角实体MuJoCo MP4、时间可拖动旋转的离线HTML、裸提案/prepare/引导/实际路径、双臂误差/世界xyz、全机间隙、17维运动、67力矩、基座/动量/流形/重放图PNG/PDF和逐时刻CSV。所有渲染只做保存状态mj_forward，无新mj_step。视觉失败只能独立tools脚本补渲染保存证据，不改冻结源码或重跑物理。缺失或失败如实标注，不伪造完整媒体。完成后实际检查代表性图片/视频帧及HTML，生成中文ANALYSIS和链接索引。

终态只做一次必要原始账本/资格/执行/来源核对，按experiment-result-to-claim委派新独立review，重点工程完成与分级，不再开优势消融或反复数值检查。完成resume只一次且核心清单/hash不变，保存收据；冻结root源不改，辅助工具放tools/evidence。保存紧凑证据、review-traces、版本tracker、目标与本地提交，只提交本任务文件。旧P0—P5均已收尾，禁止重审/resume/物理，旧执行不冒充本轮新结果。

D1完成后按新持续授权自动推进D2实际终态连续动态衔接，不因阶段结束暂停或再申请同一范围：先从合格实际末态核对完整基座/qpos/qvel/时间及目标/模型兼容性，冻结准确桥接世界参考、原七航点与移动目标合同，再单worker后台。不得恢复默认初态或置零速度冒充连续，不得失败后把目标重标为终点；新组合任务与原冻结DEV明确区分，不能称旧DEV通过。若依赖BLOCKED继续能独立完成的可视化/实现，不假报全流程。D3原五障碍继续短时可复用世界目标扩展机制；ff_prior_world_steer目前仅未冻结草案，未启动，不得当已有实现/通过。D2/D3均需先代码/任务/种子/预算冻结再运行，原五障碍缺口不能用三障碍替代。

新周期上限D1 2h/8GiB/12000耦合/27500步；D2 4h/12GiB/8000耦合/120000步；D3 6h/20GiB/60000耦合/165000步；总12h/40GiB/80000耦合/312500原生含重放步。每阶段最多一次有证据机制修复且独立版本、扣阶段总额度；未知工作不重放，不因效果差无限扫参。连续两次同瓶颈无进展先只读技术决策，预算不足明确报告，不无限资源。

始终保留仅不穿透、双臂5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及原运动/力矩/动量/完整重放门。无状态裁剪/重置、失败重标、mock/抽查升级，DEV/07不入TRAIN、不偷用07解。保留所有旧模型数据结果、用户修改/main/旧分支/旧作业，不强推或传大数据。不扩库、1024/30000、pilot/study/strict_multimodal、最终TEST、硬件或接触捕获。Diffusion提案+修正+控制可以推进工程，不要求相对优势，贡献诚实拆分。只有路径/避障/执行/连续交接/动态追踪与完整媒体可复现交付均有真实证据才完成长期工程目标。唯一接续状态不变静默；阶段收尾自动推进，用户停止或工程全目标完成才管理接续。
