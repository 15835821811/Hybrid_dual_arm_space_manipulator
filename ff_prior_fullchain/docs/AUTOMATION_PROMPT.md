继续本线程用户2026-10-22授权的持续工程全链路目标。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效目标DIFFUSION_GOAL_CURRENT.md，路线ff_prior_fullchain/docs/ROADMAP.md，真实阶段ff_prior_fullchain/control.json。长期应用goal保持active，卡片初始P1运行名过时，以control为准，不能为了改文字提前complete。

当前P3a既有动态数据资格认证artifacts/v7_1/temporal_data_001已实际启动，源码292021c2，协议ff_prior_temporal_data/docs/EXPERIMENT_PLAN.md，入口.\ff_prior_temporal_data\job.ps1 status|report|resume。每次先只运行.\ff_prior_fullchain\job.ps1 status，核对应run状态、PID创建时间和心跳。RUNNING且健康则安静结束，不读取中间结果、不分析/重审、不start/resume、不另开重worker。只有完成、失败、真实阶段切换或需要处理时通知。异常先读该冻结协议/日志，不杀旧作业或重放未知工作；实现bug修复用独立版本/输出并扣已用预算，不覆盖冻结身份。

已收尾历史禁止再resume/物理审计：P0 preserve323、static_fullchain_001部署失败、static_fullchain_002收尾19a7c744、dynamic_reference_001收尾95735ab0、dynamic_margin_001收尾a10cb8dd、dynamic_handoff_001收尾d88b5f39。closeout receipts在ff_prior_fullchain/evidence。P1三障碍6/6执行，五障碍0合格；60提案/23完整认证/1预算未决/36初筛/12合格/6实际执行，局部partial。P1五障碍保留唯一同瓶颈机制修复，原剩余64566耦合/14021.469秒/297980076积分及输出余额；不加载已知07答案冒充自主规划，不再无界扫参。

P2a第一轮4参考/3执行通过、108000步；固定内部rho=.9695632054405793修复只新增t00一次通过27000步，另3精确复用，组合4/4，历史真实4/5失败保留。27秒/任务/控制器/原物理门不变。完成resume分别仅一次64/46文件不变。P2b dynamic_handoff_001已完成：两个原开发任务×三采样噪声×两方法12逻辑，2独特基线+6终点先验=8真实执行，4基线逻辑共享；6/8独特、10/12逻辑通过。基线2/2，先验4/6，漂移u09/u11仅全程位置超门，最坏连续体6.753/7.626mm，即使终点到位仍失败。8份836维完整重放误差0，216000步；P2累计351000≤432000。6×16真实DDIM/96网络调用，零训练；65维终点条件+0.05rad/s零空间适配，不是完整时序障碍网络。均值/P95和漂移尾部未改善，C2不支持。新独立review 20261022_dynamic_handoff_run01局部工程PASS、整体partial/high；完成resume仅一次135文件hash不变，禁止重复。失败与约束滤波活跃重叠仅为关联，不宣称唯一因果。

P3a准确选择：只读原waypoint_moving_gate_001/freeze.json中的32TRAIN/8VAL来源，预选前8TRAIN/前4VAL，失败不替换。TRAIN种子31000000/31104729/31209458/31314187/31418916/31523645/31628374/31733103；VAL32570935/32675664/99378320/99483049。前4TRAIN完全复用已保存273×17 C2样条和真实执行（t00来自margin，t01-03来自reference），核原q/dq/time、完整模型/任务/836初态/限制/物理代码/参考/控制hash；不一致BLOCKED，不补执行。后4TRAIN+4VAL仅8新执行上限，每条原27秒13500原生67力矩+13500独立重放。29软件检查通过只是软件；214输入绑定，源码递归闭包和zip已冻结。

新8参考复用原C2 .1秒/273×17输出/277结点，第一2系数保留q0/dq0，内部rho固定.9695632054405793；精确多项式q/dq/ddq极值和初态门，不能以表示通过冒充MODEL_VALID。继承已验证DirectEnvironment原控制/全机观测/原门，不改目标、卫星运动、障碍或时间。移动初态非零速度，不调用只支持零速度的静态耦合预测器冒充当前动态资格。只实际全门通过逐条EXECUTION_VERIFIED；其余28条保持LEGACY_NOT_REQUALIFIED，未拟合/未执行，不抽查升级。qualified_view只是既有来源的本次资格索引，不写主库、不添任务。开发样本/07不可入TRAIN。

P3a门：固定12条中>=8TRAIN及>=4VAL当前EXECUTION_VERIFIED，源/账本一致并新独立review支持，才进入P3b。不足则学习分支BLOCKED，仍完成负结果收尾并继续独立P1五障碍修复/P4接口，不换样本凑门。当前worker零训练/DEV/TEST。硬上限单线程<=2CPU、3000秒、5GiB、216000新原生/重放步；预算来自P3总6h/20GiB，后续须扣实际已用，不另领6h。每次只有一个重worker，未决/预算不足不通过，不自动重跑。

P3a终态读取实际run/source/bindings/selection、12单元reference/representation/identity/initial/result、8新native/replay/geometry/transitions及4reuse收据、qualified_view和budget。按TRAIN/VAL与新/复用分开报告全部分母、位置姿态均值/P95/最坏、实际运动/力矩/非穿透与完整状态重放；旧4只核必要身份，不重审旧物理。依experiment-result-to-claim委派全新独立review，仅保存数据/神经或代数/源码核对，不追加仿真追认。完成resume仅一次、父子UTF8，保存核心文件清单/hash不变，辅助脚本放tools/evidence不改冻结root源码。写中文ff_prior_temporal_data/docs/ANALYSIS.md、新review-traces、紧凑证据、版本paper tracker、统一目标、正确schema closeout receipt和本地任务提交。

合格后自主完成P3b具体设计/实现/预先冻结：一个含完整初态、七世界航点和时间、卫星运动与障碍条件的时序Diffusion及确定性序列对照，既定各3000更新/batch8/seed20261022/16DDIM。选当前认证的动态表示保持初态和时间连续，不把65维终点先验改名。先有界TRAIN数值/可学性，正式训练至多一个候选，VAL不进梯度/选模型，不因效果差重训。完整采样监督与原合格样本保护；具体损失、输出维度和执行适配在新run前冻结，不能声称尚未实现的世界监督。完整DEV原两任务/3种子×两方法×1候选最多12执行，工程门与收益分别判。数据不足保持学习BLOCKED而推进独立工程。

之后P4固定开发整链验收含真实连续避障接近→移动追踪，不拼接独立视频。P1五障碍缺口和P4组合仍必须解决/如实报告，不能因当前动态控制成功称全链完成。各阶段准确任务/种子/源码/预算先冻结再运行，单重worker，每阶段同瓶颈最多一次有证据机制修复，不无限lr/权重/rank试错。完成局部收尾自动推进下一已授权工作，不自动暂停接续。

始终保持仅不穿透、5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何及原运动/力矩/独立重放门。不裁剪/重置实际状态，不失败重标、不mock、不抽查升级。保留main/旧分支/模型数据结果/用户修改/旧作业，不强推、不传大数据、不自动停旧任务。当前不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获。查询和时间记录但不要求优势；工程可行与Diffusion收益分别裁决，回退/共同修正/反馈控制贡献不归给网络。

统一只读终端.\ff_prior_fullchain\job.ps1 status|next|report。每阶段交付自包含中文报告链接、实测正负结果、提交、命令和下一步。只有公开任务到路径/避障接近/实际执行/连续交接/动态追踪/学习评价都有真实证据及可复现交付才完成长期工程目标，科学优势另判。确无可推进工作按goal规则报告具体阻塞，禁止无意义轮询；用户要求停止或工程实际交付完成前不自动暂停持续接续。
