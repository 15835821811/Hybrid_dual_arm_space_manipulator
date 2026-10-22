继续本线程用户2026-10-22最新授权的持续全链路目标：从公开初态、障碍、世界目标和时间条件出发，完成路径生成、全机避障、刚性臂预抓捕接近、原生67力矩执行与独立重放，再接移动卫星和连续体七不规则航点追踪，以及真实训练/完整采样/公平对照。工作区E:\桌面\mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main\Hybrid_dual_arm_space_manipulator。有效长期目标DIFFUSION_GOAL_CURRENT.md，详细路线ff_prior_fullchain/docs/ROADMAP.md，当前阶段ff_prior_fullchain/control.json。不是再做一个孤立的小损失试验。工程全链路与Diffusion相对收益分开裁决；优势未成立不禁止推进工程阶段，失败不能写通过，回退规划器/共同控制器的成功不能归给网络。

每次先执行只读 .\ff_prior_fullchain\job.ps1 status，读取control.json所指active_run的status及PID/创建时间。RUNNING、PID身份匹配且心跳健康时安静结束，不读中间结果、不分析、不重复审计、不启动第二worker或重做resume。只在完成、失败、阶段真正切换或需要用户处理时通知。当前唯一run是artifacts/v7_1/preserve323_bounded_001，源码d5e1b7c5，冻结协议ff_prior_preserve323/docs/EXPERIMENT_PLAN.md，原入口 .\ff_prior_preserve323\job.ps1 status|report|resume。该运行完整保持原预算、数据隔离、数值/学习门与条件下游；新长期目标不得改其源码或放行失败门。

当前P0仍采用323输出、旧aligned低102冻结、221细节训练、分层fit16×2与检查16×2、检查任务对应34行排除、14188适配TRAIN/48VAL。M1最多3轮完整323Jacobian；fit门及独立检查门通过才训练一个6000更新候选，后续96无引导、原DEV新180/36及应用新10/2，旧身份合格对照复用；未运行不报候选成功率。具体门与预算以冻结协议和源码为准，不用本摘要替代。2线程/6h/20GiB/160000耦合。原始结果完整保留。状态异常先诊断，未知工作不重放；必要代码bug修复使用新版本/输出，不能覆盖冻结身份或因效果不好追加训练。

P0终态后只做本轮必要收尾：核原始TRAIN_baseline/fit/check/gate、分层原资格保护、低102保持、独立排除、完整梯度/神经VJP/保存参数增量，若后续运行核实际训练消费/重载、无引导、原DEV与静态应用逐任务种子配对、实际执行与历史复用。核189源码及准确绑定清单，数量以原preflight为准。依experiment-result-to-claim技能委派一次全新无历史独立审查，不复用旧review；使用保存数据的只读审计，不追加物理仿真来追认。完成resume仅用本包tools/check_completed_resume.py一次，保存核心清单/hash不变，不重做旧run的resume。写中文ff_prior_preserve323/docs/ANALYSIS.md、紧凑证据、新review-traces，版本tracker和本地提交，只提交任务文件。写ff_prior_fullchain/evidence/preserve323_closeout.json，字段run、closed、commit、analysis、independent_review、completed_resume指真实文件。closed仅表示有界工作收尾，不表示全部物理或科学门通过。

P0完成后不要按旧定时提示暂停整个接续。保留局部失败结论、使用合格旧模型组件，按新路线自动继续已授权的有界开发阶段：P1静态三/五障碍任务输入→323提案→共同精细碰撞/目标修正→执行重放；P2修复旧4TRAIN动态表示的加速度问题并接已有合格控制器，验证规划到追踪状态连续；P3既有合格TRAIN/VAL上的一个有时间/完整任务/障碍条件的候选和确定性对照；P4固定开发套件的整链验收与交付。先读相关已完成源码/报告，复用身份匹配证据，不重做已经解决的rank/prepare/存在性诊断。DEV07路径只做回归，不加载为部署答案、不入TRAIN。下一阶段准确协议、任务/种子/预算/源码冻结后才启动独立后台worker，并更新control.json、统一目标与版本tracker。每阶段预算和停止/分支规则见ROADMAP；同一时刻只一个新重任务，不无限重复局部调参。一个分支失败，完成报告后继续不依赖它的工程工作；不能把所有后续都错误绑定到Diffusion必须先有优势。

应用目标卡当前旧有界目标显示blocked，实际worker活着；状态不是工作完成证据。工具不支持原地改未完成目标正文或恢复状态，不因此阻断文件目标和heartbeat。只有P0原有界工作客观全部收尾后才能把旧卡complete，然后创建新的长期目标卡，内容取DIFFUSION_GOAL_CURRENT.md的持续目标与终止门。不要为更名提前complete，不把局部卡片complete解释为长期科学完成。

所有阶段保持仅不穿透、双臂世界5mm/1度、2mm几何步长、完整初态、17→67、自由漂浮、全机几何、运动/力矩/独立重放原物理门；动态原合同不放宽。原软件测试不是物理证据，未执行仅MODEL_VALID，逐条实际力矩与重放通过才EXECUTION_VERIFIED。新数据/旧数据/新执行/历史复用分开，失败和缺失分母保留。不失败重标、mock、固定基座、隐藏重置或抽查升级。保留main/旧分支、所有旧模型数据结果、用户修改和旧作业；不强推、不上传大数据、不停止旧进程。

当前仍不扩库、不1024/30000生产、不pilot/study/strict_multimodal、不最终TEST、不硬件或接触捕获试验。对既有数据重新认证可以按冻结预算进行，开发路径不得回灌TRAIN。用户的“不惜一切代价”表示持续推进和机制改进，不是放宽物理/诚信合同、无上限资源或预设一定有优势。TEST只记录就绪门及待授权，不接触或调参。查询和耗时只记录，不要求胜出。

终端 .\ff_prior_fullchain\job.ps1 status|next|report 始终显示真实当前阶段；命令本身只读，无后台重复启动。每个阶段最终交付自包含中文报告、实测结果和失败边界、提交、下一阶段和命令。只有全链路开发交付条件客观满足才完成长期工程目标，再分别说明科学优势与TEST状态。若确实无可推进的独立工作且需要用户输入，具体说明缺口；遵守goal工具blocked规则，避免无意义轮询。不要因为局部阶段完成自动暂停持续接续；用户要求停止，或本轮定义的全链路工程交付已完成，才按实际授权收尾管理接续。
