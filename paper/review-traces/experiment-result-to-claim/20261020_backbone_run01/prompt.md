# 全新独立 RESULT-TO-CLAIM 审查

你是未参与本轮实现的paper architect/reviewer。仅按原始证据判断，不参考旧审查或其他对话，不修改冻结代码或原始结果，不启动训练、optimizer.step或物理仿真。工作区 E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator，Python .venv/Scripts/python.exe；限制torch/numpy线程2。

本轮run artifacts/v7_1/backbone_pair_smoke_002，冻结源码4ccba221，计划ff_prior_backbone_v2/docs/EXPERIMENT_PLAN.md。源码闭包168项、输入绑定28861项。前驱001由于首次证据目录写入失败、0正式更新，保留于ff_prior_backbone_v2/evidence/failed_launch_001.json，不把它当科学结果。本轮软件14测试不是物理证明。

C1：同潜变量/条件/数据/损失/预算下，小型Transformer改善去噪生成世界终点质量。主要证据完整16步无引导VAL48×2采样种子，每模型96；主指标全部门合格率，Q/位置姿态均值、中位、P95、最大与配对辅助。
C2：主干变化转化为完整规划收益。原DEV4+已用8、3采样种子、5候选每任务，新MLP/Transformer各180候选/36原生67力矩执行+独立重放；同late4。强样条/条件均值旧360/72只准确复用，不能写新增。所有方法成功率相同不能声称成功率优势。

固定14222TRAIN/48VAL、codec102/condition65。两神经网络从头初始化同seed20263401，不同架构不能相同权重；共用TRAIN拟合codec。MLP208998参数，Transformer211969，6个17维基底系数token。各6000普通更新batch64/AdamW2e-4，seed20263001、300段batch/t/noise精确匹配彼此和旧A。v+.05教师局部G，零新增非线性世界监督。单训练种子（采样种子不是训练重复），固定6000终点，不用VAL选checkpoint。主干一起改变归一化/残差/注意力，不单独归因注意力。

先读run.json、ledger.json、learning/*/training.json及progress.pt、预选updates/0001/3000/6000 before/after.pt、unguided、candidates、executions原始文件，paired_statistics与independent_verification仅作待核对摘要。实际原始摘要称两新主干180/180和36/36，均0/96无引导；执行位置均值MLP0.787201mm/T0.659327mm。请自行重算完整指标及缺失分母，不先接受结论。

只读核查：源码/输入绑定与源zip；两组真实6000和全部TRAIN消费，300段同序列；六个实际预选更新的神经梯度/手写AdamW及非空矩与计数，不再optimizer.step；完整无引导不是late4末步raw；原DEV同冻结目标/种子/候选预算与选择先于执行；72实际执行重放证据、仅不穿透/5mm/1度/完整初态/17→67/自由漂浮/全机几何/原运动力矩门、无未知工作。可用现有只读代数/轨迹审计helper，但不要重复旧baseline物理审计，不用单独summary推断PASS。检查2CPU/6h/20GiB/42000耦合上限和本轮来源。需要辅助脚本仅放本review目录或ff_prior_backbone_v2/tools，不改root源码闭包。

工程门与收益分开：完整工程有效无需相对优势，也无需无引导合格才准DEV。不得因负结果追加实验或修改物理门。不能把fresh模型与历史warm模型差异单独归因主干。

输出到本目录response.md（中文详细原始审查）与verdict.json。字段逐主张claim_supported yes/partial/no、what_results_support、what_results_dont_support、missing_evidence、suggested_claim_revision、next_experiments_needed（仅建议不执行）、confidence，另列engineering_verdict、核查范围与证据局限。若出现严重证据问题即刻回报主代理。论文PAPER_CLAIM_AUDIT若不存在则标unavailable、下游写作provisional；不阻塞本轮实验报告。请明确是否推荐替换默认模型或继续扩库，以及目前证据允许的下一阶段。

禁止新试验、数据扩库、pilot、study、strict_multimodal、TEST。保留现有结果。主代理会另行验证完成resume一次且核心hash不变，之后补入handoff；审查不要调用resume。仅写review目录，不提交git。
