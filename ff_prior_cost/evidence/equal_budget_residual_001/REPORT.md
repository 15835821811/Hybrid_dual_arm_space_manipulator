# 同预算残差学习与采样 smoke 报告

状态：FINISHED。
学习修正成本收益：NOT_ESTABLISHED；相对样条的Diffusion必要性：NOT_ESTABLISHED。

| 分组 | 方法 | 模型有效 | 世界目标合格 | 同时合格 | 执行+重放 | 总耦合查询 | 候选秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| DEV | structured_final | 20/20 | 20/20 | 20/20 | 4/4 | 1110 | 18.838 |
| DEV | frozen_diffusion_two | 20/20 | 20/20 | 20/20 | 4/4 | 1935 | 31.838 |
| DEV | frozen_diffusion_final | 20/20 | 20/20 | 20/20 | 4/4 | 1251 | 21.527 |
| DEV | trained_diffusion_final | 20/20 | 20/20 | 20/20 | 4/4 | 1251 | 21.495 |
| VAL | structured_final | 30/30 | 30/30 | 30/30 | 6/6 | 1705 | 29.132 |
| VAL | frozen_diffusion_two | 30/30 | 30/30 | 30/30 | 6/6 | 2995 | 49.138 |
| VAL | frozen_diffusion_final | 30/30 | 30/30 | 30/30 | 6/6 | 1985 | 34.219 |
| VAL | trained_diffusion_final | 30/30 | 30/30 | 30/30 | 6/6 | 1985 | 34.097 |

合格教师标签：{'TRAIN': 30, 'VAL': 6}。TRAIN仅用于梯度，VAL仅用于固定损失检查及权重选择；标签仅MODEL_VALID，未批量升级执行等级。
微调状态：PASS。选择更新：0。小样本拟合：{'claim': 'pipeline capacity sanity only; no generalization claim', 'final_loss': 0.06104006618261337, 'forward_calls': 82, 'initial_loss': 0.8641008138656616, 'ratio': 0.07063998228348277, 'status': 'PASS', 'training_ids': ['cost_TRAIN_000', 'cost_TRAIN_001', 'cost_TRAIN_002', 'cost_TRAIN_003', 'cost_TRAIN_004', 'cost_TRAIN_005', 'cost_TRAIN_006', 'cost_TRAIN_007']}。

DEV开始前按VAL冻结推荐：structured_final。四种对照及种子预先固定，DEV结果不用于再次调整。

## 误差来源

- TRAIN归一化12维世界位姿误差范数中位数：{'initializer': 2.809228010641033, 'first_network': 4.320318177891068, 'repaired_14': 0.03168438835034987, 'network_15': 0.5409995868116437, 'prior_15': 0.7965909513900293, 'smooth_15': 0.7972099319912043, 'repaired_15': 0.005616591575892588}。
- VAL归一化12维世界位姿误差范数中位数：{'initializer': 3.186885622507379, 'first_network': 4.722562766026353, 'repaired_14': 0.058039557954612485, 'network_15': 0.5229278768356471, 'prior_15': 0.989655372311982, 'smooth_15': 0.9926833310543357, 'repaired_15': 0.003069911021186704}。

## 判据与边界

所有方法5候选/任务、最多110次修正耦合查询+3次完整模型验证查询。两步方法为末步保留36次查询；没有达到预算上限的候选不补用剩余额度。
共享17方向修正、全机几何及原物理门槛。仅不穿透>=0m；5mm/1度；2mm几何细化；17→67映射；完整初态、自由基座、真实67力矩mj_step、独立ctrl重放。
收益须VAL与DEV都不降低合格数/执行成功，耦合查询至少少20%且候选wall不更高。额外训练成本另列，未算成免费收益。
每任务每方法预选一条执行。候选模型通过不等于全部真实执行。背景诊断探测独立计账，不计入正式候选，也不隐瞒为零。
world目标误差实际前向诊断；训练是关节残差监督，并未声称实现可微全动力学损失。控制点多样性不等于避障拓扑或strict_multimodal。
本DEV已多轮开发，只支持smoke结论；不支持独立泛化、硬件或微重力是唯一失败原因。
阶段wall：{'diagnostic': 20.403040699660778, 'teacher': 36.52323449915275, 'training': 5.6722465003840625, 'candidates': 243.24228299921378, 'execution': 434.3989784014411}；尝试数：{'diagnostic': 12, 'teacher': 36, 'training': 1, 'candidates': 200, 'execution': 40}。所有失败保留，依赖失败为BLOCKED，未运行pilot/TEST。
