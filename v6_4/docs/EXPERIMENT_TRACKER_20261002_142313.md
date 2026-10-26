# V6.4-A Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| P01 | M0 | 独立任务协议 | 三族 12 task groups | 6 train / 3 val / 3 test | task SHA、group split、中段 FK/间隙 | MUST | FROZEN | `v6_4/output/tasks/protocol_b_01.json`；任何新增物理前冻结 |
| R01 | M0 | 旧行为不变 | planner disabled | 历史五场景 | 25/25、11/11、6755/12350、5×90 非计时 arrays | MUST | COMPLETE | root 保存回归证据；不计新增学习测试 |
| R02 | M0 | 实际机制验证 | 新任务 fixed EE train000 | train | 27s + 新任务/执行/区间/几何 | MUST | COMPLETE | `teacher_bootstrap/fixed_ee_train_000_01`；初次 evaluator 缺模块失败保留，随后独立补评，不重跑物理 |
| R03 | M1 | 首个真实 Teacher | EE train000 route0 | train | 27s + complete/evidence_valid/task_success | MUST | COMPLETE | `teacher_closed_loop/ee_train_000_route0_01`；单例成功不代表三族完成 |
| R04 | M1 | 三族训练数据 | 各 train task 3 starts | train | 每族至少一任务完整通过；distinct routes 实测 | MUST | TODO | 6 tasks；所有失败保存，目标每任务至少两条成功不同路线 |
| R05 | M2 | 训练链路 | 小 Transformer DDPM overfit | train | finite loss、checkpoint、数据/normalizer SHA | MUST | TODO | 不宣称泛化 |
| R06 | M2 | 首版学习模型 | train-only fit，val 选权重 | train / val | DDIM20 生成、loss/权重/来源 | MUST | TODO | 不使用 test 轨迹或未来实际状态 |
| R07 | M3 | 主组对照 | fixed / Teacher / Diffusion K1 | test 三任务 | raw/conditional/total completion、全部失败 | MUST | TODO | 9 单元；无修复无 fallback |
| R08 | M4 | 搜索归因 | 同方法 K8 | test 三任务 | 8 候选、同预算/全部成本、来源 | MUST | TODO | 与 K1 独立分母；完整仿真筛选需单独声明 |
| R09 | M4 | 工程增强归因 | repair / fallback | test | 来源/开销/直接模型分子分离 | NICE | NOT_RUN | 不用未运行机制补主组成功 |
| R10 | M4 | 结果与局限 | 全尝试自动报告 | 所有 splits 单列 | 任务/执行/安全/性能/归因 + manifest | MUST | TODO | 小测试集、名义模型、非实时研究；不改历史失败 |

状态随实际输出更新；本 tracker 中的完成项仅对应已存在原件，其余均不预报结果。
