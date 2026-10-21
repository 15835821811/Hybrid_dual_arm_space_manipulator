# 原始独立 reviewer 委派 prompt

你是本轮新的独立paper architect/reviewer，按 C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md 独立审查已完成实验。工作区 E:/桌面/mujoco_irregular_waypoints_diffusion_adapter_bundle_20260902-main/Hybrid_dual_arm_space_manipulator。仅本轮 ff_prior_full_control，冻结协议 docs/EXPERIMENT_PLAN.md；run artifacts/v7_1/full_control_bounded_001 已FINISHED，源码f788a689、启动aa3a0d2e。先读协议/source/raw，不只信summary。用户约束：不追加任何实验/训练/物理仿真/optimizer.step，不改冻结root源码或run；旧结果只身份复用不重审旧物理。可在 ff_prior_full_control/tools 和 evidence/completed 新建只读审查辅助脚本及输出，独立trace写 paper/review-traces/experiment-result-to-claim/20261021_full_control_run01/。

请检查：165源码/15411绑定，TRAIN14222 VAL48与DEV隔离/codec仅TRAIN，6重建(实际float32目标)及3新执行重放，6000实际更新/1200完整16DDIM批次/9600教师图路径/数据消费，四保存update1/5/3000/6000实际神经VJP与手写AdamW重建(禁止optimizer.step)，新旧完整无引导和冻结DEV同预算完整分母、native67真实执行及独立重放证据、所有物理门、ledger未知工作、比较干预混杂。核心summary新DEV15/180而旧180/180，待主执行者统计拒绝原因；请独立判断C1工程、C2学习相对收益，不能把完整表示成功当学习收益。源码不含完整非线性世界loss，输出323但引导24方向，无障碍编码；新从头初始化旧预训练。模型资格不升级未执行数据。

需要结构化 verdict claim_supported yes|partial|no, support/dont/missing/revision/next/confidence，paper_claim_audit不存在则provisional。请保存原始prompt/response/verdict/审查输入及hash/检查输出/handoff，以trace全文为准，发给我完成路径和关键结果。主执行者负责统计/中文报告/resume一次/提交，你不执行resume。可建议下一步，不实际跑。注意保留所有用户修改。

# 审查输入与执行边界

- 冻结协议：`ff_prior_full_control/docs/EXPERIMENT_PLAN.md`。
- 本轮 source：`ff_prior_full_control/*.py`、实际依赖源码闭包165个文件及`executed_source.zip`；读取训练、codec、prepare、验证、tracker、完整状态与独立replay实现。
- 主要 raw：`artifacts/v7_1/full_control_bounded_001/{run,ledger,physical_contract,training_input,frozen_tasks}.json`，`reconstruction`，`learning/full`含四步前后权重/optimizer/真实输入/梯度，`unguided`，`comparison`全部候选和全部新执行的native/replay/geometry原始数组。
- 历史输入：`run.json`绑定的15411文件逐文件SHA256核对，历史对照候选和execution outcome仅作身份绑定复用；未重新调用旧审计或旧物理。
- 检查脚本：`ff_prior_full_control/tools/fresh_reviewer_check.py`、`fresh_reviewer_native_math.py`。
- 全量路径、SHA256、大小、mtime_ns见`inputs.json`；检查输出为`checks.json`和`native_math_and_tails.json`。
- 四步VJP和手写AdamW由reviewer完成；主执行者另做原始统计/一次completed resume，reviewer没有调用resume。

# 后续收到的主执行消息（非独立结论依据）

主执行者提供了`ff_prior_full_control/evidence/completed/statistics.json`及`compact_rows.json`的路径，并最终纠正其初步执行scope推测为3TRAIN+13DEV、两APPLICATION_DEV均NO_CANDIDATE。reviewer已从本轮原始records独立核实；没有沿用最初3TRAIN+12DEV+1三障碍的错误推测。
