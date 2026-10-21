# Fresh independent paper architect/reviewer task

你是全新的独立 paper architect/reviewer，按 experiment-result-to-claim 技能审查 ff_prior_control_transfer 本轮，不能复用前轮 review。唯一 run 为 artifacts/v7_1/control_transfer_bounded_001，冻结协议 ff_prior_control_transfer/docs/EXPERIMENT_PLAN.md；源码 df13c4e1，启动记录 7d32e4e2。

只读原始数据，不改冻结 root 源码、运行目录或旧文件；严禁额外训练、optimizer.step、物理仿真、旧审计或 resume。允许新包 tools 内写只读审查脚本，新 trace 写 paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01/。已有 summary 只能作为待验证线索。

独立核 170 源码/15412 输入绑定、旧低102权重/codec/随机流与221细节 clean 零初始化、完整基和 TRAIN 有界支持、TRAIN/VAL/DEV 隔离。核6个实际float32重建及3个native67/replay数组；16迁移原始控制与重新神经推理的完整旧/新无引导采样等价。核固定8TRAIN×2种子、临时256AdamW、首末实际before/after与gradients，用全16rollout VJP及手写AdamW核首末，不得 optimizer.step。核前后16保存物理预测、prepare、全部门、归一化Q和世界位置/姿态，原已合格样本退化；临时状态不转候选、初态可恢复，M2/M3/M4阻断与账本无未知工作。教师执行不得冒充网络执行。

特别追溯 coverage_train_00288 seed20270213 的源教师、缓存Z、冻结目标、完整初态和原资格，排除已知标签/缓存错配后再解释控制损失与世界误差不一致。检查主执行的中文 ANALYSIS 数字与主张。只给下一阶段建议，不追加。

判断 C1 功能保持/有限 TRAIN 学习与 C2 相对收益；给出 partial/no 与 not_evaluated 的区别，证据边界、missing evidence、claim revision、next experiments 和 confidence。paper claim audit 缺失则 unavailable/provisional。输出 prompt.md、response.md、verdict.json、inputs.json、checks/handoff、ff_prior_control_transfer/docs/CLAIM_VERDICT.md。主执行负责根 findings/tracker、一次完成resume、统计、提交及暂停接续。

Input list: frozen plan and sources/archive; run identity/binding/selection/codec; original aligned model/codec; expand256 cached arrays and every bound record; six reconstruction arrays/predictions; TRAIN_diagnostic before/after checkpoints, first/last updates and gradients, migration NPZ, all 32 controls/predictions/grids/gates; three native/replay/geometry/initial-state packages; ledger/budget/status; original coverage_train_00288 record, arrays, generation proof and complete initial state; final ANALYSIS.md. Exact reviewed paths, SHA256, size and timestamps are in inputs.json.
