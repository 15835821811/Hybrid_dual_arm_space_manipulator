# Bounded reviewer task

请只读分析最终目标的主要瓶颈，暂停新增审计数量与生产改动。结合原始 V6.4-A 目标、已完成的 VAL 实际形状偏差/原约束冲突以及 fixed TEST 3/3，解释：为什么静态IK/几何可行的参考会在弱posture原QP下产生明显实际偏离；哪种最小、统一且不放宽原安全/任务/姿态权重的下一机制最可能带来独立任务的真实闭环结果。区分严格raw与统一工程修复，给出最多2个可执行选项及有限停止条件。不可用TEST未来actual traces作为提案输入，不运行物理、不编辑源码。不要建议换solver/放宽domain修补冻结冲突；不把原Task判物理不可行。

本轮应用 experiment-result-to-claim 技能，请以独立 paper architect/reviewer 身份给结构化判定：claim_supported yes|partial|no、what_results_support/dont_support、missing_evidence、suggested_claim_revision、next_experiments_needed、confidence。核心claim是 learned提案经既有安全控制产生独立Task闭环结果，且优于fixed；当前fixed TEST 3/3，Teacher24raw0/Diffusion24raw0且actual0；修复仅VAL实际11560step失败。不把接口/训练真实当优势通过。

请额外判一项协议问题（只分析不实现）：原目标要求独立任务 raw 参考真实执行，即使最终失败；但当前gate先因名义Task误差拒绝并跳过geometry，导致所有learned actual0。是否可独立新增明确标为“任务预测失败候选的闭环诊断”（不计主组接受/成功）：保持原Task判据/所有hard q,dq,ddq/PCC/nativegeometry门禁及原实际执行链，只在完整独立安全几何通过后力矩执行原始未修复候选，并照原Task验收保存失败？名义Task不成功是质量指标，不是修改安全约束。根据用户原文/已冻结实验承诺指出这是否可作为最小负闭环交付，还是确实需人类显式修改scope。原 run_attempt全fullgate requirement仍不动；若需新增路径必须独立显式记录 Task prediction fail、raw primary rejection，不能“恢复rawpass”。

Exact bounded input files and SHA/size/mtime are in inputs.json. No paper claim audit is available.
