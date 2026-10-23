# horizon_restore_001 已收尾 / 双侧布局待确认新运行范围

|单元|固定分母|状态|证据|
|---|---:|---|---|
|horizon_restore_001/u00 原五障碍|1|CLOSED_EXECUTION_VERIFIED|MODEL_VALID、67力矩执行、完整836重放全通过；C1/Caux yes/high；一次完成resume PASS|
|原五障碍动态 horizon_tracking_001|未冻结|NOT_STARTED_SUPERSEDED_BY_LAYOUT_STEERING|实现草稿保留；未启动物理，不偷换场景|
|刚性2 / 连续体5新布局|布局选择1|DESIGN_READY_NEW_RUN_SCOPE_PENDING|原7航点、臂身约束、一次固定初态几何检查；新物理0|

准确分析：ff_prior_horizon_restore/docs/ANALYSIS.md。新布局与预算：ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md。
原五障碍Git基点bfc4ea9加142实际源码hash/zip及233输入为身份；保留一个用户已有hierarchical_qp归档差异，非纯clean-tree提交。
2681耦合、1987216模型区间、27500原生含重放、468.375秒。2249检查中2248直接通过，1项Git差异解释，0未决。
旧run禁止再审查/resume/物理；新布局不能在旧末态中变更障碍后拼接。长期目标未完成，无训练/扩库/对照/TEST。
