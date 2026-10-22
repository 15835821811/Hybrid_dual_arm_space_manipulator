# Experiment Tracker：持续工程全链路

更新2026-10-22。P1第二轮真实结束并完成新独立审查；P2a独立实现已冻结，等待源码提交后启动。长期工程目标ACTIVE，P1五障碍未通过；科学收益单列。

|Run|阶段|真实状态|证据与限制|
|---|---|---|---|
|preserve323_bounded_001|P0|CLOSED_NEGATIVE|3e659ce4已收尾，禁止重复resume/审计|
|static_fullchain_001|P1|CLOSED_DEPLOYMENT_FAILURE|旧长路径故障，未知调用保留，不resume|
|static_fullchain_002|P1|CLOSED_PARTIAL|60提案、24修正、23最终认证、1预算未决、36仅初筛；12合格，6实际执行重放通过；三障碍6/6单元，五障碍0/6且未执行|
|dynamic_reference_001|P2a|FROZEN_NOT_STARTED|原4TRAIN来源，C2样条受约束重建，27秒/原门，最多4执行重放；3数值软件测试通过不等于物理通过|
|动态DEV交接|P2b|QUEUED_DEPENDENCY|P2a通过后另冻原动态任务/12执行上限|
|时序候选|P3|QUEUED_DEPENDENCY|现有32TRAIN/8VAL逐条认证；单候选，数据不足可BLOCKED，不扩库|
|固定整链开发验收|P4|QUEUED_DEPENDENCY|五场景、最多30执行；静态五障碍仍缺，不能提前通过|

P1新review：paper/review-traces/experiment-result-to-claim/20261022_static_fullchain_v2_run01/verdict.json。C1 partial/high；C2 partial/medium仅三障碍执行精度，成功率优势no。三种子保持段最坏臂位置均值2.577016→0.078901mm；五障碍11完成验证全部终点失败，8同时碰撞。251完整修正记录中226内层SLSQP未报告success，不能写成收敛。4211+39只读核对通过，未重跑物理。

完成resume仅一次：2476核心文件不变、exit0，stdout因编码捕获失败未保存，收据如实记录。报告ff_prior_static_fullchain_v2/docs/ANALYSIS.md。没有新训练、TRAIN新增或TEST；6未执行合格路径只模型级，6实际执行才执行级。

控制文件与运行status为实时事实，不以本表排队状态代替运行状态。P1缺口不阻断独立P2；同瓶颈至多一次机制修复、消耗仍计原总预算。唯一持续heartbeat保留，不在局部收尾时暂停。
