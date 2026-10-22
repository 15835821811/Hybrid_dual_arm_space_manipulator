# Experiment Tracker：持续工程全链路

更新2026-10-22。P2a第一版结束、独立审查和完成resume已收尾；唯一响应余量修复已冻结并后台启动。长期工程目标ACTIVE；P1五障碍、P2b和学习链仍未完成。

|Run/阶段|真实状态|证据与限制|
|---|---|---|
|P0 preserve323|CLOSED_NEGATIVE|3e659ce4，禁止重复resume/审计|
|static_fullchain_001|CLOSED_DEPLOYMENT_FAILURE|未知调用保留|
|static_fullchain_002|CLOSED_PARTIAL|三障碍6/6单元执行重放，五障碍0/6；60提案、23完整验证、1预算未决、36仅筛选，12合格6实际执行|
|dynamic_reference_001|CLOSED_PARTIAL|4/4参考、3/4真实执行，四次完整重放差0；t00仅实际加速度失败。108000步，未新增TRAIN或训练|
|dynamic_margin_001|RUNNING|固定rho=.9695632054405793；最多1新执行+3精确历史复用，原任务门不变|
|P2b动态DEV交接|BLOCKED_DEPENDENCY|4/4门未过；12逻辑单位、297000新步剩余，至少1精确复用，不改任务凑数|
|P3时序候选|QUEUED|原32TRAIN/8VAL资格；一个候选，数据不足则BLOCKED|
|P4整链开发验收|QUEUED|最多30执行，五障碍与完整连续组合仍缺|

新独立review：paper/review-traces/experiment-result-to-claim/20261022_dynamic_reference_run01；C1局部partial/high，C2 NOT_EVALUATED；缺少PAPER_CLAIM_AUDIT，论文结论provisional。166源码/86输入和完整状态/控制/几何/原目标核对通过；不能把表示过门当执行通过。完成resume仅一次，64核心文件hash完全保持，UTF-8完整输出已存。

报告ff_prior_dynamic_reference/docs/ANALYSIS.md。修复经验rho来自已保存TRAIN实际加速度与参考之差，须新执行；旧失败永久保留。P2总步数432000不增，修复27000从未启动P2b转移；只运行一次，不调门、不扩库、不启动TEST。控制文件与run/status为实时事实。

修复源码d5ba2106，启动回执ff_prior_dynamic_margin/evidence/launch_receipt.json（PID创建时间匹配）；唯一接续已更新，原P2a收尾提交95735ab0。实时结果以统一status为准。
