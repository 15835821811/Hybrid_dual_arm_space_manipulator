# 有界覆盖扩库接续

2026-10-16：代码c8611237，最终源码5项真实模型软件测试通过，预启动只读审阅发现的路径/分层/恢复计数问题全部关闭。

qualified_coverage_001已实际启动，worker135524。已通过前置10k闭环和退出检查，旧来源及新场景合同校验完成，进入coverage_forward_generation；启动核验时已实际准入104条前向样本。动态状态以job.ps1 status为准，启动证据保留于evidence/launch_001.json，不代表本批完成。

待完成：2048条前向先验、24目标各4次修正尝试、48预选执行与独立重放、原始证据审计和训练接续。未执行条目只为MODEL_VALID，未启动最终TEST或自动10万条。查询/时间不作为相对优胜门槛。

终端：`./ff_prior_coverage/job.ps1 status`、`./ff_prior_coverage/job.ps1 report`、`./ff_prior_coverage/job.ps1 resume`。运行中不重复start/resume。
