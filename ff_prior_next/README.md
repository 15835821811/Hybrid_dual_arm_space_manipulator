# V7.1 后台残差先验 smoke

详细目标、预算、判据见 `docs/EXPERIMENT_PLAN.md`。这个独立包复用原物理链，不改历史 `ff_prior` 实现；没有 pilot/TEST/生产入口。终端查询和自动报告不调用 LLM。

从仓库根目录运行：

```powershell
# 启动一次隐藏窗口的独立后台进程；默认目录已运行过则拒绝重复启动。
.\ff_prior_next\job.ps1 start
# 只读查询 PID、阶段、进度、心跳及累计时间。
.\ff_prior_next\job.ps1 status
# 运行结束后直接打印自动报告。
.\ff_prior_next\job.ps1 report
# 异常退出后显式恢复；完成项不重做，失败/中断候选不补样。
.\ff_prior_next\job.ps1 resume
# 需要实时尾随日志时使用终端即可，无需聊天轮询。
Get-Content artifacts/v7_1/public_seed_residual_001/events.jsonl -Tail 10 -Wait
```

若本机PowerShell脚本策略阻止 `.ps1`，使用等价Python命令（不改变系统策略）：

```powershell
.venv\Scripts\python.exe -m ff_prior_next.job status
.venv\Scripts\python.exe -m ff_prior_next.job report
```

输出包括 `run.json`（冻结协议/源哈希）、`executed_source.zip`、训练checkpoint、逐候选JSON、预选记录、真实执行与独立重放报告、`summary.json`、`REPORT.md`、`status.json` 和日志。后台进程能够在聊天结束后继续，但需要电脑保持开机；本程序不改变系统电源设置。

没有自动定时唤醒Codex或LLM报告生成。分析报告由本地程序根据固定判据生成；终端 status/report 查询不重新计算实验。完整运行后的 resume 检查身份后返回 `NO_WORK_ALREADY_COMPLETE`。
