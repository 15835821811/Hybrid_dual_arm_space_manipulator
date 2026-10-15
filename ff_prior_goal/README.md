# 耦合世界目标约束采样后台验证

从仓库根目录执行：

```powershell
.\ff_prior_goal\job.ps1 start
.\ff_prior_goal\job.ps1 status
.\ff_prior_goal\job.ps1 report
.\ff_prior_goal\job.ps1 resume
Get-Content artifacts/v7_1/goal_projection_001/events.jsonl -Tail 10 -Wait
```

新目录启动，不覆盖旧结果；完成后的resume核对源码/任务/checkpoint身份后不重复工作。Windows隐藏后台进程，5秒心跳，终端查询与报告不调用LLM。异常恢复不退还已预留候选或执行预算。

冻结上一轮已训练模型，只改变低噪声采样的曲率正则和世界目标修正。详见docs/EXPERIMENT_PLAN.md和docs/RESEARCH.md。没有pilot/TEST或生产开关。
