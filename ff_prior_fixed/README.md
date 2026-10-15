# 固定 TRAIN 目标修正与一致性诊断后台程序

详细预算和判据见 `docs/EXPERIMENT_PLAN.md`。本包独立于历史 `ff_prior`/`ff_prior_next`，保留它们的代码及结果。

从仓库根目录运行：

```powershell
.\ff_prior_fixed\job.ps1 start
.\ff_prior_fixed\job.ps1 status
.\ff_prior_fixed\job.ps1 report
.\ff_prior_fixed\job.ps1 resume
Get-Content artifacts/v7_1/fixed_train_consistency_001/events.jsonl -Tail 10 -Wait
```

等价入口 `.venv\Scripts\python.exe -m ff_prior_fixed.job status`。start只允许新目录，已存在的结果不覆盖。完成后的resume只读核对，不重做训练或物理执行。异常退出后显式resume，仅在确认旧PID已死后恢复锁；中断候选不补样。状态与自动报告不调用LLM。

默认一轮：8个固定TRAIN目标、16次有界修正、最多4个预选TRAIN执行/独立重放、200步残差训练、原4个DEV任务×3方法×5候选。没有pilot/TEST/生产开关，也没有自动调参。

状态见 `status.json`；自动报告 `REPORT.md`；逐候选、修正失败、预选诊断、完整状态/力矩、独立重放和源码哈希均保存在独立运行目录。只有通过MODEL_VALID和原冻结目标的修正样本加入TRAIN；本轮已知执行失败的标签排除，未执行数据不升级。

首次启动前必须存在与当前源码匹配的回归证明 `evidence/software_tests.json`。改代码后先运行 `pytest ff_prior/tests ff_prior_next/tests ff_prior_fixed/tests` 并重新记录JUnit及源码哈希。旧checkpoint和原始数据不变。
