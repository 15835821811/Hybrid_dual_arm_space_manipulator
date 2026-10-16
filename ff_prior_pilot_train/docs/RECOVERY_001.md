# 10,000条学习接续的文件写入故障修复

2026-10-16：qualified_pilot_train_001 已实际FAILED，PID46588已退出。运行3463.5秒，提交4949个PASS敏感度，第4950个未提交。未完成权重训练或DEV，不构成Diffusion失效或新模型通过的证据。

原sensitivity_failure.json记录PermissionError(13，拒绝访问)，但缺失原异常堆栈，无法确定是哪次文件操作或哪个外部进程占用。随后experiment尝试继续独立基线，Ledger中仍有未结束条目，产生“one owner/one in-flight work item required”，导致整体退出。没有把文件锁猜测写成已证实根因。

修复仅限本训练包：JSON/NPZ/账目/检查点写入遇PermissionError最多重试8次，总等待不超过2.12秒；不重试物理计算、不放宽验证。进度写入异常进入已知调用的失败记账；证明写入最终失败也先关闭账目条目。外层未能确定调用的异常明确记录未知，不阻塞独立基线继续。敏感度异常现在保存堆栈。

接续使用独立qualified_pilot_train_002。旧9917个文件全部SHA256冻结于evidence/recovery_001/failed_run_manifest.json，001不修改。只复用4949个已提交PASS条目：逐个核验准确TRAIN任务、轨迹哈希、样本ID、矩阵形状/有限性、半步/混合方向校验；核对codec、所有物理依赖源哈希，并对原执行源码ZIP中的build/directions/coefficient_map/fit_codec函数比较AST。数值函数或输入不同即拒绝复用。

旧未提交锚点不复用，在新作业中重新计算。其原额外耦合调用未知，保留上限39次的未决成本说明，不算零。已提交缓存的193011次耦合调用计入接续账目。原3463.5秒从14400秒总活跃预算扣除，接续的导入和核验同样占用剩余预算。旧失败不能被新PASS抹去。

12000次更新、batch64、固定训练种子、原4个DEV、3采样种子、5候选、4方法、真实执行与独立重放、物理门槛全部不变。8项软件测试通过，含真实小批训练与耦合矩阵、注入文件故障的恢复测试；不把故障注入或小批测试当成正式实验。

终端：

```powershell
.\ff_prior_pilot_train\job.ps1 status
.\ff_prior_pilot_train\job.ps1 report
```

默认查询002；查原失败使用 `-Output artifacts/v7_1/qualified_pilot_train_001`。仅在确认当前worker退出后使用resume。最终报告须分别说明软件修复、实际训练、采样/执行结果和Diffusion收益。
