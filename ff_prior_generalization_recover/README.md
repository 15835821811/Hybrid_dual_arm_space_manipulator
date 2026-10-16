# 独立恢复新验证的诊断和审计

原`independent_new_validation_001`在完成720候选、144选定和12教师执行/重放后，首次诊断的进度调用使用了重复`stage`参数，抛出TypeError。原状态保持FAILED，所有文件和失败记录保留。

本程序只恢复未执行的96项引导前独立验证和最终审计。它从原作业导入完全相同的候选、任务、控制和原始轨迹字节，并逐项核查来源哈希；没有重新调用网络采样或重新执行任何native力矩/重放。新的进度明细字段为`proposal_stage`，原冻结源码149项不修改，物理协议、任务、选择和各方法预算不改变。

恢复只接受这一种已证明发生于物理调用之前的异常：traceback、原源码调用边界、Monitor参数签名、首个pending预留项和无任何诊断调用记录必须全部吻合。沿用原预留项，实际诊断尝试仍为96，未退款或增加已执行候选预算。任意未知中断不得按零成本处理或自动重做。原耗时3705.781秒计入7200秒总预算；诊断600秒上限扣除原失败阶段的时间上界，20GiB新增文件上限保留。新增运行时初始化有准备开销，不能把恢复称为全部计算零成本。

原始实验审计继续调用原审计器，另核查恢复代码ZIP、全部导入不可变字节和原失败目录仍未改变。软件修复、完整验证闭合和Diffusion收益分开判定；原结果不因为恢复成功被追溯改写。

```powershell
.\ff_prior_generalization_recover\job.ps1 start
.\ff_prior_generalization_recover\job.ps1 status
.\ff_prior_generalization_recover\job.ps1 report
```

默认输出`artifacts/v7_1/independent_new_validation_recovery_001`。已经启动时不要重复start；完成后的resume先核查身份并返回NO_WORK_ALREADY_COMPLETE。恢复中发生未知物理中断时保留证据，不自动重复预留工作。

4项软件测试使用实际失败记录和真实Monitor，验证字段修复、篡改拒绝、预留计数及未知中断拒绝；不以测试中的账本单元记录充当物理通过。
