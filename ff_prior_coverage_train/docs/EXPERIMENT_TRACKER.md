# 覆盖扩库的训练接续

最新接续：父数据作业通过全部数据门槛与48条执行/重放审计并退出；queue已FINISHED且worker81992退出，含义仅为成功派发。qualified_coverage_train_001已实际启动，worker182540，加载12192实际数组（12144 TRAIN/48 VAL）。权重更新、敏感度和冻结DEV尚未完成，数据成功不替代训练结论。下方为启动时历史快照。

2026-10-17：实现提交61143fb5。最终源码4项实际保存数组/缓存/权重更新测试通过，测试不计正式训练。只读预启动审阅的20GiB阈值和源码证明问题均已关闭。

coverage_training_queue_001已启动，实际worker81992存活，固定等待数据worker135524退出并通过准入、48条执行重放及原始审计。父作业已实际完成2048条前向准入，65条拒绝保留，正在96次固定目标修正阶段。启动证据：evidence/queue_launched_001.json。

qualified_coverage_train_001尚未开始，6000更新、全体TRAIN消费、新敏感度及冻结DEV结果均待实测。队列完成只表示派发，不等同训练完成。每个阶段以自己的job.ps1 status/report为准。无最终TEST或自动10万条任务。

终端：`./ff_prior_coverage/job.ps1 status`，`./ff_prior_coverage_train/queue.ps1 status`，`./ff_prior_coverage_train/job.ps1 status`及`report`。
