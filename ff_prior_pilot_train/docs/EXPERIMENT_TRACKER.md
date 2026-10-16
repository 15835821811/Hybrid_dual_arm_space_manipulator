# 后台训练接续跟踪

- qualified_pilot_001：数据生成中，worker105972；不能预写数据审计通过。
- pilot_training_queue_001：RUNNING，worker114180；实际等待父worker退出与数据验收，源码8a332a9a。
- qualified_pilot_train_001：尚未开始；正式12000次更新、10000 TRAIN及冻结DEV结果均待真实运行。
- 四项真实小数据软件测试通过，独立预启动审查未发现阻断项；这些证据不替代正式实验。

终端：ff_prior_pilot_train/queue.ps1 status；训练开始后用job.ps1 status/report/resume。查询和时间为辅助记录，最终TEST保留。
