# 后台训练接续跟踪

## 2026-10-16 文件故障修复后的接续（当前状态）

- qualified_pilot_train_001已FAILED，worker46588退出；4949个合格敏感度已提交，未完成训练。完整旧结果保留，原因与修复见RECOVERY_001.md。
- qualified_pilot_train_002已启动，实际worker116820；4949个结果经输入、数值函数和矩阵证据核对后导入，原故障锚点重新计算并PASS，正在推进剩余9912个任务锚点。
- 旧3463.5秒计入14400秒活跃预算；原未提交锚点额外调用未知且明确保留，不能写零或隐藏旧失败。8项修复测试通过，不代表正式训练通过。
- job.ps1默认查询002；高噪声诊断ten_k也指向002，待实际训练完成后才运行。冻结DEV、最终审计和新模型收益均待实际结果。

## 2026-10-16 21:19 实测更新（覆盖下方启动时快照）

- qualified_pilot_001已FINISHED，10000 TRAIN/48 VAL，48/48执行重放及独立原始审计PASS，原worker已退出。
- pilot_training_queue_001已FINISHED，含义仅为成功派发训练，原队列worker已退出。
- qualified_pilot_train_001已实际启动，worker46588；完整数组加载后正在构建9912个准确TRAIN任务的耦合敏感度。12000次模型更新、冻结DEV及最终审计尚未完成。
- 数据阶段报告见ff_prior_pilot/docs/ANALYSIS.md。相对效率不作为继续推进门槛；Diffusion新收益仍待实测。

## 以下为原始启动时快照

- qualified_pilot_001：数据生成中，worker105972；不能预写数据审计通过。
- pilot_training_queue_001：RUNNING，worker114180；实际等待父worker退出与数据验收，源码8a332a9a。
- qualified_pilot_train_001：尚未开始；正式12000次更新、10000 TRAIN及冻结DEV结果均待真实运行。
- 四项真实小数据软件测试通过，独立预启动审查未发现阻断项；这些证据不替代正式实验。

终端：ff_prior_pilot_train/queue.ps1 status；训练开始后用job.ps1 status/report/resume。查询和时间为辅助记录，最终TEST保留。
