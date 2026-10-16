# 覆盖扩库的训练接续

## 已完成：qualified_coverage_train_001

2026-10-17，正式worker182540已FINISHED并退出，实际耗时4624.078秒。原始证据审计PASS，独立结果复核支持有限闭环主张；不将全部长期目标记完成。

- 实际加载12144 TRAIN＋48 VAL；6000更新、384000样本消费，全TRAIN每行31–32次；温启动、新权重改变与重载证明通过。
- 11960准确任务敏感度：9912精确旧缓存、2048新计算；历史386568及本次79872耦合调用分账，旧失败unknown≤39保留。
- 原冻结DEV四方法各60/60合格，各种子20/20；各12/12选定真实力矩执行及独立重放通过。共240候选、48执行、504000物理步。
- 新模型修正前pose-pass10/60，旧10k为3/60；最终成功率均饱和，同噪声结果混合，不能宣称全面优势或新场景泛化。

中文分析`ANALYSIS.md`；完成证据`../evidence/completed_001/`；独立复核`../../paper/review-traces/experiment-result-to-claim/20261016_expansion_run01/coverage_training_final_review.md`。终端使用`./ff_prior_coverage_train/job.ps1 status`及`report`，无需重启已完成作业。相对效率不设门槛，未启动最终TEST或十万规模。

## 启动与运行历史

最新接续：父数据作业通过全部数据门槛与48条执行/重放审计并退出；queue已FINISHED且worker81992退出，含义仅为成功派发。qualified_coverage_train_001已实际启动，worker182540，加载12192实际数组（12144 TRAIN/48 VAL）。权重更新、敏感度和冻结DEV尚未完成，数据成功不替代训练结论。下方为启动时历史快照。

2026-10-17：实现提交61143fb5。最终源码4项实际保存数组/缓存/权重更新测试通过，测试不计正式训练。只读预启动审阅的20GiB阈值和源码证明问题均已关闭。

coverage_training_queue_001已启动，实际worker81992存活，固定等待数据worker135524退出并通过准入、48条执行重放及原始审计。父作业已实际完成2048条前向准入，65条拒绝保留，正在96次固定目标修正阶段。启动证据：evidence/queue_launched_001.json。

qualified_coverage_train_001尚未开始，6000更新、全体TRAIN消费、新敏感度及冻结DEV结果均待实测。队列完成只表示派发，不等同训练完成。每个阶段以自己的job.ps1 status/report为准。无最终TEST或自动10万条任务。

终端：`./ff_prior_coverage/job.ps1 status`，`./ff_prior_coverage_train/queue.ps1 status`，`./ff_prior_coverage_train/job.ps1 status`及`report`。
