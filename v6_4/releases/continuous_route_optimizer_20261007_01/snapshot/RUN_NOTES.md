# C.1 运行附注

45项搜索与原执行/参考回归单测通过，0新物理步。正式任务、两偏好、12槽预算和producer均在候选前冻结。原执行模块、QP、模型资产及全部安全容差无源码变更。

主调度器按母场景00的左右任务顺序执行搜索；母场景01左右任务由两个独立进程调用同一已冻结optimize API。进程拥有各自私有模型/数据/控制器，候选目录互不重叠。主调度器随后仅核验它们的封存selection，不重复积分。全部四selection封存后才开始actual。并行只改变任务调度，逐任务候选算法、顺序、中心、预算均保持冻结。

最初两个并行包装脚本因文件入口未包含仓库sys.path而在导入阶段退出。0评价槽、0预测积分；原日志保留，task_worker_v2只修正路径。原算法producer不变，脚本版本SHA与恢复收据分别存于scheduling_plan.json、scheduling_recovery.json和scheduling_receipts。没有重试任何已消费候选。

每候选及每实际执行账本的elapsed_wall_s为其服务时间，多个并行预测的服务时间之和不等于总经过时间。command_receipts和scheduling_receipts记录各入口UTC开始/结束与退出状态。几何、QP、main integration、嵌套十步preview及独立replay按操作/阶段分别保存，累加工作量不受并行调度影响。全部墙钟仍只作诊断。

旧B.2已使用建议种子，故运行前按next-unused选择source index12/13；没有读取新任务结果择种子。旧28行只读质量整理不增加任何旧物理重放。
