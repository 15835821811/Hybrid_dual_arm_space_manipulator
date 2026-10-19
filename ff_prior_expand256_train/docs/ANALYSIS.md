# 扩库后训练：待依赖门

程序已实现，8/8软件及既有模型/缓存来源检查通过。256槽扩库仍在运行；必须待其完成、审计和工程门全部通过，再冻结合并TRAIN准确数量并训练。当前没有新训练或DEV实测结果，不得写成通过。

独立等待器负责一次条件启动。终端 .\ff_prior_expand256_train\job.ps1 status|report|resume。协议EXPERIMENT_PLAN.md；不继续扩库，不运行最终TEST。
