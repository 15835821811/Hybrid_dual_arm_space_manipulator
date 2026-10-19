# 当前执行跟踪 delivery_v1

2026-10-19；基点04b54adf；参考900d781。协议paper/EXPERIMENT_PLAN.md。

|块|证据/运行|状态|下一步|
|---|---|---|---|
|B0旧完整机制|qualified_origin_learn_002，稳定13774模型，180/180与36/36|PASS_REUSED|不重跑|
|当前库|13838 TRAIN/48 VAL，分级标签保留|PASS_REUSED|按当前索引使用|
|旧四组训练|D=0.914089；原DEV未运行|HISTORICAL_BLOCKED|不晋级C/D，不阻塞新主线|
|B1采样整合|delivery_guidance_smoke_001|PASS|四方法各80/80、16/16；原始审计PASS；选择late4；收益未成立|
|B2小批补库|delivery_dataset_128_001|PASS|128/128合格、32目标、16/16执行；TRAIN13966/VAL48|
|B3训练和DEV|delivery_train_dev_001|RUNNING|6000更新+720候选/144执行上限|
|B4交付|delivery_release_001|NOT_READY|数据/模型/报告和下一阶段协议|
|TEST/旧1024/pilot|无新运行|NOT_STARTED|本轮禁止|
|研究优势|强基线对照未支持稳定优势|NOT_ESTABLISHED|独立报告|
|应用目标卡片|用户清除旧目标后已实际创建delivery_v1|ACTIVE|按M1→M4推进，同一接续不重复启动|

本轮M1已实际启动，9/9软件检查通过；尚无新物理通过结论。后续以独立目录status/PID/心跳和原始结果为准。

M1耗时1583.625秒，完成resume不重复。M2实现中，尚未启动补库。

M2：7/7软件检查通过，目标/执行单元预冻结，源码待提交后真实启动。

M2实际启动，源码4aeced8a，后台PID/心跳核实；新资格结果待结束。

M2耗时923.265秒，完成resume无重复；M3实施中，不重复M1/M2审计。

M3：7/7软件检查通过，源码冻结后启动；尚无本轮训练结果。

M3已实际启动，源码09ba8bae，独立输出delivery_train_dev_001；PID/心跳已核实，实际训练与DEV结果等待运行结束。
