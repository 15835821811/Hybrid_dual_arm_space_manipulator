# 当前执行跟踪 delivery_v1

2026-10-19；基点04b54adf；参考900d781。协议paper/EXPERIMENT_PLAN.md。

|块|证据/运行|状态|下一步|
|---|---|---|---|
|B0旧完整机制|qualified_origin_learn_002，稳定13774模型，180/180与36/36|PASS_REUSED|不重跑|
|当前库|13838 TRAIN/48 VAL，分级标签保留|PASS_REUSED|按当前索引使用|
|旧四组训练|D=0.914089；原DEV未运行|HISTORICAL_BLOCKED|不晋级C/D，不阻塞新主线|
|B1采样整合|delivery_guidance_smoke_001|READY_NOT_IMPLEMENTED|一次编码检查+320 TRAIN候选|
|B2小批补库|delivery_dataset_128_001|PLANNED|32×4槽，最多256尝试|
|B3训练和DEV|delivery_train_dev_001|PLANNED|6000更新+720候选/144执行上限|
|B4交付|delivery_release_001|NOT_READY|数据/模型/报告和下一阶段协议|
|TEST/旧1024/pilot|无新运行|NOT_STARTED|本轮禁止|
|研究优势|强基线对照未支持稳定优势|NOT_ESTABLISHED|独立报告|
|旧应用卡片|paused|SUPERSEDED_TEXT_NOT_CLEARED|用户/系统清除操作|

本轮目标/协议与状态入口已建立，无新增物理通过；实际运行须由独立目录status/PID/心跳确认。
