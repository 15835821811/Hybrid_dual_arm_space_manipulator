# 独立初态覆盖执行跟踪

| Run ID | 阶段 | 目的 | 分区 | 状态 | 说明 |
|---|---|---|---|---|---|
| software_001 | M0 | 真实物理、球障碍坐标、固定目标、分区保护、真实执行重放、恢复去重 | 软件测试 | PASS | 最终源码6/6，26.413秒；初次4/6通过、2失败XML已保存；不当作正式结果 |
| qualified_origin_coverage_001 | M1–M4 | 8新验证先冻结、512独立TRAIN初态、32固定修正尝试、32真实执行审计 | TRAIN＋NEW_VAL | TODO | 尚未启动；总5400秒/2线程/20GiB |
| origin_coverage_training_closure | M5 | 实际消费新旧TRAIN训练、原DEV及新验证三种子同预算对照 | TRAIN/原VAL/DEV/NEW_VAL | PENDING_DATA_GATE | 待数据门通过后另冻结；当前不声称新训练或优势 |

查询与耗时仅辅助，不设相对胜出门槛。长期目标保持进行中，原结果和物理门槛保留，最终TEST未启动。
