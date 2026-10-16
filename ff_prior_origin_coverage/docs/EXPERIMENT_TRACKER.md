# 独立初态覆盖执行跟踪

| Run ID | 阶段 | 目的 | 分区 | 状态 | 说明 |
|---|---|---|---|---|---|
| software_001 | M0 | 真实物理、球障碍坐标、固定目标、分区保护、真实执行重放、恢复去重 | 软件测试 | PASS | 最终源码6/6，26.413秒；初次4/6通过、2失败XML已保存；不当作正式结果 |
| qualified_origin_coverage_001 | M1–M4 | 8新验证先冻结、512独立TRAIN初态、32固定修正尝试、32真实执行审计 | TRAIN＋NEW_VAL | RUNNING | 已实际启动worker201448，源码b133c0b5；8新验证已冻结，最近完整快照315个新初态轨迹准入；总5400秒/2线程/20GiB；门未通过 |
| origin_coverage_training_closure | M5 | 实际消费新旧TRAIN训练、原DEV及新验证三种子同预算对照 | TRAIN/原VAL/DEV/NEW_VAL | PENDING_DATA_GATE | 待数据门通过后另冻结；当前不声称新训练或优势 |

查询与耗时仅辅助，不设相对胜出门槛。长期目标保持进行中，原结果和物理门槛保留，最终TEST未启动。

运行中诊断：最早6个integration_state错误来自实际初态球障碍穿透、只有warmstart不同，0耦合/物理步/网络。当前已有障碍槽位四次尝试耗尽，原协议不能记为512/512或无ERROR通过。后续初始化模块ff_prior_scene_guard先拒绝真实初态穿透，再保留全部187项完整状态转移检查，3/3真实测试通过；不接入当前冻结源码，不重分类原ERROR或升级数据。诊断`../../ff_prior_pilot_tools/evidence/origin_contact_diagnostic_001/`；当前执行审计仍待实际完成。
