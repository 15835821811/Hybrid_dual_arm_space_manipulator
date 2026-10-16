# 独立初态覆盖执行跟踪

| Run ID | 阶段 | 目的 | 分区 | 状态 | 说明 |
|---|---|---|---|---|---|
| software_001 | M0 | 真实物理、球障碍坐标、固定目标、分区保护、真实执行重放、恢复去重 | 软件测试 | PASS | 最终源码6/6，26.413秒；初次4/6通过、2失败XML已保存；不当作正式结果 |
| qualified_origin_coverage_001 | M1–M4 | 8新验证先冻结、512独立TRAIN初态、32固定修正尝试、32真实执行审计 | TRAIN＋NEW_VAL | FINISHED_WITH_BLOCKED_ITEMS | worker201448退出；503/512初态、119/128场景、32/32固定修正；535新TRAIN，总12679，初态43→546；32/32执行重放、336000步骤、原始审计PASS；2219.875秒；原覆盖门FAIL |
| origin_coverage_training_closure | M5 | 实际消费新旧TRAIN训练、原DEV及新验证三种子同预算对照 | TRAIN/原VAL/DEV/NEW_VAL | BLOCKED_DATA_GATE | 9覆盖槽位缺失、39原ERROR保留，旧资格false；需另冻结有界补充及合并资格，再实际训练，不能手改原字段 |

查询与耗时仅辅助，不设相对胜出门槛。长期目标保持进行中，原结果和物理门槛保留，最终TEST未启动。

全部39个integration_state错误已逐例诊断为实际初态球障碍穿透、仅warmstart不同，0耦合/物理步/网络；66 REJECTED均geometry。9个障碍槽位四次耗尽，原协议不能记为512/512或无ERROR通过。后续初始化模块ff_prior_scene_guard先拒绝真实初态穿透，再保留全部187项完整转移检查，3/3真实测试通过；不接入冻结源码，不重分类原ERROR或升级数据。诊断`../../ff_prior_pilot_tools/evidence/origin_contact_diagnostic_002/`，001亦保留；最终[ANALYSIS.md](ANALYSIS.md)。独立评审partial/high，C1仅部分支持、C2本轮未评估，保留合格样本但不解除原训练BLOCKED。
