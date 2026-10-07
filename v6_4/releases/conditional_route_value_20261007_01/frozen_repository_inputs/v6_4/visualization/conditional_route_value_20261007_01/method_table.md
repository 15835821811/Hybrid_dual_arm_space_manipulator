# V6.4-B.3 方法表

以下为固定开发 pilot 的六个实际槽位；净空单位为 m，干预单位为 rad/s。空值未填入完整质量比较。

| slot_id | task_id | method | status | saved_horizon_s | full_task_success | I_route_rad_s | I_full_rad_s | continuum_path_length_m | route_clearance_m | full_clearance_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PILOT_00 | b3_mother_00_c_plus | z0 | TASK_COMPLETED | 27.0 | True | 0.10860163231078468 | 0.09460137190674829 | 1.1102763433179839 | 0.025884292882607234 | 0.025884292882607234 |
| PILOT_01 | b3_mother_00_c_plus | z+ | TASK_COMPLETED | 27.0 | True | 0.11196314201535783 | 0.09514228383243532 | 1.1177147967767762 | 0.025777815320296507 | 0.025777815320296507 |
| PILOT_02 | b3_mother_00_c_plus | z- | TASK_COMPLETED | 27.0 | True | 0.1055704751670103 | 0.09407614046917046 | 1.105529998169196 | 0.02586127943175947 | 0.02586127943175947 |
| PILOT_03 | b3_mother_00_c_minus | z0 | TASK_COMPLETED | 27.0 | True | 0.11206975487291264 | 0.09510896574166713 | 1.1102911148904617 | 0.025877743096047352 | 0.025877743096047352 |
| PILOT_04 | b3_mother_00_c_minus | z+ | TASK_COMPLETED | 27.0 | True | 0.1077965504580734 | 0.09441994548253373 | 1.1055413513407002 | 0.02590019716891505 | 0.02590019716891505 |
| PILOT_05 | b3_mother_00_c_minus | z- | TASK_COMPLETED | 27.0 | True | 0.11433806112317475 | 0.09541743713007009 | 1.1177397385971588 | 0.025766457256907287 | 0.025766457256907287 |

## 正式新 TEST

| 方法 | 状态 | 完整成功数 |
|---|---|---|
| Z0 | NOT_RUN_PILOT_STOP | 未评价 |
| R0 | NOT_RUN_PILOT_STOP | 未评价 |
| U0 | NOT_RUN_PILOT_STOP | 未评价 |
| D_true | NOT_RUN_PILOT_STOP | 未评价 |
| D_swap | NOT_RUN_PILOT_STOP | 未评价 |

预声明 pilot 停止后，teacher、训练、新模型采样及正式 TEST 均未运行。NOT_RUN 不等于 0/4 失败。
