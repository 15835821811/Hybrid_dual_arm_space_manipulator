# B.2 新私有轨迹上的原任务跟踪指标

场景 `v6_lite_scenario_01` 共核对 13500 个 2 ms 状态。指标公式和阈值沿用原 A.1 配置。

| 指标 | 新轨迹 | 原阈值 | 通过 |
| --- | ---: | ---: | :---: |
| rigid_final_error_m | 5.72149136e-07 | 0.0001 | True |
| rigid_steady_rmse_m | 6.06129503e-07 | 0.00015 | True |
| continuum_active_path_rmse_m | 7.9064922e-05 | 0.00018 | True |
| rigid_orientation_full_max_deg | 0.0379353561 | 0.25 | True |
| continuum_orientation_full_max_deg | 0.0263527271 | 0.25 | True |

按原任务状态和 4 次插值核对整机净空：0.0149985 m，阈值 0.005 m；continuum-target 最小净空 0.208225 m。
本程序没有完成原 26/11 项合同或在线时限验收。
