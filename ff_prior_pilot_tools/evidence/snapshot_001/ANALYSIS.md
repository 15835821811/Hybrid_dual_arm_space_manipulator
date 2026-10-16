# 扩库数据的保存数组与覆盖诊断

数据作业状态：RUNNING / pilot_forward_generation；开始检查时进程存活：True。
本次实际核验已提交新数组 3079 条；异常 0；新增物理调用0、训练调用0。
若仍运行，本文件只是跨文件实时快照，不是最终同时刻统计或整轮验收。

验证等级：{'MODEL_VALID_ONLY': 3079}。不同初态 15、不同世界任务 3079、家族标签 1254。
提案形状：{'smooth_baseline': 619, 'correlated_low_frequency_perturbation': 626, 'continuum_leads': 624, 'rigid_arm_leads': 610, 'joint_group_motion_allocation': 600}；运动倍率：{'1.0': 534, '0.75': 533, '1.25': 505, '1.5': 498, '0.5': 510, '1.75': 499}；时长秒：{'10.0': 3079}。
控制点最大偏移rad范围：{'minimum': 0.028324663226634295, 'maximum': 0.5147486427639714}；目标位置误差m范围：{'minimum': 1.3947684642920175e-11, 'maximum': 8.441304858561605e-08}。
保存几何列为间距m/最大步距m：{'minimum': [0.00013907554631501125, 5.7602735508698836e-05], 'maximum': [0.008560813041592136, 0.001979010453861155]}。

拒绝原因：{'INDEPENDENT_VALIDATION_REJECTED': 46}；失败门槛：{'geometry': 46}。
拒绝上下文：{'obstacle_1': 24, 'obstacle_2': 15, 'obstacle_0': 5, 'origin_0': 2}；拒绝运动倍率：{'1.0': 6, '1.5': 12, '1.75': 11, '1.25': 10, '0.75': 7}。
拒绝样本穿透深度m范围：{'minimum': 0.0005635445268147455, 'maximum': 0.027383818427688576}；几何对：{"('v7_detour', 'collision_0063')": 30, "('v7_detour', 'collision_0061')": 9, "('v7_detour', 'collision_0029')": 5, "('collision_0070', 'target_satellite_collision')": 1, "('collision_0072', 'target_satellite_collision')": 1}。

每条核验真实控制点/训练视图一致性、完整初态哈希、冻结世界目标、生成与独立准入证据。未执行者只能称MODEL_VALID；离散几何步距与收敛检查不等同于连续时间严格几何证明。
初态/目标/形状计数不证明新障碍场景、统计独立性或不同绕行同伦，也不证明Diffusion学习收益。
具体异常：[]。
