# 扩库数据的保存数组与覆盖诊断

数据作业状态：FINISHED / complete；开始检查时进程存活：False。
本次实际核验已提交新数组 9000 条；异常 0；新增物理调用0、训练调用0。
若仍运行，本文件只是跨文件实时快照，不是最终同时刻统计或整轮验收。

验证等级：{'MODEL_VALID_ONLY': 8952, 'EXECUTION_VERIFIED': 48}。不同初态 15、不同世界任务 9000、家族标签 3654。
提案形状：{'smooth_baseline': 1807, 'correlated_low_frequency_perturbation': 1825, 'continuum_leads': 1819, 'rigid_arm_leads': 1795, 'joint_group_motion_allocation': 1754}；运动倍率：{'1.0': 1509, '0.75': 1515, '1.25': 1507, '1.5': 1497, '0.5': 1503, '1.75': 1469}；时长秒：{'10.0': 9000}。
控制点最大偏移rad范围：{'minimum': 0.02039522644284042, 'maximum': 0.5229694365095068}；目标位置误差m范围：{'minimum': 1.3947684642920175e-11, 'maximum': 8.79513806798755e-08}。
保存几何列为间距m/最大步距m：{'minimum': [1.0535792949227865e-05, 5.7602735508698836e-05], 'maximum': [0.008560813041592136, 0.001999721381349647]}。

拒绝原因：{'INDEPENDENT_VALIDATION_REJECTED': 126}；失败门槛：{'geometry': 125, 'geometry_refinement': 1}。
仅几何不通过且冻结目标仍通过的拒绝样本：125。origin上下文没有额外添加障碍，不表示没有默认目标卫星等几何。
拒绝上下文：{'obstacle_1': 65, 'obstacle_2': 44, 'obstacle_0': 12, 'origin_0': 3, 'origin_9': 2}；拒绝运动倍率：{'1.0': 21, '1.5': 33, '1.75': 31, '1.25': 23, '0.75': 15, '0.5': 3}。
拒绝样本穿透深度m范围：{'minimum': 0.0003010380654720566, 'maximum': 0.05162173910910184}；几何对：{"('v7_detour', 'collision_0063')": 88, "('v7_detour', 'collision_0061')": 21, "('v7_detour', 'collision_0029')": 12, "('collision_0070', 'target_satellite_collision')": 2, "('collision_0072', 'target_satellite_collision')": 2, "('collision_0070', 'collision_0072')": 1}。

每条核验真实控制点/训练视图一致性、完整初态哈希、冻结世界目标、生成与独立准入证据。未执行者只能称MODEL_VALID；离散几何步距与收敛检查不等同于连续时间严格几何证明。
初态/目标/形状计数不证明新障碍场景、统计独立性或不同绕行同伦，也不证明Diffusion学习收益。
具体异常：[]。
