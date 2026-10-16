# 扩库数据的保存数组与覆盖诊断

数据作业状态：FINISHED / complete；开始检查时进程存活：False。
本次实际核验已提交新数组 2144 条；异常 0；新增物理调用0、训练调用0。
若仍运行，本文件只是跨文件实时快照，不是最终同时刻统计或整轮验收。

验证等级：{'MODEL_VALID_ONLY': 2096, 'EXECUTION_VERIFIED': 48}。不同初态 24、不同世界任务 2048、家族标签 532。
提案形状：{'fixed_world_target_sine_bow': 96, 'smooth_baseline': 408, 'correlated_low_frequency_perturbation': 414, 'continuum_leads': 411, 'rigid_arm_leads': 404, 'joint_group_motion_allocation': 411}；运动倍率：{'not_applicable_fixed_goal': 96, '1.0': 369, '0.75': 375, '1.25': 325, '1.5': 324, '0.5': 331, '1.75': 324}；时长秒：{'10.0': 2144}。
控制点最大偏移rad范围：{'minimum': 0.02774650131455221, 'maximum': 0.4845330092060345}；目标位置误差m范围：{'minimum': 1.7466001503143804e-11, 'maximum': 0.0028590138311408485}。
保存几何列为间距m/最大步距m：{'minimum': [0.000269910048748467, 0.00010720419971010494], 'maximum': [0.008578391871811256, 0.001993250031085818]}。

拒绝原因：{'INDEPENDENT_VALIDATION_REJECTED': 65}；失败门槛：{'geometry': 65}。
仅几何不通过且冻结目标仍通过的拒绝样本：65。origin上下文没有额外添加障碍，不表示没有默认目标卫星等几何。
拒绝上下文：{'obstacle_2': 2, 'obstacle_4': 7, 'obstacle_6': 20, 'obstacle_7': 7, 'obstacle_11': 4, 'obstacle_5': 7, 'obstacle_9': 3, 'obstacle_8': 7, 'obstacle_10': 4, 'obstacle_1': 2, 'obstacle_0': 1, 'obstacle_3': 1}；拒绝运动倍率：{'1.0': 15, '0.75': 9, '1.25': 12, '1.5': 12, '1.75': 12, '0.5': 5}。
拒绝样本穿透深度m范围：{'minimum': 1.1901287718630096e-05, 'maximum': 0.027550750597496326}；几何对：{"('v7_detour', 'collision_0029')": 4, "('v7_detour', 'collision_0063')": 49, "('v7_detour', 'collision_0061')": 10, "('v7_detour', 'collision_0027')": 2}。

每条核验真实控制点/训练视图一致性、完整初态哈希、冻结世界目标、生成与独立准入证据。未执行者只能称MODEL_VALID；离散几何步距与收敛检查不等同于连续时间严格几何证明。
初态/目标/形状计数不证明新障碍场景、统计独立性或不同绕行同伦，也不证明Diffusion学习收益。
具体异常：[]。
