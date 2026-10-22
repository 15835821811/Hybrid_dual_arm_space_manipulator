# 三障碍单次Diffusion完整运行与可视化

2026-10-22。运行`artifacts/v7_1/diffusion_demo_001`已完成，后台总耗时366.5秒（含规划、执行、重放、图表与两段视频）。这一次**路径生成、全机避障、原生67通道力矩执行、完整状态独立重放均通过**。仅证明这一原三障碍开发场景的完整系统运行，未把它替代五障碍与动态连续组合；本轮未做优势对照。

## 实际结果

|环节|真实结果|范围|
|---|---|---|
|模型与采样|旧已验证模型实际加载；5×16=80次网络调用，完整323维表示|65维条件没有障碍条件，无新训练|
|路径资格|5条提案中预选2条修正；2条原门MODEL_VALID|未选3条仅筛选，不能写成完整验证失败|
|选定路径|候选04；依据冻结最低合格位姿分数选定|执行失败不补选，本次一次即通过|
|真实执行|13750个原生MuJoCo步，27秒运动＋0.5秒保持|67通道力矩驱动，无实际状态裁剪/写回|
|独立重放|另建状态，重放13750步；836维INTEGRATION最大差0|不逐步复制实际状态|
|物理资格|15项原门全部通过；EXECUTION_VERIFIED|未执行候选02仅MODEL_VALID|
|可视化|两视角MP4、交互HTML、7组PNG/PDF、逐帧CSV已生成|只读实际保存状态，额外物理步0|

保持段双臂最大位置误差为**0.0546055 / 0.00497906 mm**，最大姿态误差为**0.006549 / 0.0002479°**（连续体／刚性臂）。终点位置误差为0.0544849 / 0.00493095 mm。接近过程尚未抵达终点时有较大世界目标误差，静态位姿门按原协议在保持段判断，未放宽门。

全机最小间隙**1.543031 mm**；碰撞检查27501个状态，测得最大几何步长0.113786 mm，小于原2 mm。保留“仅不穿透”与原碰撞对策略，不是连续时间无碰撞数学证明。原始力矩/限值最大比值**0.00492407**，饱和比例0；17维参考跟踪最大误差1.49137e-5 rad，17→67流形最大误差6.25450e-9 rad。自由基座最大平移5.78523 mm，原动量门通过，未固定基座。

本轮3932次实际耦合调用、27500原生含重放步；2304124为账本模型积分预留数，不冒充逐步独立实测。上限12000耦合/27500原生步/2小时未超出。无新训练、扩库、TEST或比较方法。

## 机制与结论

实际链路是：Diffusion提案→公开初始化与prepare→完整323维世界目标/全机避障修正→原门资格→浮动基座前馈＋力矩PD→独立重放。5条prepare提案都未直接通过位姿和非穿透筛选；候选02与04各经6次共同修正迭代后合格。因此，**Diffusion参与的完整系统能工作**；本结果不把修正器或控制器的贡献归给裸网络，也不评判相对优势。用户已要求推进工程，本轮成功足以接下一段。

## 直接查看

- [交互三维＋全部视频与曲线](../../artifacts/v7_1/diffusion_demo_001/visuals/index.html)
- [实体主视角视频](../../artifacts/v7_1/diffusion_demo_001/visuals/overview.mp4) · [实体俯视视频](../../artifacts/v7_1/diffusion_demo_001/visuals/top.mp4)
- [误差、间隙、基座、力矩、动量](../../artifacts/v7_1/diffusion_demo_001/visuals/01_summary.png)
- [17维运动](../../artifacts/v7_1/diffusion_demo_001/visuals/02_joint_motion.png) · [67力矩与跟踪](../../artifacts/v7_1/diffusion_demo_001/visuals/03_torque_tracking.png)
- [基座与流形](../../artifacts/v7_1/diffusion_demo_001/visuals/04_base_manifold.png) · [分阶段世界路径](../../artifacts/v7_1/diffusion_demo_001/visuals/05_world_paths.png)
- [独立重放](../../artifacts/v7_1/diffusion_demo_001/visuals/06_independent_replay.png) · [世界坐标](../../artifacts/v7_1/diffusion_demo_001/visuals/07_world_coordinates.png)

交互视图是来自模型body位置的骨架示意，实体几何见MuJoCo视频。每段视频400个实际下采样帧，15fps播放26.667秒，对应27.5秒仿真；不是实时速度证书。CSV保存全部13751个时刻。图像/视频24个文件的来源与SHA256见visuals/manifest.json。

## 可复现性与后续

生产提交8456203e加启动时冻结的176个源码哈希与executed_source.zip，87输入绑定。工作区已有用户修改`v6_lite/hierarchical_qp.py`也如实归档，未改动或提交它；不能说纯净Git树。该旧QP不负责本轮floating_ff_torque_pd_v1执行。原始账本审计PASS；完成resume仅核一次，191文件清单/大小/哈希完全不变。新独立审查保存于paper/review-traces/experiment-result-to-claim/20261022_diffusion_demo：本次静态C1 yes/high，68项必要证据检查通过；C2 NOT_EVALUATED，长期目标未完成。未有paper claim audit，论文主张仍provisional，但不阻塞已授权工程推进。

继续D2：从本次绝对时间27.5秒的完整836实际末态接原七不规则航点，保留三障碍与控制命令，不从默认初态重开。前驱卫星速度为零，不能通过突然赋速度冒充真实连续漂移；本段刚性臂保持原预抓捕位姿、连续体跟踪动态航点。真正移动卫星与原五障碍仍单列缺口。不再增加优势对照，不扩库、不启动TEST。

终端：`.\ff_prior_diffusion_demo\job.ps1 status|report`；当前阶段统一查看`.\ff_prior_fullchain\job.ps1 status|next|report`。
