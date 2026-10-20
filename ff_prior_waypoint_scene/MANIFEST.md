# 本任务输出

| 输出 | 内容 |
|---|---|
| docs/EXPERIMENT_PLAN.md | 场景、配对方法、物理门、资源和范围 |
| docs/ANALYSIS.md | 中文实测分析与结果边界 |
| docs/INDEPENDENT_REVIEW.md | 新的独立只读结果审查 |
| docs/GOAL_STATUS.md | 本线程场景目标状态，不覆盖主干训练目标 |
| evidence/ | 原始与修正紧凑结果、源码绑定、旧未跟踪依赖快照、resume不重复核验、浏览器检查、输出文件清单 |
| ../../artifacts/v7_1/waypoint_scene_001/index.html | 四方法完整视频与曲线入口 |
| ../../artifacts/v7_1/waypoint_scene_001/*/*/native_trace.npz | 原始每步状态与力矩，留本地不推送 |
| 线程可视化目录/waypoint-scene.html | 交互骨架、期望/实际轨迹、误差和工作域 |

源码历史保留原始报告误判，独立audit_saved.py按原jnt_limited掩码只读修正。旧tracking_diffusion目录是既有未跟踪用户文件，本任务没有修改或将整个目录提交；确切运行依赖副本保存在evidence/executed_tracking_dependency.zip以供复查。
