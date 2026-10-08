# C.1 保存状态媒体补充（2026-10-08）

本次按用户追加要求，同步更新五视角、连续体侧视、双臂末端轨迹、位置/姿态跟踪误差、基座漂移、净空和控制诊断。入口为[媒体目录](../v6_4/visualization/continuous_route_media_20261008_01/README.md)、[HTML目录](../v6_4/visualization/continuous_route_media_20261008_01/index.html)和[结果总览](../v6_4/visualization/continuous_route_optimizer_20261007_01/index.html)。

16个逻辑方法槽对应12条独立actual轨迹（10条完整27秒，2条拒绝前缀）、3个别名和1个NO_PLAN。每条独立轨迹提供front、side、top、iso、overview五个640×480视频、1920×960合成视频及960×720连续体聚焦视频，共84视频；每条另提供轨迹、跟踪误差、净空/基座、控制诊断4图，共48图。别名页面明确指向同一证据资产，不伪装成新运行。NO_PLAN页面没有视频或误差曲线。

视频读取原始actual trace的initial_qpos/qvel、actual_full_qpos/qvel和time。全部12条与原独立力矩重放的全状态逐位一致，时钟偏差小于1e-9秒后，才复用其保存的末端运动学。模型运行合同、源资产、碰撞对策略及编译模型标识也与封存证据一致。摄像机及编码复用原有媒体工具，历史媒体代码/资产未改动。

呈现帧率15FPS，选择呈现时刻之后的第一个保存状态，包含准确保存终点，不插值、不外推。Z0/G0失败视频分别止于11.08/10.80秒，不补齐27秒；它们仍是原来的RAMP_MICROSTATE_OUTSIDE_DECLARED_DOMAIN拒绝。编码时长包括终点帧，可略长于物理保存区间，完整帧索引/时钟在replay_metadata.json中。

误差图使用实际消耗的20毫秒QP输入，与同一pre-step状态的运动学对齐。完整轨迹1350点，失败前缀554/540点；拒绝调用额外保存但未消耗的参考行被排除。位置误差为欧氏范数，姿态误差为SO(3)主角，单位分别mm/deg。全执行区间RMS包括刚性臂初始接近过程；失败前缀统计不充当完整任务指标。轨迹与原始Task路线偏差另列，绕行量不等同Task失败。

紫色阴影表示原冻结W_support并集；失败尾部用灰色斜线明确表示未执行。相关球净空只绘制原保存W_support样本，窗口间不连线；与robot-target原生净空保留不同对集/适用范围。机器人目标5mm硬门禁和C.1的30mm质量偏好分别标注。CSV保留实际时间、状态索引和数值，支持核对曲线。

渲染仅调用mj_forward；上下文禁止mj_step、mj_step1、mj_step2、mj_geomDistance和已加载OSQP.solve。新增物理步、距离查询、QP求解、模型采样和训练均为0。可视化不是新安全评估，部署仍为NOT_MET。原始1208项manifest逐文件校验保持通过，未更改封存实验结果或运行时源码黄金哈希。

验证包括84视频ffprobe的尺寸、FPS、帧数和时长、文件SHA256、12条全状态一致性、每条准确终点、16槽覆盖与别名绑定、48图及36CSV的存在和来源核对。代表性的完整/失败视频预览及诊断图通过静态图像检查。浏览器实时预览未验证；HTML使用原生video controls和普通相对链接，GitHub请使用Markdown媒体目录或下载后打开HTML。

从完整本地封存实验复现媒体（clone不包含大型原始NPZ；遗漏清单沿用原release_manifest.json）：

```powershell
python -B -X utf8 -m v6_4.visualization.continuous_route_media --source v6_4/output/continuous_route_optimizer_20261007_01 --output v6_4/visualization/continuous_route_media_20261008_01 --workers 2
python -B -X utf8 -m v6_4.visualization.finalize_continuous_route_media --source v6_4/output/continuous_route_optimizer_20261007_01 --output v6_4/visualization/continuous_route_media_20261008_01
```

collection.json包含全部来源/状态一致性/成本记录；media_manifest.json是本次派生资产清单。原release的raw_manifest.json和snapshot继续保留原实验封存语义；publication_manifest.json仅刷新本次明确追加的发布文件哈希，不重写原实验manifest。
