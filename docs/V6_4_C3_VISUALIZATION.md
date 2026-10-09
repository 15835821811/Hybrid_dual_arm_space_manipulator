# V6.4-C.3 当前实际执行可视化

入口：[C.3 查看器](../v6_4/visualization/search_aware_warmstart_20261008_01/index.html)。下载或 clone 分支后本地打开 HTML；GitHub 的源码查看页不会直接运行交互式播放器。也可从仓库根目录运行 `python -m http.server 8766 --bind 127.0.0.1`，再打开对应本地页面。

查看器覆盖四个冻结 TEST Task、R8/R12/N8/S8/D8、A/B，共 40 个逻辑槽。28 次独立实际执行对应 196 个 MP4：overview/front/side/top/iso、五视角拼图、连续体侧向近景。6 个严格别名复用原实际媒体；6 个 NO_PLAN 槽没有视频或轨迹图。每个独立实际执行附双臂末端轨迹、位置/姿态误差图及原生 2ms CSV。全部媒体已渲染并完成最终 QA。

所有画面读取保存的 actual qpos/qvel，以 `mj_forward` 求显示姿态。没有新的 `mj_step`、QP、安全几何查询、模型样本或训练更新；不是播放预测力矩，也不是补做实验。完整轨迹保留 27s 精确末状态。15fps 编码包含起点和末点，编码时长约 27.067s，与原物理时长分开说明。

绿色是冻结生成参考，红色为 Task/base 路径，青色为实际轨迹，RGB 表示 XYZ。连续体侧视淡化刚性臂以观察连续体弯曲及局部交互；五视角的第六格留空。邻近 waypoint 的空间标签可能重叠，应结合轨迹图查看坐标，不由视频估计净空。原独立五门禁决定研究验收。

位置误差区分连续体相对生成参考、连续体相对 Task/base、刚性臂相对当前抓取目标。图中单位 mm/degree，CSV 保留 m/rad；任务时间窗以阴影显示。不同初值的偏离、瞬态峰值和缺测原样保留。

两张核心结果图复制自[封存报告](../v6_4/releases/search_aware_warmstart_20261008_01/snapshot/REPORT.md)：预算—覆盖/近质量，以及 actual 质量—冷规划成本。4 槽曲线是名义预测；只对实际执行端点显示 actual，R8 冷请求总成本缺测，不冒用 R12 成本。A/B 共享搜索，成本不重复相加。

结论保持：D8 未建立优于 R/N/S 的收益，默认 C.1。全部状态为研究仿真；DATA_LIMITED、deployment NOT_MET、连续时间与硬件安全 NOT_ESTABLISHED。历史 C.1/C.2/B 查看器与原始资产均保留。

媒体与 portable release 独立发布。精简 release 缺少明确列出的原证书/计时档案，不能承诺仅凭 `snapshot/` 重建全部媒体；已完成的 MP4/PNG/CSV/HTML 可整体迁移使用。所有可用字节由各自 manifest 绑定。

核验回执保留于 [delivery 目录](audit_receipts/c3_delivery/copy_manifest.json)：[CSV 与媒体字节](audit_receipts/c3_delivery/media_numeric_01.json)、[196 视频首/中/末帧和 84 预览视觉检查](audit_receipts/c3_delivery/media_visual_review_01.json)、[56 张轨迹/误差图](audit_receipts/c3_delivery/figures_review_01/visual_review.json)、[280 个浏览器选择组合](audit_receipts/c3_delivery/browser_ui_01.json)、[迁移后的实际播放和 CSV 下载](audit_receipts/c3_delivery/browser_relocation_01.json)。视频机器解码覆盖全部帧；视觉检查抽查每个视频的首、中、末帧，不声称逐帧人工连续观看。原 machine-only 和 in-progress 回执保留当时状态，由后续独立回执补足。

最终任务书逐项核对见 [C3_FINAL_COMPLETION_AUDIT.md](C3_FINAL_COMPLETION_AUDIT.md)。媒体 manifest SHA-256：`10f9174dc8419ea8561e547e67bb02de45dd1120598730ae076de5ff5bf2bbae`。
