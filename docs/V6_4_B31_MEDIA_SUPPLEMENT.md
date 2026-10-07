# V6.4-B.3.1 保存状态媒体补充

本补充承接已发布研究提交 `e67f8cfee583fb3d42fad69b8372fbb8dd0e8d16`，覆盖当前固定开发集的全部四任务 × 七参考 = 28 槽。它刷新媒体展示，不新增闭环实验，不修改原研究判断。

[完整交互目录](../v6_4/visualization/execution_aware_media_20261007_01/index.html) · [GitHub 可读目录与下载](../v6_4/visualization/execution_aware_media_20261007_01/README.md) · [视频来源](../v6_4/visualization/execution_aware_media_20261007_01/videos.json) · [图表、数值与来源](../v6_4/visualization/execution_aware_media_20261007_01/figures/plots_final.json) · [媒体清单](../v6_4/visualization/execution_aware_media_20261007_01/media_manifest.json) · [本地验证](../v6_4/visualization/execution_aware_media_20261007_01/validation.json)

每槽提供总览、正面、侧面、俯视、等轴、五视角组合、专用连续体侧视，共 **196 个完整视频**。常规单视角为 640×480，组合为 1920×960，连续体侧视为 960×720，15 fps。原任务均完整 27 s；406 个呈现帧包含 t=0 和精确保存末状态，编码片长约 27.067 s。连续体侧视使用独立相机，刚性臂半透明；不是组合视频的裁切。

默认海报另从视频中精确提取冻结路线窗口中点附近的第 153 帧（呈现时间 10.2 s），方便查看局部路线作用。28 组各保留组合与侧视海报，共 56 张；原视频中点海报与其原哈希不修改。[路线海报来源与帧号](../v6_4/visualization/execution_aware_media_20261007_01/route_posters/poster_manifest.json)分别记录源 MP4、实际保存状态时刻和 PNG 哈希；海报提取只解码视频，不调用物理或前向计算。

图表包括双臂末端轨迹、位置和姿态误差、基座平移与旋转漂移、原查询范围内保存的净空诊断，以及保存的命令、67 通道力矩与 PCC/区间诊断。每槽同时提供 PNG、PDF 和数值 CSV，另提供每任务七参考的完整比较。冻结路线区间与任务锚点保持原定义。状态 CSV 每 20 ms 选择原保存状态；全状态统计仍使用 13,501 个原 2 ms 状态。六条旧原件未保存的 QP 向量保持缺测，不重建。

视频直接读取原实际执行的初态与完整 qpos/qvel/time；独立同力矩重放 fresh 数组仅在逐槽全状态精确一致、时钟浮点误差检查通过后，作为已有运动学的来源。EA 比较槽位、六条旧 PILOT 原来源、22 条新增执行来源及 SHA 分开记录。绿色为按保存时钟求值的解析残差参考，红色为独立 Task/base 曲线，青色为实际末端轨迹。

执行跟踪误差使用原控制器真正消费的参考和对应控制边界；相对独立 Task/base 曲线的偏差另列。任务允许锚点之间合法绕行，因此 base 曲线偏差不能替代原 Task 锚点、朝向、终点与安全验收。原 Task 与门禁通过 **28/28**；无训练、模型采样或独立泛化 TEST，部署状态保持 `NOT_MET`，未建立硬实时或连续时间安全保证。

渲染仅选择保存状态并调用 `mj_forward`，不积分、不续接、不插值，不重新运行控制器、QP、独立力矩重放或几何安全验收。新增物理步、QP 求解、训练更新和模型采样均为零；渲染和编码成本另记。

原 `execution_aware_route_teacher_20261007_01` 的 OUT、两图 VIS、portable RELEASE、run03 审查及各清单保持原字节。旧提交和收据证明原发布时点；当前 README、说明与入口在此后增加媒体链接。原本机全量封存的 external ledger 包含这些当时可变的文档路径，因此不能把文档更新后的当前路径重新宣称为原封存时点的完整本机外部路径校验通过。原精确冻结副本与 portable 校验保持有效，新媒体使用独立清单、验证和远端提交收据。

当前入口在 `v6_4/visualization/index.html` 与 `v6_lite/visualization/index.html`。原[两张研究图、候选表与有限教师](../v6_4/visualization/execution_aware_route_teacher_20261007_01/index.html)以及[原报告](../v6_4/releases/execution_aware_route_teacher_20261007_01/report.md)继续可访问。GitHub 的 Markdown 目录提供图片和 MP4 直接链接；交互 HTML 使用嵌入数据，无外部库或在线 API。
