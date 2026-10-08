# V6.4-C.2 当前可视化入口

固定预算的 C.2 数据、真实训练、搜索及最终 actual 协议已执行。独立科学审查确认有限工程交付完成，支持局部先导结果，判定 `claim_supported=partial`、置信度 medium；学习收益保持 `NOT_ESTABLISHED`，默认保留 C.1 规则初始化。媒体已完成刷新并通过清单、资产哈希与保存末状态核验：12 条唯一 actual、84 个视频，覆盖全部 32 个逻辑槽的可用媒体及诚实的 `NO_PLAN` 状态。

[打开 C.2 当前查看器](../v6_4/visualization/preference_warmstart_20261008_01/index.html) · [媒体完成清单](../v6_4/visualization/preference_warmstart_20261008_01/visualization_manifest.json) · [封存报告](../v6_4/releases/preference_warmstart_20261008_01/snapshot/REPORT.md) · [机器结论](../v6_4/releases/preference_warmstart_20261008_01/snapshot/summary.json) · [独立科学审查](../paper/review-traces/experiment-result-to-claim/2026-10-08_run01/response.md) · [审查机器结论](../paper/review-traces/experiment-result-to-claim/2026-10-08_run01/verdict.json) · [GitHub 独立分支](https://github.com/15835821811/Hybrid_dual_arm_space_manipulator/tree/v6.4-c2-preference-diffusion-warmstart) · [C.1 历史查看器](../v6_4/visualization/continuous_route_optimizer_20261007_01/index.html)

真实新模型完成 4,000 optimizer updates，VAL 选中 update 250。正式 TEST 共生成八个指定 Diffusion 初值，七个 raw 合法、一个原样拒绝并占槽。最终 32 个逻辑 actual 槽为 22 个完整 Task 与原五门禁通过、10 个 `NO_PLAN`；12 条唯一执行、10 个严格别名，无 actual 执行失败或工具错误。N8-A 四任务均达到 R12-A 的预声明近质量带，D8-A 为三任务。偏好 B 在各端点均为 2/4 完整通过且达到 30mm，其他任务的 `NO_PLAN` 保留在分母内。与 R12-B 的两个完整任务配对时，D8 为 2/2 达到近质量带，N8 为 1/2，另外两任务均为 N/A；N8 在 `test0_plus` 的路径增量为 5.26878mm，超过 5mm 带宽。D8-B 在这两个完整配对任务中的路线比 N8-B 短，但净空较低且仍达到 30mm；该局部差异不建立整体神经收益。两种八槽初始化均未建立保持整体质量的摊销收益。结果仅代表四个 TEST 任务、两个新母场景与单个训练 seed，不构成广泛泛化或总体非劣保证。

当前查看器的固定路径为 `v6_4/visualization/preference_warmstart_20261008_01/index.html`。它按四个新 TEST Task、R8 / R12 / N8 / D8 端点及 A / B 偏好展示全部 32 个逻辑 actual 槽。规则组的 R8 来自同一 R12 搜索运行在第八槽前缀的封存结果。四槽曲线只表示预测结果。

本轮 12 条唯一 actual 各有七种视频，共 84 个：总览、正面、侧面、俯视、斜视、连续体特写和五视角组合。完成清单绑定全部资产，核验了视频对应的精确保存末状态及 actual 与独立重放状态的逐位一致性。相同 Task 下完整计划与执行身份一致的十个别名槽共用同一组媒体，明确显示其来源，不制造第二次执行。本轮没有 actual 失败前缀；生成工具仍保留失败视频只到实际保存精确末状态、不补齐至 27 s 的规则。十个 `NO_PLAN` 槽保留在全部方法的分母内，执行步数为零，不提供虚构视频。

末端轨迹图分别显示连续体与刚性臂的实际路径。连续体跟踪图同时提供相对冻结生成参考和原 Task/base 路径的误差；刚性臂误差相对保存的当前目标位姿所定义的 Task 抓取位姿计算。位置与姿态误差使用保存的物理时刻，原要求窗口另作标记，并提供原生时间分辨率 CSV。全曲线误差是描述性诊断，不能替代任务窗口与原五项独立门禁。

查看器还展示两张核心图：封存预算前缀的预测质量与覆盖，以及包含初始化/推理的规划成本与最终 actual 结果。冷规划时间以实测区间表示：下界为原内部端点计时，上界为包围 selection 的 worker 调用至返回区间；R8 使用同一 R worker 的启动/收尾残差加第八槽封存计时。它不冒充精确的进程启动到 selection 封存时延。四 worker 共享资源，未做隔离重复时延实验；warm 路径只是去除已测 import/load 的分解估计。所有四个任务的失败和缺测保留；只在具备完整合格证据的子集中显示完整路线质量。八槽比十二槽少四个配额属于协议设置，不自动构成实测学习收益。

媒体只读取已封存 actual 的状态和独立重放保存的运动学结果，并校验初态、时钟、计划身份与状态一致性。渲染仅执行前向运动学，不新增物理积分、净空查询、QP 求解、网络样本或训练更新。候选预测资格、最终反馈 actual、独立重放及可视化分别记账；媒体刷新不产生新的任务或安全认证。

部署状态保持 `NOT_MET`，连续时间安全与硬件安全保持 `NOT_ESTABLISHED`。所有带日期的历史图集、报告、视频和封存清单继续保留原件。

在克隆分支的仓库根目录执行便携检查：

```sh
python -B -X utf8 -m v6_4.visualization.portable_preference_warmstart --release v6_4/releases/preference_warmstart_20261008_01 --action inspect
```

该检查核验发布副本并重建 dataset/scaler/sampler，不执行采样或物理。大型原生状态和力矩日志按发布遗漏账本保留本地；便携检查与字节校验不等于重新完成物理认证。可视化构建入口为 [build_preference_warmstart_media.py](../v6_4/visualization/build_preference_warmstart_media.py)。

发布复核将一个可选 R8 warm 诊断字段改为 null，避免沿用 R12 全流估计；原展示文件和绑定已保留在 `metadata_revision_01` 及发布目录的 prior metadata binding 中，实验报告、cold 区间、模型和全部视频字节不变。查看器的内嵌 JavaScript 语法、全部 32 槽、154 个视频引用、66 个轨迹/误差/CSV 引用及别名/NO_PLAN 关系通过[静态复核](C2_VIEWER_STATIC_CHECKS.json)。Tabbit 浏览器连接未能创建页面，因此真实浏览器点击检查记为不可用，不冒称通过。
