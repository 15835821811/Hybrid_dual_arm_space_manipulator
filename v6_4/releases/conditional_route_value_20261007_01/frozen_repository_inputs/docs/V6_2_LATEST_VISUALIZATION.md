> **历史归档（V6.2）**：本页及其35视频保留原始结果，不是 B.3 新实验。当前分支为 [V6.4-B.3 两图与方法表](V6_4_B3_VISUALIZATION.md)：pilot 6/6 完整安全通过，路线干预改善 2.79% / 3.81% 未达预声明 10% 门槛，新训练与五组正式 TEST 均为 `NOT_RUN_PILOT_STOP`，部署 `NOT_MET`。[B.2 全套历史可视化](V6_4_B2_VISUALIZATION.md)独立保留。

# V6.2 最新结果全套可视化

本页是 V6.2 历史展示入口，覆盖其名义研究五场景、保守性与形状诊断、2 倍速度全部失败/成功尝试和 15 组惯量敏感性。历史 B.2/V6.1 图与原始失败仍在原目录保留。

名义验收 **PASSED（25/25 功能、11/11 合同、6,755 边界与 12,350 区间行重算）**；压力实验 **FAILED / INCOMPLETE**；惯量为 **DIAGNOSTIC COMPLETE**；墙钟部署 **NOT_MET**。模拟周期仍为 20 ms / 2 ms，墙钟统计单列性能。

[完整清单](../v6_lite/visualization/latest/visualization_manifest.json) · [独立校验](../v6_lite/visualization/latest/visualization_validation.json) · [交互式本地总览](../v6_lite/visualization/latest/index.html) · [结构化结果](../v6_lite/visualization/latest/result_data.json)

GitHub 页面可直接查看下方 PNG/GIF；MP4 点击打开或下载。交互式 HTML 在本地浏览器打开或由仓库内的本地 HTTP 服务查看，不依赖外部网站。

## 名义五场景图、计时与执行

### 五场景六面板跟踪误差与基座漂移

![五场景六面板跟踪误差与基座漂移](../v6_lite/visualization/latest/error_curves.png)

最新名义非实时研究；保存数据的有限仿真观察。

### 五场景离散间隙

![五场景离散间隙](../v6_lite/visualization/latest/safety_clearance_summary.png)

最新名义非实时研究；保存数据的有限仿真观察。

### QP 候选、实际命令与斜坡合同

![QP 候选、实际命令与斜坡合同](../v6_lite/visualization/latest/execution_contract.png)

最新名义非实时研究；保存数据的有限仿真观察。

### 17 维实际命令与 67 路执行力矩

![17 维实际命令与 67 路执行力矩](../v6_lite/visualization/latest/actual_actions.png)

最新名义非实时研究；保存数据的有限仿真观察。

### 全部规划、发布及力矩计时

![全部规划、发布及力矩计时](../v6_lite/visualization/latest/full_control_timing.png)

最新名义非实时研究；保存数据的有限仿真观察。

### 基座平移与姿态漂移动画

![基座平移与姿态漂移动画](../v6_lite/visualization/latest/base_pose_drift.gif)

最新名义非实时研究；保存数据的有限仿真观察。

## 五场景完整视频与路径

视频从完整保存状态渲染，不重新积分。每场 5 个视角、1 个组合、1 个连续体单侧，共 35 个视频；初始与终态包含在 811 帧中。末端坐标系、七个航点与基座漂移逐帧展示。

### v6_lite_scenario_00

![v6_lite_scenario_00 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_five_view_preview.png)

![v6_lite_scenario_00 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_continuum_focus_preview.png)

[overview](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_overview.mp4) · [front](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_front.mp4) · [side](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_side.mp4) · [top](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_top.mp4) · [iso](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_iso.mp4) · [five_view_grid](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_five_view_grid.mp4) · [continuum_focus](../v6_lite/visualization/latest/videos/v6_lite_scenario_00_continuum_focus.mp4)

![v6 lite scenario 00 tracking paths 3d](../v6_lite/visualization/latest/v6_lite_scenario_00_tracking_paths_3d.png)

![distance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_00/plots/distance_comparison.png)

![gradient comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_00/plots/gradient_comparison.png)

![minimum clearance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_00/plots/minimum_clearance_comparison.png)

### v6_lite_scenario_01

![v6_lite_scenario_01 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_five_view_preview.png)

![v6_lite_scenario_01 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_continuum_focus_preview.png)

[overview](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_overview.mp4) · [front](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_front.mp4) · [side](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_side.mp4) · [top](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_top.mp4) · [iso](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_iso.mp4) · [five_view_grid](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_five_view_grid.mp4) · [continuum_focus](../v6_lite/visualization/latest/videos/v6_lite_scenario_01_continuum_focus.mp4)

![v6 lite scenario 01 tracking paths 3d](../v6_lite/visualization/latest/v6_lite_scenario_01_tracking_paths_3d.png)

![distance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_01/plots/distance_comparison.png)

![gradient comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_01/plots/gradient_comparison.png)

![minimum clearance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_01/plots/minimum_clearance_comparison.png)

### v6_lite_scenario_02

![v6_lite_scenario_02 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_five_view_preview.png)

![v6_lite_scenario_02 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_continuum_focus_preview.png)

[overview](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_overview.mp4) · [front](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_front.mp4) · [side](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_side.mp4) · [top](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_top.mp4) · [iso](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_iso.mp4) · [five_view_grid](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_five_view_grid.mp4) · [continuum_focus](../v6_lite/visualization/latest/videos/v6_lite_scenario_02_continuum_focus.mp4)

![v6 lite scenario 02 tracking paths 3d](../v6_lite/visualization/latest/v6_lite_scenario_02_tracking_paths_3d.png)

![distance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_02/plots/distance_comparison.png)

![gradient comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_02/plots/gradient_comparison.png)

![minimum clearance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_02/plots/minimum_clearance_comparison.png)

### v6_lite_scenario_03

![v6_lite_scenario_03 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_five_view_preview.png)

![v6_lite_scenario_03 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_continuum_focus_preview.png)

[overview](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_overview.mp4) · [front](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_front.mp4) · [side](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_side.mp4) · [top](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_top.mp4) · [iso](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_iso.mp4) · [five_view_grid](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_five_view_grid.mp4) · [continuum_focus](../v6_lite/visualization/latest/videos/v6_lite_scenario_03_continuum_focus.mp4)

![v6 lite scenario 03 tracking paths 3d](../v6_lite/visualization/latest/v6_lite_scenario_03_tracking_paths_3d.png)

![distance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_03/plots/distance_comparison.png)

![gradient comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_03/plots/gradient_comparison.png)

![minimum clearance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_03/plots/minimum_clearance_comparison.png)

### v6_lite_scenario_04

![v6_lite_scenario_04 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_five_view_preview.png)

![v6_lite_scenario_04 预览](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_continuum_focus_preview.png)

[overview](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_overview.mp4) · [front](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_front.mp4) · [side](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_side.mp4) · [top](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_top.mp4) · [iso](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_iso.mp4) · [five_view_grid](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_five_view_grid.mp4) · [continuum_focus](../v6_lite/visualization/latest/videos/v6_lite_scenario_04_continuum_focus.mp4)

![v6 lite scenario 04 tracking paths 3d](../v6_lite/visualization/latest/v6_lite_scenario_04_tracking_paths_3d.png)

![distance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_04/plots/distance_comparison.png)

![gradient comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_04/plots/gradient_comparison.png)

![minimum clearance comparison](../v6_lite/visualization/latest/pcc_monitor/v6_lite_scenario_04/plots/minimum_clearance_comparison.png)

## 区间代理保守性

### Conservatism: all 1,024 frozen configurations

![Conservatism: all 1,024 frozen configurations](../v6_lite/visualization/latest/studies/conservatism_population.png)

Source 2d76555 | B1 budget 31, fixed 5 mm gate and calibrated radii. Distal continuum geometry to target OBB only.
190 = 172 proxy BELOW + 18 UNKNOWN; false-safe 0. Offline cases are not closed-loop task failures.

### Finite budget diagnosis: 18 UNKNOWN cases, seven arms

下表展开图中的预算标签。每行只对应原先 18 个 UNKNOWN 案例；更大预算不代表重新测试全部 1,024 例。

| 查询系列 | 点预算 | 叶预算 | SAFE | BELOW | UNKNOWN | 查询 p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1_distance_bounds | 255 | — | 13 | 0 | 5 | 21.484 |
| B1_distance_bounds | 510 | — | 13 | 0 | 5 | 21.690 |
| B1_distance_bounds | 1020 | — | 13 | 0 | 5 | 21.636 |
| BatchedDistanceDecisionQuery | 255 | 128 | 18 | 0 | 0 | 4.632 |
| BatchedDistanceDecisionQuery | 255 | 512 | 18 | 0 | 0 | 4.774 |
| BatchedDistanceDecisionQuery | 510 | 512 | 18 | 0 | 0 | 4.599 |
| BatchedDistanceDecisionQuery | 1020 | 512 | 18 | 0 | 0 | 4.626 |

![Finite budget diagnosis: 18 UNKNOWN cases, seven arms](../v6_lite/visualization/latest/studies/conservatism_subset_budgets.png)

Source 2d76555 | B1: 13 SAFE + 5 UNKNOWN, max 155 points, tolerance stop. Decision 255/128: 18 SAFE, max 133 points / 69 leaves.
Online decision over all 1,024 cases was NOT_RUN. Subset timing is not population p95; early-stop gap is not an accuracy curve.

## 形状与包络诊断

### Shape and envelope diagnosis: all 172 selected configurations

![Shape and envelope diagnosis: all 172 selected configurations](../v6_lite/visualization/latest/studies/shape_decomposition.png)

Source 32c3b04 | Fixed original radii and 5 mm gate; 25 same-R classification flips are static cases, not closed-loop successes.
Segment/capsule negative unsafe indicators are not exact penetration depths. Envelope support is finite and state-local, not a global bound.

### Saved geometric witnesses: two declared illustrations

![Saved geometric witnesses: two declared illustrations](../v6_lite/visualization/latest/studies/shape_saved_witnesses.png)

Source 32c3b04 | All-case statistics appear separately. Left: minimum envelope margin; right: the sole capsule BELOW case (index 791).
Lines show saved discrete module centerlines; colored points show saved witnesses. No reconstructed surface or exact penetrating depth is claimed.

## 2 倍目标速度压力实验

### Target translation speed x2: FAILED / INCOMPLETE

![Target translation speed x2: FAILED / INCOMPLETE](../v6_lite/visualization/latest/studies/velocity_completion_and_constraints.png)

Source 9c7120e | All five declared scenes retained. Full cohort 25/11/interval and timing verdict: NOT_RUN.
Scalar certificates keep both original 1e-4 rad/s tolerances; conclusions concern the three recorded states, not all possible x2 trajectories.

### All five speed-stress histories, including the refused trials

![All five speed-stress histories, including the refused trials](../v6_lite/visualization/latest/studies/velocity_all_five_error_histories.png)

Sources x1 9cd1831 / x2 9c7120e | Curves use each producer's saved post-step cache errors. Gray tail is unexecuted, not interpolated.
Pressure task logs include a final rejected planning attempt; it is not an executed servo step. Display curves decimated; no full-cohort pass is inferred.

## 惯量模型与执行敏感性

### Inertia sensitivity: 15 full groups, 202,500 same-torque steps

![Inertia sensitivity: 15 full groups, 202,500 same-torque steps](../v6_lite/visualization/latest/studies/inertia_all_15_summary.png)

Source 19ad85c; input 9cd1831 | Only principal body inertia scaled, 73 robot bodies. Five nominal runs reproduce nine saved fields exactly.
Fresh-current-state diagnostic errors differ from original cached trace metrics. Not a new 25/11/interval acceptance, feedback robustness or hardware test.

### All 15 inertia groups: two declared geometry scopes

![All 15 inertia groups: two declared geometry scopes](../v6_lite/visualization/latest/studies/inertia_all_15_clearance_histories.png)

Source 19ad85c | Target: 13,501 actual states x 75 pairs/run. Whole body: 5,401 interpolated configurations x 2,927 pairs/run.
Triangles indicate censored lower bounds; solid curves use uncensored minima. Whole-body scope is not every-2ms/all-pairs or continuous-time CCD.

### All 15 fresh-state tracking histories

![All 15 fresh-state tracking histories](../v6_lite/visualization/latest/studies/inertia_all_15_tracking_histories.png)

Source 19ad85c | No new feedback / QP / runtime executor; the same frozen torque is applied open loop. Initial state excluded from errors.
Dashed 0.10 / 0.18 mm lines are original nominal diagnostic comparisons, not per-sample pass gates. Continuum path statistics use the shaded 4.5-25.5 s window.

### Ten paired execution responses across all five scenes

![Ten paired execution responses across all five scenes](../v6_lite/visualization/latest/studies/inertia_all_10_paired_tip_responses.png)

Source 19ad85c | Each perturbed full trajectory compared with its own alpha=1.00 fresh-state trajectory. These are tip offsets, not target errors.
No run selection. JSON also retains low-level joint, base pose and tip orientation response statistics for all ten pairs. Display curves decimated.

### Paired clearance response: censor-aware differences

![Paired clearance response: censor-aware differences](../v6_lite/visualization/latest/studies/inertia_all_10_paired_clearance_responses.png)

Source 19ad85c | Difference is plotted only when both target minima are uncensored; gaps are retained.
Ten pairs: 131,555 exact states, 3,455 excluded; total 135,010. Exact range -5.967862 to +5.203110 mm. Not a robust uncertainty bound.

## 计算性能与墙钟部署

### Research compute and native wall continuation: separate verdicts

原生墙钟五次尝试的结果如下；四次拒绝均未执行下一物理步。

| 场景 | 状态 | 已执行仿真时间 (s) | 拒绝时晚唤醒 (ms) |
| --- | --- | ---: | ---: |
| 00 | MISSED_SERVO_WINDOW | 13.288 | 3.735 |
| 01 | COMPLETED | 27.000 | — |
| 02 | MISSED_SERVO_WINDOW | 5.848 | 3.760 |
| 03 | MISSED_SERVO_WINDOW | 5.962 | 4.311 |
| 04 | MISSED_SERVO_WINDOW | 2.476 | 4.608 |

![Research compute and native wall continuation: separate verdicts](../v6_lite/visualization/latest/studies/nominal_compute_and_c11_wall_status.png)

Nominal source 9cd1831: simulation PASSED (25 functional + 11 contract + interval), p95 sampling target PASSED; engineering 16ms/p99 20ms NOT_MET.
Wall source c2b5077: one of five complete; formal repeats/new native 26/11/interval not complete. Admin DEFERRED/platform BLOCKED; hard RT and hardware NOT_ESTABLISHED.

## 来源与范围

名义运行：`research_acceptance_01`，生成源码 `9cd1831`；惯量完整合并：`research_inertia_shadow_resumed_01`，生成源码 `19ad85c`。精确输入 SHA 见清单；14 组观察复用、1 组续作及原中断文件保留。

新鲜当前状态曲线与原缓存日志统计分别标注，原名义验收数字不被图表改写。压力图包含完整 01/02 和部分 00/03/04，拒绝后的轨迹没有延伸。惯量只改变 body inertia，使用同一 67 路力矩；没有重新求解反馈闭环，也未建立模型误差界、连续时间安全或硬件保证。

PCC 三类监控图从本次名义运行逐字节复制；旧监控绑定占位字段不作为区间 CBF 活跃性证据，区间有效性使用原独立重算报告。

重新生成须使用新目录，原始实验文件不覆盖：

```powershell
python -m v6_lite.visualization.generate_latest_full --output-dir v6_lite/visualization/latest_rebuild
python -m v6_lite.visualization.validate_latest_full --output-dir v6_lite/visualization/latest_rebuild
```

本地交互展示：

```powershell
python -m http.server 8765 --bind 127.0.0.1
# http://127.0.0.1:8765/v6_lite/visualization/latest/index.html
```
