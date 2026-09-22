# V6-lite：确定性混合双臂跟踪与避障

> 当前运行合同：`v6_lite_6`（2026-09-21）

V6-lite 是一个单独目录内的最小闭环实现。它不加载训练集、权重、归一化器或任何 Diffusion 模块，也不生成、投影或排序多条候选轨迹。

在线链路只有：

```text
卫星指定抓捕位姿 ─────┐
连续体 irregular waypoints ─┼→ 单一优先级加权约束速度 QP（50 Hz）
整机/目标卫星有符号距离 ─┤                  ↓
目标卫星 6D 外生速度 ───┘
                       67 维饱和模型补偿力矩伺服（500 Hz）
                                      ↓
                              data.ctrl + mj_step
```

## 控制定义

- 规划坐标仍使用已审计的 17 维表示：连续体 10 维形状坐标和刚性臂 7 维关节；执行层为 67 个 MuJoCo 力矩执行器。
- 刚性臂追踪目标卫星本体坐标系中的固定点 `[0, -0.205, 0] m`及指定姿态。目标卫星为自由关节刚体，每场景具有不同的线速度和角速度。
- 连续体真实末端定义为 `link_30` 刚体原点加局部偏置 `[0.0475, 0, 0] m`，因此零构型初始世界坐标为 `[1.930, 0.626, 0] m`；位置、点 Jacobian、误差、航点、trace、验证和视频统一使用该定义。
- 连续体末端恢复原 `irregular_waypoints` 合同：7 个不规则航点、原径向/前后随机范围、世界坐标 -Y 方向 0.15 比例平移、段长比例分配的 21 s 最小加加速度参考。场景 1 保留种子 `20260801` 与原相对几何，并锚定到修复后的真实末端初始位置；其余场景使用独立种子。
- QP 同时处理两臂的位置与 SO(3) 姿态任务，刚性抓捕任务权重高于连续体任务；关节边界、速度/加速度变化范围及整机距离 barrier 为硬约束。
- `v6_lite_6` 以 opt-in 方式把目标卫星加入 V6 的碰撞 pair 策略。含两个工作区障碍时共有 2,927 个 pair，其中目标相关 75 个：`continuum_target=62`、`rigid_target=7`、`base_target=6`。只有 `collision_0072/end_link` 和 `collision_0073/end_effector_r` 享有终端预期接触豁免；连续体、基座以及其他刚性几何均不豁免。
- 在线安全缓冲按 pair 分类：`continuum_target`、`base_target` 及其他普通 pair 为 25 mm，`rigid_target` 为 5 mm；统一在 80 mm 内激活、90 mm 查询上限下形成硬 CBF 行。
- 目标卫星自由关节的实测 6D `qvel` 不属于 17 维决策变量，而是移动目标的外生漂移。对最近点法向 `n`、双方点 Jacobian `J_a,J_b` 和反作用映射 `G`，使用 `d_dot=A qdot_plan+d_dot_T`，其中 `A=n^T(J_a-J_b)G`、`d_dot_T=n^T(J_a-J_b)qvel_T^exo`；QP 约束为 `A qdot_plan >= -8(d-d_safe,pair)-d_dot_T`，同时覆盖目标平移与自转。
- 自由漂浮基座不被冻结。末端与距离 Jacobian 使用质量矩阵得到的零动量反作用映射。
- 基座位姿漂移相对仿真初始化完成后的精确自由关节位姿计算：平移漂移为位置差的二范数，姿态漂移为四元数符号不变的 SO(3) 测地角 `2 acos(|q₀ᵀq|)`。
- 500 Hz 执行层对相邻 QP 速度指令做 20 ms 线性斜坡，再用质量矩阵消去未驱动基座加速度，形成临界阻尼的模型补偿力矩。该层没有第二个优化器。
- 每个 20 ms 任务周期只调用一次 QP；若求解失败只允许安全停止，不调用 oracle。是否出现失败或安全停止由当前合同的正式五场景结果与独立验证决定。

## 固定验收门槛

| 指标 | 门槛 | `v6_lite_6` 五场景结果 |
| --- | ---: | ---: |
| 刚性臂抓捕点终误差 | ≤ 0.10 mm | **0.009011 mm，通过** |
| 刚性臂稳态 RMSE | ≤ 0.15 mm | **0.009617 mm，通过** |
| 连续体 irregular-waypoint 活动路径 RMSE | ≤ 0.18 mm | **0.159450 mm，通过** |
| 刚性臂全程最大姿态误差 | < 0.25° | **0.041769°，通过** |
| 连续体全程最大姿态误差 | < 0.25° | **0.019170°，通过** |
| 整机 4×细分最小有符号间隙 | ≥ 5 mm | **14.995557 mm，通过** |
| 连续体—目标卫星 4×细分最小间隙 | ≥ 5 mm | **24.995132 mm，通过** |
| 连续体—目标卫星全部原生 500 Hz 状态间隙 | ≥ 5 mm，且低于门槛/穿透状态均为 0 | **24.994694 mm；0/0，通过** |
| 50 Hz 任务全链路 p95 | ≤ 20 ms | **7.809830 ms，通过** |
| 500 Hz 力矩计算 p95 | ≤ 2 ms | **0.339205 ms，通过** |

基座漂移仍是报告指标，不额外虚构验收阈值；五场景最大平移/姿态漂移为 8.633348 mm/3.773870°。旧 `v6_lite_5` 数值不覆盖目标卫星 pair，不能作为 `v6_lite_6` 的通过证据。正式运行完成 67,500 次力矩更新和 6,750 次 QP，QP 失败为 0，共有 1,031 次 clearance 约束绑定。独立验证为 **26/26** 通过。

## 运行与验证

从仓库根目录运行：

```powershell
python -m v6_lite.run_v6_lite
python -m v6_lite.validate_v6_lite
python -m unittest v6_lite.test_v6_lite v6_lite.test_target_collision_policy -v
```

权威结果是 `output/v6_lite_metrics.json`；`output/validation.json` 从保存的力矩序列重放全部 MuJoCo 步，重新计算末端位置、末端姿态和基座漂移。碰撞审计包含两条互补路径：对 50 Hz `task_qpos` 做 4 倍构型细分的整机离散验证，以及对包括初始状态在内的全部原生 500 Hz 重放状态逐一扫描 62 个 `continuum_target` pair。验证器还重算 pair 数量与策略 SHA-256，检查目标漂移/白名单元数据；`output/artifact_manifest.json` 绑定指标和 trace。

## 证据边界

目标卫星不再被整体排除：除 `collision_0072/end_link` 与 `collision_0073/end_effector_r` 的精确终端接触豁免外，目标相关几何均进入正间隙策略。当前证据的含义仍严格受限于已哈希 MuJoCo 模型、固定场景和离散检查时刻：整机验证是 50 Hz 状态间的 4 倍细分，连续体—目标专项验证是原生 500 Hz（2 ms）状态扫描；两者都不是相邻采样时刻之间的连续时间/CCD 安全证书。当前也没有模拟夹爪闭合、接触后的组合体动力学，不声称硬件迁移或相对学习规划器的优势。

## 跟踪可视化与五视角视频

```powershell
python -m v6_lite.visualization.generate_visualizations
python -m v6_lite.visualization.validate_visualizations
```

`visualization/output/error_curves.png` 的六个面板同时显示刚性抓捕点误差、连续体 W1–W7 跟踪误差、两臂姿态误差、基座平移漂移、基座姿态漂移和整机在线最小间隙；`safety_clearance_summary.png` 用两个面板分别呈现整机全局最小间隙和连续体—目标最小间隙；`base_pose_drift.gif` 动态展开五个场景的两条基座漂移曲线；`tracking_paths_3d.png` 标出 W1–W7 及目标/实际轨迹。

`visualization/output/videos/` 保存 overview、front、side、top、iso 五个 640×480、30 FPS、27 s 的独立视频，以及一个 1920×960 五视角组合视频。每帧除了刚性/连续体目标与末端坐标系，还绘制基座初始/当前坐标系和 W1–W7 的七个固定坐标系；front 视角直接标注 W1–W7。RGB 分别为 XYZ，顶部叠加基座平移与姿态漂移的逐帧数值。所有视频均由正式 trace 中的力矩序列重新执行 MuJoCo 后渲染。

当前图、GIF、视频及其 manifest 均由已通过验证的 `v6_lite_6` 五场景 trace 重新生成；旧合同生成的可视化只能作为历史产物，不能作为当前验收证据。
