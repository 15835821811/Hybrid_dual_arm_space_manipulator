# V6.4-B.3 固定配对路线价值可视化

[静态总览](../v6_4/visualization/conditional_route_value_20261007_01/index.html) · [可移植研究报告](../v6_4/releases/conditional_route_value_20261007_01/report.md) · [方法表 CSV](../v6_4/visualization/conditional_route_value_20261007_01/method_table.csv) · [方法表 Markdown](../v6_4/visualization/conditional_route_value_20261007_01/method_table.md)

本轮终态为 `ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION`。两侧共六个固定 pilot 槽位均完成 27 s，并通过完整 Task 及原五项独立门禁；较好非零方向的数值排名随障碍换边反转，但相对零残差的 `I_route` 降低仅为 2.79% / 3.81%，未达到冻结的 10% 路线价值门槛。所有候选都成功，也未满足“一条非零成功、零残差或相反方向失败”的判据。因此按计划停止 P2/P3/P4，没有新 teacher、主训练或正式新 TEST。

已有非学习可执行性见证，当前瓶颈是路线质量差额不足。有限负结果不否定 Diffusion；本轮未评价新模型相对错条件、无条件经验抽样或检索的收益。

## 两张固定配对路线图

| 开发任务 | 图像 | PDF | 范围 |
|---|---|---|---|
| `b3_mother_00_c_plus` | [c+ PNG](../v6_4/visualization/conditional_route_value_20261007_01/01_pilot_c_plus_routes.png) | [c+ PDF](../v6_4/visualization/conditional_route_value_20261007_01/01_pilot_c_plus_routes.pdf) | z0 / z+ / z− 的参考、独立保存实际路径、固定窗口与相关球净空 |
| `b3_mother_00_c_minus` | [c− PNG](../v6_4/visualization/conditional_route_value_20261007_01/02_pilot_c_minus_routes.png) | [c− PDF](../v6_4/visualization/conditional_route_value_20261007_01/02_pilot_c_minus_routes.pdf) | 同一开发母场景的另一障碍侧，三个候选分别实际执行 |

![B.3 c+ 侧路线与质量](../v6_4/visualization/conditional_route_value_20261007_01/01_pilot_c_plus_routes.png)

![B.3 c− 侧路线与质量](../v6_4/visualization/conditional_route_value_20261007_01/02_pilot_c_minus_routes.png)

两张图分别展示各自保存的实际状态，不镜像重用轨迹。虚线参考与实际曲线分别标示；端点的正交投影不等同于全连续体净空或全臂主动形状规划。图表与静态页面只读取已保存证据，新增物理步、DDIM、optimizer 更新及几何查询均为 0，不生成新视频。

## 六个开发槽位

| 槽位 | 障碍侧 | 候选 | 完整 Task + 原独立门禁 | I_route / rad·s⁻¹ | 相关球窗口最小净空 / mm |
|---|---|---|---|---:|---:|
| PILOT_00 | c+ | z0 | PASS，27 s | 0.108601632 | 25.884293 |
| PILOT_01 | c+ | z+ | PASS，27 s | 0.111963142 | 25.777815 |
| PILOT_02 | c+ | z− | PASS，27 s | 0.105570475 | 25.861279 |
| PILOT_03 | c− | z0 | PASS，27 s | 0.112069755 | 25.877743 |
| PILOT_04 | c− | z+ | PASS，27 s | 0.107796550 | 25.900197 |
| PILOT_05 | c− | z− | PASS，27 s | 0.114338061 | 25.766457 |

该表覆盖 **2 个开发任务、1 个母场景、6 次实际尝试**，不是 6 个独立任务或正式 TEST。固定作用窗口为 `[8.461298845015024, 11.977496008196567] s`，各方法使用相同完整闭区间的 50 Hz 消费样本。

`I_route = sqrt(mean(task_avoidance_intervention²))` 精确复用原日志中的 17 维 QP 名义速度与实际选中速度差的范数，不是 raw proposal 与 selected 的差。名义值经过原速度边界裁剪，两个源向量未保存。相关净空使用全部连续体碰撞几何与移动球的对应几何对、独立保存的 2 ms 状态；不以 rigid-target 主导的整机最小值宣称绕障收益。

完整质量比较要求 Task、执行合同、区间独立重算、原生几何和参考消费绑定全部通过。失败前缀或证据缺失必须单列，不填入完整任务均值。本轮六槽均完整通过。

## 正式五组 TEST 与旧权重诊断

| 预声明新 TEST 方法 | 状态 | 成功数 |
|---|---|---|
| Z0：零残差 | `NOT_RUN_PILOT_STOP` | 未评价 |
| R0：TRAIN 高质量检索 | `NOT_RUN_PILOT_STOP` | 未评价 |
| U0：TRAIN 高质量经验抽样 | `NOT_RUN_PILOT_STOP` | 未评价 |
| D_true：正确障碍条件 Diffusion | `NOT_RUN_PILOT_STOP` | 未评价 |
| D_swap：错配障碍条件 Diffusion | `NOT_RUN_PILOT_STOP` | 未评价 |

正式 TRAIN / VAL / TEST 的 6 / 2 / 4 个任务已冻结，pilot 标签未进入正式数据。停止后 teacher actual 为 0、新训练为 0、optimizer 更新为 0、新模型 DDIM 为 0、正式 TEST actual 为 0。`NOT_RUN` 不能写成 0/4 失败，也不能据此宣布检索胜出。

P0 使用旧 B.2 update250 权重做 32 次 DDIM：16/16 组同噪声对照观察到条件响应，27/32 原始输出幅值合法。原始输出未裁剪、替换，公开旧 TEST 只用于诊断，物理执行与训练均为 0。响应变化不证明正确适应，也不能代替新模型五组对照。

## 成本与验收范围

| 成本项 | 次数 / 步数 |
|---|---:|
| 消耗的 actual 槽位 | 6 |
| actual 物理步 | 81,000 |
| 私有预演物理步 | 81,000 |
| 独立保存力矩重放步 | 81,000 |
| 执行及证据流程原生几何查询 | 201,187,130 |
| 额外路线质量几何查询 | 5,022,372 |
| 输入几何预检查询 | 41,846 |
| 旧保存状态接线检查查询 | 186 |

预演、重放和几何查询分别计费，不算新独立闭环。分阶段计时、原始 dispatch 长尾、五个研究问题的直接回答和完整成本字典见[报告](../v6_4/releases/conditional_route_value_20261007_01/report.md)。本轮无新模型，相关训练/检索/采样对照计时保持未运行。

20 ms 规划、2 ms 物理、27 s 任务与原 QP、67 路力矩、私有预演、安全门禁及 20 mm 幅值上限保持。墙钟 20 ms 不作本研究门槛。robot-target 原生 500 Hz；whole-body 为 50 Hz 边界加配置空间 subdivisions4。部署 **`NOT_MET`**，不建立硬实时、连续时间或模型失配保证。

## 身份、校验与历史入口

- 本轮实际执行 producer：`f5f1687f5ba0b119c91bbc385dfdc841b56d091e`。
- B.2 已发布基点：`28ef7889be16b4a504d9449f52c50e3a99e82a31`；B.2 原算法 producer：`7d0a3fd4bd17f41b99b388c30c5ac4897ac4d31d`。
- [原两图 manifest](../v6_4/visualization/conditional_route_value_20261007_01/manifest.json)绑定绘图输入与 PNG / PDF；[dashboard manifest](../v6_4/visualization/conditional_route_value_20261007_01/dashboard_manifest.json)独立绑定页面、表格与源报告；[绘图数据](../v6_4/visualization/conditional_route_value_20261007_01/plot_data.json)保留两侧来源。
- [可移植发布清单](../v6_4/releases/conditional_route_value_20261007_01/release_manifest.json)记录原件、源码、映射与省略项；省略重轨迹不等于拥有全部可重放原始证据。
- [B.2 历史总览](../v6_4/visualization/task_anchored_residual_20261007_01/index.html)与[图表/回放说明](V6_4_B2_VISUALIZATION.md)保留其原结果和失败。
- [V6.2 历史总览](../v6_lite/visualization/latest/index.html)与[历史可视化说明](V6_2_LATEST_VISUALIZATION.md)保留原 35 个视频及各自验收范围。

静态 HTML 不依赖 CDN 或脚本；GitHub 可直接查看 PNG / PDF 和 Markdown 方法表，HTML 源文件本身不代表已启用 GitHub Pages。
